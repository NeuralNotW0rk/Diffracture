# Diffracture

**Diffracture** is a modular PyTorch library for intervening in trained models without editing their source code. It handles low-rank adapters (LoRA, DoRA) and *network bending* (deterministic transforms applied to intermediate activations) through a single interface. Interventions are stored as portable, self-describing files that can be attached to a model, toggled, trained, extracted, merged, or removed at runtime.

Diffracture keeps three things separate:

- **what** an intervention is (its parameters and metadata),
- **how** it is computed (the math), and
- **where and by what mechanism** it enters the model (module grafting, forward hooks, weight parametrization, or a direct in-place merge).

Each concern can be swapped independently. The same LoRA file can be grafted for training, hooked for lightweight inference, or merged permanently for deployment. A new adapter type only needs a new kernel. It does not need a new injection system.

> **Status:** early alpha (`0.1.0`). The API may change between releases.

---

## Contents

- [Installation](#installation)
- [Core concepts](#core-concepts)
- [Quickstart](#quickstart)
  - [Train a LoRA](#train-a-lora)
  - [Load and apply a saved grating](#load-and-apply-a-saved-grating)
  - [Network bending](#network-bending)
  - [Feature clustering](#feature-clustering)
- [Injection strategies](#injection-strategies)
- [Built-in kernels](#built-in-kernels)
- [Adapting non-standard modules](#adapting-non-standard-modules)
- [The grating file format](#the-grating-file-format)
- [Extending Diffracture](#extending-diffracture)
- [Scripts and tests](#scripts-and-tests)
- [Known limitations](#known-limitations)
- [Related projects](#related-projects)
- [Acknowledgements and prior work](#acknowledgements-and-prior-work)
- [License](#license)

---

## Installation

Diffracture is not yet published on PyPI. Install it from source:

```bash
pip install git+https://github.com/NeuralNotW0rk/Diffracture.git
```

or, for development:

```bash
git clone https://github.com/NeuralNotW0rk/Diffracture.git
cd Diffracture
pip install -e .
```

**Requirements:** Python 3.10 (the only version tested so far), PyTorch ≥ 2.0, `safetensors`, and `torchvision`.

---

## Core concepts

| Concept | Role |
|---|---|
| **Element** | A data container (`nn.Module`) for a single intervention at a single module address. It holds trainable parameters (for example `lora_down` and `lora_up`) and serializable metadata (for example rank, alpha, angle, channel indices). It has no computation logic of its own. |
| **Grating** | A collection of Elements, and the unit you save, load, move between devices, and optimize. It is an `nn.Module`, so `.to()`, `.parameters()` and `.state_dict()` work as expected. |
| **Kernel** | The stateless math for a given element type. A kernel can compute an intervention in *activation space* (`__call__`, `forward_delta`), in *weight space* (`compute_delta`), or both, and it reports which of these it supports for a given module. |
| **Injector** | The mechanism that attaches a grating to a model: `graft`, `hook`, `parametrize`, or `direct`. It also handles freezing, extraction, merging and cleanup. |
| **Actant** | The high-level controller that binds a grating to a target model and manages the injection lifecycle. |

Elements, kernels, injectors and analysis strategies are looked up by string keys in a global registry. Element and kernel types share a key (an element with `kernel_type="lora"` is computed by the kernel registered as `"lora"`), which lets a grating be reconstructed from a file without importing anything by hand.

### Lifecycle

```python
actant = Actant(model)
actant.activate(grating, injection_strategy="graft", strength=1.0)  # attach
grating.set_multiplier(0.5)        # adjust intensity at runtime
outputs = actant.extract()         # harvest trained element state_dicts
actant.collapse_and_cleanup()      # merge permanently and remove all wrappers/hooks
# or
actant.deactivate()                # detach and restore the original model
```

`activate` can be called more than once to stack gratings. `deactivate` reverses them in reverse order.

There are two ways to scale an intervention:

- **`strength`** is set once at activation time.
- **`multiplier`** can be changed at any time with `grating.set_multiplier()`, or per element with `element.multiplier`.

The two values multiply together. Setting `element.active = False` or a multiplier of `0.0` bypasses an element at negligible cost, without any surgery on the model.

---

## Quickstart

### Train a LoRA

```python
import torch
from diffracture import Actant, Grating
from diffracture.topology.lora import LoRAElement, find_lora_targets

model = ...  # any nn.Module
model.requires_grad_(False)  # freeze the base model before activation

grating = Grating()
for address, param, weight in find_lora_targets(model, include=r"attn.*\.weight$"):
    grating.add_element(LoRAElement.for_parameter(address, weight, rank=8, param=param))
grating.to(device)

actant = Actant(model)
actant.activate(grating, injection_strategy="graft")

optimizer = torch.optim.AdamW(grating.parameters(), lr=1e-4)
for batch in loader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

grating.save("my_lora.safetensors")
actant.deactivate()
```

`find_lora_targets` lists every parameter with at least 2 dimensions that a module owns directly, whatever the module's type. It takes optional `include` and `exclude` regexes, matched against full parameter names. `LoRAElement.for_parameter` sizes the element from the tensor it will adapt and infers the [weight layout](#adapting-non-standard-modules).

### Load and apply a saved grating

```python
from diffracture import Actant, Grating

grating = Grating.load("my_lora.safetensors").to(device)

actant = Actant(model)
actant.activate(grating, injection_strategy="hook", strength=0.8)
out = model(x)

actant.deactivate()  # model is restored exactly
```

To bake the grating into the model's weights for zero-overhead deployment, use `injection_strategy="direct"`, or call `actant.collapse_and_cleanup()` after any other strategy.

### Network bending

Bending elements apply deterministic transforms to a module's output activations. By default they apply to all channels. Pass `indices` to restrict them to specific channels.

```python
from diffracture import Actant, Grating
from diffracture.topology.bending import RotateElement, AblateElement

grating = Grating()
grating.add_element(RotateElement("synthesis.block3.conv", angle=45.0, indices=[0, 4, 7]))
grating.add_element(AblateElement("synthesis.block5.conv", indices=list(range(16))))

actant = Actant(model)
actant.activate(grating, injection_strategy="hook", strength=0.5)  # 50% blend with the original
image = model(z)
actant.deactivate()
```

### Feature clustering

The `analysis` subpackage finds groups of channels in a layer that behave alike, so they can be bent together. It does this in three steps:

1. Record the layer's activations over many forward passes.
2. Train a small *exemplar classifier* to tell the channels apart by their activation maps.
3. Cluster the channels by their averaged bottleneck embeddings with k-means.

```python
import torch
from diffracture.analysis import FeatureClusteringPipeline
from diffracture.topology.bending import ScaleElement

address = "synthesis.block3.conv"
pipeline = FeatureClusteringPipeline(model, strategy_name="cnn")  # "cnn" for [B, C, H, W], "transformer" for sequences

pipeline.collector.start_collecting([address])
with torch.no_grad():
    for z in latents:
        model(z)
pipeline.collector.stop_collecting()

cluster_map = pipeline.run_clustering(address, channel_dim=1, num_clusters=5, epochs=10)
# [{"feature_index": 0, "cluster_index": 3}, ...]

element = ScaleElement(address, scale_factor=1.5, cluster=2, cluster_map=cluster_map)
```

When an element has `cluster` and `cluster_map` set and no explicit `indices`, the kernel works out the channel indices at runtime. The cluster map is saved inside the grating, so the file stays self-contained.

`ActivationCollector.export_to_disk()` can also write the collected activations in an `ImageFolder`-compatible layout, for training external classifiers.

---

## Injection strategies

| Strategy | Mechanism | Reversible | Best for |
|---|---|---|---|
| `graft` | Replaces each target module with a `GraftedModule` wrapper that calls the kernel. Train/eval mode is copied from the original. | Yes | Training, DoRA, bending. Kernels that need to control the whole forward pass. |
| `hook` | Registers a forward hook that adds `kernel.forward_delta(...)` to the module's output. The module tree and `state_dict` keys are untouched. If a weight-space kernel has no activation path for a module, it falls back to `parametrize` for that element, so one grating can mix both kinds. | Yes | Lightweight inference with additive kernels (LoRA). Bending. |
| `parametrize` | Uses `torch.nn.utils.parametrize` to add `kernel.compute_delta(...)` to the target parameter just before each forward pass. Chains with other parametrizations on the same tensor. | Yes | Weight-space kernels on any module that reads the parameter in its own `forward`, including custom layers. |
| `direct` | Adds `compute_delta` to the base weights in place, immediately. | **No** | Final deployment with zero overhead. |

All reversible strategies freeze the parameters of the modules they target when they activate. Freezing the rest of the model is up to you.

### Compatibility

| Kernel | `graft` | `hook` | `parametrize` | `direct` / collapse |
|---|---|---|---|---|
| `lora` | `nn.Linear`, `nn.Conv1d`, `nn.Conv2d` (exact types, standard layout) | Same types via the activation path. Any other module via the parametrize fallback. | Any module with a ≥2-D target parameter | ✓ |
| `dora` | Same as LoRA | Same as LoRA (the activation path recomputes the fused forward, so it is roughly 2× the cost) | Any module with a ≥2-D target parameter | ✓ |
| bending | Modules whose `forward` takes a single input | ✓ | ✗ (skipped) | ✗ (activation-only) |

---

## Built-in kernels

### LoRA (`"lora"`)

Standard Low-Rank Adaptation. The update is defined in weight space as ΔW = (α / r) · strength · s_ref · B·A, where:

- B is `lora_up`, initialized to zeros,
- A is `lora_down`, initialized with Kaiming-uniform,
- s_ref is the element's `reference_scale` (see below).

For known module types the kernel also provides an activation-space path that computes B(A(x)) directly with the module's own functional op (`F.linear`, `F.conv1d`, `F.conv2d`). This never materializes the full weight delta. Convolutional LoRA uses an A of shape `[r, in, *kernel]` and a B of shape `[out, r, 1, ...]`.

### DoRA (`"dora"`)

Weight-Decomposed Low-Rank Adaptation. The base weight plus the LoRA delta is renormalized row by row (per output) to a learned `magnitude` vector. Because the result is a fused, normalized weight rather than an additive delta, DoRA is most efficient with `graft` or `parametrize`.

### Network bending

| Key | Element | Parameters | Input |
|---|---|---|---|
| `erode` | `ErodeElement` | `radius` | 4-D `[B, C, H, W]` |
| `dilate` | `DilateElement` | `radius` | 4-D |
| `translate` | `TranslateElement` | `offset_x`, `offset_y` (normalized coords) | 4-D |
| `scale` | `ScaleElement` | `scale_factor` | 4-D |
| `rotate` | `RotateElement` | `angle` (degrees) | 4-D |
| `resize` | `ResizeElement` | `scale_x`, `scale_y` | 4-D, all channels |
| `flip-h` | `FlipHorizontalElement` | none | ≥3-D |
| `flip-v` | `FlipVerticalElement` | none | ≥3-D |
| `invert` | `InvertElement` | none | any |
| `binary-thresh` | `BinaryThresholdElement` | `threshold` | any |
| `scalar-multiply` | `ScalarMultiplyElement` | `factor` | any |
| `ablate` | `AblateElement` | none | any |

Every bending element also accepts `indices`, `cluster` and `cluster_map` to choose which channels it targets. The channel axis is taken as the last dimension for `nn.Linear` outputs and dimension 1 for convolutions and 4-D tensors. Strength below 1.0 blends linearly between the original and the transformed activations.

---

## Adapting non-standard modules

Many generative models use layers that are not plain `nn.Linear` or `nn.Conv2d`. Examples are StyleGAN's modulated convolutions, with a `[1, out, in, k, k]` weight, and equalized-learning-rate layers, whose raw weights are N(0, 1) and rescaled at runtime. Diffracture's weight-space LoRA and DoRA handle these with two pieces of metadata.

- **`WeightLayout`** describes how a weight tensor folds into the `[out, in]` matrix the low-rank delta lives in. You specify `out_dim` and `in_dims`, and every other dimension must have size 1. `WeightLayout.infer(shape)` recognizes the common leading-singleton broadcast layout automatically.
- **`reference_scale`** expresses the delta relative to the base weight's RMS, compared with the standard 1/√fan_in init scale. This means a given LoRA magnitude has the same effect on ordinary and equalized-lr layers. `LoRAElement.for_parameter(..., relative_scale=True)` computes it for you.

Elements can target any parameter by name through `param` (default `"weight"`). When a module has no activation-space fast path, the `hook` strategy falls back to parametrization automatically. With `parametrize`, the kernel reads the unparametrized base tensor through a proxy, which avoids recursion.

---

## The grating file format

A grating is saved as one `.safetensors` file. Element parameters are stored as tensors, and a JSON **topology** is embedded in the file's metadata:

```json
[
    {
        "address": "synthesis.block3.conv",
        "kernel_type": "rotate",
        "metadata": { "angle": 45.0, "indices": [0, 4, 7], "cluster": null, "cluster_map": null }
    }
]
```

`Grating.save(path, topology_path=...)` can also write the topology to a separate JSON file for hand-editing. `Grating.load(path, topology_path=...)` lets that external file override the embedded one. Gratings without weights, such as pure bending, are fully supported.

---

## Extending Diffracture

Every component is registered by name. Importing `diffracture` auto-discovers every module in the package, so registration happens on import. Custom components defined outside the package register as soon as their module is imported.

```python
import torch.nn as nn
from diffracture.registry import register_element, register_kernel
from diffracture.topology.base_element import Element
from diffracture.kernels.bending_kernel import BendingKernel


@register_element("clamp")
class ClampElement(Element):
    def __init__(self, address: str, limit: float):
        super().__init__(address, kernel_type="clamp")
        self.metadata["limit"] = float(limit)

    @classmethod
    def from_metadata(cls, address, metadata):
        return cls(address, limit=metadata.get("limit", 1.0))


@register_kernel("clamp")
class ClampKernel(BendingKernel):
    def _apply_transform(self, x, element, original_module):
        limit = element.metadata["limit"]
        return x.clamp(-limit, limit)
```

The relevant extension points are:

- **`Element`**: store trainable tensors in `self.params` (an `nn.ParameterDict`) and anything JSON-serializable in `self.metadata`. Implement `from_metadata` so the element can be rebuilt on load.
- **`Kernel`**: implement whichever of `__call__`, `forward_delta` and `compute_delta` apply. Then report support through `supports_activation` and `supports_weights`. The descriptive `traits` dictionary (`is_additive`, `requires_weight_fusion`) records the kernel's mathematical character.
- **`Injector`** (`register_injector`): implement `inject` and `cleanup`, and optionally `on_inject`, `on_extract` and `on_collapse`.
- **`ExemplarStrategy`** (`register_exemplar`): provide a classifier that returns `(embedding, logits)` and an input preprocessor, to support new activation shapes in clustering.

---

## Scripts and tests

| Path | Description |
|---|---|
| `scripts/convert_loraw.py` | Converts legacy [LoRAW](https://github.com/NeuralNotW0rk/LoRAW) checkpoints (LoRA/DoRA for `stable-audio-tools`) into gratings: `python scripts/convert_loraw.py <input> <output> [--alpha A] [--multiplier M] [--save-topology]`. Accepts a single file or a directory. |
| `scripts/test_bending.py` | Smoke test for bending registration, hook injection and restoration on a toy model. |
| `scripts/test_clustering.py` | End-to-end check of activation collection, exemplar training and clustering. |
| `scripts/validate_injection.py` | Example of loading a grating into a Stable Audio Open model (requires `stable-audio-tools`. Edit the paths before running). |
| `tests/test_weight_lora.py` | Unit tests for weight-space LoRA and DoRA, layouts, reference scaling, and StyleGAN-style modules. |

```bash
python -m unittest discover tests
python scripts/test_bending.py
```

---

## Known limitations

- **One element per module address per grating.** A grating maps each address to a single element. To apply several interventions to the same module, put them in separate gratings and activate each one. Stacking works with `hook` and `parametrize`. `graft` refuses to wrap an already grafted module.
- **Grafting assumes single-input modules.** `GraftedModule.forward(x)` passes only one argument. For modules with multi-argument `forward` methods, use `hook` or `parametrize`.
- **Bending kernels are activation-only.** They cannot be collapsed or used with `direct`.
- **`resize`** changes the activation shape. It ignores `indices`, and it only works with `graft` at full strength.
- **No automatic strategy selection yet.** Kernel `traits` are intended to drive a future `injection_strategy="auto"`.
- Injectors currently report progress with `print`.

---

## Related projects

- [StemmA2A](https://github.com/NeuralNotW0rk/StemmA2A): a desktop workbench for generative audio and image models. It uses Diffracture gratings for network bending and for evolved LoRA genomes.
- [Neutral Selection](https://github.com/NeuralNotW0rk/NeutralSelection): a typed, composable evolutionary-computation framework, used alongside Diffracture to evolve LoRA gratings.

---

## Acknowledgements and prior work

Diffracture builds directly on the following research and software.

**Network bending.** The bending transforms, the idea of grouping features by the spatial similarity of their activation maps, and the clustering pipeline (activation collection → exemplar classifier → k-means on bottleneck embeddings) are adapted from:

> Terence Broad, Frederic Fol Leymarie, and Mick Grierson. *Network Bending: Expressive Manipulation of Deep Generative Models.* EvoMUSART 2021. [arXiv:2005.12420](https://arxiv.org/abs/2005.12420) · [github.com/terrybroad/network-bending](https://github.com/terrybroad/network-bending)

Diffracture reimplements the transforms in pure PyTorch, using affine grids and pooling instead of the original TorchScript/OpenCV operators. It generalizes them to arbitrary target modules and channel-axis conventions, and adds strength blending. The CNN exemplar classifier follows the ShuffleNetV2-based classifier in the network-bending repository.

**ShuffleNet V2.** The inverted-residual and channel-shuffle blocks in the CNN exemplar classifier come from:

> Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. *ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design.* ECCV 2018. [arXiv:1807.11164](https://arxiv.org/abs/1807.11164)

**LoRA.**

> Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**DoRA.**

> Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML 2024. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)

**StyleGAN2.** The weight-layout and reference-scale handling exists to support modulated convolutions and equalized-learning-rate layers, introduced in:

> Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. *Analyzing and Improving the Image Quality of StyleGAN.* CVPR 2020. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)

**Adapter libraries.** Diffracture's injector design came out of working with, and comparing against, existing adapter tooling, in particular:

- Hugging Face [PEFT](https://github.com/huggingface/peft)
- [minLoRA](https://github.com/cccntu/minLoRA), whose use of `torch.nn.utils.parametrize` for adapters informed the `parametrize` strategy

**LoRAW.** Diffracture is the successor to [LoRAW](https://github.com/NeuralNotW0rk/LoRAW), a LoRA/DoRA training library for `stable-audio-tools`. `scripts/convert_loraw.py` migrates its checkpoints.

Diffracture also depends on [PyTorch](https://pytorch.org/), [torchvision](https://github.com/pytorch/vision), and [safetensors](https://github.com/huggingface/safetensors).

---

## License

Diffracture is released under the [MIT License](LICENSE).

Code adapted from third-party projects remains subject to the terms of its original authors. See [Acknowledgements and prior work](#acknowledgements-and-prior-work).
