import copy
import math
import os
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffracture import Actant, Grating
from diffracture.topology.layout import WeightLayout
from diffracture.topology.lora import LoRAElement, find_lora_targets
from diffracture.topology.dora import DoRAElement
from diffracture.kernels.lora_kernel import LoRAKernel


class ModulatedConv(nn.Module):
    """StyleGAN2-style modulated conv: [1, out, in, k, k] equalized-lr weight, modulated and demodulated per sample."""
    def __init__(self, in_channel, out_channel, kernel_size, style_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.modulation = nn.Linear(style_dim, in_channel)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        s = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * s
        weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(batch, -1, 1, 1, 1)
        weight = weight.view(-1, in_channel, *weight.shape[3:])
        out = F.conv2d(x.view(1, -1, height, width), weight, padding=self.padding, groups=batch)
        return out.view(batch, -1, height, width)


class EqualLinear(nn.Module):
    """Equalized-lr linear: raw N(0, 1) weight scaled at runtime, so it is not an nn.Linear."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale)


def randomize_factors(element, std=0.1):
    with torch.no_grad():
        element.params["lora_up"].normal_(0, std)


def lora_grating(*elements):
    grating = Grating()
    for element in elements:
        grating.add_element(element)
    return grating


class TestWeightLayout(unittest.TestCase):
    def test_round_trip(self):
        cases = [
            ((6, 4), WeightLayout()),
            ((6, 4, 3, 3), WeightLayout()),
            ((1, 6, 4, 3, 3), WeightLayout.infer((1, 6, 4, 3, 3))),
            ((4, 6, 3, 3), WeightLayout(out_dim=1, in_dims=(0, 2, 3))),  # ConvTranspose2d
        ]
        for shape, layout in cases:
            w = torch.randn(shape)
            m = layout.to_matrix(w)
            self.assertEqual(m.shape, (6, 36 if len(shape) > 2 else 4))
            self.assertTrue(torch.equal(layout.from_matrix(m, shape), w))

    def test_transposed_layout_groups_by_output_channel(self):
        w = torch.randn(4, 6, 3, 3)
        m = WeightLayout(out_dim=1, in_dims=(0, 2, 3)).to_matrix(w)
        self.assertTrue(torch.equal(m[2], w[:, 2].reshape(-1)))

    def test_infer(self):
        self.assertEqual(WeightLayout.infer((8, 4)).to_metadata(), None)
        self.assertEqual(WeightLayout.infer((8, 4, 3, 3)).to_metadata(), None)
        self.assertEqual(WeightLayout.infer((1, 8, 4, 3, 3)).to_metadata(), {"out_dim": 1, "in_dims": [2, 3, 4]})

    def test_unassigned_dims_must_be_singleton(self):
        with self.assertRaises(ValueError):
            WeightLayout(out_dim=0, in_dims=(1,)).split_shape((4, 3, 2))


class TestWeightSpaceLoRA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_hook_and_weight_paths_agree_on_linear(self):
        model = nn.Sequential(nn.Linear(8, 6))
        x = torch.randn(3, 8)
        element = LoRAElement.for_parameter("0", model[0].weight, rank=2, relative_scale=False)
        randomize_factors(element)
        kernel = LoRAKernel()
        self.assertTrue(kernel.supports_activation(model[0], element))

        actant = Actant(model)
        actant.activate(lora_grating(element), injection_strategy="hook")
        hooked = model(x)
        actant.deactivate()

        actant.activate(lora_grating(element), injection_strategy="parametrize")
        parametrized = model(x)
        actant.deactivate()

        merged = copy.deepcopy(model)
        Actant(merged).activate(lora_grating(element), injection_strategy="direct")

        self.assertTrue(torch.allclose(hooked, parametrized, atol=1e-6))
        self.assertTrue(torch.allclose(hooked, merged(x), atol=1e-6))

    def test_hook_falls_back_to_parametrization_for_modulated_conv(self):
        model = nn.Module()
        model.conv = ModulatedConv(4, 6, 3, style_dim=5)
        x, style = torch.randn(2, 4, 8, 8), torch.randn(2, 5)
        base = model.conv(x, style)

        element = LoRAElement.for_parameter("conv", model.conv.weight, rank=2)
        self.assertEqual(element.metadata["layout"], {"out_dim": 1, "in_dims": [2, 3, 4]})
        self.assertEqual(tuple(element.params["lora_down"].shape), (2, 4, 3, 3))
        self.assertEqual(tuple(element.params["lora_up"].shape), (6, 2, 1, 1))
        randomize_factors(element)

        kernel = LoRAKernel()
        self.assertFalse(kernel.supports_activation(model.conv, element))
        self.assertTrue(kernel.supports_weights(model.conv, element))

        actant = Actant(model)
        actant.activate(lora_grating(element), injection_strategy="hook")
        bent = model.conv(x, style)
        actant.deactivate()

        merged = copy.deepcopy(model)
        Actant(merged).activate(lora_grating(element), injection_strategy="direct")

        self.assertFalse(torch.allclose(bent, base))
        self.assertTrue(torch.allclose(bent, merged.conv(x, style), atol=1e-5))
        self.assertTrue(torch.equal(model.conv(x, style), base))
        self.assertFalse(hasattr(model.conv, "parametrizations"))

    def test_reference_scale_makes_delta_independent_of_weight_parametrization(self):
        # Same effective weights, stored plainly vs. equalized-lr (raw values sqrt(in) times larger)
        plain = nn.Linear(16, 4, bias=False)
        equal = EqualLinear(16, 4)
        with torch.no_grad():
            equal.weight.copy_(plain.weight / equal.scale)
        x = torch.randn(3, 16)
        self.assertTrue(torch.allclose(plain(x), equal(x), atol=1e-5))

        changes = []
        for module in (plain, equal):
            model = nn.Module()
            model.layer = module
            element = LoRAElement.for_parameter("layer", module.weight, rank=2)
            torch.manual_seed(1)
            with torch.no_grad():
                element.params["lora_down"].normal_()
                element.params["lora_up"].normal_()
            base = module(x)
            actant = Actant(model)
            actant.activate(lora_grating(element), injection_strategy="parametrize")
            changes.append(module(x) - base)
            actant.deactivate()

        self.assertTrue(torch.allclose(changes[0], changes[1], atol=1e-4))

    def test_gradients_reach_factors_not_base(self):
        model = nn.Module()
        model.conv = ModulatedConv(4, 6, 3, style_dim=5)
        element = LoRAElement.for_parameter("conv", model.conv.weight, rank=2)
        actant = Actant(model)
        actant.activate(lora_grating(element), injection_strategy="hook")

        model.conv(torch.randn(2, 4, 8, 8), torch.randn(2, 5)).pow(2).mean().backward()
        # lora_up starts at zero, so only it receives gradient on the first step
        self.assertIsNotNone(element.params["lora_up"].grad)
        self.assertGreater(element.params["lora_up"].grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.conv.parametrizations.weight.original.grad)
        actant.deactivate()

    def test_stacked_gratings_on_one_parameter(self):
        model = nn.Module()
        model.layer = EqualLinear(8, 4)
        x = torch.randn(2, 8)
        base = model.layer(x)
        a = LoRAElement.for_parameter("layer", model.layer.weight, rank=1)
        b = LoRAElement.for_parameter("layer", model.layer.weight, rank=1)
        randomize_factors(a)
        randomize_factors(b)

        actant = Actant(model)
        actant.activate(lora_grating(a), injection_strategy="hook")
        only_a = model.layer(x)
        actant.activate(lora_grating(b), injection_strategy="hook")
        both = model.layer(x)
        self.assertEqual(len(model.layer.parametrizations.weight), 2)
        actant.deactivate()

        self.assertFalse(torch.allclose(only_a, both))
        self.assertTrue(torch.equal(model.layer(x), base))
        self.assertFalse(hasattr(model.layer, "parametrizations"))

    def test_collapse_merges_weight_space_delta(self):
        model = nn.Module()
        model.conv = ModulatedConv(4, 6, 3, style_dim=5)
        x, style = torch.randn(2, 4, 8, 8), torch.randn(2, 5)
        element = LoRAElement.for_parameter("conv", model.conv.weight, rank=2)
        randomize_factors(element)

        actant = Actant(model)
        actant.activate(lora_grating(element), injection_strategy="hook")
        bent = model.conv(x, style)
        actant.collapse_and_cleanup()

        self.assertFalse(hasattr(model.conv, "parametrizations"))
        self.assertTrue(torch.allclose(model.conv(x, style), bent, atol=1e-5))

    def test_save_load_keeps_target_metadata(self):
        weight = torch.randn(1, 6, 4, 3, 3)
        element = LoRAElement.for_parameter("conv", weight, rank=2)
        randomize_factors(element)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "g.safetensors")
            lora_grating(element).save(path)
            loaded = Grating.load(path).nodes["conv"]
        self.assertEqual(loaded.metadata["layout"], element.metadata["layout"])
        self.assertAlmostEqual(loaded.metadata["reference_scale"], element.metadata["reference_scale"])
        self.assertTrue(torch.equal(loaded.params["lora_up"], element.params["lora_up"]))

    def test_find_lora_targets(self):
        model = nn.Module()
        model.conv = ModulatedConv(4, 6, 3, style_dim=5)
        model.head = EqualLinear(6, 2)
        names = [f"{m}.{p}" for m, p, _ in find_lora_targets(model)]
        self.assertEqual(names, ["conv.weight", "conv.modulation.weight", "head.weight"])
        names = [f"{m}.{p}" for m, p, _ in find_lora_targets(model, exclude=r"modulation")]
        self.assertEqual(names, ["conv.weight", "head.weight"])

    def test_dora_on_modulated_conv_is_identity_at_init(self):
        model = nn.Module()
        model.conv = ModulatedConv(4, 6, 3, style_dim=5)
        x, style = torch.randn(2, 4, 8, 8), torch.randn(2, 5)
        base = model.conv(x, style)

        layout = WeightLayout.infer(model.conv.weight.shape)
        element = DoRAElement(
            "conv", rank=2, alpha=2.0, in_features=4, out_features=6, kernel_size=(3, 3),
            layout=layout.to_metadata()
        )
        with torch.no_grad():
            element.params["magnitude"].copy_(layout.to_matrix(model.conv.weight).norm(dim=1))

        actant = Actant(model)
        actant.activate(lora_grating(element), injection_strategy="hook")
        self.assertTrue(hasattr(model.conv, "parametrizations"))
        self.assertTrue(torch.allclose(model.conv(x, style), base, atol=1e-5))
        actant.deactivate()


if __name__ == "__main__":
    unittest.main()
