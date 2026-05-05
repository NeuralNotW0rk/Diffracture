import argparse
import torch
import json
from typing import Dict
from pathlib import Path

from diffracture.kernels.lora_kernel import LoRAKernel
from diffracture.kernels.dora_kernel import DoRAKernel
from diffracture.topology.lora import LoRAPrism
from diffracture.topology.dora import DoRAPrism
from diffracture.topology.lattice import Lattice

def convert_loraw_to_lattice(old_state_dict: Dict[str, torch.Tensor], multiplier: float, alpha: float = None) -> Lattice:
    """
    Converts a legacy LoRAW state_dict into a modern Diffracture Lattice.
    """
    lattice = Lattice()
    lattice.add_kernel("lora", LoRAKernel())
    lattice.add_kernel("dora", DoRAKernel())
    
    grouped_weights = {}

    for key, tensor in old_state_dict.items():
        # Ignore biases saved by legacy nn.Linear modules
        if key.endswith(".bias"):
            continue
            
        if "/lora_down.weight" in key:
            base_address = key.split('/lora_down')[0]
            param_type = "lora_down"
        elif "/lora_up.weight" in key:
            base_address = key.split('/lora_up')[0]
            param_type = "lora_up"
        elif "/dora_mag.weight" in key:
            base_address = key.split('/dora_mag')[0]
            param_type = "magnitude"
        else:
            continue
        
        base_address = base_address.replace('/', '.')
        
        if base_address not in grouped_weights:
            grouped_weights[base_address] = {}
        grouped_weights[base_address][param_type] = tensor

    for address, weights in grouped_weights.items():
        down_w = weights["lora_down"]
        up_w = weights["lora_up"]
        mag_w = weights.get("magnitude")
        
        rank = down_w.size(0)
        
        # If alpha wasn't explicitly provided, assume it equals the rank (scale = 1.0)
        layer_alpha = alpha if alpha is not None else float(rank)
        # Combine legacy multiplier and alpha into a single effective alpha
        effective_alpha = layer_alpha * multiplier
        
        is_dora = (mag_w is not None)
        prism_cls = DoRAPrism if is_dora else LoRAPrism

        prism = prism_cls(
            address=address,
            rank=rank,
            alpha=effective_alpha,
            in_features=down_w.size(1),
            out_features=up_w.size(0),
            kernel_size=down_w.shape[2:] if down_w.dim() > 2 else None
        )
        
        with torch.no_grad():
            prism.params["lora_down"].copy_(down_w)
            prism.params["lora_up"].copy_(up_w)
            if is_dora:
                prism.params["magnitude"].copy_(mag_w.view(-1))
        
        lattice.add_prism(prism)
        
    return lattice

def convert_path(input_path: Path, output_path: Path, fallback_multiplier: float, fallback_alpha: float = None, save_topology: bool = False):
    multiplier = fallback_multiplier
    alpha = fallback_alpha
    
    # Dynamically match and load the config file
    parts = input_path.stem.split("_")
    if len(parts) > 1:
        name = "_".join(parts[:-1])
        config_path = input_path.parent / f"{name}_config.json"
        if config_path.exists():
            print(f"Found matching config: {config_path.name}")
            with open(config_path, "r") as f:
                config = json.load(f)
            
            lora_cfg = config.get("lora", config)
            multiplier = lora_cfg.get("multiplier", fallback_multiplier)
            alpha = lora_cfg.get("alpha", fallback_alpha)
            print(f"  -> Using alpha={alpha}, multiplier={multiplier}")

    print(f"Loading legacy weights from {input_path}...")
    old_state_dict = torch.load(input_path, map_location="cpu")
    
    print("Converting to Diffracture Lattice...")
    lattice = convert_loraw_to_lattice(old_state_dict, multiplier, alpha=alpha)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topology_path = str(output_path.with_suffix('.json')) if save_topology else None
    
    lattice.save(str(output_path), topology_path=topology_path)
    print(f"Saved Diffracture Lattice to: {output_path}")
    if save_topology:
        print(f"Exported optional topology JSON to: {topology_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert legacy LoRAW checkpoints to Diffracture Lattices.")
    parser.add_argument("input", type=str, help="Path to a single .ckpt/.pt file, or a directory of checkpoints.")
    parser.add_argument("output", type=str, help="Path to save the converted .safetensors Lattice, or output directory.")
    parser.add_argument("--alpha", type=float, default=None, help="The alpha value used during legacy training (defaults to rank).")
    parser.add_argument("--multiplier", type=float, default=1.0, help="The multiplier used during legacy training.")
    parser.add_argument("--save-topology", action="store_true", help="Export the topology to a separate JSON file.")
    args = parser.parse_args()
    
    in_path = Path(args.input)
    out_path = Path(args.output)
    
    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        for file_path in in_path.glob("*.*"):
            if file_path.suffix in [".pt", ".ckpt", ".safetensors", ".bin"]:
                dest_file = out_path / f"{file_path.stem}.safetensors"
                convert_path(file_path, dest_file, fallback_multiplier=args.multiplier, fallback_alpha=args.alpha, save_topology=args.save_topology)
    else:
        # If the user specified a directory as the output for a single file, append the filename
        if out_path.is_dir() or out_path.suffix == "":
            out_path = out_path / f"{in_path.stem}.safetensors"
        convert_path(in_path, out_path, fallback_multiplier=args.multiplier, fallback_alpha=args.alpha, save_topology=args.save_topology)

if __name__ == "__main__":
    main()