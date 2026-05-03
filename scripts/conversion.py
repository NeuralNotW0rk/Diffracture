import torch
import torch.nn as nn
from typing import Dict

# Import your newly drafted components
# (Ensure these are in your python path or the same directory)
import sys
import os
import json

from diffracture.kernels.lora_kernel import LoRAKernel, LORA_HANDLERS
from diffracture.injection.graft_injector import GraftInjector
from diffracture.topology.lora import LoRAPrism
from diffracture.topology.lattice import Lattice
from diffracture.actant import Actant

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict



def convert_loraw_to_lattice(old_state_dict: Dict[str, torch.Tensor]) -> Lattice:
    """
    Converts a legacy LoRAW state_dict into a modern Diffracture Lattice.
    """
    lattice = Lattice()
    lattice.add_kernel("lora", LoRAKernel())
    
    # Group 'up' and 'down' weights by their base layer address
    grouped_weights = {}

    for key, tensor in old_state_dict.items():
        # LoRAW keys usually look like: 'path.to.module.lora_down.weight'
        # 1. Strip the parameter suffix to find the base module address
        print(key)

        if ".lora_down" in key:
            base_address = key.split('.lora_down')[1]
            param_type = "lora_down"
        elif ".lora_up" in key:
            base_address = key.split('.lora_up')[1]
            param_type = "lora_up"
        else:
            continue
        
        if base_address not in grouped_weights:
            grouped_weights[base_address] = {}
        grouped_weights[base_address][param_type] = tensor

    for address, weights in grouped_weights.items():
        # 2. Convert dots to slashes for 'Self-Documenting' format
        # Note: You may need to adjust this based on how LoRAW stored paths
        clean_address = address.replace('.', '/') 
        
        down_w = weights["lora_down"]
        up_w = weights["lora_up"]
        
        # 3. Initialize the LoRAPrism with inferred shapes
        # rank is the size of the inner dimension
        prism = LoRAPrism(
            address=clean_address,
            rank=down_w.size(0),
            alpha=down_w.size(0), # Default alpha=rank if not explicitly in old dict
            in_features=down_w.size(1),
            out_features=up_w.size(0),
            # Check if it's a Conv layer by looking at dimension count
            kernel_size=down_w.shape[2:] if down_w.dim() > 2 else None
        )
        
        # 4. Copy old weights into the new ParameterDict
        with torch.no_grad():
            prism.params["lora_down"].copy_(down_w)
            prism.params["lora_up"].copy_(up_w)
        
        lattice.add_prism(prism)
        
    return lattice

def run_validation_test(model: nn.Module, old_checkpoint_path: str):
    """
    Main test execution flow.
    """
    print(f"--- Starting Validation Test ---")
    
    # 1. Load legacy weights
    print(f"Loading old checkpoint from {old_checkpoint_path}...")
    old_state_dict = torch.load(old_checkpoint_path, map_location="cpu")
    
    # 2. Convert to Lattice
    print("Converting to Manifracture Lattice structure...")
    lattice = convert_loraw_to_lattice(old_state_dict)
    
    # 3. Move Lattice to your 4070 Ti Super
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    lattice.to(device)
    
    # 4. Orchestrate Injection via Actant
    actant = Actant(model, lattice, GraftInjector(), LORA_HANDLERS)
    
    print("Activating model divergence...")
    actant.activate()
    
    # 5. Verification
    print(f"Injection Complete. Verified device: {next(model.parameters()).device}")
    
    # Dummy inference to ensure no shape mismatches
    # Adjust dummy_input shape to match your specific audio model requirements
    # dummy_input = torch.randn(1, 1, 16000).to(device) 
    # output = model(dummy_input)
    # print(f"Inference successful. Output shape: {output.shape}")

if __name__ == "__main__":
    # To run this, replace 'your_audio_model' with an instance of your model 
    # and provide the path to an old .pt or .safetensors checkpoint.
    config_path = "C:/Users/griff/Proton Drive/griffinpage9/My files/Models/stable_audio/stable_audio_open_1/model_config.json"
    base_ckpt = "C:/Users/griff/Proton Drive/griffinpage9/My files/Models/stable_audio/stable_audio_open_1/model.safetensors"
    lora_ckpt = "C:/Users/griff/Proton Drive/griffinpage9/My files/Models/stable_audio/stable_audio_open_1/loras/gwf128_6000.ckpt"

    with open(config_path, 'r') as cf:
        config_json = cf.read()
        config = json.loads(config_json)

    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(base_ckpt))
    
    run_validation_test(model, lora_ckpt)