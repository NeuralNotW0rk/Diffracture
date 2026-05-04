import torch
import torch.nn as nn
import json
from pathlib import Path

from diffracture.topology.lattice import Lattice
from diffracture.actant import Actant

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

def run_validation_test(model: nn.Module, lattice_ckpt: str):
    """
    Main execution flow to validate Lattice injection into a target model.
    """
    print(f"--- Starting Validation Test ---")
    
    print(f"Loading Lattice from {lattice_ckpt}...")
    lattice = Lattice.load(lattice_ckpt)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    lattice.to(device)
    
    # Orchestrate Injection via Actant
    print("Orchestrating Actant...")
    actant = Actant(model, lattice)
    
    print("Activating...")
    actant.activate(injector_name="graft")
    
    print(f"Injection Complete. Verified device: {next(model.parameters()).device}")

if __name__ == "__main__":
    lora_name = "gwf128_6000"
    
    model_dir = Path("C:/Users/griff/Proton Drive/griffinpage9/My files/Models/stable_audio/stable_audio_open_1/")
    config_path = model_dir / "model_config.json"
    base_ckpt = model_dir / "model.safetensors"
    lattice_ckpt = model_dir / "diffracture" / "loras" / f"{lora_name}.safetensors"
    
    with open(config_path, 'r') as cf:
        config = json.loads(cf.read())

    print("Loading base model...")
    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(str(base_ckpt)))
    
    run_validation_test(model, str(lattice_ckpt))