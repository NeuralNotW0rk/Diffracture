import torch
import torch.nn as nn
from diffracture.topology.grating import Grating
from diffracture.actant import Actant
from diffracture.registry import ELEMENT_REGISTRY, KERNEL_REGISTRY
from diffracture.topology.bending import RotateElement, InvertElement, DilateElement

def test_bending_transforms():
    print("=== Testing Network Bending Elements & Kernels ===")
    
    # 1. Verify Registration
    expected_keys = [
        "erode", "dilate", "translate", "scale", "rotate", "resize",
        "flip-h", "flip-v", "invert", "binary-thresh", "scalar-multiply", "ablate"
    ]
    
    print("Registered Elements:", list(ELEMENT_REGISTRY.keys()))
    print("Registered Kernels:", list(KERNEL_REGISTRY.keys()))
    
    for key in expected_keys:
        assert key in ELEMENT_REGISTRY, f"Element '{key}' not registered!"
        assert key in KERNEL_REGISTRY, f"Kernel '{key}' not registered!"
    print("[OK] All 12 elements and kernels are successfully registered.")

    # 2. Build a mock model containing a Conv2d and Linear layer
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
            # Initialize weights to all ones to make checks predictable
            nn.init.constant_(self.conv.weight, 1.0)
            
            self.fc = nn.Linear(8, 4, bias=False)
            nn.init.constant_(self.fc.weight, 0.5)

        def forward(self, x):
            x = self.conv(x)
            x_flat = x.mean(dim=[2, 3]) # Simple pooling to feed into linear
            return self.fc(x_flat)

    model = MockModel()
    
    # Create dummy spatial input (B=1, C=3, H=4, W=4)
    # Fill with a specific pattern
    x = torch.ones(1, 3, 4, 4) * 0.5

    # Run base model forward pass
    base_out = model(x)
    print("Base output:", base_out)

    # 3. Build Grating with Bending Elements
    grating = Grating()
    
    # Apply Invert transform on channels [0, 1] of self.conv
    # Invert: 1.0 - activation. Since conv weights are ones, activation output will be positive.
    # We will invert it.
    invert_el = InvertElement(address="conv", indices=[0, 1])
    grating.add_element(invert_el)
    
    # Apply Rotate transform on channels [2, 3] of self.conv
    rotate_el = RotateElement(address="conv", angle=90.0, indices=[2, 3])
    grating.add_element(rotate_el)

    # Apply Dilate transform on channels [4, 5] of self.conv
    dilate_el = DilateElement(address="conv", radius=1, indices=[4, 5])
    grating.add_element(dilate_el)

    # 4. Activate Grating using Actant (Hook Injection)
    actant = Actant(model)
    print("Activating Hook injection...")
    actant.activate(grating, injection_strategy="hook")
    
    # Run forward pass with hooks active
    hooked_out = model(x)
    print("Hooked output:", hooked_out)
    
    # Deactivate and restore model
    print("Deactivating hooks...")
    actant.deactivate()
    
    restored_out = model(x)
    print("Restored output:", restored_out)
    
    # 5. Assertions to confirm it works
    assert torch.allclose(restored_out, base_out), "Model was not successfully restored to base state!"
    assert not torch.allclose(hooked_out, base_out), "Hooked output did not diverge from base output!"
    print("[OK] Hook-based activation intervention verified successfully!")

if __name__ == "__main__":
    test_bending_transforms()
