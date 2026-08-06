import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_kernel import Kernel
from ..registry import register_kernel


class BendingKernel(Kernel):
    """
    Base stateless mathematical kernel for network-bending activation transforms.
    
    Bending kernels apply modifications directly to activations (non-additive relative
    to the module parameterization itself).
    """
    traits = {
        "is_additive": False,
        "requires_weight_fusion": False
    }

    def is_supported(self, original_module: nn.Module) -> bool:
        # Activation transforms can theoretically run on any layer that produces output activations.
        return True

    def _get_channel_dim(self, x: Tensor, original_module: nn.Module) -> int:
        if isinstance(original_module, nn.Linear):
            return x.ndim - 1
        elif isinstance(original_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return 1
        else:
            if x.ndim == 4:
                return 1
            return x.ndim - 1

    def _get_indices_tensor(self, x: Tensor, element, channel_dim: int) -> Tensor:
        indices = element.metadata.get("indices")
        cluster = element.metadata.get("cluster")
        cluster_map = element.metadata.get("cluster_map")
        
        # Resolve indices dynamically if cluster and map are provided and indices is empty
        if (indices is None or len(indices) == 0) and cluster is not None and cluster_map is not None:
            resolved = []
            try:
                target_cluster = int(cluster)
            except (ValueError, TypeError):
                target_cluster = cluster
                
            for item in cluster_map:
                item_cluster = item.get("cluster_index")
                try:
                    curr_cluster = int(item_cluster) if item_cluster is not None else None
                except (ValueError, TypeError):
                    curr_cluster = item_cluster
                if curr_cluster == target_cluster:
                    resolved.append(item.get("feature_index"))
            indices = resolved

        if indices is None or len(indices) == 0:
            # Fallback: Apply to all channels along the resolved channel axis
            indices = list(range(x.size(channel_dim)))
        return torch.tensor(indices, dtype=torch.long, device=x.device)

    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor, element, original_module) -> Tensor:
        if not element.active:
            return original_module(x)

        base_output = original_module(x)
        modified_output = self._apply_transform(base_output, element, original_module)

        # Apply strength blending (using multiplier/strength attributes)
        multiplier = getattr(element, "multiplier", 1.0)
        strength = getattr(element, "strength", 1.0)
        effective_multiplier = multiplier * strength

        if effective_multiplier == 1.0:
            return modified_output
        elif effective_multiplier == 0.0:
            return base_output
        else:
            return (1.0 - effective_multiplier) * base_output + effective_multiplier * modified_output

    def forward_delta(self, x: Tensor, base_output: Tensor, element, original_module) -> Tensor:
        # Standard hook injector calculates: output + forward_delta
        # Returning (modified - base) achieves standard override: base + (modified - base) = modified.
        modified_output = self._apply_transform(base_output, element, original_module)

        multiplier = getattr(element, "multiplier", 1.0)
        strength = getattr(element, "strength", 1.0)
        effective_multiplier = multiplier * strength

        delta = modified_output - base_output
        return delta * effective_multiplier

    def compute_delta(self, element, original_module) -> dict:
        raise NotImplementedError("Bending kernels do not support static weight fusion (collapse).")


@register_kernel("erode")
class ErodeKernel(BendingKernel):
    """
    Kernel for morphological erosion.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"ErodeKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        radius = element.metadata.get("radius", 1)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        kernel_size = 2 * radius + 1
        eroded = -F.max_pool2d(-sliced, kernel_size=kernel_size, stride=1, padding=radius)

        x_mod.index_copy_(channel_dim, idx_tensor, eroded)
        return x_mod


@register_kernel("dilate")
class DilateKernel(BendingKernel):
    """
    Kernel for morphological dilation.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"DilateKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        radius = element.metadata.get("radius", 1)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        kernel_size = 2 * radius + 1
        dilated = F.max_pool2d(sliced, kernel_size=kernel_size, stride=1, padding=radius)

        x_mod.index_copy_(channel_dim, idx_tensor, dilated)
        return x_mod


@register_kernel("translate")
class TranslateKernel(BendingKernel):
    """
    Kernel for 2D spatial translation.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"TranslateKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        offset_x = element.metadata.get("offset_x", 0.0)
        offset_y = element.metadata.get("offset_y", 0.0)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        theta = torch.tensor([
            [1.0, 0.0, offset_x],
            [0.0, 1.0, offset_y]
        ], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(sliced.size(0), 1, 1)
        grid = F.affine_grid(theta, sliced.size(), align_corners=False)
        translated = F.grid_sample(sliced, grid, align_corners=False, mode="bilinear", padding_mode="zeros")

        x_mod.index_copy_(channel_dim, idx_tensor, translated)
        return x_mod


@register_kernel("scale")
class ScaleKernel(BendingKernel):
    """
    Kernel for 2D spatial scaling.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"ScaleKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        scale_factor = element.metadata.get("scale_factor", 1.0)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        inv_scale = 1.0 / (scale_factor + 1e-8)
        theta = torch.tensor([
            [inv_scale, 0.0, 0.0],
            [0.0, inv_scale, 0.0]
        ], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(sliced.size(0), 1, 1)
        grid = F.affine_grid(theta, sliced.size(), align_corners=False)
        scaled = F.grid_sample(sliced, grid, align_corners=False, mode="bilinear", padding_mode="zeros")

        x_mod.index_copy_(channel_dim, idx_tensor, scaled)
        return x_mod


@register_kernel("rotate")
class RotateKernel(BendingKernel):
    """
    Kernel for 2D spatial rotation.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"RotateKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        angle = element.metadata.get("angle", 0.0)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        import math
        angle_rad = angle * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        theta = torch.tensor([
            [cos_a, sin_a, 0.0],
            [-sin_a, cos_a, 0.0]
        ], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(sliced.size(0), 1, 1)
        grid = F.affine_grid(theta, sliced.size(), align_corners=False)
        rotated = F.grid_sample(sliced, grid, align_corners=False, mode="bilinear", padding_mode="zeros")

        x_mod.index_copy_(channel_dim, idx_tensor, rotated)
        return x_mod


@register_kernel("resize")
class ResizeKernel(BendingKernel):
    """
    Kernel for 2D spatial resizing.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"ResizeKernel expects a 4D spatial tensor (B, C, H, W), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        scale_x = element.metadata.get("scale_x", 1.0)
        scale_y = element.metadata.get("scale_y", 1.0)

        new_h = int(x.size(2) * scale_y)
        new_w = int(x.size(3) * scale_x)
        if new_h == x.size(2) and new_w == x.size(3):
            return x

        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    def __call__(self, x: Tensor, element, original_module) -> Tensor:
        if not element.active:
            return original_module(x)

        base_output = original_module(x)
        modified_output = self._apply_transform(base_output, element, original_module)

        if base_output.shape != modified_output.shape:
            # Check effective multiplier
            multiplier = getattr(element, "multiplier", 1.0)
            strength = getattr(element, "strength", 1.0)
            effective_multiplier = multiplier * strength
            if effective_multiplier != 1.0 and effective_multiplier != 0.0:
                raise ValueError(
                    f"ResizeKernel does not support shape changes (from {base_output.shape} to {modified_output.shape}) "
                    f"when strength blending is active (strength/multiplier = {effective_multiplier}). "
                    f"Blending requires identical spatial dimensions."
                )

        # Apply strength blending safely
        multiplier = getattr(element, "multiplier", 1.0)
        strength = getattr(element, "strength", 1.0)
        effective_multiplier = multiplier * strength

        if effective_multiplier == 1.0:
            return modified_output
        else:
            return base_output

    def forward_delta(self, x: Tensor, base_output: Tensor, element, original_module) -> Tensor:
        modified_output = self._apply_transform(base_output, element, original_module)
        if base_output.shape != modified_output.shape:
            raise ValueError(
                f"ResizeKernel does not support shape changes (from {base_output.shape} to {modified_output.shape}) "
                f"during Hook injection or delta computation because they require identical spatial dimensions. "
                f"Consider using 'graft' strategy with strength=1.0 instead."
            )
        return super().forward_delta(x, base_output, element, original_module)


@register_kernel("flip-h")
class FlipHorizontalKernel(BendingKernel):
    """
    Kernel for horizontal reflection.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim < 3:
            raise ValueError(
                f"FlipHorizontalKernel expects a tensor with at least 3 dimensions "
                f"to resolve the spatial width axis (last dimension), "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        flipped = sliced.flip([-1])

        x_mod.index_copy_(channel_dim, idx_tensor, flipped)
        return x_mod


@register_kernel("flip-v")
class FlipVerticalKernel(BendingKernel):
    """
    Kernel for vertical reflection.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        if x.ndim < 3:
            raise ValueError(
                f"FlipVerticalKernel expects a tensor with at least 3 dimensions "
                f"to resolve the spatial height axis, "
                f"but received a {x.ndim}D tensor of shape {x.shape}."
            )
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        flip_dim = -2 if x.ndim >= 3 else -1
        flipped = sliced.flip([flip_dim])

        x_mod.index_copy_(channel_dim, idx_tensor, flipped)
        return x_mod


@register_kernel("invert")
class InvertKernel(BendingKernel):
    """
    Kernel for activation value inversion.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        inverted = 1.0 - sliced

        x_mod.index_copy_(channel_dim, idx_tensor, inverted)
        return x_mod


@register_kernel("binary-thresh")
class BinaryThresholdKernel(BendingKernel):
    """
    Kernel for thresholding activation values.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        threshold = element.metadata.get("threshold", 0.0)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        threshed = (sliced > threshold).to(dtype=sliced.dtype)

        x_mod.index_copy_(channel_dim, idx_tensor, threshed)
        return x_mod


@register_kernel("scalar-multiply")
class ScalarMultiplyKernel(BendingKernel):
    """
    Kernel for channel scalar multiplication.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        factor = element.metadata.get("factor", 1.0)
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        multiplied = sliced * factor

        x_mod.index_copy_(channel_dim, idx_tensor, multiplied)
        return x_mod


@register_kernel("ablate")
class AblateKernel(BendingKernel):
    """
    Kernel for zeroing/ablating activations.
    """
    def _apply_transform(self, x: Tensor, element, original_module) -> Tensor:
        channel_dim = self._get_channel_dim(x, original_module)
        idx_tensor = self._get_indices_tensor(x, element, channel_dim)

        x_mod = x.clone()
        sliced = x.index_select(channel_dim, idx_tensor)

        ablated = sliced * 0.0

        x_mod.index_copy_(channel_dim, idx_tensor, ablated)
        return x_mod
