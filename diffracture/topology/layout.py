import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class WeightLayout:
    """
    Describes how a weight tensor folds into the [out, in] matrix a low-rank delta is built in.
    Any dimension that is neither out_dim nor one of in_dims must have size 1 (e.g. the leading
    broadcast dim of StyleGAN's [1, out, in, k, k] modulated conv weight).
    """
    out_dim: int = 0
    in_dims: Optional[tuple] = None  # None: every dimension after out_dim

    @classmethod
    def from_metadata(cls, layout: Optional[dict]) -> "WeightLayout":
        if not layout:
            return cls()
        in_dims = layout.get("in_dims")
        return cls(out_dim=int(layout.get("out_dim", 0)), in_dims=tuple(in_dims) if in_dims is not None else None)

    @classmethod
    def infer(cls, shape: Sequence[int]) -> "WeightLayout":
        """
        Guesses the layout from a shape alone: the standard [out, in, *kernel] unless the tensor has
        a leading singleton broadcast dim on top of a conv layout. A wrong guess still gives a valid
        low-rank factorization of the same tensor, only grouped along different axes.
        """
        if len(shape) >= 3 and shape[0] == 1:
            return cls(out_dim=1, in_dims=tuple(range(2, len(shape))))
        return cls()

    def to_metadata(self) -> Optional[dict]:
        """Serializes to grating metadata; the standard layout is stored as None."""
        if self.out_dim == 0 and self.in_dims is None:
            return None
        return {"out_dim": self.out_dim, "in_dims": list(self.in_dims) if self.in_dims is not None else None}

    def resolve(self, ndim: int) -> tuple[int, tuple]:
        """Returns (out_dim, in_dims) as non-negative indices for a tensor of the given rank."""
        out_dim = self.out_dim % ndim
        if self.in_dims is None:
            in_dims = tuple(range(out_dim + 1, ndim))
        else:
            in_dims = tuple(d % ndim for d in self.in_dims)
        if out_dim in in_dims or len(set(in_dims)) != len(in_dims):
            raise ValueError(f"Invalid weight layout {self} for a {ndim}-d tensor: dims must be distinct.")
        return out_dim, in_dims

    def is_standard(self, ndim: int) -> bool:
        """True when the tensor is already [out, in, *kernel], the layout torch's linear and conv ops expect."""
        return self.resolve(ndim) == (0, tuple(range(1, ndim)))

    def split_shape(self, shape: Sequence[int]) -> tuple[int, tuple]:
        """Returns (out_features, in_sizes) for a tensor of the given shape."""
        out_dim, in_dims = self.resolve(len(shape))
        for d, size in enumerate(shape):
            if d != out_dim and d not in in_dims and size != 1:
                raise ValueError(
                    f"Weight layout {self} leaves dim {d} of shape {tuple(shape)} unassigned, but it has size {size}."
                )
        return shape[out_dim], tuple(shape[d] for d in in_dims)

    def to_matrix(self, weight: Tensor) -> Tensor:
        """Folds a weight tensor into its [out, in] matrix."""
        out_dim, in_dims = self.resolve(weight.dim())
        out_features, in_sizes = self.split_shape(weight.shape)
        rest = [d for d in range(weight.dim()) if d != out_dim and d not in in_dims]
        return weight.permute(out_dim, *in_dims, *rest).reshape(out_features, math.prod(in_sizes))

    def from_matrix(self, matrix: Tensor, shape: Sequence[int]) -> Tensor:
        """Unfolds an [out, in] matrix back into a tensor of the given weight shape."""
        out_dim, in_dims = self.resolve(len(shape))
        out_features, in_sizes = self.split_shape(shape)
        if matrix.shape != (out_features, math.prod(in_sizes)):
            raise ValueError(
                f"Delta of shape {tuple(matrix.shape)} does not fold into weight shape {tuple(shape)} "
                f"(expected [{out_features}, {math.prod(in_sizes)}])."
            )
        order = (out_dim, *in_dims)
        # Put the out/in axes back in their original relative order; unassigned dims are size 1 so reshape restores them
        unfolded = matrix.reshape(out_features, *in_sizes)
        unfolded = unfolded.permute(*sorted(range(len(order)), key=lambda i: order[i]))
        return unfolded.reshape(torch.Size(shape))
