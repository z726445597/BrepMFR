import torch


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    """fairseq 0.12.3 without the optional apex backend == torch.nn.LayerNorm."""
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)