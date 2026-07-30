import torch
import torch.nn.functional as F


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        def gelu(x: torch.Tensor) -> torch.Tensor:
            return F.gelu(x.float()).type_as(x)
        return gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError(f"--activation-fn {activation} not supported")