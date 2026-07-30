def quant_noise(module, p, block_size):
    """fairseq 0.12.3 trimmed to the only branch BrepMFR uses (q_noise=0.0)."""
    if p <= 0:
        return module
    raise NotImplementedError(
        "quant_noise with p>0 is not implemented in the local fairseq shim; "
        "BrepMFR uses q_noise=0.0 everywhere."
    )