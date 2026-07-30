"""Local minimal fairseq shim for BrepMFR (adaptation #1).
BrepMFR uses only six fairseq symbols: utils.softmax, utils.get_activation_fn,
FairseqDropout, quant_noise, LayerNorm, LayerDropModuleList.
Math-equivalent subset of fairseq 0.12.3 (MIT), avoiding fairseq's heavy
import-time deps (hydra/omegaconf/sacrebleu)."""
from . import utils  # noqa: F401

__version__ = "0.12.3-shim"