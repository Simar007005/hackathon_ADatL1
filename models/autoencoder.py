"""PyTorch anomaly detection models.

Every model here is an autoencoder: a plain ``nn.Module`` that maps an event ``x`` to its
reconstruction ``x_hat``. train.ipynb minimises the masked reconstruction error on normal
(zero-bias) events, so the model learns to reconstruct normal events well, and events it
reconstructs badly get a high anomaly score (see ``masked_mse``).

Constructors receive ``input_shape``: the shape of one event without the batch dimension,
e.g. (117,) for "flat" input or (39, 3) for "objects" input.
"""

import math
import warnings

warnings.filterwarnings("ignore", message=".*fast_hadamard_transform.*")  # optional Brevitas speed-up, not needed
import brevitas.nn as qnn  # noqa: E402
import numpy as np
import torch
from torch import nn

from brevitas.quant import (  # noqa: E402
    Int8ActPerTensorFixedPoint, Int8BiasPerTensorFixedPointInternalScaling, Int8WeightPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
)

from .registry import register


def expand_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast the per-object mask (N, n_objects) to one entry per value of ``x``."""
    if x.dim() == 2:  # flat input
        return mask.repeat_interleave(x.shape[1] // mask.shape[1], dim=1)
    return mask.unsqueeze(-1).expand_as(x)


def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-event mean squared error, ignoring padded objects."""
    mask = expand_mask(mask, x).to(x.dtype)
    x_hat = x_hat.reshape(x.shape)
    err = ((x_hat - x) ** 2 * mask).flatten(1).sum(1)
    return err / mask.flatten(1).sum(1).clamp(min=1.0)


def reconstruction_error(x: np.ndarray, mask: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Masked reconstruction error of every event, given its reconstruction ``x_hat``."""
    x_hat = torch.as_tensor(np.asarray(x_hat, dtype=x.dtype))
    return masked_mse(x_hat, torch.as_tensor(x), torch.as_tensor(mask)).numpy()


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """The model's output for every event, computed in batches on the model's device."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch_size):
        x_hat = model(torch.as_tensor(x[i:i + batch_size], device=device))
        out.append(getattr(x_hat, "value", x_hat).cpu().numpy())  # Brevitas may return a QuantTensor
    return np.concatenate(out)


def anomaly_score(model: nn.Module, x: np.ndarray, mask: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Masked reconstruction error of every event: higher = more anomalous."""
    return reconstruction_error(x, mask, predict(model, x, batch_size))


@register("dense-ae", input_type="flat")
def dense_autoencoder(input_shape, latent_dim: int = 8):
    """A small fully connected autoencoder: input -> 32 -> 16 -> 8 -> 16 -> 32 -> input."""
    n_in = math.prod(input_shape)
    return nn.Sequential(
        nn.Linear(n_in, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, latent_dim),  # latent space
        nn.Linear(latent_dim, 16), nn.ReLU(),
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, n_in),
    )


# ---------------------------------------------------------------------------
# Quantized layers (Brevitas) for quantization-aware training. During training, weights and
# activations are rounded to `bits` bits, as they will be on the FPGA, so the model learns to cope.
# Use these instead of brevitas.nn directly: they use power-of-two scales and quantized biases,
# which is what hls4ml needs to convert the model (see convert_hls4ml.ipynb).
# ---------------------------------------------------------------------------
def quant_input(bits: int = 8):
    """Quantize the values going into the next layer (e.g. the model inputs) to ``bits`` bits."""
    return qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, bit_width=bits)


def quant_dense(n_in: int, n_out: int, bits: int = 8):
    """A dense (Linear) layer with ``bits``-bit weights."""
    return qnn.QuantLinear(n_in, n_out, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=bits,
                           bias_quant=Int8BiasPerTensorFixedPointInternalScaling)


def quant_relu(bits: int = 8):
    """ReLU, with the output quantized to ``bits`` bits."""
    return qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=bits)


@register("dense-qae", input_type="flat")
def quantized_dense_autoencoder(input_shape, latent_dim: int = 8, bits: int = 6):
    """dense-ae with quantization-aware training: ``bits``-bit weights and activations.

    Fewer bits = fewer bit operations (see utils.model_cost).
    """
    n_in = math.prod(input_shape)
    return nn.Sequential(
        quant_input(bits),
        quant_dense(n_in, 32, bits), quant_relu(bits),
        quant_dense(32, 16, bits), quant_relu(bits),
        quant_dense(16, latent_dim, bits),  # latent space
        quant_input(bits),
        quant_dense(latent_dim, 16, bits), quant_relu(bits),
        quant_dense(16, 32, bits), quant_relu(bits),
        quant_dense(32, n_in, bits),
    )


# TODO(participants): add your own models here, register them with @register(...) so the
# notebooks can find them. Ideas: a smaller autoencoder that fits on a trigger FPGA, a
# variational autoencoder, a model with "objects" input that treats each particle
# separately (deep sets / transformer), ...
