"""Collection of smooth functions"""
import torch


def rcos(x: torch.Tensor) -> torch.Tensor:
  """Raised cosine function"""
  theta = 0.5 * (1 - torch.cos(torch.pi * x))
  theta[x <= 0] = 0
  theta[x >= 1] = 1
  return theta


VALID_SMOOTH_FUNC = {
    'rcos': rcos
}