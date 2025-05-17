"""Filters for the ladder structure network."""
import torch

_FNAME_DICT = {
    'pkva': torch.tensor([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144]),
    'pkva12': torch.tensor([0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144]),
    'pkva8': torch.tensor([0.6302, -0.1924, 0.0930, -0.0403]),
    'pkva6': torch.tensor([0.6261, -0.1794, 0.0688])
}


def ldfilter(fname, dtype, device):
  """Generate filter for the ladder structure network."""
  v = _FNAME_DICT[fname]

  # Symmetric impulse response.
  f = torch.cat([v.flip(0), v], 0).to(dtype).to(device)
  return f
