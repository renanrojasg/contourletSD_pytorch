"""Normalization scaling utilities."""
import copy
import os

import torch
from scipy.io import loadmat

_SDMM_CONFIGS = ['480_15', '512_1', '512_2']


def get_norm_scaling(image_size, SDmm_dir, Pyr_mode, device):
  """Load pre-computed norm scaling factors for each subband"""
  sdmm_config = f'{image_size}_{Pyr_mode}'
  assert f'{image_size}_{Pyr_mode}' in _SDMM_CONFIGS, \
      'Unavailable SDmm configuration. Check transform size and Pyr_mode parameters.'
  E = loadmat(os.path.join(SDmm_dir, f'SDmm_{sdmm_config}'))['E']

  # Squeeze pre-computed norm scaling.
  E = E.squeeze(-1)
  for idx in range(len(E)):
    E[idx] = E[idx].squeeze(0)
    if len(E[idx]) > 1:
      E[idx] = [torch.from_numpy(e).to(device) for e in E[idx]]
    else:
      E[idx] = torch.from_numpy(E[idx]).to(device)

  return E


def hard_thresholding(y, sigma, E):
  """Apply hard thresholding on coefficients"""
  y_th = copy.deepcopy(y)
  for m in range(1, len(y)):
    thresh = 3 * sigma + sigma * (m == len(y) - 1)
    for k in range(len(y[m])):
      y_th[m][k] = y[m][k] * (torch.abs(y[m][k]) > thresh * E[m][k])
  return y_th
