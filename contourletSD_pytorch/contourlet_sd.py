"""Contourlets Decomposition and Reconstruction"""
from torch import nn

from .contourlet_sd_dec import (VALID_COLOR_MODES, VALID_LADDER_DFILTS,
                                PyrNDDEC_mm, dfbdec_l)
from .contourlet_sd_rec import PyrNDRec_mm, dfbrec_l


class ContourletSD(nn.Module):
  """ContourletSD Decomposition"""

  def __init__(self, nlevs, Pyr_mode, smooth_func, dfilt, color_mode='gray'):
    super().__init__()
    assert color_mode in VALID_COLOR_MODES, 'Invalid color mode. Check color_mode input argument.'

    self.nlevs = nlevs
    self.Pyr_mode = Pyr_mode
    self.smooth_func = smooth_func
    self.dfilt = dfilt
    self.L = len(nlevs)
    self.color_mode = color_mode

    if self.dfilt in VALID_LADDER_DFILTS:
      # Use the ladder structure (very efficient).
      self.dfbdec = dfbdec_l
      self.dfbrec = dfbrec_l
    else:
      # General case
      raise ValueError('General method currently unimplemented.')

  def group_rgb_coeffs(self, y, num_channels):
    for idx in range(len(y)):
      if isinstance(y[idx], list):
        y[idx] = [c.squeeze(1).unflatten(dim=0, sizes=(
            c.shape[0]//num_channels, num_channels)) for c in y[idx]]
      else:
        y[idx] = y[idx].squeeze(1).unflatten(
            dim=0, sizes=(y[idx].shape[0]//num_channels, num_channels))
    return y

  def ungroup_rgb_coeffs(self, y):
    for idx in range(len(y)):
      if isinstance(y[idx], list):
        y[idx] = [c.flatten(0, 1).unsqueeze(1) for c in y[idx]]
      else:
        y[idx] = y[idx].flatten(0, 1).unsqueeze(1)
    return y

  def forward_dec(self, x):
    """Compute Contourlet coefficients"""
    if self.color_mode == 'rgb':
      # Merge image channels with batches to process each channel independently.
      num_channels = x.shape[1]
      x = x.flatten(0, 1).unsqueeze(1)

    y = PyrNDDEC_mm(
        X=x,
        OutD='S',
        L=self.L,
        Pyr_mode=self.Pyr_mode,
        smooth_func=self.smooth_func,
    )

    for k in range(1, self.L+1):
      # DFB on the bandpass image.
      y[k] = self.dfbdec(x=y[k], f=self.dfilt,
                         n=self.nlevs[k-1], spatial_dims=(-2, -1))

    if self.color_mode == 'rgb':
      # Group back into channels.
      y = self.group_rgb_coeffs(
          y=y,
          num_channels=num_channels,
      )
    return y

  def forward_rec(self, y):
    """Reconstruct images from their Contourlet coefficients."""
    if self.color_mode == 'rgb':
      # Merge channels with batches to recover each channel independently.
      num_channels = y[0].shape[1]
      y = self.ungroup_rgb_coeffs(y=y)

    for k in range(2, self.L + 2):
      # Reconstruct the bandpass image from DFB.
      y[k - 1] = self.dfbrec(y=y[k - 1], f=self.dfilt)

    x = PyrNDRec_mm(
        subs=y,
        InD='S',
        Pyr_mode=self.Pyr_mode,
        smooth_func=self.smooth_func,
    )

    if self.color_mode == 'rgb':
      x = x.squeeze(1).unflatten(
          0, sizes=(x.shape[0]//num_channels, num_channels))
    return x

  def forward(self, x, reconstruct=False):
    if reconstruct:
      ret = self.forward_rec(y=x)
    else:
      ret = self.forward_dec(x=x)
    return ret
