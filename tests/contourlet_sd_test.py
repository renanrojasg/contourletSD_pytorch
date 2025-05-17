"""Unit-tests for contourlet_sd"""
import unittest

import torch

from contourletSD_pytorch.contourlet_sd import ContourletSD
from contourletSD_pytorch.contourlet_sd_dec import VALID_INPUT_PRECISION
from utils.smooth_functions import VALID_SMOOTH_FUNC


class TestContourletSD(unittest.TestCase):
  def test_single_scale_contourletsd_reconstructs_image(self):
    """Check if a single scale ContourletSD transform accurately reconstructs an image."""

    # Contourlet settings.
    nlev_SD = [1]
    Pyr_mode = 1
    smooth_func = 'rcos'
    dfilt = 'pkva'
    color_images = False
    input_precision = 'double'

    # Get random image.
    x = 255 * torch.rand(4, 1, 32, 32)
    x = x.to(VALID_INPUT_PRECISION[input_precision])

    # Set Contourlet transform.
    contourlet_sd = ContourletSD(
      nlevs=nlev_SD,
      Pyr_mode=Pyr_mode,
      smooth_func=VALID_SMOOTH_FUNC[smooth_func],
      dfilt=dfilt,
      color_mode='rgb' if color_images else 'gray',
    )

    # Get reconstruction.
    y = contourlet_sd(x=x)
    x_hat = contourlet_sd(x=y, reconstruct=True)

    # Check if the reconstruction is almost equal to the original image.
    self.assertTrue(torch.allclose(x, x_hat, atol=1e-5))

  def test_multi_scale_contourletsd_reconstructs_image(self):
    """Check if a multiscale ContourletSD transform accurately reconstructs an image."""

    # Contourlet settings.
    nlev_SD = [1, 2, 2, 4]
    Pyr_mode = 1
    smooth_func = 'rcos'
    dfilt = 'pkva'
    color_images = False
    input_precision = 'double'

    # Get random image.
    x = 255 * torch.rand(4, 1, 64, 64)
    x = x.to(VALID_INPUT_PRECISION[input_precision])

    # Set Contourlet transform.
    contourlet_sd = ContourletSD(
      nlevs=nlev_SD,
      Pyr_mode=Pyr_mode,
      smooth_func=VALID_SMOOTH_FUNC[smooth_func],
      dfilt=dfilt,
      color_mode='rgb' if color_images else 'gray',
    )

    # Get reconstruction.
    y = contourlet_sd(x=x)
    x_hat = contourlet_sd(x=y, reconstruct=True)

    # Check if the reconstruction is almost equal to the original image.
    self.assertTrue(torch.allclose(x, x_hat, atol=1e-5))

  def test_single_scale_contourletsd_keeps_size(self):
    """Check if a single scale ContourletSD transform output keeps the input dimensions."""

    # Contourlet settings.
    nlev_SD = [1]
    Pyr_mode = 1
    smooth_func = 'rcos'
    dfilt = 'pkva'
    color_images = False
    input_precision = 'double'

    # Get random image.
    x = 255 * torch.rand(4, 1, 32, 32)
    x = x.to(VALID_INPUT_PRECISION[input_precision])

    # Set Contourlet transform.
    contourlet_sd = ContourletSD(
      nlevs=nlev_SD,
      Pyr_mode=Pyr_mode,
      smooth_func=VALID_SMOOTH_FUNC[smooth_func],
      dfilt=dfilt,
      color_mode='rgb' if color_images else 'gray',
    )

    # Get reconstruction.
    y = contourlet_sd(x=x)
    x_hat = contourlet_sd(x=y, reconstruct=True)

    # Check if the reconstruction is almost equal to the original image.
    self.assertTrue(x.shape==x_hat.shape)

  def test_multiscale_contourletsd_keeps_size(self):
    """Check if a multiscale ContourletSD transform output keeps the input dimensions."""

    # Contourlet settings.
    nlev_SD = [1, 2, 2, 4]
    Pyr_mode = 1
    smooth_func = 'rcos'
    dfilt = 'pkva'
    color_images = False
    input_precision = 'double'

    # Get random image.
    x = 255 * torch.rand(4, 1, 64, 64)
    x = x.to(VALID_INPUT_PRECISION[input_precision])

    # Set Contourlet transform.
    contourlet_sd = ContourletSD(
      nlevs=nlev_SD,
      Pyr_mode=Pyr_mode,
      smooth_func=VALID_SMOOTH_FUNC[smooth_func],
      dfilt=dfilt,
      color_mode='rgb' if color_images else 'gray',
    )

    # Get reconstruction.
    y = contourlet_sd(x=x)
    x_hat = contourlet_sd(x=y, reconstruct=True)

    # Check if the reconstruction is almost equal to the original image.
    self.assertTrue(x.shape==x_hat.shape)

if __name__ == '__main__':
  unittest.main()
