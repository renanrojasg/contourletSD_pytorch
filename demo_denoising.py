"""Demo for image denoising using Contourlets"""
import copy
import os
from argparse import ArgumentParser
from datetime import datetime

import torch
from torchvision.io import read_image

from contourletSD_pytorch.contourlet_ops import (get_norm_scaling,
                                                 hard_thresholding)
from contourletSD_pytorch.contourlet_sd import ContourletSD
from contourletSD_pytorch.contourlet_sd_dec import VALID_INPUT_PRECISION
from utils.image_utils import export_images
from utils.smooth_functions import VALID_SMOOTH_FUNC

if __name__ == "__main__":
  # Parse arguments.
  parser = ArgumentParser()
  parser.add_argument('--dfilt', type=str, default='pkva',
                      help='Filter name for the directional decomposition step.')
  parser.add_argument('--nlev_SD', nargs='+', type=int, default=[
                      2, 2, 3, 4, 5], help='vector of numbers of directional filter bank '
                      'decomposition levels at each pyramidal level (from coarse to fine scale).')
  parser.add_argument('--smooth_func', choices=list(VALID_SMOOTH_FUNC.keys()), default='rcos',
                      help='Function handle to generate the filter for the pyramid decomposition.')
  parser.add_argument(
      '--Pyr_mode', choices=[1, 2], default=1, help='Decomposition mode.')
  parser.add_argument('--image_path', type=str,
                      default='./datasets/test_images/peppers_gray.png', help='Demo test image.')
  parser.add_argument('--sigma', type=float, default=30.0,
                      help='Noise intensity.')
  parser.add_argument('--SDmm_dir', type=str, default='./extras/sdmm_matlab',
                      help='Precomputed norm scaling factors.')
  parser.add_argument('--out_dir', type=str,
                      default='./output', help='Output folder.')
  parser.add_argument('--out_ext', type=str,
                      default='jpg', help='Output file extension.')
  parser.add_argument('--color_images', action='store_true',
                      default=False, help='Process color images.')
  parser.add_argument('--input_precision', default='single',
                      choices=list(VALID_INPUT_PRECISION.keys()), help='Input numerical precision.')
  args = parser.parse_args()

  # Date and output folder.
  args.exp_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  args.out_path = os.path.join(args.out_dir, args.exp_name)
  os.makedirs(args.out_path, exist_ok=True)

  # Get contourlets operator.
  contourlet_sd = ContourletSD(
      nlevs=args.nlev_SD,
      Pyr_mode=args.Pyr_mode,
      smooth_func=VALID_SMOOTH_FUNC[args.smooth_func],
      dfilt=args.dfilt,
      color_mode='rgb' if args.color_images else 'gray',
  )

  # Read grayscale image.
  X = read_image(args.image_path).unsqueeze(0)
  X = X.to(VALID_INPUT_PRECISION[args.input_precision])

  # Add Gaussian noise.
  Xn = X + args.sigma * torch.randn_like(X)

  # Get contourlet coefficients.
  Y = contourlet_sd(x=Xn)

  if args.sigma > 0.0:
    # Load pre-computed norm scaling factors for each subband.
    E = get_norm_scaling(
        image_size=Xn.shape[-1],
        SDmm_dir=args.SDmm_dir,
        Pyr_mode=args.Pyr_mode,
        device=Xn.device,
    )

    # Apply hard thresholding on coefficients.
    Yth = hard_thresholding(
        y=Y,
        sigma=args.sigma,
        E=E,
    )
  else:
    Yth = copy.deepcopy(Y)

  # Reconstruct.
  Xd = contourlet_sd(x=Yth, reconstruct=True)

  # Export results.
  export_images(
      images=X.clamp(0, 255) / 255,
      out_name='input',
      out_path=args.out_path,
      out_ext=args.out_ext,
  )
  export_images(
      images=Xn.clamp(0, 255) / 255,
      out_name='noisy',
      out_path=args.out_path,
      out_ext=args.out_ext,
  )
  export_images(
      images=Xd.clamp(0, 255) / 255,
      out_name='reconstruction',
      out_path=args.out_path,
      out_ext=args.out_ext,
  )
