"""Demo for image denoising using Contourlets"""
import copy
import os
from argparse import ArgumentParser
from datetime import datetime

import torch
from tqdm import tqdm

from contourletSD_pytorch.contourlet_ops import (get_norm_scaling,
                                                 hard_thresholding)
from contourletSD_pytorch.contourlet_sd import ContourletSD
from contourletSD_pytorch.contourlet_sd_dec import VALID_INPUT_PRECISION
from utils.image_utils import export_images, get_dataloader
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
  parser.add_argument('--dataset_path', type=str,
                      default='./datasets/Set12', help='Demo test dataset.')
  parser.add_argument('--sigma', type=float, default=30.0,
                      help='Noise intensity.')
  parser.add_argument('--SDmm_dir', type=str, default='./extras/sdmm_matlab',
                      help='Precomputed norm scaling factors.')
  parser.add_argument('--out_dir', type=str,
                      default='./output', help='Output folder.')
  parser.add_argument('--out_ext', type=str,
                      default='jpg', help='Output file extension.')
  parser.add_argument('--device', type=str,
                      default='cuda', help='Accelerator (device).')
  parser.add_argument('--transform_size', type=int, default=512,
                      help='Preprocessing transform size.')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Images per batch.')
  parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of workers.')
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

  # Create dataloader.
  dataloader = get_dataloader(
      dataset_path=args.dataset_path,
      transform_size=args.transform_size,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
  )

  # Load pre-computed norm scaling factors for each subband.
  if args.sigma > 0.0:
    E = get_norm_scaling(
        image_size=args.transform_size,
        SDmm_dir=args.SDmm_dir,
        Pyr_mode=args.Pyr_mode,
        device=args.device,
    )

  iterator = tqdm(dataloader, total=len(dataloader))
  for data in iterator:
    images, fnames = data
    images = images.to(device=args.device).type(VALID_INPUT_PRECISION[args.input_precision])

    # Add Gaussian noise.
    images_noisy = images + args.sigma * torch.randn_like(images)

    # Get contourlet coefficients.
    coeffs = contourlet_sd(x=images_noisy)

    if args.sigma > 0:
      # Apply hard thresholding on coefficients.
      coeffs_th = hard_thresholding(
          y=coeffs,
          sigma=args.sigma,
          E=E,
      )
    else:
      coeffs_th = copy.deepcopy(coeffs)

    # Reconstruct.
    images_rec = contourlet_sd(x=coeffs_th, reconstruct=True)

    # Export results.
    export_images(
        images=images.clamp(0, 255) / 255,
        fnames=fnames,
        out_name='input',
        out_path=args.out_path,
        out_ext=args.out_ext,
    )
    export_images(
        images=images_noisy.clamp(0, 255) / 255,
        fnames=fnames,
        out_name='noisy',
        out_path=args.out_path,
        out_ext=args.out_ext,
    )
    export_images(
        images=images_rec.clamp(0, 255) / 255,
        fnames=fnames,
        out_name='rec',
        out_path=args.out_path,
        out_ext=args.out_ext,
    )
