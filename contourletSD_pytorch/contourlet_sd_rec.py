"""ContourletSD Reconstruction Tools"""

import copy
from math import ceil, log2, sqrt

import torch

from .contourlet_sd_dec import (VALID_PYR_MODES,
                                VALID_QPDEC_TYPES,
                                VALID_RESAMP_TYPES, ccsym,
                                resamp, sefilter2)

from .ldfilter import ldfilter


def PrySDrec_onestep(Lp, Hp, w, tbw, D, smooth_func, spatial_dims):
  """N-dimensional multiscale pyramid reconstruction - One Step."""
  # The dimension of the problem.
  N = len(spatial_dims)
  # Spatial support.
  szX = torch.Tensor([Hp.shape[d] for d in spatial_dims]).int()

  # The original full size.
  # Spatial support.
  szF = torch.Tensor([Hp.shape[d] for d in spatial_dims]).int()
  szF[-1] = (szX[-1] - 1) * 2

  # Passband index arrays.
  pbd_array = []

  # Lowpass filter (nonzero part).
  szLf = 2 * torch.ceil(szF / 2 * (w + tbw)) + 1
  szLf[-1] = (szLf[-1] + 1) / 2
  Lf = torch.ones(szLf.int().tolist(), device=Lp.device)

  szR = torch.ones(N, dtype=torch.int32)  # For resizing.

  for n in range(N):
    # Current Fourier domain resolution.
    nr = szF[n]
    szR[n] = szLf[n]

    # Passband.
    pbd = torch.arange(ceil(nr / 2 * (w + tbw)) + 1, device=Lp.device)

    # Passband value.
    pbd_value_ = smooth_func(
        (pbd[-1] - pbd) / (torch.ceil(nr / 2 * (w + tbw)) - torch.floor(nr / 2 * (w - tbw))))
    assert all(pbd_value_ >= 0), 'Invalid values, must be nonnegative.'
    pbd_value = torch.sqrt(pbd_value_)

    # See if we need to consider the symmetric part.
    if n != N - 1:
      pbd = torch.cat(
          [pbd, torch.arange(nr - pbd[-1], nr, device=Lp.device)], 0)
      pbd_value = torch.cat([pbd_value, pbd_value[1:].flip(0)], 0)

    pbd_array.append(pbd + 1)
    pbd_value = torch.reshape(pbd_value, szR.tolist())
    Lf = Lf * pbd_value.repeat((szLf / szR).int().tolist())

    szR[n] = 1

  # Get the reconstruction.
  rec = copy.deepcopy(Hp)
  grid01, grid02 = torch.meshgrid(
      pbd_array[0] - 1, pbd_array[1] - 1, indexing='ij')

  Lf_ = Lf**2
  assert torch.all(Lf_.real == Lf_), 'Error, complex power outputs.'

  Lf_ = torch.sqrt(1 - Lf_)
  assert torch.all(Lf_.real == Lf_), 'Error, complex square root outputs.'

  rec_ = torch.stack([Lf_ * r[grid01, grid02] for r in rec.flatten(0, 1)],
                     0).unflatten(0, [rec.shape[0], rec.shape[1]])
  rec[:, :, grid01, grid02] = rec_

  # Get the lowpass subband.
  assert torch.all(szF % D) == 0, \
      'The downsampling factor must be able to divide the size of the FFT matrix!'
  szLp = torch.div(szF, D, rounding_mode='trunc').int()
  szLp[-1] = szLp[-1] / 2 + 1

  pbd_array_sml = copy.deepcopy(pbd_array)
  for n in range(N-1):
    pbd = copy.deepcopy(pbd_array[n])
    pbd[(len(pbd) + 3) // 2 - 1:] = pbd[(len(pbd) + 3) // 2 - 1:] + szLp[n] - szX[n]
    pbd_array_sml[n] = copy.deepcopy(pbd)

  grid01, grid02 = torch.meshgrid(
      pbd_array[0] - 1, pbd_array[1] - 1, indexing='ij')
  grid_sml01, grid_sml02 = torch.meshgrid(
      pbd_array_sml[0] - 1, pbd_array_sml[1] - 1, indexing='ij')

  if rec.shape[0] == rec.shape[1] == 1:
    # Single layer, single batch, apply filter directly.
    rec[0, 0, grid01, grid02] = rec[0, 0, grid01, grid02] + \
        Lp[0, 0, grid_sml01, grid_sml02] * (Lf * D**(N / 2))
  else:
    rec_ = (rec[:, :, grid01, grid02])
    rec_ += torch.stack([(Lf * D**(N / 2)) * Lp_[grid_sml01, grid_sml02]
                        for Lp_ in Lp.flatten(0, 1)], 0).unflatten(0, [Lp.shape[0], Lp.shape[1]])
    rec[:, :, grid01, grid02] = rec_

  return rec


def PyrNDRec_mm(subs, InD, Pyr_mode, smooth_func):
  """N-dimensional multiscale pyramid reconstruction - with multiple modes."""

  InD = InD.upper()
  N = len(subs[0].shape) - 2
  L = len(subs) - 1

  assert Pyr_mode in VALID_PYR_MODES, 'Invalid Decomposition (Pyr) mode.'
  if Pyr_mode == 1:
    # the cutoff frequencies at each scale.
    w_array = torch.cat((0.25 * torch.ones(L - 1), torch.tensor([0.5])), 0)

    # the transition bandwidths at each scale.
    tbw_array = torch.cat(
        (1 / 12 * torch.ones(L - 1), torch.tensor([1 / 6])), 0)

    # the downsampling factor at each scale.
    # no downsampling at the finest scale.
    D_array = torch.cat((2 * torch.ones(L - 1), torch.Tensor([1])), 0)
  elif Pyr_mode == 1.5:
    # the cutoff frequencies at each scale.
    w_array = torch.cat((3 / 8 * torch.ones(L - 1), torch.Tensor([0.5])), 0)

    # the transition bandwidths at each scale.
    tbw_array = torch.cat(
        (1 / 9 * torch.ones(L - 1), torch.Tensor([1 / 7])), 0)

    # the downsampling factor at each scale.
    # no downsampling at the finest scale.
    D_array = torch.cat(
        (2 * torch.ones(L - 1), torch.tensor(1.5).unsqueeze(0)), 0)
  else:  # Pyr_mode == 2:
    # the cutoff frequencies at each scale.
    w_array = 1 / 3 * torch.ones(L)

    # the transition bandwidths at each scale.
    tbw_array = 1 / 7 * torch.ones(L)

    # the downsampling factor at each scale.
    # no downsampling at the finest scale.
    D_array = 2 * torch.ones(L)

  Lp = copy.deepcopy(subs[0])
  subs[0] = []
  if InD == 'S':
    Lp = torch.fft.fftn(Lp, dim=(-2, -1))
  Lp = ccsym(x=Lp, k=N-1, ccsym_type='c', spatial_dims=(-2, -1))

  for n in range(1, L+1):
    Hp = copy.deepcopy(subs[n])
    subs[n] = []

    if InD == 'S':
      Hp = torch.fft.fftn(Hp, dim=(-2, -1))
    Hp = ccsym(x=Hp, k=N-1, ccsym_type='c', spatial_dims=(-2, -1))
    Lp = PrySDrec_onestep(
        Lp=Lp,
        Hp=Hp,
        w=w_array[n-1],
        tbw=tbw_array[n-1],
        D=D_array[n-1],
        smooth_func=smooth_func,
        spatial_dims=(-2, -1)
    )

  Lp = ccsym(x=Lp, k=N-1, ccsym_type='e', spatial_dims=(-2, -1))
  rec = torch.real(torch.fft.ifftn(Lp, dim=(-2, -1)))

  return rec


def rebacksamp(y, spatial_dims):
  """Re-backsampling the subband images of the DFB."""
  assert spatial_dims == (-2, -
                          1), 'Spatial dimensions currently not supported.'
  n = log2(len(y))
  assert n == round(
      n) and n >= 1, 'Input must be a cell vector of dyadic length.'
  n = int(n)

  if n == 1:
    # One level, the reconstruction filterbank shoud be Q1r.
    # Redo the first resampling (Q1r = R2 * D1 * R3).
    for k in [0, 1]:
      y[k][..., ::2] = resamp(x=y[k][..., ::2],
                                   resamp_type=2, spatial_dims=spatial_dims)
      y[k][..., 1::2] = resamp(x=y[k][..., 1::2],
                                    resamp_type=2, spatial_dims=spatial_dims)
      y[k] = resamp(x=y[k], resamp_type=3, spatial_dims=spatial_dims)

  elif n > 2:
    N = 2**(n-1)

    for k in range(1, 2**(n-2)+1):
      shift = 2*k - (2**(n-2) + 1)

      # The first half channels.
      y[2*k-2] = resamp(x=y[2*k-2], resamp_type=3, shift=-
                        shift, spatial_dims=spatial_dims)
      y[2*k-1] = resamp(x=y[2*k-1], resamp_type=3, shift=-
                        shift, spatial_dims=spatial_dims)

      # The second half channels.
      y[2*k-2+N] = resamp(x=y[2*k-2+N], resamp_type=1,
                          shift=-shift, spatial_dims=spatial_dims)
      y[2*k-1+N] = resamp(x=y[2*k-1+N], resamp_type=1,
                          shift=-shift, spatial_dims=spatial_dims)

  return y


def qprec(p0, p1, spatial_dims, qprec_type='1r'):
  """Quincunx Polyphase Reconstruction."""
  assert spatial_dims == (-2, -
                          1), 'Spatial dimensions currently not supported.'
  y_shape = list(p0.shape)

  assert qprec_type in VALID_QPDEC_TYPES, 'Invalid polyphase decomposition type.'
  if qprec_type == '1r':
    y_shape[spatial_dims[0]] *= 2
    y = torch.zeros(y_shape, dtype=p0.dtype, device=p0.device)
    y[..., ::2, :] = resamp(x=p0, resamp_type=4, spatial_dims=spatial_dims)
    y[..., 1::2, list(range(1, y_shape[spatial_dims[-1]])) + [0]
      ] = resamp(x=p1, resamp_type=4, spatial_dims=spatial_dims)
    x = resamp(x=y, resamp_type=1, spatial_dims=spatial_dims)

  elif qprec_type == '1c':
    y_shape[spatial_dims[1]] *= 2
    y = torch.zeros(y_shape, dtype=p0.dtype, device=p0.device)
    y[..., :, ::2] = resamp(x=p0, resamp_type=1, spatial_dims=spatial_dims)
    y[..., :, 1::2] = resamp(x=p1, resamp_type=1, spatial_dims=spatial_dims)
    x = resamp(x=y, resamp_type=4, spatial_dims=spatial_dims)

  elif qprec_type == '2r':
    y_shape[spatial_dims[0]] *= 2
    y = torch.zeros(y_shape, dtype=p0.dtype, device=p0.device)
    y[..., ::2, :] = resamp(x=p0, resamp_type=3, spatial_dims=spatial_dims)
    y[..., 1::2, :] = resamp(x=p1, resamp_type=3, spatial_dims=spatial_dims)
    x = resamp(x=y, resamp_type=2, spatial_dims=spatial_dims)

  else:  # qprec_type=='2c':
    y_shape[spatial_dims[1]] *= 2
    y = torch.zeros(y_shape, dtype=p0.dtype, device=p0.device)
    y[..., :, ::2] = resamp(x=p0, resamp_type=2, spatial_dims=spatial_dims)
    y[..., list(range(1, y_shape[spatial_dims[0]])) + [0],
      1::2] = resamp(x=p1, resamp_type=2, spatial_dims=spatial_dims)
    x = resamp(x=y, resamp_type=3, spatial_dims=spatial_dims)

  return x


def pprec(p0, p1, spatial_dims, pprec_type):
  """Parallelogram Polyphase Reconstruction."""
  assert spatial_dims == (-2, -
                          1), 'Spatial dimensions currently not supported.'

  x_shape = list(p0.shape)
  assert pprec_type in VALID_RESAMP_TYPES, 'Invalid polyphase decomposition type.'

  if pprec_type == 1:  # P1 = R1 * Q1 = D1 * R3
    x_shape[spatial_dims[0]] *= 2
    x = torch.zeros(x_shape, dtype=p0.dtype, device=p0.device)
    x[..., ::2, :] = resamp(x=p0, resamp_type=4, spatial_dims=spatial_dims)
    x[..., 1::2, list(range(1, x_shape[spatial_dims[1]])) + [0]
      ] = resamp(x=p1, resamp_type=4, spatial_dims=spatial_dims)

  elif pprec_type == 2:  # P2 = R2 * Q2 = D1 * R4
    x_shape[spatial_dims[0]] *= 2
    x = torch.zeros(x_shape, dtype=p0.dtype, device=p0.device)
    x[..., ::2, :] = resamp(x=p0, resamp_type=3, spatial_dims=spatial_dims)
    x[..., 1::2, :] = resamp(x=p1, resamp_type=3, spatial_dims=spatial_dims)

  elif pprec_type == 3:  # P3 = R3 * Q2 = D2 * R1
    x_shape[spatial_dims[1]] *= 2
    x = torch.zeros(x_shape, dtype=p0.dtype, device=p0.device)
    x[..., :, ::2] = resamp(x=p0, resamp_type=2, spatial_dims=spatial_dims)
    x[..., list(range(1, x_shape[spatial_dims[0]])) + [0],
      1::2] = resamp(x=p1, resamp_type=2, spatial_dims=spatial_dims)

  else:  # pprec_type==4:  # P4 = R4 * Q1 = D2 * R2
    x_shape[spatial_dims[1]] *= 2
    x = torch.zeros(x_shape, dtype=p0.dtype, device=p0.device)
    x[..., :, ::2] = resamp(x=p0, resamp_type=1, spatial_dims=spatial_dims)
    x[..., :, 1::2] = resamp(x=p1, resamp_type=1, spatial_dims=spatial_dims)

  return x


def fbrec_l(y0, y1, f, type1, type2, spatial_dims, extmod='per'):
  """Two-channel 2D Filterbank Reconstruction using Ladder Structure."""
  # Modulate f. Avoid in-place operation.
  f_ = f.clone()
  f_[::2] = -f_[::2]

  # Ladder network structure.
  p1 = (-1 / sqrt(2)) * (y1 + sefilter2(x=y0, f1=f_,
                                        f2=f_, extmod=extmod, spatial_dims=spatial_dims))
  p0 = sqrt(2) * y0 + sefilter2(x=p1, f1=f_,
                                f2=f_, extmod=extmod, shift=[1, 1], spatial_dims=spatial_dims)

  # Polyphase reconstruction.
  if type1.lower()[0] == 'q':
    # Quincunx polyphase reconstruction.
    x = qprec(p0=p0, p1=p1, qprec_type=type2, spatial_dims=spatial_dims)
  elif type1.lower()[0] == 'p':
    # Parallelogram polyphase decomposition.
    x = pprec(p0=p0, p1=p1, pprec_type=type2, spatial_dims=spatial_dims)
  else:
    raise ValueError('Invalid argument type1.')

  return x


def dfbrec_l(y, f):
  """Directional Filterbank Reconstruction using Ladder Structure."""
  n = log2(len(y))

  assert n == round(
      n) and n >= 1, 'Number of reconstruction levels must be a non-negative integer'
  n = int(n)

  if n == 0:
    # Simply copy input to output.
    x = copy.deepcopy(y[0])
  else:
    # Ladder filter.
    if isinstance(f, str):
      f = ldfilter(f, dtype=y[0].dtype, device=y[0].device)

    # Flip back the order of the second half channels.
    y[2**(n-1):] = (y[2**(n-1):])[::-1]

    # Undo backsampling.
    y = rebacksamp(y, spatial_dims=(-2, -1))

    # Tree-structured filter banks.
    if n == 1:
      # Simplest case, one level.
      x = fbrec_l(y0=y[0], y1=y[1], f=f, type1='q',
                  type2='1r', extmod='qper_col', spatial_dims=(-2, -1))
    else:
      # For the cases that n >= 2.
      # Recombine subband outputs to the next level.
      for l in range(n, 2, -1):
        y_old = copy.deepcopy(y)
        y = []

        # The first half channels use R1 and R2.
        for k in range(1, 2**(l-2) + 1):
          i = (k-1) % 2 + 1
          y.append(fbrec_l(y0=y_old[2*k-1], y1=y_old[2*k-2],
                   f=f, type1='p', type2=i, extmod='per', spatial_dims=(-2, -1)))

        # The second half channels use R3 and R4.
        for k in range(2**(l-2)+1, 2**(l-1)+1):
          i = (k-1) % 2 + 3
          y.append(fbrec_l(y0=y_old[2*k-1], y1=y_old[2*k-2],
                   f=f, type1='p', type2=i, extmod='per', spatial_dims=(-2, -1)))

      # Second level.
      x0 = fbrec_l(y0=y[1], y1=y[0], f=f, type1='q',
                   type2='2c', extmod='per', spatial_dims=(-2, -1))
      x1 = fbrec_l(y0=y[3], y1=y[2], f=f, type1='q',
                   type2='2c', extmod='per', spatial_dims=(-2, -1))

      # First level.
      x = fbrec_l(y0=x0, y1=x1, f=f, type1='q', type2='1r',
                  extmod='qper_col', spatial_dims=(-2, -1))

  return x
