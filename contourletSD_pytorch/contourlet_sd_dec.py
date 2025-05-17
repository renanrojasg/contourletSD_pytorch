"""ContourletSD Decomposition Tools"""
import copy
from math import ceil, floor, log2, sqrt

import torch
from torch import nn

from .extend2 import extend2
from .ldfilter import ldfilter

_VALID_CCSYM_TYPES = ['c', 'e']
VALID_LADDER_DFILTS = ['pkva6', 'pkva8', 'pkva12', 'pkva']
VALID_QPDEC_TYPES = ['1r', '1c', '2r', '2c']
VALID_RESAMP_TYPES = [1, 2, 3, 4]
VALID_PYR_MODES = [1, 1.5, 2]
VALID_COLOR_MODES = ['gray', 'rgb']
VALID_INPUT_PRECISION = {
    'single': torch.float32,
    'double': torch.float64,
}


def backsamp(y, spatial_dims):
  """Input and output are cell vector of dyadic length."""
  assert spatial_dims == (-2, -
                          1), 'Spatial dimensions currently not supported.'
  n = log2(len(y))

  assert n == round(
      n) and n >= 1, 'Input must be a cell vector of dyadic length.'
  n = int(n)
  if n == 1:
    # One level, the decomposition filterbank shoud be Q1r.
    # Undo the last resampling (Q1r = R2 * D1 * R3)
    for k in [0, 1]:
      y[k] = resamp(x=y[k], resamp_type=4, spatial_dims=spatial_dims)
      y[k][..., ::2] = resamp(x=y[k][..., ::2],
                            resamp_type=1, spatial_dims=spatial_dims)
      y[k][..., 1::2] = resamp(x=y[k][..., 1::2],
                             resamp_type=1, spatial_dims=spatial_dims)
  elif n > 2:
    N = int(2**(n-1))

    for k in range(1, 2**(n-2)+1):
      shift = 2*k - (2**(n-2) + 1)

      # The first half channels.
      y[2*k-2] = resamp(x=y[2*k-2], resamp_type=3,
                        shift=shift, spatial_dims=spatial_dims)
      y[2*k-1] = resamp(x=y[2*k-1], resamp_type=3,
                        shift=shift, spatial_dims=spatial_dims)

      # The scond half channels.
      y[2*k-2+N] = resamp(x=y[2*k-2+N], resamp_type=1,
                          shift=shift, spatial_dims=spatial_dims)
      y[2*k-1+N] = resamp(x=y[2*k-1+N], resamp_type=1,
                          shift=shift, spatial_dims=spatial_dims)

  return y


def sefilter2(x, f1, f2, spatial_dims, extmod='per', shift=(0, 0)):
  """2D separable filtering with extension handling."""

  # Make sures filters are row vectors.
  f1 = torch.flatten(f1)
  f2 = torch.flatten(f2)

  # Periodic extension.
  lf1 = (f1.shape[0] - 1) / 2
  lf2 = (f2.shape[0] - 1) / 2

  y = extend2(
      x=x,
      ru=floor(lf1) + shift[0],
      rd=ceil(lf1) - shift[0],
      cl=floor(lf2) + shift[1],
      cr=ceil(lf2) - shift[1],
      extmod=extmod,
      spatial_dims=spatial_dims,
  )

  # Separable filter.
  # TODO check if Conv2d uses valid by default.
  y = nn.functional.conv2d(y, f1.unsqueeze(0).unsqueeze(0).unsqueeze(0))
  y = nn.functional.conv2d(y, f2.unsqueeze(0).unsqueeze(0).unsqueeze(-1))
  return y


def resamp(x, resamp_type, spatial_dims, shift=1, extmod='per'):
  """Resampling in 2D filterbank."""
  assert extmod == 'per', 'Undefined extension mode.'
  assert resamp_type in VALID_RESAMP_TYPES, 'Invalid resampling type.'

  if resamp_type == 1:
    y = [torch.gather(x_, spatial_dims[0], (torch.arange(x.shape[spatial_dims[0]], device=x.device)[
        :, None] + shift * torch.arange(x.shape[spatial_dims[1]], device=x.device)) % x.shape[spatial_dims[0]]) for x_ in x.flatten(0, 1)]
  elif resamp_type == 2:
    y = [torch.gather(x_, spatial_dims[0], (torch.arange(x.shape[spatial_dims[0]], device=x.device)[
        :, None] - shift * torch.arange(x.shape[spatial_dims[1]], device=x.device)) % x.shape[spatial_dims[0]]) for x_ in x.flatten(0, 1)]
  elif resamp_type == 3:
    y = [torch.gather(x_, spatial_dims[1], (torch.arange(
        x.shape[spatial_dims[1]], device=x.device) + shift * torch.arange(x.shape[spatial_dims[0]], device=x.device)[:, None]) % x.shape[spatial_dims[1]]) for x_ in x.flatten(0, 1)]
  else:  # resamp_type==4:
    y = [torch.gather(x_, 1, (torch.arange(
        x.shape[spatial_dims[1]], device=x.device) - shift * torch.arange(x.shape[spatial_dims[0]], device=x.device)[:, None]) % x.shape[spatial_dims[1]]) for x_ in x.flatten(0, 1)]

  y = torch.stack(y, 0).unflatten(0, [x.shape[0], x.shape[1]])
  return y


def ccsym(x, k, ccsym_type, spatial_dims=(-2, -1)):
  """Exploit the complex conjugate symmetry in the fourier transform of real-valued signals."""
  assert ccsym_type in _VALID_CCSYM_TYPES, 'Invalid ccsym type.'

  N = len(spatial_dims)  # Spatial dimensions.
  szX = [x.shape[d] for d in spatial_dims]  # Spatial support.

  if ccsym_type == 'c':
    # Initialize the subscript array.
    y = torch.narrow(x, spatial_dims[k], 0, int(szX[k] / 2) + 1)
  else:
    # Subscript mapping for complex conjugate symmetric signal recovery.
    szX[k] = (szX[k] - 1) * 2
    sub_conj = []

    for m in range(N):
      sub_conj.append(
          torch.cat([torch.tensor([1]), torch.arange(szX[m], 1, -1)], 0))

    sub_conj[k] = torch.arange(szX[k] / 2, 1, -1, dtype=torch.int64)

    # Recover full signal.
    grid01, grid02 = torch.meshgrid(
        sub_conj[0] - 1, sub_conj[1] - 1, indexing='ij')
    if x.shape[0] == x.shape[1] == 1:
      y = torch.cat([x, torch.conj(x[:, :, grid01, grid02])], spatial_dims[k])
    else:
      y = torch.stack(
          [torch.cat([x_, torch.conj(x_[grid01, grid02])], k)
           for x_ in x.flatten(0, 1)],
          0).unflatten(0, [x.shape[0], x.shape[1]])
  return y


def ppdec(x, ppdec_type, spatial_dims):
  """Parallelogram Polyphase Decomposition"""
  assert spatial_dims == (-2, -
                          1), 'Spatial dimensions currently not supported.'
  assert ppdec_type in VALID_RESAMP_TYPES, 'Invalid polyphase decomposition type.'
  if ppdec_type == 1:  # P1 = R1 * Q1 = D1 * R3
    p0 = resamp(x=x[..., ::2, :], resamp_type=3, spatial_dims=spatial_dims)

    # R1 * [0; 1] = [1; 1]
    p1 = resamp(x=x[..., 1::2, list(range(1, x.shape[spatial_dims[1]])
                                    ) + [0]], resamp_type=3, spatial_dims=spatial_dims)

  elif ppdec_type == 2:  # P2 = R2 * Q2 = D1 * R4
    p0 = resamp(x=x[..., ::2, :], resamp_type=4, spatial_dims=spatial_dims)

    # R2 * [1; 0] = [1; 0]
    p1 = resamp(x=x[..., 1::2, :], resamp_type=4, spatial_dims=spatial_dims)

  elif ppdec_type == 3:  # P3 = R3 * Q2 = D2 * R1
    p0 = resamp(x=x[..., :, ::2], resamp_type=1, spatial_dims=spatial_dims)

    # R3 * [1; 0] = [1; 1]
    p1 = resamp(x=x[..., list(range(1, x.shape[spatial_dims[0]])) +
                [0], 1::2], resamp_type=1, spatial_dims=spatial_dims)

  else:  # ppdec_type==4:  # P4 = R4 * Q1 = D2 * R2
    p0 = resamp(x=x[..., :, ::2], resamp_type=2, spatial_dims=spatial_dims)

    # R4 * [0; 1] = [0; 1]
    p1 = resamp(x=x[..., :, 1::2], resamp_type=2, spatial_dims=spatial_dims)

  return p0, p1


def qpdec(x, spatial_dims, qpdec_type='1r'):
  """Quincunx Polyphase Decomposition."""
  assert spatial_dims == (-2, -1), \
      'Spatial dimensions currently not supported.'
  assert qpdec_type in VALID_QPDEC_TYPES, 'Invalid polyphase decomposition type.'
  if qpdec_type == '1r':  # Q1 = R2 * D1 * R3
    y = resamp(x=x, resamp_type=2, spatial_dims=spatial_dims)
    p0 = resamp(x=y[..., ::2, :], resamp_type=3, spatial_dims=spatial_dims)

    # inv(R2) * [0; 1] = [1; 1]
    p1 = resamp(x=y[..., 1::2, list(range(1, x.shape[spatial_dims[1]])
                                    ) + [0]], resamp_type=3, spatial_dims=spatial_dims)

  elif qpdec_type == '1c':  # Q1 = R3 * D2 * R2
    y = resamp(x=x, resamp_type=3, spatial_dims=spatial_dims)
    p0 = resamp(x=y[..., :, ::2], resamp_type=2, spatial_dims=spatial_dims)

    # inv(R3) * [0; 1] = [0; 1]
    p1 = resamp(x=y[..., :, 1::2], resamp_type=2, spatial_dims=spatial_dims)

  elif qpdec_type == '2r':  # Q2 = R1 * D1 * R4
    y = resamp(x=x, resamp_type=1, spatial_dims=spatial_dims)
    p0 = resamp(x=y[..., ::2, :], resamp_type=4, spatial_dims=spatial_dims)

    # inv(R1) * [1; 0] = [1; 0]
    p1 = resamp(x=y[..., 1::2, :], resamp_type=4, spatial_dims=spatial_dims)

  elif qpdec_type == '2c':  # Q2 = R4 * D2 * R1
    y = resamp(x=x, resamp_type=4, spatial_dims=spatial_dims)
    p0 = resamp(x=y[..., :, ::2], resamp_type=1, spatial_dims=spatial_dims)

    # inv(R4) * [1; 0] = [1; 1]
    p1 = resamp(x=y[..., list(range(1, x.shape[spatial_dims[0]])) +
                [0], 1::2], resamp_type=1, spatial_dims=spatial_dims)
  else:
    raise ValueError('Invalid argument type')
  return p0, p1


def fbdec_l(x, f, type1, type2, spatial_dims, extmod='per'):
  """Two-channel 2D Filterbank Decomposition using Ladder Structure"""
  # Modulate f. Avoid in-place operation.
  f_ = f.clone()
  f_[::2] = -f_[::2]

  assert min([x.shape[d] for d in spatial_dims]
             ) > 1, 'Input is a vector, unpredicted output!'

  # Polyphase decomposition of the input image.
  if type1.lower()[0] == 'q':
    # Quincunx polyphase decomposition.
    p0, p1 = qpdec(x=x, qpdec_type=type2, spatial_dims=spatial_dims)
  elif type1.lower()[0] == 'p':
    # Parallelogram polyphase decomposition.
    p0, p1 = ppdec(x=x, ppdec_type=type2, spatial_dims=spatial_dims)
  else:
    raise ValueError('Invalid argument type1.')

  # Ladder network structure.
  y0 = (1 / sqrt(2)) * (p0 - sefilter2(x=p1,
                                       f1=f_, f2=f_, extmod=extmod, shift=[1, 1], spatial_dims=(-2, -1)))
  y1 = (-sqrt(2)*p1) - sefilter2(x=y0, f1=f_,
                                 f2=f_, extmod=extmod, spatial_dims=(-2, -1))
  return y0, y1


def dfbdec_l(x, f, n, spatial_dims):
  """Directional Filterbank Decomposil.tion using Ladder Structure."""
  assert round(n) == n and n >= 0, \
      'Number of decomposition levels must be a non-negative integer.'

  if n == 0:
    # No decomposition, simply copy input to output.
    return [x]

  # Ladder filter.
  if isinstance(f, str):
    f = ldfilter(fname=f, dtype=x.dtype, device=x.device)

  # Tree-structured filter banks.
  if n == 1:
    # Simplest case, one level.
    y_1, y_2 = fbdec_l(x=x, f=f, type1='q', type2='1r',
                       extmod='qper_col', spatial_dims=spatial_dims)
    y = [y_1, y_2]
  else:
    # For the cases that n >= 2.
    # First level.
    x0, x1 = fbdec_l(x=x, f=f, type1='q', type2='1r',
                     extmod='qper_col', spatial_dims=spatial_dims)

    # Second level.
    y_2, y_1 = fbdec_l(x=x0, f=f, type1='q', type2='2c',
                       extmod='per', spatial_dims=spatial_dims)
    y_4, y_3 = fbdec_l(x=x1, f=f, type1='q', type2='2c',
                       extmod='per', spatial_dims=spatial_dims)
    y = [y_1, y_2, y_3, y_4]

    # Now expand the rest of the tree.
    for l in range(3, n+1):
      # Allocate space for the new subband outputs.
      y_old = copy.deepcopy(y)
      y = []

      # The first half channels use R1 and R2.
      for k in range(1, 2**(l-2) + 1):
        i = (k - 1) % 2 + 1
        y_a, y_b = fbdec_l(x=y_old[k-1], f=f, type1='p',
                           type2=i, extmod='per', spatial_dims=spatial_dims)
        y.append(y_b)
        y.append(y_a)

      # The second half channels use R3 and R4.
      for k in range(2**(l - 2) + 1, 2**(l - 1) + 1):
        i = (k-1) % 2 + 3
        y_a, y_b = fbdec_l(x=y_old[k-1], f=f, type1='p',
                           type2=i, extmod='per', spatial_dims=spatial_dims)
        y.append(y_b)
        y.append(y_a)

  # Backsampling.
  y = backsamp(y=y, spatial_dims=spatial_dims)

  # Flip the order of the second half channels.
  y[2**(n-1):] = (y[2**(n-1):])[::-1]
  return y


def PrySDdec_onestep(X, w, tbw, D, smooth_func, spatial_dims):
  """N-dimensional multiscale pyramid decomposition - One Step."""
  # The dimension of the problem.
  N = len(spatial_dims)
  # Spatial support.
  szX = torch.tensor([X.shape[d] for d in spatial_dims], device=X.device)

  # The original full size.
  # Spatial support.
  szF = torch.tensor([X.shape[d] for d in spatial_dims], device=X.device)
  szF[-1] = (szX[-1] - 1) * 2

  # Passband index arrays.
  pbd_array = []

  # Lowpass filter (nonzero part).
  szLf = 2 * torch.ceil(szF / 2 * (w + tbw)) + 1
  szLf[-1] = (szLf[-1] + 1) / 2
  szLf = szLf.int()

  Lf = torch.ones(szLf.int().tolist(), device=X.device, dtype=torch.complex64)

  szR = torch.ones(N, dtype=torch.int32, device=X.device)  # For resizing.

  for n in range(N):
    # Current Fourier domain resolution.
    nr = szF[n]
    szR[n] = szLf[n]

    # Passband.
    pbd = torch.arange(ceil((nr) / 2 * (w + tbw)) + 1, device=X.device)

    # Passband value.
    pbd_value_ = smooth_func(
        (pbd[-1] - pbd) / (torch.ceil(nr / 2 * (w + tbw)) - torch.floor(nr / 2 * (w - tbw))))
    assert all(pbd_value_ >= 0), 'Invalid values, must be nonnegative.'
    pbd_value = torch.sqrt(pbd_value_)

    # See if we need to consider the symmetric part.
    if n != N - 1:
      pbd = torch.cat(
          [pbd, torch.arange(nr - pbd[-1], nr, device=X.device)], 0)
      pbd_value = torch.cat([pbd_value, (pbd_value[1:]).flip(0)], 0)

    pbd_array.append(pbd + 1)
    pbd_value = torch.reshape(pbd_value, szR.tolist())
    Lf = Lf * pbd_value.repeat((szLf / szR).int().tolist())

    szR[n] = 1

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

  # Get full dimension of lowpass filter (batch dim, channels and spatial support)
  szLp_full = torch.tensor(list(X.shape), dtype=torch.int, device=X.device)
  szLp_full[list(spatial_dims)] = szLp[list(spatial_dims)]

  Lp = torch.tensor(0, dtype=X.dtype, device=X.device).repeat(
      szLp_full.tolist())

  # TODO: generalize this.
  grid_sml01, grid_sml02 = torch.meshgrid(
      pbd_array_sml[0] - 1, pbd_array_sml[1] - 1, indexing='ij')
  grid01, grid02 = torch.meshgrid(
      pbd_array[0] - 1, pbd_array[1] - 1, indexing='ij')

  if X.shape[0] == X.shape[1] == 1:
    # Single layer, single batch, apply filter directly.
    Lp[0, 0, grid_sml01, grid_sml02] = (
        Lf / D**(N / 2)) * X[0, 0, grid01, grid02]
  else:
    Lp_ = torch.stack([(Lf / D**(N / 2)) * x[grid01, grid02]
                       for x in X.flatten(0, 1)], 0).unflatten(0, [X.shape[0], X.shape[1]])
    Lp[:, :, grid_sml01, grid_sml02] = Lp_

  # Get the highpass subband.
  Lf_ = Lf**2
  assert torch.all(Lf_.real == Lf_), 'Error, complex power outputs.'

  Lf_ = torch.sqrt(1 - Lf_)
  assert torch.all(Lf_.real == Lf_), 'Error, complex square root outputs.'

  Hp = X.clone()
  if X.shape[0] == X.shape[1] == 1:
    # Single layer, single batch, apply filter directly.
    Hp[0, 0, grid01, grid02] = Lf_ * X[0, 0, grid01, grid02]
  else:
    Hp_ = torch.stack([Lf_ * x[grid01, grid02] for x in X.flatten(0, 1)],
                      0).unflatten(0, [X.shape[0], X.shape[1]])
    Hp[:, :, grid01, grid02] = Hp_

  return Lp, Hp


def PyrNDDEC_mm(X, OutD, L, Pyr_mode, smooth_func):
  """N-dimensional multiscale pyramid decomposition - with multiple modes."""
  OutD = OutD.upper()

  # Get shape, do not count batch dim. nor channels.
  N = len(X.shape) - 2

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
  elif Pyr_mode == 2:
    # the cutoff frequencies at each scale.
    w_array = 1 / 3 * torch.ones(L)

    # the transition bandwidths at each scale.
    tbw_array = 1 / 7 * torch.ones(L)

    # the downsampling factor at each scale.
    # no downsampling at the finest scale.
    D_array = 2 * torch.ones(L)
  else:
    raise ValueError('Unsupported Pyr mode.')

  # Compute the FFT in the spatial domain.
  X = torch.fft.fftn(X, dim=(-2, -1))

  # We assume a real-valued input signal X. Half of its Fourier
  # coefficients can be removed due to conjugate symmetry.
  X = ccsym(x=X, k=N-1, ccsym_type='c', spatial_dims=(-2, -1))
  subs = []

  for n in range(L - 1, -1, -1):
    # One level of the pyramid decomposition
    Lp, Hp = PrySDdec_onestep(
        X=X,
        w=w_array[n],
        tbw=tbw_array[n],
        D=D_array[n],
        smooth_func=smooth_func,
        spatial_dims=(-2, -1),
    )

    X = Lp.clone()
    Hp = ccsym(x=Hp, k=N-1, ccsym_type='e', spatial_dims=(-2, -1))
    if OutD == 'S':
      # Go back to the spatial domain
      subs.insert(0, torch.real(torch.fft.ifftn(Hp, dim=(-2, -1))))
    else:
      subs.insert(0, Hp)

  X = ccsym(x=X, k=N-1, ccsym_type='e')
  if OutD == 'S':
    subs.insert(0, torch.real(torch.fft.ifftn(X, dim=(-2, -1))))
  else:
    subs.insert(0, X)

  return subs
