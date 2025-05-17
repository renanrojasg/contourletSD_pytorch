"""2D extension functions."""
import torch

_VALID_EXTENSION_MODES = ['per', 'qper_row', 'qper_col']


def getPerIndices(lx, lb, le):
  """Get periodic indices."""
  I = list(range(lx - lb + 1, lx + 1)) + \
      list(range(1, lx + 1)) + list(range(1, le + 1))

  if lx < lb or lx < le:
    I = [i % lx for i in I]
  return I


def extend2(x, ru, rd, cl, cr, extmod, spatial_dims):
  """2D extension."""
  assert spatial_dims == (-2, -1), 'Unsupported spatial dimensions.'
  assert extmod in _VALID_EXTENSION_MODES, 'Invalid extension mode.'

  rx, cx = [x.shape[d] for d in spatial_dims]

  if extmod == 'per':
    I = getPerIndices(lx=rx, lb=ru, le=rd)
    y = x[..., torch.tensor(I).long()-1, :]

    I = getPerIndices(lx=cx, lb=cl, le=cr)
    y = y[..., torch.tensor(I).long()-1]

  elif extmod == 'qper_row':
    rx2 = round(rx / 2)
    y1 = torch.cat([x[..., range(rx2, rx), range(cx-cl, cx)],
                   x[..., :rx2, range(cx-cl, cx)]], 0)
    y2 = torch.cat([x[..., range(rx2, rx), :cr], x[..., :rx2, :cr]], 0)
    y = torch.cat([y1, x, y2], 1)

    I = getPerIndices(lx=rx, lb=ru, le=rd)
    y = y[..., torch.tensor(I).long()-1, :]

  else:   # extmod=='qper_col':
    cx2 = round(cx / 2)
    y1 = torch.cat([x[..., rx-ru: rx, cx2: cx],
                   x[..., rx-ru: rx, :cx2]], spatial_dims[1])
    y2 = torch.cat([x[..., :rd, cx2: cx], x[..., :rd, :cx2]], spatial_dims[1])
    y = torch.cat([y1, x, y2], spatial_dims[0])

    I = getPerIndices(cx, cl, cr)
    y = y[..., torch.tensor(I).long()-1]

  return y
