"""File export utilities"""
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.image_transforms import resize_transform

_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']


def export_images(images, out_name, out_path, out_ext='jpg', fnames=None):
  """Save output images"""
  if fnames:
    assert len(images) == len(
        fnames), 'The same number of images and filenames are required.'
    iterator = zip(images, fnames)
  else:
    iterator = images

  for b, data in enumerate(iterator):
    image = data[0] if fnames else data
    image_pil = transforms.ToPILImage()(image.detach().cpu())

    fname = f'{data[1]}_{out_name}.{out_ext}' if fnames else f"{b:04d}_{out_name}.{out_ext}"
    image_pil.save(os.path.join(out_path, fname))
  return


def is_image_file(fname):
  return any(fname.endswith(ext) for ext in _IMAGE_EXTENSIONS)


class DatasetWithFnames(torch.utils.data.Dataset):
  def __init__(self, dataset_path, transform_size):
    super(DatasetWithFnames, self).__init__()
    self.dataset_path = dataset_path
    self.image_list = [x for x in os.listdir(
        self.dataset_path) if is_image_file(x)]
    self.transform = resize_transform(size=transform_size)

  def __getitem__(self, index):
    image_path = os.path.join(self.dataset_path, self.image_list[index])
    image = self.transform(Image.open(image_path))

    fname = self.image_list[index].rsplit('/', 2)[-1].rsplit('.', 2)[0]
    return image, fname

  def __len__(self):
    return len(self.image_list)


def get_dataloader(dataset_path, transform_size, batch_size, num_workers):
  dataset = DatasetWithFnames(
      dataset_path=dataset_path,
      transform_size=transform_size,
  )
  dataloader = DataLoader(
      dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
  )
  return dataloader
