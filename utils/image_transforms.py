"""Image preprocessing transforms"""
from torchvision import transforms


def resize_transform(size):
  return transforms.Compose([
      transforms.Resize((size, size)),
      transforms.ToTensor(),
      lambda x: x * 255,  # Scale to [0, 255]
  ])
