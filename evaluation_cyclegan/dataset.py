import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random
import pdb


class dataset_single(data.Dataset):
  def __init__(self, opts, setname):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index])
    return data

  def load_img(self, img_name):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    return img

  def __len__(self):
    return self.size



class dataset_single_maps(data.Dataset):
  def __init__(self, opts, setname):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)

    # setup image transformation
    transforms1 = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms1.append(CenterCrop(opts.crop_size))
    transforms1.append(ToTensor())
    self.transforms1 = Compose(transforms1)
    
    transforms2 = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms2.append(CenterCrop(opts.crop_size))
    transforms2.append(ToTensor())
    transforms2.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms2 = Compose(transforms2)

    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index])
    return data

  def load_img(self, img_name):
    img = Image.open(img_name).convert('RGB')
    img_left = img.crop((0, 0, 600, 600)) 
    img_right = img.crop((600, 0, 1200, 600))
    im_left = self.transforms1(img_left)
    im_right = self.transforms1(img_right)
    im_l = self.transforms2(img_left)
    im_r = self.transforms2(img_right)
    return {'left':im_left, 'right':im_right, 'img_left': im_l,'img_right': im_r}

  def __len__(self):
    return self.size



class dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    if opts.phase == 'train':
      transforms.append(RandomCrop(opts.crop_size))
    else:
      transforms.append(CenterCrop(opts.crop_size))
    if not opts.no_flip:
      transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[random.randint(0, self.A_size - 1)], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    return data_A, data_B

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.dataset_size
