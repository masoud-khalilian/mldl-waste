import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import ColorJitter as jitter
from torchvision.transforms import RandomResizedCrop as RRC
from torchvision.transforms import RandomRotation as RR
from config import cfg


import torch
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size   
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw  = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class ColorJitter(object):
    def __call__(self,img, mask):
        transform = jitter(0.4, 0.6, 0.3, 0.2)
        return transform(img),mask
    
class RandomResizedCrop(object):
    def __call__(self,img,mask):
        transform = RRC(cfg.TRAIN.IMG_SIZE, ratio = (0.5, 1))
        return transform(img),transform(mask)
    
class Rotate_none(object):
    def __call__(self,img,mask):
        return img,mask 
    
class Rotate_90(object):
    def __call__(self,img, mask):
        return img.transpose(Image.TRANSPOSE).transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.TRANSPOSE).transpose(Image.FLIP_TOP_BOTTOM)

class Rotate_180(object):
    def __call__(self,img, mask):
        return img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)

class Rotate_270(object):
    def __call__(self,img, mask):
        return img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.TRANSPOSE),mask.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.TRANSPOSE)


class FreeScale(object):
    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size  # (h, w)
        self.interpolation = interpolation

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), self.interpolation), mask.resize(self.size, self.interpolation)
    
class RandomRotate(object):
        def __call__(self,img,mask):
            transform = RR(360)
            return transform(img), transform(mask)

class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)           
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)



# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class ChangeLabel(object):
    def __init__(self, ori_label, new_label):
        self.ori_label = ori_label
        self.new_label = new_label

    def __call__(self, mask):
        mask[mask == self.ori_label] = self.new_label
        return mask

