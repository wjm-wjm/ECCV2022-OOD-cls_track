import random
import numpy as np
import torch
import pdb

import torchvision.transforms as T
from PIL import Image
import random
import torch.nn.functional as F
import torch
from torch.nn.parallel import parallel_apply
from io import BytesIO
from PIL import Image as PILImage

import skimage as sk
from skimage.filters import gaussian
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates


def jpeg_compression_2dcc(x):
    c = random.randint(1, 25)

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate_2dcc(x):
    c = random.uniform(0.1, 0.6)
    img_size = x.size

    x = x.resize((int(img_size[0] * c), int(img_size[1] * c)), PILImage.BOX)
    x = x.resize((img_size[0], img_size[1]), PILImage.BOX)

    return x


def shot_noise_2dcc(x):
    c = random.randint(1, 60)

    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

    x = Image.fromarray(np.uint8(x))

    return x


def impulse_noise_2dcc(x):
    c = random.uniform(0.02, 0.27)

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    x = np.clip(x, 0, 1) * 255
    x = Image.fromarray(np.uint8(x))

    return x


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur_2dcc(x):
    c1 = random.randint(0,10)
    c2 = random.uniform(0,0.5)
    c = (c1, c2)

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
    x = np.clip(channels, 0, 1) * 255

    x = Image.fromarray(np.uint8(x))

    return x


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()

    return maparray / maparray.max()


def fog_2dcc(x):
    c1 = random.uniform(1.5, 3.5)
    c2 = random.uniform(2, 1.1)
    c = (c1, c2)

    img_size = x.size
    x = x.resize((224, 224))

    x = np.array(x) / 255.
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
    x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
    x = Image.fromarray(np.uint8(x))

    x = x.resize(img_size)

    return x


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def zoom_blur_2dcc(x):
    c1 = 1
    c2 = random.uniform(1.08, 1.32)
    c3 = random.uniform(0.08, 0.32)
    c = np.arange(c1, c2, c3)

    img_size = x.size
    x = x.resize((224, 224))

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    x = np.clip(x, 0, 1) * 255
    x = Image.fromarray(np.uint8(x))

    x = x.resize(img_size)

    return x


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmove_2dcc(x):
    lam = np.random.beta(1.0, 1.0)
    x = np.array(x)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.shape, lam)
    bbx3, bby3, bbx4, bby4 = rand_bbox(x.shape, lam)
    if (bbx2-bbx1) > (bbx4-bbx3):
        bbx2 = bbx1 + (bbx4-bbx3)
    else:
        bbx4 = bbx3 + (bbx2-bbx1)
    if (bby2-bby1) > (bby4-bby3):
        bby2 = bby1 + (bby4-bby3)
    else:
        bby4 = bby3 + (bby2-bby1)
    # print("{} {} {} {}".format(bbx1, bby1, bbx2, bby2))
    # print("{} {} {} {}".format(bbx3, bby3, bbx4, bby4))
    # print(x.shape)
    # print(x[bbx1:bbx2, bby1:bby2, :].shape)
    # print(x[bbx3:bbx4, bby3:bby4, :].shape)
    x[bbx1:bbx2, bby1:bby2, :] = x[bbx3:bbx4, bby3:bby4, :]
    x = Image.fromarray(np.uint8(x))

    return x

