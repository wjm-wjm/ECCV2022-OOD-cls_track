import os
import random
import copy
import math
import time
import cv2
import collections
import numpy as np
import torch
from PIL import Image
from utils import mkdir_if_missing


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def is_valid_file(filename, extensions):
    return filename.lower().endswith(extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def add_haze1(img, num, param):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    (row, col, chs) = img.shape

    A = param[2]  # brightness [150, 250]
    beta = param[3]  # haze concentration [0.05, 0.2]
    size = math.sqrt(max(row, col))  # haze size
    center = (row / num * param[0], col / num * param[1])  # haze center
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img[j][l][:] = img[j][l][:] * td + A * (1 - td)

    return Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    tt = time.time()

    dataset_path = "../../../../dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose"
    file_name = "train/Images"
    file_name_data_augmentation = "train_addhaze/Images"
    file_path = os.path.join(dataset_path, file_name)
    file_data_augmentation_path = os.path.join(dataset_path, file_name_data_augmentation)

    mkdir_if_missing(file_data_augmentation_path)

    classes = sorted(entry.name for entry in os.scandir(file_path) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    num = 20
    id = 0
    bright = np.arange(150, 250, 1)
    dense = np.arange(0.05, 0.2, 0.01)
    for i in range(1, num + 1):
        for j in range(1, num + 1):
            if (i == 1) and (j == 1):
                param = np.array([[i, j, bright[id % bright.shape[0]], dense[id % dense.shape[0]]]])
            else:
                param = np.concatenate((param, np.array([[i, j, bright[id % bright.shape[0]], dense[id % dense.shape[0]]]])), axis=0)
            id += 1


    num_img = 0
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(file_path, target_class)
        mkdir_if_missing(os.path.join(file_data_augmentation_path, target_class))
        if not os.path.isdir(target_dir):
            continue
        for _, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname, IMG_EXTENSIONS):
                    path = os.path.join(file_path, target_class, fname)
                    path_data_augmentation_origin = os.path.join(file_data_augmentation_path, target_class, fname)
                    path_data_augmentation_addhaze1 = os.path.join(file_data_augmentation_path, target_class, fname.split(".")[0] + "_addhaze1.jpg")
                    img = pil_loader(path)
                    img_data_augumentation_addhaze1 = add_haze1(copy.deepcopy(img), num, param[num_img % param.shape[0]])
                    img.save(path_data_augmentation_origin)
                    img_data_augumentation_addhaze1.save(path_data_augmentation_addhaze1)
                    num_img += 1

    print("Consuming Time: {:.4f}".format(time.time() - tt))
    print("Number Of Images For Origin Dataset: {}".format(num_img))






