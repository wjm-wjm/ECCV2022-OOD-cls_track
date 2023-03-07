import os
import math
import cv2
import copy
from PIL import Image
import numpy as np
from utils import mkdir_if_missing


def read_img(img_path):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

        return img


def add_haze1(img):
    img_f = copy.deepcopy(img)
    (row, col, chs) = img.shape

    A = 200.0  # brightness [150, 250]
    beta = 0.1  # haze concentration [0.05, 0.2]
    size = math.sqrt(max(row, col))  # haze size
    center = (row // 2, col // 2)  # haze center
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    return img_f


def add_haze2(img):
    A = np.random.uniform(0.6, 0.95)
    t = np.random.uniform(0.3, 0.95)
    img_h = img * t + A * (1 - t)

    return img_h


def main_addhaze():
    file_path = "test_haze_image"
    mkdir_if_missing(file_path)

    # img_path = "/home/wjm/wjm/dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose/train/Images/aeroplane/imagenet_aeroplane_n02690373_1002_0.jpg"
    img_path = "/home/wjm/wjm/dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose/train/Images/aeroplane/imagenet_aeroplane_n02690373_1161_0.jpg"
    img = read_img(img_path)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img_haze1 = add_haze1(img)
    img_haze2 = add_haze2(img)
    img_haze1 = Image.fromarray(cv2.cvtColor(np.uint8(img_haze1), cv2.COLOR_BGR2RGB))
    img_haze2 = Image.fromarray(cv2.cvtColor(np.uint8(img_haze2), cv2.COLOR_BGR2RGB))
    img = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))
    img.save(os.path.join(file_path, img_path.split("/")[-1].split(".")[0] + ".jpg"))
    img_haze1.save(os.path.join(file_path, img_path.split("/")[-1].split(".")[0] + "_haze1.jpg"))
    img_haze2.save(os.path.join(file_path, img_path.split("/")[-1].split(".")[0] + "_haze2.jpg"))


def grabcut(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img = img * mask2[:, :, np.newaxis]

    img = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))

    return img


def grabcut_test():
    file_path = "test_grabcut_image"
    mkdir_if_missing(file_path)

    # img_path = "/home/wjm/wjm/dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose/train/Images/aeroplane/imagenet_aeroplane_n02690373_1002_0.jpg"
    # img_path = "/home/wjm/wjm/dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose/train/Images/aeroplane/imagenet_aeroplane_n02690373_1161_0.jpg"
    img_path = "/home/wjm/wjm/dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose/nuisances/occlusion/Images/aeroplane/imagenet_aeroplane_n02690373_10111_0.jpg"
    img = read_img(img_path)
    img.save(os.path.join(file_path, img_path.split("/")[-1].split(".")[0] + ".jpg"))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (400, 1, img.shape[1], img.shape[0])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img = img * mask2[:, :, np.newaxis]

    img = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))

    img.save(os.path.join(file_path, img_path.split("/")[-1].split(".")[0] + "_grabcut.jpg"))


if __name__ == "__main__":
    grabcut_test()