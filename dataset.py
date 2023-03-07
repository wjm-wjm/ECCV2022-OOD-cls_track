import os
import random
import math
import cv2
import collections
import numpy as np
import torch
from PIL import Image
from corruption import gaussian_blur, glass_blur, defocus_blur, motion_blur, zoom_blur, fog, snow, spatter, contrast, saturate, jpeg_compression, pixelate, elastic_transform, gaussian_noise, shot_noise, impulse_noise, speckle_noise
from corruption_2dcc import jpeg_compression_2dcc, pixelate_2dcc, shot_noise_2dcc, impulse_noise_2dcc, defocus_blur_2dcc, fog_2dcc, zoom_blur_2dcc, cutmove_2dcc


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def is_valid_file(filename, extensions):
    return filename.lower().endswith(extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def add_haze1(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    (row, col, chs) = img.shape

    A = 200.0  # brightness
    beta = 0.1  # haze concentration
    size = math.sqrt(max(row, col))  # haze size
    center = (row // 2, col // 2)  # haze center
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img[j][l][:] = img[j][l][:] * td + A * (1 - td)

    return Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))


def augment(img):
    img_size = img.size
    img_aug_size = (224, 224)
    img = img.resize(img_aug_size)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    p = random.random()
    if p < (1./17):
        severity_id = random.randint(1, 5)
        img_aug = gaussian_noise(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (2./17):
        severity_id = random.randint(1, 5)
        img_aug = shot_noise(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (3./17):
        severity_id = random.randint(1, 5)
        img_aug = impulse_noise(img, severity=severity_id, seed=3407)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (4./17):
        severity_id = random.randint(1, 5)
        img_aug = speckle_noise(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (5./17):
        severity_id = random.randint(1, 5)
        img_aug = gaussian_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (6./17):
        severity_id = random.randint(1, 5)
        img_aug = glass_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (7./17):
        severity_id = random.randint(1, 5)
        img_aug = defocus_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (8./17):
        severity_id = random.randint(1, 5)
        img_aug = motion_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (9./17):
        severity_id = random.randint(1, 5)
        img_aug = zoom_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (10./17):
        severity_id = random.randint(1, 5)
        img_aug = fog(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (11./17):
        severity_id = random.randint(1, 5)
        img_aug = snow(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (12./17):
        severity_id = random.randint(1, 5)
        img_aug = spatter(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (13./17):
        severity_id = random.randint(1, 5)
        img_aug = contrast(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (14./17):
        severity_id = random.randint(1, 5)
        img_aug = saturate(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (15./17):
        severity_id = random.randint(1, 5)
        img_aug = jpeg_compression(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (16./17):
        severity_id = random.randint(1, 5)
        img_aug = pixelate(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    else:
        severity_id = random.randint(1, 5)
        img_aug = elastic_transform(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug


def augment_2dcc(img):
    p = random.random()

    if p < (1./8):
        # return img
        img_size = img.size
        img_aug_size = (224, 224)
        img = img.resize(img_aug_size)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        severity_id = random.randint(1, 5)
        img_aug = motion_blur(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (2./8):
        return jpeg_compression_2dcc(img)
    elif p < (3./8):
        return pixelate_2dcc(img)
    elif p < (4./8):
        # return shot_noise_2dcc(img)
        img_size = img.size
        img_aug_size = (224, 224)
        img = img.resize(img_aug_size)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        severity_id = random.randint(1, 5)
        img_aug = snow(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (5./8):
        # return impulse_noise_2dcc(img)
        img_size = img.size
        img_aug_size = (224, 224)
        img = img.resize(img_aug_size)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        severity_id = random.randint(1, 5)
        img_aug = elastic_transform(img, severity=severity_id)
        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
        img_aug = img_aug.resize(img_size)

        return img_aug
    elif p < (6./8):
        return defocus_blur_2dcc(img)
    elif p < (7./8):
        return fog_2dcc(img)
    # elif p < (8./9):
    #     img_size = img.size
    #     img_aug_size = (224, 224)
    #     img = img.resize(img_aug_size)
    #     img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    #     severity_id = random.randint(1, 5)
    #     img_aug = speckle_noise(img, severity=severity_id)
    #     img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
    #     img_aug = img_aug.resize(img_size)
    #
    #     return img_aug
    # elif p < (9./10):
    #     return impulse_noise_2dcc(img)
    else:
        return zoom_blur_2dcc(img)


def print_dataset(args_config, class2num_list):
    print("Dataset {} Statistics:".format(args_config.dataset))
    print("  ------------------------------------------------------------------------------------------------------------")
    print("  |     Subset     | aeroplane | bicycle | boat | bus | car | chair | diningtable | motorbike | sofa | train |")
    print("  |      Train     | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[0]["aeroplane"], class2num_list[0]["bicycle"], class2num_list[0]["boat"], class2num_list[0]["bus"], class2num_list[0]["car"], class2num_list[0]["chair"], class2num_list[0]["diningtable"], class2num_list[0]["motorbike"], class2num_list[0]["sofa"], class2num_list[0]["train"]))
    print("  |    Test_iid    | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[1]["aeroplane"], class2num_list[1]["bicycle"], class2num_list[1]["boat"], class2num_list[1]["bus"], class2num_list[1]["car"], class2num_list[1]["chair"], class2num_list[1]["diningtable"], class2num_list[1]["motorbike"], class2num_list[1]["sofa"], class2num_list[1]["train"]))
    print("  |   Test_shape   | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[2]["aeroplane"], class2num_list[2]["bicycle"], class2num_list[2]["boat"], class2num_list[2]["bus"], class2num_list[2]["car"], class2num_list[2]["chair"], class2num_list[2]["diningtable"], class2num_list[2]["motorbike"], class2num_list[2]["sofa"], class2num_list[2]["train"]))
    print("  |    Test_pose   | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[3]["aeroplane"], class2num_list[3]["bicycle"], class2num_list[3]["boat"], class2num_list[3]["bus"], class2num_list[3]["car"], class2num_list[3]["chair"], class2num_list[3]["diningtable"], class2num_list[3]["motorbike"], class2num_list[3]["sofa"], class2num_list[3]["train"]))
    print("  |  Test_texture  | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[4]["aeroplane"], class2num_list[4]["bicycle"], class2num_list[4]["boat"], class2num_list[4]["bus"], class2num_list[4]["car"], class2num_list[4]["chair"], class2num_list[4]["diningtable"], class2num_list[4]["motorbike"], class2num_list[4]["sofa"], class2num_list[4]["train"]))
    print("  |  Test_context  | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[5]["aeroplane"], class2num_list[5]["bicycle"], class2num_list[5]["boat"], class2num_list[5]["bus"], class2num_list[5]["car"], class2num_list[5]["chair"], class2num_list[5]["diningtable"], class2num_list[5]["motorbike"], class2num_list[5]["sofa"], class2num_list[5]["train"]))
    print("  | Test_occlusion | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[6]["aeroplane"], class2num_list[6]["bicycle"], class2num_list[6]["boat"], class2num_list[6]["bus"], class2num_list[6]["car"], class2num_list[6]["chair"], class2num_list[6]["diningtable"], class2num_list[6]["motorbike"], class2num_list[6]["sofa"], class2num_list[6]["train"]))
    print("  |  Test_weather  | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(class2num_list[7]["aeroplane"], class2num_list[7]["bicycle"], class2num_list[7]["boat"], class2num_list[7]["bus"], class2num_list[7]["car"], class2num_list[7]["chair"], class2num_list[7]["diningtable"], class2num_list[7]["motorbike"], class2num_list[7]["sofa"], class2num_list[7]["train"]))
    print("  ------------------------------------------------------------------------------------------------------------")


def make_dataset(class_to_idx, directory):
    instances = []
    class2num = collections.defaultdict(int)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname, IMG_EXTENSIONS):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)
                    class2num[target_class] += 1

    return instances, class2num


def make_dataset_final(directory):
    instances = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_file(fname, IMG_EXTENSIONS):
                path = os.path.join(root, fname)
                item = path
                instances.append(item)

    return instances


class ROBINDataset_train(torch.utils.data.Dataset):
    def __init__(self, directory, transform):
        self.classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples, self.class2num = make_dataset(self.class_to_idx, directory)
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        # sample = augment(sample)
        sample = augment_2dcc(sample)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        return len(self.samples)


class ROBINDataset_test(torch.utils.data.Dataset):
    def __init__(self, directory, transform):
        self.classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples, self.class2num = make_dataset(self.class_to_idx, directory)
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, path

    def __len__(self):
        return len(self.samples)


class ROBINDataset_test_final(torch.utils.data.Dataset):
    def __init__(self, directory, transform):
        self.samples = make_dataset_final(directory)
        self.transform = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path

    def __len__(self):
        return len(self.samples)



