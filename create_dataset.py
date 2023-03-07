import os
import time
import numpy as np
import cv2
from PIL import Image
from utils import mkdir_if_missing
from corruption import glass_blur, defocus_blur, motion_blur, zoom_blur, fog, snow, spatter, contrast, saturate, jpeg_compression, pixelate, elastic_transform


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
corruptions = [("glass_blur", glass_blur),
               ("defocus_blur", defocus_blur),
               ("motion_blur", motion_blur),
               ("zoom_blur", zoom_blur),
               ("fog", fog),
               ("snow", snow),
               ("spatter", spatter),
               ("contrast", contrast),
               ("saturate", saturate),
               ("jpeg", jpeg_compression),
               ("pixelate", pixelate),
               ("elastic", elastic_transform)]


def is_valid_file(filename, extensions):
    return filename.lower().endswith(extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


if __name__ == "__main__":
    tt = time.time()

    dataset_path = "../../../../dataset/ROBIN/processed_dataset/ROBINv1.1-cls-pose"
    file_name = "train/Images"
    file_name_data_augmentation = "train_corruption/Images"
    file_path = os.path.join(dataset_path, file_name)
    file_data_augmentation_path = os.path.join(dataset_path, file_name_data_augmentation)

    mkdir_if_missing(file_data_augmentation_path)

    classes = sorted(entry.name for entry in os.scandir(file_path) if entry.is_dir())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

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
                    img = pil_loader(path)
                    img_size = img.size
                    img_aug_size = (224, 224)
                    img = img.resize(img_aug_size)
                    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    for i, corruption in enumerate(corruptions):
                        severity_id = 1
                        img_aug = corruption[1](img, severity=severity_id)
                        img_aug = Image.fromarray(cv2.cvtColor(np.uint8(img_aug), cv2.COLOR_BGR2RGB))
                        img_aug = img_aug.resize(img_size)
                        path_corruption = os.path.join(file_data_augmentation_path, target_class, fname.split(".")[0] + "_" + corruption[0] + "_severity" + str(severity_id) + ".jpg")
                        img_aug.save(path_corruption)
                    img = Image.fromarray(cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB))
                    img = img.resize(img_size)
                    path_origin = os.path.join(file_data_augmentation_path, target_class, fname)
                    img.save(path_origin)
                    num_img += 1

    print("Consuming Time: {:.4f}".format(time.time() - tt))
    print("Number Of Images For Origin Dataset: {}".format(num_img))






