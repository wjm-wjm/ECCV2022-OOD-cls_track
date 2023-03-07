import os
import time
import copy
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import pandas as pd
from utils import Summary, AverageMeter, ProgressMeter, mkdir_if_missing

from torchvision.utils import save_image
import torchvision.transforms as transforms

import ttach as tta
import kornia as K


CSV_EXTENSIONS = (".csv")

class_name = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]
class_name_dict = {"aeroplane":0, "bicycle":1, "boat":2, "bus":3, "car":4, "chair":5, "diningtable":6, "motorbike":7, "sofa":8, "train":9}


def is_valid_file(filename, extensions):
    return filename.lower().endswith(extensions)


def simple_ensemble():
    file_name = "logs/ensemble/res_simple_ensemble/"
    mkdir_if_missing(file_name)

    df = pd.read_csv(os.path.join("../cls_submission/results.csv"))
    imgs_list = collections.defaultdict(list)

    for id in range(len(df["imgs"])):
        for i in range(len(class_name)):
            imgs_list[df["imgs"][id]].append(0)

    for root, _, fnames in sorted(os.walk("logs/ensemble/", followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_file(fname, CSV_EXTENSIONS):
                path = os.path.join(root, fname)
                df_ = pd.read_csv(path)
                for id in range(len(df["imgs"])):
                    print(id)
                    print(df_["pred"][id])
                    imgs_list[df["imgs"][id]][class_name_dict[df_["pred"][id]]] += 1

    for id in range(len(df["imgs"])):
        if np.max(np.array(imgs_list[df["imgs"][id]])) == 1:
            continue
            # df["pred"][id] = df_["pred"][id]
        else:
            df["pred"][id] = class_name[np.argmax(np.array(imgs_list[df["imgs"][id]]))]

    df.to_csv(os.path.join(file_name, "results.csv"), index=None)


if __name__ == "__main__":
    simple_ensemble()