# datasets/base_dataset.py

import os
import yaml
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class BaseDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

    def load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def load_image(self, path):
        return Image.open(path).convert('RGB')

    def load_pcd(self, path):
        # 如果是 .npy 格式
        if path.endswith(".npy"):
            return np.load(path)
        # TODO: 如果是 .pcd/.bin，你可以用 Open3D 或自定义解析器加载
        raise NotImplementedError("Unsupported point cloud format.")

    def apply_transform(self, img, transform):
        return transform(img) if transform else img

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
