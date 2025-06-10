import os
from data_utils.base_dataset import BaseDataset

class OPV2VImageDataset(BaseDataset):
    def __init__(self, data_root, agent_id, transform=None):
        super().__init__(data_root)
        self.agent_path = os.path.join(data_root, agent_id)
        self.image_pairs = self._find_image_pairs()
        self.transform = transform

    def _find_image_pairs(self):
        files = sorted(os.listdir(self.agent_path))
        image_pairs = []
        for f in files:
            if "_camera0.png" in f:
                base = f.replace("_camera0.png", "")
                if f"{base}_camera2.png" in files:
                    image_pairs.append((f, f"{base}_camera2.png"))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        cam0, cam2 = self.image_pairs[idx]
        img0 = os.path.join(self.agent_path, cam0)
        img2 = os.path.join(self.agent_path, cam2)
        return {
            "front_image": img0,
            "rear_image": img2,
        }