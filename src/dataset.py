"""SIRST dataset loader.

Images come in mixed modes (L / RGB / RGBA) and varying sizes; all are
converted to single-channel grayscale and resized to a fixed `base_size`.
Masks are binary (0/255) -> {0,1}.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SIRSTDataset(Dataset):
    def __init__(self, root, split="train", base_size=480, augment=False):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.base_size = base_size
        self.augment = augment and split == "train"

        split_file = os.path.join(root, "idx_427", f"{split}.txt")
        with open(split_file) as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

        # ImageNet-style single-channel normalization (grayscale).
        self.mean = 0.485
        self.std = 0.229

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, name + ".png")).convert("L")
        mask = Image.open(
            os.path.join(self.mask_dir, name + "_pixels0.png")).convert("L")
        return img, mask, name

    def __getitem__(self, idx):
        img, mask, name = self._load(idx)
        s = self.base_size
        img = img.resize((s, s), Image.BILINEAR)
        mask = mask.resize((s, s), Image.NEAREST)

        img = np.asarray(img, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:           # horizontal flip
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:           # vertical flip
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()

        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).unsqueeze(0)   # 1 x H x W
        mask = torch.from_numpy(mask).unsqueeze(0)  # 1 x H x W
        return img, mask, name
