import os
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

class WaterbirdsDataset(Dataset):
    """Dataset for the Waterbirds dataset."""

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.download()
        self.load_data()

    def download(self):
        waterbirds_dir = osp.join(self.root, "waterbirds")
        if not osp.isdir(waterbirds_dir):
            url = (
                "http://worksheets.codalab.org/rest/bundles/"
                "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
            )
            download_and_extract_archive(
                url,
                waterbirds_dir,
                filename="waterbirds.tar.gz",
            )

    def load_data(self):
        waterbirds_dir = osp.join(self.root, "waterbirds")
        metadata_df = pd.read_csv(osp.join(waterbirds_dir, "metadata.csv"))
        self.data = np.asarray(metadata_df["img_filename"].values)
        self.data = np.asarray([osp.join(waterbirds_dir, d) for d in self.data])
        self.targets = np.asarray(metadata_df["y"].values)
        background = np.asarray(metadata_df["place"].values)
        landbirds = np.argwhere(self.targets == 0).flatten()
        waterbirds = np.argwhere(self.targets == 1).flatten()
        land = np.argwhere(background == 0).flatten()
        water = np.argwhere(background == 1).flatten()
        self.groups = [
            np.intersect1d(landbirds, land),
            np.intersect1d(landbirds, water),
            np.intersect1d(waterbirds, land),
            np.intersect1d(waterbirds, water),
        ]
        
        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        if self.split == 'train':
            self.indices = self.train_indices
        elif self.split == 'val':
            self.indices = self.val_indices
        else:
            self.indices = self.test_indices

        # Adds group indices into targets for metrics.
        targets = []
        for j, t in enumerate(self.targets):
            g = [k for k, group in enumerate(self.groups) if j in group][0]
            targets.append([t, g])
        self.targets = np.asarray(targets)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target)

class WaterbirdsDataModule:
    """DataModule for the Waterbirds dataset."""

    def __init__(self, root, batch_size=32, num_workers=4):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def augmented_transforms(self):
        return Compose([
            RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def default_transforms(self):
        return Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def setup(self):
        self.train_dataset = WaterbirdsDataset(self.root, split='train', transform=self.augmented_transforms())
        self.val_dataset = WaterbirdsDataset(self.root, split='val', transform=self.default_transforms())
        self.test_dataset = WaterbirdsDataset(self.root, split='test', transform=self.default_transforms())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)