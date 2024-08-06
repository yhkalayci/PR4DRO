import os
import numpy as np
import pickle
from transformers import BertTokenizer
import wilds
import torch
from torch.utils.data import Dataset, DataLoader

def to_np(tensor):
    return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

class CivilCommentsDataset(Dataset):
    """Dataset for the CivilComments dataset."""

    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.load_data()

    def load_data(self):
        dataset = wilds.get_dataset(
            dataset="civilcomments",
            download=True,
            root_dir=self.root,
        )
        spurious_names = ["male", "female", "LGBTQ", "black", "white",
                          "christian", "muslim", "other_religions"]
        column_names = dataset.metadata_fields
        spurious_cols = [column_names.index(name) for name in spurious_names]
        spurious = to_np(dataset._metadata_array[:, spurious_cols].sum(-1).clip(max=1))
        prefix = os.path.join(self.root, "civilcomments_v1.0")
        data_file = os.path.join(prefix, "civilcomments_token_data.pt")
        targets_file = os.path.join(prefix, "civilcomments_token_targets.pt")
        if not os.path.isfile(data_file):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            def tokenize(text):
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=220,
                    return_tensors="pt",
                )
                return torch.squeeze(torch.stack((
                    tokens["input_ids"], tokens["attention_mask"], 
                    tokens["token_type_ids"]), dim=2), dim=0)
            data = []
            targets = []
            ln = len(dataset)
            for j, d in enumerate(dataset):
                print(f"Caching {j}/{ln}")
                data.append(tokenize(d[0]))
                targets.append(d[1])
            data = torch.stack(data)
            targets = torch.stack(targets)
            torch.save(data, data_file)
            torch.save(targets, targets_file)
        self.data = torch.load(data_file).numpy()
        self.targets = torch.load(targets_file).numpy()
        self.groups = [
            np.intersect1d((~self.targets+2).nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d((~self.targets+2).nonzero()[0], spurious.nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], spurious.nonzero()[0]),
        ]
        
        split = dataset._split_array
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
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx])

class CivilCommentsDataModule:
    """DataModule for the CivilComments dataset."""

    def __init__(self, root, batch_size=32, num_workers=4):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        self.train_dataset = CivilCommentsDataset(self.root, split='train')
        self.val_dataset = CivilCommentsDataset(self.root, split='val')
        self.test_dataset = CivilCommentsDataset(self.root, split='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)