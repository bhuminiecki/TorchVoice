import os
import torch
import torchaudio
from torch.utils.data import Dataset


class VoiceGenderDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir=root_dir
        self.transform=transform
        self.entries = []
        self.labels = []
        for cat in os.walk(root_dir):
            for file in cat[2]:
                self.entries.append(os.path.join(cat[0], file))
                self.labels.append(cat[0][-1])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = torchaudio.load(self.entries[idx])
        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def print_entries(self):
        print(self.entries)
        print(self.labels)


vgd = VoiceGenderDataset("data/train")

vgd.print_entries()

