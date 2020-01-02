import os
import torch
import torchaudio
from torch.utils.data import Dataset


class VoiceGenderDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
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

        sample = torchaudio.transforms.Resample(sample[1], 32000)(sample[0])

        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        if not sample.dtype.is_floating_point:
            sample = torch.tensor(sample.to(torch.float32)[0])

        if len(sample) == 2:
            sample = torch.mean(sample, 0, True)

        return sample[0][0][40:350], torch.tensor([int(target == "k"), int(target == "m")])

    def print_entries(self):
        print(self.entries)
        print(self.labels)


"""vgd = VoiceGenderDataset("data/train", transform=torchaudio.transforms.Spectrogram())

for x in range(5):
    tensor = vgd.__getitem__(x)[0]
    print((tensor).size())
    print(tensor)
    print("############")
"""
