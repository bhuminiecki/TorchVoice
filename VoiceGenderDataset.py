import os
import torch
#import torchaudio
from torch.utils.data import Dataset
from librosa import load, piptrack
import numpy as np
from entropy import spectral_entropy

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

        '''sample = torchaudio.load(self.entries[idx])
        sample = torchaudio.transforms.Resample(sample[1], 32000)(sample[0])

        if not sample.dtype.is_floating_point:
            sample = torch.tensor(sample.to(torch.float32)[0])

        if len(sample) == 2:
            sample = torch.mean(sample, 0, True)

        if self.transform is not None:
            sample = self.transform(sample)

        sample = sample[0][0:10000]

        out = [0.0 for i in range(10000)]

        for i in range(len(sample)):
            out[i] = sample[i]

        out = torch.tensor(out)

        out.resize_(1, 10000)'''

        target = self.labels[idx]

        target = torch.tensor(1 if (target == "k") else 0).to(torch.long)

        y, sr = load(self.entries[idx])
        pitches, magnitudes = piptrack(y=y, sr=sr, fmin=75, fmax=275)
        q25 = 0.0
        iqr = 0.0
        fun = 0.0

        if len(pitches[np.nonzero(pitches)]) > 0:
            q25 = np.percentile(pitches[np.nonzero(pitches)], 25)
            iqr = np.percentile(pitches[np.nonzero(pitches)], 75) - q25
            entropy = spectral_entropy(y, 36000)

        return torch.tensor([q25, iqr, entropy]), target

    def print_entries(self):
        print(self.entries)
        print(self.labels)


"""vgd = VoiceGenderDataset("data/train", transform=None)

for x in range(5):
    tensor = vgd.__getitem__(x)[0]
    print((tensor).size())
#    print(tensor)
    print("############")
"""
