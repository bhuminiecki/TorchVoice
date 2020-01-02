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
#        print(sample[0])
        sample = torchaudio.transforms.Resample(sample[1], 32000)(sample[0])
        print(sample.size())
  #      print(sample.dim())
        target = self.labels[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        print(sample.size())

        if not sample.dtype.is_floating_point:
            sample = sample.to(torch.float32)

        #print(len(sample))
        if len(sample) == 2:
#            print(torch.mean(sample, 0, True))
            sample = torch.mean(sample, 0, True)

        #print(len(sample))

 #       print("#####")

        return sample, torch.tensor([int(target == "k"), int(target == "m")])

    def print_entries(self):
        print(self.entries)
        print(self.labels)


vgd = VoiceGenderDataset("data/train", transform=torchaudio.transforms.MelSpectrogram(f_min=0.0,
                                                                                      f_max=20000.0,
                                                                                      pad=1,
                                                                                      n_fft=2000,
                                                                                      sample_rate=32000))

for x in range(5):
    tensor = vgd.__getitem__(x)[0]
    print((tensor).size())
    print(tensor)
    print("############")

