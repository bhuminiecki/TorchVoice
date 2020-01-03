from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from VoiceGenderDataset import VoiceGenderDataset
from torchaudio.transforms import Spectrogram

import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=21421, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

torch.manual_seed(opt.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = VoiceGenderDataset("data/train", transform=Spectrogram())
testing_set = VoiceGenderDataset("data/test", transform=Spectrogram())
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=testing_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

model = Net().to(device)
criterion = nn.SmoothL1Loss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input = batch[0].to(device)
        target = batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        #loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    avg_loss = epoch_loss / len(training_data_loader)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
    return avg_loss


def test():
    avg_acc = 0
    avg_loss = 0
    total = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0], batch[1]
            prediction = model(input.to(device))

            res = criterion(prediction, target.to(device)).item()
            avg_acc += (prediction.argmax().item() == target.argmax().item())
            avg_loss += res

    avg_loss = avg_loss / len(testing_data_loader)
    avg_acc = avg_acc / len(testing_data_loader)
    print("===> Avg. Accuracy: {:.4f}, Testing Loss: {:.4f}".format(avg_acc, avg_loss))
    return avg_loss, avg_acc


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


train_history = []
test_history = []
acc_history = []
for epoch in range(1, opt.nEpochs + 1):
    train_history.append(train(epoch))
    res = test()
    test_history.append(res[0])
    acc_history.append((res[1]))
    checkpoint(epoch)

plt.subplot(1, 2, 1)
plt.plot(range(1, opt.nEpochs + 1), train_history, label='Training Loss')
plt.plot(range(1, opt.nEpochs + 1), test_history, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, opt.nEpochs + 1), acc_history, label='Accuracy')
plt.legend(loc='upper right')
plt.title('Validation Accuracy')

plt.show()