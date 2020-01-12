from __future__ import print_function

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Net
from VoiceGenderDataset import VoiceGenderDataset
#from torchaudio.transforms import Spectrogram

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

model = torch.load("model/model_epoch_18.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testing_set = VoiceGenderDataset("data/input")
testing_data_loader = DataLoader(dataset=testing_set, num_workers=4, batch_size=1, shuffle=False)

model = model.to(device).eval()
criterion = nn.CrossEntropyLoss()

classes = ["m", "k"]


def test():
    avg_acc = 0
    avg_loss = 0
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0], batch[1]
            prediction = model(input.to(device))

            predicted_class = prediction.argmax().item()
            target_class = target.item()

            print(classes[predicted_class])

            targets.append(target_class)
            predictions.append(predicted_class)

            res = criterion(prediction, target.to(device)).item()

            avg_acc += (predicted_class == target_class)
            avg_loss += res

    print('Accuracy: ' + avg_acc/len(testing_data_loader))
    confused = confusion_matrix(targets, predictions)
    return confused

res = test()
fig, ax = plt.subplots()
im = ax.imshow(res)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

plt.setp(ax.get_xticklabels(), ha="right",
         rotation_mode="anchor")

for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, res[i, j],
                       ha="center", va="center", color="gray")

ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title("Confusion matrix")
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
fig.tight_layout()
plt.show()

