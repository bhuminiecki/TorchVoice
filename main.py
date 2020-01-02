import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam, adadelta
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from config import TRAIN_DIR, TRAIN_SIZE, TEST_DIR, TEST_SIZE, EPOCHS, CLASS_NAMES
from sklearn.metrics import confusion_matrix
from time import gmtime, strftime
from AudioDataGenerator import AudioDataGenerator

def generate_data(image):
    orb = cv.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)
    return des


# CREATE DATA GENERATORS

train_sound_generator = AudioDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                shift=.2,
                horizontal_flip=True,
                zca_whitening=True
)

validation_sound_generator = AudioDataGenerator()

train_data_generator = train_sound_generator.flow_from_directory(
    directory=TRAIN_DIR,
    batch_size=TRAIN_SIZE
)

validation_data_generator = validation_sound_generator.flow_from_directory(
    directory=TEST_DIR,
    batch_size=TEST_SIZE
)

# CREATE MODEL

model = keras.Sequential([
    keras.layers.Conv1D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling1D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(24, activation="softmax")
])

# COMPILE MODEL

model.compile(optimizer=adam(learning_rate=0.001, amsgrad=True),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# TRAIN MODEL

history = model.fit_generator(train_data_generator, validation_data=validation_data_generator, epochs=EPOCHS)

# SAVE MODEL

model.save("output/model.h5")

# VISUALISE LEARNING PROCESS

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

progress = plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

Y_pred = model.predict_generator(validation_data_generator)
y_pred = np.argmax(Y_pred, axis=1)
confused = confusion_matrix(validation_data_generator.classes, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(confused)

ax.set_xticks(np.arange(len(CLASS_NAMES)))
ax.set_yticks(np.arange(len(CLASS_NAMES)))

ax.set_xticklabels(CLASS_NAMES)
ax.set_yticklabels(CLASS_NAMES)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        text = ax.text(j, i, confused[i, j],
                       ha="center", va="center", color="w")

ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_title("Confusion matrix")
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')
fig.tight_layout()
plt.show()

time = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
progress.savefig("output/visualisation/progress-"+time+".jpg")
fig.savefig("output/visualisation/confusionmattrix-"+time+".jpg")
