import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.utils import to_categorical


def load_images_from_folder(folder):
    images = []
    labels = []
    # mapping emotions to int values
    emotion_mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

    for emotion_label in os.listdir(folder):
        label = emotion_mapping.get(emotion_label)
        if label is not None:
            emotion_folder = os.path.join(folder, emotion_label)
            for filename in os.listdir(emotion_folder):
                img_path = os.path.join(emotion_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)


# Load images from 'train' folder for visualization
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# Function to visualize the data
def visualize_data(X, y, num_per_emotion=5):
    rows = 7
    cols = num_per_emotion

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 12))

    for row in range(rows):
        emotion_indices = np.where(y == row)[0]
        selected_indices = np.random.choice(emotion_indices, size=num_per_emotion, replace=False)

        for col, img_index in enumerate(selected_indices):
            img = X[img_index].reshape(48, 48)
            label = emotions[row]

            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(f"Emotion: {label} - {row}", fontsize=10)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


# Load images from 'train' folder for visualization
X_visualize, y_visualize = load_images_from_folder('train')
visualize_data(X_visualize, y_visualize, num_per_emotion=5)

# Load images from 'train' and 'test' folders
X_train, y_train = load_images_from_folder('train')
X_test, y_test = load_images_from_folder('test')

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Build the model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 output classes for the 7 emotions

# Compile the model with different learning rates or optimizers
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

# Save the model in HDF5 format
model.save('emotion_model.h5')
