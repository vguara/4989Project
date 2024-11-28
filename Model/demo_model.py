import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Paths to the datasets
dataset1_path = "ImageData/SunoAI/SunoCaps"
dataset2_path = "ImageData/Real/GTZAN"

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

def load_images_from_folder(folder, label):
    """
    Load images from a folder and assign the given label.
    """
    images = []
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if filepath.endswith(".png"):  # Ensure it's an image
            img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load Dataset1 and Dataset2
print("Loading Dataset1...")
dataset1_images, dataset1_labels = load_images_from_folder(dataset1_path, 0)

print("Loading Dataset2...")
dataset2_images, dataset2_labels = load_images_from_folder(dataset2_path, 1)

# Combine the datasets
images = np.concatenate([dataset1_images, dataset2_images], axis=0)
labels = np.concatenate([dataset1_labels, dataset2_labels], axis=0)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=32
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Save the model
model.save("spectrogram_classifier.h5")
print("Model saved as 'spectrogram_classifier.h5'")
