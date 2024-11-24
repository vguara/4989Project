import os
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image_dataset_from_directory


if __name__ == "__main__":
    dataset_path = input("Enter the path of the spectrogram dataset: ")
    batch_size = 32
    img_size = (141, 291)

    train_data, test_data = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="both",
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="grayscale"
    )

    # good for performance according to my good friend chatGPT
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    cnn_model = Sequential()
    # cnn_model.add(Input(shape=(28, 28, 1)))
    # cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    # cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(units=100, activation='relu'))
    # cnn_model.add(Dense(units=num_classes, activation='softmax'))


