import tensorflow as tf
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
        color_mode="grayscale",
        seed=42
    )

    # good for performance according to my good friend chatGPT
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    cnn_model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(141, 291, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=128, activation='relu'),
        Dense(units=2, activation='sigmoid')
    ])

    cnn_model.summary()

    # Compile the model
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fit the model
    cnn_model.fit(train_data, epochs=10, verbose=1)

    # Evaluate the model
    loss, accuracy = cnn_model.evaluate(test_data)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Save the model
    cnn_model.save("spectrogram_cnn_model.h5")
