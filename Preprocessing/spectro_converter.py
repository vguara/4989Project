import os

import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
import librosa.display
import librosa


def generate_spectrogram(audio_path: str, output_path: str):
    """
    Converts an audio file into a black-and-white spectrogram and saves it as a PNG.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)

        # Create a Mel spectrogram
        spect_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spect_db = librosa.power_to_db(spect_array, ref=np.max)

        # Plot the spectrogram and save it
        plt.figure(figsize=(10, 4))
        plt.axis("off")  # Turn off axes
        librosa.display.specshow(spect_db, sr=sr, cmap="gray", fmax=8000)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved spectrogram to: {output_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def process_audio_files(input_folder: str, output_folder: str):
    """
    Recursively processes all audio files in a folder, generating App_Spectrograms
    and preserving the folder structure in the output folder.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp3"):  # Supported audio formats
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path, f"{os.path.splitext(file)[0]}.png")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                generate_spectrogram(input_path, output_path)


if __name__ == "__main__":
    input_folder = input("Enter the path to the folder containing audio files: ")
    output_folder = input("Enter the path to the output folder: ")

    process_audio_files(input_folder, output_folder)
