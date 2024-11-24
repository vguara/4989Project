import os

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.signal import spectrogram


def generate_spectrogram(audio_path: str, output_path: str):
    """
    Converts an audio file into a black-and-white spectrogram and saves it as a PNG.
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # Compute the spectrogram
        frequencies, times, spectro_data = spectrogram(samples, fs=sample_rate)

        # Convert spectrogram data to dB for better visualization
        spectro_data = 10 * np.log10(spectro_data + 1e-10)  # Avoid log(0)

        # Plot and save the spectrogram
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times, frequencies, spectro_data, shading='gouraud', cmap='gray')
        plt.axis('off')  # Remove axes for a clean image
        plt.tight_layout()

        # Save as PNG
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved spectrogram to: {output_path}")

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def process_audio_files(input_folder: str, output_folder: str):
    """
    Recursively processes all audio files in a folder, generating spectrograms
    and preserving the folder structure in the output folder.
    """
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp3"):  # Supported audio formats
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path, f"{os.path.splitext(file)[0]}.png")
                generate_spectrogram(input_path, output_path)


if __name__ == "__main__":
    input_folder = input("Enter the path to the folder containing audio files: ")
    output_folder = input("Enter the path to the output folder: ")

    process_audio_files(input_folder, output_folder)
