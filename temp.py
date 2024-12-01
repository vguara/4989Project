import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# Constants
MP3_FOLDER = "App_Audio"  # Input folder containing .mp3 files
SPECTROGRAM_FOLDER = "App_Spectrograms"  # Output folder for App_Spectrograms

def generate_spectrogram(mp3_path, output_path):
    """
    Generate a spectrogram from an MP3 file and save it as a black-and-white PNG.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(mp3_path, sr=None)

        # Create a Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot the spectrogram and save it
        plt.figure(figsize=(10, 4))
        plt.axis("off")  # Turn off axes
        librosa.display.specshow(S_dB, sr=sr, cmap="gray", fmax=8000)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Spectrogram saved: {output_path}")
    except Exception as e:
        print(f"Error processing {mp3_path}: {e}")

def convert_all_mp3s_to_spectrograms(mp3_folder, spectrogram_folder):
    """
    Convert all MP3 files in a folder to spectrogram PNG files.
    """
    if not os.path.exists(spectrogram_folder):
        os.makedirs(spectrogram_folder)

    for file in os.listdir(mp3_folder):
        if file.endswith(".mp3"):
            mp3_path = os.path.join(mp3_folder, file)
            output_name = os.path.splitext(file)[0] + ".png"
            output_path = os.path.join(spectrogram_folder, output_name)

            generate_spectrogram(mp3_path, output_path)

if __name__ == "__main__":
    # Ensure input and output folders are correct
    convert_all_mp3s_to_spectrograms(MP3_FOLDER, SPECTROGRAM_FOLDER)
