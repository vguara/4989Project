import os
import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pygame  # For audio playback

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
SPECTROGRAM_DIR = "App_Spectrograms"
COVERS_DIR = "App_Covers"
MP3_DIR = "App_Audio"
MODEL_PATH = "spectrogram_classifier.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Global state for currently selected track
current_track_row = None
playing_song_button = None


def predict_spectrogram(image_path):
    """
    Predict whether a spectrogram belongs to AI-Generated or Not.
    """
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = img_array[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_array)
    return "AI-Generated" if prediction[0] < 0.5 else "Not AI-Generated"


def get_cover_image(song_name, size=(50, 50)):
    """
    Get the album cover for a song or return a placeholder, scaled to the given size.
    """
    cover_path = os.path.join(COVERS_DIR, song_name + ".jpeg")
    if os.path.exists(cover_path):
        cover_image = Image.open(cover_path).resize(size)
    else:
        cover_image = Image.open("App_Covers/placeholder.jpeg").resize(size)
    return ImageTk.PhotoImage(cover_image)


def play_audio(song_name, button):
    """
    Play the audio associated with the song. Toggle between play and stop states.
    """
    global current_track_row, playing_song_button
    audio_path = os.path.join(MP3_DIR, song_name + ".mp3")

    # Stop playback if the same song is already playing
    if pygame.mixer.music.get_busy() and playing_song_button == button:
        pygame.mixer.music.stop()
        button.config(text="Play")
        playing_song_button = None
        return

    # Stop the currently playing song (if any) before starting a new one
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        if playing_song_button:
            playing_song_button.config(text="Play")

    # Load and play the new song
    if os.path.exists(audio_path):
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        button.config(text="Stop")
        playing_song_button = button

    # Highlight the selected track
    for widget in song_list_frame.winfo_children():
        widget.configure(bg="white")  # Reset all rows to default

    if current_track_row:
        current_track_row.configure(bg="white")  # Reset the previous row

    # Highlight the new row
    for widget in song_list_frame.winfo_children():
        if widget.winfo_name() == f"row_{song_name}":
            widget.configure(bg="lightgrey")
            current_track_row = widget
            break


def display_prediction(song_name, spectrogram_path):
    """
    Display the prediction and corresponding album cover, scaled up.
    """
    # Get prediction
    prediction = predict_spectrogram(spectrogram_path)
    prediction_label.config(text=f"Prediction: {prediction}")

    # Update cover display (scaled up)
    cover_image = get_cover_image(song_name, size=(200, 200))
    cover_label.config(image=cover_image)
    cover_label.image = cover_image


def load_songs():
    """
    Load all songs in the spectrogram directory and display them with buttons.
    """
    files = [f for f in os.listdir(SPECTROGRAM_DIR) if f.endswith(".png")]

    for widget in song_list_frame.winfo_children():
        widget.destroy()  # Clear existing items

    for file in files:
        song_name, _ = os.path.splitext(file)

        # Create a row for each song
        row_frame = tk.Frame(song_list_frame, name=f"row_{song_name}", bg="white")
        row_frame.pack(fill="x", pady=5)

        # Small album cover
        cover_image = get_cover_image(song_name)
        cover_label = tk.Label(row_frame, image=cover_image, bg="white")
        cover_label.image = cover_image
        cover_label.pack(side="left", padx=10)

        # Song name label
        tk.Label(row_frame, text=song_name, anchor="w", bg="white").pack(side="left", padx=10)

        # Play/Stop button
        play_button = tk.Button(row_frame, text="Play", bg="lightblue")
        play_button.config(command=lambda sn=song_name, btn=play_button: play_audio(sn, btn))
        play_button.pack(side="right", padx=10)

        # Prediction button
        spectrogram_path = os.path.join(SPECTROGRAM_DIR, file)
        tk.Button(
            row_frame,
            text="Run Detection",
            command=lambda sn=song_name, sp=spectrogram_path: display_prediction(sn, sp),
            bg="lightgreen"
        ).pack(side="right", padx=10)


# Initialize Tkinter
root = tk.Tk()
root.title("Spectrogram AI Detector")
root.geometry("800x600")

# Top frame: Song list
song_list_frame = tk.Frame(root)
song_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Bottom frame: Display prediction and cover
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill="x", padx=10, pady=10)

# Cover display (scaled-up size)
cover_label = tk.Label(bottom_frame, text="Album Cover", width=200, height=200)
cover_label.pack(side="left", padx=10)

# Prediction result
prediction_label = tk.Label(bottom_frame, text="Prediction: None", font=("Arial", 14))
prediction_label.pack(side="left", padx=10)

# Load songs on app start
load_songs()

# Run the app
root.mainloop()
