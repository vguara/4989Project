import os
from pydub import AudioSegment


def split_mp3_files(folder_path, chunk_duration=30):
    """
    Splits all .mp3 files in the specified folder into chunks of specified duration.

    Parameters:
        folder_path (str): Path to the folder containing .mp3 files.
        chunk_duration (int): Duration of each chunk in seconds (default: 30 seconds).
    """
    output_folder = os.path.join(folder_path, "split_audio")
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp3"):
            file_path = os.path.join(folder_path, file_name)
            try:
                audio = AudioSegment.from_file(file_path)
                duration_in_seconds = len(audio) / 1000  # Convert milliseconds to seconds

                if duration_in_seconds < chunk_duration:
                    print(f"Skipping {file_name}: Not enough data for a 30-second chunk.")
                    continue

                # Split audio into chunks
                base_name = os.path.splitext(file_name)[0]
                num_chunks = int(duration_in_seconds // chunk_duration)

                for i in range(num_chunks):
                    start_time = i * chunk_duration * 1000  # Convert to milliseconds
                    end_time = start_time + chunk_duration * 1000
                    chunk = audio[start_time:end_time]

                    # Save each chunk
                    chunk_file_name = f"{base_name}_chunk{i + 1}.mp3"
                    chunk_path = os.path.join(output_folder, chunk_file_name)
                    chunk.export(chunk_path, format="mp3", bitrate="192k")
                    print(f"Exported {chunk_file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing .mp3 files: ")
    split_mp3_files(folder_path)
