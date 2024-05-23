import re
import os
import whisper
import youtube_dl
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from youtube_dl.utils import DownloadError

def sanitize_filename(filename):
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

def capitalize_names(text):
    capitalized_text = re.sub(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b", lambda x: x.group().title(), text)
    return capitalized_text

def format_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}"

def extract_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")
    return None

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', None)
    except DownloadError as e:
        print(f"An error occurred while downloading the video: {str(e)}")
        video_title = None
    return video_title

def main(url, timestamp_interval=30):
    try:
        print("Starting transcription process...")

        # Extract the video ID from the URL
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube video URL")
        print(f"Video ID: {video_id}")

        # Get the current date in the format YYYYMMDD
        current_date = datetime.now().strftime("%Y%m%d")

        # Check if the 'videos' folder exists, and create it if it doesn't
        videos_folder = "videos"
        if not os.path.exists(videos_folder):
            os.makedirs(videos_folder)
            print(f"Created folder: {videos_folder}")

        # Download the video using youtube-dl
        print("Downloading video...")
        video_file_path = os.path.join(videos_folder, f"video_{current_date}.mp4")
        video_title = download_video(url, video_file_path)

        if video_title is None:
            print("Video download failed. Skipping transcription.")
            return

        print(f"Video Title: {video_title}")

        # Sanitize the video title for the file name
        sanitized_title = sanitize_filename(video_title)
        print(f"Sanitized Title: {sanitized_title}")

        # Create the file name with the sanitized video title and current date
        file_name = f"{sanitized_title}_{current_date}.txt"
        print(f"File Name: {file_name}")

        # Check if the 'transcriptions' folder exists, and create it if it doesn't
        transcriptions_folder = "transcriptions"
        if not os.path.exists(transcriptions_folder):
            os.makedirs(transcriptions_folder)
            print(f"Created folder: {transcriptions_folder}")

        # Create the full file path for the transcription file
        transcription_file_path = os.path.join(transcriptions_folder, file_name)
        print(f"Transcription File Path: {transcription_file_path}")

        # Load the Whisper model
        model = whisper.load_model("base")
        print("Whisper model loaded.")

        # Transcribe the video using Whisper
        print("Transcribing video...")
        result = model.transcribe(video_file_path)
        print("Video transcription completed.")

        # Process the transcription
        transcription = result["text"]

        # Capitalize names using the regular expression pattern
        transcription = capitalize_names(transcription)

        # Capitalize names of the month and the pronoun "I"
        transcription = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", lambda x: x.group().capitalize(), transcription)
        transcription = re.sub(r"\bi\b", "I", transcription)

        # Split the transcription into segments based on the timestamp interval
        segments = []
        current_segment = ""
        current_timestamp = 0

        for segment in result["segments"]:
            start = segment["start"]
            text = segment["text"]

            if start >= current_timestamp + timestamp_interval:
                if current_segment:
                    segments.append((current_timestamp, current_segment))
                current_segment = ""
                current_timestamp = start

            current_segment += " " + text

        if current_segment:
            segments.append((current_timestamp, current_segment))

        # Open the file in write mode
        with open(transcription_file_path, "w") as f:
            f.write(f"{video_title}\n\n")

            for timestamp, segment in segments:
                formatted_timestamp = format_timestamp(timestamp)
                f.write(f"[{formatted_timestamp}] {segment.strip()}\n\n")

        print(f"Transcription saved as {transcription_file_path}")

    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python whisper_transcribe_video.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    main(url)
