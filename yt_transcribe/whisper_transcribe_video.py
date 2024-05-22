import re
import os
import whisper
from pytube import YouTube
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from pytube.exceptions import VideoUnavailable, RegexMatchError

def sanitize_filename(filename):
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

def capitalize_names(text):
    capitalized_text = re.sub(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b", lambda x: x.group().title(), text)
    return capitized_text

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

def main(url, timestamp_interval=30):
    try:
        print("Starting transcription process...")

        # Extract the video ID from the URL
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube video URL")
        print(f"Video ID: {video_id}")

        # Create the YouTube object using the video ID
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        video = YouTube(video_url)
        video_title = video.title
        print(f"Video Title: {video_title}")

        # Sanitize the video title for the file name
        sanitized_title = sanitize_filename(video_title)
        print(f"Sanitized Title: {sanitized_title}")

        # Get the current date in the format YYYYMMDD
        current_date = datetime.now().strftime("%Y%m%d")

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

        # Check if the 'videos' folder exists, and create it if it doesn't
        videos_folder = "videos"
        if not os.path.exists(videos_folder):
            os.makedirs(videos_folder)
            print(f"Created folder: {videos_folder}")

        # Create the full file path for the video file
        video_file_path = os.path.join(videos_folder, f"{sanitized_title}_{current_date}.mp4")
        print(f"Video File Path: {video_file_path}")

        # Download the video
        print("Downloading video...")
        try:
            video.streams.get_highest_resolution().download(output_path=videos_folder, filename=f"{sanitized_title}_{current_date}.mp4")
            print("Video downloaded successfully.")
        except AttributeError as e:
            print(f"An error occurred while downloading the video: {str(e)}")
            print("Skipping video download and proceeding with transcription.")
            video_file_path = None
        except (VideoUnavailable, RegexMatchError) as e:
            print(f"An error occurred while accessing the video: {str(e)}")
            print("Skipping video download and proceeding with transcription.")
            video_file_path = None

        # Load the Whisper model
        model = whisper.load_model("base")
        print("Whisper model loaded.")

        # Transcribe the video using Whisper
        print("Transcribing video...")
        if video_file_path:
            result = model.transcribe(video_file_path)
            print("Video transcription completed.")
        else:
            print("Video file not available. Skipping transcription.")
            return

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
