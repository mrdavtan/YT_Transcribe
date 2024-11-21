import re
import os
import whisper
import yt_dlp
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import argparse

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

def main(url):
    try:
        print("Starting transcription process...")

        # Extract the video ID from the URL
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube video URL")
        print(f"Video ID: {video_id}")

        # Get video info using yt-dlp
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            video_info = ydl.extract_info(url, download=False)
            video_title = video_info['title']
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

        # Download the video using yt-dlp
        print("Downloading video...")
        ydl_opts = {
            'format': 'best',
            'outtmpl': video_file_path,
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Video downloaded successfully.")

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

        # Open the file in write mode
        with open(transcription_file_path, "w") as f:
            f.write(f"{video_title}\n\n")

            # Write segments with their timestamps
            for segment in result["segments"]:
                timestamp = format_timestamp(segment["start"])
                text = segment["text"].strip()
                f.write(f"[{timestamp}] {text}\n\n")

        print(f"Transcription saved as {transcription_file_path}")

    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe YouTube videos using Whisper')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--url', help='Single YouTube video URL')
    group.add_argument('-f', '--file', help='File containing YouTube URLs (one per line)')
    args = parser.parse_args()

    if args.url:
        # Process single video
        print(f"Processing single video: {args.url}")
        main(args.url)
    else:
        # Process multiple videos from file
        print(f"Processing videos from file: {args.file}")
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]

        for i, url in enumerate(urls, 1):
            print(f"\nProcessing video {i} of {len(urls)}")
            print(f"URL: {url}")
            main(url)
