import re
import os
import whisper
import yt_dlp
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import argparse
import subprocess
from tqdm import tqdm
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
from moviepy import VideoFileClip


def sanitize_filename(filename):
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

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

def check_video_exists(video_id):
    videos_dir = "videos"
    if os.path.exists(videos_dir):
        for video_file in os.listdir(videos_dir):
            if video_id in video_file:
                return os.path.join(videos_dir, video_file)
    return None

class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total, unit='B', unit_scale=True)

    def update(self, count):
        self.pbar.update(count)

    def close(self):
        self.pbar.close()

def download_video(url, video_path):
    progress_bar = None

    def progress_hook(d):
        nonlocal progress_bar
        if d['status'] == 'downloading':
            if progress_bar is None and 'total_bytes' in d:
                progress_bar = ProgressBar(d['total_bytes'])
            if progress_bar and 'downloaded_bytes' in d:
                progress_bar.update(d['downloaded_bytes'] - progress_bar.pbar.n)
        elif d['status'] == 'finished' and progress_bar:
            progress_bar.close()

    ydl_opts = {
        'format': 'best',
        'outtmpl': video_path,
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return info.get('title', None)
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None

def convert_to_wav(video_path):
    wav_path = video_path.rsplit('.', 1)[0] + '.wav'
    if not os.path.exists(wav_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(wav_path)
        video.close()
    return wav_path

def get_speaker_diarization(audio_path):
    wav_path = convert_to_wav(audio_path)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    )
    diarization = pipeline(wav_path)

    segments = []
    for turn, _, speaker in diarization.itertracks():
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    return segments

def assign_speakers_to_transcript(transcript_segments, diarization_segments):
    for t_segment in transcript_segments:
        start_time = t_segment['start']
        for d_segment in diarization_segments:
            if d_segment['start'] <= start_time <= d_segment['end']:
                t_segment['speaker'] = d_segment['speaker']
                break
    return transcript_segments

def is_sentence_end(text):
    return bool(re.search(r'[.!?][\s"]*$', text.strip()))

def main(url):
    try:
        print("Starting transcription process...")
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube video URL")
        print(f"Video ID: {video_id}")

        existing_video = check_video_exists(video_id)
        if existing_video:
            print(f"Video already exists: {existing_video}")
            video_file_path = existing_video
            video_title = os.path.splitext(os.path.basename(existing_video))[0]
        else:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                try:
                    video_info = ydl.extract_info(url, download=False)
                    video_title = video_info['title']
                    duration = video_info.get('duration', 0)

                    if duration > 4 * 3600:
                        raise ValueError("Video is too long (over 4 hours)")

                except Exception as e:
                    raise ValueError(f"Could not fetch video info: {str(e)}")

            print(f"Video Title: {video_title}")
            sanitized_title = sanitize_filename(video_title)
            print(f"Sanitized Title: {sanitized_title}")

            for folder in ['transcriptions', 'videos']:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    print(f"Created folder: {folder}")

            current_date = datetime.now().strftime("%Y%m%d")
            video_file_path = os.path.join('videos', f"{sanitized_title}_{current_date}.mp4")
            print(f"Video File Path: {video_file_path}")

            print("Downloading video...")
            if not download_video(url, video_file_path):
                raise ValueError("Failed to download video")
            print("Video downloaded successfully.")

        print("Running speaker diarization...")
        diarization_segments = get_speaker_diarization(video_file_path)
        print("Speaker diarization completed.")

        transcription_file_path = os.path.join('transcriptions',
                                             f"{os.path.splitext(os.path.basename(video_file_path))[0]}.txt")

        model = whisper.load_model("base")
        print("Whisper model loaded.")

        print("Transcribing video...")
        result = model.transcribe(video_file_path)
        print("Video transcription completed.")

        result["segments"] = assign_speakers_to_transcript(result["segments"], diarization_segments)

        with open(transcription_file_path, "w", encoding='utf-8') as f:
            f.write(f"{video_title}\n\n")

            current_timestamp = None
            current_speaker = None
            current_sentence = []

            for segment in result["segments"]:
                text = segment["text"].strip()
                speaker = segment.get("speaker", "UNKNOWN")

                if not current_timestamp:
                    current_timestamp = segment["start"]
                    current_speaker = speaker

                if current_speaker != speaker:
                    if current_sentence:
                        timestamp = format_timestamp(current_timestamp)
                        complete_sentence = ' '.join(current_sentence)
                        f.write(f"[{timestamp}] {current_speaker}: {complete_sentence}\n\n")
                        current_sentence = []
                        current_timestamp = segment["start"]
                        current_speaker = speaker

                current_sentence.append(text)

                if is_sentence_end(text):
                    timestamp = format_timestamp(current_timestamp)
                    complete_sentence = ' '.join(current_sentence)
                    f.write(f"[{timestamp}] {current_speaker}: {complete_sentence}\n\n")
                    current_sentence = []
                    current_timestamp = None
                    current_speaker = None

            if current_sentence:
                timestamp = format_timestamp(current_timestamp or result["segments"][-1]["start"])
                complete_sentence = ' '.join(current_sentence)
                f.write(f"[{timestamp}] {current_speaker}: {complete_sentence}\n\n")

        print(f"Transcription saved as {transcription_file_path}")

    except Exception as e:
        import traceback
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    # Check huggingface login
    try:
        subprocess.run(["huggingface-cli", "whoami"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Please login first using: huggingface-cli login")
        exit(1)

    parser = argparse.ArgumentParser(description='Transcribe YouTube videos using Whisper')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-u', '--url', help='Single YouTube video URL')
    group.add_argument('-f', '--file', help='File containing YouTube URLs (one per line)')
    args = parser.parse_args()

    if args.url:
        print(f"Processing single video: {args.url}")
        main(args.url)
    else:
        print(f"Processing videos from file: {args.file}")
        with open(args.file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]

        for i, url in enumerate(urls, 1):
            print(f"\nProcessing video {i} of {len(urls)}")
            print(f"URL: {url}")
            main(url)
