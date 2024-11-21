# YT_Transcribe

![symb](https://github.com/mrdavtan/YT_Transcribe/assets/21132073/af830f85-cf54-46ef-8bb5-9cf701e90349)

# YouTube Video Transcription CLI Tool

## Description

The YouTube Video Transcription CLI Tool is a Python script that allows you to easily transcribe YouTube videos and save the transcriptions as formatted text files. It utilizes `yt-dlp` for video downloading, the OpenAI `whisper` library for transcription, and includes features for text processing and formatting.

The tool supports both single video transcription and batch processing of multiple videos from a list of URLs.

## Features

- Transcribe single or multiple YouTube videos using Whisper's speech recognition
- Download videos efficiently using yt-dlp
- Preserve natural speech segments with accurate timestamps
- Automatically format and capitalize text for improved readability
- Generate organized output with video title and timestamps
- Save transcriptions in a structured directory format
- Process multiple videos from a text file

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/youtube-transcription-tool.git
```

2. Navigate to the project directory:
```bash
cd youtube-transcription-tool
```

3. Create a Python virtual environment:
```bash
python -m venv .venv
```

4. Activate the virtual environment:
```bash
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

5. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Video Transcription
To transcribe a single YouTube video:
```bash
python whisper_transcribe_videos.py -u "https://youtube.com/watch?v=..."
# or
python whisper_transcribe_videos.py --url "https://youtube.com/watch?v=..."
```

### Multiple Videos Transcription
To transcribe multiple videos, create a text file with one YouTube URL per line, then:
```bash
python whisper_transcribe_video.py -f urls.txt
# or
python whisper_transcribe_video.py --file urls.txt
```

Example `urls.txt` format:
```text
https://www.youtube.com/watch?v=video1
https://www.youtube.com/watch?v=video2
https://www.youtube.com/watch?v=video3
```

### Output
The script will:
1. Create a `videos` directory for downloaded videos
2. Create a `transcriptions` directory for output files
3. Generate transcription files with the format: `{video_title}_{date}.txt`
4. Include timestamps and properly formatted text in the transcription

### Output Format
The transcription file will contain:
```text
Video Title

[00:00:00.000] First segment of speech...

[00:00:04.123] Next segment of speech...

[00:00:08.456] And so on...
```

## License

This project is licensed under the MIT License.
