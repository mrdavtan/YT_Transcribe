# YT_Transcribe

![symb](https://github.com/mrdavtan/YT_Transcribe/assets/21132073/af830f85-cf54-46ef-8bb5-9cf701e90349)

YouTube Video Transcription CLI Tool
====================================

This Python script allows you to transcribe YouTube videos and save the transcription as a text file. The script uses the `youtube_transcript_api` library to retrieve the video transcript, the `deepmultilingualpunctuation` library to restore punctuation, and the `pytube` library to extract the video title.

Requirements
------------

-   Python 3.x
-   `youtube_transcript_api` library
-   `deepmultilingualpunctuation` library
-   `nltk` library
-   `pytube` library

Installation
------------

1.  Clone the repository or download the `transcribe_cli.py` file.
2.  Install the required libraries by running the following command:A command-line tool for transcribing YouTube videos and saving the transcriptions as formatted text files.

## Description
-----------

* * * * *

The YouTube Video Transcription CLI Tool is a Python script that allows you to easily transcribe YouTube videos and save the transcriptions as formatted text files. It utilizes the `youtube_transcript_api` library to retrieve video transcripts, the `deepmultilingualpunctuation` library to restore punctuation, and the `pytube` library to extract video titles.

With this tool, you can quickly obtain transcriptions of YouTube videos in your desired language, with the text properly formatted, capitalized, and saved with a descriptive file name.

## Features
--------

* * * * *

-   Transcribe YouTube videos by providing the video URL and desired language
-   Restore punctuation in the transcribed text using a multilingual punctuation model
-   Capitalize sentences and specific words for improved readability
-   Extract the video title and use it to generate a formatted file name
-   Save the transcription as a text file with the video title and current date

## Installation
------------

* * * * *

To use the YouTube Video Transcription CLI Tool, follow these steps:

1.  Clone the repository:

```bash

git clone https://github.com/your-username/youtube-transcription-tool.git

```

2. Navigate to the project directory:

```bash

cd youtube-transcription-tool

```

3. Create a Python virtual environment

```bash
python -m venv .venv
```

4. Activate the virtual environment

```bash

source .venv/bin/actviate

```

5. Install the required dependencies:

```bash
pip install -r requirements.txt

```

## Usage

To transcribe a YouTube video, run the script with the following command:

```bash

python transcribe_cli.py <youtube_url> <language>

```

Replace <youtube_url> with the URL of the YouTube video you want to transcribe, and <language> with the language code of the desired transcript language (e.g., "en" for English, "es" for Spanish).
The script will retrieve the transcript, restore punctuation, capitalize sentences and specific words, and save the transcription as a text file with a formatted file name based on the video title and current date.

## License

This project is licensed under the MIT License.

