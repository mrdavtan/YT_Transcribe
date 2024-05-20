import re
import sys
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs
from deepmultilingualpunctuation import PunctuationModel
import nltk
from nltk.tokenize import sent_tokenize
from pytube import YouTube
from datetime import datetime
import os
import spacy


def text_to_paragraphs(text, max_sentences_per_paragraph=10, max_paragraph_length=300):
    sentence_delimiters = [".", "!", "?"]
    sentences = []
    current_sentence = ""
    paragraphs = []
    current_paragraph = ""
    paragraph_length = 0

    for char in text:
        current_sentence += char
        if char in sentence_delimiters:
            sentences.append(current_sentence)
            current_sentence = ""

    if current_sentence:
        sentences.append(current_sentence)

    for sentence in sentences:
        sentence = sentence.strip()  # Remove leading/trailing whitespace and newlines
        if sentence:
            if current_paragraph:
                if (paragraph_length + len(sentence) + 1 <= max_paragraph_length) and (
                    len(current_paragraph.split(".")) < max_sentences_per_paragraph
                ):
                    current_paragraph += " "  # Add space between sentences
                    current_paragraph += sentence
                    paragraph_length += len(sentence) + 1  # +1 for the space
                else:
                    paragraphs.append(current_paragraph)
                    current_paragraph = sentence
                    paragraph_length = len(sentence)
            else:
                current_paragraph = sentence
                paragraph_length = len(sentence)

    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs

def sanitize_filename(filename):
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized


def capitalize_names(text):
    capitalized_text = re.sub(r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b", lambda x: x.group().title(), text)
    return capitalized_text


def main(url, max_segment_duration=600):  # Default segment duration of 10 minutes
    try:
        parsed_url = urlparse(url)
        video_id = ""
        if "youtu.be" in url:
            path_segments = parsed_url.path.split("/")
            video_id = path_segments[-1]
        else:
            parsed_dict = parse_qs(parsed_url.query)
            video_id = parsed_dict["v"][0]

        # Extract the video title using pytube
        video = YouTube(url)
        video_title = video.title

        # Sanitize the video title for the file name
        sanitized_title = sanitize_filename(video_title)

        # Get the current date in the format YYYYMMDD
        current_date = datetime.now().strftime("%Y%m%d")

        # Create the file name with the sanitized video title and current date
        file_name = f"{sanitized_title}_{current_date}.txt"

        # Check if the 'transcriptions' folder exists, and create it if it doesn't
        transcriptions_folder = "transcriptions"
        if not os.path.exists(transcriptions_folder):
            os.makedirs(transcriptions_folder)

        # Create the full file path for the transcription file
        file_path = os.path.join(transcriptions_folder, file_name)

        # Open the file in append mode
        with open(file_path, "a") as f:
            f.write(f"{video_title}\n\n")

            # Get the video duration
            video_duration = video.length

            # Process the video in segments
            start_time = 0
            while start_time < video_duration:
                end_time = min(start_time + max_segment_duration, video_duration)
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                segment_transcript = [
                    segment for segment in transcript
                    if start_time <= segment['start'] < end_time
                ]

                # Process each segment of the transcript
                for segment in segment_transcript:
                    segment_start_time = segment['start']
                    segment_text = segment['text']

                    # Load the multilingual punctuation model
                    model = PunctuationModel()

                    # Restore punctuation in the text
                    punctuated_text = model.restore_punctuation(segment_text)

                    # Capitalize the first letter of each sentence
                    sentences = sent_tokenize(punctuated_text)
                    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
                    capitalized_text = " ".join(capitalized_sentences)

                    # Capitalize names using the regular expression pattern
                    capitalized_text = capitalize_names(capitalized_text)

                    # Capitalize names of the month and the pronoun "I"
                    capitalized_text = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", lambda x: x.group().capitalize(), capitalized_text)
                    capitalized_text = re.sub(r"\bi\b", "I", capitalized_text)

                    # Write the timestamp and processed text to the file
                    f.write(f"[{segment_start_time:.2f}] {capitalized_text}\n")

                f.write("\n")
                start_time = end_time

        print(f"Transcription saved as {file_path}")
    except NoTranscriptFound:
        print("No transcript is available for this YouTube video.")
    except TranscriptsDisabled:
        print("Transcripts are disabled for this YouTube video.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_cli.py <youtube_url>")
        sys.exit(1)

    url = sys.argv[1]
    main(url)
