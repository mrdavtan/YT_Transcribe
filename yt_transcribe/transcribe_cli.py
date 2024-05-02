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

def main(url, language):
    try:
        parsed_url = urlparse(url)
        video_id = ""
        if "youtu.be" in url:
            path_segments = parsed_url.path.split("/")
            video_id = path_segments[-1]
        else:
            parsed_dict = parse_qs(parsed_url.query)
            video_id = parsed_dict["v"][0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])

        # Concatenate the text of each segment
        transcript_text = " ".join([segment["text"] for segment in transcript])

        # Load the multilingual punctuation model
        model = PunctuationModel()

        # Restore punctuation in the text
        punctuated_text = model.restore_punctuation(transcript_text)

        # Capitalize the first letter of each sentence
        sentences = sent_tokenize(punctuated_text)
        capitalized_sentences = [sentence.capitalize() for sentence in sentences]
        capitalized_text = " ".join(capitalized_sentences)

        # Capitalize names, names of the month, and the pronoun "I"
        capitalized_text = re.sub(r"\b(Stuart|Alexander|Bruce|Paul|Dimitri|Jared|Michael|Nassim|Brian)\b", lambda x: x.group().capitalize(), capitalized_text)
        capitalized_text = re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", lambda x: x.group().capitalize(), capitalized_text)
        capitalized_text = re.sub(r"\bi\b", "I", capitalized_text)

        # Split the text into paragraphs
        paragraphs = text_to_paragraphs(capitalized_text)

        # Extract the video title using pytube
        video = YouTube(url)
        video_title = video.title

        # Get the current date in the format YYYYMMDD
        current_date = datetime.now().strftime("%Y%m%d")

        # Create the file name with the video title and current date
        file_name = f"{video_title}_{current_date}.txt"

        with open(file_name, "w") as f:
            f.write(f"{video_title}\n\n")
            for paragraph in paragraphs:
                f.write(f"{paragraph}\n\n")
        print(f"Transcription saved as {file_name}")
    except NoTranscriptFound:
        print(f"No {language} transcript is available for this YouTube video.")
    except TranscriptsDisabled:
        print("Transcripts are disabled for this YouTube video.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transcribe_cli.py <youtube_url> <language>")
        sys.exit(1)

    url = sys.argv[1]
    language = sys.argv[2]
    main(url, language)
