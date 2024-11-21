import requests
from bs4 import BeautifulSoup
import re
import sys
from datetime import datetime

if len(sys.argv) < 2:
    print("Please provide the session URL as a command-line argument.")
    sys.exit(1)

session_url = sys.argv[1]

session = requests.Session()

try:
    response = session.get(session_url)
    print(f"Response status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the <iframe> tag with the YouTube video URL
    iframe = soup.find('iframe', src=lambda src: src and 'youtube.com/embed' in src)

    if iframe:
        youtube_embed_url = iframe['src']
        print(f"Found YouTube embed URL: {youtube_embed_url}")

        # Extract the YouTube video ID from the embed URL
        video_id_match = re.search(r'youtube\.com/embed/([^?]+)', youtube_embed_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            youtube_video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Extracted YouTube video URL: {youtube_video_url}")

            current_date = datetime.now().strftime("%Y%m%d")
            output_file = f"url_{current_date}.txt"
            with open(output_file, 'w') as file:
                file.write(youtube_video_url)

            print(f"YouTube video URL saved to {output_file}.")
        else:
            print("Could not extract YouTube video ID from the embed URL.")
    else:
        print("No YouTube video found on the session page.")

except requests.exceptions.RequestException as e:
    print(f"Error occurred during the request: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
