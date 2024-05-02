import os
import sys
import json
import uuid
from datetime import datetime
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_uuid():
    return str(uuid.uuid4())

def get_current_date():
    return datetime.now().strftime("%Y%m%d")

def get_current_time():
    return datetime.now().strftime("%H:%M:%S UTC")

def create_video_url(video_id):
    return f"https://www.youtube.com/watch?v={video_id}"

def search_videos(search_terms):
    # Create a YouTube Data API client
    youtube = build("youtube", "v3", developerKey=API_KEY)

    results = {}

    # Perform the search for each search term
    for term in search_terms:
        # Execute the search request
        search_response = youtube.search().list(
            q=term,
            type="video",
            part="id,snippet",
            maxResults=10
        ).execute()

        # Process the search results
        term_results = []
        for search_result in search_response.get("items", []):
            video_title = search_result["snippet"]["title"]
            video_id = search_result["id"]["videoId"]
            video_url = create_video_url(video_id)
            video_description = search_result["snippet"]["description"]
            channel_title = search_result["snippet"]["channelTitle"]
            published_at = search_result["snippet"]["publishedAt"]

            video_data = {
                "uuid": generate_uuid(),
                "source": channel_title,
                "url": video_url,
                "title": video_title,
                "published": published_at,
                "description": video_description,
                "body": "",
                "summary": "",
                "keywords": [],
                "image_url": search_result["snippet"]["thumbnails"]["default"]["url"],
                "robots_permission": True
            }

            term_results.append(video_data)

        results[term] = term_results

    return results

def save_to_json(data, file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    search_list_dir = os.path.join(current_dir, "search_list")

    # Create the "search_list" directory if it doesn't exist
    os.makedirs(search_list_dir, exist_ok=True)

    output_data = {
        "uuid": generate_uuid(),
        "current_dir": current_dir,
        "date": get_current_date(),
        "time": get_current_time(),
        "search_terms": sys.argv[1:],
        "data": data
    }

    file_path = os.path.join(search_list_dir, file_name)
    with open(file_path, "w") as file:
        json.dump(output_data, file, indent=4)

    return file_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_videos.py <search_term1> <search_term2> ...")
        sys.exit(1)

    search_terms = sys.argv[1:]
    search_results = search_videos(search_terms)

    current_date = get_current_date()
    search_terms_underscored = [term.replace(" ", "_") for term in search_terms]
    output_file = f"{('_'.join(search_terms_underscored))}_{current_date}.json"
    file_path = save_to_json(search_results, output_file)
    print(f"Search results saved to {file_path}")
