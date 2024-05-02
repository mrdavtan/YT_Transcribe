import os
import sys
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")

def search_videos(search_terms):
    # Create a YouTube Data API client
    youtube = build("youtube", "v3", developerKey=API_KEY)

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
        print(f"Search results for term: {term}")
        for search_result in search_response.get("items", []):
            video_title = search_result["snippet"]["title"]
            video_id = search_result["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            print(f"Title: {video_title}")
            print(f"URL: {video_url}")
            print()

        print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_videos.py <search_term1> <search_term2> ...")
        sys.exit(1)

    search_terms = sys.argv[1:]
    search_videos(search_terms)
