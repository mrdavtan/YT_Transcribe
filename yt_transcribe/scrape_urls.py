import requests
from bs4 import BeautifulSoup
import sys

if len(sys.argv) < 2:
    print("Please provide the URL as a command-line argument.")
    sys.exit(1)

url = sys.argv[1]

session = requests.Session()

try:
    response = session.get(url)
    print(f"Response status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a', href=True)
    print(f"Found {len(links)} links.")

    urls = [link['href'] for link in links]

    print("Extracted URLs:")
    for url in urls:
        print(url)

except requests.exceptions.RequestException as e:
    print(f"Error occurred during the request: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
