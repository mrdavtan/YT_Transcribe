import requests
from bs4 import BeautifulSoup
import sys

if len(sys.argv) < 2:
    print("Please provide the URL as a command-line argument.")
    sys.exit(1)

url = sys.argv[1]

session = requests.Session()

login_url = 'https://dataaisummit.databricks.com/flow/db/dais2024/GVIR/login'
login_data = {
    'username': 'virtualpaul@gmail.com',
    'password': 'MP24af@data'
}
response = session.post(login_url, data=login_data)

response = session.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
youtube_links = soup.find_all('a', class_='youtube-link')

urls = [link['href'] for link in youtube_links]

for url in urls:
    print(url)
