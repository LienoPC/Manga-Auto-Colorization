import requests
import os
import time
from xml.etree import ElementTree as ET

# Define parameters
base_url = "https://safebooru.org/index.php?page=dapi&s=post&q=index"
save_dir = "./images"
os.makedirs(save_dir, exist_ok=True)


# Function to download images
def download_images(tag, pages=10, limit=100):
    for pid in range(pages):
        params = {"tags": tag, "limit": limit, "pid": pid}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            continue

        # Parse XML response for image URLs
        try:
            root = ET.fromstring(response.content)
            for post in root.findall("post"):
                image_url = post.get("file_url")
                if image_url:
                    try:
                        img_response = requests.get(image_url, timeout=10)
                        img_response.raise_for_status()
                        filename = os.path.join(save_dir, os.path.basename(image_url))
                        with open(filename, "wb") as f:
                            f.write(img_response.content)
                        print(f"Downloaded: {filename}")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to download {image_url}: {e}")
        except ET.ParseError as e:
            print(f"XML parsing failed: {e}")

        # Avoid overwhelming the server
        time.sleep(1)


# Start downloading
download_images(tag="one_piece", pages=10)