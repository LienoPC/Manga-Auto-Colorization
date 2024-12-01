
import requests
import os

# Define parameters
base_url = "https://safebooru.org/index.php?page=dapi&s=post&q=index"
save_dir = "./images"
os.makedirs(save_dir, exist_ok=True)

# Function to download images
def download_images(tag, pages=10, limit=100):
    for pid in range(pages):
        params = {
            "tags": tag,
            "limit": limit,
            "pid": pid
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            continue

        # Parse XML response for image URLs
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        for post in root.findall("post"):
            image_url = post.get("file_url")
            if image_url:
                try:
                    img_data = requests.get(image_url).content
                    filename = os.path.join(save_dir, os.path.basename(image_url))
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"Downloaded: {filename}")
                except Exception as e:
                    print(f"Failed to download {image_url}: {e}")

# Example: Download images tagged 'anime'
download_images(tag="", pages=300)
