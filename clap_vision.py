import requests
from langchain_community.llms import Ollama
import base64
import base64
import requests
from pathlib import Path
import os
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)
ollama_chat = Ollama(
    model="llava:13b-v1.5-fp16",
)

# Prepare headers and data

def download_image(url, filepath):
    response = requests.get(url, stream=True, headers={'Authorization':'Bearer xoxb-80968956049-4810800004850-8JTSCFIupHSxWG1DThbS9mZZ'})
    if response.status_code == 200:
        with open(filepath, 'wb') as image_file:
            for chunk in response.iter_content(1024):
                if chunk:
                    image_file.write(chunk)
        logging.info("Image downloaded successfully")
    else:
        logging.info(f"Failed to download the image. Status code: {response.status_code}")

def save_image_locally(url, filename):
    filepath = Path('/Users/scottbelisle/Documents/code/clapvontrap/downloads/') / filename
    download_image(url, str(filepath))

def vision(url, query):

    filename = "screenshot.png"  # You can change this to any desired filename
    save_image_locally(url, filename)

    # base64 encode the image
    with open('/Users/scottbelisle/Documents/code/clapvontrap/downloads/screenshot.png', 'rb') as image_file:
        base64string = base64.b64encode(image_file.read())

    # convert base64string to string
    base64string = base64string.decode('utf-8')
    url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "llava:13b-v1.5-fp16",
        "prompt": query,
        "images": [base64string]
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    str_response = response.content.decode('utf-8')
    list_response = str_response.split('\n')
    message = ""
    for i in list_response[:-1]:
        i = json.loads(i)
        message = message + i['response']
    return message

if __name__ == "__main__":
    foo = vision("https://files.slack.com/files-tmb/T2CUGU41F-F06D3R45L6M-e2041accd5/screenshot_2024-01-04_at_11.20.05___am_480.png", "what is this?")
    print(foo)

