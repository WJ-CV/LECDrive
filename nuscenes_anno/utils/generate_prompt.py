# drive_prompt.py
import jsonlines
import openai
import base64
import os
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm  # 导入 tqdm 库
from PIL import Image
import io
import re
import time
import sys
import argparse
import pickle
import json
import numpy as np

class DrivePromptEngine:
    def __init__(self):
        self.template = """
        
        ### 4. Current State
        - Lidar: {lidar}
        ```
        """
    def encode_image_to_base64(self, image_path):
        """Encodes an image from a file path to base64."""
        try:
            with open(image_path, "rb") as image_file:
                img = Image.open(image_path)
                img_resized = img.resize((960, 540))  # Resize to 960x540

                # Save the resized image to a buffer
                buffer = io.BytesIO()
                img_resized.save(buffer, format="JPEG")
                buffer.seek(0)

                # Encode the resized image to base64
                encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
                return encoded_image
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None

    def images_to_image_urls(self, images):
        """Encodes a list of images from file paths to base64."""
        image_urls = []
        for image_path in images:
            encoded_image = self.encode_image_to_base64(image_path)
            if encoded_image is None:
                print(f"Skipping image {image_path} due to encoding failure.")
                continue
            image_url = {"url": f"data:image/jpeg;base64,{encoded_image}"}
            image_urls.append(image_url)
        return image_urls

    def generate(self, dataset):
        prompts = []
        for idx in range(len(dataset)):
            entry = dataset[idx]
            image_paths = entry["images"]
            lidar = entry["lidar_path"]
            replacements = {
                "lidar": lidar,
            }
            user_prompt = self.template.format(**replacements)
            prompt = {
                "images": image_paths,
                "user_prompt": user_prompt,
            }
            prompts.append(prompt)
        return prompts

    def generate_with_batch(self, dataset, batch_prompts_file_path):
        # format example:
        # {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!  List 3 NBA players and tell a story"}],"max_tokens": 300}}
        for idx in range(len(dataset)):
        # for idx in range(2):
            user_prompt = "what happened in the images?"
            image_paths = dataset[idx]["images"]
            image_urls = self.images_to_image_urls(image_paths)
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        *[
                            {"type": "image_url", "image_url": image_url}
                            for image_url in image_urls
                        ],
                    ],
                }
            ],
            body = {
                "model": "Qwen2.5-VL-72B-Instruct",
                "messages": messages,
                "max_tokens": 300
            }
            custom_id = f"request-{idx + 1}"
            method = "POST"
            url = "/v1/chat/completions"
            entry = {
                "custom_id": custom_id,
                "method": method,
                "url": url,
                "body": body
            }
            if not os.path.exists(batch_prompts_file_path):
                os.makedirs(os.path.dirname(batch_prompts_file_path), exist_ok=True)
                with open(batch_prompts_file_path, "a") as file:
                    file.write(json.dumps(entry) + "\n")
                
        print(f"Batch prompts saved to {batch_prompts_file_path}")