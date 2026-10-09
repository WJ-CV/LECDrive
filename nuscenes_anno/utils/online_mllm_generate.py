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

class OnlineMLLMGenerate:
    def __init__(self):
        self.openai_host = "http://10.96.208.216:8000/v1"
        self.api_key = "e2e-vlm-token"
        self.client = OpenAI(api_key=self.api_key, base_url=self.openai_host)
        self.model_name = self.client.models.list().data[0].id
        self.models = self.client.models.list()
        self.model_id = self.models.data[0].id  # Or specify your desired model_id directly, e.g., "gpt-4-vision-preview"


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

    def process_entry(self, entry):
        """Processes a single entry from the JSONL file, saving only the total score, with retry mechanism."""
        image_paths = entry.get("images", [])
        image_urls = self.images_to_image_urls(image_paths)

        user_prompt = entry.get("user_prompt", "")
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
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
            )
            model_response = completion.choices[0].message.content
            print(f"Model response: {model_response}")

        except openai.APIError as e:
            print(f"APIError for id {entry['id']}: {e}")
            return {**entry, "error": f"APIError: {e}"}

    def process_batch(self, batch):
        return [self.process_entry(entry) for entry in batch]

    def generate_with_batch(self, dataset, batch_size, output_file):
        batch = []
        total_entries = len(dataset)
        try:
            with jsonlines.open(
                output_file, "a"
            ) as writer:  # Open output file in append mode
                with tqdm(
                    total=total_entries, desc="Processing entries"
                ) as pbar:  # Initialize progress bar
                    for idx in range(total_entries):
                        entry = dataset[idx]
                        batch.append(entry)

                        if len(batch) == batch_size:
                            processed_batch =  self.process_batch(batch)
                            writer.write_all(
                                processed_batch
                            )  # Write processed batch to output file
                            batch = []
                            pbar.update(batch_size)  # Update progress bar

                    if batch:  # Process remaining entries
                        processed_batch =  self.process_batch(batch)
                        writer.write_all(
                            processed_batch
                        )  # Write remaining batch to output file
                        pbar.update(len(batch))  # Update progress bar

            print(
                f"Processed entries in batches of {batch_size}. Output written to {output_file} after each batch."
            )

        except Exception as e:
            print(f"An error occurred: {e}")

    def generate(self, dataset, batch_size=1, output_file="auxiliary_task_results.jsonl"):
        self.generate_with_batch(dataset, batch_size, output_file)