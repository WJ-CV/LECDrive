      
import jsonlines
import openai
import asyncio
import base64
import os
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm  # 导入 tqdm 库
from PIL import Image
import io
import re
class OnlineMLLMGenerateAsyncio:
    def __init__(self):
        self.openai_host = "http://10.96.208.216:8000/v1"
        self.api_key = "e2e-vlm-token"
        self.aclient = AsyncOpenAI(
            api_key=self.api_key, base_url=self.openai_host
        )
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

    async def process_entry(self, entry, retry_count=0, max_retries=3):
        """Processes a single entry from the JSONL file, saving only the total score, with retry mechanism."""
        image_paths = entry.get("image_paths", [])
        image_urls = self.images_to_image_urls(image_paths)

        user_prompt = entry.get("user_prompt", "")

        try:
            completion = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            *[
                                {"type": "image_url", "url": image_url}
                                for image_url in image_urls
                            ],
                        ],
                    }
                ],
            )
            model_response = completion.choices[0].message.content

            return model_response

        except openai.APIError as e:
            print(f"APIError {e}")
            return {**entry, 'error': f"APIError: {e}"}
        except Exception as e:
            print(f"Error processing ： {e}")
            return {**entry, 'error': f"Processing Error: {e}"}

    async def process_batch(self, batch):
        """Processes a batch of entries concurrently."""
        return await asyncio.gather(*[self.process_entry(entry) for entry in batch])

    async def generate_asyncio(self, input_file, output_file, batch_size):
        batch = []
        updated_entries = [] #  不再需要一次性存储所有 updated_entries

        try:
            with jsonlines.open(input_file, 'r') as reader:
                total_entries = sum(1 for _ in reader)  # 计算总条目数，用于进度条
            with jsonlines.open(input_file, 'r') as reader:  # 重新打开 reader
                with jsonlines.open(output_file, 'a') as writer: # 以 'append' 模式打开输出文件
                    with tqdm(total=total_entries, desc="Processing entries") as pbar:  # 初始化 tqdm 进度条
                        for entry in reader:
                            batch.append(entry)
                            if len(batch) == batch_size:
                                processed_batch = await self.process_batch(batch)
                                writer.write_all(processed_batch) #  处理完一批次后立即写入
                                batch = []
                                pbar.update(batch_size)  # 更新进度条

                    if batch:  # 处理剩余的条目
                        processed_batch = await self.process_batch(batch)
                        writer.write_all(processed_batch) # 处理剩余批次后立即写入
                        pbar.update(len(batch))  # 更新进度条

            print(f"Processed entries in batches of {batch_size}. Output written to {output_file} after each batch.") # 修改提示信息

        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    async def generate(self, input_file, output_file, batch_size):
        asyncio.run(self.generate_asyncio(input_file, output_file, batch_size))

    