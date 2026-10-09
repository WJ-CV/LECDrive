import json
import os
import time

import openai

class OpenAIBatchProcessor:
    def __init__(self):
        self.openai_host = "http://10.96.195.34:60101/v1"
        self.api_key = "token-evad-infra-abc123"
        self.client = openai.Client(base_url=self.openai_host, api_key=self.api_key)

    def process_batch(self, input_file_path, endpoint, completion_window, result_file_name):
        with open(input_file_path, "rb") as file:
            uploaded_file = self.client.files.create(file=file, purpose="batch")
            
        batch_job = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint=endpoint,
            completion_window=completion_window
        )

        while batch_job.status not in ["completed", "failed", "cancelled"]:
            time.sleep(3)
            print(f"Batch job status: {batch_job.status}")
            batch_job = self.client.batches.retrieve(batch_job.id)

        if batch_job.status == "failed":
            print("Batch job failed.")
            return None

        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            file_response = self.client.files.content(result_file_id)
            result_content = file_response.read()
                
            with open(result_file_name, "wb") as result_file:
                result_file.write(result_content)
            
            results = []
            with open(result_file_name, "r", encoding="utf-8") as result_file:
                for line in result_file:
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            
            return results

        else:
            print(f"batch job status: {batch_job.status}")
            return None
