import sys
sys.path.append("nuscenes_anno")
from utils.generate_prompt import DrivePromptEngine
from utils.online_mllm_generate import OnlineMLLMGenerate
from utils.online_mllm_generate_with_batch import OpenAIBatchProcessor
from utils.online_mllm_generate_asyncio import OnlineMLLMGenerateAsyncio
import pickle
import argparse
import asyncio

class GenerateAuxiliaryTasks:
    def __init__(self, data_path, batch_size, output_file, batch_prompts_file_path):
        self.data_path = data_path
        self.batch_size = batch_size
        self.output_file = output_file
        self.batch_prompts_file_path = batch_prompts_file_path
        self.generator = DrivePromptEngine()
        self.online_mllm_generate = OnlineMLLMGenerate()
        self.openai_batch_processor = OpenAIBatchProcessor()
        self.online_mllm_generate_asyncio = OnlineMLLMGenerateAsyncio()

    def generate_tasks_single(self):
        data = pickle.load(open(self.data_path, "rb"))
        data = data["infos"]

        prompts = self.generator.generate(data)
    
        self.online_mllm_generate.generate(prompts, self.batch_size, self.output_file)

    def generate_tasks_with_batch(self):
        data = pickle.load(open(self.data_path, "rb"))
        data = data["infos"]

        self.generator.generate_with_batch(data, self.batch_prompts_file_path)
    
        results = self.openai_batch_processor.process_batch(
            input_file_path=self.batch_prompts_file_path,
            endpoint="v1/chat/completions",
            completion_window="24h",
            result_file_name=self.output_file,
        )

    async def generate_tasks_asyncio(self):
        data = pickle.load(open(self.data_path, "rb"))
        data = data["infos"]

        # self.generator.generate_with_batch(data, self.batch_prompts_file_path)
    
        results = await self.online_mllm_generate_asyncio.generate_asyncio(
            input_file=self.batch_prompts_file_path,
            output_file=self.output_file,
            batch_size=self.batch_size,
        )

async def main():
    argparser = argparse.ArgumentParser(
        description="Generate auxiliary tasks for NuScenes planning dataset."
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing entries.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="auxiliary_tasks.jsonl",
        help="Output file path for processed entries.",
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default="nuscenes_anno/utils/nusc_converter/nusc_data/mini/nusc_infos_val.pkl",
        help="Input file path for data.",
    )
    argparser.add_argument(
        "--batch_prompts_file_path",
        type=str,
        default="nuscenes_anno/utils/nusc_converter/nusc_data/mini/batch_prompts.jsonl",
        help="file path for batch prompts.",
    )
    args = argparser.parse_args()
    generator = GenerateAuxiliaryTasks(
        data_path=args.data_path,
        batch_size=args.batch_size,
        output_file=args.output_file,
        batch_prompts_file_path=args.batch_prompts_file_path,
    )

    # generator.generate_tasks_single()
    await generator.generate_tasks_asyncio()
    print(f"Generated auxiliary tasks and saved to {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main())
