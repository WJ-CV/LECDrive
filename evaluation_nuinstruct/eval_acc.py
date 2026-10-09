import json
from collections import defaultdict
import re
import os
import argparse
import pickle
###########
pkl_path = 'outputs_NuInstruct_ablation/NuInstruct_test/11_04_07_11_03_vggt_replace/accuracytask/results.pkl'
##########
NuInstruct_task_json = 'evaluation_nuinstruct/cache/NuInstruct_test_accuracytask.json'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
json_path = pkl_path[:-4] + '.json'

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


with open(NuInstruct_task_json, 'r', encoding='utf-8') as f:
    source_data = json.load(f)

with open(json_path, 'r', encoding='utf-8') as f:
    result_data = json.load(f)

answer_to_task = {}

for case in source_data:
    gpt_answer = None
    for conv in case.get('conversations', []):
        if conv['from'] == 'gpt':
            gpt_answer = conv['value']
            break
    if gpt_answer:
        answer_to_task[gpt_answer.strip()] = case.get('task')

for item in result_data['predictions']:
    gt = item.get('answer_gt', '').strip()

    matched_task = answer_to_task.get(gt, None)

    if matched_task is None:
        raise ValueError(f"No matching task found for answer_gt: {gt}")

    item['task'] = matched_task

json_task_path = pkl_path[:-4] + '_task.json'
with open(json_task_path, 'w', encoding='utf-8') as f:
    json.dump(result_data, f, ensure_ascii=False, indent=4)
print("补充task完成，保存为 result_task.json")

parser = argparse.ArgumentParser()
json_path = json_task_path

parser.add_argument('--model_output_file', default=json_path, type=str, help='model inference output file path')
args = parser.parse_args()
if args.model_output_file.endswith('.json'):
    outputs = json.load(open(args.model_output_file,'r'))
elif args.model_output_file.endswith('.jsonl'): 
    outputs = [json.loads(line) for line in open(args.model_output_file,'r')]

# gts = [json.loads(line) for line in open(gts_file,'r')]
# assert len(preds) == len(gts)

# 初始化字典来存储每个key的正确和总数
accuracy_dict = defaultdict(lambda: {"correct": 0, "total": 0})

# 遍历outputs
for output in outputs["predictions"]:
    task = output['task']
    accuracy_dict[task]["total"] += 1
    pattern = r"<(.*?)>"
    predstr = re.search(pattern, output['answer_pre']).group(1) if re.search(pattern, output['answer_pre']) else output['answer_pre']
    gtstr = re.search(pattern, output['answer_gt']).group(1) if re.search(pattern, output['answer_gt']) else output['answer_gt']
    if predstr == gtstr:
        accuracy_dict[task]["correct"] += 1


# accuracy_result = {task: value["correct"] / value["total"] for task, value in accuracy_dict.items()}

accuracies = []

for task, data in accuracy_dict.items():
    correct = data["correct"]
    total = data["total"]
    accuracy = correct / total if total != 0 else 0.0
    accuracy_dict[task]["accuracy"] = accuracy
    accuracies.append(accuracy)

mean_accuracy = sum(accuracies) / len(accuracies)
accuracy_dict["mean_accuracy"] = mean_accuracy



base = os.path.basename(args.model_output_file)               
stem, _ = os.path.splitext(base)                           
prefix = '_'.join(stem.split('_')[:2])                        
results_file_name = os.path.join(os.path.dirname(args.model_output_file), "metric.json")

# save accuracy_result to results_file_name
## TODO：need to modify the acc_dict format
with open(results_file_name, 'w') as f:
    json.dump(accuracy_dict, f, indent=4, sort_keys=True)