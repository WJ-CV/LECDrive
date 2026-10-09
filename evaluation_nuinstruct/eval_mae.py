import json
from collections import defaultdict
import re
import os
import argparse
import pickle
###########
pkl_path = 'outputs_NuInstruct_ablation/NuInstruct_test/11_04_07_11_03_vggt_replace/maetask/results.pkl'
##########
NuInstruct_task_json = 'evaluation_nuinstruct/cache/NuInstruct_test_maetask.json'
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
parser.add_argument('--model_output_file', default=json_task_path, type=str, help='model inference output file path')
args = parser.parse_args()
if args.model_output_file.endswith('.json'):
    outputs = json.load(open(args.model_output_file,'r'))
elif args.model_output_file.endswith('.jsonl'): 
    outputs = [json.loads(line) for line in open(args.model_output_file,'r')]

def find_number(answer):
    pattern = r'\(([^,]+),([^)]+)\)'
    matches = re.findall(pattern, answer)[0]
    
    return matches


def perception(gt, predict, task):
    # if task == 'closest' or task == 'status' or task == 'status_ego' or task == 'status_others':
    #     if gt == predict:
    #         return 1
    #     else:
    #         return 0
    # if task == 'distance' or task == 'speed' or task == 'instance_count':
    #     try:
    #         mse = abs(float(gt) - float(predict))
    #         return mse
    #     except:
    #         return abs(float(gt))
    # elif task == 'motion_ego' or task == 'motion_other':
    #     x1,y1 = find_number(predict)
    #     x2,y2 = find_number(gt)
    #     mae = (abs(float(x1) - float(x2)) + abs(float(y1) - float(y2))) // 2
    #     return mae
    
    if task == 'speed' or task == 'instance_count':
        try:
            mse = abs(float(gt) - float(predict))
            return mse
        except:
            return abs(float(gt))
    elif task == 'motion_ego' or task == 'motion_other' or task == 'distance':
        try:
            x1,y1 = find_number(predict)
            x2,y2 = find_number(gt)
            mae = (abs(float(x1) - float(x2)) + abs(float(y1) - float(y2))) // 2
        except:
            x2,y2 = find_number(gt)
            return abs(float(x2))+abs(float(y2))
        return mae
    else:
        assert False, f'{task} not found'
         
    

mse_dict = defaultdict(lambda: [])
for output in outputs["predictions"]:
    task = output['task'].split('-')[1]
    mse_dict[task].append(perception(output['answer_gt'], output['answer_pre'], task))
mse_result = {task: sum(value) / len(value) for task, value in mse_dict.items()}

values = list(mse_result.values())
total_sum = sum(values)
mean_value = total_sum / len(values)

mse_result['_total_sum'] = total_sum
mse_result['_mean'] = mean_value

base = os.path.basename(args.model_output_file)               
stem, _ = os.path.splitext(base)                           
prefix = '_'.join(stem.split('_')[:2])                        
results_file_name = os.path.join(os.path.dirname(args.model_output_file), "metric.json")

# save accuracy_result to results_file_name
## TODO：need to modify the acc_dict format
with open(results_file_name, 'w') as f:
    json.dump(mse_result, f, indent=4, sort_keys=True)