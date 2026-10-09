import pickle
import json
import os

Drive_LM_test_json = 'evaluation_DriveLM/DriveLM_dataset/cache/Drivelm_Qwen_test_15480.json'
with open(Drive_LM_test_json, 'r') as js_data:
    test_data = json.load(js_data)

sample_token_to_id = {}
for item in test_data:
    sample_token_to_id[item['sample_token']] = item['id']

############   只需要改这里的预测resluts_all.json  ##################
json_path = './results.json'

with open(json_path, 'rb') as js:
    json_data = json.load(js)

for pred in json_data['predictions']:
    sample_token = pred['sample_token']
    if sample_token in sample_token_to_id:
        pred['id'] = sample_token_to_id[sample_token]
    else:
        print(f"Warning: sample_token {sample_token} not found in mapping")

# result_json_path = pkl_path[:-4]+'.json'
# with open(result_json_path, 'w', encoding='utf-8') as f:
#     json.dump(pkl_data, f, ensure_ascii=False, indent=4)

id_to_answer_pre = {}
for item1 in json_data.get('predictions', []):
    id_to_answer_pre[item1['id']] = item1['answer_pre']


template_json = 'evaluation_DriveLM/DriveLM_template.json'
with open(template_json, 'r', encoding='utf-8') as f:
    results_data = json.load(f)

for result in results_data.get('results', []):
    _id = result.get('id')
    if _id in id_to_answer_pre:
        result['answer'] = id_to_answer_pre[_id]

output_json = os.path.dirname(json_path) + '/results_submission.json'
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=4)

print("替换完成，保存为 results_submission.json")

