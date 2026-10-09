
## BLEU
import os
from coco_caption.pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
import json
import argparse
import pickle

def coco_caption_eval(coco_annotations , coco_results):
    coco = COCO()
    coco.dataset = coco_annotations
    coco.createIndex()
    coco_result = coco.loadRes(coco_results)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    
    results = {}
    for metric, score in coco_eval.eval.items():
        results[metric] = score
    return results


if __name__ == "__main__":
    ###########
    pkl_path = 'outputs_NuInstruct_ablation/NuInstruct_test/11_04_07_11_03_vggt_replace/reasoningtask/results.pkl'
    ##########
    NuInstruct_task_json = 'evaluation_nuinstruct/cache/NuInstruct_test_reasoningtask.json'
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


    def lists_to_coco_format(outputs):
        annotations = []
        results = []
        for i, output in enumerate(outputs):
            annotations.append({
                "image_id": i,
                "id": i,
                "caption": output['answer_gt']
            })
            results.append({
                "image_id": i,
                "caption": output['answer_pre']
            })
        
        coco_annotations = {
            "images": [{"id": i} for i in range(len(outputs))],
            "annotations": annotations
        }
        
        return coco_annotations, results


    # 转换为COCO格式
    coco_annotations, coco_results = lists_to_coco_format(outputs["predictions"])
    metrics = coco_caption_eval(coco_annotations, coco_results)
    
    base = os.path.basename(args.model_output_file)               
    stem, _ = os.path.splitext(base)                           
    prefix = '_'.join(stem.split('_')[:2])                        
    results_file_name = os.path.join(os.path.dirname(args.model_output_file), "metric.json")

    # save accuracy_result to results_file_name
    ## TODO：need to modify the metrics format
    with open(results_file_name, 'w') as f:
        json.dump(metrics, f, indent=4, sort_keys=True)
