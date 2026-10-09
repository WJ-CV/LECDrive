## MAP
import json
import re
import numpy as np
import os
import argparse
import pickle

cam_id_to_key = {
    'CAM_FRONT_LEFT': 1,
    'CAM_FRONT': 0,
    'CAM_FRONT_RIGHT': 2,
    'CAM_BACK_LEFT': 4,
    'CAM_BACK': 3,
    'CAM_BACK_RIGHT': 5
}

def parse_detections(input_str, class_name_to_id, is_gt=False):
    input_str = input_str.strip().rstrip(';')
    
    detections = input_str.split(';')
    result = []
    for det in detections:
        det = det.strip()
        if not det:
            continue
        
        class_match = re.search(r'<([^>]+)>',det)
        if class_match:
            class_name = class_match.group(1)
            class_id = class_name_to_id.get(class_name.lower(), -1)
            if class_id == -1:
                continue
        else:
            continue
        
        bbox_match = re.search(r'\[([^\]]+)\]',det)
        if bbox_match:
            bbox_info = bbox_match.group(1)
            parts = bbox_info.split(',')
            if len(parts) == 5:
                try:
                    if len(parts[0].strip()) == 2:
                        camera_idx = parts[0].strip()[-1]
                    else:
                        camera_idx = cam_id_to_key[parts[0].strip()]
                except:
                    print("#" * 20 + f"{parts[0].strip()} not in cam_id_to_key!" + "#" * 20)
                    camera_idx = None

                try:
                    x1, y1, x2, y2 = map(float, parts[1:])
                except:
                    # assert False
                    continue
                detection = {
                    'class_id': class_id,
                    'camera_idx': camera_idx,
                    # 'confidence': confidence,
                    'confidence': 1.0,
                    'bbox': [x1, y1, x2, y2],
                    'matched': False
                }
        result.append(detection)
    return result


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    area1 = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0
    else:
        return inter_area / union_area

def compute_ap(recalls,  precisions):
    max_precisions = []
    for recall_threshold in [i/10 for i in range(0,11)]:
        precisions_at_recall = [p for r, p in zip(recalls, precisions) if r >= recall_threshold]
        if precisions_at_recall:
            max_precisions.append(max(precisions_at_recall))
        else:
            max_precisions.append(0)
    ap = sum(max_precisions) / 11
    return ap

def compute_map(predictions, ground_truths, class_name_to_id, iou_threshold=0.5):
    aps = []
    num_classes = len(class_name_to_id)
    for class_name, class_id in class_name_to_id.items():
        preds = [p for p in predictions if p['class_id'] == class_id]
        gts = [g for g in ground_truths if g['class_id'] == class_id]
        
        npos = len(gts)
        if npos == 0 and len(preds) == 0:
            continue
        elif npos == 0:
            aps.append(0)
            continue
        
        preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        for idx, pred in enumerate(preds):
            max_iou = 0
            max_gt_idx = -1
            for gt_idx, gt in enumerate(gts):
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            if max_iou >= iou_threshold:
                if not gts[max_gt_idx]['matched']:
                    tp[idx] = 1
                    gts[max_gt_idx]['matched'] = True
                else:
                    fp[idx] = 1
            else:
                fp[idx] = 1
                
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recalls = tp / npos
        precisions = tp / (tp + fp)
        ap = compute_ap(recalls, precisions)
        aps.append(ap) 
    if aps:
        mAP = sum(aps)/len(aps)#sum(aps) / num_classes
    else:
        mAP = 0.0
    return mAP   

class_name_to_id = {
    'animal': 1,
    'barrier': 2,
    'bicycle': 3,
    'bicycle_rack': 4,
    'bus': 5,
    'car': 6,
    'category_name': 7,
    'construction': 8,
    'debris': 9,
    'emergency': 10,
    'motorcycle': 11,
    'pedestrian': 12,
    'pushable_pullable': 13,
    'trafficcone': 14,
    'trailer': 15,
    'truck': 16
}

def scale_coordinates(answer_pre):
    # 正则表达式，用于匹配格式 <class>[camera,x1,y1,x2,y2];
    pattern = r'<(\w+)>\[([\w_]+),([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)\];'
    
    # 尝试匹配answer_pre
    match = re.match(pattern, answer_pre)
    if match:
        # 提取类名、相机类型和四个坐标值
        class_name = match.group(1)
        camera_view = match.group(2)
        x1, y1, x2, y2 = map(float, [match.group(3), match.group(4), match.group(5), match.group(6)])

        scale_x = 16
        scale_y = 9
        scaled_x1 = round(x1 * scale_x, 3)
        scaled_y1 = round(y1 * scale_y, 3)
        scaled_x2 = round(x2 * scale_x, 3)
        scaled_y2 = round(y2 * scale_y, 3)

        return f"<{class_name}>[{camera_view},{scaled_x1},{scaled_y1},{scaled_x2},{scaled_y2}];"
    else:
        return answer_pre

if __name__ == "__main__":
    ########### 
    pkl_path = 'output_NuInstruct_sota/NuInstruct_test_/10_16_12_07_35_full_tuning_ep5/maptask/results.pkl'
    ###########
    NuInstruct_task_json = 'evaluation_nuinstruct/cache/NuInstruct_test_maptask.json'
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
    parser.add_argument('--model_output_file',default=json_task_path, type=str, help='model inference output file path')
    args = parser.parse_args()

    if args.model_output_file.endswith('.json'):
        outputs = json.load(open(args.model_output_file,'r'))
    elif args.model_output_file.endswith('.jsonl'): 
        outputs = [json.loads(line) for line in open(args.model_output_file,'r')]

    pred_answers = []
    gt_answers = []
    tasks = []
    for output in outputs["predictions"]:
        answer_pre_ = scale_coordinates(output['answer_pre'])
        pred_answers.append(answer_pre_)
        gt_answers.append(output['answer_gt'])
        tasks.append(output['task'])

    # 初始化空列表来存储所有样本的预测和真实值
    all_predictions = []
    all_ground_truths = []

    # 解析每个样本的预测和真实值，并添加到总列表中
    for predictions_str, ground_truths_str in zip(pred_answers, gt_answers):
        predictions = parse_detections(predictions_str, class_name_to_id, is_gt=False) ## TODO: 预测坐标的格式是否一致，eg <car>[CAM_BACK_RIGHT,563,405,1600,900]
        ground_truths = parse_detections(ground_truths_str, class_name_to_id, is_gt=True)
        all_predictions.append(predictions)
        all_ground_truths.append(ground_truths)

    # 初始化字典来存储每个任务的预测和真实值
    task_predictions = {task: [] for task in set(tasks)}
    task_ground_truths = {task: [] for task in set(tasks)}

    # 将预测和真实值按任务分类
    for pred, gt, task in zip(all_predictions, all_ground_truths, tasks):
        # if gt == []:
        #    continue
        task_predictions[task].append(pred)
        task_ground_truths[task].append(gt)
                        
    # 计算每个任务的mAP
    task_mAPs = {}
    iou_thresholds = [x / 100.0 for x in range(50, 100, 5)]  # 0.5, 0.55, ..., 0.95

    for task in task_predictions.keys():
        if task == 'risk-non':
            continue
        task_mAPs[task] = {iou_threshold: [] for iou_threshold in set(iou_thresholds)}
        for iou_threshold in iou_thresholds: ## 不同阈值
            for task_prediction, task_ground_truth in zip(task_predictions[task], task_ground_truths[task]):
                if len(task_prediction) == 0 and len(task_ground_truth) == 0:
                    continue
                # mAP += compute_map(task_prediction, task_ground_truth, class_name_to_id, iou_threshold=0.5)
                # task_mAPs[task] = mAP / len(task_predictions[task])
                for i in range(len(task_prediction)):
                    task_prediction[i]['matched'] = False
                for i in range(len(task_ground_truth)):
                    task_ground_truth[i]['matched'] = False
                
                task_mAPs[task][iou_threshold].append(compute_map(task_prediction, task_ground_truth, class_name_to_id, iou_threshold=iou_threshold))
            task_mAPs[task][iou_threshold] = sum(task_mAPs[task][iou_threshold]) / len(task_mAPs[task][iou_threshold])

    # 初始化字典来存储每个task的平均值
    task_averages = {}

    # 遍历task_mAPs字典
    for task, mAPs in task_mAPs.items():
        total = sum(mAPs.values())
        count = len(mAPs)
        average = total / count
        task_averages[task] = average
    #     print(f"Task {task} AP: {mAPs}")

    # print("Average AP per task:", task_averages)

    base = os.path.basename(args.model_output_file)               
    stem, _ = os.path.splitext(base)                           
    prefix = '_'.join(stem.split('_')[:2])                        
    results_file_name = os.path.join(os.path.dirname(args.model_output_file), "metric.json")

    excluded_tasks = {"risk-braking", "risk-overtaking"}
    filtered_values = [v for k, v in task_averages.items() if k not in excluded_tasks]
    average_value = sum(filtered_values) / len(filtered_values)
    task_averages["average"] = round(average_value, 5) 
    
    # save accuracy_result to results_file_name
    ## TODO：need to modify the acc_dict format
    with open(results_file_name, 'w') as f:
        json.dump(task_averages, f, indent=4, sort_keys=True)
