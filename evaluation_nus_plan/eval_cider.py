import pickle


def aggregate_omni_result(results):
    from coco_caption.pycocoevalcap.bleu.bleu import Bleu
    from coco_caption.pycocoevalcap.rouge.rouge import Rouge
    from coco_caption.pycocoevalcap.cider.cider import Cider
    refs = {i: [res["answer_gt"]] for i, res in enumerate(results)}
    hyps = {i: [res["answer_pre"]] for i, res in enumerate(results)}
    total_scores = {}
    # BLEU
    bleu = Bleu(4)
    bleu_score, _ = bleu.compute_score(refs, hyps)
    for i in range(4):
        total_scores[f"Bleu_{i+1}"] = bleu_score[i]
    total_scores["Bleu"] = sum(bleu_score) / 4
    # ROUGE_L
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(refs, hyps)
    total_scores["ROUGE_L"] = rouge_score
    # CIDEr
    cider = Cider()
    cider_score, _ = cider.compute_score(refs, hyps)
    total_scores["CIDEr"] = cider_score
    return total_scores


pkl_path = 'outputs_Omnidrive/omni_conv_desc_test/10_25_08_18_17_full_tuning/results.pkl'

with open (pkl_path, 'rb') as pkl:
    pre_gt_valdata = pickle.load(pkl)

print(len(pre_gt_valdata['predictions']))

omni_scores = aggregate_omni_result(pre_gt_valdata['predictions'])
print(omni_scores)
