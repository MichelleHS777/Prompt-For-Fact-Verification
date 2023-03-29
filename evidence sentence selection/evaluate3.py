import torch
import re
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pred_file = open('data/evidence/prompt_test_threshold.json', 'r', encoding='utf-8')
pred_file = list(pred_file)
gold_file = open('data/evidence/gold_test.json', 'r', encoding='utf-8')
gold_file = list(gold_file)


def get_dict_by_id(list, id):
    """
    Get a list of dictionaries with the same ID from list.
    :param list: a list of dictionaries
    :param id: the ID to search for
    :return: a list of dictionaries with the same ID
    """
    result = []
    for d in list:
        d = eval(str(d))
        if d.get('claimId') == int(id):
            result.append(d)
    return result


def get_sameId_evidences(file, id):
        same_id_dataset = get_dict_by_id(file, int(id))
        for data in same_id_dataset:
            evidences = data['evidence']
            return evidences


precision = []
recall = []
f1 = []
for data in tqdm(gold_file, desc='Evaluating'):
    id = eval(data)['claimId']
    pred_dataset = get_dict_by_id(pred_file, id)
    gold_dataset = get_dict_by_id(gold_file, id)
    pred_evidence = get_sameId_evidences(pred_dataset, id)
    gold_evidence = get_sameId_evidences(gold_dataset, id)

    tp = 0
    for pred in pred_evidence:
        if pred in gold_evidence:
            tp += 1
    if len(pred_evidence) == 0:
        each_precision = 0
    else:
        each_precision = tp / len(pred_evidence)
    if len(gold_evidence) == 0:
        each_recall = 0
    else:
        each_recall = tp / len(gold_evidence)
    if each_precision + each_recall != 0:
        each_f1 = 2 * each_precision * each_recall / (each_precision + each_recall)
        precision.append(each_precision)
        recall.append(each_recall)
        f1.append(each_f1)
    else:
        continue

precision = sum(precision) / len(precision)
recall = sum(recall) / len(recall)
f1 = sum(f1) / len(f1)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"f1: {f1:.2f}")

