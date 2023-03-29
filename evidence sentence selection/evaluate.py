import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict
from tqdm import tqdm

def get_dict_by_id(list, id):
    """
    Get a list of dictionaries with the same ID from list.
    :param list: a list of dictionaries
    :param id: the ID to search for
    :return: a list of dictionaries with the same ID
    """
    result = []
    for d in list:
        d = eval(d)
        if d.get('claimId') == id:
            result.append(d)
    return result


pred_file = open('result/0311/PromptBERT_result.json', 'r', encoding='utf-8')
gold_file = open('data/CHEF/test_evidence.json', 'r', encoding='utf-8')
gold_file = list(gold_file)
pred_file = list(pred_file)

claimId = [eval(data)['claimId'] for data in gold_file]
claimId = list(OrderedDict.fromkeys(claimId))

precision = []
recall = []
micro_f1 = []

for id in tqdm(claimId, desc='evaluating...'):
    pred_dataset = get_dict_by_id(pred_file, id)
    y_pred = [data['label'] for data in pred_dataset]
    gold_dataset = get_dict_by_id(gold_file, id)
    y_true = [data['label'] for data in gold_dataset]
    pre, re, f1, _ = precision_recall_fscore_support(y_pred, y_true, average='micro')
    precision.append(pre)
    recall.append(re)
    micro_f1.append(f1)

avg_precision = sum(precision) / len(precision)
avg_recall = sum(recall) / len(recall)
avg_f1 = sum(micro_f1) / len(micro_f1)
print("Precision (micro): {:.2%}".format(avg_precision))
print("   Recall (micro): {:.2%}".format(avg_recall))
print("       F1 (micro): {:.2%}".format(avg_f1))
