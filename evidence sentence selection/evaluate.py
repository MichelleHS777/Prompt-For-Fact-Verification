import numpy as np
from sklearn.metrics import precision_recall_fscore_support

pred_file = open('result/0311/semantic_result.json', 'r', encoding='utf-8')
gold_file = open('data/CHEF/test_evidence.json', 'r', encoding='utf-8')
y_pred = []
y_true = []

for pred in pred_file:
    pred = eval(pred)
    y_pred.append(pred['label'])
for gold in gold_file:
    gold = eval(gold)
    y_true.append(gold['label'])

pre, recall, f1,_= precision_recall_fscore_support(y_pred, y_true, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
pre, recall, f1,_= precision_recall_fscore_support(y_pred, y_true, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))

