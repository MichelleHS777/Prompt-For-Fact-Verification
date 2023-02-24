import argparse
import json

parser = argparse.ArgumentParser(description='Count the avg of the golden evidence')
parser.add_argument('--train_dataset_path', type=str, default='./datasets/unpreprocess/train.json', help='train dataset path')
parser.add_argument('--valid_dataset_path', type=str, default='./datasets/unpreprocess/dev.json', help='validation dataset path')
parser.add_argument('--test_dataset_path', type=str, default='./datasets/unpreprocess/test.json', help='test dataset path')
args = parser.parse_args()

train_dataset = json.load(open(args.train_dataset_path, 'r', encoding='utf-8'))
valid_dataset = json.load(open(args.valid_dataset_path, 'r', encoding='utf-8'))
test_dataset = json.load(open(args.test_dataset_path, 'r', encoding='utf-8'))

avg_train_goldenEvidence = []
for row in range(len(train_dataset)):
    gold_evidences = [train_dataset[row]['gold evidence'][str(i)]['text'] for i in range(5) if train_dataset[row]['gold evidence'][str(i)]['text'] != '']
    # if len(gold_evidences) != 0:
    avg_train_goldenEvidence.append(len(gold_evidences))
print(sum(avg_train_goldenEvidence)/len(avg_train_goldenEvidence))

avg_valid_goldenEvidence = []
for row in range(len(valid_dataset)):
    gold_evidences = [valid_dataset[row]['gold evidence'][str(i)]['text'] for i in range(5) if valid_dataset[row]['gold evidence'][str(i)]['text'] != '']
    # if len(gold_evidences) != 0:
    avg_valid_goldenEvidence.append(len(gold_evidences))
print(sum(avg_valid_goldenEvidence)/len(avg_valid_goldenEvidence))

avg_test_goldenEvidence = []
for row in range(len(test_dataset)):
    gold_evidences = [test_dataset[row]['gold evidence'][str(i)]['text'] for i in range(5) if test_dataset[row]['gold evidence'][str(i)]['text'] != '']
    # if len(gold_evidences)!=0:
    avg_test_goldenEvidence.append(len(gold_evidences))
print(sum(avg_test_goldenEvidence)/len(avg_test_goldenEvidence))