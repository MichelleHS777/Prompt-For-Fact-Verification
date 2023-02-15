import argparse
parser = argparse.ArgumentParser(description='Count Dataset')
parser.add_argument('--train_dataset_path', type=str, default='datasets/train.json', help='train dataset path')
# parser.add_argument('--dev_dataset_path', type=str, default='datasets/dev.json', help='validation dataset path')
parser.add_argument('--test_dataset_path', type=str, default='datasets/test.json', help='test dataset path')
args = parser.parse_args()

# Load Dataset
train_dataset = open(args.train_dataset_path,'r', encoding='utf-8').readlines()
# validation_dataset = open(args.dev_dataset_path,'r', encoding='utf-8').readlines()
test_dataset = open(args.test_dataset_path,'r', encoding='utf-8').readlines()

# Count dataset
# Train Dataset
train_label = {0:0, 1:0, 2:0}
for train in train_dataset:
    train = eval(train)
    train_label[train['label']]+=1
print('len of train:', len(train_dataset))
print('train labels:', train_label)

# Test Dataset
test_label = {0:0, 1:0, 2:0}
for test in test_dataset:
    test = eval(test)
    test_label[test['label']]+=1
print('len of test:', len(test_dataset))
print('test labels:', test_label)

# Dev Dataset
# dev_label = {0:0, 1:0, 2:0}
# for dev in validation_dataset:
#     dev = eval(dev)
#     dev_label[dev['label']]+=1
# print('len of dev:', len(validation_dataset))
# print('dev labels:', dev_label)
