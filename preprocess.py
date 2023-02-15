import json
import argparse

parser = argparse.ArgumentParser(description='Preprocess Dataset')
parser.add_argument('--dataset_path', type=str, default='datasets/train.json', help='load dataset to preprocess')
parser.add_argument('--save_dataset_path', type=str, default='datasets/train.json', help='save dataset to path')
args = parser.parse_args()

# load json file
datalist = json.load(open(args.dataset_path, 'r', encoding='utf-8'))
# save json file 
jsonFile = open(args.save_dataset_path, "w", encoding="utf8")

"=====Preprocessing====="
evidences = []
evidence_sentence = ''
datasets = []
for row in range (len(datalist)):
    # get claimID, claim, label
    claimID = datalist[row]['claimId']
    claim = datalist[row]['claim']
    label = datalist[row]['label']
    # get gold evidence
    evidences = [datalist[row]['gold evidence'][str(i)]['text'] for i in range(5) if datalist[row]['gold evidence'][str(i)]['text'] != '']
    # write in json file
    # if len(evidences)!=0:
    evidence_sentence = ''.join([str(evidence) for evidence in evidences])
    data = {'claimId':int(claimID), 'claim':claim, 'evidences':evidence_sentence, 'label':label}
    jsonString = json.dumps(data, ensure_ascii=False)
    # jsonString = json.dumps(data, indent=4, ensure_ascii=False)
    jsonFile.write(jsonString + "\n")
jsonFile.close()
print("=====Preprocess Finish=====")