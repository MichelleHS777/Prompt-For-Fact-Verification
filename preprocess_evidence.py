import re
# import regex as re
import argparse
import json

parser = argparse.ArgumentParser(description='Preprocess Dataset')
parser.add_argument('--dataset_path', type=str, default='datasets/unpreprocess/test.json', help='load dataset to preprocess')
parser.add_argument('--save_dataset_path', type=str, default='datasets/test_evidence.json', help='save dataset to path')
args = parser.parse_args()

datalist = json.load(open(args.dataset_path, 'r', encoding='utf-8'))
jsonFile = open(args.save_dataset_path, "w", encoding="utf8")

def search_nonEvidence(doc_list, gold_evidence_list):
    ev_sents = []
    preprocess_doc_i = None
    for doc_i in doc_list: # claim_i match to doc_i with 5 doc
        for gold in gold_evidence_list: # claim_i match to gold evidence_i with 5 sentences
            doc_i = doc_i.replace(gold, '') # delete gold evidence from document
            # find = re.findall(gold, doc_i)
            # if len(find) != 0:
            #     doc_i = re.sub(find[0], '', doc_i)
        ev_sents += re.split(r'[？：。！（）.“”…\t\n]', doc_i)
        preprocess_doc_i = [sent for sent in ev_sents if len(sent) > 10]
    return preprocess_doc_i

print("==================Preprocess==================")
for row in range(len(datalist)):
    claimID = datalist[row]['claimId']
    claim = datalist[row]['claim']
    doc_evidences = [datalist[row]['evidence'][str(i)]['text'] for i in range(5)
                     if datalist[row]['evidence'][str(i)]['text'] != '']
    gold_evidences = [datalist[row]['gold evidence'][str(i)]['text'] for i in range(5)
                     if datalist[row]['gold evidence'][str(i)]['text'] != '']
    # try:
    search_not_evidence = search_nonEvidence(doc_evidences, gold_evidences)
    # except:
    #     print(search_not_evidence)
    not_evidence = {'claimId':int(claimID), 'claim':claim, "evidence":search_not_evidence, "label":0}
    gold_evidence = {'claimId': int(claimID), 'claim':claim, "evidence":gold_evidences, "label":1}
    jsonString = json.dumps(not_evidence, ensure_ascii=False)
    jsonFile.write(jsonString + "\n")
    jsonString = json.dumps(gold_evidence, ensure_ascii=False)
    jsonFile.write(jsonString + "\n")
jsonFile.close()
print("==================Finish==================")
