from tqdm import tqdm
import re
import json

dataset = json.load(open('./datasets/unpreprocess/dev.json', 'r', encoding='utf-8'))
save_file = open('./datasets/evidence_splitGold/dev.json', 'w', encoding='utf-8')


for data in tqdm(dataset, desc='Preprocess...'):
    claimId = data['claimId']
    claim = data['claim']
    label = data['label']
    evidence_doc = [data['evidence'][str(i)]['text'] for i in range(5)]
    gold_evidence = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    for evidence_text in evidence_doc:
        for gold_evidence_text in gold_evidence:
            evidence_text = evidence_text.replace(gold_evidence_text, '')
        evidence_text = re.split(r'[？：。！（）.“”…\t\n]', evidence_text)
        evidence_text = [evidence for evidence in evidence_text if len(evidence)>5]
        data = json.dumps({'claimId': int(claimId), 'claim': claim, 'evidences': evidence_text + gold_evidence, 'label': label}, ensure_ascii=False)
    save_file.write(data + "\n")
save_file.close()
