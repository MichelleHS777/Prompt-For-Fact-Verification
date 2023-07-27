from tqdm import tqdm
import json

dataset = json.load(open('./datasets/unpreprocess/test.json', 'r', encoding='utf-8'))
save_file = open('datasets/preprocessed/test.json', 'w', encoding='utf-8')


for data in tqdm(dataset, desc='Preprocess...'):
    claimId = data['claimId']
    claim = data['claim']
    evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    evidences = ''.join(evidences)
    label = data['label']
    data = json.dumps({'claimId': claimId, 'claim': claim, 'evidences': evidences, 'label': label}, ensure_ascii=False)
    save_file.write(data + "\n")
save_file.close()
