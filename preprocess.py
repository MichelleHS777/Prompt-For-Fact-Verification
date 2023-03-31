from tqdm import tqdm
import json

dataset = json.load(open('./data/unpreprocess/test.json', 'r', encoding='utf-8'))
save_file = open('datasets/claim verification/test.json', 'w', encoding='utf-8')


for data in tqdm(dataset, desc='Preprocess...'):
    claimId = data['claimId']
    claim = data['claim']
    evidences = [data['gold evidences'][str(i)]['text'] for i in range(5)]
    data = json.dumps({'claimId': claimId, 'claim': claim, 'evidences': evidences}, ensure_ascii=False)
    save_file.write(data + "\n")
save_file.close()
