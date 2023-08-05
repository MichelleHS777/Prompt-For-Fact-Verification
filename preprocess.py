from tqdm import tqdm
import json

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Preprocess Datasets')
parser.add_argument('--dataset', type=str, default="'./datasets/unpreprocess/test.json', help='dataset path')
parser.add_argument('--save_file', type=str, default='datasets/preprocessed/test.json', help='save file path')

args = parser.parse_args()
dataset = json.load(open(args.dataset, 'r', encoding='utf-8'))
save_file = open(args.save_file, 'w', encoding='utf-8')

for data in tqdm(dataset, desc='Preprocess...'):
    claimId = data['claimId']
    claim = data['claim']
    evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)] 
    evidences = ''.join(evidences)
    label = data['label']
    data = json.dumps({'claimId': claimId, 'claim': claim, 'evidences': evidences, 'label': label}, ensure_ascii=False)
    save_file.write(data + "\n")
save_file.close()
