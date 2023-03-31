from random import sample
import json
from config import set_args
from tqdm import tqdm

args = set_args()
train_dataset = open(args.train_file, 'r', encoding='utf-8').readlines()
save_file = open('train_fake_NE_4000.json', 'w', encoding='utf-8')

fake_id = []
del_nei = []

for data in train_dataset:
    data = eval(data)
    data['label'] = int(data['label'])
    if data["label"] == 2:
        continue
    del_nei.append(data)

print("Generating fake data...")
while len(fake_id) != 4000:
    random_sample = sample(del_nei, 2)
    if random_sample[0]['claimId'] in fake_id:
        continue
    if len(random_sample[1]['evidencess']) == 0:
        continue

    claimId = random_sample[0]['claimId']
    claim = random_sample[0]['claim']
    evidences = random_sample[1]['evidencess']
    fake_id.append(claimId) # count the collected numbers of data

    # write claim + "" in json
    # if len(fake_id) <= 3200:
    fake = {'claimId': claimId, 'claim': claim, 'evidencess': "", 'label': 2}
    # write claim+non-evidences in json
    # else:
    # fake = {'claimId': claimId, 'claim': claim, 'evidencess': evidences, 'label': 2}
    data = json.dumps(fake, ensure_ascii=False)
    save_file.write(data + '\n')

