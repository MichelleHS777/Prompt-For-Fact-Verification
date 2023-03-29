from collections import Counter
import sys, json, re
import math
from tqdm import tqdm
from config import set_args

def main():
    args = set_args()
    test_data = json.load(open(args.test_data_path, 'r', encoding='utf-8'))
    save_file = open(args.save_file, 'a+', encoding='utf-8')
    # count idf
    idf = {}
    idf = Counter()
    for row in test_data:
        sentList = []
        for ev in row['evidence'].values():
            sentList += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
        # remove duplicates
        sentList = list(set(sentList))
        for sent in sentList:
            idf[sent] += 1
    document_count = len(test_data)
    for key in idf.keys():
        idf[key] = math.log(document_count/idf[key]+1)
    
    # count tf and select
    get_evidence_num = 5
    tdidf_ev = []
    for row in tqdm(test_data, desc='Getting similar sentences...'):
        sentList = []
        claimId = row['claimId']
        claim = row['claim']
        label = row['label']
        ev_sent = row['evidence']
        for ev in row['evidence'].values():
            sentList += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
        sentList = [sent for sent in sentList if len(sent) > 5]
        tf = Counter()
        for sent in sentList:
            tf[sent] += 1
        for key in tf.keys():
            tf[key] = tf[key] / len(tf.keys())
        tf_idf = Counter()
        for sent in sentList:
            tf_idf[sent] = tf[sent] * idf[sent]
        tmp = list(tf_idf.items())
        tmp.sort(key=lambda s: s[1], reverse=True)
        # tmp = [ele[0] for ele in tmp if len(ele[0]) > 5]
        ev_sent = [s[0] for s in tmp[:5]]

        data = json.dumps(
            {'claimId': claimId, 'claim': claim, 'sentence': ev_sent, 'label': label},
            ensure_ascii=False
        )
        save_file.write(data + "\n")
if __name__ == '__main__':
    main()

