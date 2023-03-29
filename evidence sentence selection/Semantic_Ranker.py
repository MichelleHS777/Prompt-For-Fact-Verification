from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
from config import set_args
import json
from tqdm import tqdm
from collections import OrderedDict
import numpy as np


device = torch.device("cuda")
args = set_args()
# load file and save file
dataset = open(args.test_data_path, 'r', encoding='utf-8')
dataset = list(dataset)
save_file = open(args.save_file, 'a+', encoding='utf-8')

# get unique Id
claimId = [eval(data)['claimId'] for data in dataset]
claimId = list(OrderedDict.fromkeys(claimId))

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
model = model.to(device)

def get_dict_by_id(list, id):
    """
    Get a list of dictionaries with the same ID from list.
    :param list: a list of dictionaries
    :param id: the ID to search for
    :return: a list of dictionaries with the same ID
    """
    result = []
    for d in list:
        d = eval(str(d))
        if d.get('claimId') == id:
            result.append(d)
    return result


def cosSimilarity(sent1, sent2, model, tokenizer):
    # sentence 1
    encoded_dict = tokenizer.encode_plus(
        sent1,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )
    input_ids = torch.tensor(encoded_dict['input_ids']).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_state = outputs[0]
    # CLS 对应的向量
    sent1_vec = last_hidden_state[0][0].detach().cpu().numpy()

    # sentence 2
    encoded_dict = tokenizer.encode_plus(
        sent2,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True
    )
    input_ids = torch.tensor(encoded_dict['input_ids']).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_state = outputs[0]

    # CLS 对应的向量
    sent2_vec = last_hidden_state[0][0].detach().cpu().numpy()
    cos_sim = np.dot(sent1_vec, sent2_vec) / (np.linalg.norm(sent1_vec) * np.linalg.norm(sent2_vec))
    return cos_sim.item()


if __name__ == '__main__':
    # get the sentences from same Id
    for id in tqdm(claimId, desc='getting similar sentence...'):
        sent2sim = {}
        same_id_dataset = get_dict_by_id(dataset, id)
        claimId = id
        for data in same_id_dataset:
            claim = data['claim']
            ev_sent = data['evidence']
            label = data['label']
            if ev_sent in sent2sim:
                continue
            sent2sim[ev_sent] = cosSimilarity(claim, ev_sent, model, tokenizer)
        sent2sim = list(sent2sim.items())
        sent2sim.sort(key=lambda s: s[1], reverse=True)
        ev_sent = [s[0] for s in sent2sim[:5] if s[1]>0.8]
        data = json.dumps({'claimId': claimId, 'claim': claim, 'evidence': ev_sent, 'label':label}, ensure_ascii=False)
        save_file.write(data + "\n")
