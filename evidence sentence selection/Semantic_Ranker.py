from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM
)
import torch
import os
import re
import sys
import copy
import random
import time, datetime
from time import sleep
import json, csv
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np


device = torch.device("cuda")
max_length = 256
save_file = open('result/0311/semantic_result.json', 'a+', encoding='utf-8')
def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model = model.to(device)

    data_list = open('./data/CHEF/test_evidence.json', 'r', encoding='utf-8')

    # similar_evs = []
    for row in tqdm(data_list):
        row = eval(row)
        claimID = row['claimId']
        claim = row['claim']
        ev_sent = row['evidence']
        sentence_score = cosSimilarity(claim, ev_sent, model, tokenizer)
        if sentence_score > 0.8:
            label = 1
            data = json.dumps({'claimId': claimID, 'claim': claim, 'sentence': ev_sent, 'label': label}, ensure_ascii=False)
        else:
            label = 0
            data = json.dumps({'claimId': claimID, 'claim': claim,'sentence': "", 'label': label},ensure_ascii=False)
        save_file.write(data + "\n")


def cosSimilarity(sent1, sent2, model, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        sent1,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_length,  # Pad & truncate all sentences.
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
    encoded_dict = tokenizer.encode_plus(
        sent2,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=max_length,  # Pad & truncate all sentences.
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
    main()
