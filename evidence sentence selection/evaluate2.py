import json
import numpy as np
import torch
import transformers
import numpy
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance


dataset = json.load(open('data/CHEF_evidence/test.json', 'r', encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")
device = torch.device("cuda")
model = model.to(device)


def cal_cossim(sent1, sent2, model, tokenizer):
    model.eval()
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

    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_state = outputs[0]
    # CLS 对应的向量
    sent1_vec = last_hidden_state[0][0].detach().cpu().numpy()
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


precision = []
recall = []
for data in dataset:
    pred = []
    gold_evidence = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    tfidf_evidence = [data['tfidf'][i] for i in range(5)]
    cossim_evidence = [data['cossim'][i] for i in range(5)]
    ranksvm_evidence = [data['ranksvm'][i] for i in range(5)]

    for pred_sent in tfidf_evidence:
        sim_score = [cal_cossim(gold, pred_sent, model, tokenizer) for gold in gold_evidence]
        print('sim_score:', sim_score)
        result = any(score > 0.8 for score in sim_score)
        if result:
            pred.append(1)
        else:
            pred.append(0)

    # num of gold evidence
    retrieved_num = len([evidence for evidence in gold_evidence if evidence != ""])
    # num of retrieved
    relevant_num = len([evidence for evidence in tfidf_evidence if evidence != ""])
    # num of retrieved relevant
    retrieved_relevant = pred.count(1)

    precision.append(retrieved_relevant/relevant_num)
    recall.append(retrieved_relevant/retrieved_num)
    print('retrieved_relevant:', retrieved_relevant, 'relevant_num:', relevant_num, 'retrieved_num:', retrieved_num)
    print('pred:', pred)
    print('precision:', precision)
    print('recall:', recall)

