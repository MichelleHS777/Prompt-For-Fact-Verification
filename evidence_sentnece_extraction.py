import re
import json
from tqdm import tqdm
# import torch
# from transformers import  AdamW, get_linear_schedule_with_warmup
# from datasets import load_dataset
# import argparse
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix,  multilabel_confusion_matrix
# from openprompt.data_utils import InputExample
# from openprompt import PromptDataLoader
# from openprompt.prompts import ManualTemplate, SoftTemplate, PTRTemplate, PrefixTuningTemplate, PtuningTemplate,MixedTemplate
# from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, KnowledgeableVerbalizer, AutomaticVerbalizer, GenerationVerbalizer, ProtoVerbalizer
# from openprompt import PromptForClassification, PromptModel
# from openprompt.plms import load_plm

# ------------------------init parameters----------------------------
# parser = argparse.ArgumentParser(description='Prompt Tuning For CHEF')
# parser.add_argument('--cuda', type=str, default="0",help='appoint GPU devices')
# parser.add_argument('--use_cuda', type=bool, default=True, help='if use GPU or not')
# parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
# parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
# parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
# parser.add_argument('--epochs', type=int, default=8, help='training epochs for labeled data')
# parser.add_argument('--freeze', type=bool, default=False, help='freeze plm or not, default is False')
# parser.add_argument('--plm_eval_mode', type=bool, default=False, help='the dropout of the model is turned off')
# parser.add_argument("--template", type=int,
# help="Set template (0 for manual, 1 for soft, 2 for Ptuning, 3 for PrefixTuning, 4 for PTR, 5 for mix)", default=0)
# parser.add_argument("--verbalizer", type=int,
# help="Set template (0 for manual, 1 for soft, 2 for knowledge)", default=0)
# args = parser.parse_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = {}
dataset['train'] = []

# Load dataset
train_dataset = json.load(open('./datasets/unpreprocess/train2.json', 'r', encoding='utf-8'))

# # Load PLM
# plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-chinese")
#
# # template
# template_text = '請問「{"placeholder":"text_a","shortenable":True}」與 「{"placeholder":"text_b"}」有關嗎? {"mask":None, "length":2}'
# template = ManualTemplate(tokenizer=tokenizer, text=template_text)
#
# # verbalizer
# verbalizer = ManualVerbalizer(
#     classes=[0, 1],
#     num_classes=3,
#     label_words={0: ["無關"], 1: ["相關"]},
#     tokenizer=tokenizer
# )
#
# # Load prompt model
# prompt_model = PromptForClassification(
#     plm=plm,
#     template=template,
#     verbalizer=verbalizer,
# )
gold_evidence = []
non_evidence = []
inter_evidence = []
for row in tqdm(train_dataset):
    claim = row['claim']
    gold_evidence.append([row['gold evidence'][str(i)]['text'] for i in range(5) if row['gold evidence'][str(i)]['text'] != ''])
    other_sentence = []
    for ev in row['evidence'].values():
        other_sentence += re.split(r'[？：。！（）.“”…\t\n]', ev['text'])
    non_evidence.append([sent for sent in other_sentence if len(sent) > 5])
    inter_evidence.append([e for e in gold_evidence if e == non_evidence])
print('other_sentence\n', non_evidence[1])
print('gold_evidence\n', gold_evidence[1])
print('inter_evidence\n', inter_evidence)




