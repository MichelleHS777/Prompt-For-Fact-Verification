import argparse
import os
import argparse

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    InputExample
from tqdm import tqdm

batch_size = 4
model_name_or_path = "bert-base-chinese"
peft_type = PeftType.PREFIX_TUNING
device = "cuda"
num_epochs = 8

peft_config = PrefixTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=10)
lr = 1e-2

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load Dataset
train_path = 'datasets/unpreprocess/train.json'
dev_path = 'datasets/unpreprocess/dev.json'
test_path = 'datasets/unpreprocess/test.json'
datasets = DatasetDict.from_json({'train': train_path, 'dev': dev_path, 'test': test_path})

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default
    outputs = tokenizer(examples["claim"], examples["evidences"], truncation=True, max_length=512)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["claimId", "claim", "evidences"],
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

def collate_fn(examples):
    return tokenizer.pad(examples, return_tensors="pt")

# Instantiate dataloaders.
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
dev_dataloader = DataLoader(
    tokenized_datasets["dev"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=3)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.to(device)

optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    # ========================================
    #               Training
    # ========================================
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        print(batch)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.sum()
        loss.sum().backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # ========================================
    #               Validation
    # ========================================
    model.eval()
    valid_y_pred = []
    valid_y_true = []
    for step, batch in enumerate(tqdm(dev_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.softmax(dim=-1)
        predictions = torch.argmax(predictions, dim=-1)
        labels = batch['labels']
        valid_y_true.extend(labels.cpu().tolist())
        valid_y_pred.extend(predictions.cpu().tolist())
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro')
    microf1 = f1
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro')
    if f1 > best_macrof1:
        best_microf1 = microf1
        best_macrof1 = f1
        torch.save(model.state_dict(), f"./checkpoint/model.ckpt")
    print("Epoch {}, f1 {}".format(epoch, f1), flush=True)


# ========================================
#               Test
# ========================================
model.load_state_dict(torch.load(f"./checkpoint/model.ckpt"))
model = model.to(device)
model.eval()
test_y_pred = []
test_y_true = []
for step, batch in enumerate(tqdm(test_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.softmax(dim=-1)
    predictions = torch.argmax(predictions, dim=-1)
    labels = batch['labels']
    test_y_true.extend(labels.cpu().tolist())
    test_y_pred.extend(predictions.cpu().tolist())
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))