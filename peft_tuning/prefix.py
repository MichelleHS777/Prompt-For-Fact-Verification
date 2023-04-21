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
device = "cpu"
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
train_path = 'datasets/preprocessed/train.json'
test_path = 'datasets/preprocessed/test.json'
datasets = DatasetDict.from_json({'train':train_path, 'test':test_path})

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default
    outputs = tokenizer(examples["claim"], examples["evidences"], truncation=True, max_length=256)
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
eval_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, num_labels=3)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = AdamW(params=model.parameters(), lr=lr)
# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
print(model)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.softmax(dim=-1)
        predictions, references = predictions, batch["labels"]


