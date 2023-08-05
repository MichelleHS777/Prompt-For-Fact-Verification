import json
import os
import torch
import torch.nn.functional as F
from config import set_args
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix, accuracy_score
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate, PtuningTemplate, MixedTemplate, PrefixTuningTemplate
from openprompt.prompts import ManualVerbalizer, AutomaticVerbalizer, KnowledgeableVerbalizer
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration, PromptModel
from openprompt.plms import load_plm
from sklearn.metrics import classification_report
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForMaskedLM, BertTokenizer, \
    AutoModelForCausalLM, ErnieForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, AutoModel, AutoModelForSeq2SeqLM
from openprompt.plms import MLMTokenizerWrapper, T5TokenizerWrapper
# from pytorchtools import EarlyStopping


# Load arguments
args = set_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 計算模型的訓練參數量
def get_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load Dataset
train_dataset = open(args.train_file, 'r', encoding='utf-8').readlines()
if args.data_aug:
    fake_train_dataset = open(args.fake_train_file, 'r', encoding='utf-8').readlines()
    train_dataset = train_dataset + fake_train_dataset
validation_dataset = open(args.valid_file, 'r', encoding='utf-8').readlines()
test_dataset = open(args.test_file, 'r', encoding='utf-8').readlines()

# wrap datasets to input format
dataset = {}
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
for data in train_dataset:
    data = eval(data)
    if len(data['evidences'])==0:
        data['evidences'] = ''
    else:
        data['evidences'] = ''.join(data['evidences'])
    train_input_example = InputExample(
        text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['train'].append(train_input_example)

for data in validation_dataset:
    data = eval(data)
    if len(data['evidences'])==0:
        data['evidences'] = ''
    else:
        data['evidences'] = ''.join(data['evidences'])
    dev_input_example = InputExample(
        text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['validation'].append(dev_input_example)

for data in test_dataset:
    data = eval(data)
    if len(data['evidences'])==0:
        data['evidences'] = ''
    else:
        data['evidences'] = ''.join(data['evidences'])
    test_input_example = InputExample(
        text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['test'].append(test_input_example)

# Load PLM
if args.plm == 'bert':
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-chinese") #yechen/bert-large-chinese
elif args.plm == 'bert-large':
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "yechen/bert-large-chinese")
elif args.plm == 'roberta':
    tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")
    plm = AutoModelForMaskedLM.from_pretrained("uer/chinese_roberta_L-12_H-768")
    WrapperClass = MLMTokenizerWrapper
elif args.plm == 'ernie':
    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
    plm = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
    WrapperClass = MLMTokenizerWrapper
elif args.plm == 'ernie-large':
    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-xbase-zh")
    plm = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-xbase-zh")
    WrapperClass = MLMTokenizerWrapper
elif args.plm == 't5':
    plm, tokenizer, model_config, WrapperClass = load_plm("t5", "ClueAI/PromptCLUE-base-v1-5")  # yechen/bert-large-chinese

# Constructing Template
if args.template == 0:
    template_text = '證據:{"placeholder":"text_a"} 宣稱:{"placeholder":"text_b"}' \
                    '請問宣稱是對的嗎? {"mask":None, "length":2}'
    # template_text = '證據:{"placeholder":"text_a"} 宣稱:{"placeholder":"text_b"}' \
    #                     '要評估宣稱和證據的結果是支持或反對，或者未知 {"mask":None, "length":2}'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

elif args.template == 1:
    template_text = '證據:{"placeholder":"text_a"} 宣稱:{"placeholder":"text_b"}' \
                    '{"soft":"要評估宣稱和證據的結果是支持或反對，或者未知"} {"mask":None, "length":2}'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="lstm", text=template_text)

elif args.template == 2:
    template_text = '{"soft":None, "duplicate":10} 證據:{"placeholder":"text_a"} 宣稱:{"placeholder":"text_b"}' \
                    '{"soft":"要評估宣稱和證據的結果是支持或反對，或者未知"} {"mask":None, "length":2}'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="lstm", text=template_text)
# elif args.template == 3:
#     template_text = '{"soft":None, "duplicate":10} {"placeholder":"text_a"} {"placeholder":"text_b"} {"mask":None, "length":2}'
#     template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="lstm", text=template_text)



# Load dataloader
train_dataloader = PromptDataLoader(
    dataset=dataset['train'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, truncate_method="tail", max_seq_length=args.max_length,
)

validation_dataloader = PromptDataLoader(
    dataset=dataset['validation'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, truncate_method="tail", max_seq_length=args.max_length,
)

test_dataloader = PromptDataLoader(
    dataset=dataset['test'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=False, truncate_method="tail", max_seq_length=args.max_length,
)

# Define the verbalizer
verbalizer = ManualVerbalizer(
    classes=[0, 1, 2],
    num_classes=3,
    label_words={0: ["支持"], 1: ["反對"], 2: ["未知"]},
    tokenizer=tokenizer
)


# Load prompt model
if args.plm=='t5':
    prompt_model = PromptForGeneration(
        plm=plm,
        template=template,
        freeze_plm=args.freeze
    )
else:
    print('args.freeze:', args.freeze)
    prompt_model = PromptForClassification(
        plm=plm,
        template=template,
        verbalizer=verbalizer,
        freeze_plm=args.freeze
    )

prompt_model = prompt_model.to(device)


# Now the training is standard
optimizer = AdamW(prompt_model.parameters(), eps=args.initial_eps, lr=args.initial_lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * args.epochs)
)
# early_stopping = EarlyStopping(patience=args.patience, verbose=True)


best_val_acc = 0
best_microf1 = 0
best_macrof1 = 0
best_recall = 0
best_precision = 0
loss_func = torch.nn.CrossEntropyLoss()

# Training
for epoch in range(args.epochs):
    # ========================================
    #               Training
    # ========================================
    print("Training...")
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(tqdm(train_dataloader, desc="Training")):
        inputs = inputs.to(device)
        prompt_model.zero_grad()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.requires_grad_(True)
        tot_loss += loss.sum().item()
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    print("Epoch {}, train_loss {}".format(epoch, tot_loss / len(train_dataloader)), flush=True)

    # ========================================
    #               Validation
    # ========================================
    valid_y_pred = []
    valid_y_true = []
    total_val_loss = 0
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(validation_dataloader, desc="Valid")):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            val_loss = loss_func(logits, labels)
            total_val_loss += val_loss.sum().item()
            valid_y_true.extend(labels.cpu().tolist())
            pred = F.softmax(logits, dim=-1)
            valid_y_pred.extend(torch.argmax(pred, dim=-1).cpu().tolist())

    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro')
    microf1 = f1
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro')
    if f1 > best_macrof1:
        best_microf1 = microf1
        best_macrof1 = f1
        torch.save(prompt_model.state_dict(), f"./checkpoint/model.ckpt")
        # if not os.path.exists(args.model_save_dir):
        #     os.makedirs(args.model_save_dir)
        #
        # print("Saving model to %s" % args.model_save_dir)
        # model_to_save = prompt_model.module if hasattr(prompt_model, 'module') else prompt_model
        # # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.model_save_dir)
    print("Epoch {}, valid_loss {}".format(epoch, total_val_loss / len(validation_dataloader)), flush=True)
    print("Epoch {}, valid f1 {}".format(epoch, f1), flush=True)

    # early_stopping(total_val_loss, prompt_model)
    # # 若满足 early stopping 要求
    # if early_stopping.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break

# ========================================
#               Test
# ========================================
print("Prediction...")
test_y_pred = []
test_y_true = []
prompt_model.load_state_dict(torch.load(f"./checkpoint/model.ckpt"))
prompt_model = prompt_model.to(device)
prompt_model.eval()
with torch.no_grad():
    for step, inputs in enumerate(tqdm(test_dataloader, desc="Test")):
        inputs = inputs.to(device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        test_y_true.extend(labels.cpu().tolist())
        pred = F.softmax(logits, dim=-1)
        test_y_pred.extend(torch.argmax(pred, dim=-1).cpu().tolist())

print('true:', test_y_true, '\n', 'pred:', test_y_pred)
# 检查索引位置上的不匹配
mismatches = [{'index': i + 1, 'pred': test_y_pred[i], 'true': test_y_true[i]} for i in range(len(test_y_pred)) if
              test_y_pred[i] != test_y_true[i]]
# 打印不匹配的索引位置
print("Mismatched indices:", mismatches)

pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("Precision (micro): {:.2%}".format(pre))
print("   Recall (micro): {:.2%}".format(recall))
print("       F1 (micro): {:.2%}".format(f1))

pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))

print("confusion_matrix:\n", confusion_matrix(test_y_true, test_y_pred, labels=[0, 1, 2]))
print("multilabel_confusion_matrix:\n", multilabel_confusion_matrix(test_y_true, test_y_pred, labels=[0, 1, 2]))

each_label_result = classification_report(test_y_true, test_y_pred)
print("Each label result:\n", each_label_result)
