import json
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
    AutoModelForCausalLM, ErnieForMaskedLM
from openprompt.plms import MLMTokenizerWrapper


# Load arguments
args = set_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load Dataset
dataset = {}
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
train_dataset = json.load(open(args.train_file, 'r', encoding='utf-8'))
# fake_train_dataset = open(args.fake_train_file, 'r', encoding='utf-8').readlines()
# train_dataset = train_dataset + fake_train_dataset
validation_dataset = json.load(open(args.valid_file, 'r', encoding='utf-8'))
test_dataset = json.load(open(args.test_file, 'r', encoding='utf-8'))


for data in train_dataset:
    evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    train_input_example = InputExample(
        text_a=''.join(evidences),
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['train'].append(train_input_example)

for data in validation_dataset:
    evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    dev_input_example = InputExample(
        text_a=''.join(evidences),
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['validation'].append(dev_input_example)

for data in test_dataset:
    evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    test_input_example = InputExample(
        text_a=''.join(evidences),
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


# Constructing Template
if args.template == 0:
    template_text = '{"placeholder":"text_a"} {"placeholder":"text_b"}' \
                    '要評估宣稱和證據的結果是支持或反對，或者未知 {"mask":None, "length":2}'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

elif args.template == 1:
    template_text = '{"placeholder":"text_a"} {"placeholder":"text_b"}' \
                    '{"soft":"要評估宣稱和證據的結果是支持或反對，或者未知"} {"mask":None, "length":2}'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="mlp", text=template_text)
    # wrapped_example = template.wrap_one_example(dataset['train'][1])
    # wrapped_tokenizer = WrapperClass(max_seq_length=512, tokenizer=tokenizer,truncate_method="head")
    # tokenized_example = wrapped_tokenizer.tokenize_one_example(wrapped_example, teacher_forcing=False)
    # print(tokenized_example)
    # print(tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))

elif args.template == 2:
    template_text = '{"placeholder":"text_a"} {"placeholder":"text_b"}' \
                    '{"soft":None, "duplicate":10} ' \
                    '要評估宣稱和證據的結果是支持或反對，或者未知 {"mask":None, "length":2}'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="mlp", text=template_text)


# Load dataloader
train_dataloader = PromptDataLoader(
    dataset=dataset['train'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, truncate_method="tail", max_seq_length=args.max_length
)

validation_dataloader = PromptDataLoader(
    dataset=dataset['validation'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, truncate_method="tail", max_seq_length=args.max_length
)

test_dataloader = PromptDataLoader(
    dataset=dataset['test'], template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, truncate_method="tail", max_seq_length=args.max_length
)

# Define the verbalizer
verbalizer = ManualVerbalizer(
    classes=[0, 1, 2],
    num_classes=3,
    label_words={0: ["正確", "支持"], 1: ["錯誤", "反對"], 2: ["未知"]},
    tokenizer=tokenizer
)


# Load prompt model
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


# def get_each_label():
#     train_label = {0: 0, 1: 0, 2: 0}
#     for train in train_dataset:
#         train = eval(train)
#         train_label[train['label']] += 1
#     return train_label[0], train_label[1], train_label[2]


# sup, ref, nei = get_each_label()
tot_loss = 0
best_val_acc = 0
best_microf1 = 0
best_macrof1 = 0
best_recall = 0
best_precision = 0
loss_func = torch.nn.CrossEntropyLoss()
# weight = [1, 1, ref/nei*10]
# class_weight = torch.FloatTensor(weight).to(device)
# loss_func = torch.nn.CrossEntropyLoss(weight=class_weight)

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
    total_eval_loss = 0
    prompt_model.eval()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(validation_dataloader, desc="Valid")):
            inputs = inputs.to(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            val_loss = loss_func(logits, labels)
            total_eval_loss += val_loss.sum().item()
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
    print("Epoch {}, valid_loss {}".format(epoch, total_eval_loss / len(validation_dataloader)), flush=True)
    print("Epoch {}, f1 {}".format(epoch, f1), flush=True)

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

print("Accuracy: {:.2%}".format(accuracy_score(test_y_true, test_y_pred)))
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))
print("confusion_matrix:\n", confusion_matrix(test_y_true, test_y_pred, labels=[0, 1, 2]))
print("multilabel_confusion_matrix:\n", multilabel_confusion_matrix(test_y_true, test_y_pred, labels=[0, 1, 2]))
each_label_result = classification_report(test_y_true, test_y_pred)
print("Each label result:\n", each_label_result)
