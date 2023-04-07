import torch
import torch.nn.functional as F
from config import set_args
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate, PtuningTemplate, MixedTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, AutomaticVerbalizer, \
    GenerationVerbalizer
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.plms import load_plm
from sklearn.metrics import classification_report
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForMaskedLM, BertTokenizer, AutoModelForCausalLM
from openprompt.plms import MLMTokenizerWrapper, LMTokenizerWrapper


# Load arguments
args = set_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
dataset = {}
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
train_dataset = open(args.train_file, 'r', encoding='utf-8').readlines()
# fake_train_dataset = open(args.fake_train_file, 'r', encoding='utf-8').readlines()
# train_dataset = train_dataset + fake_train_dataset
validation_dataset = open(args.valid_file, 'r', encoding='utf-8').readlines()
test_dataset = open(args.test_file, 'r', encoding='utf-8').readlines()


for data in train_dataset:
    data = eval(data)
    train_input_example = InputExample(
        text_a='[SEP]'.join(data['evidences']),
        # text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['train'].append(train_input_example)

for data in validation_dataset:
    data = eval(data)
    dev_input_example = InputExample(
        # text_a=data['evidences'],
        text_a='[SEP]'.join(data['evidences']),
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['validation'].append(dev_input_example)

for data in test_dataset:
    data = eval(data)
    test_input_example = InputExample(
        # text_a=data['evidences'],
        text_a='[SEP]'.join(data['evidences']),
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['test'].append(test_input_example)

# Load PLM
if args.plm == 'bert':
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-chinese")
if args.plm == 'roberta':
    tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-12_H-768")
    plm = AutoModelForMaskedLM.from_pretrained("uer/chinese_roberta_L-12_H-768")
    WrapperClass = MLMTokenizerWrapper


# Constructing Template
if args.template == 0:
    template_text = '{"placeholder":"text_a","shortenable":True}' \
                    '這是 {"mask":None, "length":2} 「{"placeholder":"text_b"}」'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)
elif args.template == 2:
    template_text = '{"placeholder":"text_a","shortenable":True} {"soft":None, "duplicate":10}' \
                    '這是 {"mask":None, "length":2} 「{"placeholder":"text_b"}」'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="lstm", text=template_text)
elif args.template == 3:
    template_text = '{"placeholder":"text_a","shortenable":True} {"soft":"這是"} {"mask":None, "length":2} ' \
                    '「 {"placeholder":"text_b"} 」}'
    template = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)

# Load dataloader
train_dataloader = PromptDataLoader(
    dataset=dataset['train'],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail", max_seq_length=args.max_length
)

validation_dataloader = PromptDataLoader(
    dataset=dataset['validation'],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    predict_eos_token=False, truncate_method="tail", max_seq_length=args.max_length
)

test_dataloader = PromptDataLoader(
    dataset=dataset['test'],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    predict_eos_token=False, truncate_method="tail", max_seq_length=args.max_length
)

# Define the verbalizer
if args.verbalizer == 0:
    verbalizer = ManualVerbalizer(
        classes=[0, 1, 2],
        num_classes=3,
        label_words={0: ["正确"], 1: ["错误"], 2: ["未知"]},
        tokenizer=tokenizer
    )
elif args.verbalizer == 1:
    verbalizer = SoftVerbalizer(
        tokenizer=tokenizer,
        model=plm,
        num_classes=3,
        label_words=[["正确"], ["错误"], ["未知"]]
    )
elif args.verbalizer == 2:
    verbalizer = AutomaticVerbalizer(
        tokenizer,
        num_classes=3,
    )
elif args.verbalizer == 3:
    verbalizer = GenerationVerbalizer(
        tokenizer,
        label_words={0: ["正确"], 1: ["错误"], 2: ["未知"]},
        classes=[0, 1, 2],
        is_rule=False
    )
# Load prompt model
if args.template == 0 or 3:
    prompt_model = PromptForClassification(
        plm=plm,
        template=template,
        verbalizer=verbalizer,
        freeze_plm=args.freeze
    )

else:
    prompt_model = PromptForGeneration(
        plm=plm,
        tokenizer=tokenizer,
        template=template,
        freeze_plm=args.freeze,
        plm_eval_mode=args.plm_eval_mode
    )

prompt_model = prompt_model.to(device)

# Now the training is standard
optimizer = AdamW(prompt_model.parameters(), eps=args.initial_eps, lr=args.initial_lr)
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


def get_each_label():
    train_label = {0: 0, 1: 0, 2: 0}
    for train in train_dataset:
        train = eval(train)
        train_label[train['label']] += 1
    return train_label[0], train_label[1], train_label[2]


sup, ref, nei = get_each_label()
tot_loss = 0
log_loss = 0
best_val_acc = 0
best_microf1 = 0
best_macrof1 = 0
best_recall = 0
best_precision = 0
weight = [1, 1, ref/nei*10]
class_weight = torch.FloatTensor(weight).to(device)
loss_func = torch.nn.CrossEntropyLoss(weight=class_weight)

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
    loss_func = torch.nn.CrossEntropyLoss()
    prompt_model.eval()
    valid_y_pred = []
    valid_y_true = []
    total_eval_loss = 0
    pbar = tqdm(validation_dataloader, desc="Valid")
    for step, inputs in enumerate(pbar):
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
prompt_model.load_state_dict(torch.load(f"./checkpoint/model.ckpt"))
prompt_model = prompt_model.to(device)
prompt_model.eval()
test_y_pred = []
test_y_true = []
pbar = tqdm(test_dataloader, desc="Test")
for step, inputs in enumerate(pbar):
    inputs = inputs.to(device)
    logits = prompt_model(inputs)
    labels = inputs['label']
    test_y_true.extend(labels.cpu().tolist())
    pred = F.softmax(logits, dim=-1)
    test_y_pred.extend(torch.argmax(pred, dim=-1).cpu().tolist())
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
