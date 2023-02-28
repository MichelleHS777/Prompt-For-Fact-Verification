import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import argparse
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix,  multilabel_confusion_matrix
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate, SoftTemplate, PTRTemplate, PrefixTuningTemplate, PtuningTemplate,MixedTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, KnowledgeableVerbalizer, AutomaticVerbalizer, GenerationVerbalizer, ProtoVerbalizer
from openprompt import PromptForClassification, PromptModel
from openprompt.plms import load_plm
from sklearn.metrics import classification_report

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Prompt Tuning For CHEF')
parser.add_argument('--cuda', type=str, default="0",help='appoint GPU devices')
parser.add_argument('--use_cuda', type=bool, default=True, help='if use GPU or not')
parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=8, help='training epochs for labeled data')
parser.add_argument('--freeze', type=bool, default=False, help='freeze plm or not, default is False')
parser.add_argument('--plm_eval_mode', type=bool, default=False, help='the dropout of the model is turned off')
parser.add_argument("--template", type=int,
help="Set template (0 for manual, 1 for soft, 2 for Ptuning, 3 for PrefixTuning, 4 for PTR, 5 for mix)", default=0)
parser.add_argument("--verbalizer", type=int,
help="Set template (0 for manual, 1 for soft, 2 for knowledge)", default=0)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
dataset = {}
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
train_dataset = open('datasets/evidence extraction/train_evidence.json','r', encoding='utf-8').readlines()
validation_dataset = open('datasets/evidence extraction/dev_evidence.json','r', encoding='utf-8').readlines()
test_dataset = open('datasets/evidence extraction/test_evidence.json','r', encoding='utf-8').readlines()

for data in train_dataset:
    data = eval(data)
    train_input_example = InputExample(
        text_a = data['evidence'],
        text_b = data['claim'],
        label=int(data['label'])
    )
    dataset['train'].append(train_input_example)

for data in validation_dataset:
    data = eval(data)
    dev_input_example = InputExample(
        text_a = data['evidence'],
        text_b = data['claim'],
        label=int(data['label'])
    )
    dataset['validation'].append(dev_input_example)

for data in test_dataset:
    data = eval(data)
    test_input_example = InputExample(
        text_a = data['evidence'],
        text_b = data['claim'],
        label=int(data['label'])
    )
    dataset['test'].append(test_input_example)

# Load PLM
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-chinese")

# Constructing Template
if args.template == 0:
    template_text = '請問 「{"placeholder":"text_a","shortenable":True}」和 「{"placeholder":"text_b"}」相關嗎?  答案:{"mask":None, "length":2}'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)
elif args.template == 1:
    # template_text = '{"placeholder":"text_a","shortenable":True} {"placeholder":"text_b"} 请问这是对的吗?答案: {"mask":None, "length":2}'
    template_text = '{"placeholder":"text_a","shortenable":True} 這是 {"mask":None, "length":2} 的「{"placeholder":"text_b"}」'
    template = SoftTemplate(model=plm, tokenizer=tokenizer, text=template_text)
elif args.template == 2:
    # template_text = '{"placeholder":"text_a"} {"placeholder":"text_b", "shortenable":False} {"soft": "请问这是对的吗?答案: "} {"mask":None, "length":2}'
    # template_text = '{"placeholder":"text_a","shortenable":True} {"placeholder":"text_b"} {"soft":None, "duplicate":20} {"mask":None, "length":2}'
    template_text = '{"placeholder":"text_a","shortenable":True} {"soft":None, "duplicate":5} {"placeholder":"text_b"} {"soft":None, "duplicate":10} {"soft":"答案:"} {"mask":None, "length":2}'
    template = PtuningTemplate(model=plm, tokenizer=tokenizer, prompt_encoder_type="lstm", text=template_text)
# elif args.template == 3:
#     template_text = '{"soft": "请问这是对的吗?答案: "} {"mask":None, "length":2} {"placeholder":"text_b", "shortenable":False} {"placeholder":"text_a"}'
#     template_text = '{"placeholder":"text_a"} {"placeholder":"text_b", "shortenable":False} {"soft":None, "duplicate":20} {"mask":None, "length":2}'
#     template = PrefixTuningTemplate(tokenizer=tokenizer, model=plm, text=template_text, using_decoder_past_key_values=False)
elif args.template == 5:
    template_text = '{"soft":"請問"} 「{"placeholder":"text_a","shortenable":True}」{"soft":"和"} 「{"placeholder":"text_b"}」{"soft":"相關"} {"soft":"嗎"}?  答案:{"mask":None, "length":2}'
    template = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)

# Load dataloader
train_dataloader = PromptDataLoader(
    dataset=dataset["train"],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, batch_size=args.batch_size,
    shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail", max_seq_length=args.max_length
)

validation_dataloader = PromptDataLoader(
    dataset=dataset["validation"],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    predict_eos_token=False, truncate_method="tail", max_seq_length=args.max_length
)

test_dataloader = PromptDataLoader(
    dataset=dataset["test"],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size, shuffle=True, teacher_forcing=False,
    predict_eos_token=False, truncate_method="tail", max_seq_length=args.max_length
)

# Define the verbalizer
if args.verbalizer == 0:
    verbalizer = ManualVerbalizer(
        classes = [0,1],
        num_classes = 2,
        # label_words = {0:["是的"], 1:["不是"], 2:["未知"]},
        label_words = {0:["无关"], 1:["相关"]},
        tokenizer = tokenizer
    )
elif args.verbalizer == 1:
    verbalizer = SoftVerbalizer(
        tokenizer=tokenizer,
        model = plm,
        num_classes = 2,
    )
elif args.verbalizer == 2: #TODO:Add knowledge verbalizer txt ref:Calibration
    verbalizer = KnowledgeableVerbalizer(tokenizer, plm, num_classes=3)
elif args.verbalizer == 3:
    verbalizer = AutomaticVerbalizer(tokenizer, num_classes=3)
elif args.verbalizer == 4:
    verbalizer = GenerationVerbalizer(
        tokenizer,
        # label_words={0:"是的", 1:"不是", 2: "未知"},
        label_words = {0:["无关"], 1:["相关"]},
        is_rule=False
    )
elif args.verbalizer == 5:
    verbalizer = ProtoVerbalizer(
        classes = [0,1],
        tokenizer=tokenizer,
        model=plm,
        label_words={0:["无关"], 1:["相关"]},
    )

# Load prompt model
if args.template == 0 or 1 or 5:
    prompt_model = PromptForClassification(
        plm=plm,
        template=template,
        verbalizer=verbalizer,
    )

elif args.template == 2 or 3:
    prompt_model = PromptModel(
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
        num_warmup_steps = 0,
        num_training_steps = total_steps
)

tot_loss = 0
log_loss = 0
best_val_acc = 0
best_microf1 = 0
best_macrof1 = 0
best_recall = 0
best_precision = 0
# weight = [1, 1, 4349/776*10]
# class_weight = torch.FloatTensor(weight).to(device)
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

    # ========================================
    #               Validation
    # ========================================
    # val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
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
        valid_y_pred.extend(labels.cpu().tolist())
        valid_y_true.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='micro')
    microf1 = f1
    pre, recall, f1, _ = precision_recall_fscore_support(valid_y_true, valid_y_pred, average='macro')
    if f1 > best_macrof1:
        best_microf1 = microf1
        best_macrof1 = f1
        torch.save(prompt_model.state_dict(),f"./checkpoint/model.ckpt")
    print("Epoch {}, f1 {}".format(epoch, f1), flush=True)

# ========================================
#               Test
# ========================================
print("Prediction...")
# torch.cuda.empty_cache()
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
    test_y_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
microf1 = f1
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))
print("confusion_matrix:\n", confusion_matrix(test_y_true, test_y_pred, labels=[0,1,2]))
print("multilabel_confusion_matrix:\n", multilabel_confusion_matrix(test_y_true, test_y_pred, labels=[0,1,2]))
each_label_result = classification_report(test_y_true, test_y_pred)
print("Each label result:\n", each_label_result)
