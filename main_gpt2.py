import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer
from openprompt.plms import LMTokenizerWrapper
from config import set_args
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, multilabel_confusion_matrix
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from sklearn.metrics import classification_report
from openprompt.data_utils.data_sampler import FewShotSampler
from transformers import AutoModelForCausalLM


# Load arguments
args = set_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Load Dataset
dataset = {}
dataset['train'] = []
dataset['validation'] = []
dataset['test'] = []
train_dataset = open('datasets/claim verification/gpt35.json', 'r', encoding='utf-8').readlines()
test_dataset = open('datasets/claim verification/test.json', 'r', encoding='utf-8').readlines()

# sampler = FewShotSampler(num_examples_per_label=args.shot_num)
for data in train_dataset:
    data = eval(data)
    train_input_example = InputExample(
        text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['train'].append(train_input_example)
# dataset['train'] = sampler(dataset['train'], seed=777)

for data in test_dataset:
    data = eval(data)
    test_input_example = InputExample(
        text_a=data['evidences'],
        text_b=data['claim'],
        label=int(data['label'])
    )
    dataset['test'].append(test_input_example)


# Load PLM
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
plm = AutoModelForCausalLM.from_pretrained("plm/gpt2-base-chinese")
WrapperClass = LMTokenizerWrapper

# Constructing Template
template_text = '請幫我判斷這個主張是符合以下哪種驗證結果。請只要回傳驗證結果選項驗證結果選項: 正确\n错误\n未知' \
                '主張:「{"placeholder":"text_b","shortenable":True}」'\
                '证据:「{"placeholder":"text_a","shortenable":True}」' \
                '驗證結果: {"mask":None, "length":2}'
template = ManualTemplate(tokenizer=tokenizer, text=template_text)

# Define the verbalizer
verbalizer = ManualVerbalizer(
    classes=[0, 1, 2],
    num_classes=3,
    label_words={0: ["正确"], 1: ["错误"], 2: ["未知"]},
    tokenizer=tokenizer
)

# Load dataloader
train_dataloader = PromptDataLoader(
    dataset=dataset['train'],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size,
    shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail", max_seq_length=args.max_length
)

test_dataloader = PromptDataLoader(
    dataset=dataset['test'],
    template=template, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=args.batch_size, shuffle=True,
    teacher_forcing=False,
    predict_eos_token=False, truncate_method="tail",
    max_seq_length=args.max_length
)

# Load prompt model
prompt_model = PromptForClassification(
    plm=plm,
    template=template,
    verbalizer=verbalizer,
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

tot_loss = 0
log_loss = 0
best_val_acc = 0
best_microf1 = 0
best_macrof1 = 0
best_recall = 0
best_precision = 0

loss_func = torch.nn.CrossEntropyLoss()

# Training
if args.shot:
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
#               Test
# ========================================
print("Prediction...")
# prompt_model.load_state_dict(torch.load(f"./checkpoint/model.ckpt"))
device = torch.device('cpu')
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

pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='macro')
print("Precision (macro): {:.2%}".format(pre))
print("   Recall (macro): {:.2%}".format(recall))
print("       F1 (macro): {:.2%}".format(f1))
pre, recall, f1, _ = precision_recall_fscore_support(test_y_true, test_y_pred, average='micro')
print("       F1 (micro): {:.2%}".format(f1))
