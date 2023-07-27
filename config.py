import argparse


def set_args():
    parser = argparse.ArgumentParser(description='Prompt Tuning For CHEF')
    parser.add_argument('--cuda', type=str, default="0", help='appoint GPU devices')
    parser.add_argument('--data_aug', action="store_true", help='If data augmentation or not')
    parser.add_argument('--train', action="store_true", help='If train or not')
    parser.add_argument('--eval', action="store_true", help='If evaluate or not')
    parser.add_argument('--train_file', type=str, default='datasets/preprocessed/train.json', help='train dataset path')
    parser.add_argument('--fake_train_file', type=str, default='datasets/preprocessed/train_fake_SUP_NE_1000.json', help='fake nei dataset for training')
    parser.add_argument('--valid_file', type=str, default='datasets/preprocessed/dev.json', help='validation dataset path')
    parser.add_argument('--test_file', type=str, default='datasets/preprocessed/test.json', help='test dataset path')
    parser.add_argument('--plm', type=str, default='bert', help='bert, roberta, ernie')
    parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
    parser.add_argument('--max_length', type=int, default=512,
                        help='max token length of the sentence for bert tokenizer')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
    parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
    parser.add_argument('--epochs', type=int, default=8, help='training epochs for labeled data')
    parser.add_argument('--patience', type=int, default=5, help="Early stopping")
    parser.add_argument('--shot_num', type=int, default=1, help='few shot numbers')
    parser.add_argument('--shot', action="store_true", help="do one-shot or few shot")
    parser.add_argument('--freeze', action="store_true", help='set if freeze plm parameters')
    parser.add_argument('--soft_token_num', type=int, default=10, help='set the number of soft token')
    parser.add_argument('--plm_eval_mode', action="store_true", help='the dropout of the model is turned off')
    parser.add_argument("--template", type=int,
                        help="Set template (0 for manual, 1 for soft, 2 for Ptuning, 3 for PrefixTuning, 4 for PTR, 5 for mix)",
                        default=0)
    parser.add_argument("--verbalizer", type=int,
                        help="Set template (0 for manual, 1 for soft, 2 for knowledge)", default=0)
    parser.add_argument("--init_from_vocab", action="store_false")
    parser.add_argument('--model_save_dir', type=str, default='./checkpoint/', help='save model path')
    return parser.parse_args()