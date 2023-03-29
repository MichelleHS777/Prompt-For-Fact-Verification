import argparse


def set_args():
    parser = argparse.ArgumentParser('--PromptBert')
    parser.add_argument('--train', action="store_true", help="If train or not")
    parser.add_argument('--eval', action="store_true", help='If evaluate or not')
    parser.add_argument('--train_data_path', default='./datasets/CHEF/train.data', type=str, help='训练数据集')
    parser.add_argument('--test_data_path', default='./datasets/CHEF/test_evidence.json', type=str, help='测试数据集')
    parser.add_argument('--save_file', default='result/0329/semantic_test.json', type=str, help='save file path')
    parser.add_argument('--max_len', default=512, type=int, help='句子的最大长度')
    parser.add_argument('--train_batch_size', default=4, type=int, help='训练批次的大小')
    parser.add_argument('--dev_batch_size', default=4, type=int, help='训练批次的大小')
    parser.add_argument('--num_train_epochs', default=3, type=int, help='训练几轮')
    parser.add_argument('--learning_rate', default=9e-6, type=float, help='学习率大小')
    parser.add_argument('--bert_pretrain_path', default='./roberta_pretrain', type=str, help='预训练模型路径')
    parser.add_argument('--output_dir', default='./checkpoint', type=str, help='模型输出目录')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚的大小')
    parser.add_argument('--model', default='./checkpoint/PromptBERT_CHEF.h5', type=str, help='save file path')
    # parser.add_argument('--train_file', default='data/CHEF/train_evidence.json', type=str, help='训练数据集')
    # parser.add_argument('--valid_file', default='data/CHEF/dev_evidence.json', type=str, help='驗證数据集')
    # parser.add_argument('--test_file', default='data/CHEF/test_evidence.json', type=str, help='測試数据集')
    # parser.add_argument('--eval_dev_data_path', default='./data/CHEF/dev.data', type=str, help='测试数据集')
    return parser.parse_args()
