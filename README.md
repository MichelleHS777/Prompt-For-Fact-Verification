# The Study of Prompt Based Learning for Chinese Fact Checking
* Claim Verification: We survey the prompt tuning and parameter efficient fine-tuning methods to implement on chinese fact checking  
* Template Engineering: We generate the prompt by APE
## Generate prompts by APE?
    python APE.py --dataset='PATH/TO/DATASET'
1. We get prompts by APE from 20 samples (random choose), and save the file
2. Next Implement the code, and define the dataset path    
## Preprocess
    python preprocess.py 
`--dataset` choose the dataset path you want to preprocess (default='datasets/unpreprocess/train.json')   
`--save_file` save the preprocess file (default='datasets/preprocessed/train.json')  
## Claim Verification  
* P-Tuning   
### Basic Usage
    python main.py 
### Parameters
`--train_file` default='datasets/preprocessed/train.json'  
`--valid_file` default='datasets/preprocessed/dev.json'  
`--test_file` default='datasets/preprocessed/test.json'  
`--plm` default='bert'  
`--template` default=0 
( 0: manual / 2: Ptuning (soft: 10 + APE))  
`--verbalizer` default=0 (manual verbalizer)
## Checkpoint
* [Checkpoint](https://drive.google.com/drive/folders/16XMpllhRwVSntn17gZxjqFOusbXB9Myq?usp=sharing)
