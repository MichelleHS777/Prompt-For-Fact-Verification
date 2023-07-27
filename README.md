# The Study of Prompt Based Learning for Chinese Fact Checking
We survey the prompt tuning and parameter efficient fine-tuning methods to implement on chinese fact checking
## How do we find proper prompt?
We use Automatic Prompt Engineer to generate the prompts
## Claim Verification  
* P-Tuning   
## Basic Usage
python main.py 

## Parameters
`--train_file` default='datasets/preprocessed/train.json'
`--valid_file` default='datasets/preprocessed/dev.json'  
`--test_file` default='datasets/preprocessed/test.json'  
`--plm` default='bert'  
`--template` default=0 (0: manual, 1: Ptuning (soft APE), 2: Ptuning (soft: 10 + APE), 3: PTuning (soft: 10) )  
`--verbalizer` default=0 (manual verbalizer)
