# The Study of Prompt Based Learning for Chinese Fact Checking
* Claim Verification: We survey the prompt tuning and parameter efficient fine-tuning methods to implement on chinese fact checking  
* Template Engineering: We generate the prompt by APE
## Generate prompts by APE?
    python APE.py
## Claim Verification  
* P-Tuning   
## Basic Usage
    python main.py 
## Parameters
`--train_file` default='datasets/preprocessed/train.json'  
`--valid_file` default='datasets/preprocessed/dev.json'  
`--test_file` default='datasets/preprocessed/test.json'  
`--plm` default='bert'  
`--template` default=0 
( 0: manual / 1: Ptuning (soft APE) / 2: Ptuning (soft: 10 + APE))  
`--verbalizer` default=0 (manual verbalizer)
