import openai
import json

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Generate Prompts/Templates')
parser.add_argument('--dataset', type=str, default=None, help='dataset path')
args = parser.parse_args()

# set dataset path
dataset = json.load(open('PATH/TO/DATASET', 'r', encoding='utf-8'))
# set openai api key
openai.api_key = 'KEY'

# First, let's define a simple dataset consisting of claim and label
label_mapping = {0:'支持',1:'反对',2:'信息不足'}
claim_evidence = []
for i in range(len(dataset)):
    claim = dataset[i]['claim']
    evidences = [dataset[i]['gold evidence'][str(j)]['text'] for j in range(1)] # get one evidence
    evidences = ''.join(evidences) 
    claim_evidence_text = '主张: ' + claim + '证据: ' + evidences
    claim_evidence.append(claim_evidence_text)
label = [label_mapping[dataset[i]['label']] for i in range(20)] # get labels from 20 samples

# Now, we need to define the format of the prompt that we are using.
eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

# Now, let's use APE to find prompts that generate antonyms for each word.
from automatic_prompt_engineer import ape

result, demo_fn = ape.simple_ape(
    dataset=(claim_evidence, label),
    eval_template=eval_template,
)

# Let's see the results.
print(result)

# if you want to evalauate the score of Prompts
# from automatic_prompt_engineer import ape

# manual_prompt = "要評估輸入和證據的結果是支持或反對，或者信息不足"

# human_result = ape.simple_eval(
#     dataset=(claim_evidence, label),
#     eval_template=eval_template,
#     prompts=[manual_prompt],
# )
# print(human_result)
