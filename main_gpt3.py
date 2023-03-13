import os
import random

import openai
import json


openai.api_key = "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"

classifications = ["0", "1", "2"] # 可能的類別
prompt = "請問"


train_dataset = open("datasets/claim verification/train.json", 'r', encoding='utf-8').readlines()
test_dataset = open("datasets/claim verification/test.json", 'r', encoding='utf-8').readlines()
save_file = open("results/230315/GPT-3/test_0shot.json", "w", encoding="utf-8")

# train_example = ''
# train_sample = random.sample(train_dataset, 1)
# for data in train_sample:
#     data = eval(data)
#     train_example += "根據「" + data['evidences'] + "」" + \
#          "如果0代表「正確」、1代表「錯誤」、2代表「未知」" + \
#          prompt + "「" + data['claim'] + \
#          "」請填寫答案為".join(classifications) \
#          + '\n答案:'+ str(data['label']) + '\n\n'
# print(train_example)

for data in test_dataset:
  data = eval(data)
  text =  "根據「" + data['evidences'] + "」" + \
         "如果0代表「正確」、1代表「錯誤」、2代表「未知」" + \
         prompt + "「" + data['claim'] + \
         "」請填寫答案為".join(classifications) \
         + '\n答案:'
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=text,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )

  label = response.choices[0].text
  result = json.dumps({"claimId":data['claimId'],
                       "claim":data['claim'], "label":label}, ensure_ascii=False)
  save_file.write(result + '\n')
  print(result)
