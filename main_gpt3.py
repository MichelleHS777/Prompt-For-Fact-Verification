import os
import random

import openai
import json

openai.organization = 'org-KqFL3LbeGGyb4RcMCPBCR8kd'
openai.api_key = "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"

classifications = ["0", "1", "2"] # 可能的類別
prompt = "請問"


train_dataset = open("datasets/claim verification/train.json", 'r', encoding='utf-8').readlines()
test_dataset = open("datasets/claim verification/test.json", 'r', encoding='utf-8').readlines()
save_file = open("results/230315/GPT-3/test_0shot.json", "w", encoding="utf-8")

def get_example():
    train_example = ''
    train_sample = random.sample(train_dataset, 5)
    for data in train_sample:
        data = eval(data)
        train_example += "根據「" + data['evidences'] + "」" + \
             "\n如果0代表「正確」、1代表「錯誤」、2代表「未知」\n" + \
             prompt + "「" + data['claim'] + \
             "」的答案是" + classifications[0] + '、' + classifications[1] +  \
             "還是" + classifications[2] + '呢?\n答案:'+ str(data['label']) + '\n\n'
    return train_example
    # print(train_example)

for data in test_dataset:
  data = eval(data)
  # train_example = get_example()
  text = "根據「" + data['evidences'] + "」" + \
         "\n如果0代表「正確」、1代表「錯誤」、2代表「未知」\n" + \
         prompt + "「" + data['claim'] + \
         "」的答案是" + classifications[0] + '、' + classifications[1] +  \
         "還是" + classifications[2] + '呢?\n答案:'

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
  result = json.dumps({"claimId":data['claimId'], "claim":data['claim'], "label":label}, ensure_ascii=False)
  save_file.write(result + '\n')
  print(result)
