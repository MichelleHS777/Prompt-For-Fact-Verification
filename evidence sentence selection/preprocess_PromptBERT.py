def convert_json_to_data(original_file, save_file):
    for data in original_file:
        data = eval(data)
        for sentence in data['evidence']:
            save_file.write(data['claim']+"\t"+sentence+"\t"+str(data['label'])+'\n')
    save_file.close()

print('preprocess...')
train_dataset = open('datasets/evidence extraction/train.json', 'r', encoding='utf-8').readlines()
save_file = open('datasets/train.data', 'w', encoding='utf-8')
convert_json_to_data(train_dataset, save_file)

dev_dataset = open('datasets/evidence extraction/dev.json', 'r', encoding='utf-8').readlines()
save_file = open('datasets/dev.data', 'w', encoding='utf-8')
convert_json_to_data(dev_dataset, save_file)

test_dataset = open('datasets/evidence extraction/test.json', 'r', encoding='utf-8').readlines()
save_file = open('datasets/test.data', 'w', encoding='utf-8')
convert_json_to_data(test_dataset, save_file)
print('finish...')
