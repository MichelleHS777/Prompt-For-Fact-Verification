python main.py --eval --test_data_path='./data/CHEF/train_evidence.json' --save_file='result/0329/prompt_train.json'
python main.py --eval --test_data_path='./data/CHEF/dev_evidence.json' --save_file='result/0329/prompt_dev.json'
#python main.py --eval --test_data_path='./data/unpreprocess/test.json' --save_file='result/0322/prompt_test.json'

#python Surface_Ranker.py  --test_data_path='./data/unpreprocess/train.json' --save_file='result/0322/tfidf_train.json'
#python Surface_Ranker.py  --test_data_path='./data/unpreprocess/dev.json' --save_file='result/0322/tfidf_dev.json'
#python Surface_Ranker.py  --test_data_path='./data/unpreprocess/test.json' --save_file='result/0322/tfidf_test.json'

python Semantic_Ranker.py  --test_data_path='./data/CHEF/train_evidence.json' --save_file='result/0329/semantic_train.json'
python Semantic_Ranker.py  --test_data_path='./data/CHEF/dev_evidence.json' --save_file='result/0329/semantic_dev.json'
#python Semantic_Ranker.py  --test_data_path='./data/unpreprocess/test.json' --save_file='result/0322/semantic_test.json'