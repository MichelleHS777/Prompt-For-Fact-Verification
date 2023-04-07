# ========================================
#               BERT
# ========================================
#python3 main.py --template=0 --verbalizer=0 --plm='bert' --train_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_train.json' --valid_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_dev.json' --test_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_test.json' > ./results/230405/evidence/prompt/prompt_manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0 --plm='bert' --train_file='../Evidence Extraction/datasets/evidence/semantic/semantic_train.json' --valid_file='../Evidence Extraction/datasets/evidence/semantic/semantic_dev.json' --test_file='../Evidence Extraction/datasets/evidence/semantic/semantic_test.json' > ./results/230405/evidence/semantic/semantic_manualTemplate_manualVerbalizer1.log
python3 main.py --template=0 --verbalizer=0 --plm='bert' --train_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_train2.json' --valid_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_dev2.json' --test_file='../Evidence Extraction/datasets/evidence/promptbert/prompt_test2.json' > ./results/230405/evidence/prompt/prompt_SEP_manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/train.json' --valid_file='datasets/evidences selection/dev.json' --test_file='datasets/evidences selection/test.json' > ./results/manualTemplate_manualVerbalizer3.log
#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/train.json' --valid_file='datasets/evidences selection/dev.json' --test_file='datasets/evidences selection/test.json' > ./results/manualTemplate_manualVerbalizer4.log
#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/train.json' --valid_file='datasets/evidences selection/dev.json' --test_file='datasets/evidences selection/test.json' > ./results/manualTemplate_manualVerbalizer5.log

#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/tfidf_train.json' --valid_file='datasets/evidences selection/tfidf_dev.json' --test_file='datasets/evidences selection/tfidf_test.json' > ./results/230321/evidences/tfidf_manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/semantic_train.json' --valid_file='datasets/evidences selection/semantic_dev.json' --test_file='datasets/evidences selection/semantic_test.json' > ./results/230321/evidences/semantic_manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0 --train_file='datasets/evidences selection/prompt_train.json' --valid_file='datasets/evidences selection/prompt_dev.json' --test_file='datasets/evidences selection/prompt_test.json' > ./results/230321/evidences/prompt_manualTemplate_manualVerbalizer1.log

#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_nei.json'  > ./results/manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_nei.json' > ./results/manualTemplate_manualVerbalizer2.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_nei.json' > ./results/manualTemplate_manualVerbalizer3.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_nei.json' > ./results/manualTemplate_manualVerbalizer4.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_nei.json' > ./results/manualTemplate_manualVerbalizer5.log

#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_3224.json'  > ./results/manualTemplate_manualVerbalizer11.log
#python3 main.py --template=0 --verbalizer=0 --fake_tr  ain_file='datasets/claim verification/train_fake_noevidences_3224.json'  > ./results/manualTemplate_manualVerbalizer12.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_0.8E_0.2NE_3224.json'  > ./results/manualTemplate_manualVerbalizer13.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_E_1220.json'  > ./results/E_1220.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_NE_4000.json'  > ./results/NE_4000.log

#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_E_2000.json'  > ./results/train_fake_E_2000.log
#python3 main.py --template=0 --verbalizer=0 --fake_train_file='datasets/claim verification/train_fake_NE_2000.json'  > ./results/train_fake_NE_2000.log

#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer1.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer2.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer3.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer4.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer5.log

#python3 main.py --template=5 --verbalizer=0  > ./results/mixTemplate_manualVerbalizer1.log
#python3 main.py --template=5 --verbalizer=0  > ./results/mixTemplate_manualVerbalizer2.log
#python3 main.py --template=5 --verbalizer=0  > ./results/mixTemplate_manualVerbalizer3.log
#python3 main.py --template=5 --verbalizer=0  > ./results/mixTemplate_manualVerbalizer4.log
#python3 main.py --template=5 --verbalizer=0  > ./results/mixTemplate_manualVerbalizer5.log
#
#python3 main.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer1.log
#python3 main.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer2.log
#python3 main.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer3.log
#python3 main.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer4.log
#python3 main.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer5.log

#python3 main.py --template=2  --freeze=False > ./results/230329/P-tuning/P-Tuning1.log
#python3 main.py --template=2  --freeze=False > ./results/230329/P-tuning/P-Tuning2.log
#python3 main.py --template=2  --freeze=False > ./results/230329/P-tuning/P-Tuning3.log
#python3 main.py --template=2  --freeze=False > ./results/230329/P-tuning/P-Tuning4.log
#python3 main.py --template=2  --freeze=False > ./results/230329/P-tuning/P-Tuning5.log

#python3 main.py --template=2  --freeze=False --plm='roberta' > ./results/230405/P-tuning/RoBERTa/template2/RoBERTa_P-Tuning1.log
#python3 main.py --template=2  --freeze=False --plm='roberta' > ./results/230405/P-tuning/RoBERTa/template2/RoBERTa_P-Tuning2.log
#python3 main.py --template=2  --freeze=False --plm='roberta' > ./results/230405/P-tuning/RoBERTa/template2/RoBERTa_P-Tuning3.log
#python3 main.py --template=2  --freeze=False --plm='roberta' > ./results/230405/P-tuning/RoBERTa/template2/RoBERTa_P-Tuning4.log
#python3 main.py --template=2  --freeze=False --plm='roberta' > ./results/230405/P-tuning/RoBERTa/template2/RoBERTa_P-Tuning5.log

# ========================================
#               GPT-2
# ========================================
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0   > ./results/230321/0-shot/0shot_manualTemplate_manualVerbalizer1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0   > ./results/230321/0-shot/0shot_manualTemplate_manualVerbalizer2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0  > ./results/230321/0-shot/0shot_manualTemplate_manualVerbalizer3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0  > ./results/230321/0-shot/0shot_manualTemplate_manualVerbalizer4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0  > ./results/230321/0-shot/0shot_manualTemplate_manualVerbalizer5.log

#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --shot --template=0 --verbalizer=0 --shot_num=1  > ./results/230321/1shot_manualTemplate_manualVerbalizer1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --shot --template=0 --verbalizer=0 --shot_num=1  > ./results/230321/1shot_manualTemplate_manualVerbalizer2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --shot --template=0 --verbalizer=0 --shot_num=1 > ./results/230321/1shot_manualTemplate_manualVerbalizer3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --shot --template=0 --verbalizer=0 --shot_num=1 > ./results/230321/1shot_manualTemplate_manualVerbalizer4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --shot --template=0 --verbalizer=0 --shot_num=1 > ./results/230321/1shot_manualTemplate_manualVerbalizer5.log

#python3 main_gpt2.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer1.log
#python3 main_gpt2.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer2.log
#python3 main_gpt2.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer3.log
#python3 main_gpt2.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer4.log
#python3 main_gpt2.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer5.log

#python3 main_gpt2.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer1.log
#python3 main_gpt2.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer2.log
#python3 main_gpt2.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer3.log
#python3 main_gpt2.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer4.log
#python3 main_gpt2.py --template=5 --verbalizer=1  > ./results/mixTemplate_softVerbalizer5.log

#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=2 --freeze=False > ./results/P-Tuning1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=2 --freeze=False > ./results/P-Tuning2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=2 --freeze=False > ./results/P-Tuning3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=2 --freeze=False > ./results/P-Tuning4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=2 --freeze=False > ./results/P-Tuning5.log

# ========================================
#               RoBERTa
# ========================================
#python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer1.log
#python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer2.log
#python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer3.log
#python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer4.log
#python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer5.log


#python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer1.log
#python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer2.log
#python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer3.log
#python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer4.log
#python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer5.log

#python3 main_roberta.py --template=3 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer1.log
#python3 main_roberta.py --template=3 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer2.log
#python3 main_roberta.py --template=3 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer3.log
#python3 main_roberta.py --template=3 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer4.log
#python3 main_roberta.py --template=3 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer5.log

#python3 main_roberta.py --template=3 --verbalizer=0  > ./results/roberta/mixTemplate_manualVerbalizer1.log
#python3 main_roberta.py --template=3 --verbalizer=0  > ./results/roberta/mixTemplate_manualVerbalizer2.log
#python3 main_roberta.py --template=3 --verbalizer=0  > ./results/roberta/mixTemplate_manualVerbalizer3.log
#python3 main_roberta.py --template=3 --verbalizer=0  > ./results/roberta/mixTemplate_manualVerbalizer4.log
#python3 main_roberta.py --template=3 --verbalizer=0  > ./results/roberta/mixTemplate_manualVerbalizer5.log


