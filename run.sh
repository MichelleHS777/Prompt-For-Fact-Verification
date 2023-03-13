# ========================================
#               BERT
# ========================================
#python3 main.py --template=0 --verbalizer=0  > ./results/manualTemplate_manualVerbalizer1.log
#python3 main.py --template=0 --verbalizer=0  > ./results/manualTemplate_manualVerbalizer2.log
#python3 main.py --template=0 --verbalizer=0  > ./results/manualTemplate_manualVerbalizer3.log
#python3 main.py --template=0 --verbalizer=0  > ./results/manualTemplate_manualVerbalizer4.log
#python3 main.py --template=0 --verbalizer=0  > ./results/manualTemplate_manualVerbalizer5.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer1.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer2.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer3.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer4.log
#python3 main.py --template=0 --verbalizer=1  > ./results/manualTemplate_softVerbalizer5.log
#
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

#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning5.log

# ========================================
#               GPT-2
# ========================================
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt.py --template=0 --verbalizer=0 --shot_num=16  > ./results/0shot_manualTemplate_manualVerbalizer.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=16  > ./results/0-shot/0shot_manualTemplate_manualVerbalizer2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=16 > ./results/0-shot/0shot_manualTemplate_manualVerbalizer3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=16 > ./results/0-shot/0shot_manualTemplate_manualVerbalizer4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=16 > ./results/0-shot/0shot_manualTemplate_manualVerbalizer5.log

#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=64  > ./results/64shot_manualTemplate_manualVerbalizer1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=64  > ./results/64shot_manualTemplate_manualVerbalizer2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=64 > ./results/64shot_manualTemplate_manualVerbalizer3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=64 > ./results/64shot_manualTemplate_manualVerbalizer4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main_gpt2.py --template=0 --verbalizer=0 --shot_num=64 > ./results/64shot_manualTemplate_manualVerbalizer5.log

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
python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer.log
python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer2.log
python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer3.log
python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer4.log
python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer5.log

python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer1.log
python3 main_roberta.py --template=0 --verbalizer=0   > ./results/roberta/manualTemplate_manualVerbalizer2.log
python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer3.log
python3 main_roberta.py --template=0 --verbalizer=0  > ./results/roberta/manualTemplate_manualVerbalizer4.log python3 main_roberta.py --template=0 --verbalizer=0 --shot_num=64 > ./results/64shot_manualTemplate_manualVerbalizer5.log

python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer1.log
python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer2.log
python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer3.log
python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer4.log
python3 main_roberta.py --template=0 --verbalizer=1  > ./results/roberta/manualTemplate_softVerbalizer5.log

python3 main_roberta.py --template=5 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer1.log
python3 main_roberta.py --template=5 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer2.log
python3 main_roberta.py --template=5 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer3.log
python3 main_roberta.py --template=5 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer4.log
python3 main_roberta.py --template=5 --verbalizer=1  > ./results/roberta/mixTemplate_softVerbalizer5.log

python3 ../BERT_Classsfication/train.py > ./results/roberta/finetune1.log
python3 ../BERT_Classsfication/train.py > ./results/roberta/finetune2.log
python3 ../BERT_Classsfication/train.py > ./results/roberta/finetune3.log
python3 ../BERT_Classsfication/train.py > ./results/roberta/finetune4.log
python3 ../BERT_Classsfication/train.py > ./results/roberta/finetune5.log

