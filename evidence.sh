python3 evidence_sentnece_extraction.py --template=0 --verbalizer=0  > ./results/evidence/manualTemplate_manualVerbalizer1.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=0  > ./results/evidence/manualTemplate_manualVerbalizer2.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=0  > ./results/evidence/manualTemplate_manualVerbalizer3.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=0  > ./results/evidence/manualTemplate_manualVerbalizer4.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=0  > ./results/evidence/manualTemplate_manualVerbalizer5.log

python3 evidence_sentnece_extraction.py --template=0 --verbalizer=1  > ./results/evidence/manualTemplate_softVerbalizer1.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=1  > ./results/evidence/manualTemplate_softVerbalizer2.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=1  > ./results/evidence/manualTemplate_softVerbalizer3.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=1  > ./results/evidence/manualTemplate_softVerbalizer4.log
python3 evidence_sentnece_extraction.py --template=0 --verbalizer=1  > ./results/evidence/manualTemplate_softVerbalizer5.log

python3 evidence_sentnece_extraction.py --template=5 --verbalizer=0  > ./results/evidence/mixTemplate_manualVerbalizer1.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=0  > ./results/evidence/mixTemplate_manualVerbalizer2.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=0  > ./results/evidence/mixTemplate_manualVerbalizer3.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=0  > ./results/evidence/mixTemplate_manualVerbalizer4.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=0  > ./results/evidence/mixTemplate_manualVerbalizer5.log

python3 evidence_sentnece_extraction.py --template=5 --verbalizer=1  > ./results/evidence/mixTemplate_softVerbalizer1.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=1  > ./results/evidence/mixTemplate_softVerbalizer2.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=1  > ./results/evidence/mixTemplate_softVerbalizer3.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=1  > ./results/evidence/mixTemplate_softVerbalizer4.log
python3 evidence_sentnece_extraction.py --template=5 --verbalizer=1  > ./results/evidence/mixTemplate_softVerbalizer5.log
#
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning1.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning2.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning3.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning4.log
#CUDA_LAUNCH_BLOCKING=1 python3 main.py --template=2 --freeze=False > ./results/P-Tuning5.log