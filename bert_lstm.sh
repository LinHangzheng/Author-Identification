#!/bin/sh
#BSUB -q gpuq
#BSUB -o identification_bertlstm.out
#BSUB -e identification_bertlstm.err
#BSUB -n 1
#BSUB -J bert
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=2]"
#BSUB -a python
nvidia-smi
 
CURDIR=$PWD
cd $CURDIR
python3 bert_lstm/bert_lstm.py

