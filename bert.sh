#!/bin/sh
#BSUB -q gpuq
#BSUB -o identification.out
#BSUB -e identification.err
#BSUB -n 1
#BSUB -R span[ptile=1]   
#BSUB -J bert
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=24]"
#BSUB -a python
nvidia-smi
 
CURDIR=$PWD
cd $CURDIR
python3 author_identification.py

