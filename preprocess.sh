#BSUB -q gpuq
#BSUB -o preprocess.out
#BSUB -e preprocess.err
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=24]"
#BSUB -R span[ptile=2]
#BSUB -a python
 
CURDIR=$PWD
cd $CURDIR
python preprocess.py

