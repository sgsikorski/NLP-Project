#!/bin/bash -l                                                                                                                                              

#SBATCH --ntasks=1
#SBATCH -A grad-nlp-cs6501
#SBATCH -t 1:00:00 # Hour:Minute:Second
#SBATCH -p gpu # Partation type
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB # CPU memory

export TRANSFORMERS_CACHE="/scratch/nqj5ak/huggingface/hub"
export HF_HOME="/scratch/nqj5ak/huggingface/hub"
export HF_DATASETS_CACHE="/scratch/nqj5ak/huggingface/datasets"

# Load the modules
module load anaconda cuda cudnn

# Activate the virtual Python environment
conda activate test

# Run the test code                                                                                                                                         
python main.py

# Or, run the training part using
# python demo.py --task train