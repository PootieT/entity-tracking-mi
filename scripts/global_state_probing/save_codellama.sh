#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_memory=80G
#$ -o logs/$JOB_ID_global_13b.log
#$ -j y
#$ -m e
#$ -M zhaoqiao@bu.edu

module load miniconda
module load cuda/11.8

conda activate <your_env>

export WANDB_PROJECT=entity-tracking-probing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# API keys and caching dirs
export HF_HOME="<your hf home>"
export HF_TOKEN="<your hf token>"
export NDIF_APIKEY="<your NDIF apikey>" # only if you are running large model on remote

#export CUDA_LAUNCH_BLOCKING=1
index=$(($SGE_TASK_ID-1))
ROOT=$(pwd)/probe_experiments

python probe_experiments/train_probe.py \
    --model_type llama \
    --exp_name global \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path /projectnb/tin-lab/sebastian/box-models/checkpoints/CodeLlama-13b-hf \
    --layer 40 \
    --epo 64 \
    --condition_on the \
    --checkpoint_root probe_checkpoints/codellama-13b/global_the \
    --save_model_representation \
    --model_representation_path representations/codellama-13b/include_empty_conditioned_on_the \
    --dataset_subset \