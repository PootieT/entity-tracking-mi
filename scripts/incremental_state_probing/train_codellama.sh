#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -l h_rt=24:00:00
#$ -o logs/$JOB_ID_incremental_codellama-13b.log
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
for layer in {1..40}
do

    python probe_experiments/train_probe.py \
        --model_type llama \
        --exp_name incremental_local_state \
        --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
        --model_path codellama/CodeLlama-13b-hf \
        --layer $layer \
        --epo 64 \
        --binary_probe \
        --condition_on number \
        --checkpoint_root probe_experiments/probe_checkpoints/codellama-13b/incremental_local_state \
        --save_model_representation \
        --model_representation_path probe_experiments/representations/codellama-13b/incremental_local_state \
        --dataset_subset \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \
        --incremental_local_state \

done
