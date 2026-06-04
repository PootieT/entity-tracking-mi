#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=7.0
#$ -o logs/$JOB_ID_mention_70b.log
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


python train_probe_LLM.py \
    --model_type llama \
    --exp_name mention \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path meta-llama/Meta-Llama-3.1-70B\
    --layer 80\
    --epo 64 \
    --binary \
    --exclude_empty \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/llama3-70b/mention_probing \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/llama3-70b/exclude_empty_conditioned_on_the \
    --subsample \
    --remote \
    --act_batch_size 16 \
    --object_vocabulary_file data/llama_friendly_objects.csv \
