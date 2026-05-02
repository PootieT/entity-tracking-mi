#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=80G
#$ -l h_rt=24:00:00
#$ -o logs/probes_codellama/ICL_free_incremental_local_state/$JOB_ID_train_codellama-13b.log
#$ -j y
#$ -m e
#$ -M zhaoqiao@bu.edu

module load miniconda
module load cuda/11.8

conda activate box-models

echo "Probing layer ${1}"

python train_probe_LLM.py \
    --model_type llama \
    --exp_name incremental_local_state \
    --dataset_path data/boxes-dataset-v1/few_shot_boxes_nso_exp2_max3 \
    --model_path //projectnb/tin-lab/qazhao/ICML26/entity-tracking-gemma/data/boxes_altAlways_default_maxop12_5k \
    --layer $1 \
    --epo 32 \
    --learning_rate 1e-3 \
    --binary_probe \
    --condition_on number \
    --checkpoint_root probe_checkpoints/codellama-13b/incremental_local_state \
    --save_model_representation \
    --model_representation_path representations/codellama-13b/ICL_Free/Altform/incremental_local_state \
    --subsample \
    