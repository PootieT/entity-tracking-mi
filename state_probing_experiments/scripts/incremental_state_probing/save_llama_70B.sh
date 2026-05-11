
module load miniconda
module load cuda/11.8

conda activate <your_env>
export HF_HOME="<your hf home>"


echo "Probing layer ${1}"

python train_probe_LLM.py \
    --model_type llama \
    --exp_name incremental_local_state \
    --dataset_path data/boxes_altAlways_default_maxop12_5k \
    --model_path meta-llama/Meta-Llama-3.1-70B\
    --layer $1 \
    --epo 64 \
    --binary \
    --condition_on number \
    --checkpoint_root probe_experiments/probe_checkpoints/llama3-70b/altform/incremental_local_state \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/llama3-70b/ICL_Free/Altform/incremental_local_state \
    --subsample \
    --remote \
    --save_activations_ckpts \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \
