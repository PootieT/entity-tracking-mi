

module load miniconda
module load cuda/11.8

conda activate <your_env>
export HF_HOME="<your hf home>"


echo "Probing layer ${1}"

python train_probe_LLM.py \
    --model_type llama \
    --exp_name binary \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path meta-llama/Meta-Llama-3.1-70B\
    --layer $layer \
    --epo 64 \
    --binary \
    --exclude_empty \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/llama3-70b/binray_probing \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/llama3-70b/exclude_empty_conditioned_on_the \
    --subsample \
    --remote \
    --act_batch_size 16 \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \
