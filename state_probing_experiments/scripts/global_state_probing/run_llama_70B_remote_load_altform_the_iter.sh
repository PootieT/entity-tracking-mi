module load miniconda
module load cuda/11.8

conda activate box-models

for layer in $(seq 1 81);
do
    echo --------PROBING LAYER $layer--------
    python train_probe_LLM.py \
        --model_type llama \
        --exp_name global \
        --dataset_path data/boxes_altAlways_default_maxop12_5k \
        --model_path meta-llama/Meta-Llama-3.1-70B\
        --layer $layer \
        --epo 64 \
        --condition_on the \
        --checkpoint_root probe_experiments/probe_checkpoints/llama3-70b/global_probing \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/llama3-70b/include_empty_conditioned_on_the \
        --subsample \
        --remote \
        --act_batch_size 16 \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \

done
