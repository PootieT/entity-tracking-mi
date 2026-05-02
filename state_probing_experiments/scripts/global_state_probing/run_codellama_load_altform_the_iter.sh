

module load miniconda
module load cuda/11.8

conda activate box-models

for layer in $(seq 1 41);
do 
    echo --------PROBING LAYER $layer--------
    python train_probe_LLM.py \
        --model_type llama \
        --exp_name global \
        --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
        --model_path codellama/CodeLlama-13b-hf \
        --layer $layer \
        --epo 64 \
        --condition_on the \
        --checkpoint_root probe_checkpoints/codellama-13b/global_the \
        --load_model_representation \
        --model_representation_path representations/codellama-13b/include_empty_conditioned_on_the \
        --subsample  \
        --object_vocabulary_file data/llama_friendly_objects.csv \
        
done