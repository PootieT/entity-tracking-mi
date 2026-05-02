
module load miniconda
module load cuda/11.8

conda activate box-models


for layer in $(seq 1 81);
do 
    echo --------PROBING LAYER $layer--------
    python train_probe_LLM.py \
        --model_type llama \
        --exp_name incremental_local_state \
        --dataset_path /projectnb/tin-lab/qazhao/ICML26/entity-tracking-gemma/data/boxes_altAlways_default_maxop12_5k \
        --model_path meta-llama/Meta-Llama-3.1-70B\
        --layer $layer \
        --epo 64 \
        --binary \
        --condition_on number \
        --checkpoint_root probe_checkpoints/llama3-70b/altform/incremental_local_state \
        --load_model_representation \
        --model_representation_path representations/llama3-70b/ICL_Free/Altform/incremental_local_state \
        --subsample \
        --remote \
        --save_activations_ckpts \
        --object_vocabulary_file /projectnb/tin-lab/qazhao/ICML26/entity-tracking-gemma/data/objects/llama_friendly_objects.csv \

done