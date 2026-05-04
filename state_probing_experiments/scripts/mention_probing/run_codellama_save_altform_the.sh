

module load miniconda
module load cuda/11.8

conda activate box-models

echo "Probing layer ${1}"

python train_probe_LLM.py \
    --model_type llama \
    --exp_name mention \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path /projectnb/tin-lab/sebastian/box-models/checkpoints/CodeLlama-13b-hf \
    --layer $1 \
    --epo 64 \
    --binary_probe \
    --exclude_empty \
    --condition_on the \
    --checkpoint_root probe_checkpoints/codellama-13b/mention_the \
    --save_model_representation \
    --model_representation_path representations/codellama-13b/exclude_empty_conditioned_on_the \
    --subsample 