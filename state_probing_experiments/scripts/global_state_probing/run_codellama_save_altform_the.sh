

module load miniconda
module load cuda/11.8

conda activate <your_env>
export HF_HOME="<your hf home>"

echo "Probing layer ${1}"

python train_probe_LLM.py \
    --model_type llama \
    --exp_name global \
    --dataset_path data/boxes_altAlways_default_maxop12_5k \
    --model_path codellama/CodeLlama-13b-hf \
    --layer $1 \
    --epo 64 \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/codellama-13b/global_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/codellama-13b/include_empty_conditioned_on_the \
    --subsample 
