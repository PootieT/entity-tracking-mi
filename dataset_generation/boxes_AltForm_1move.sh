python generate_boxes_data_modified.py \
    --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
    --output_dir "../data/boxes_AltForm_1move" \
    --alternative_forms "always" \
    --num_samples 5000 \
    --allowed_operations "move" \
    --num_operations 1