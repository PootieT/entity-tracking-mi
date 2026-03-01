python generate_boxes_data_modified.py \
    --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
    --output_dir "../data/boxes_altAlways_1remove" \
    --alternative_forms "always" \
    --num_samples 7000 \
    --allowed_operations "remove" \
    --num_operations 1