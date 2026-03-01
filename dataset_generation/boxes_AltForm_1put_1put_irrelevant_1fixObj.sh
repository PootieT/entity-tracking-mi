python generate_boxes_data_modified.py \
    --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
    --output_dir "../data/boxes_AltForm_1put_1put_irrelevant_1fixObj" \
    --alternative_forms "always" \
    --num_samples 5000 \
    --allowed_operations "put" \
    --num_operations 2 \
    --fix_object_count_per_phrase 1