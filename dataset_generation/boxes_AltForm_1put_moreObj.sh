python generate_boxes_data_modified.py \
    --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
    --output_dir "../data/boxes_AltForm_1put_moreObj" \
    --alternative_forms "always" \
    --num_samples 50000 \
    --allowed_operations "put" \
    --num_operations 1 \
    --fix_object_count_per_phrase 1 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 4