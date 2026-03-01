python pipelines/generates_boxes_data_modified.py \
  --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
  --output_dir "../data/boxes_AltForm_default" \
  --alternative_forms "always" \
  --num_samples 10000 \
  --num_operations 12