# Box Data Generation

The script `generate_boxes_data_modified.py` is used to generate all data used in the experiments.
It is modified from the original data generation code from [Kim and Schuster (2023)](https://github.com/sebschu/entity-tracking-lms/tree/main) 
with following changes:
* Initial State for all boxes is non-empty.

Each script name corresponds to the dataset name (in `/data`), which are listed in the paper Appendix 
(i.e. `boxes_AltForm_default.sh` generates `./data/boxes_AltForm_default`)

Dataset `AltForm_remove_invalid` and `AltForm_remove_duplicate` are generate separately using TODO.

A few of the existing datasets can be downloaded from this [Google Drive Link](https://drive.google.com/drive/folders/1UY6odU1hj-j7raBUwlOK2O4IYQS0R2NW?usp=share_link), (generating it is often quicker!)