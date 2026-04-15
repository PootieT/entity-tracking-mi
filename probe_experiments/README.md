# entity-tracking-probing
probing experiments to understand entity tracking task

# Local / Global / Mention Probes  

To cache model residual streams (last token):
```commandline
./probe_experiments/cache_representation_codellama13b.qsub  # using 2GPUs local
./probe_experiments/cache_representation_llama3_70B.qsub  # using NDIF
```

To train Local Probes

```commandline

```

To train Global probes

```commandline

```

To train mention probes

```commandline

```

# Prior State Probes

### Script
To cache hidden states
```commandline
./scripts/cache_ternary_probe_activations_codellama13b.qsub  # local 2GPU
./scripts/cache_ternary_probe_activations_llama3_70B # NDIF remote
```

To train probe
```commandline
./scripts/load_and_train_probe_llama3_70B.qsub  # for llama405b
./scripts/load_and_train_probe_codellama13b.qsub  # for codellama 13b
./scripts/load_and_train_probe_codellama13b_moveContent.qsub  # for codellama13b with moveContent split
```



# Remove Mechanism
## Ternary Probe Training

In my code, I refer to these ternary probes as `phrase probe`.
### Script
To cache the model representation, see
```commandline
./scripts/cache_codellama13b_phrase_probe_activations.qsub
./scripts/cache_gpt2_phrase_probe_activations.qsub
```
some important arguments here are
- `condition_on`: which token hidden states to condition the probe on: 
  - `object_all_local`: condition on object, local states
  - `number_all_local`: condition on box_id (in code I often refer to as `number`), local states
  - `number_all_cumulative`: condition on box_id, global states. When caching, this and `number_all_local` results in the same cache, so just use `number_all_local` when caching

For `codellama13b`, make sure to use 2gpu torch run distributed w/ 16bit. (8bit cache does not result in good probes).
You will also notice qsub this with `#$ -pe omp 28`, this is needed because we are storing a lot of hidden states needs lots of memories.

Now to load and train the probes, see
```commandline
./scripts/load_and_train_phrase_probe_codellama13b.qsub
```

Since we need to train #layers amount of probes, for `codellama13b` I usually submit 4 jobs, each for-looping 10 probes (each probe takes around 20-30min to train)
but customize however you want

### Data
The training data used here is in `/projectnb/mcnet/peter/entity-tracking-gemma/data/boxes_altAlways_default_maxop12_5k`.
And specifically training uses `train-gpt.jsonl` and test uses `test-subsample-states-gpt.jsonl`. We need full train split
because the class label is very imbalanced with 700 probes.

## Intervention with Ternary Probes
Before running intervention, we need to run baseline model inference to 1) get model behavioral accuracy and 2) get 
examples where model succeeds. The most important scripts are 
```commandline
./scripts/intervene_phrase_probe_codellama13b_8bit_null_1put.qsub  # null the 1 put operation in query box
./scripts/intervene_phrase_probe_codellama13b_8bit_null_1remove.qsub  # null the 1 remove operation in query box
./scripts/intervene_phrase_probe_codellama13b_8bit_null_1remove_put_globally_removed.qsub  # for specific dataset with putting globally removed object in
```

since we are only doing 100 examples in most cases, these should be <10 min each run/layer


# Utilities that maybe helpful

## Anything Data format related
Conversion between json/tsv, subsampling, checkout 
```commandline
./entity-tracking-probing/utils/*
```
Plotting/analysis scripts
```commandline
entity-tracking-probing/*
```
run them like `python -m src.analysis.plot_phrase_probe_results`

#### Font issue in plots
If you have issue with `Times New Roman` font when plotting (which I 
had to deal with because SCC doesn't have it), download the font .ttf file
from [here](https://github.com/justrajdeep/fonts/blob/master/Times%20New%20Roman.ttf)

`scp` or `rsync` this file to your SCC home directory
```commandline
rsync -ravz ~/Downloads/Times\ New\ Roman.ttf username@scc1.bu.edu:your/home/dir/.local/share/fonts/
```
Delete `matplotlib` font cache:
```commandline
rm /your/home/dir/.cache/matplotlib
```
re-start python and run this to rebuild cache, and it should work.
```
>>> import matplotlib.font_manager as fm
>>> [f.name for f in fm.fontManager.ttflist if "Times" in f.name]
['Times New Roman']
```
