import os
# set up logging
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
import time
import csv

import numpy as np
import pandas as pd
import argparse
import torch
from src.dataset import ProbeDataLoader, LMDataloader, GPTDataloaderForInference, ObjectLocationProbeDataLoader, \
    BinaryProbeDataLoader
from src.probe_trainer import Trainer, TrainerConfig
from src.probe_model import BatteryProbeClassification, ObjectLocationProbeClassification, \
    BatteryProbeClassificationTwoLayer
import pickle



_MAX_SOURCE_TEXT_LENGTH = {
    "t5": 512,
    "gpt": 512,
    "llama": 2048
}

_MAX_TARGET_TEXT_LENGTH = 100

_INPUT_DIMENSIONS = {
    "t5": 768,
    "gpt": 1600,
    "llama": 16384

}

# make deterministic
torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--model_type",
                        required=True,
                        choices=["t5", "gpt", "llama"],
                        help="'t5', 'gpt' or 'llama' supported.")
    parser.add_argument("--dataset_path",
                        required=True,
                        type=str)
    parser.add_argument("--model_path",
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--checkpoint_root',
                        default="./probe_checkpoints", type=str)
    parser.add_argument(
        "--object_vocabulary_file",
        type=str,
        default="data/objects_with_bnc_frequency.csv",
        help='Path to a .csv file with a string field "object_names".')
    parser.add_argument('--layer',
                        required=True,
                        default=-1,
                        type=int)
    parser.add_argument('--epo',
                        default=16,
                        type=int)
    parser.add_argument('--condition_on',
                        choices=["box", "period", "the", "number", "contains"],
                        type=str,
                        dest='condition_on',
                        default='number')
    parser.add_argument('--num_prior_state',
                        required=True,
                        default=-1,
                        type=int)

    # Non-linear probe
    parser.add_argument('--mid_dim',
                        default=128,
                        type=int)
    parser.add_argument('--twolayer',
                        dest='twolayer',
                        action='store_true')
    parser.add_argument('--object_location',
                        dest='object_location',
                        action='store_true')
    parser.add_argument('--binary_probe',
                        dest='binary_probe',
                        action='store_true')
    parser.add_argument('--random',
                        dest='random',
                        action='store_true')
    parser.add_argument('--eval_only',
                        dest='eval_only',
                        action='store_true')
    parser.add_argument('--exclude_empty',
                        dest='exclude_empty',
                        action='store_true')
    parser.add_argument('--condition_on_obj',
                        default=0,
                        type=int)

    parser.add_argument('--model_representation_path',
                        default=None,
                        type=str)

    parser.add_argument('--save_model_representation',
                        dest="save_model_representation",
                        action="store_true")

    parser.add_argument('--load_model_representation',
                        dest="load_model_representation",
                        action="store_true")

    parser.add_argument('--include_prompt',
                        dest="include_prompt",
                        action="store_true")

    args, _ = parser.parse_known_args()

    if (args.condition_on not in ["number", "contains"]) and args.model_type == 't5':
        raise ValueError("--condition_on must be set to 'number' or 'contains' when training a probe on T5.")
    if args.eval_only:
        # TODO(Sebastian): debug eval_only
        raise ValueError("--eval_only is buggy, do not use for now")

    if args.exclude_empty and not args.binary_probe:
        raise ValueError("--exclude_empty only works with --binary_probe")

    if args.exclude_empty and args.condition_on not in ["contains", "the"]:
        raise ValueError("--exclude_empty can only be used with --condition_on 'contains' or 'the'")

    if args.condition_on in ["contains", "the"] and not args.exclude_empty:
        raise ValueError("--condition_on 'contains' or 'the' can only be used with --exclude_empty")

    if args.save_model_representation and args.model_representation_path is None:
        raise ValueError("--save_model_representation requires --model_representation_path to be set")

    if args.load_model_representation and args.model_representation_path is None:
        raise ValueError("--load_model_representation requires --model_representation_path to be set")

    if args.load_model_representation and args.save_model_representation:
        raise ValueError("--load_model_representation and --save_model_representation cannot be used together")

    folder_name = f"probing/state"

    if args.twolayer:
        folder_name = folder_name + f"_tl{args.mid_dim}"  # tl for probes without batchnorm
    if args.random:
        folder_name = folder_name + "_random"
    if args.object_location:
        folder_name = folder_name + "_object_location"
    if args.binary_probe:
        folder_name = folder_name + "_binary"
    if args.exclude_empty:
        folder_name = folder_name + "_exclude_empty"
    if args.condition_on_obj > 0:
        folder_name = folder_name + f"_condition_on_obj_{args.condition_on_obj}"
    if args.num_prior_state != -1:
        folder_name = folder_name + f"_prior_state_{args.num_prior_state}"

    print(f"Running experiment for {folder_name}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print("[Data]: Reading data...\n")

    # Load data
    data_type = "t5" if args.model_type == "t5" else "gpt"
    dataset_path_train = os.path.join(args.dataset_path, f'train-subsample-states-{data_type}.jsonl')
    print("Train dataset:", dataset_path_train)
    dataset_path_test = os.path.join(args.dataset_path, f'test-subsample-states-{data_type}.jsonl')

    train_df = pd.read_json(dataset_path_train, orient='records', lines=True)
    test_df = pd.read_json(dataset_path_test, orient='records', lines=True)

    if args.eval_only:
        train_df = train_df.head(0)

    if args.model_type == "t5":
        train_df = train_df[["sentence_masked", "masked_content"]]
        test_df = test_df[["sentence_masked", "masked_content"]]

    # Load object names
    object_map = {}
    object_list = []
    with open(args.object_vocabulary_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            object_map[row["object_name"]] = i
            object_list.append(row["object_name"])

    act_container_train = []
    act_all_container_train = []
    act_container_test = []
    act_all_container_test = []

    if args.load_model_representation:  # This need to be modified to load all representations sequentially, file names are given by representaions_train_{batch}.p, each of shape [Batch_size, num_layers, hidden_size]
        # get numer of files to load:

        # load pre-computed representations
        train_rep_path = os.path.join(args.model_representation_path, "representations_train.p")
        test_rep_path = os.path.join(args.model_representation_path, "representations_test.p")

        with open(train_rep_path, "rb") as rep_f:
            act_all_container_train = pickle.load(rep_f)
        print("total train data shape:", act_all_container_train.shape)
        act_container_train = act_all_container_train[args.layer - 1].to(torch.float32)
        print("sliced train data shape:", act_container_train.shape)
        # for act in act_all_container_train:
        # act_container_train.append(act[args.layer - 1])

        act_container_train = list(torch.unbind(act_container_train, dim=0))
        # act_all_container_train.clear()
        del act_all_container_train

        with open(test_rep_path, "rb") as rep_f:
            act_all_container_test = pickle.load(rep_f)

        print("total test data shape:", act_all_container_test.shape)
        act_container_test = act_all_container_test[args.layer - 1].to(torch.float32)
        print("sliced test data shape:", act_container_test.shape)
        act_container_test = list(torch.unbind(act_container_test, dim=0))

        # for act in act_all_container_test:
        #     act_container_test.append(act[args.layer - 1]) # check with sebastian about how to choosing args.layer

        # act_all_container_test.clear()
        del act_all_container_test

    probe_class = 8 if not args.binary_probe else 2
    input_dim = _INPUT_DIMENSIONS[args.model_type]
    if args.object_location:
        probing_dataset_train = ObjectLocationProbeDataLoader(act_container_train, dataset_path_train)
        probing_dataset_test = ObjectLocationProbeDataLoader(act_container_test, dataset_path_test)
    elif args.binary_probe:
        probing_dataset_train = BinaryProbeDataLoader(act_container_train, dataset_path_train, object_map,
                                                      include_empty=not args.exclude_empty,
                                                      min_prev_objects=args.condition_on_obj,
                                                      local_operation_order=args.num_prior_state)
        probing_dataset_test = BinaryProbeDataLoader(act_container_test, dataset_path_test, object_map,
                                                     include_empty=not args.exclude_empty,
                                                     min_prev_objects=args.condition_on_obj,
                                                     local_operation_order=args.num_prior_state)
    else:
        probing_dataset_train = ProbeDataLoader(act_container_train, dataset_path_train, object_map)
        probing_dataset_test = ProbeDataLoader(act_container_test, dataset_path_test, object_map)

    # train_size = int(0.8 * len(probing_dataset))
    # test_size = len(probing_dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])
    train_dataset, test_dataset = probing_dataset_train, probing_dataset_test
    sampler = None
    # train_loader = DataLoader(probing_dataset_dev, shuffle=False, sampler=sampler, pin_memory=True, batch_size=128, num_workers=1)
    # test_loader = DataLoader(probing_dataset_test, shuffle=True, pin_memory=True, batch_size=128, num_workers=1)

    if args.object_location:
        if args.twolayer:
            raise ValueError("Parameter --twolayer is not supported when using the object location probe.")
        probe = ObjectLocationProbeClassification(device,
                                                  input_dim=input_dim,
                                                  probe_class=probe_class,
                                                  ce_weights=probing_dataset_train.get_weights().to(device,
                                                                                                    dtype=torch.float32))
    else:
        if args.twolayer:
            probe = BatteryProbeClassificationTwoLayer(device,
                                                       input_dim=input_dim,
                                                       probe_class=probe_class,
                                                       num_task=100,
                                                       mid_dim=args.mid_dim,
                                                       ce_weights=probing_dataset_train.get_weights().to(device,
                                                                                                         dtype=torch.float32),
                                                       )
        else:
            probe = BatteryProbeClassification(device,
                                               input_dim=input_dim,
                                               probe_class=probe_class,
                                               num_task=100,
                                               ce_weights=probing_dataset_train.get_weights().to(device,
                                                                                                 dtype=torch.float32),
                                               )

    max_epochs = args.epo
    # print probe shape:
    # print(probe['proj.weight'].shape)
    # print(probe.proj.weight.shape)
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=1024, learning_rate=3e-3,
        betas=(.9, .999),
        lr_decay=True, warmup_tokens=len(train_dataset) * 5,
        final_tokens=len(train_dataset) * max_epochs,
        num_workers=4, weight_decay=0.,
        ckpt_path=os.path.join(args.checkpoint_root, folder_name, f"layer{args.layer}_token1")
    )
    trainer = Trainer(probe, train_dataset, test_dataset, tconf)
    if not args.eval_only:
        predictions_matrix = trainer.train(prt=True).astype(int)
        trainer.save_traces()
        trainer.save_checkpoint()
    else:
        trainer.load_checkpoint()
        predictions_matrix = trainer.predict(prt=True).astype(int)

    predictions_file = os.path.join(tconf.ckpt_path, "predictions.txt")
    header = " ".join(object_list)
    np.savetxt(predictions_file, predictions_matrix, delimiter=" ", fmt='%i', header=header, comments="")
    # save plot
    fig_file = os.path.join(tconf.ckpt_path, "predictions.pdf")
    trainer.flush_plot(fig_file)


if __name__ == "__main__":
    main()


