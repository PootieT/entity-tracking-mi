import os
import pdb
import json
from typing import List
from pathlib import Path

from tqdm import tqdm
from scipy import stats
import statannot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fix_fonts(title=20, label=20, xtick=15, ytick=15, default=15):
    # Set the global font family to 'Times New Roman'
    # keep running into
    plt.rc('font', family='serif', serif=['Times New Roman'])

    # Set the global default font size (e.g., to 14)
    plt.rcParams["font.size"] = default
    plt.rcParams["xtick.labelsize"] = xtick  # Optional: specific size for x-axis ticks
    plt.rcParams["ytick.labelsize"] = ytick  # Optional: specific size for y-axis ticks
    plt.rcParams["axes.labelsize"] = label  # Optional: specific size for axis labels
    plt.rcParams["axes.titlesize"] = title  # Optional: specific size for plot titles


def plot_global_local_remove(exp_dir: str, ax=None, metric="Logit Diff", split_behavioral=True, all_tests=False):
    rows = []
    for f in os.listdir(exp_dir):
        if not f.endswith(".json"):
            continue
        removal_target = f.replace(".json", "")
        with open(os.path.join(exp_dir, f), "r") as f:
            data = json.load(f)
        for row in data["full_results"]:
            correct_ctf_label = [l for l in row["labels"] if l !=row["ctf_label"]]
            # (rank_ctf_obj - rank_noop_obj) - (rank_ctf_target - rank_noop_target)
            if row.get("target_objs_ctf_rank"):
                rank_diff_offset = sum(-row["target_objs_ctf_rank"][i] + row["target_objs_rank"][i] for i in range(len(row["target_objs_ctf_rank"])))/len(row["target_objs_ctf_rank"])
            else:
                rank_diff_offset = 0
            rows.append({
                "Object Type": removal_target.capitalize() + " Remove",
                "Logit Diff": row["logit_diff"],
                "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                "dataset_index": row.get("dataset_index"),
                "Rank Diff": row["rank_diff"] if "rank_diff" in row else None,
                "Rank Diff From Target": (row["rank_diff"] +rank_diff_offset) if "rank_diff" in row else None,
                "target_cnt": len(row["target_objs"]),
            })
            if "target_objs_logit_diff" in row:
                for i, target_diff in enumerate(row["target_objs_logit_diff"]):
                    rows.append({
                        "Object Type": "Target",
                        "Logit Diff": target_diff,
                        "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                        "dataset_index": row.get("dataset_index"),
                        "Rank Diff": row["target_objs_rank_diff"][i] if "target_objs_rank_diff" in row else None,
                        "Rank Diff From Target": (row["target_objs_rank_diff"][i]  +rank_diff_offset) if "target_objs_rank_diff" in row else None,
                        "target_cnt": len(row["target_objs"]),
                    })
            if "other_objs_logit_diff" in row:
                for i, other_diff in enumerate(row["other_objs_logit_diff"]):
                    rows.append({
                        "Object Type": "Other",
                        "Logit Diff": other_diff,
                        "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                        "dataset_index": row.get("dataset_index"),
                        "Rank Diff": row["other_objs_rank_diff"][i] if "other_objs_rank_diff" in row else None,
                        "Rank Diff From Target": (row["other_objs_rank_diff"][i]  +rank_diff_offset) if "other_objs_rank_diff" in row else None,
                        "target_cnt": len(row["target_objs"]),
                    })
    df = pd.DataFrame(rows)

    if metric != "Logit Diff":
        df[metric] = df[metric].clip(lower=-20, upper=20)
    if not split_behavioral:
        df = df[df["Model Correctness"]=="Correct"]
    # pdb.set_trace()
    model_name = exp_dir.split("/")[-1]

    fix_fonts()
    if ax is None:
        plt.figure(figsize=(7, 3.5))
    if split_behavioral:
        axis = sns.violinplot(
            data=df, x="Object Type", y=metric, ax=ax,
            hue="Model Correctness", hue_order=["Incorrect", "Correct"],
            order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
        )
    else:
        axis = sns.violinplot(
            data=df, x="Object Type", y=metric, ax=ax,
            order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
        )
    if axis.get_legend():
        axis.get_legend().set_title("")
    axis.axhline(y=0, color='black', linestyle='--', linewidth=1)  #
    if split_behavioral:
        axis.axhline(y=df[(df["Object Type"] == "Other") & (df["Model Correctness"] == "Correct")][metric].mean(), color='red',linestyle='--', linewidth=1)  # Other object baseline
    else:
        axis.axhline(y=df[(df["Object Type"] == "Other")][metric].mean(), color='red', linestyle='--', linewidth=1)
    plt.xticks(rotation=10)

    # do some statistical significance test
    if split_behavioral:
        pairs = [
            (("Query Remove", "Correct"), ("Irrelevant Remove", "Correct")),
            (("Other", "Correct"), ("Irrelevant Remove", "Correct")),
            (("Query Remove", "Correct"), ("Other", "Correct")),
            (("Query Remove", "Correct"), ("Query Remove", "Incorrect")),
            (("Irrelevant Remove", "Correct"), ("Irrelevant Remove", "Incorrect")),
            (("Target", "Correct"), ("Target", "Incorrect")),
            (("Other", "Correct"), ("Other", "Incorrect")),
        ]
        if all_tests:
            pairs.extend([
                (("Query Remove", "Correct"), ("Target", "Correct")),
                (("Irrelevant Remove", "Correct"), ("Target", "Correct")),
                (("Target", "Correct"), ("Other", "Correct")),
            ])
            result = stats.wilcoxon(df[(df["Object Type"]=="Irrelevant Remove")&(df["Model Correctness"]=="Correct")][metric], y=None)
            print("Irrelevant remove vs 0 test:\n{}".format(result))

        test_results = statannot.add_stat_annotation(
            axis, data=df, x="Object Type", y=metric,
            hue="Model Correctness", hue_order=["Incorrect", "Correct"],
            order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
            box_pairs=pairs,
            test="Mann-Whitney",  # Mann-Whitney
            text_format='star', # full, star
            loc='inside', verbose=2,
            text_offset = -7,
        )
    else:
        test_results = statannot.add_stat_annotation(
            axis, data=df, x="Object Type", y=metric,
            order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
            box_pairs=[
                ("Query Remove", "Irrelevant Remove"),
                ("Other", "Irrelevant Remove"),
                ("Query Remove", "Other"),
            ],
            test="Mann-Whitney",  # Mann-Whitney
            text_format='star',  # full, star
            loc='inside', verbose=2,
            text_offset=-7,
        )

    if ax is None:
        if split_behavioral:
            plt.title(f"REMOVE Avg. Logit Argmax Accuracy={(df['Model Correctness']=='Correct').mean():.2f}")
        elif all_tests:
            plt.title(f"Irrelevant Remove vs. 0 pvalue={result.pvalue}")

        plt.tight_layout()
        file_path = os.path.join(exp_dir, "hist.png" if metric=="Logit Diff" else "hist_rank.png" if metric=="Rank Diff" else "hist_rank_target.png")
        if not split_behavioral:
            file_path = file_path.replace(".png", "_no_split.png")
        if all_tests:
            file_path = file_path.replace(".png", "_all_tests.png")
        plt.savefig(file_path, dpi=600)
        plt.close()
    else:
        if split_behavioral:
            ax.set_title(f"{model_name}, removal_accuracy={(df[df['Object Type'].contains('Remove')]['Model Correctness']=='Correct').mean():.2f}")
    print(df.groupby(["Object Type"])['Model Correctness'].value_counts())




def plot_global_local_remove_2_v_0_shot_model_family(exp_dirs: List[str], family_name: str, metric: str = "Logit Diff"):
    plt.rc('font', family='serif', serif=['Times New Roman'])

    # plot a 2x3 grid of the above experiments
    n_cols = len(exp_dirs) // 2
    fig, axs = plt.subplots(2, n_cols, figsize=(12, 5))
    plt.rcParams["axes.titlesize"] = 15
    for i, exp_dir in enumerate(exp_dirs):
        # if not exists, leave a empty plot
        if n_cols > 1:
            cur_ax = axs[i // n_cols, i % n_cols]
        else:
            cur_ax = axs[i // n_cols]
        if not os.path.exists(exp_dir):
            # leave an empty plot, with boundary
            cur_ax.spines['top'].set_visible(True)
            cur_ax.spines['right'].set_visible(True)
            cur_ax.spines['left'].set_visible(True)
            cur_ax.spines['bottom'].set_visible(True)
            # add title, but indicate missing
            cur_ax.set_title(f"Missing: {exp_dir.split('/')[-1]}")
            continue

        rows = []
        for f in os.listdir(exp_dir):
            mode = "Two-Shot" if ("two-shot" in exp_dir or "PROMPT_ALTFORM" in exp_dir) else "Zero-Shot"
            if not f.endswith(".json"):
                continue
            removal_target = f.replace(".json", "")
            try:
                with open(os.path.join(exp_dir, f), "r") as f_in:
                    data = json.load(f_in)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
            for row in data["full_results"]:
                correct_ctf_label = [l for l in row["labels"] if l != row["ctf_label"]]
                rows.append({
                    "Object Type": removal_target.capitalize() + " Remove",
                    "Logit Diff": row["logit_diff"],
                    "Model Correctness": "Correct" if row["ctf_argmax_token"].strip() in correct_ctf_label else "Incorrect",
                    "Rank Diff": row["rank_diff"] if "rank_diff" in row else None,
                })
                if "target_objs_logit_diff" in row:
                    for obj_i, target_diff in enumerate(row["target_objs_logit_diff"]):
                        rows.append({
                            "Object Type": "Target",
                            "Logit Diff": target_diff,
                            "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                            "dataset_index": row.get("dataset_index"),
                            "Rank Diff": row["target_objs_rank_diff"][obj_i] if "target_objs_rank_diff" in row else None,
                        })
                if "other_objs_logit_diff" in row:
                    for obj_i, other_diff in enumerate(row["other_objs_logit_diff"]):
                        rows.append({
                            "Object Type": "Other",
                            "Logit Diff": other_diff,
                            "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                            "dataset_index": row.get("dataset_index"),
                            "Rank Diff": row["other_objs_rank_diff"][obj_i] if "other_objs_rank_diff" in row else None,
                        })
        df = pd.DataFrame(rows)
        if metric != "Logit Diff":
            df[metric] = df[metric].clip(lower=-20, upper=20)

        model_name = exp_dir.split("/")[-1]

        sns.violinplot(data=df, x="Object Type", y=metric,
                    hue="Model Correctness", hue_order=["Incorrect", "Correct"],
                    order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
                    ax = cur_ax,
                    )
        test_results = statannot.add_stat_annotation(
            cur_ax, data=df, x="Object Type", y=metric,
            hue="Model Correctness", hue_order=["Incorrect", "Correct"],
            order=["Query Remove", "Irrelevant Remove", "Target", "Other"],
            box_pairs=[
                (("Query Remove", "Correct"), ("Irrelevant Remove", "Correct")),
                (("Other", "Correct"), ("Irrelevant Remove", "Correct")),
                (("Query Remove", "Correct"), ("Other", "Correct")),
                (("Query Remove", "Correct"), ("Query Remove", "Incorrect")),
                (("Irrelevant Remove", "Correct"), ("Irrelevant Remove", "Incorrect")),
                (("Other", "Correct"), ("Other", "Incorrect")),
                (("Target", "Correct"), ("Target", "Incorrect")),
            ],
            test="Mann-Whitney",  # Mann-Whitney
            text_format='star',  # full, star
            loc='inside', verbose=2,
            text_offset=-4,
            # line_offset=2
        )
        # print(test_results)

        cur_ax.axhline(y=0, color='black', linestyle='--',linewidth=1)  #
        cur_ax.axhline(y=df[df["Object Type"] == "Other"][metric].mean(), color='red', linestyle='--', linewidth=1)  # Other object baseline
        if cur_ax.get_legend():
            cur_ax.get_legend().set_title("")
        cur_ax.set_title(f"{model_name.replace('_PROMPT_ALTFORM','')}, Logit Acc.={(df[df['Object Type'].str.contains('Remove')]['Model Correctness']=='Correct').mean():.2f}")
        # indicate: top row is two-shot, bottom row is zero-shot on Y axis

        if i % n_cols == 0:
            cur_ax.set_ylabel(f"{mode}\n{metric}")
        else:
            cur_ax.set_ylabel("")
        # indicate: left column is Llama-3.1-8B, middle column
        if i // n_cols == 0:  # top row
            cur_ax.set_xlabel("")
            cur_ax.set_xticklabels([])
        else:
            cur_ax.tick_params(axis='x', labelrotation=10 if n_cols <3 else 50)

        if not ((i % n_cols == n_cols - 1) and (i // n_cols == 1)):
            cur_ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(f"{Path(exp_dirs[0]).parent}/{family_name}_family{'_rank' if metric=='Rank Diff' else ''}.png", dpi=600)
    plt.close()


def plot_global_local_put(exp_dir: str, ax=None, metric="Logit Diff"):
    rows = []
    for f in os.listdir(exp_dir):
        if not f.endswith(".json"):
            continue
        removal_target = f.replace(".json", "")
        with open(os.path.join(exp_dir, f), "r") as f:
            data = json.load(f)
        for row in data["full_results"]:
            correct_ctf_label = [l.strip() for l in row["ctf_label"]] + [l.strip() for l in row["labels"]]
            correct = "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect"
            rows.append({
                "Object Type": removal_target.capitalize() + " Put",
                "Logit Diff": row["logit_diff"],
                "Model Correctness": correct,
                "dataset_index": row.get("dataset_index"),
                "Rank Diff": row["rank_diff"] if "rank_diff" in row else None,
            })
            if "target_objs_logit_diff" in row:
                for i, target_diff in enumerate(row["target_objs_logit_diff"]):
                    rows.append({
                        "Object Type": "Target",
                        "Logit Diff": target_diff,
                        "Model Correctness": correct,
                        "dataset_index": row.get("dataset_index"),
                        "Rank Diff": row["target_objs_rank_diff"][i] if "target_objs_rank_diff" in row else None,
                    })
            if "other_objs_logit_diff" in row:
                for i, other_diff in enumerate(row["other_objs_logit_diff"]):
                    rows.append({
                        "Object Type": "Other",
                        "Logit Diff": other_diff,
                        "Model Correctness": correct,
                        "dataset_index": row.get("dataset_index"),
                        "Rank Diff": row["other_objs_rank_diff"][i] if "other_objs_rank_diff" in row else None,
                    })

    df = pd.DataFrame(rows)
    if metric != "Logit Diff":
        df[metric] = df[metric].clip(lower=-1000, upper=1000)
    model_name = exp_dir.split("/")[-1]

    fix_fonts()
    plt.figure(figsize=(7, 3.5))

    axis = sns.violinplot(
        data=df, x="Object Type", y=metric, ax=ax,
        hue="Model Correctness", hue_order=["Incorrect", "Correct"],
        order=["Query Put", "Irrelevant Put", "Target", "Other"],
    )
    test_results = statannot.add_stat_annotation(
        axis, data=df, x="Object Type", y=metric,
        hue="Model Correctness", hue_order=["Incorrect", "Correct"],
        order=["Query Put", "Irrelevant Put", "Target", "Other"],
        box_pairs=[
            (("Query Put", "Correct"), ("Irrelevant Put", "Correct")),
            (("Other", "Correct"), ("Irrelevant Put", "Correct")),
            (("Query Put", "Correct"), ("Other", "Correct")),
            (("Query Put", "Correct"), ("Query Put", "Incorrect")),
            (("Irrelevant Put", "Correct"), ("Irrelevant Put", "Incorrect")),
        ],
        test="Mann-Whitney",  # Mann-Whitney
        text_format='star',  # full, star
        loc='inside', verbose=2,
        text_offset=-7,
        # line_offset=2
    )
    axis.get_legend().set_title("")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  #
    plt.axhline(y=df[df["Object Type"] == "Other"][metric].mean(), color='red', linestyle='--',linewidth=1)  # Other object baseline
    if ax is None:
        plt.title(f"PUT Logit Argmax Accuracy={(df['Model Correctness']=='Correct').mean():.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "hist.png" if metric == "Logit Diff" else "hist_rank.png"), dpi=600)
    else:
        ax.set_title(f"{model_name}, removal_accuracy={(df['Model Correctness']=='Correct').mean():.2f}")


def get_obj_box_id(phrases: List[str], obj: str) -> int:
    for i, phrase in enumerate(phrases):
        if f"he {obj} " in phrase:
            return i
    raise ValueError(f"No object found for {obj}:\n{phrases}")

def plot_global_local_remove_by_box_position(exp_dir: str, ax=None, metric="Logit Diff", test=False):
    rows = []
    for f in os.listdir(exp_dir):
        if not f.endswith(".json"):
            continue
        removal_target = f.replace(".json", "")
        with open(os.path.join(exp_dir, f), "r") as f:
            data = json.load(f)
        for row in data["full_results"]:
            correct_ctf_label = [l for l in row["labels"] if l !=row["ctf_label"]]
            query_box = int(row["sentence"].split()[-3])
            rows.append({
                "Object Type": removal_target.capitalize() + " Remove",
                "Logit Diff": row["logit_diff"],
                "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                "dataset_index": row.get("dataset_index"),
                "Box": query_box if removal_target == "query" else int(row["ctf_sentence"].split(". ")[-2].split()[-1]) if "Description" not in row["ctf_sentence"] else int(row["ctf_sentence"].split(".\nStatement")[-2].split()[-1]),
                "Query Box": query_box,
                "Rank Diff": row["rank_diff"] if "rank_diff" in row else None,
            })
            if "target_objs_logit_diff" in row:
                for i, target_diff in enumerate(row["target_objs_logit_diff"]):
                    rows.append({
                        "Object Type": "Target",
                        "Logit Diff": target_diff,
                        "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                        "dataset_index": row.get("dataset_index"),
                        "Box": query_box,
                        "Query Box": query_box,
                        "Rank Diff": row["target_objs_rank_diff"][i] if "target_objs_rank_diff" in row else None,
                    })
            if "other_objs_logit_diff" in row:
                desc_phrases = row["sentence"].split(". ")[0].split(", ") if "Description" not in row["sentence"] else row["sentence"].split("\n\nDescription: ")[-1].split("\nStatement: ")[0].split(". ")[0].split(", ")
                for i, (obj_name, obj_diff) in enumerate(zip(row["other_objs"],row["other_objs_logit_diff"])):
                    box_id = get_obj_box_id(desc_phrases, obj_name)
                    rows.append({
                        "Object Type": "Other",
                        "Logit Diff": obj_diff,
                        "Model Correctness": "Correct" if row["ctf_argmax_token"].strip().lower() in correct_ctf_label else "Incorrect",
                        "dataset_index": row.get("dataset_index"),
                        "Box": box_id,
                        "Query Box": query_box,
                        "Rank Diff": row["other_objs_rank_diff"][i] if "other_objs_rank_diff" in row else None,
                    })
    df = pd.DataFrame(rows)
    df = df[df["Model Correctness"]=="Correct"]
    model_name = exp_dir.split("/")[-1]
    if metric != "Logit Diff":
        df[metric] = df[metric].clip(lower=-20, upper=20)

    fix_fonts()
    fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(12, 20), sharex=True, sharey=True)
    for col, remove_type in enumerate(["Query Remove", "Irrelevant Remove"]):
        for row, query_box in enumerate(range(7)):
            hue_order = ["Other", "Target", "Query Remove"] if remove_type == "Query Remove" else ["Target", "Irrelevant Remove"]
            sub_df = df[df["Query Box"] == query_box]
            sns.violinplot(
                data=sub_df, x="Box", y=metric, ax=axes[row, col], fill=False, linewidth=1,
                hue="Object Type", hue_order=hue_order, palette=sns.color_palette()[:3] if remove_type == "Query Remove" else sns.color_palette()[1:3],
            )
            if test:
                # automatically construct pairwise tests to check if
                # - query remove different from target (no longer testing)
                # - one-adjacent box to irrelevant remove are higher than ones farther away
                # - one-adjacent box other obj are higher than ones farther away
                # despite the tuple being ordered, the package seem to sort the data of the two distribution
                # by the order in which they appear in the graph (left to right), so we have to manually use 2
                # set of tests for smaller and greater tests.
                ls_pairs, gt_pairs = [], []
                if remove_type == "Query Remove":
                    # ls_pairs.append(((query_box, "Query Remove"), (query_box, "Target")))
                    for left_farther_box in range(0, query_box-1):
                        ls_pairs.append(((query_box-1, "Other"), (left_farther_box, "Other")))
                    for right_farther_box in range(query_box+2, 7):
                        gt_pairs.append(((query_box+1, "Other"), (right_farther_box, "Other")))
                else:
                    for left_farther_box in range(0, query_box-1):
                        ls_pairs.append(((query_box-1, "Irrelevant Remove"), (left_farther_box, "Irrelevant Remove")))
                    for right_farther_box in range(query_box+2, 7):
                        gt_pairs.append(((query_box+1, "Irrelevant Remove"), (right_farther_box, "Irrelevant Remove")))
                if ls_pairs:
                    test_results1 = statannot.add_stat_annotation(
                        axes[row, col], data=sub_df, x="Box", y=metric, linewidth=1, # fill does not work with test
                        hue="Object Type", hue_order=hue_order,
                        box_pairs=ls_pairs,
                        test="Mann-Whitney-ls",  # Mann-Whitney
                        text_format='star',  # full, star
                        loc='inside', verbose=2,
                        text_offset=-2,
                        # line_offset=2
                    )
                if gt_pairs:
                    test_results1 = statannot.add_stat_annotation(
                        axes[row, col], data=sub_df, x="Box", y=metric, linewidth=1,  # fill does not work with test
                        hue="Object Type", hue_order=hue_order,
                        box_pairs=gt_pairs,
                        test="Mann-Whitney-gt",  # Mann-Whitney
                        text_format='star',  # full, star
                        loc='inside', verbose=2,
                        text_offset=-2,
                        # line_offset=2
                    )


            axes[row, col].axhline(y=0, color='black', linestyle='--', linewidth=1)  # no difference baseline
            axes[row, col].axhline(y=sub_df[sub_df["Object Type"]=="Other"][metric].mean(), color='red', linestyle='--', linewidth=1)  # Other object baseline
            axes[row, col].set_title(f"Query Box=Box {query_box}, {remove_type}")
            if not (row==6):
                axes[row, col].get_legend().remove()

    # plt.suptitle(f"REMOVE Avg. Logit Argmax Accuracy={(df[df['Object Type'].isin(['Query Remove', 'Irrelevant Remove'])]['Model Correctness']=='Correct').mean():.2f}")
    # plt.suptitle(f"{model_name} {metric} from Remove (by position)")
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "hist_by_pos.png" if metric=="Logit Diff" else "hist_rank_by_pos.png"), dpi=600)
    plt.close()

    # plot a 2x3 grid of the above experiments
    n_cols = len(exp_dirs) // 2
    n_rows = 2
    fig, axs = plt.subplots(2, n_cols, figsize=(n_cols * 6, n_rows * 3.5))
    shared_handles, shared_labels = None, None  # ADD

    for i, exp_dir in enumerate(exp_dirs):
        # if not exists, leave an empty plot
        if not os.path.exists(exp_dir):
            # leave an empty plot, with boundary
            axs[i // n_cols, i % n_cols].spines['top'].set_visible(True)
            axs[i // n_cols, i % n_cols].spines['right'].set_visible(True)
            axs[i // n_cols, i % n_cols].spines['left'].set_visible(True)
            axs[i // n_cols, i % n_cols].spines['bottom'].set_visible(True)
            # add title, but indicate missing
            axs[i // n_cols, i % n_cols].set_title(f"Missing: {exp_dir.split('/')[-1]}")
            continue

        rows = []
        for f in os.listdir(exp_dir):
            mode = "Two-Shot" if "two-shot" in exp_dir else "Zero-Shot"
            if f == "hist.png":
                continue
            removal_target = f.replace(".json", "")
            with open(os.path.join(exp_dir, f), "r") as f:
                data = json.load(f)
            for row in data["full_results"]:
                correct_ctf_label = [l for l in row["labels"] if l != row["ctf_label"]]
                rows.append({
                    "REMOVE Box": removal_target.capitalize(),
                    "Logit Diff": row["logit_diff"],
                    "Model Correctness": "Correct" if row["ctf_argmax_token"].strip() in correct_ctf_label else "Incorrect",
                })
        df = pd.DataFrame(rows)
        model_name = exp_dir.split("/")[-1]

        sns.boxplot(data=df, x="REMOVE Box", y=f"Logit Diff",
                    hue="Model Correctness", hue_order=["Incorrect", "Correct"], ax=axs[i // n_cols, i % n_cols])
        axs[i // n_cols, i % n_cols].axhline(y=0, color='black', linestyle='--', linewidth=1)  #
        axs[i // n_cols, i % n_cols].set_title(f"{model_name}, Logit Acc.={(df['Model Correctness']=='Correct').mean():.2f}")
        # fix legend location
        axs[i // n_cols, i % n_cols].legend(loc='upper left')
        # ADD: capture legend once, then remove per-axes legend
        leg = axs[i // n_cols, i % n_cols].get_legend()

        # indicate: top row is two-shot, bottom row is zero-shot on Y axis
        if i % n_cols == 0:
            axs[i // n_cols, i % n_cols].set_ylabel(f"{mode}\nLogit Diff")
        else:
            axs[i // n_cols, i % n_cols].set_ylabel("")
        # indicate: left column is Llama-3.1-8B, middle column

    plt.tight_layout()
    plt.savefig(
        f"../outputs/behavioral_global_local_remove/logit_diff_1remove_{family_name}_family.png",
        dpi=600,
        bbox_inches="tight"
    )


def plot_remove_main_fig(out_dir):

    fix_fonts()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7, 5),
        sharex=True,
    )
    plt.rcParams["axes.titlesize"] = 15
    plot_global_local_remove(f"{out_dir}", metric="Logit Diff", split_behavioral=False, ax=axes[0])
    plot_global_local_remove(f"{out_dir}", metric="Rank Diff", split_behavioral=False, ax=axes[1])
    # axes[0].set_title(f"REMOVE Avg. Logit Argmax Accuracy={axes[0].get_title().split('=')[1]}")
    axes[0].set_xlabel("")
    axes[1].set_title("")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/main_fig.png",dpi=600)


if __name__ == "__main__":
    out_dir = "../outputs/behavioral_global_local_remove_swap_2obj"
    for model in tqdm(["codellama-13b","llama-2-7b", "llama-2-13b", "llama-3.1-8b", "mistral-7b","gemma-2-2b", "gemma-2-9b", "qwen-3-1.7b", "qwen-3-4b", "qwen-3-8b", "qwen-3-14b"]):
        plot_global_local_remove_by_box_position(f"{out_dir}_4k/{model}", metric="Rank Diff", test=True)
        plot_global_local_remove_by_box_position(f"{out_dir}_4k/{model}_PROMPT_ALTFORM", metric="Rank Diff", test=True)
        for metric in ["Logit Diff", "Rank Diff"]:
            for all_tests in [True, False]:
                plot_global_local_remove(f"{out_dir}/{model}", metric=metric, all_tests=all_tests)
                plot_global_local_remove(f"{out_dir}/{model}_PROMPT_ALTFORM", metric=metric, all_tests=all_tests)

    plot_remove_main_fig("../outputs/behavioral_global_local_remove_swap_2obj/codellama-13b")

    for model in ["codellama-13b", "gemma-2-2b"]:
        for shot in ["", "_PROMPT_ALTFORM"]:
            for metric in ["Logit Diff", "Rank Diff"]:
                plot_global_local_put(f"../outputs/behavioral_global_local_put/{model}{shot}", metric=metric)


    LOG_DIR2 = "../outputs/behavioral_global_local_remove_swap_2obj"

    # NEW PLOTS (with illegal remove)
    # Plot Llama Family: 2*2
    llama_exp_dirs = [
        os.path.join(LOG_DIR2, "llama-2-7b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "llama-3.1-8b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "llama-2-13b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "codellama-13b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "llama-2-7b"),
        os.path.join(LOG_DIR2, "llama-3.1-8b"),
        os.path.join(LOG_DIR2, "llama-2-13b"),
        os.path.join(LOG_DIR2, "codellama-13b"),
    ]
    plot_global_local_remove_2_v_0_shot_model_family(llama_exp_dirs, "llama", metric="Logit Diff")
    plot_global_local_remove_2_v_0_shot_model_family(llama_exp_dirs, "llama", metric="Rank Diff")


    # Plot Qwen Family: 2 * 4
    qwen_exp_dirs = [
        os.path.join(LOG_DIR2, "qwen-3-1.7b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2,  "qwen-3-4b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "qwen-3-8b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "qwen-3-14b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "qwen-3-1.7b"),
        os.path.join(LOG_DIR2,  "qwen-3-4b"),
        os.path.join(LOG_DIR2, "qwen-3-8b"),
        os.path.join(LOG_DIR2, "qwen-3-14b"),
    ]
    plot_global_local_remove_2_v_0_shot_model_family(qwen_exp_dirs, "qwen", metric="Logit Diff")
    plot_global_local_remove_2_v_0_shot_model_family(qwen_exp_dirs, "qwen", metric="Rank Diff")
    # Plot Gemma Family: 2 * 2
    gemma_exp_dirs = [
        os.path.join(LOG_DIR2, "gemma-2-2b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "gemma-2-9b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "gemma-2-2b"),
        os.path.join(LOG_DIR2, "gemma-2-9b"),
    ]
    plot_global_local_remove_2_v_0_shot_model_family(gemma_exp_dirs, "gemma", metric="Logit Diff")
    plot_global_local_remove_2_v_0_shot_model_family(gemma_exp_dirs, "gemma", metric="Rank Diff")

    # Plot mistral Family: 1 * 2
    mistral_exp_dirs = [
        os.path.join(LOG_DIR2, "mistral-7b_PROMPT_ALTFORM"),
        os.path.join(LOG_DIR2, "mistral-7b"),
    ]
    plot_global_local_remove_2_v_0_shot_model_family(mistral_exp_dirs, "mistral", metric="Logit Diff")
    plot_global_local_remove_2_v_0_shot_model_family(mistral_exp_dirs, "mistral", metric="Rank Diff")
