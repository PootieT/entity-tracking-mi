# USED TO CONVERT THE PROBING LOGS TO A CSV FILE. NEED TO REORGANIZE THE LOG FILES MANUALLY

import os   
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import argparse

# NUM_LAYERS = 126 # for llama3 405b model
NUM_LAYERS = 81 # 1 + 80 for Llama 70B, 1 + 40 for Llama 13B

def list_files_in_directory(directory):
    """
    List all files in a given directory.
    """
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def extract_result_from_file(file_path):
    """
    Extract the result from a given file.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming the result is in the last line of the file
        file.close()
        
    result = {
        "test_acc": None,
        "test_non_triv_acc": None,
        "Layer": None,
        "recall": None,
        "precision": None,
        "FP non-triv rate": None,
        "FP triv rate": None,
        
    }
    # format of the line contains the layer: Probing layer <#LAYER>
    layer = re.search(r'Probing layer (\d+)', lines[0])
    if layer:
        result["Layer"] = int(layer.group(1))
    else:
        raise ValueError("Layer not found in the file.")
    
    last_line = lines[-1]
    test_acc = re.search(r'test acc ([\d.]+)%', last_line)
    if test_acc:
        result["test_acc"] = float(test_acc.group(1))
    else:
        raise ValueError("Test accuracy not found in the last line.")
    test_non_triv_acc = re.search(r'test non-triv acc ([\d.]+)%', last_line)
    if test_non_triv_acc:
        result["test_non_triv_acc"] = float(test_non_triv_acc.group(1))
    else:
        raise ValueError("Test non-trivial accuracy not found in the last line.")
    
    
    recall = re.search(r'recall ([\d.]+)%', last_line)
    if recall:
        result["recall"] = float(recall.group(1))
    else:
        raise ValueError("Recall not found in the last line.")
    precision = re.search(r'precision ([\d.]+)%', last_line)
    if precision:
        result["precision"] = float(precision.group(1))
    else:
        raise ValueError("Precision not found in the last line.")   
    fp_non_triv_rate = re.search(r'FP non-triv rate ([\d.]+)%', last_line)
    if fp_non_triv_rate:
        result["FP non-triv rate"] = float(fp_non_triv_rate.group(1))
    else:
        raise ValueError("FP non-triv rate not found in the last line.")
    fp_triv_rate = re.search(r'FP triv rate ([\d.]+)%', last_line)
    if fp_triv_rate:
        result["FP triv rate"] = float(fp_triv_rate.group(1))
    else:
        raise ValueError("FP triv rate not found in the last line.")
    return result
    
    
def reformat_logs_to_csv(log_directory, output_csv_dir):
    """
    Reformat the logs in the given directory to a CSV file.
    """
    files = list_files_in_directory(log_directory)
    data = []
    
    for file in files:
        file_path = os.path.join(log_directory, file)
        time_stamp = os.path.getmtime(file_path)
        try:
            result = extract_result_from_file(file_path)
            result["time_stamp"] = time_stamp
            data.append(result)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    # for all the duplicated layers, keep the latest one
    data = sorted(data, key=lambda x: (x["Layer"], -x["time_stamp"]))
    # num unique layers
    num_unique_layers = len(set(entry["Layer"] for entry in data))
    print(f"Found {num_unique_layers} unique layers in the logs.")
    print(num_unique_layers)
    # drop layer 0 -- equivlaent to the final output layer
    data = [entry for entry in data if entry["Layer"] != 0]
    unique_data = {}
    for entry in data:
        layer = entry["Layer"]
        if layer not in unique_data:
            unique_data[layer] = entry
    data = list(unique_data.values())
    assert len(data) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, but got {len(data)}"
    # handle duplicate layers by keeping the latest one
    
    
    df = pd.DataFrame(data)
    # if not exists, create the directory
    if not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir)
        
    # save the dataframe to a csv file
    df = df.sort_values(by="Layer")
    output_csv_path = os.path.join(output_csv_dir, "result.csv")
    df.to_csv(output_csv_path, index=False)
    plot_results(df, output_csv_dir)
    print(f"Reformatted logs saved to {output_csv_path}")
    
    
def plot_results(df, output_csv_dir):
    # plot the curves for test acc and test non-triv acc
    plt.figure(figsize=(10, 5))
    plt.plot(df["Layer"], df["test_acc"], label="Test Acc")
    plt.plot(df["Layer"], df["test_non_triv_acc"], label="Test Non-Triv Acc")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy (%)")
    plt.title("Probing Results")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_csv_dir, "probing_results.png")) # for google sheet
    plt.savefig(os.path.join(output_csv_dir, "probing_results.pdf")) # for paper
    
    print(f"Plot saved to {os.path.join(output_csv_dir, 'probing_results.png')}")
    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat probing logs to CSV.")
    parser.add_argument("--log_directory", type=str, help="Directory containing the log files.", default="logs/probes_405b")
    parser.add_argument("--output_csv_dir", type=str, help="Path to save the output CSV file.",
                        default="results/probes_405b")
    
    args = parser.parse_args()
    
    reformat_logs_to_csv(args.log_directory, args.output_csv_dir)