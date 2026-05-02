import argparse
import json

def t5_to_gpt_format(jsonl_file, format):
  """
  Converts a .jsonl file to a .txt file.

  Args:
    jsonl_file: The path to the .jsonl file.

  Returns:
    The path to the .txt file.
  """

  with open(jsonl_file, "r") as f:
    jsonl_data = f.readlines()

  save_file = None

  if format == "txt":
    save_file = jsonl_file.replace(".jsonl", ".txt")
    save_file = save_file.replace("-t5", "-gpt")
    with open(save_file, "w") as f:
      for line in jsonl_data:
        json_data = json.loads(line)
        sentence = json_data["sentence"]
        f.write(f"{sentence}\n")

  elif format == "tsv":
    save_file = jsonl_file.replace(".jsonl", ".tsv")
    save_file = save_file.replace("-t5", "-gpt")
    with open(save_file, "w") as f:
      f.write(f"sentence\tprefix\n")
      for line in jsonl_data:
        json_data = json.loads(line)
        sentence = json_data["sentence"]
        prefix = json_data["sentence_masked"].split(" <extra_id_0>")[0]
        f.write(f"{sentence}\t{prefix}\n")

  elif format == "jsonl":
    save_file = jsonl_file.replace("-t5", "-gpt")
    with open(save_file, "w") as f:
      for line in jsonl_data:
        json_data = json.loads(line)
        sentence = json_data["sentence"]
        prefix = json_data["sentence_masked"].split(" <extra_id_0>")[0]
        masked_content = sentence.removeprefix(prefix + " ").strip(".")
        new_json = {"sentence": sentence, "prefix": prefix, "masked_content": masked_content, "numops": json_data["numops"]}
        f.write(json.dumps(new_json) + "\n")

  return save_file
      

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", help="The path to the .jsonl file to be converted.", required=True)
  parser.add_argument("--format", help="tsv or txt", required=True)
  args = parser.parse_args()

  print(f"Wrote converted file to {t5_to_gpt_format(args.input_file, args.format)}")
