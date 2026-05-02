import json
import argparse
import random


def main():
    
    parser = argparse.ArgumentParser(description="Randomize order of objects in each box.")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    
    args = parser.parse_args()
    
    with open(args.input, "r", encoding="UTF-8") as in_f, open(args.output, "w", encoding="UTF-8") as out_f:
        for line in in_f:
            d = json.loads(line)
            sents = d["sentence"][:-1].split(". ")
            
            init_statements  =sents[0].split(", ")
            init_statements_shuf = []
            for s in init_statements:
                if " contains " in s and " and " in s:
                    i = s.index(" contains ") + 10
                    contents = s[i:].split(" and ")
                    random.shuffle(contents)
                    contents_shuffled = " and ".join(contents)
                    init_statements_shuf.append(s[:i] + contents_shuffled)
                else:
                    init_statements_shuf.append(s)
            
            sents[0] = ", ".join(init_statements_shuf)
            
            # shuffle contents in operations
            if len(sents) > 2:
                for i, s in enumerate(sents[1:-1]):
                    if " and " in s:
                        toks = s.split()
                        prep_index = toks.index("from") if "from" in toks else toks.index("into")
                        contents = " ".join(toks[1:prep_index]).split(" and ")
                        random.shuffle(contents)
                        contents_shuffled = " and ".join(contents)
                        sents[i+1] = toks[0] + " " + contents_shuffled + " " + " ".join(toks[prep_index:])

            s = sents[-1]
            if " contains " in s and " and " in s:
                i = s.index(" contains ") + 10
                contents = s[i:].split(" and ")
                random.shuffle(contents)
                contents_shuffled = " and ".join(contents)
                sents[-1] = s[:i] + contents_shuffled
                d["masked_content"] = "<extra_id_0> " + contents_shuffled

            
            
            d["sentence"] = ". ".join(sents) + "."
            d["sentence_masked"] = ". ".join(sents[:-1]) + ". " + d["sentence_masked"].split(". ")[-1]

            print(json.dumps(d), file=out_f)

    
if __name__ == "__main__":
    main()