

def eval_helper(file_path):
    with open(file_path, "r") as myfile:
        seq = []
        label = []
        for line in myfile:
            if line.strip() == "-DOCSTART-	O":                        # skip first two line
                continue
            if line == "\n":
                if len(seq) < 1:
                    continue
                text = ' '.join(seq)
                with open("input_sentence", "a") as myfile:
                    myfile.write(text + "\n")
                with open("label_eval", "a") as myfile:
                    for l in label:
                        myfile.write(l + "\n")
                    myfile.write("\n")
                seq = []
                label = []
            else:
                val = line.strip().split()
                seq = seq+[val[0]]
                label = label + [val[1]]

if __name__ == "__main__":
    eval_helper()