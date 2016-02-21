from sklearn.metrics import f1_score

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
                with open("input_sentence_train", "a") as myfile:
                    myfile.write(text + "\n")
                with open("label_eval_train", "a") as myfile:
                    for l in label:
                        myfile.write(l + "\n")
                seq = []
                label = []
            else:
                val = line.strip().split()
                seq = seq+[val[0]]
                label = label + [val[1]]

def eval (file_true, file_predict):
    true_label = []
    pred_label = []
    with open(file_true, "r") as myfile:
        for line in myfile:
            true_label.append(line.strip())
    with open(file_predict, "r") as myfile:
        for line in myfile:
            pred_label.append(line.strip())
    return f1_score(true_label, pred_label, average='macro')

if __name__ == "__main__":
    # eval_helper("../ner/train")
    print eval ("label_eval_train", "label_eval_train")