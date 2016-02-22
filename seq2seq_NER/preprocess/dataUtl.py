from sklearn.metrics import f1_score

WINDOW_SIZE = 5
START = ["_BOS", "_BOS"]

def writeFile(seq, tag):
    with open("../data/train_data", "a") as myfile:
        for word in seq:
            myfile.write(word + " ")
        myfile.write("\n")
    with open("../data/train_label", "a") as myfile:
        myfile.write(tag+'\n')

def genTrain(train_file_path):
    with open(train_file_path, "r") as myfile:
        seq = START
        label = []
        for line in myfile:
            if line.strip().__contains__("-DOCSTART-"):                        # skip first two line
                continue
            if line == "\n":
                if len(seq) <= WINDOW_SIZE/2:
                    continue
                for i in range(WINDOW_SIZE-len(seq)):
                    seq = seq+["_EOS"]
                writeFile(seq, label[0])
                for i in range(WINDOW_SIZE/2):
                    seq = seq[1:]+["_EOS"]
                    if i+1>=len(label):
                        break
                    writeFile(seq, label[i+1])
                seq = START
                label = []
            else:
                val = line.strip().split()
                if len(seq) < WINDOW_SIZE:
                    seq = seq+[val[0].lower()]
                    label = label + [val[3]]
                else:
                    writeFile(seq, label[0])
                    seq = seq[1:]+[val[0].lower()]
                    label = label[1:]+[val[3]]

def genSentence(file_path):
    with open(file_path, "r") as myfile:
        seq = []
        label = []
        for line in myfile:
            if line.strip().__contains__("-DOCSTART-"):                        # skip first two line
                continue
            if line == "\n":
                if len(seq) < 1:
                    continue
                text = ' '.join(seq)
                if file_path.__contains__("train"):
                    with open("../data/sentence_train", "a") as myfile:
                        myfile.write(text + "\n")
                    with open("../data/label_train", "a") as myfile:
                        for l in label:
                            myfile.write(l + "\n")
                else:
                    with open("../data/sentence_test", "a") as myfile:
                        myfile.write(text + "\n")
                    with open("../data/label_test", "a") as myfile:
                        for l in label:
                            myfile.write(l + "\n")
                seq = []
                label = []
            else:
                val = line.strip().split()
                seq = seq+[val[0].lower()]
                label = label + [val[3]]

def parser(str):
    words = str.strip().split(" ")
    seq = START
    rst = []
    for word in words:
        if len(seq) < WINDOW_SIZE:
            seq = seq+[word]
        else:
            rst.append(seq)
            seq = seq[1:]+[word]
    for i in range(WINDOW_SIZE-len(seq)):
        seq = seq+["_EOS"]
    rst.append(seq)
    for i in range(min(len(words)-1, WINDOW_SIZE/2)):
        seq = seq[1:]+["_EOS"]
        rst.append(seq)
    for i in range(len(rst)):
        rst[i] = ' '.join(rst[i])
    return rst


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
    print eval ("../data/label_train", "../data/label_train")