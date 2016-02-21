
WINDOW_SIZE = 5
FILE_PATH = "../ner/train"
START = ["_BOS", "_BOS"]

def writeFile(seq, tag):
    with open("input", "a") as myfile:
        for word in seq:
            myfile.write(word + " ")
        myfile.write("\n")
    with open("output", "a") as myfile:
        myfile.write(tag+'\n')


with open(FILE_PATH, "r") as myfile:
    seq = START
    label = []
    for line in myfile:
        if line.strip() == "-DOCSTART-	O":                        # skip first two line
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
                seq = seq+[val[0]]
                label = label + [val[1]]
            else:
                writeFile(seq, label[0])
                seq = seq[1:]+[val[0]]
                label = label[1:]+[val[1]]

# a = [1,2,3]
# a+[4]
# print a