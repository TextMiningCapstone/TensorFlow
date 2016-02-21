
WINDOW_SIZE = 5
START = ["_BOS", "_BOS"]

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

if __name__ == "__main__":
    str = raw_input('Input Sentence:')
    print parser(str)
