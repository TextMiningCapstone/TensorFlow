with open("input", "r") as myfile:
    for line in myfile:
        val = line.strip().split(" ")
        print len(val)