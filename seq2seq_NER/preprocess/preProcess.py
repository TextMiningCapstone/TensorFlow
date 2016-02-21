import dataUtl

train_file_path = "../ner/eng.train.txt"
test_file_path = "../ner/eng.test.txt"

dataUtl.genTrain(train_file_path)
dataUtl.genSentence(train_file_path)
dataUtl.genSentence(test_file_path)