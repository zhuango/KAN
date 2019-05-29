singleSentences = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_sentence_win3_39debug.txt"
crossSentences  = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_sentence_whole2Sentence_39debug.txt"
fullCorpus = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_win3_39debug.txt"

# singleSentences = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_sentence_ForTest_win3_39debug.txt"
# crossSentences  = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_sentence_ForTest_whole2Sentence_39debug.txt"
# fullCorpus = "/home/laboratory/lab/ACL2017/corpus/BioCreative/corpus_train_ForTest_win3_39debug.txt"

pairDict = {}

with open(singleSentences, 'r') as f:
    for line in f:
        items = line.strip().split("\t")
        uniStr0= "_".join([items[0], items[2], items[3]])
        uniStr1= "_".join([items[0], items[3], items[2]])
        if uniStr0 in pairDict:
            pairDict[uniStr0].append(line)
        elif uniStr1 in pairDict:
            pairDict[uniStr1].append(line)
        else:
            pairDict[uniStr0] = []
            pairDict[uniStr0].append(line)
with open(crossSentences, 'r') as f:
    for line in f:
        items = line.strip().split("\t")
        pairStr0 = "_".join([items[0], items[2], items[3]])
        pairStr1 = "_".join([items[0], items[3], items[2]])
        if pairStr0 in pairDict:
            pairDict[pairStr0].append(line)
        elif pairStr1 in pairDict:
            pairDict[pairStr1].append(line)
        else:
            pairDict[uniStr0] = []
            pairDict[uniStr0].append(line)

fullStream = open(fullCorpus, 'w')
for key in pairDict:
    fullStream.write("{}\n".format(len(pairDict[key])))
    for sentence in pairDict[key]:
        fullStream.write(sentence)
fullStream.close()