#!/usr/bin/python3

import os
import time
import argparse

from KAN import *
from LoadDataVariables import *
from merge import mergeResult

cuda = False#torch.cuda.is_available()
softmax = torch.nn.Softmax()
loss_function = torch.nn.NLLLoss()

def prf(predicPath, goldPath):
    p = 0.0
    r = 0.0
    f = 0.0
    sampleCount = 0
    goldCount = 0
    ppCount = 0

    goldDict = {}
    with open(goldPath, 'r') as goldStream:
        for line in goldStream:
            items= line.strip().split("\t")
            if items[0] not in goldDict:
                goldDict[items[0]] = []
            goldDict[items[0]].append(items[1] + "\t" + items[2])
            goldCount += 1

    with open(predicPath, 'r') as predicStream:
        for line in predicStream:
            items = line.strip().split("\t")
            if items[0] not in goldDict:
                #print(items[0])
                #sampleCount += 1
                continue
            gold = goldDict[items[0]]
            pairStr0 = items[1] + "\t" + items[2]
            pairStr1 = items[2] + "\t" + items[1]
            
            if pairStr0 in gold:
                ppCount += 1
            elif pairStr1 in gold:
                ppCount += 1

            sampleCount += 1
    try:
        # print(sampleCount)
        p = float(ppCount) / float(sampleCount)
        r = float(ppCount) / float(goldCount)
        f = 2 * p * r / (p + r)
    except:
        return 0.0, 0.0, 0.0
    return p*100, r*100, f*100


def splitData(trainset, docR=0.15):
    pmids = {}
    validSet = []
    trainSet = []

    # Calculate instance of each document.
    for sample in trainset:
        if sample[0] not in pmids:
            pmids[sample[0]] = 0
        if sample[-1] == "1":
            pmids[sample[0]] += 1

    sortedpmids = sorted(pmids.keys(), key=lambda pmid: pmids[pmid])
    # Entities in testset are annotated by GNormPlus, which miss many interacting protein entities.
    # Selecting the document that having least positive instances as valid set makes valid set and test set have similar distribution.
    docCount = int(docR * len(pmids))
    validPmids = set(sortedpmids[0:docCount])
    # Add sample which pmid are in validPmids to validSet.
    for sample in trainset:
        if sample[0] in validPmids:
            validSet.append(sample)
        else:
            trainSet.append(sample)

    return trainSet, validSet

def train(kan, trainset, validsetR, testset, trainGold, testGold, lr, wdecay, paraPathPref='./parameters/model'):

    optimizer = optim.Adadelta(kan.parameters(), lr=lr, weight_decay=wdecay)
    
    trainset, validset = splitData(trainset, validsetR)
    trainsetSize = len(trainset)
    test_idx = 0
    maxf  = 0.0
    maxvf = 0.0
    
    for epoch_idx in range(numEpoches):
        myRandom.shuffle(trainset)
        sum_loss = VariableDevice(torch.zeros(1), cuda)
        print("=====================================================================")
        print("epoch " + str(epoch_idx) + ", trainSize: " + str(trainsetSize))
        print("hop size: ", str(hopNumber))

        correct = 0
        totalLoss = 0
        instanceCount = 0
        time0 = time.time()  
        optimizer.zero_grad()
        for sample in trainset:
            sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample
            finallinearLayerOut = kan.forward(contxtWords, e1, e2, e1p, e2p, relation, sentLength)

            calssification = softmax(finallinearLayerOut.view(1, classNumber))
            predict = np.argmax(calssification.cpu().data.numpy())
            if predict == label:
                correct += 1

            log_prob = F.log_softmax(finallinearLayerOut.view(1, classNumber))
            loss = loss_function(log_prob, VariableDevice(torch.LongTensor([label]), cuda))
            sum_loss += loss
            instanceCount += 1
        ###################Update#######################
            if instanceCount >= batchSize:
                sum_loss = sum_loss / instanceCount
                totalLoss += sum_loss.cpu().data.numpy()[0]
                instanceCount = 0
                sum_loss.backward()
                optimizer.step()
                sum_loss = VariableDevice(torch.zeros(1), cuda)
                optimizer.zero_grad()
        if instanceCount != 0:
            sum_loss = sum_loss / instanceCount
            totalLoss += sum_loss.cpu().data.numpy()[0]
            sum_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ####################Update#######################

        time1 = time.time()
        print("Iteration", epoch_idx, "Loss", totalLoss / trainsetSize, "train Acc: ", float(correct / trainsetSize) , "time: ", str(time1 - time0))
        
        # Valid
        currentValidResult = resultOutput + "result_valid_" + str(epoch_idx) + ".txt"
        mergedValidResult = currentValidResult + ".merged"
        resultValidStream = open(currentValidResult, 'w')

        test(kan, validset, resultValidStream)
        resultValidStream.close()
        mergeResult(currentValidResult, mergedValidResult)
        p, r, f = prf(mergedValidResult, trainGold)
        print("valid P: {} R: {} F: {}".format(p, r, f))

        if f > maxvf:
            maxvf = f
            # Test
            currentResult = resultOutput + "result_" + str(test_idx) + ".txt"
            mergedResult = currentResult + ".merged"
            resultStream = open(currentResult, 'w')
            probPath   = resultOutput + "prob_" + str(test_idx) + ".txt"
            test(kan, testset, resultStream, probPath)
            resultStream.close()
            
            mergeResult(currentResult, mergedResult)
            p, r, f = prf(mergedResult, testGold)
            torch.save(kan.state_dict(), paraPathPref)
            print("test P: {} R: {} F: {}".format(p, r, f))
            test_idx += 1

def test(kan, testSet, resultStream=None, probPath=None):
    count = 0
    correct = 0
    time0 = time.time()
    probs = []
    for sample in testSet:
        sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample

        finallinearLayerOut = kan.forward(contxtWords, e1, e2, e1p, e2p, relation, sentLength, False)
        calssification = softmax(finallinearLayerOut.view(1, classNumber))
                
        prob = calssification.cpu().data.numpy().reshape(classNumber)
        predict = np.argmax(prob)
        probs.append(prob)
        if resultStream and predict == 1:
            resultStream.write("\t".join([sentid, e1ID, e2ID]) + "\n")

        if predict == label:
                correct += 1.0
        count += 1
    if probPath:
        np.savetxt(probPath, probs, '%.5f',delimiter=' ')
    time1 = time.time()
    acc = correct/count
    print("test Acc: ", acc)
    print("time    : ", str(time1 - time0))

def GetInstanceProperty(sample):
    words = sample[1].split(" ")
    e1Index = words.index("$1")
    e2Index = words.index("$2")
    e1p = []
    e2p = []
    for i in range(len(words)):
        if e1Index == i or e2Index == i:
            continue
        e1p.append(e1Index - i)
        e2p.append(e2Index - i)
    words = sample[1].replace("$1", " ").replace("$2", " ").replace("  ", "").split(" ")
    n = len(words)
    contxtWords = VariableDevice(torch.LongTensor([word2id[words[i]] for i in range(0, len(words))]), cuda)

    sentid = sample[0]
    e1ID = sample[2]
    e1   = "-1" 
    if e1ID in entity2id:
        e1 = entity2id[e1ID]
        e1V = VariableDevice(torch.LongTensor([int(e1)]), cuda)
        e1V = entityEmbed(e1V).transpose(0,1)
    else:
        entityWords = sample[4].split(" ")
        e1V = torch.sum(wordEmbed(VariableDevice(torch.LongTensor([word2id[entityWord] for entityWord in entityWords]), cuda)), 0) / len(entityWords)
        e1V = e1V.view(wordVectorLength,1)

    e2ID = sample[3]
    e2   = "-1"
    if e2ID in entity2id:
        e2 = entity2id[e2ID]
        e2V = VariableDevice(torch.LongTensor([int(e2)]), cuda)
        e2V = entityEmbed(e2V).transpose(0,1)
    else:
        entityWords = sample[5].split(" ")
        e2V = torch.sum(wordEmbed(VariableDevice(torch.LongTensor([word2id[entityWord] for entityWord in entityWords]), cuda)), 0) / len(entityWords)
        e2V = e2V.view(wordVectorLength,1)

    pairStr0 = str(e1) + "_" + str(e2)
    pairStr1 = str(e2) + "_" + str(e1)
    if pairStr0 in triples:
        relation = relation2vector[triples[pairStr0]]
    elif pairStr1 in triples:
        relation = relation2vector[triples[pairStr1]]
    else:
        # a zero vector.
        relation = relation2vector['unknow']
    
    label = int(sample[-1])

    return sentid, contxtWords, e1ID, e1V, e2ID, e2V, relation, n, label, e1p, e2p


# Hyperparameter setting:
# python3 main.py 
# --trainPath ../data/train.txt
# --testPath ../data/test.txt
# --batchSize 100 
# --wd 100 
# --ed 100 
# --hop 2 
# --clas 2 
# --epoch 20 
# --wePath ../data/wordEmb/bio-word2id100
# --w2IDPath ../data/wordEmb/bio-embed100
# --eePath ../data/KB/entity2vec.vec
# --rePath ../data/KB/relation2vec.vec
# --t2idPath ../data/KB/triple2id.txt
# --e2idPath ../data/KB/entity2id.txt
# --paraPath ./parameters/
# --results ./results/

parser = argparse.ArgumentParser()
parser.add_argument("--trainPath", default="../data/train.txt")
parser.add_argument("--testPath", default="../data/test.txt")
parser.add_argument("--validPath", default="../data/valid.txt")
parser.add_argument("--trainGold", default="../data/trainGold.txt")
parser.add_argument("--testGold", default="../data/testGold.txt")
parser.add_argument("--batchSize", default=100, type=int)
parser.add_argument("--wd", default=100, type=int)
parser.add_argument("--ed", default=100, type=int)
parser.add_argument("--hop", default=2, type=int)
parser.add_argument("--clas", default=2, type=int)
parser.add_argument("--epoch", default=40, type=int)
parser.add_argument("--wePath", default="../data/wordEmb/bio-word2id100")
parser.add_argument("--w2IDPath", default="../data/wordEmb/bio-embed100")
parser.add_argument("--eePath", default="../data/KB/entity2vec.vec")
parser.add_argument("--rePath", default="../data/KB/relation2vec.vec")
parser.add_argument("--t2idPath", default="../data/KB/triple2id.txt")
parser.add_argument("--e2idPath", default="../data/KB/entity2id.txt")
parser.add_argument("--paraPath", default="./parameters/kan")
parser.add_argument("--results", default="./results/")
parser.add_argument("--training", default=True, type=bool)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--wdecay", default=0.0, type=float)
parser.add_argument("--validsetR", default=0.15, type=float)
args = parser.parse_args()

batchSize           = args.batchSize
wordVectorLength    = args.wd
entityVecSize       = args.ed
hopNumber           = args.hop
classNumber         = args.clas
numEpoches          = args.epoch
paraPath            = args.paraPath
training            = args.training
lr                  = args.lr
wdecay              = args.wdecay

print(training)
print("Loading word id...")
word2id = LoadWord2id(args.wePath)
print("Loading word vectors...")
ten = np.loadtxt(args.w2IDPath)
wordEmbed = nn.Embedding(ten.shape[0], wordVectorLength)
wordEmbed.weight = ParameterDevice(torch.FloatTensor(ten), cuda)

print("Loading entity vectors...")
entity2vector = np.loadtxt(args.eePath)
entityEmbed = nn.Embedding(entity2vector.shape[0], entity2vector.shape[1])
entityEmbed.weight = ParameterDevice(torch.FloatTensor(entity2vector), cuda)

print("Loading relation vectors...")
relation2vector = LoadRelationVectors(args.rePath, entityVecSize, cuda)
print("Loading triples...")
triples = LoadTriples(args.t2idPath)
print("Loading entity id mapping...")
entity2id = LoadEntity2Id(args.e2idPath)

resultOutput = args.results
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)

testsPath = args.testPath
testGold  = args.testGold
print("Load test samples...")
testSet = LoadSamples(testsPath)
testset = []
for sample in testSet:
    sampleTuple = GetInstanceProperty(sample)
    testset.append(sampleTuple)
    
kan = KAN(wordEmbed, entityEmbed, wordVectorLength, entityVecSize, hopNumber, classNumber, cuda=cuda)

if training:
    validsetR = args.validsetR
    trainPath = args.trainPath
    trainGold = args.trainGold
    print("Load training samples...")
    trainSet = LoadSamples(trainPath)

    trainset = []
    for sample in trainSet:
        sampleTuple = GetInstanceProperty(sample)
        trainset.append(sampleTuple)

    train(kan, trainset, validsetR, testset, trainGold, testGold, lr, wdecay, paraPath)
else:
    kan.load_state_dict(torch.load(paraPath))
    
    currentResult = resultOutput + "result.txt"
    mergedResult = currentResult + ".merged"
    resultStream = open(currentResult, 'w')
    probPath   = resultOutput + "prob.txt"
    test(kan, testset, resultStream, probPath)
    resultStream.close()
            
    mergeResult(currentResult, mergedResult)
    p, r, f = prf(mergedResult, testGold)
    print("test P: {} R: {} F: {}".format(p, r, f))