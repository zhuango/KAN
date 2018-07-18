#!/usr/bin/python3

import numpy as np
import os
import matplotlib.pyplot as plt
import math

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from collections import defaultdict
import time
import argparse

from LoadDataVariables import *
from maxLen import maxLen
from merge import mergeResult


torch.manual_seed(4324)
myRandom = np.random.RandomState(2543)

def VariableDevice(data, cuda=True, requires_grad=False):
    if cuda:
        return Variable(data.cuda(), requires_grad=requires_grad)
    else:
        return Variable(data, requires_grad=requires_grad)

def ParameterDevice(data, cuda=True, requires_grad=False):
    if cuda:
        return torch.nn.Parameter(data.cuda(), requires_grad=requires_grad)
    else:
        return torch.nn.Parameter(data, requires_grad=requires_grad)

class EncoderLayer():
    def __init__(self, headSize, vq, vk, vv, vectorLength, hiddenLayer=400, cuda=True, dropout = 0.0):
        self.parameters = []
        self.head =  headSize
        self.vectorLength = vectorLength
        self.softmax = nn.Softmax()
        self.headLength = int(self.vectorLength / self.head)
        self.multiAtt_Ws = []

        self.Wo = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (vectorLength, vectorLength))), cuda, requires_grad=True)
        for i in range(self.head):
            headProject_wq = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vq))), cuda, requires_grad=True)
            headProject_wk = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vk))), cuda, requires_grad=True)
            headProject_wv = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vv))), cuda, requires_grad=True)            
            ws = {}
            ws['wq'] = headProject_wq
            ws['wk'] = headProject_wk
            ws['wv'] = headProject_wv

            self.multiAtt_Ws.append(ws)

            self.parameters.append(headProject_wq)
            self.parameters.append(headProject_wk)
            self.parameters.append(headProject_wv)
        self.parameters.append(self.Wo)

        if cuda:
            self.ff_W1 = nn.Conv1d(vectorLength, hiddenLayer, 1).cuda()
            self.ff_W2 = nn.Conv1d(hiddenLayer, vectorLength, 1).cuda()
        else:
            self.ff_W1 = nn.Conv1d(vectorLength, hiddenLayer, 1)
            self.ff_W2 = nn.Conv1d(hiddenLayer, vectorLength, 1)
        self.parameters += list(self.ff_W1.parameters())
        self.parameters += list(self.ff_W2.parameters())

        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.a_2 = ParameterDevice(torch.ones(vectorLength), cuda, requires_grad=True)
        self.b_2 = ParameterDevice(torch.zeros(vectorLength), cuda, requires_grad=True)
        self.parameters.append(self.a_2)
        self.parameters.append(self.b_2)

    def AddNorm(self, z, eps=1e-3):
        if z.size(0) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

    def ff(self, input):
        output = self.ReLU(self.ff_W1(input.unsqueeze(0)))
        output = self.ff_W2(output)
        output = output.squeeze(0)
        output = self.dropout(output)
        return output

    def multiHeadAttention(self, Q, K, V):
        mask = torch.zeros(Q.size()[1], Q.size()[1])
        for i in range(Q.size()[1]):
            mask[i][i] = -100000.0
        mask = VariableDevice(mask, cuda)
        heads = []
        for i in range(self.head):
            Vq = torch.mm(self.multiAtt_Ws[i]['wq'], Q)
            Vp = torch.mm(self.multiAtt_Ws[i]['wv'], V)
            Vk = torch.mm(self.multiAtt_Ws[i]['wk'], K)
            scores = torch.mm(Vq.transpose(0,1), Vk)
            weights = self.softmax(scores / math.sqrt(self.headLength) + mask)
            head = torch.mm(weights, Vp.transpose(0, 1))
            heads.append(head)
        output = torch.cat(heads, 1)
        output = torch.mm(output, self.Wo)
        output = self.AddNorm(V.transpose(0, 1) + self.dropout(output))
        return output.transpose(0,1)

    def __call__(self, Q, K, V):
        output = self.multiHeadAttention(Q, K, V)
        output = self.AddNorm(output.transpose(0, 1) + self.ff(output).transpose(0, 1)).transpose(0, 1)
        return output

class KAN():
    def __init__(self, wordEmbed, entityEmbed, batchSize, wordVectorLength, hopNumber, classNumber, numEpoches, cuda=True):
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.batchSize = batchSize
        self.wordVectorLength = wordVectorLength
        self.vectorLength = wordVectorLength
        self.hopNumber = hopNumber
        self.classNumber = classNumber
        self.numEpoches = numEpoches
        self.cuda = cuda
        self.parameters = []
        self.encodings = {}

        self.attention_W0 = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (vectorLength, vectorLength))), cuda, requires_grad=True)
        self.attention_b0 = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (vectorLength,1))), cuda, requires_grad=True)
        self.parameters.append(self.attention_W0)
        self.parameters.append(self.attention_b0)

        self.softmax_W = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, vectorLength*2 + wordVectorLength))), cuda, requires_grad=True)
        self.softmax_b = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, 1))), cuda, requires_grad=True)
        self.parameters.append(self.softmax_W)
        self.parameters.append(self.softmax_b)
        
        self.encoderLayer0 = EncoderLayer(4, vectorLength*2, vectorLength, vectorLength, vectorLength, cuda=cuda, dropout=0.1)
        self.parameters += self.encoderLayer0.parameters

        self.encoderLayer1 = EncoderLayer(4, vectorLength*2, vectorLength, vectorLength, vectorLength, cuda=cuda, dropout=0.1)
        self.parameters += self.encoderLayer1.parameters

        self.softmax = torch.nn.Softmax()

    def positionEncoding(self, posList):
        posMatrix = []
        for pos in posList:
            if pos not in self.encodings:
                encoding = np.zeros((1, self.wordVectorLength))
                for i in range(self.wordVectorLength):
                    if i % 2 == 0:
                        encoding[0][i] = np.sin(pos / np.power(10000, i / self.wordVectorLength))
                    else:
                        encoding[0][i] = np.cos(pos / np.power(10000, i / self.wordVectorLength))
                self.encodings[pos] = VariableDevice(torch.FloatTensor(encoding), cuda)
            posMatrix.append(self.encodings[pos])
        return torch.cat(posMatrix)
        
    def forward(self, contxtWords, e1, e2, e1p, e2p, relation, sentLength):

        softmax_W = self.softmax_W
        softmax_b = self.softmax_b
        vectorLength = self.vectorLength

        contxtWords = self.wordEmbed(contxtWords)
        contxtWords0 = contxtWords + self.positionEncoding(e1p)
        contxtWords1 = contxtWords + self.positionEncoding(e2p)

        contxtWords0 = contxtWords0.transpose(0,1)
        contxtWords1 = contxtWords1.transpose(0,1)

        output0 = contxtWords0
        for hop in range(self.hopNumber):
            contxtWords0withEntity = torch.cat([output0, e1.expand(entityVecSize,sentLength)])
            output0 = self.encoderLayer0(contxtWords0withEntity, output0, output0)

        output1 = contxtWords1
        for hop in range(self.hopNumber):
            contxtWords1withEntity = torch.cat([output1, e2.expand(entityVecSize,sentLength)])
            output1 = self.encoderLayer1(contxtWords1withEntity, output1, output1)

        attentionA = torch.mm(self.attention_W0, output0) + self.attention_b0.expand(vectorLength, sentLength)
        output0 = torch.sum(output0 * self.softmax(torch.tanh(attentionA)), 1, keepdim=True)
        
        attentionA = torch.mm(self.attention_W0, output1) + self.attention_b0.expand(vectorLength, sentLength)
        output1 = torch.sum(output1 * self.softmax(torch.tanh(attentionA)), 1, keepdim=True)

        output = torch.cat([output0, output1, relation])
        finallinearLayerOut = torch.mm(softmax_W, output) + softmax_b
        return finallinearLayerOut

def train(kan, trainset, paraPathPref='./parameters/model'):
    maxp = 0
    maxr = 0
    maxf = 0
    maxacc = 0
    trainsetSize = len(trainset)

    optimizer = optim.Adadelta(kan.parameters, lr=0.1)
         
    for epoch_idx in range(kan.numEpoches):
        myRandom.shuffle(trainset)
        sum_loss = VariableDevice(torch.zeros(1), cuda)
        print("=====================================================================")
        print("epoch " + str(epoch_idx) + ", trainSize: " + str(trainsetSize))
        print("hop size: ", str(kan.hopNumber))

        correct = 0
        instanceCount = 0
        time0 = time.time()  
        optimizer.zero_grad()  
        for sample in trainset:
            sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample
            finallinearLayerOut = kan.forward(contxtWords, 
                                            e1, 
                                            e2, 
                                            e1p,
                                            e2p,
                                            relation,
                                            sentLength)

            calssification = kan.softmax(finallinearLayerOut.view(1, classNumber))
            predict = np.argmax(calssification.cpu().data.numpy())
            if predict == label:
                correct += 1

            log_prob = F.log_softmax(finallinearLayerOut.view(1, classNumber))
            loss = loss_function(log_prob, VariableDevice(torch.LongTensor([label]), cuda))
            sum_loss += loss
            instanceCount += 1
        ###################Update#######################
            if instanceCount >= kan.batchSize:
                sum_loss = sum_loss / instanceCount
                instanceCount = 0
                sum_loss.backward()
                optimizer.step()
                sum_loss = VariableDevice(torch.zeros(1), cuda)
                for para in kan.parameters:
                    para._grad.data.zero_()
                optimizer.zero_grad()
        if instanceCount != 0:
            sum_loss = sum_loss / instanceCount
            sum_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ####################Update#######################

        time1 = time.time()
        print("Iteration", epoch_idx, "Loss", sum_loss.cpu().data.numpy()[0] / kan.batchSize, "train Acc: ", float(correct / trainsetSize) , "time: ", str(time1 - time0))
            
        currentResult = resultOutput + "result_" + str(epoch_idx) + ".txt"
        mergedResult = currentResult + ".merged"
        resultStream = open(currentResult, 'w')
        probPath   = resultOutput + "prob_" + str(epoch_idx) + ".txt"
        test(testset, resultStream, probPath)
        resultStream.close()
                
        mergeResult(currentResult, mergedResult)

def test(kan, testSet, resultStream=None, probPath=None):
    count = 0
    correct = 0
    time0 = time.time()
    probs = []
    for sample in testSet:
        sentid, contxtWords, e1ID, e1, e2ID, e2, relation, sentLength, label, e1p, e2p = sample

        finallinearLayerOut = kan.forward(contxtWords, 
                                            e1, 
                                            e2,
                                            e1p,
                                            e2p,
                                            relation,
                                            sentLength)
        calssification = kan.softmax(finallinearLayerOut.view(1, classNumber))
                
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
    return acc


def GetSampleProperty(sample):
    contxtWords = []
    n = []
    e1ps = []
    e2ps = []
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

batchSize = 100
wordVectorLength = 100
vectorLength = wordVectorLength
positinoVecLen = vectorLength
entityVecSize = 100
hopNumber = 2
classNumber = 2
numEpoches = 20
cuda = torch.cuda.is_available()

print("Load word id...")
word2id = LoadWord2id("../data/wordEmb/bio-word2id100")
print("Load word vectors...")
ten = np.loadtxt("../data/wordEmb/bio-embed100")
wordEmbed = nn.Embedding(ten.shape[0], wordVectorLength)
wordEmbed.weight = ParameterDevice(torch.FloatTensor(ten), cuda)

print("Load entity vectors...")
entity2vector = np.loadtxt("../data/KB/entity2vec.vec")
entityEmbed = nn.Embedding(entity2vector.shape[0], entity2vector.shape[1])
entityEmbed.weight = ParameterDevice(torch.FloatTensor(entity2vector), cuda)

print("Load relation vectors...")
relation2vector = LoadRelationVectors("../data/KB/relation2vec.vec", wordVectorLength, cuda)
print("Load triples...")
triples = LoadTriples("../data/KB//triple2id.txt")
print("Load entity id mapping...")
entity2id = LoadEntity2Id("../data/KB/entity2id.txt")

loss_function = torch.nn.NLLLoss()

fold = 1
paraPathPref = './parameters/'
resultOutput = './results/'
if not os.path.exists(resultOutput):
    os.makedirs(resultOutput)
if not os.path.exists(paraPathPref):
    os.makedirs(paraPathPref)
trainPath = "../data/train.txt"
testsPath = "../data/test.txt"
print("Load training samples...")
trainSet = LoadSamples(trainPath)
print("Load test samples...")
testSet = LoadSamples(testsPath)
trainset = []
for sample in trainSet:
    sampleTuple = GetSampleProperty(sample)
    trainset.append(sampleTuple)

testset = []
for sample in testSet:
    sampleTuple = GetSampleProperty(sample)
    testset.append(sampleTuple)
print(len(testset))
kan = KAN(wordEmbed, entityEmbed, batchSize, wordVectorLength, hopNumber, classNumber, numEpoches, cuda)
train(kan, trainset, paraPathPref)
