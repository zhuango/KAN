#!/usr/bin/python3

import numpy as np
import math

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


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
        self.cuda = cuda

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
        mask = VariableDevice(mask, self.cuda)
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
    def __init__(self, wordEmbed, entityEmbed, wordVectorLength, entityVecSize, hopNumber, classNumber, cuda=True):
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.wordVectorLength = wordVectorLength
        self.vectorLength = wordVectorLength
        self.entityVecSize = entityVecSize
        self.hopNumber = hopNumber
        self.classNumber = classNumber
        self.cuda = cuda
        self.parameters = []
        self.encodings = {}

        self.attention_W0 = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength, self.vectorLength))), cuda, requires_grad=True)
        self.attention_b0 = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength,1))), cuda, requires_grad=True)
        self.parameters.append(self.attention_W0)
        self.parameters.append(self.attention_b0)

        self.softmax_W = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, self.vectorLength*2 + wordVectorLength))), cuda, requires_grad=True)
        self.softmax_b = VariableDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, 1))), cuda, requires_grad=True)
        self.parameters.append(self.softmax_W)
        self.parameters.append(self.softmax_b)
        
        self.encoderLayer0 = EncoderLayer(4, self.vectorLength*2, self.vectorLength, self.vectorLength, self.vectorLength, cuda=cuda, dropout=0.1)
        self.parameters += self.encoderLayer0.parameters

        self.encoderLayer1 = EncoderLayer(4, self.vectorLength*2, self.vectorLength, self.vectorLength, self.vectorLength, cuda=cuda, dropout=0.1)
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
                self.encodings[pos] = VariableDevice(torch.FloatTensor(encoding), self.cuda)
            posMatrix.append(self.encodings[pos])
        return torch.cat(posMatrix)
        
    def forward(self, contxtWords, e1, e2, e1p, e2p, relation, sentLength):

        softmax_W = self.softmax_W
        softmax_b = self.softmax_b
        vectorLength = self.vectorLength
        entityVecSize = self.entityVecSize

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
