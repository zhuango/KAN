#!/usr/bin/python3

import numpy as np
import math

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module


torch.manual_seed(2333)
myRandom = np.random.RandomState(2333)

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

class EncoderLayer(Module):
    def __init__(self, headSize, vq, vk, vv, vectorLength, hiddenLayer=400, cuda=True, dropout = 0.0):
        super(EncoderLayer, self).__init__()
        self.head =  headSize
        self.vectorLength = vectorLength
        self.softmax = nn.Softmax()
        self.headLength = int(self.vectorLength / self.head)
        self.multiAtt_Ws = []
        self.cuda = cuda

        self.Wo = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (vectorLength, vectorLength))), cuda, requires_grad=True)
        for i in range(self.head):
            headProject_wq = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vq))), cuda, requires_grad=True)
            headProject_wk = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vk))), cuda, requires_grad=True)
            headProject_wv = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.1, 0.1, (self.headLength, vv))), cuda, requires_grad=True)            
            ws = {}
            ws['wq'] = headProject_wq
            ws['wk'] = headProject_wk
            ws['wv'] = headProject_wv
            self.register_parameter("pro_q_h{}".format(i), headProject_wq)
            self.register_parameter("pro_k_h{}".format(i), headProject_wk)
            self.register_parameter("pro_v_h{}".format(i), headProject_wv)

            self.multiAtt_Ws.append(ws)

        if cuda:
            self.ff_W1 = nn.Conv1d(vectorLength, hiddenLayer, 1).cuda()
            self.ff_W2 = nn.Conv1d(hiddenLayer, vectorLength, 1).cuda()
        else:
            self.ff_W1 = nn.Conv1d(vectorLength, hiddenLayer, 1)
            self.ff_W2 = nn.Conv1d(hiddenLayer, vectorLength, 1)

        self.ReLU = nn.ReLU()
        self.dropoutTrain = nn.Dropout(dropout)
        self.dropoutTest  = nn.Dropout(0.0)
        self.dropout = self.dropoutTrain
        
        self.a_2 = ParameterDevice(torch.ones(vectorLength), cuda, requires_grad=True)
        self.b_2 = ParameterDevice(torch.zeros(vectorLength), cuda, requires_grad=True)

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

    def __call__(self, Q, K, V, training=True):
        if training:
            self.dropout = self.dropoutTrain
        else:
            self.dropout = self.dropoutTest
        output = self.multiHeadAttention(Q, K, V)
        output = self.AddNorm(output.transpose(0, 1) + self.ff(output).transpose(0, 1)).transpose(0, 1)
        return output

class KAN(Module):
    def __init__(self, wordEmbed, entityEmbed, wordVectorLength, entityVecSize, hopNumber, classNumber, cuda=True):
        super(KAN, self).__init__()
        self.wordEmbed = wordEmbed
        self.entityEmbed = entityEmbed
        self.wordVectorLength = wordVectorLength
        self.vectorLength = wordVectorLength
        self.entityVecSize = entityVecSize
        self.hopNumber = hopNumber
        self.classNumber = classNumber
        self.cuda = cuda
        self.encodings = {}
        self.training = True

        self.attention_W0 = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength, self.vectorLength))), cuda, requires_grad=True)
        self.attention_b0 = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (self.vectorLength,1))), cuda, requires_grad=True)

        self.softmax_W = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, self.vectorLength*2 + wordVectorLength))), cuda, requires_grad=True)
        self.softmax_b = ParameterDevice(torch.FloatTensor(myRandom.uniform(-0.01, 0.01, (classNumber, 1))), cuda, requires_grad=True)
        
        self.encoderLayer0 = EncoderLayer(4, self.vectorLength*2, self.vectorLength, self.vectorLength, self.vectorLength, cuda=cuda, dropout=0.5)
        self.encoderLayer1 = EncoderLayer(4, self.vectorLength*2, self.vectorLength, self.vectorLength, self.vectorLength, cuda=cuda, dropout=0.5)

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
        
    def forward(self, contxtWords, e1, e2, e1p, e2p, relation, sentLength, training=True):

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
            output0 = self.encoderLayer0(contxtWords0withEntity, output0, output0, training)

        output1 = contxtWords1
        for hop in range(self.hopNumber):
            contxtWords1withEntity = torch.cat([output1, e2.expand(entityVecSize,sentLength)])
            output1 = self.encoderLayer1(contxtWords1withEntity, output1, output1, training)

        attentionA = torch.mm(self.attention_W0, output0) + self.attention_b0.expand(vectorLength, sentLength)
        output0 = torch.sum(output0 * self.softmax(torch.tanh(attentionA)), 1, keepdim=True)
        
        attentionA = torch.mm(self.attention_W0, output1) + self.attention_b0.expand(vectorLength, sentLength)
        output1 = torch.sum(output1 * self.softmax(torch.tanh(attentionA)), 1, keepdim=True)

        output = torch.cat([output0, output1, relation])
        finallinearLayerOut = torch.mm(softmax_W, output) + softmax_b
        return finallinearLayerOut
