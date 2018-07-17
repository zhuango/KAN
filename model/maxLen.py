#!/usr/bin/python3

def maxLen(filename):
    maxlen = 0
    maxELen = 0
    with open(filename) as f:
        for line in f:
            items = line.strip().split("\t")
            if len(items) < 2:
                continue
            curlen= len(items[1].split(" "))
            if curlen > maxlen:
                maxlen = curlen
            curELen = max(len(items[4].split(" ")),len(items[5].split(" ")))
            if curELen > maxELen:
                maxELen = curELen
    return maxlen + maxELen*2

if __name__ == "__main__":
    maxlen = maxLen("/home/laboratory/lab/sem2013/DDI/DDIcorpus/testSamples.txt")
    print(maxlen)
