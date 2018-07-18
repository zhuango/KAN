#!/usr/bin/python3

def mergeResult(resultPath, mergedResultPath):
    dic = {}
    with open(resultPath,'r') as f:
        for line in f:
            items = line.strip().split("\t")
            if items[0] not in dic:
                dic[items[0]] = []
            pairstr0 = items[1] + "\t" + items[2]
            pairstr1 = items[2] + "\t" + items[1]
            if pairstr0 in dic[items[0]]:
                continue
            if pairstr1 in dic[items[0]]:
                continue
            dic[items[0]].append(items[1] + "\t" + items[2])
    with open(mergedResultPath, 'w') as f:
        for key in dic:
            for pairStr in dic[key]:
                f.write(key + "\t" + pairStr + "\n")