# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:23:11 2013

@author: masahiko
"""

dataFile = open('./detectionResult.txt')

TP = 0
FP = 0
FN = 0
TN = 0

test=[]

for line in dataFile:
    data = line.split(' ')
    if data[0] == data[1]:
        print str('correct')
        test.append(data)
    else:
        print str('incorrect')

dataFile.close()
