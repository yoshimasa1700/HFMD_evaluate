#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sys
import os

argvs = sys.argv
argc = len(argvs)

if(argc > 1):
	inputFileName = argvs[1]
else:
	inputFileName = './detectionResult.txt'

if(argc > 2):
	outputFileName = argvs[2]
else:
	outputFileName = './analyzedResult.txt'

if(argc > 3):
	os.system('cp %s ./%s'% (inputFileName, argvs[3]))
	
sta = 0.00
sto = 20.0
ste = 0.1
preval = 0.0

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

TPPERT = []
FPPERT = []
FPPERT_SUM = 0
outFile = open(outputFileName, 'w')

for th in drange(sta, sto , ste):
	print th
	resultFile = open(inputFileName)
	dataonly = resultFile.readline()
	data = []
	TP = 0
	TN = 0
	FP = 0
	FN = 0

	for line in resultFile:
		data = line.split()
		if data[0] == data[1]:
			if float(data[2]) > th:
				TP += 1
			else:
				FP += 1
		else:
			if float(data[2]) > th:
				TN += 1
			else:
				FN += 1
	resultFile.close()
	print "TP = %d , FP = %d" % (TP, FP)
	print "TN = %d , FN = %d" % (TN, FN)
	precision = float(TP) / ( float(TP) + float(FP) + 0.0000001)
	recall = float(TP) / (float(TP) + float(FN) + 0.0000001)

	FPF = float(TN) / (float(TN) + float(FN))
	F = 2 * recall * precision / (recall + precision + 0.000001)
	accuracy = (float(TP) + float(TN)) / (float(TP) + float(TN) + float(FP) + float(FN))
	print "precision = %lf , recall = %lf , F = %lf , accuracy = %lf , FPF = %lf" % (precision,recall, F, accuracy, FPF)

	outFile.write("%lf %lf %lf %lf% lf %lf\n" %(th, precision, recall, F, accuracy, FPF))
	TPPERT.append(precision)
	FPPERT.append(FPF)
	FPPERT_SUM += precision * (FPF - preval)
	preval = FPF

print FPPERT_SUM
outFile.write("%lf"%(FPPERT_SUM))
outFile.close()

plt.plot(TPPERT,FPPERT)
plt.show()
