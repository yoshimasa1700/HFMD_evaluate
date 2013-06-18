# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def drange(start, stop, step):
	r = start
	while r < stop:
		yield r
		r += step

TPPERT = []
FPPERT = []

for th in drange(0.00, 0.10 , 0.001):
	print th
	resultFile = open('../detectionResult.txt')
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
	precision = float(TP) / ( float(TP) + float(FP) )
	recall = float(TP) / (float(TP) + float(FN))

	FPF = float(TN) / (float(TN) + float(FN))
	F = 2 * recall * precision / (recall + precision + 0.000001)
	accuracy = (float(TP) + float(TN)) / (float(TP) + float(TN) + float(FP) + float(FN))
	print "precision = %lf , recall = %lf , F = %lf , accuracy = %lf , FPF = %lf" % (precision,recall, F, accuracy, FPF)
	TPPERT.append(precision)
	FPPERT.append(FPF)

plt.plot(TPPERT,FPPERT,'r*')
plt.show()