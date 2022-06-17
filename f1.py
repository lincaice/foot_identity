import numpy as np

precision = [0.68488371,0.84603282,0.92660737,0.94675101,0.80574554,0.92660737,0.84603282,0.84603282,0.90646373,0.94675101]
# 召回率矩阵
recall = [0.68,0.84,0.92,0.94,0.8,0.92,0.84,0.84,0.9,0.94]
f1=[]
for i, val in enumerate(precision):
    x = 2 * val * recall[i]
    y = val + recall[i]
    z = x / y
    f1.append(z)
print(f1)
print(np.sum(f1))