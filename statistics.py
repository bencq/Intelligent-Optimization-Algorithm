import os
import numpy as np
# E:\document\研究生\课程\算法\展示\SA_OUT
rootDir = input()
fileNames = os.listdir(rootDir)

BEST_FITNESS = 33523.708507

fitness_list = []
for fileName in fileNames:
    filePath = os.path.join(rootDir, fileName)
    with open(filePath, 'r') as f:
        lines = f.readlines()
        fitness = float(lines[0].split(":")[1])
        fitness_list.append(fitness)

fitness_list.sort()
meanFitness = np.mean(fitness_list)
maxFitness = np.max(fitness_list)
minFitness = np.min(fitness_list)
stdFitness = np.std(fitness_list)
cntBestFitness = fitness_list.count(BEST_FITNESS)
print(fitness_list)
print("meanFitness", meanFitness, "maxFitness", maxFitness, "minFitness", minFitness, "stdFitness", stdFitness, "cntBestFitness", cntBestFitness)



