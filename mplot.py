import matplotlib.pyplot as plt
import numpy as np
import functools
COLORS = "bgrcmyk"

data = np.loadtxt('in.txt', int)
seq_data = np.loadtxt('seq.txt', int)

seq = seq_data
seqInd2OriInd = dict(zip(seq, range(len(seq))))


data = np.array(sorted(data, key=functools.cmp_to_key(lambda t0, t1:seqInd2OriInd[t0[0]]-seqInd2OriInd[t1[0]])))


SEQ_NUM = data[:, 0]
X = data[:, 1]
Y = data[:, 2]
N = SEQ_NUM.shape[0]

total_dist = 0
for ind in range(N):
    ind1 = ind
    ind2 = (ind + 1) % N
    dX = X[ind1] - X[ind2]
    dY = Y[ind1] - Y[ind2]
    dist = np.sqrt(dX * dX + dY * dY)
    # print(dist)
    total_dist += dist

print(total_dist)

plt.figure(figsize=(12, 8))
# plt.title('TSP Graph')
plt.title('SIMULATED ANNEALING: ' + str(total_dist))
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X, Y, s=140)
for ind in range(N):
    plt.plot((X[ind], X[(ind+1) % N]), (Y[ind], Y[(ind+1) % N]), color=COLORS[ind % len(COLORS)], linewidth=2, alpha=0.5) # 折线
    plt.annotate(SEQ_NUM[ind], (X[ind], Y[ind]), size=15)# 标签

plt.axis('equal')
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
# plt.savefig("SA.png")
plt.show()




# best dist = 33523.708507