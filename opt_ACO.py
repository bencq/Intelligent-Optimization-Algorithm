import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import operators


data_cities = np.loadtxt('in.txt', int)
COLORS = "bgrcmyk"

num_points = len(data_cities)


points_coordinate = data_cities[:, 1:]
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


def cal_total_distance(routine):
    '''The objective function. input routine, return total distance.
    cal_total_distance(np.arange(num_points))
    '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])




from sko.ACA import ACA_TSP

# size_pop * max_iter
aca_tsp = ACA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=100, max_iter=300, distance_matrix=distance_matrix, alpha=0.01, beta=14, rho=0.1, )

best_points, best_distance = aca_tsp.run()

X = data_cities[best_points, 1]
Y = data_cities[best_points, 2]

plt.figure(figsize=(12, 8))
# plt.title('TSP Graph')
plt.title('ACO: ' + str(best_distance))
plt.xlabel('X')
plt.ylabel('Y')

plt.scatter(X, Y, s=140)
for ind in range(num_points):
    plt.plot((X[ind], X[(ind+1) % num_points]), (Y[ind], Y[(ind+1) % num_points]), color=COLORS[ind % len(COLORS)], linewidth=2, alpha=0.5) # 折线
    plt.annotate(best_points[ind]+1, (X[ind], Y[ind]), size=15)# 标签

plt.axis('equal')
plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
plt.savefig("ACO.png")
plt.show()

# fig, ax = plt.subplots(1, 2)
# best_points_ = np.concatenate([best_points, best_points[0:1]])
# best_points_coordinate = points_coordinate[best_points_, :]
# ax[0].set_title("dist: " + str(best_distance))
# ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
# ax[1].plot(ga_tsp.generation_best_Y)
# plt.show()