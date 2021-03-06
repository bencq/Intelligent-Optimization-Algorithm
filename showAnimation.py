import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation
import argparse
import sko
import operators
from types import MethodType

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='in.txt', type=str)
args = parser.parse_args()

if __name__ == '__main__':


    file_name = args.file_path
    points_coordinate = np.loadtxt(file_name)[:, 1:]
    num_points = points_coordinate.shape[0]
    distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')


    def cal_total_distance(routine):
        '''The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        '''
        num_points, = routine.shape
        return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


    # %%
    tsp_obj = sko.SA.SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=3, T_min=1, L=100 * num_points)
    tsp_obj.cool_down = MethodType(operators.cool_down, tsp_obj)
    # tsp_obj = sko.GA.GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=100, max_iter=200, prob_mut=0.07)

    best_points, best_distance = tsp_obj.run()
    print("best_points", best_points, "best_distance", best_distance)
    # %% Plot the best routine
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots(1, 2)

    best_points_ = np.concatenate([best_points, [best_points[0]]])
    best_points_coordinate = points_coordinate[best_points_, :]

    fig.set_figwidth(24)
    fig.set_figheight(8)
    # ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].plot(tsp_obj.best_y_history)
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Distance")
    ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
               marker='o', markerfacecolor='b', color='c', linestyle='-')
    # ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    # plt.show()

    # %% Plot the animation


    best_x_history = tsp_obj.best_x_history

    fig2, ax2 = plt.subplots(1, 1)
    fig2.set_figwidth(12)
    fig2.set_figheight(8)
    ax2.set_aspect('equal', adjustable="datalim")
    # ax2.set_title('title', loc='center')
    line = ax2.plot(points_coordinate[:, 0], points_coordinate[:, 1], marker='o', markerfacecolor='b', color='c', linestyle='-')
    # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    # plt.show()


    def update_scatter(frame):
        ax2.set_title('iter = ' + str(frame))
        print(frame)
        points = best_x_history[frame]
        points = np.concatenate([points, [points[0]]])
        point_coordinate = points_coordinate[points, :]
        plt.setp(line, 'xdata', point_coordinate[:, 0], 'ydata', point_coordinate[:, 1])
        return line


    ani = FuncAnimation(fig2, update_scatter, blit=False, interval=125, frames=len(best_x_history))
    plt.show()
    # ani.save('sa_tsp.gif', writer='pillow')