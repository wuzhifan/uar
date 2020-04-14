import random
import math

from uar.landform import generator
from uar.route.aco import ACO, Graph
from uar.route.plot import plot

import matplotlib.pyplot as plt


def distance(city1: dict, city2: dict):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def plot_round_bests(aco: dict):
    fig = plt.figure(figsize=(13, 6.5))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    plt.sca(ax1)
    for l, a in aco.items():
        plt.plot(range(a.generations), [c for c in a.round_costs_mean],
                 label=l)
    plt.legend(loc='upper right')
    plt.xlabel('Rounds')
    plt.ylabel('Distance mean cost')

    plt.sca(ax2)
    for l, a in aco.items():
        plt.plot(range(a.generations), [c for c in a.round_best_cost], label=l)
    plt.legend(loc='upper right')
    plt.xlabel('Rounds')
    plt.ylabel('Distance best cost')

    plt.sca(ax3)
    for l, a in aco.items():
        plt.plot(range(a.generations), [c for c in a.round_angle_costs_mean],
                 label=l)
    plt.legend(loc='upper right')
    plt.xlabel('Rounds')
    plt.ylabel('Angle mean cost')

    plt.sca(ax4)
    for l, a in aco.items():
        plt.plot(range(a.generations), [c for c in a.round_angle_cost],
                 label=l)
    plt.legend(loc='upper right')
    plt.xlabel('Rounds')
    plt.ylabel('Angle best cost')

    plt.show()


ANT_COUNT = 30
GENERATIONS = 200
ALPHA = 1.0
BETA = 5.0
RHO = 0.1
Q = 50


def run(cost_matrix: list, cost_angle_matrix: list, method: int):
    aco = ACO(ANT_COUNT, GENERATIONS, ALPHA, BETA, RHO, Q, method)
    graph = Graph(cost_matrix, cost_angle_matrix, len(cost_matrix[0]))
    path, cost = aco.solve(graph)
    print('cost: {}, path: {}'.format(cost, path))
    #plot(points, path)
    return aco


def main():
    # points = []
    # cities = []
    # with open('/Users/djy/Developer/go/dep/ml/uar/route/data/chn31.txt') as f:
    #     for line in f.readlines():
    #         city = line.split(' ')
    #         cities.append(
    #             dict(index=int(city[0]), x=int(city[1]), y=int(city[2])))
    #         points.append((int(city[1]), int(city[2])))
    land = generator.LandForm(100, 100)
    rank = len(land.cities)
    cost_matrix = []
    cost_angle_matrix = [[0 for j in range(rank)] for i in range(rank)]
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(distance(land.cities[i], land.cities[j]))
            cost_angle_matrix[i][j] = cost_angle_matrix[j][i]
            if cost_angle_matrix[i][j] == 0:
                cost_angle_matrix[i][j] = random.randint(1, 100)
        cost_matrix.append(row)
    plot_round_bests({
        "ant-density":
        run(cost_matrix, cost_angle_matrix, 2),
        # "f-ant-density":
        # run(cost_matrix, cost_angle_matrix, 3),
        "bellman-ant-density":
        run(cost_matrix, cost_angle_matrix, 4),
    })


if __name__ == '__main__':
    main()
