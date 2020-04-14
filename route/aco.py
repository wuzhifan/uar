import random


class Graph(object):
    def __init__(self, cost_matrix: list, cost_angle_matrix: list, rank: int):
        """
        :param cost_matrix:
        :param rank: rank of the cost matrix
        """
        self.matrix = cost_matrix
        self.angle_matrix = cost_angle_matrix
        self.rank = rank
        # noinspection PyUnusedLocal
        self.pheromone = [[1 / (rank * rank) for j in range(rank)]
                          for i in range(rank)]


class ACO(object):
    def __init__(self, ant_count: int, generations: int, alpha: float,
                 beta: float, rho: float, q: int, strategy: int):
        """
        :param ant_count:
        :param generations:
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        :param strategy: pheromone update strategy. 0 - ant-cycle, 1 - ant-quality, 2 - ant-density, 3 - bellman
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations
        self.update_strategy = strategy

        self.round_costs_mean = []
        self.round_angle_costs_mean = []
        self.round_best_cost = []
        self.round_angle_cost = []

    def _update_pheromone(self, graph: Graph, ants: list):
        for i, row in enumerate(graph.pheromone):
            for j, col in enumerate(row):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    # noinspection PyProtectedMember
    def solve(self, graph: Graph):
        """
        :param graph:
        """
        best_cost = float('inf')
        angle_cost = float('inf')
        best_solution = []
        for gen in range(self.generations):
            round_cost_sum = 0.0
            round_angle_cost_sum = 0.0
            # noinspection PyUnusedLocal
            ants = [_Ant(self, graph) for i in range(self.ant_count)]
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant._select_next()
                ant.total_cost += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
                ant.total_angle_cost += graph.angle_matrix[ant.tabu[-1]][
                    ant.tabu[0]]
                ant.tabu.append(ant.tabu[0])
                round_cost_sum += ant.total_cost
                round_angle_cost_sum += ant.total_angle_cost
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    angle_cost = ant.total_angle_cost
                    best_solution = [] + ant.tabu
                # update pheromone
                ant._update_pheromone_delta()
            self.round_costs_mean.append(round_cost_sum / self.ant_count)
            self.round_angle_costs_mean.append(round_angle_cost_sum /
                                               self.ant_count)
            self.round_best_cost.append(best_cost)
            self.round_angle_cost.append(angle_cost)
            self._update_pheromone(graph, ants)
            # print('generation #{}, best cost: {}, path: {}'.format(gen, best_cost, best_solution))
        return best_solution, best_cost


class _Ant(object):
    def __init__(self, aco: ACO, graph: Graph):
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.total_angle_cost = 0.0
        self.tabu = []  # tabu list
        self.pheromone_delta = []  # the local increase of pheromone
        self.allowed = [i for i in range(graph.rank)
                        ]  # nodes which are allowed for the next selection
        self.gama = 0.06
        self.angle_eta_weight = 0.012
        self.angle_q = 13
        self.eta = []
        for i in range(graph.rank):
            row = []
            for j in range(graph.rank):
                if i == j:
                    row.append(0)
                    continue
                v = 1 / graph.matrix[i][j]
                if self.colony.update_strategy in [3, 4]:
                    v += self.angle_eta_weight / graph.angle_matrix[i][j]
                row.append(v)
            self.eta.append(row)
        # self.eta = [[
        #     0 if i == j else
        #     (1 / graph.matrix[i][j] + 2 / graph.angle_matrix[i][j])
        #     for j in range(graph.rank)
        # ] for i in range(graph.rank)]  # heuristic information
        start = random.randint(0, graph.rank - 1)  # start from any node
        self.tabu.append(start)
        self.current = start
        self.allowed.remove(start)

    def _select_next(self):
        denominator = 0
        for i in self.allowed:
            denominator += self.graph.pheromone[
                self.current][i]**self.colony.alpha * self.eta[
                    self.current][i]**self.colony.beta
        # noinspection PyUnusedLocal
        # probabilities for moving to a node in the next step
        probabilities = [0 for i in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.allowed.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing
        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break
        # if selected not in self.allowed:
        #     print(probabilities)
        #     print(self.graph.pheromone[self.current])
        #     print(rand, selected)
        #     raise Exception("unknown")
        self.allowed.remove(selected)
        self.tabu.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.total_angle_cost += self.graph.angle_matrix[
            self.current][selected]
        self.current = selected

    # noinspection PyUnusedLocal
    def _update_pheromone_delta(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)]
                                for i in range(self.graph.rank)]
        if self.colony.update_strategy in [0, 1, 2]:
            for _ in range(1, len(self.tabu)):
                i = self.tabu[_ - 1]
                j = self.tabu[_]
                if self.colony.update_strategy == 1:  # ant-quality system
                    self.pheromone_delta[i][j] = self.colony.Q
                elif self.colony.update_strategy == 2:  # ant-density system
                    # noinspection PyTypeChecker
                    self.pheromone_delta[i][j] = \
                        self.colony.Q / self.graph.matrix[i][j]
                else:  # ant-cycle system
                    self.pheromone_delta[i][
                        j] = self.colony.Q / self.total_cost
        elif self.colony.update_strategy == 3:
            last = 0
            for _ in range(len(self.tabu) - 1, 0, -1):
                i = self.tabu[_ - 1]
                j = self.tabu[_]
                cr = self.colony.Q / self.graph.matrix[i][
                    j] + self.angle_q / self.graph.angle_matrix[i][j]
                self.pheromone_delta[i][j] = self.gama * last + cr
                last = self.pheromone_delta[i][j]
        elif self.colony.update_strategy == 4:

            for _ in range(len(self.tabu) - 1, 0, -1):
                i = self.tabu[_ - 1]
                j = self.tabu[_]

                last = 0
                if _ != len(self.tabu) - 1:
                    denominator = 0
                    for ii in range(self.graph.rank):
                        denominator += self.graph.pheromone[ii][
                            i]**self.colony.alpha * self.eta[ii][
                                i]**self.colony.beta
                    for ii in range(self.graph.rank):
                        p = self.graph.pheromone[ii][i] ** self.colony.alpha * \
                            self.eta[ii][i] ** self.colony.beta / denominator
                        last += p * self.graph.pheromone[ii][i]

                cr = self.colony.Q / self.graph.matrix[i][
                    j] + self.angle_q / self.graph.angle_matrix[i][j]
                self.pheromone_delta[i][j] = self.gama * last + cr
