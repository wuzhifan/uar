#!/usr/local/bin/python3.8

import random
import math

from uar.landform import generator

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

NODE_INIT_ENERGY = 2000
NODE_DEAD_ENERGY_THRESHOLD = 0


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def leach_threshould(r, p):
    return p / (1 - p * (r % int(1 / p)))


def sigmoid_threshould(dr, er):
    return 1 / (1 + math.exp(-(0.01 * dr + 7 * er - 2.6)))


def absorption_coefficient(f=10):
    return 0.11 * f * f / (1 + f * f) + 44 * f * f / (
        4100 + f * f) + 0.000275 * f * f + 0.003


h = 30 * 0.1
Lb = 50
P0 = 0.001
Pr = 0.0002
Eda = 5
Ti = 100
Tf = 400
a = math.pow(10, 0.1 * absorption_coefficient())

MAX_BROADCAST_DISTANCE = 100 * 0.1


class Cluster(object):
    def __init__(self, nodes_coord, nodes=None, head=None, center_coord=None):
        self.nodes_coord = nodes_coord
        self.nodes_energy = {}
        self.head = head
        self.center_coord = center_coord
        self.exclude_head_candidates = set()

    def add_node(self, node, energy=None):
        if energy == None:
            energy = NODE_INIT_ENERGY
        self.nodes_energy[node] = energy

    def collect_once(self):
        consumption = 0
        max_distance = 0

        def sub_nodes_add_consum(i, v):
            nonlocal consumption
            self.nodes_energy[i] -= v
            consumption += v

        for i in self.nodes_energy:
            if self.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                continue
            if i != self.head:
                d = distance(self.nodes_coord[i],
                             self.nodes_coord[self.head]) * 0.1
                if d > MAX_BROADCAST_DISTANCE:
                    max_distance = MAX_BROADCAST_DISTANCE
                    continue
                if d > max_distance:
                    max_distance = d

                # print("node d {0} send cons {1} receive cons {2}".format(
                #     d, self.formal_node_send_consumption(d),
                #     self.formal_node_receive_broadcast_consumption()))
                sub_nodes_add_consum(i, self.formal_node_send_consumption(d))
                sub_nodes_add_consum(
                    i, self.formal_node_receive_broadcast_consumption())
        # print(
        #     "head max d {0} to auv cons {1} fusion cons {2} rec cons {3} br cons {4}"
        #     .format(max_distance, self.formal_head_to_auv_consumption(),
        #             self.formal_fusion_consumption(),
        #             self.formal_head_receive_consumption(),
        #             self.formal_head_broadcast_consumption(max_distance)))

        if self.nodes_energy[self.head] > NODE_DEAD_ENERGY_THRESHOLD:
            sub_nodes_add_consum(self.head,
                                 self.formal_head_to_auv_consumption())
            sub_nodes_add_consum(self.head, self.formal_fusion_consumption())
            sub_nodes_add_consum(self.head,
                                 self.formal_head_receive_consumption())
            sub_nodes_add_consum(
                self.head,
                self.formal_head_broadcast_consumption(max_distance))

        return consumption

    def formal_head_broadcast_consumption(self, distance):
        return Lb * P0 * math.pow(distance, 1.5) * math.pow(a, distance)

    def formal_fusion_consumption(self):
        return Eda * Tf * 0.005

    def formal_head_receive_consumption(self):
        alive_node_count = len(self.nodes_energy) - self.count_dead_node()
        if alive_node_count <= 1:
            return 0
        return (alive_node_count - 1) * Ti * Pr

    def formal_head_to_auv_consumption(self):
        return Tf * P0 * math.pow(h, 1.5) * math.pow(a, h)

    def formal_node_send_consumption(self, distance):
        return Ti * P0 * math.pow(distance, 1.5) * math.pow(a, distance)

    def formal_node_receive_broadcast_consumption(self):
        return Lb * Pr

    def count_dead_node(self):
        s = 0
        for _, e in self.nodes_energy.items():
            if e <= NODE_DEAD_ENERGY_THRESHOLD:
                s += 1
        return s

    def is_all_dead(self):
        return self.count_dead_node() == len(self.nodes_energy)

    def remain_energy(self):
        r = 0
        for _, e in self.nodes_energy.items():
            if e > 0:
                r += e
        return r

    def refresh_exclude_head_condidate(self, r):
        if len(self.exclude_head_candidates) >= len(
                self.nodes_energy) - self.count_dead_node():
            self.exclude_head_candidates = set()
        if r % 10 == 0:
            self.exclude_head_candidates = set()


class KMeansCluster(object):
    def __init__(self, land):
        self.nodes_coord = land.cities
        self.kmeans = None
        self.clusters = []
        self.init_clusters(self.find_k_elbow())

    def init_clusters(self, k):
        self.clusters = [
            Cluster(self.nodes_coord,
                    center_coord=self.kmeans.cluster_centers_[i])
            for i in range(k)
        ]
        for i in range(len(self.nodes_coord)):
            label = self.kmeans.labels_[i]
            self.clusters[label].add_node(i)
        for c in self.clusters:
            self.find_head_center_nearest(c)

    def find_head_stochastic(self, cluster, r):
        if cluster.is_all_dead():
            return
        cluster.refresh_exclude_head_condidate(r)
        while 1:
            for i in cluster.nodes_energy:
                if cluster.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                    continue
                if i in cluster.exclude_head_candidates:
                    continue
                d = distance(cluster.nodes_coord[i], cluster.center_coord)
                if d == 0:
                    d = 1
                t = sigmoid_threshould(
                    1 / d, cluster.nodes_energy[i] / NODE_INIT_ENERGY)
                rdn = random.random()
                if rdn <= t:
                    cluster.head = i
                    cluster.exclude_head_candidates.add(i)
                    return

    def find_head_center_nearest(self, cluster):
        if cluster.is_all_dead():
            return
        min_dis = -1
        for i in cluster.nodes_energy:
            if cluster.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                continue
            d = distance(cluster.nodes_coord[i], cluster.center_coord)
            if min_dis == -1 or min_dis > d:
                cluster.head = i
                min_dis = d

    def find_k_elbow(self):
        lki = KMeans(n_clusters=1).fit(self.nodes_coord).inertia_
        max_slope = 0
        i = 2
        while i <= len(self.nodes_coord):
            self.kmeans = KMeans(n_clusters=i).fit(self.nodes_coord)
            slope = self.kmeans.inertia_ - lki
            if max_slope == 0:
                max_slope = slope
            if slope / max_slope < 0.008:
                break
            lki = self.kmeans.inertia_
            i += 1
        return i

    def show(self):
        plt.scatter([c[0] for c in self.nodes_coord],
                    [c[1] for c in self.nodes_coord],
                    marker='o',
                    c=self.kmeans.labels_)
        print(self.kmeans.cluster_centers_)
        plt.show()

    def collect_once(self):
        consumption = 0
        remain = 0
        dead_node_count = 0
        for c in self.clusters:
            consumption += c.collect_once()
            remain += c.remain_energy()
            dead_node_count += c.count_dead_node()

            if c.nodes_energy[c.head] <= NODE_DEAD_ENERGY_THRESHOLD:
                self.find_head_center_nearest(c)
        return consumption, remain, dead_node_count

    def rehead_and_collect_once(self, r):
        consumption = 0
        remain = 0
        dead_node_count = 0
        for c in self.clusters:
            consumption += c.collect_once()
            remain += c.remain_energy()
            dead_node_count += c.count_dead_node()
            self.find_head_stochastic(c, r)
        return consumption, remain, dead_node_count


class LeachCluster(object):
    def __init__(self, land, p=0.1):
        self.p = p
        self.nodes_coord = land.cities
        self.nodes_energy = {
            i: NODE_INIT_ENERGY
            for i in range(len(self.nodes_coord))
        }
        self.exclude_head_candidates = set()
        self.all_dead = False
        self.alive_node_count = len(self.nodes_energy)

    def cluster(self, r):
        if r % int(1 / self.p) == 0 or \
            self.alive_node_count <= len(self.exclude_head_candidates):
            self.exclude_head_candidates = set()

        self.clusters = []
        self.current_heads = set()
        while len(self.clusters) == 0:
            for i in range(len(self.nodes_coord)):
                if self.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                    continue
                if i in self.exclude_head_candidates:
                    continue

                t = leach_threshould(r, self.p)
                rdn = random.random()
                if rdn <= t:
                    nc = Cluster(self.nodes_coord, head=i)
                    nc.add_node(i, self.nodes_energy[i])
                    self.clusters.append(nc)
                    self.current_heads.add(i)
                    self.exclude_head_candidates.add(i)

        for i in range(len(self.nodes_coord)):
            if i in self.current_heads:
                continue
            tc = None
            d = -1
            for c in self.clusters:
                cd = distance(self.nodes_coord[i], self.nodes_coord[c.head])
                if d == -1 or cd < d:
                    d = cd
                    tc = c
            tc.add_node(i, self.nodes_energy[i])

    def collect_once(self, r):
        if self.all_dead:
            return 0, 0, len(self.nodes_coord)
        self.cluster(r)
        consumption = 0
        remain = 0
        dead_node_count = 0
        for c in self.clusters:
            consumption = c.collect_once()
            remain += c.remain_energy()
            dead_node_count += c.count_dead_node()

            self.nodes_energy.update(c.nodes_energy)
        self.alive_node_count = len(self.nodes_coord) - dead_node_count
        if dead_node_count == len(self.nodes_coord):
            self.all_dead = True
        return consumption, remain, dead_node_count


if __name__ == "__main__":
    l = generator.LandForm(200, 200)
    c = KMeansCluster(l)
    c.show()