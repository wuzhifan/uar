#!/usr/local/bin/python3.8

import random
import math

from uar.landform import generator

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

NODE_INIT_ENERGY = 4500
NODE_DEAD_ENERGY_THRESHOLD = 10


def distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def leach_threshould(r, p):
    return p / (1 - p * (r % int(1 / p)))


def sigmoid_threshould(dr, er):
    return 1 / (1 + math.exp(-(0.01 * dr + 7 * er - 2.6)))


class Cluster(object):
    def __init__(self, nodes_coord, nodes=None, head=None, center=None):
        self.nodes_coord = nodes_coord
        self.nodes = []
        self.nodes_energy = {}
        if nodes != None:
            for n in nodes:
                self.add_node(n)
        self.head = head
        self.center_coord = center
        self.all_dead = None
        self.largest_distance = None
        self.exclude_head_candidates = set()

    def add_node(self, node, energy=None):
        self.nodes.append(node)
        if energy == None:
            energy = NODE_INIT_ENERGY
        self.nodes_energy[node] = energy

    def collect_once(self):
        consumption = 0
        remain = 0
        for i in self.nodes:
            if self.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                continue
            if i == self.head:
                c = self.head_to_auv_consumption()
                self.nodes_energy[i] -= c
                consumption += c
            else:
                d = distance(self.nodes_coord[i], self.nodes_coord[self.head])
                c = self.node_send_consumption(d)
                self.nodes_energy[i] -= c
                consumption += c
                if self.nodes_energy[i] > 0:
                    remain += self.nodes_energy[i]
                c = self.head_receive_consumption(d)
                self.nodes_energy[self.head] -= c
                consumption += c
        if self.nodes_energy[self.head] > 0:
            remain += self.nodes_energy[self.head]
        return consumption, remain

    def head_to_auv_consumption(self):
        return 10

    def head_receive_consumption(self, distance):
        return distance * 0.005

    def node_send_consumption(self, distance):
        return distance * 0.01

    def count_dead_node(self):
        s = 0
        for i, e in self.nodes_energy.items():
            if e <= NODE_DEAD_ENERGY_THRESHOLD:
                s += 1
        return s

    def is_all_dead(self):
        if self.all_dead == True:
            return self.all_dead

        for _, e in self.nodes_energy.items():
            if e > NODE_DEAD_ENERGY_THRESHOLD:
                return False
        self.all_dead = True
        return True

    def get_largest_distance(self):
        if self.largest_distance != None:
            return self.largest_distance

        for i in self.nodes:
            d = distance(self.nodes_coord[i], self.center_coord)
            if self.largest_distance == None or d > self.largest_distance:
                self.largest_distance = d
        return self.largest_distance


class KMeansCluster(object):
    def __init__(self, land):
        self.nodes_coord = land.cities
        self.k = 0
        self.kmeans = None
        self.find_k_elbow()
        self.clusters = [
            Cluster(self.nodes_coord, center=self.kmeans.cluster_centers_[i])
            for i in range(self.k)
        ]
        for i in range(len(self.nodes_coord)):
            label = self.kmeans.labels_[i]
            self.clusters[label].add_node(i)
        for c in self.clusters:
            self.find_head(c, 0)

    def find_head(self, cluster, r):
        if cluster.is_all_dead():
            return
        if len(cluster.exclude_head_candidates) >= len(
                cluster.nodes) - cluster.count_dead_node():
            cluster.exclude_head_candidates = set()
        if r % 10 == 0:
            cluster.exclude_head_candidates = set()
        while 1:
            # print("find head cluster nodes {0} energy {1}".format(
            #     cluster.nodes, cluster.nodes_energy))
            for i in cluster.nodes:
                if cluster.nodes_energy[i] <= NODE_DEAD_ENERGY_THRESHOLD:
                    continue
                if i in cluster.exclude_head_candidates:
                    continue
                d = distance(cluster.nodes_coord[i], cluster.center_coord)
                t = sigmoid_threshould(
                    1 / d, cluster.nodes_energy[i] / NODE_INIT_ENERGY)
                rdn = random.random()
                print(
                    "find head d {0} largest distance {1} energy {2} random {3} t {4}"
                    .format(d, cluster.get_largest_distance(),
                            cluster.nodes_energy[i], rdn, t))
                if rdn <= t:
                    cluster.head = i
                    cluster.exclude_head_candidates.add(i)
                    return

    def find_k_elbow(self):
        max_len = len(self.nodes_coord)
        lk = KMeans(n_clusters=1).fit(self.nodes_coord)
        lki = lk.inertia_
        # distorsions = []
        max_slope = 0

        for i in range(2, max_len + 1):
            k = KMeans(n_clusters=i).fit(self.nodes_coord)
            ki = k.inertia_
            slope = ki - lki
            if max_slope == 0:
                max_slope = slope
            print("k:{0} intertia:{1} slope:{2}".format(i, ki, slope))
            if slope / max_slope < 0.01:
                self.k = i
                self.kmeans = k
                break
            # distorsions.append(i)
            lk = k
            lki = ki

        # fig = plt.figure(figsize=(15, 5))
        # plt.plot(range(2, max_len + 1), distorsions)
        # plt.grid(True)
        # plt.title('Elbow curve')
        # plt.show()

    def show(self):
        plt.scatter([c[0] for c in self.nodes_coord],
                    [c[1] for c in self.nodes_coord],
                    marker='o',
                    c=self.kmeans.labels_)
        print(self.kmeans.cluster_centers_)
        plt.show()

    def collect_once(self, r):
        consumption = 0
        remain = 0
        dead_node_count = 0
        for c in self.clusters:
            cs, r = c.collect_once()
            consumption += cs
            remain += r
            dead_node_count += c.count_dead_node()

            if c.nodes_energy[c.head] <= NODE_DEAD_ENERGY_THRESHOLD:
                self.find_head(c, r)
        return consumption, remain, dead_node_count

    def rehead_and_collect_once(self, r):
        consumption = 0
        remain = 0
        dead_node_count = 0
        for c in self.clusters:
            cs, r = c.collect_once()
            consumption += cs
            remain += r
            dead_node_count += c.count_dead_node()
            self.find_head(c, r)

        return consumption, remain, dead_node_count


class LeachCluster(object):
    def __init__(self, land, p=0.1):
        self.p = p
        self.nodes_coord = land.cities
        self.nodes_energy = {
            i: NODE_INIT_ENERGY
            for i in range(len(self.nodes_coord))
        }
        self.clusters = []
        self.exclude_head_candidates = set()
        self.current_heads = set()
        self.all_dead = False

    def cluster(self, r):
        #if r % int(1 / self.p) == 0:
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
                print("round {0} node {1} random {2} t {3}".format(
                    r, i, rdn, t))
                if rdn <= t:
                    nc = Cluster(self.nodes_coord, head=i)
                    nc.add_node(i, self.nodes_energy[i])
                    self.clusters.append(nc)
                    self.current_heads.add(i)
                    self.exclude_head_candidates.add(i)
                    #print("add cluster {0} head {1}".format(nc, i))

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
        cumulated_nodes = 0
        for c in self.clusters:
            cs, rm = c.collect_once()
            consumption += cs
            remain += rm
            dead_node_count += c.count_dead_node()
            cumulated_nodes += len(c.nodes)

            self.nodes_energy.update(c.nodes_energy)
            # print(
            #     "round {0} cluster nodes {1} cluster nodes energy {2}".format(
            #         r, c.nodes, c.nodes_energy))
        #print("round {0} consumption {1} all node {2} dead node {3}".format(
        #    r, consumption, len(self.nodes_coord), dead_node_count))
        if dead_node_count == len(self.nodes_coord):
            self.all_dead = True
        return consumption, remain, dead_node_count


if __name__ == "__main__":
    l = generator.LandForm(200, 200)
    c = KMeansCluster(l)
    c.show()