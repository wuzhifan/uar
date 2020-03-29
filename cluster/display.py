#!/usr/local/bin/python3.8

from uar.cluster import methods
from uar.landform import generator

import matplotlib.pyplot as plt


class PlotDisplay(object):
    def __init__(self,
                 kmeans=True,
                 leach=True,
                 land_width=200,
                 round_count=1000):
        self.round_count = round_count
        self.land = generator.LandForm(land_width, land_width)
        self.consumptions = []
        self.remains = []
        self.dead_nodes = []
        if kmeans:
            kc, kr, kdn = self.use_kmeans()
            self.consumptions.append(kc)
            self.remains.append(kr)
            self.dead_nodes.append(kdn)
            kch, krh, khdn = self.use_kmeans_rehead()
            self.consumptions.append(kch)
            self.remains.append(krh)
            self.dead_nodes.append(khdn)

        if leach:
            lc, lr, ldn = self.use_leach()
            self.consumptions.append(lc)
            self.remains.append(lr)
            self.dead_nodes.append(ldn)
        self.show()

    def use_kmeans(self):
        kc = methods.KMeansCluster(self.land)
        kcc = []
        kcr = []
        kcdn = []
        for rd in range(self.round_count):
            c, r, dn = kc.collect_once()
            kcc.append(c)
            kcr.append(r)
            kcdn.append(dn)
        return kcc, kcr, kcdn

    def use_kmeans_rehead(self):
        kc = methods.KMeansCluster(self.land)
        kcc = []
        kcr = []
        kcdn = []
        for rd in range(self.round_count):
            c, r, dn = kc.rehead_and_collect_once(rd)
            kcc.append(c)
            kcr.append(r)
            kcdn.append(dn)
        return kcc, kcr, kcdn

    def use_leach(self):
        lc = methods.LeachCluster(self.land)
        lcc = []
        lcr = []
        lcdn = []
        for rd in range(self.round_count):
            c, r, dn = lc.collect_once(rd)
            lcc.append(c)
            lcr.append(r)
            lcdn.append(dn)
        return lcc, lcr, lcdn

    def show(self):
        fig = plt.figure(figsize=(13, 6.5))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        plt.sca(ax1)
        for c in self.consumptions:
            plt.plot(range(self.round_count), c, lw=0.7)
        plt.grid(True)
        plt.title('energy consumption')
        plt.sca(ax2)
        for r in self.remains:
            plt.plot(range(self.round_count), r, lw=0.7)
        plt.grid(True)
        plt.title('energy remain')
        plt.sca(ax3)
        for d in self.dead_nodes:
            plt.plot(range(self.round_count), d, lw=0.7)
        plt.grid(True)
        plt.title('dead node')
        plt.show()


if __name__ == "__main__":
    PlotDisplay(True, True).show()