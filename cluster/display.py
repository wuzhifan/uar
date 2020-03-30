#!/usr/local/bin/python3.8

from uar.cluster import methods
from uar.landform import generator

import matplotlib
import matplotlib.pyplot as plt


class PlotDisplay(object):
    def __init__(self,
                 kmeans=True,
                 leach=True,
                 land_width=200,
                 round_count=1500):
        self.round_count = round_count
        self.land = generator.LandForm(land_width, land_width)
        self.consumptions = {}
        self.remains = {}
        self.dead_nodes = {}
        if kmeans:
            label = "K-Means"
            kc, kr, kdn = self.use_kmeans()
            self.consumptions[label] = kc
            self.remains[label] = kr
            self.dead_nodes[label] = kdn
            label = "ECBIK"
            kch, krh, khdn = self.use_kmeans_rehead()
            self.consumptions[label] = kch
            self.remains[label] = krh
            self.dead_nodes[label] = khdn

        if leach:
            label = "LEACH"
            lc, lr, ldn = self.use_leach()
            self.consumptions[label] = lc
            self.remains[label] = lr
            self.dead_nodes[label] = ldn
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
        plt_color = {"K-Means": "red", "ECBIK": "blue", "LEACH": "black"}
        plt_linestyle = {"K-Means": "-", "ECBIK": "-", "LEACH": "--"}
        fig = plt.figure(figsize=(13, 6.5))
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        plt.sca(ax1)
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        for k, v in self.consumptions.items():
            plt.plot(range(self.round_count),
                     v,
                     lw=0.5,
                     label=k,
                     ls=plt_linestyle[k],
                     color=plt_color[k])
        plt.legend(loc='upper right')
        plt.xlim(0, self.round_count)
        plt.ylim(bottom=0)
        plt.xlabel('Rounds')
        plt.ylabel('Energy consumption')

        plt.sca(ax2)
        for k, v in self.remains.items():
            plt.plot(range(self.round_count),
                     v,
                     lw=0.5,
                     label=k,
                     ls=plt_linestyle[k],
                     color=plt_color[k])
        plt.legend(loc='upper right')
        plt.xlim(0, self.round_count)
        plt.ylim(bottom=0)
        plt.xlabel('Rounds')
        plt.ylabel('Residual energy')

        plt.sca(ax3)
        for k, v in self.dead_nodes.items():
            plt.plot(range(self.round_count),
                     v,
                     lw=0.5,
                     label=k,
                     ls=plt_linestyle[k],
                     color=plt_color[k])
        plt.legend(loc='upper left')
        plt.xlim(0, self.round_count)
        plt.ylim(bottom=0)
        plt.xlabel('Rounds')
        plt.ylabel('Dead node')

        plt.show()


if __name__ == "__main__":
    PlotDisplay(True, True).show()