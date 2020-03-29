#!/usr/local/bin/python3.8
import random

from uar.db import redis


class LandForm(object):
    def __init__(self,
                 length=0,
                 width=0,
                 use_old=True,
                 obstacle_rate=0.01,
                 city_rate=0.01):
        # if use_old == True:
        #     db_cli = redis.redis_client()

        self.length = length
        self.width = width
        self.map = [[0 for j in range(width)] for i in range(length)]
        self.cities = []
        for i in range(length):
            for j in range(width):
                if random.random() <= city_rate:
                    if self.map[i][j] != 0:
                        continue
                    self.cities.append([i, j])
                    self.map[i][j] = 1
                elif random.random() <= obstacle_rate:
                    ol, ow = self.obstacle_shap(length, width)
                    for oi in range(i, i + ol):
                        if oi >= length:
                            break
                        for oj in range(j, j + ow):
                            if oj >= width:
                                break
                            self.map[oi][oj] = 2

    def obstacle_shap(self, length, width):
        ld = length // 17
        wd = width // 17

        l = ld
        w = wd
        for _ in range(3):
            if random.random() < 0.01:
                l += ld
            if random.random() < 0.01:
                w += wd
        return l, w


if __name__ == "__main__":
    l = LandForm(200, 200)
    for i in range(l.length):
        print(l.map[i])
    print(l.cities)
