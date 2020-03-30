import math

h = 5
Lb = 50
P0 = 0.001
Pr = 0.0002
Eda = 5
Ti = 100
Tf = 230


def absorption_coefficient(f=9):
    return 0.11 * f * f / (1 + f * f) + 44 * f * f / (
        4100 + f * f) + 0.000275 * f * f + 0.003


a = math.pow(10, 0.1 * absorption_coefficient())

print(a)


def f(dis):
    print(Ti * P0 * math.pow(dis, 1.5) * math.pow(a, dis))


print(math.pow(13.801, 1.5), math.pow(a, 13.801), f(13.801))
