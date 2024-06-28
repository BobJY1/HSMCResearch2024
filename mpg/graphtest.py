from sporadics import m1
from sat_colorability_solver import allSolutions

#find coloring of a given graph

edge_list = m1[0]
n = 1
for (u, v) in edge_list:
    n = max(n, u, v)

print(allSolutions(n + 1, edge_list))