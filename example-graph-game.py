# coding: utf-8
"""
    Game between two graphs of partitions.
"""

import numpy as np
# import networkx as nx
from itertools import permutations
from partitions import *
from partitions.game import *


def payoff_sign(x, y):
    if x > y:
        return (0, 0, 1)
    elif x < y:
        return (1, 0, 0)
    return (0, 1, 0)


def payoff_permute_matrix(payoff_m):
    indexes = range(len(payoff_m))
    return payoff_reduce(
        (payoff_reduce((row[i] for i, row in zip(row_indexes, payoff_m)),
                       payoff_reduce_sign_sum)
         for row_indexes in permutations(indexes)),
        payoff_reduce_sign_sum)


def payoff_graph_vs_graph(xs, adjacency_m_xs, ys, adjacency_m_ys):
    payoff_m = game.payoff_matrix_permute(xs, ys)
    payoff_m_n = game.payoff_matrix_neighbourhood_vs_neighbourhood(
        adjacency_m_xs, adjacency_m_ys, payoff_m, game.payoff_reduce_sign_sum)
    payoff_m_n = game.matrix_apply_scoring(payoff_m_n, game.scoring_sign)


x = core.Partition([11, 9, 7, 5, 3, 1])
# y = core.Partition([9, 9, 7, 6, 3, 2])
y = core.Partition([36, 0, 0, 0, 0, 0])
print(payoff_permute_matrix(payoff_matrix(x, y, payoff_sign)))
print(game.payoff_permute(x, y))
