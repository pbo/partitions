# coding: utf-8
"""
    Calculation of the average payoff table of the partitions in the partitions
    game.

    For the results see https://docs.google.com/spreadsheets/d/16R3xHbcIHqSIUsO8Jimp_D7_Z94UCjfIS9QYYn9kM64/edit?usp=sharing
"""

import numpy as np
import pandas as pd
from functools import partial
from partitions import *


def average_payoff(pm, scoring_func):
    df = pd.DataFrame(game.matrix_apply_scoring(pm, scoring_func))
    return (-df.sum() / (len(df.columns) - 1)).values

n, m = 25, 5
partitions = core.all_partitions(n, m)
df = pd.DataFrame(partitions)

try:
    pm = data.load_payoff_matrix_lotto_permute(n, m)
except FileNotFoundError:
    pm = game.payoff_matrix_lotto_permute(partitions)

df = df.assign(avg_permute=average_payoff(
    pm,
    game.scoring_sign))
df = df.assign(avg_resource=average_payoff(
    game.payoff_matrix_zero_sum(partitions, game.payoff_lotto_resource),
    game.scoring_sign))

adjacency_m = core.family_adjacency_matrix(partitions, branches=[-1, 0, 1])
family_pm = game.payoff_matrix_graph(pm, adjacency_m)
df = df.assign(family_avg=average_payoff(
    family_pm,
    game.scoring_sign))

# Print all data frame
for t in df.itertuples():
    print("\t".join(map(str, t)))
