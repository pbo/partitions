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
    return (df.sum(axis=1) / (len(df.columns) - 1)).values

n, m = 16, 6
partitions = core.all_partitions(n, m)
df = pd.DataFrame(partitions)

# Load or calculate a payoff matrix for all partitions
try:
    pm = data.load_payoff_matrix_permute(n, m)
except FileNotFoundError:
    pm = game.payoff_matrix_permute(partitions)

# Calculate and add to the dataframe average payoff using a specified
# scoring function
df = df.assign(avg_permute=average_payoff(
    pm,
    game.scoring_sign))

# Calculate and add to the dataframe average payoff for the different type of
# payoff function based on partition resource function
df = df.assign(avg_lotto=average_payoff(
    game.payoff_matrix_zero_sum(partitions, game.payoff_lotto),
    game.scoring_sign))

# Calculate and add to the dataframe average payoff for the game between
# partition families
adjacency_m = core.family_adjacency_matrix(partitions, branches=[-1, 0, 1])
family_pm = game.payoff_matrix_adjacencies_vs_adjacencies(
    adjacency_m, pm, game.payoff_reduce_sign_sum)
df = df.assign(family_vs_family_avg=average_payoff(
    family_pm,
    game.scoring_sign))

# Calculate and add to the dataframe average payoff for the game between
# partition families vs all other partitions
df = df.assign(family_vs_all_avg=average_payoff(
    game.payoff_matrix_adjacencies_vs_all(adjacency_m, pm,
                                          game.payoff_reduce_sign_sum),
    game.scoring_sign))


# Print the whole dataframe
for t in df.itertuples():
    print("\t".join(map(str, t)))
