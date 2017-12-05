# coding: utf-8
"""
    Calculation of the average payoff table of the partitions in the partitions
    game.
"""

import numpy as np
import pandas as pd
from functools import partial
from partitions import *


def print_raiting_df(df):
    for p, row in df.items():
        print("\t".join(map(str, p)) + "\t" + str(row))


n, m = 25, 5
partitions = core.all_partitions(n, m)

# pm = game.payoff_matrix_lotto_permute(partitions)
pm = data.load_payoff_matrix_lotto_permute(n, m)
sign_pm = game.matrix_apply_scoring(pm, game.scoring_sign)
sign_pm_df = pd.DataFrame(sign_pm, index=partitions, columns=partitions)
partitions_raiting = (-sign_pm_df.sum() / (len(sign_pm_df.columns) - 1)).sort_values()
print("Invividual payoff:")
print_raiting_df(partitions_raiting)
print()

adjacency_m = core.family_adjacency_matrix(partitions, branches=[-1, 0, 1])
family_pm = game.payoff_matrix_graph(pm, adjacency_m)
sign_family_pm = game.matrix_apply_scoring(family_pm, game.scoring_sign)
sign_family_pm_df = pd.DataFrame(sign_family_pm, index=partitions, columns=partitions)
family_raitings = (sign_family_pm_df.sum() / (len(sign_family_pm_df.columns) - 1)).sort_values()
print("Families payoff:")
print_raiting_df(family_raitings)
