import numpy as np
from itertools import compress
from . import core


def payoff_blotto_sign(a, b):
    """
    Returns:
    (0, 0, 1) -- a wins, b loss;
    (0, 1, 0) -- draw;
    (1, 0, 0)-- a loss, b wins.
    """
    wins, losses = 0, 0
    for x, y in zip(a, b):
        if x > y:
            wins += 1
        elif x < y:
            losses += 1

    if wins > losses:
        return (0, 0, 1)
    elif wins < losses:
        return (1, 0, 0)

    return (0, 1, 0)


def payoff_lotto_permute(a, b):
    """
    Returns tuple of normalized (sum == 1) values of (losses, draws, wins).
    """
    if len(set(a)) < len(set(b)):
        w, d, losses = payoff_one_agaist_all(b, a.iter_permutations,
                                             payoff_blotto_sign)
        return losses, d, w
    else:
        return payoff_one_agaist_all(a, b.iter_permutations,
                                     payoff_blotto_sign)


def payoff_lotto_resource(a, b):
    """
    Returns tuple of normalized (sum == 1) values of (losses, draws, wins).
    """
    losses, draws, wins = 0, 0, 0
    for l_a in a:
        for l_b in b:
            if l_a < l_b:
                losses += 1
            elif l_a == l_b:
                draws += 1
            else:
                wins += 1
    sz = len(a) * len(b)
    return (losses / sz, draws / sz, wins / sz)


def payoff_one_agaist_all(a, partitions_b, payoff_func):
    losses, draws, wins = 0, 0, 0
    n = 0
    for b in partitions_b:
        n += 1
        l, d, w = payoff_func(a, b)
        losses += l
        draws += d
        wins += w
    return (losses / n, draws / n, wins / n)


def payoff_round_robin(paritions_a, partitions_b, payoff_func):
    losses, draws, wins = 0, 0, 0
    n = 0
    for a in paritions_a:
        for b in partitions_b:
            n += 1
            l, d, w = payoff_func(a, b)
            losses += l
            draws += d
            wins += w
    return (losses / n, draws / n, wins / n)


def payoff_graph(payoff_m, adjacency_m, player_i, player_j):
    l, d, w = 0, 0, 0
    range_i = list(compress(range(len(adjacency_m)), adjacency_m[player_i]))
    for payoff_row in compress(payoff_m, adjacency_m[player_j]):
        for i in range_i:
            l1, d1, w1 = payoff_row[i]
            l += l1
            d += d1
            w += w1
    s = l + d + w
    return (0, 0, 0) if s == 0 else (l / s, d / s, w / s)


def payoff_matrix_zero_sum(players, payoff_func, progress=None):
    length = len(players)
    matrix = [[None for col in range(length)] for row in range(length)]

    if progress is not None:
        progress.start(total=(length + 1) * length / 2)

    for i in range(length):
        for j in range(length):
            if j < i:
                continue

            if progress is not None:
                progress.progress()

            l, d, w = payoff_func(players[i], players[j])

            matrix[i][j] = (l, d, w)
            matrix[j][i] = (w, d, l)

    return matrix


def payoff_matrix_lotto_permute(partitions, progress=None):
    def payoff_helper(partitions_a, partitions_b):
        if len(partitions_a) < len(partitions_a):
            w, d, losses = payoff_one_agaist_all(partitions_b[0], partitions_a,
                                                 payoff_blotto_sign)
        else:
            losses, d, w = payoff_one_agaist_all(partitions_a[0], partitions_b,
                                                 payoff_blotto_sign)
        return losses, d, w

    all_permutations = [list(p.iter_permutations()) for p in partitions]
    return payoff_matrix_zero_sum(all_permutations, payoff_helper, progress)


def payoff_matrix_graph(payoff_m, adjacency_m, progress=None):
    def payoff_helper(i, j):
        return payoff_graph(payoff_m, adjacency_m, i, j)

    indexes = list(range(len(payoff_m)))
    return payoff_matrix_zero_sum(indexes, payoff_helper, progress)


def scoring_zero_sum(losses, draws, wins):
    return wins - losses


def scoring_sign(losses, draws, wins):
    return np.sign(wins - losses)


def scoring_bool(losses, draws, wins):
    return wins > losses


def matrix_apply_scoring(matrix, scoring_func):
    return [[scoring_func(*ldw) for ldw in row] for row in matrix]
