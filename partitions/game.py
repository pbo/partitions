import numpy as np
from itertools import compress, product
from functools import reduce
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


def payoff_lotto(a, b):
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


def payoff_permute(a, b):
    """
    Returns tuple of normalized (sum == 1) values of (losses, draws, wins).
    """
    if a.unique_parts < b.unique_parts:
        return payoff_all_vs_one(a.iter_permutations, b,
                                 payoff_blotto_sign, aggregate_sum)
    else:
        return payoff_one_vs_all(a, b.iter_permutations,
                                 payoff_blotto_sign, aggregate_sum)


def payoff_from_matrix(payoff_matrix):
    return lambda index_a, index_b: payoff_matrix[index_a][index_b]


class _Counting(object):
    def __init__(self, it):
        self._it = iter(it)
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        value = next(self._it)
        self.count += 1
        return value


def payoff_reduce(payoffs, payoff_reduce_func):
    it = _Counting(payoffs)
    result = reduce(payoff_reduce_func, it, (0, 0, 0))
    return (result[0] / it.count, result[1] / it.count, result[2] / it.count)


def payoff_reduce_sum(ldw_result, ldw_next):
    return (ldw_result[0] + ldw_next[0],
            ldw_result[1] + ldw_next[1],
            ldw_result[2] + ldw_next[2])


def payoff_reduce_sign_sum(ldw_result, ldw_next):
    l, d, w = ldw_next
    if l > w:
        return (ldw_result[0] + 1, ldw_result[1],     ldw_result[2])
    elif l < w:
        return (ldw_result[0],     ldw_result[1],     ldw_result[2] + 1)
    else:
        return (ldw_result[0],     ldw_result[1] + 1, ldw_result[2])


def payoff_one_vs_all(a, partitions_b, payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(a, b) for b in partitions_b),
                         payoff_reduce_func)


def payoff_all_vs_one(partitions_a, b, payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(a, b) for a in partitions_a),
                         payoff_reduce_func)


def payoff_round_robin(partitions_a, partitions_b,
                       payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(a, b)
                          for a, b in product(partitions_a, partitions_b)),
                         payoff_reduce_func)


def payoff_matrix_zero_sum(players, payoff_func, progress=None):
    length = len(players)
    matrix = [[None for col in range(length)] for row in range(length)]

    if progress is not None:
        progress.start(total=(length + 1) * length / 2)

    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if j < i:
                continue

            if progress is not None:
                progress.progress()

            l, d, w = payoff_func(a, b)

            matrix[i][j] = (l, d, w)
            matrix[j][i] = (w, d, l)

    return matrix


def payoff_matrix(players, payoff_func, progress=None):
    length = len(players)
    matrix = [[None for col in range(length)] for row in range(length)]

    if progress is not None:
        progress.start(total=length * length)

    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if progress is not None:
                progress.progress()

            matrix[i][j] = payoff_func(a, b)

    return matrix


def payoff_matrix_permute(partitions, progress=None):
    def payoff_helper(partitions_a, partitions_b):
        if len(partitions_a) < len(partitions_a):
            return payoff_all_vs_one(partitions_a, partitions_b[0],
                                     payoff_blotto_sign, aggregate_sum)
        else:
            return payoff_one_vs_all(partitions_a[0], partitions_b,
                                     payoff_blotto_sign, aggregate_sum)

    all_permutations = [list(p.iter_permutations()) for p in partitions]
    return payoff_matrix_zero_sum(all_permutations, payoff_helper, progress)


def payoff_matrix_adjacencies_vs_adjacencies(adjacency_m, payoff_m,
                                             payoff_reduce_func,
                                             progress=None):
    indexes = list(range(len(adjacency_m)))

    def payoff_helper(i, j):
        return payoff_round_robin(
            list(compress(indexes, adjacency_m[i])),
            list(compress(indexes, adjacency_m[j])),
            payoff_from_matrix(payoff_m), payoff_reduce_func)

    return payoff_matrix(indexes, payoff_helper, progress)


def payoff_matrix_adjacencies_vs_all(adjacency_m, payoff_m,
                                     payoff_reduce_func, progress=None):
    indexes = list(range(len(adjacency_m)))

    def payoff_helper(i, j):
        return payoff_all_vs_one(
            compress(indexes, adjacency_m[i]),
            j,
            payoff_from_matrix(payoff_m), payoff_reduce_func)

    return payoff_matrix(indexes, payoff_helper, progress)


def scoring_zero_sum(losses, draws, wins):
    return wins - losses


def scoring_sign(losses, draws, wins):
    return np.sign(wins - losses)


def scoring_bool(losses, draws, wins):
    return wins > losses


def matrix_apply_scoring(matrix, scoring_func):
    return [[scoring_func(*ldw) for ldw in row] for row in matrix]
