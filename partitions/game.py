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
        return [0, 0, 1]
    elif wins < losses:
        return [1, 0, 0]

    return [0, 1, 0]


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
    return [losses / sz, draws / sz, wins / sz]


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


def payoff_one_vs_all(a, partitions_b, payoff_func, aggregate_func):
    result = [0, 0, 0]
    n = 0
    for b in partitions_b:
        n += 1
        aggregate_func(result, payoff_func(a, b))
    return [result[0] / n, result[1] / n, result[2] / n]


def payoff_all_vs_one(partitions_a, b, payoff_func, aggregate_func):
    result = [0, 0, 0]
    n = 0
    for a in partitions_a:
        n += 1
        aggregate_func(result, payoff_func(a, b))
    return [result[0] / n, result[1] / n, result[2] / n]


def payoff_round_robin(paritions_a, partitions_b, payoff_func, aggregate_func):
    result = [0, 0, 0]
    n = 0
    for a in paritions_a:
        for b in partitions_b:
            n += 1
            aggregate_func(result, payoff_func(a, b))
    return [result[0] / n, result[1] / n, result[2] / n]


def payoff_command_vs_command(payoff_m, adjacency_m,
                              command_a_index, command_b_index,
                              aggregate_func):
    result = [0, 0, 0]
    n = len(adjacency_m)
    range_j = list(compress(range(n), adjacency_m[command_b_index]))
    for payoff_row in compress(payoff_m, adjacency_m[command_a_index]):
        for j in range_j:
            aggregate_func(result, payoff_row[j])
    s = sum(result)
    if s == 0:
        return [0, 0, 0]
    else:
        return [result[0] / s, result[1] / s, result[2] / s]


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

            matrix[i][j] = [l, d, w]
            matrix[j][i] = [w, d, l]

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


def payoff_matrix_command_vs_command(payoff_m, adjacency_m,
                                     aggregate_func, progress=None):
    def payoff_helper(i, j):
        return payoff_command_vs_command(payoff_m, adjacency_m, i, j,
                                         aggregate_func)

    indexes = list(range(len(payoff_m)))
    return payoff_matrix_zero_sum(indexes, payoff_helper, progress)


def payoff_list_command_vs_all(payoff_m, adjacency_m, scoring_func):
    scoring_m = matrix_apply_scoring(payoff_m, scoring_func)
    n = len(payoff_m)
    payoff_row_sums = [sum(row) / (n - 1) for row in scoring_m]
    return [sum(compress(payoff_row_sums, command_mask)) / sum(command_mask)
            for command_mask in adjacency_m]


def aggregate_sum(ldw_result, ldw_next):
    ldw_result[0] += ldw_next[0]
    ldw_result[1] += ldw_next[1]
    ldw_result[2] += ldw_next[2]


def aggregate_sign_sum(ldw_result, ldw_next):
    l, d, w = ldw_next
    if l > w:
        ldw_result[0] += 1
    elif l < w:
        ldw_result[2] += 1
    else:
        ldw_result[1] += 1


def scoring_zero_sum(losses, draws, wins):
    return wins - losses


def scoring_sign(losses, draws, wins):
    return np.sign(wins - losses)


def scoring_bool(losses, draws, wins):
    return wins > losses


def matrix_apply_scoring(matrix, scoring_func):
    return [[scoring_func(*ldw) for ldw in row] for row in matrix]
