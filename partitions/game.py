import numpy as np
from itertools import compress, product
from functools import reduce
from . import core


def payoff_blotto_sign(x, y):
    """
    Returns:
    (0, 0, 1) -- x wins, y loss;
    (0, 1, 0) -- draw;
    (1, 0, 0)-- x loss, y wins.
    """
    wins, losses = 0, 0
    for x_i, y_i in zip(x, y):
        if x_i > y_i:
            wins += 1
        elif x_i < y_i:
            losses += 1

    if wins > losses:
        return (0, 0, 1)
    elif wins < losses:
        return (1, 0, 0)

    return (0, 1, 0)


def payoff_lotto(x, y):
    """
    Returns tuple of normalized (sum == 1) values of (losses, draws, wins).
    """
    losses, draws, wins = 0, 0, 0
    for x_i in x:
        for y_i in y:
            if x_i < y_i:
                losses += 1
            elif x_i == y_i:
                draws += 1
            else:
                wins += 1
    count = len(x) * len(y)
    return (losses / count, draws / count, wins / count)


def payoff_permute(x, y):
    """
    Returns tuple of normalized (sum == 1) values of (losses, draws, wins).
    """
    if x.unique_parts < y.unique_parts:
        return payoff_all_vs_one(x.iter_permutations(), y,
                                 payoff_blotto_sign, payoff_reduce_sum)
    else:
        return payoff_one_vs_all(x, y.iter_permutations(),
                                 payoff_blotto_sign, payoff_reduce_sum)


def payoff_from_matrix(payoff_matrix):
    return lambda index_x, index_y: payoff_matrix[index_x][index_y]


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


def payoff_permute_matrix(payoff_m,
                          components_payoff_reduce_func=payoff_reduce_sign_sum,
                          permutations_payoff_reduce_func=payoff_reduce_sign_sum):
    indexes = range(len(payoff_m))
    return payoff_reduce(
        (payoff_reduce((row[i] for i, row in zip(row_indexes, payoff_m)),
                       components_payoff_reduce_func)
         for row_indexes in permutations(indexes)),
        permutations_payoff_reduce_func)


def payoff_one_vs_all(x, ys, payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(x, y) for y in ys), payoff_reduce_func)


def payoff_all_vs_one(xs, y, payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(x, y) for x in xs), payoff_reduce_func)


def payoff_round_robin(xs, ys, payoff_func, payoff_reduce_func):
    return payoff_reduce((payoff_func(x, y) for x, y in product(xs, ys)),
                         payoff_reduce_func)


def payoff_matrix_antisymmetric(xs, ys, payoff_func, progress=None):
    size = len(xs)

    if size != len(ys):
        raise ValueError('Lists must be the same size.')

    matrix = [[None for _ in range(size)] for _ in range(size)]

    if progress is not None:
        progress.start(total=(size + 1) * size / 2)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if j < i:
                continue

            if progress is not None:
                progress.progress()

            l, d, w = payoff_func(x, y)

            matrix[i][j] = (l, d, w)
            matrix[j][i] = (w, d, l)

    return matrix


def payoff_matrix(xs, ys, payoff_func, progress=None):
    size_xs = len(xs)
    size_ys = len(ys)
    matrix = [[None for _ in range(size_ys)] for _ in range(size_xs)]

    if progress is not None:
        progress.start(total=size_xs * size_ys)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if progress is not None:
                progress.progress()

            matrix[i][j] = payoff_func(x, y)

    return matrix


def payoff_matrix_permute(xs, ys, progress=None):
    def payoff_helper(xs, ys):
        if len(xs) < len(ys):
            return payoff_all_vs_one(xs, ys[0],
                                     payoff_blotto_sign, payoff_reduce_sum)
        else:
            return payoff_one_vs_all(xs[0], ys,
                                     payoff_blotto_sign, payoff_reduce_sum)
    if xs == ys:
        ps = [list(x.iter_permutations()) for x in xs]
        return payoff_matrix_antisymmetric(ps, ps, payoff_helper, progress)
    else:
        ps_xs = [list(x.iter_permutations()) for x in xs]
        ps_ys = [list(y.iter_permutations()) for y in ys]
        return payoff_matrix_antisymmetric(ps_xs, ps_ys, payoff_helper,
                                           progress)


def payoff_matrix_neighbourhood_vs_neighbourhood(adjacency_m_xs,
                                                 adjacency_m_ys,
                                                 payoff_m_xs_vs_ys,
                                                 payoff_reduce_func,
                                                 progress=None):
    indexes_xs = list(range(len(adjacency_m_xs)))
    indexes_ys = list(range(len(adjacency_m_ys)))

    def payoff_helper(i, j):
        return payoff_round_robin(
            list(compress(indexes_xs, adjacency_m_xs[i])),
            list(compress(indexes_ys, adjacency_m_ys[j])),
            payoff_from_matrix(payoff_m_xs_vs_ys), payoff_reduce_func)

    return payoff_matrix(indexes_xs, indexes_ys, payoff_helper, progress)


def payoff_matrix_neighbourhood_vs_all(adjacency_m, payoff_m,
                                       payoff_reduce_func, progress=None):
    indexes = list(range(len(adjacency_m)))

    def payoff_helper(i, j):
        return payoff_all_vs_one(
            compress(indexes, adjacency_m[i]),
            j,
            payoff_from_matrix(payoff_m), payoff_reduce_func)

    return payoff_matrix(indexes, indexes, payoff_helper, progress)


def scoring_zero_sum(losses, draws, wins):
    return wins - losses


def scoring_sign(losses, draws, wins):
    return np.sign(wins - losses)


def scoring_bool(losses, draws, wins):
    return wins > losses


def matrix_apply_scoring(matrix, scoring_func):
    return [[scoring_func(*ldw) for ldw in row] for row in matrix]
