# coding: utf-8

import pkg_resources
import numpy as np
import partitions


def load_e_optimal_partitions(n, m, drop_probabilities=False):
    """
    Returns two arrays (attacker and defender) of epsilon-optimal partitions
    which was calculated with the Attacker-Defender algorithm
    by Arkadi Nemirovski.

    See https://arxiv.org/pdf/1506.02444.pdf
    """

    # (120, 6)
    # ? iterations
    # Attacker: guarantee -0.02587 # of pure strategies: 227
    # Defender: guarantee 0.02040 # of pure strategies: 229

    # (15, 4)
    # 50000 iterations
    # Attacker: guarantee -0.00422 # of pure strategies: 35
    # Defender: guarantee 0.00422 # of pure strategies: 35

    # (15, 5)
    # 50000 iterations
    # Attacker: guarantee -0.00647 # of pure strategies: 45
    # Defender: guarantee 0.00647 # of pure strategies: 45

    # (36, 6)
    # 50000 iterations
    # Attacker: guarantee -0.00016 # of pure strategies: 77
    # Defender: guarantee 0.00016 # of pure strategies: 77

    # (100, 10)
    # 20000 iterations
    # Inaccurate/Solved Residual=7.498226e-01
    # 11554.95 Attacker guarantee: -7.035840e-02 Defender guarantee: 6.755791e-02
    # multiplicities: Attacker: 277 Defender: 296
    # 11576.82 Attacker guarantee: -5.590693e-02 Defender guarantee: 6.349326e-02
    # multiplicities: Attacker: 185 Defender: 182
    # Bottom line:
    #  Mixed strategies: Attacker guarantee: -5.590693e-02 Defender guarantee: 6.349326e-02
    #  Pure strategies: Attacker guarantee: -8.000000e+00 Defender guarantee: 8.000000e+00
    # multiplicities: Attacker: 185 Defender: 182
    #
    # profA =
    #      0     0     0     0     0     0     0     0     0     0
    #     29    22    29    21    35    31    34    22    30    33
    #
    # profD =
    #      0     0     0     0     0     0     0     0     0     0
    #     23    21    24    27    23    31    34    21    22    33
    #
    #
    # Attacker: guarantee -0.05591 # of pure strategires: 185
    # Defender: guarantee 0.06349 # of pure strategires: 182

    data_fname = "data/e-optimal_{}-{}.csv".format(n, m)
    data_fpath = pkg_resources.resource_filename(__name__, data_fname)
    dtype = np.dtype([('', np.object), ('', np.float)] + [('', np.int)] * m)
    data = np.loadtxt(data_fpath, dtype=dtype, delimiter=',', skiprows=1)

    result = {"attacker": [], "defender": []}
    for record in data:
        player = record[0]
        partition = partitions.Partition([record[2 + i] for i in range(m)])
        if drop_probabilities:
            result[player].append(partition)
        else:
            probability = record[1]
            result[player].append((probability, partition))

    return result["attacker"], result["defender"]


def load_payoff_matrix_lotto_permute(n, m):
    data_fname = "data/individual-payoff-matrix_{}-{}.npz".format(n, m)
    data_fpath = pkg_resources.resource_filename(__name__, data_fname)
    return np.load(data_fpath)["arr_0"]
