# coding: utf-8
"""
    Программа для изучения семейств разбиений.
"""

import cmd
import re
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from partitions import *


class FamilyShell(cmd.Cmd):
    intro = (
        "+-------------------+\n"
        "| Р А З Б И Е Н И Я |\n"
        "+-------------------+\n"
        "| v. 0.0.2       ПБ |\n"
        "+-------------------+\n"
        "\n"
        "Введите help или ? для списка доступных команд.\n"
    )
    doc_header = 'Команды (введите help <название> для информации о команде)'
    prompt = '> '
    file = None

    # PARSING #################################################################
    def parse_partition(self, arg):
        p_list = list(map(int, re.findall(r"[\d]+", arg)))
        if len(p_list) == 0:
            print("Неверно введено разбиение. Пример правильного ввода: 7 5 3 1")
        elif len(p_list) < 2:
            print("Разбиение должно содержать хотя бы два элемента")
        else:
            return core.Partition(p_list)

    # COMMANDS ################################################################
    def do_family(self, arg):
        """
        family [ranks] partition
        Характеристики семейства заданного разбиения.

        Параметры:
        ranks  : Относительные ранги разбиений, входящих в семейство.
        partition : Глава семейства.

        Примеры:
        family 7 5 3 1
        family [0] 7 5 3 1
        family [-1, 0] 7 5 3 1
        """
        ranks = None

        head = self.parse_partition(arg)

        family = head.family()
        print_family(head, family)

    def do_quit(self, arg):
        'Выход из программы'
        self.close()
        return True

    def emptyline(self):
        self.do_family("7 5 3 1")

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


def print_family(head, family):
    len_prev = sum(1 for p in family if p[0] == head[0] - 1)
    len_same = sum(1 for p in family if p[0] == head[0])
    len_next = sum(1 for p in family if p[0] == head[0] + 1)

    print("Глава семейства: {} (n = {}, m = {})".format(
        head.to_str(compact=True), head.n, head.m))

    print("Разбиений в семействе: {} шт. ({} + {} + {})".format(
        len(family), len_prev, len_same, len_next))
    print()

    family_m = np.array(core.family_adjacency_matrix(family))
    G = nx.from_numpy_matrix(family_m)

    df = pd.DataFrame([p.to_str(compact=True) for p in family], columns=["Разбиение"])

    family_m = np.array(core.family_adjacency_matrix(family))
    degrees = pd.Series(family_m.sum(axis=0), index=df.index)
    df["Степень"] = degrees
    df["Ср. степень соседей"] = pd.Series(nx.average_neighbor_degree(G), index=df.index)
    with pd.option_context('display.max_rows', None):
        print(df)
    print()

    print("Распределение степеней:")
    for degree, count in Counter(degrees).most_common():
        print("{}: {} шт.".format(degree, count))
    print()

    print("Кластерный коэффициент: {}".format(nx.average_clustering(G)))
    print("Коэффициент ассортативности: {}".format(nx.degree_assortativity_coefficient(G)))
    print()

    cliques = list(nx.find_cliques(G))
    print("Клик в семействе: {} шт.".format(len(cliques)))
    print()

    clique_sizes = map(len, cliques)
    for size, count in Counter(clique_sizes).most_common():
        print("{} клик с {} разбиениями:".format(count, size))
        for clique in cliques:
            if len(clique) == size:
                print(clique)
        print()


if __name__ == '__main__':
    FamilyShell().cmdloop()
