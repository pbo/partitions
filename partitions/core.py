# coding: utf-8

import random
from math import inf, factorial
from operator import neg, mul, itemgetter
from functools import reduce, lru_cache
from collections import Counter


class Composition(object):
    """
    The sequence of non-negative integers.
    """

    def __init__(self, l):
        self._list = l[:]

    def __repr__(self):
        return ("[" + ", ".join(map(lambda x: repr(x).rjust(2), self._list)) +
                "]")

    def to_str(self, compact=False):
        if compact:
            f = "{:" + str(len(str(self.n))) + "d}"
            return " ".join([f.format(a) for a in self._list])
        else:
            return str(self)

    def to_list(self):
        return self._list[:]

    def to_partition(self):
        """
        Convert the composition to a partition.
        """
        return Partition(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, key):
        return self._list[key]

    def count(self, value):
        return self._list.count(value)

    def __len__(self):
        return len(self._list)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._list == other._list
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self._list))

    @property
    def weight(self):
        """
        The weight of the composition (that is, the sum of the
        corresponding sequence).
        """
        return sum(self._list)

    @property
    def width(self):
        """
        The width of the composition (the number of non-zero parts).
        """
        return sum(1 for a in self._list if a > 0)

    @property
    def height(self):
        """
        The height of the composition (the max element of the sequence).
        """
        return max(self._list)

    @property
    def n(self):
        """
        The weight of the composition (that is, the sum of the
        corresponding sequence).
        """
        return self.weight

    @property
    def unique_parts(self):
        """
        Number of unique parts of the composition
        """
        return len(set(self._list))

    @property
    def m(self):
        """
        The number of parts (including trailing zeros).
        """
        return len(self._list)

    @property
    def balance(self):
        """
        Permutational balance of the composition.
        """
        return self.to_partition().balance

    @property
    def eigenresource(self):
        """
        The resource of the composition relative to itself.
        The value is in [0, Composition.eigenresource_max(self.m)].
        """
        return self.resource_relative_to(self)

    @staticmethod
    def eigenresource_max(m):
        return m * (m - 1) // 2

    def resource_relative_to(self, another):
        """
        The resource of the composition relative to another composition.
        """
        return sum(1 for x in self._list for y in another._list if x > y)

    def compact(self):
        """
        Returns the composition without zeros.
        """
        return Composition([a for a in self._list if a > 0])

    def iter_permutations(self):
        """
        Iterates throung all possible permutations.
        """
        class Part:
            def __init__(self, value, occurrences):
                self.value = value
                self.occurrences = occurrences

        def helper(parts, result_list, d):
            if d < 0:
                yield Composition(result_list)
            else:
                for part in parts:
                    if part.occurrences > 0:
                        result_list[d] = part.value
                        part.occurrences -= 1
                        for g in helper(parts, result_list, d - 1):
                            yield g
                        part.occurrences += 1

        parts = [Part(v, o) for v, o in Counter(self._list).most_common()]
        m = self.m

        return helper(parts, [0] * m, m - 1)

    def count_permutations(self):
        denom = reduce(mul, map(factorial, Counter(self._list).values()), 1)
        return int(factorial(self.m) / denom)


class Partition(Composition):
    """
    The non-increasing sequence of non-negative integers.
    """

    def __init__(self, l, already_sorted=False):
        Composition.__init__(self, l)
        if not already_sorted:
            self._list.sort(reverse=True)

    @property
    def height(self):
        """
        The height of the partition (the first element of the sequence).
        """
        # The list is already sorted so max(self._list) == self._list[0]
        return self._list[0]

    @property
    def balance_old(self):
        """
        The balance of the partition.
        The value is in [-self.m, 0].
        """
        half_len = self.m // 2
        head, tail = self._list[:half_len], self._list[-half_len:]
        zipped_head_tail = zip(reversed(head), tail)
        return (-sum(i * (h - t)
                     for i, (h, t) in enumerate(zipped_head_tail)) / self.n)

    @property
    def balance(self):
        """
        The balance of the partition.
        The value is in range [0, 0.5], where 0 stands for (n, 0, ..., 0) and
        0.5 stands for (n/m, n/m, ..., n/m).
        """
        return self.weighted_size / (self.n * (self.m - 1))

    @property
    def weighted_size(self):
        return sum(i * a for i, a in enumerate(self._list))

    def compact(self):
        """
        Returns the partition without trailing zeros.
        """
        p = Partition(self._list, already_sorted=True)
        while p._list[-1] == 0:
            p._list.pop()
        return p

    def conjugate(self):
        """
        Find the conjugate of the partition.
        """
        if not self._list:
            return Partition([], already_sorted=True)
        else:
            length = len(self._list)
            conj = [length] * self._list[-1]
            for i in range(length - 1, 0, -1):
                conj.extend([i] * (self._list[i - 1] - self._list[i]))
            return Partition(conj, already_sorted=True)

    def ferrers_diagram(self, symbol="•"):
        """
        The Ferrers diagram for the partition.
        """
        return "\n".join((symbol + " ") * p for p in self._list)

    def is_dominates(self, other):
        """
        Returns true if `self` dominates a given partition.
        """
        self_sum = 0
        other_sum = 0
        for (a, b) in zip(self._list, other._list):
            self_sum += a
            other_sum += b
            if self_sum < other_sum:
                return False
        return True

    def family(self, zeros=True, branches=[-1, 0, 1]):
        """
        Returns all sibling partitions p_i (partitions with the same n and m
        and distance_max(self, p_i) <= 1).
        """
        def helper(partition, zeros, acc, prev_head, branches=[-1, 0, 1]):
            if len(partition) == 0 or abs(acc) > len(partition):
                return []

            head, *tail = partition

            if len(partition) == 1:
                new_head = head + acc
                if new_head <= prev_head:
                    if new_head > 0 or (zeros and new_head == 0):
                        return [[new_head]]
                return []

            return [[head + h] + p for h in branches if head + h <= prev_head
                                   for p in helper(tail, zeros, acc - h,
                                                   head + h)]

        return [Partition(p, already_sorted=True)
                for p in helper(self._list, zeros, 0, inf, branches)]


def distance_cumulative(a, b):
    """
    Manhattan distance between two 2D vectors
    (the sum of the absolute difference along all parts).
    """
    return sum(abs(x - y) for x, y in zip(a, b))


def distance_max(a, b):
    """
    Chebyshev distance between two 2D vectors
    (the greatest of their differences along any part).
    """
    return max(abs(x - y) for x, y in zip(a, b))


def distance_matrix(partitions, distance_func=distance_max):
    """
    The distance from any to any partitions in a given array.
    """
    length = len(partitions)
    matrix = [[None for col in range(length)] for row in range(length)]

    for i, a in enumerate(partitions):
        for j, b in enumerate(partitions):
            if j < i:
                continue

            r = distance_func(a, b)
            matrix[i][j] = r
            matrix[j][i] = r

    return matrix


def family_adjacency_matrix(partitions, branches=[-1, 0, 1]):
    length = len(partitions)
    matrix = [[None for col in range(length)] for row in range(length)]

    for i, a in enumerate(partitions):
        for j, b in enumerate(partitions):
            if j < i:
                continue

            r = (a.height - b.height in branches) and distance_max(a, b) <= 1
            matrix[i][j] = r
            matrix[j][i] = r

    return matrix


def iter_partitions(n, m, zeros=True):
    """
    Yields all possible integer partitions of n into exactly m parts
    (which could optionally be equal to zero) in colex order.

    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it to Hindenburg, 1779.
    """
    # Guard against special cases
    if n < m and not zeros:
        return

    if m == 0:
        if n == 0:
            yield Partition([], already_sorted=True)
        return

    if m == 1:
        yield Partition([n], already_sorted=True)
        return

    p = [n] + [0] * (m - 1) + [-1] if zeros else ([n - m + 1] + [1] * (m - 1) +
                                                  [-1])
    while True:
        yield Partition(p[0:m], already_sorted=True)

        if p[0] - 1 > p[1]:
            p[0] -= 1
            p[1] += 1
        else:
            j = 2
            s = p[0] + p[1] - 1
            while j < m and p[j] >= p[0] - 1:
                s += p[j]
                j += 1
            if j >= m:
                break
            x = p[j] + 1
            p[j] = x
            j -= 1
            while j > 0:
                p[j] = x
                s -= x
                j -= 1
            p[0] = s


def all_partitions(n, m, zeros=True, order='lex'):
    """
    The list of all partitions of n into m or (if zeros == True) less parts.

    If order == 'lex', then the returning list will be in lexicographical
    order. If order == 'colex', then the returning list will be in
    colexicographical order.
    """
    partitions = list(iter_partitions(n, m, zeros))
    if order == 'lex':
        partitions.sort(key=itemgetter(*range(m)), reverse=True)
    elif order == 'colex':
        pass
    else:
        raise ValueError('Unknown order "{}"'.format(order))
    return partitions

# Draft of the new API:
#
# count_partitions(n)
# count_partitions(n, having_parts=m)
# count_partitions(n, max_part=n)
# count_partitions(n, min_part=0)
# iter_partitions(n, order='colex')
# iter_partitions(n, having_parts=m, order='colex')
# iter_partitions(n, max_part=n, order='colex')
# iter_partitions(n, min_part=0, order='colex')
# all_partitions(n, order='colex')
# all_partitions(n, having_parts=m, order='colex')
# all_partitions(n, max_part=n, order='colex')
# all_partitions(n, min_part=0, order='colex')
# iter_random_partitions(n)
# iter_random_partitions(n, having_parts=m)
# iter_random_partitions(n, max_part=n)
# iter_random_partitions(n, min_part=0)
# or
# Partitions(n)
# Partitions(n, having_parts=m, order='colex')
# Partitions(n, max_part=n, order='colex')
# Partitions(n, min_part=0, order='colex')
# (...).size() // counts all partitions in the class
# (...).list() // returns the list of all partitions in the class
# (...).random() // returns the infinite iterator of random partitions in the class


def count_partitions(*args, zeros=True):
    """
    count_partitions(n)
    returns the number of all possible partitions of n.

    count_partitions(n, m),
    count_partitions(n, m, zeros=True)
    returns the number of partitions of n with m or less parts.

    count_partitions(n, m, zeros=False)
    returns the number of partitions of n with exactly m parts.

    Modifications for speed based on the proposition that the number of
    partitions of n having m parts is equal to the number of partitions of n-m,
    if m > n/2 (for odd n) or if m >= n/2 (for even n)
    """

    def _count_unrestricted(n):
        result = 1
        p = [1] * (n + 1)
        for i in range(1, n + 1):
            result = 0
            k = 1
            l = 1
            while 0 <= i - (l + k):
                result -= (-1)**k * (p[i - l] + p[i - (l + k)])
                k += 1
                l += 3 * k - 2

            if 0 <= i - l:
                result -= (-1)**k * p[i - l]
            p[i] = result
        return result


    if zeros and len(args) == 2:
        n, m = args
        return count_partitions(n + m, m, zeros=False)

    if len(args) == 1:  # if we're finding p(n)
        return _count_unrestricted(args[0])
    elif len(args) == 2:  # if we're finding p(n, m)
        n, m = args

        if m >= n / 2.0:  # Can we use the proposition?
            return _count_unrestricted(n - m)

        result = 0
        if n == m or m == 1:
            result = 1
        elif n < m or m == 0:
            result = 0
        else:
            n1 = int(n)
            m1 = int(m)
            p = [1] * n1

            for i in range(2, m1 + 1):
                for j in range(i + 1, n1 - i + 1 + 1):
                    p[j] += p[j - i]

            result = p[n1 - m1 + 1]
    return result


@lru_cache(maxsize=None)
def count_restricted_partitions(n, m, l):
    """
    Returns the number of partitions of n into m or less parts
    """
    if n == 0:
        return 1
    elif l == 0:
        return 0
    elif m == 0:
        return 0

    return sum(count_restricted_partitions(n - i, m - 1, i)
               for i in range(1, min(n, l) + 1))


def iter_random_partitions(n, m, size, zeros=True, filter_func=None):
    """
    Yields uniformly random integer partitions if n having m (or less if
    zeros == True) parts.

    WARNING: use `filter` with caution. It may cause infinite loop if
    `filter(p) == False` for all possible `p` in `partitions(n, m)`.
    """
    i = 0
    for p in _infinite_random_partitions(n, m, zeros=zeros):
        if i == size:
            break
        if filter_func is None or filter_func(p):
            yield p
            i += 1


def _infinite_random_partitions(q, n, D={}, zeros=False):
    """
    Infinitely yields uniform random partitions of Q having N parts.

    Arguments:
    Q : Total sum across parts
    N : Number of parts to sum over
    D : a dictionary for the number of partitions of Q having N or less
    parts (or N or less as the largest part), i.e. P(Q, Q + N). Defaults
    to a blank dictionary.
    zeros : boolean if True partitions can have zero values, if False
    partitions have only positive values, defaults to False

    Returns: A list of lists
    """

    def P_with_cache(D, n, m):
        if (n, m) not in D:
            D[(n, m)] = p(n, m)
        return [D, D[(n, m)]]

    def bottom_up(partition, q, D, rand_int):
        """
        Bottom up method of generating uniform random partitions of q having
        n parts.

        Arguments:
        partition : a list to hold the partition
        q : the total sum of the partition
        D : a dictionary for the number of partitions of q having n or less
        parts (or n or less as the largest part), i.e. P(q + n, n).
        rand_int : a number representing a member of the feasible set
        """
        while q > 0:
            # loop through all possible values of the first/largest part
            for k in range(1, q + 1):
                # number of partitions of q having k or less as the largest
                # part
                D, count = P_with_cache(D, q, k)
                if count >= rand_int:
                    D, count = P_with_cache(D, q, k - 1)
                    break
            partition.append(k)
            q -= k
            if q == 0:
                break
            rand_int -= count
        return Partition(partition, already_sorted=True).conjugate()._list

    def divide_and_conquer(partition, q, n, D, rand_int):
        """
        Divide and conquer method of generating uniform random partitions of q
        having n parts.

        Arguments:
        partition : a list to hold the partition
        q : the total sum of the partition
        n : number of parts to sum over
        D : a dictionary for the number of partitions of q having n or less
        parts (or n or less as the largest part), i.e. P(q + n, n).
        rand_int : a number representing a member of the feasible set
        """
        if n >= 1 and isinstance(n, int):
            pass
        else:
            print('n must be a positive integer')

        min_int, max_int = int(1), n
        while q > 0:
            # choose a value of the largest part at random
            k = random.randrange(min_int, max_int + 1)
            D, upper = P_with_cache(D, q, k)
            D, lower = P_with_cache(D, q, k - 1)
            if lower < rand_int and rand_int <= upper:
                partition.append(k)
                q -= k
                min_int, max_int = 1, k
                num = int(upper - lower)
                rand_int = random.randrange(1, num + 1)
            elif rand_int > upper:
                min_int = k + 1
            elif rand_int <= lower:
                max_int = k - 1
        return Partition(partition, already_sorted=True).conjugate()._list

    if zeros:
        """ if zeros are allowed, then we must ask whether Q >= N. if not, then
        the total Q is partitioned among a greater number of parts than there
        are, say, individuals. In which case, some parts must be zero. A random
        partition would then be any random partition of Q with zeros appended
        at the end. But, if Q >= N, then Q is partitioned among less number of
        parts than there are individuals. In which case, a random partition
        would be any random partition of Q having N or less parts.
        """
        if q >= n:
            D, number_of_parts = P_with_cache(D, q, n)
        elif q < n:
            D, number_of_parts = P_with_cache(D, q, q)
    else:
        D, number_of_parts = P_with_cache(D, q - n, n)

    while True:
        rand_int = random.randrange(1, number_of_parts + 1)

        if zeros:
            q1 = int(q)
            partition = []
        else:
            q1 = int(q - n)
            partition = [n]

        if q < 250 or n >= q / 1.5:
            partition = bottom_up(partition, q1, D, rand_int)
        else:
            partition = divide_and_conquer(partition, q1, n, D, rand_int)
        if zeros:
            Zs = [0] * (n - len(partition))
            partition.extend(Zs)
        yield Partition(partition, already_sorted=True)
