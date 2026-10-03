"""Sanity checks for the benchmark maths in flylab.screen (Monte Carlo vs exact formulas)."""
import random
from flylab import screen as S
N, K = 30, 4
rng = random.Random(0)
first, last, within = [], [], 0
for _ in range(200000):
    o = list(range(N)); rng.shuffle(o)
    pos = [i + 1 for i, x in enumerate(o) if x < K]
    first.append(pos[0]); last.append(pos[-1]); within += pos[0] <= 3
print("first", sum(first) / len(first), S._expected_random_first(N, K))
print("all  ", sum(last) / len(last), S._expected_random_all(N, K))
print("P<=3 ", within / len(first), S._p_random_within(N, K, 3))
# tie-aware guided positions: all-equal scores == random
print("ties ", S._guided_positions([0.0] * N, [i < K for i in range(N)]), "expect", S._expected_random_first(N, K), S._expected_random_all(N, K))
# perfect ranking
print("perfect", S._guided_positions([float(N - i) for i in range(N)], [i < K for i in range(N)]))
# hits inside a tie block after 2 unique non-hits: block of 5 with 2 hits -> first = 2 + 6/3 = 4
sc = [9, 8, 1, 1, 1, 1, 1, 0, 0]
hit = [False, False, True, True, False, False, False, False, False]
print("block", S._guided_positions(sc, hit), "expect first 4.0, all 2+2*6/3=6.0")
