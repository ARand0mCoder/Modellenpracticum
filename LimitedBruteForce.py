import numpy as np
import itertools

#%%

def norm_from_dist(power, dist, norm):
    tot1 = np.zeros(len(power[0]))
    tot2 = np.zeros(len(power[0]))
    for i in dist[0]:
        tot1 += power[i]
    for i in dist[1]:
        tot2 += power[i]
    return max(norm(tot1), norm(tot2))


def k_step_perms(init_dist, k):
    # Currently only works for len(init_dist) == 2
    T1 = init_dist[0]
    T2 = init_dist[1]
    for i in range(1, k + 1):
        for comb1 in itertools.combinations(T1, i):
            for comb2 in itertools.combinations(T2, i):
                res1 = []
                res2 = []
                for j in T1:
                    if j not in comb1:
                        res1.append(j)
                for j in T2:
                    if j not in comb2:
                        res2.append(j)
                res1.extend(comb2)
                res2.extend(comb1)
                yield [res1, res2]
    

def k_step_brute_force(data, k, initial_distribution, norm):
    possible_dists = k_step_perms(initial_distribution, k)
    best_norm = norm_from_dist(data, initial_distribution, norm)
    best_dist = []
    for dist in possible_dists:
        cur_norm = norm_from_dist(data, dist, norm)
        if cur_norm < best_norm:
            best_dist = dist
            best_norm = cur_norm
    return best_dist
