import numpy as np
import itertools

#%%

'''
--------- CONTENT ---------
The code in this file can be used for performing a limited brute force. An 
initial distribution can be specified, together with a maximum number of 
allowed changes. Every configuration that is at most the specified number of 
changes away from the initial distribution is tested for its norm, and the 
distribution with the lowest norm is returned. It is possible to specify 
a penalty for each number of changes, if one prefers to get results with a 
lower amount of changes.

This only works if the initial distribution consists of 2 groups.
INPUTS for k_step_brute_force:
    - data: the input data
    - k: The maximum number of allowed changes.
    - initial_distribution
    - max_stekkers: The maximal amount of stekkers that can be connected to each
                trafo in the initial distribution.
    - norm: the desired norm function to be used
    - capacities: The maximal capacity of each trafo
    - weights: The weights of each group
    - penalties: The penalty for each change weight. So penalties[i] gives the
                 extra cost that should be added to the norm for i changes from
                 the initial distribution.
OUTPUT: The best distribution.

The helper functions are norm_from_dist, k_step_perms and change_weights.
The norm_from_dist function calculates the norm that is associated with a 
specific distribution. It can also take care of different capacities of the 
groups.

The k_step_perms function returns all the distributions that are at distance 
at most k from the initial distribution. It uses the yield statement to return
a generator instead of a list in order to save memory usage.

The change_weights calculates the distance from a given distribution to the
initial distribution, and this uses the weights.
----------------------------
'''

def norm_from_dist(power, dist, norm, capacities):
    # Calculate the norm (weighted by capacity) of a given distribution
    tot1 = np.zeros(len(power[0]))
    tot2 = np.zeros(len(power[0]))
    for i in dist[0]:
        tot1 += power[i]
    for i in dist[1]:
        tot2 += power[i]
    return max(norm(tot1) / capacities[0], norm(tot2) / capacities[1])


def k_step_perms(init_dist, k):
    # Gives all permutations at distance at most k from init_dist
    T1 = init_dist[0]
    T2 = init_dist[1]
    
    # For each allowed distance i, take all subsets comb1 of size i in T1, and 
    # take all subsets comb2 of size i in T2, and then change them. The resulting
    # distributions are then res1 and res2, which are returned.
    for i in range(1, k + 1):
        for comb1 in itertools.combinations(T1, i):
            for comb2 in itertools.combinations(T2, i):
                res1 = []
                res2 = []
                # Move everything from comb1 to res2, and everything else to res1
                for j in T1:
                    if j not in comb1:
                        res1.append(j)
                for j in T2:
                    if j not in comb2:
                        res2.append(j)
                res1.extend(comb2)
                res2.extend(comb1)
                yield [res1, res2], [comb1, comb2]


def change_weight(changes, weights):
    # For a list of weights and the list of changes (this list contains of two
    # lists, one which gives the numbers going from T1 to T2, and another giving
    # the numbers going from T2 to T1), calculate the weight of this change.
    total = 0
    for change in changes:
        change_tot = 0
        for c in change:
            # The weights list defines how big every change is.
            change_tot += weights[c]
        total += change_tot
    return int(total)
    

def k_step_brute_force(data, k, initial_distribution, max_stekkers, norm, capacities, weights, penalties = [0 for i in range(1000)]):
    # The k_step_perms only gives changes where from both groups the same amount
    # is taken. Sometimes it may be needed to move more from one group than from
    # the other, so we add some extra rows that do not matter.
    new_initial = initial_distribution.copy()
    total_groups = len(initial_distribution[0]) + len(initial_distribution[1])
    for i in range(2):
        # Make sure that we do not add too much.
        if max_stekkers[i] - len(new_initial[i]) > k:
            max_stekkers[i] = len(new_initial[i]) + k
        while len(new_initial[i]) < max_stekkers[i]:
            new_initial[i].append(total_groups)
            total_groups += 1
            data.append(np.zeros(len(data[0])))
            weights.append(0)
    
    possible_dists = k_step_perms(new_initial, k)
    best_norm = norm_from_dist(data, new_initial, norm, capacities)
    best_dist = []
    
    # Try all distributions and calculate their norms. Return the distribution
    # with the lowest norm. 
    for dist, changes in possible_dists:
        # A distribution is only allowed if the changes in comb move less than 
        # the maximum number of allowed changes, even with weigths applies
        weight = change_weight(changes, weights)
        if weight <= k:
            cur_norm = norm_from_dist(data, dist, norm, capacities) + penalties[weight]
            
            if cur_norm < best_norm:
                best_dist = dist
                best_norm = cur_norm
    
    return best_dist
