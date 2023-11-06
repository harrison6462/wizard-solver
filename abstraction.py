from typing import Callable
import numpy as np
# deck

# hands = list of all possible hands in some order(lexigraphical?)
# let b = be all possible tuples where ith component is bet size for ith hand
# For each (b1, b2):
#   let G be game induced by b1,b2 (with corresponding utilities)
#   solve with CFR self-play

# choose b1 that maximizes min over all b2?


# def perfectMatching(list1, list2):
#     for node1 in list1:
#         for node2 in list2:
#             if iso(game, node1, node2):
#                 # list1.remove(node1)
#                 list2.remove(node2)
#                 continue
#         return False
#     return True

# isoTable = {} # (p1 node , p2 node) -> bool
# # game is ___, v1 is sequence of signals (history), v2 is sequence of signals
# def iso(game, v1, v2): # 
#     if (v1, v2) in isoTable: 
#         return [(v1, v2)]
#     if leaf(v1) and leaf(v2):
#         output = True if utility(v1) == utility(v2) else False
#     else:
#         c1 = children(v1)
#         c2 = children(v2)
#         output = True if perfectMatching(c1, c2) else False
#     isoTable[(v1, v2)] = output
#     isoTable[(v2, v1)] = output
#     return output


# #union find ds for keeping merges

# F = identity
# def gameShrink(game, node, F):
#     c = children(node)
#     for i in range(len(c)):
#         for j in range(i+1, len(c)):
#             if (c[i] not already merged with c[j]) and iso(game, c[i],c[j]):
#                 F.merge(c[i], c[j])
#         gameShrink(game, c[i], F)



# _______________________________________________________________________________________________

# Potential Aware Imperfect Recall 
# https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf

# Pseudocode:
#   info tree with r+1 levels (0 ... r)
#   for each level, choose number of clusters (C_n)
#   for final rounds r_*, ... r, create (how?) C_n clusters each (mean m_{i,n} for ith cluster)
#   given abstraction for round n+1, compute abstraction for round n:
#       for each pair of means in round n+1, compute distance
#       for each point (node?) (x_n), compute histogram (H(x_n))
#       cluster histograms into C_n clusters using algo L_n 
#       compute new cluster means

num_levels = 3 # levels 0,1,2
num_clusters = [3,4,5] # index i is C_i (num clusters for lvl i)
# ^ could be set with IP
nodes_at_lvl = [] # TODO populate with list of nodes for each level of full game tree from json

# assume r_* = r, set final lvl clusters & means
num_set_lvls = 1
def abstraction_fn(num_levels : int, num_clusters : list[int], nodes_at_lvl : list[int], num_set_lvls: int):
    clusters_mean = {} # key (i,j) is m_{i,j} (mean of j'th cluster of lvl i)

    for lvl in range(num_levels-1, num_levels-1-num_set_lvls, -1): #set last num_set_lvls with non-potential aware algo (estimate equity as P(win) from rollout... P(win) might not be best since we care about how much we win by)
        nodes_cur_lvl = nodes_at_lvl[lvl]        
        equities = equity_from_rollout(nodes_cur_lvl)
        means = k_means(lambda x,y: np.linalg.norm(y-x, ord=2), num_clusters[lvl], equities, {}) #* set cluster mean with L2 on equity rollout, could use EMD on 1D equity distribution histogram instead
        for cl in range(num_clusters[lvl]):
            clusters_mean[(lvl, cl)] = means[cl]

    for lvl in range(num_levels-1-num_set_lvls, -1, -1):
        #   for each pair of means in round n+1, compute distance
        distances_mean = {}
        for i in range(num_clusters[lvl+1]): # redundant computation, should delete at some point for optimization
            for j in range(i+1, num_clusters[lvl+1]):
                dist_metric = lambda x,y : y-x if lvl == num_levels-1-num_set_lvls else EMD #TODO change from L1 to actual metric used in round (num_levels-num_set_lvls)
                dist = dist_metric(clusters_mean[lvl+1, i], clusters_mean[lvl+1, j])
                distances_mean[i,j] = dist
                distances_mean[j,i] = dist

        #  for each point/node (x_n), compute histogram (H(x_n))
        histograms = [] # one for each "?point"
        nodes_cur_lvl = nodes_at_lvl[lvl]
        for i in range(len(nodes_cur_lvl)):
            node = nodes_cur_lvl[i]
            child_list = children(node)
            histograms[i] = 0 #TODO set histogram values (?? Based on chance, but what if no chance action and just opponent move?)
        
        # precompute distances btwn histograms with EMD metric
        distances_hist = {}
        for i in range(len(nodes_cur_lvl)):
            for j in range(i+1, len(nodes_cur_lvl)):
                dist = EMD(histograms[i], histograms[j])
                distances_mean[i,j] = dist
                distances_mean[j,i] = dist

        #   compute new cluster means (cluster histograms into C_n clusters using k-means)
        means = k_means(EMD, num_clusters[lvl], histograms, distances_hist)
        for cl in range(num_clusters[lvl]):
            clusters_mean[(lvl,cl)] = means[cl]
    
def k_means(dist_metric: Callable[[list[float], list[float]], float], k: int, points: list, distance_dict: dict): #might delte distance_dict
    pass 
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html : not clear how to use custom distance metric (will need to compute distance btwn histograms)

def EMD(hist1 : list[float], hist2 : list[float]) -> float:
    pass
    # https://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance : Missing for Python3 Q-Q
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html : perhaps
    # https://github.com/wmayner/pyemd : current best option

def equity_from_rollout(nodes : list) -> float:
    pass
    # depends on the representation of node/game tree & how we decide to parse the json


# EMD heuristic
# goal: approx k means where "point" is histogram over (potential future) clusters, represented as a vector of indices of future clusters
# populate sortedDistances array st entry i,j is distance btwn next-round cluster i and j'th closest cluster to i (where current mean has nonzero probability ??) (how do you compute closeness of clusters? by their means?)
# populate orderedClusters array st entry i,j is index of j'th closest cluster to i (where current mean has nonzero probability- mean is a histogram)