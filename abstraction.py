from typing import Callable
from pyemd import emd
import numpy as np
import old_wizard_not_open_spiel as wizard
import itertools
from nltk.cluster import KMeansClusterer, euclidean_distance
import time

# ______________________________________________________________________________________________________________________________________________________________________________________________
# ISOMORPHISMS (ignore)

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



# ______________________________________________________________________________________________________________________________________________________________________________________________

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
            histograms[i] = 0 #TODO set histogram values (?? Based on chance, but what if no chance action and just opponent move?) (emailed Tuomas for clarificaton)
        
        # precompute distances btwn histograms with EMD metric
        distances_hist = {}
        for i in range(len(nodes_cur_lvl)):
            for j in range(i+1, len(nodes_cur_lvl)):
                dist = EMD(histograms[i], histograms[j])
                distances_mean[i,j] = dist
                distances_mean[j,i] = dist
        distance_matrix = np.array([[0.0 if i==j else distance_dict[i,j] for j in range(len(nodes_cur_lvl))] for i in range(len(nodes_cur_lvl))]) # defines ground distances btween elts

        #   compute new cluster means (cluster histograms into C_n clusters using k-means)
        means = k_means(EMD, num_clusters[lvl], histograms, distances_hist)
        for cl in range(num_clusters[lvl]):
            clusters_mean[(lvl,cl)] = means[cl]
    
def k_means(dist_metric: Callable[[list[float], list[float]], float], k: int, points: list, distance_dict: dict) : #might delte distance_dict
    pass 
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html : no support for custom distance metric (will need to compute distance btwn histograms)
    # https://www.nltk.org/api/nltk.cluster.kmeans.html cur best option, unfortunately seems like might return fewer means than specified?
    # https://github.com/annoviko/pyclustering 
    # http://bonsai.hgc.jp/~mdehoon/software/cluster/software.htm : also no custom distance metric D:

# hist1, hist 2 are histograms with bins corresponding to clusters in next level. distance_dict[i,j] is distance btwn histograms corresponding to mean of cluster i and mean of cluster j of next level.
def EMD(hist1 : list[float], hist2 : list[float], distance_matrix: dict) -> float:
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)
    # distance_matrix = np.array([[0.0 if i==j else distance_dict[i,j] for j in range(len(hist2))] for i in range(len(hist1))]) # defines ground distances btween elts
    return emd(hist1, hist2, distance_matrix) 
    # https://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance : Missing for Python3 Q-Q
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html : perhaps
    # https://github.com/wmayner/pyemd : current best option (currently using)

def equity_from_rollout(nodes : list) -> float:
    pass
    # depends on the representation of node/game tree & how we decide to parse the json


# EMD heuristic
# goal: approx k means where "point" is histogram over (potential future) clusters, represented as a vector of indices of future clusters
# populate sortedDistances array st entry i,j is distance btwn next-round cluster i and j'th closest cluster to i (where current mean has nonzero probability ??) (how do you compute closeness of clusters? by their means?)
# populate orderedClusters array st entry i,j is index of j'th closest cluster to i (where current mean has nonzero probability- mean is a histogram)

# ______________________________________________________________________________________________________________________________________________________________________________________________
# Wait nvm potential-aware doesn't apply because there's no chance actions after all cards are dealt. Instead try distribution aware abstraction:
# DISTR-AWARE (HAND):

# add profiling?

max_min_dict = {}
def get_equities_fn(my_hand, deck, trump):
    # on full deck, 10 s/it... 95 hours for full run :/
    start_time = time.time()
    # max_min_dict = {} # made global for DP efficiency
    global max_min_dict
    print(len(max_min_dict))
    max_guaranteed_wins_list = []
    min_guaranteed_wins_list = []
    deck = [c for c in deck] # 'deepcopy'
    for card in my_hand:
        deck.remove(card)
    opp_hands = list(itertools.combinations(deck, hand_num_cards))

    for opp_hand in opp_hands:
        minP1, maxP1, minP2, maxP2 = max_min_cards(tuple(my_hand), tuple(opp_hand), max_min_dict, trump)
        max_guaranteed_wins_list.append(maxP1)
        min_guaranteed_wins_list.append(minP1)
    # hist_max = np.histogram(max_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    # hist_min = np.histogram(min_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    # return hist_min, hist_max
    end_time = time.time()
    print(f'get_equities_fn runtime: {end_time - start_time}\n')
    return min_guaranteed_wins_list, max_guaranteed_wins_list

# calculate 'equity' histogram for every possible hand, then k-means cluster with EMD on histograms as distance metric
def distr_abstraction(num_clusters: int, cards_per_hand, deck, trump_suit) -> list:
    # for player 1
    hand_to_histogram = {}
    histogram_to_hand = {}
    hist_list = []
    hand_list = []
    possible_hands = list(itertools.combinations(deck, cards_per_hand))
    print(f'current clustering algorithm on {len(possible_hands)} possible hands\n')
    for hand in possible_hands:
        # hist_min, hist_max = get_equities_fn(hand, deck)
        list_min, list_max = get_equities_fn(hand, deck, trump_suit)
        # histogram_dict[hand] = (hist_min, hist_max)
        list_max_shifted = [i + 2*(cards_per_hand+1) for i in list_max] #so we don't mess up EMD by being too close to list_min
        list_combined = list_min + list_max_shifted
        hist = np.histogram(list_combined, bins=np.arange(3*(cards_per_hand+1)+1), density=True)
        # hand_to_histogram[hand] = hist
        # histogram_to_hand[tuple(hist[0])] = hand
        hist_list.append(hist[0])
        hand_list.append(hand)

    def EMD_basic(hist1 : list[float], hist2 : list[float]) -> float:
        start_time = time.time()
        hist1 = np.array(hist1)
        hist2 = np.array(hist2)
        distance_matrix = np.array([[float(abs(i-j)) for j in range(len(hist1))] for i in range(len(hist1))]) # defines ground distances btween elts
        
        end_time = time.time()
        print(f'EMD_basic runtime: {end_time - start_time}\n')
        # breakpoint() # some are 0? kinda sus
        return emd(hist1, hist2, distance_matrix) 

    def EMD_avg(hand1, hand2) -> float:
        hist_min1, hist_max1 = histogram_dict[hand1]
        hist_min2, hist_max2 = histogram_dict[hand2]
        return EMD_basic(hist_min1, hist_min2) + EMD(hist_max1, hist_max2)
    
    def k_means(vectors, num_clusters):
        start_time = time.time()
        clusterer = KMeansClusterer(num_clusters, EMD_basic, initial_means=None, avoid_empty_clusters=True)
        clusters = clusterer.cluster(vectors, True, trace=True)
        
        end_time = time.time()
        print(f'k_means runtime: {end_time - start_time}\n')
        return clusters

    cluster_ids = k_means(hist_list, num_clusters) #EMD_avg is average of EMD of hist_min and EMD of hist_max
    clusters = [[] for i in range(num_clusters)]
    for i in range(len(hand_list)): #assume hist_list order same as hand_list (which should be true by how i coded it)
        cluster_id = cluster_ids[i]
        hand = hand_list[i]
        clusters[cluster_id].append(hand)
    return clusters


# ______________________________________________________________________________________________________________________________________________________________________________________________
def cmp_cards(c2, c1, trump): # does c2 beat c1?
        return wizard.get_cmp_from_trump_and_initial_suit(trump, c1.suit)(c2, c1)
# dict is P1hand,P2hand -> minP1, maxP1, minP2, maxP2
def max_min_cards(p1_hand, p2_hand, max_min_dict, trump):
    # around 0.0005140304565429688 s/it
    # start_time = time.time()

    if (p1_hand, p2_hand) in max_min_dict: return max_min_dict[(p1_hand, p2_hand)]
    # print(str(p1_hand), str(p2_hand))
    hand_num_cards = len(p1_hand)
    cur_max_p1 = 0
    cur_min_p1 = hand_num_cards
    cur_max_p2 = hand_num_cards #! sus
    cur_min_p2 = 0
    for c1 in p1_hand:
        min_subgame_p1 = 0
        max_subgame_p1 = hand_num_cards
        min_subgame_p2 = hand_num_cards
        max_subgame_p2 = 0

        for c2 in p2_hand:
            subhand_1 = tuple([c for c in p1_hand if c != c1])
            subhand_2 = tuple([c for c in p2_hand if c != c2]) # p2_hand.remove(c2) wrong because needs to be pure
            if len(subhand_1) == 0: # base case
                # breakpoint()
                minP2, maxP2, minP1, maxP1 = 0,0,0,0
                p1wins = 0 if cmp_cards(c2,c1,trump) else 1
                p2wins = 1 - p1wins
            elif wizard.is_valid_move(c1, c2, p2_hand): # p1 goes first as set in the convention for max_min_dict
                if cmp_cards(c2,c1,trump): # p2 wins 
                    minP2, maxP2, minP1, maxP1 = max_min_cards(subhand_2, subhand_1, max_min_dict, trump) # cur p2 is starting player in subgame
                    p1wins = 0
                    p2wins = 1
                else: # p1 wins
                    minP1, maxP1, minP2, maxP2 = max_min_cards(subhand_1, subhand_2, max_min_dict, trump) # cur p1 is starting player in subgame
                    p1wins = 1
                    p2wins = 0
            else: continue
            min_subgame_p1 = max(minP1+p1wins, min_subgame_p1)
            max_subgame_p1 = min(maxP1+p1wins, max_subgame_p1)
            min_subgame_p2 = min(minP2+p2wins, min_subgame_p2)
            max_subgame_p2 = max(maxP2+p2wins, max_subgame_p2)
            
        cur_min_p1 = min(cur_min_p1, min_subgame_p1)
        cur_max_p1 = max(cur_max_p1, max_subgame_p1)
        cur_min_p2 = max(cur_min_p2, min_subgame_p2)
        cur_max_p2 = min(cur_max_p2, max_subgame_p2)
    max_min_dict[(p1_hand, p2_hand)] = cur_min_p1, cur_max_p1, cur_min_p2, cur_max_p2

    # end_time = time.time()
    # print(f'max_min_cards runtime: {end_time - start_time}\n')
    return max_min_dict[(p1_hand, p2_hand)]

# def max_min_cards(hand, deck, hand_num_cards, trump):
#     max_min_dict = {} #P1hand,P2hand -> minP1, maxP1, minP2, maxP2
#     for i in range(1, hand_num_cards+1):
#         print(i)
#         combined_hands = list(itertools.combinations(deck, i*2))
#         print("num combined hands:", len(combined_hands))
#         hand_pair_list = []
#         print("adding all combinations to list")
#         count = 0
#         for ch in combined_hands:
#             count +=1 
#             if count % 50000 == 0 : print(count)
#             p1_hands = list(itertools.combinations(ch, i))
#             p2_hands = [[c for c in list(ch) if c not in h] for h in p1_hands]
#             hand_pair_list += zip(p1_hands, p2_hands)
#         print("num combined hand pairs:", len(hand_pair_list))
#         print("tabulating results")
#         count = 0
#         for hand_pair in hand_pair_list:
#             count +=1 
#             if count % 50000 == 0 : print(count)
#             # print(hand_pair)
#             p1_hand = list(hand_pair[0])
#             p2_hand = list(hand_pair[1])
#             cur_max_p1 = 0
#             cur_min_p1 = i
#             cur_max_p2 = 0
#             cur_min_p2 = i
#             for c1 in p1_hand:
#                 min_subgame_p1 = i
#                 max_subgame_p1 = 0
#                 min_subgame_p2 = i
#                 max_subgame_p2 = 0

#                 for c2 in p2_hand:
#                     subhand_1 = [c for c in p1_hand if c != c1]
#                     subhand_2 = [c for c in p2_hand if c != c2] # p2_hand.remove(c2) wrong because needs to be pure
#                     if len(subhand_1) == 0: # base case
#                             minP2, maxP2, minP1, maxP1 = 0,0,0,0
#                             p1wins = 0 if cmp_cards(c2,c1,trump) else 1
#                             p2wins = 1 - p1wins
#                     elif wizard.is_valid_move(c1, c2, subhand_2): # p1 goes first as set in the convention for max_min_dict
                        
#                         if cmp_cards(c2,c1,trump): # p2 wins 
#                             minP2, maxP2, minP1, maxP1 = max_min_dict[(subhand2, subhand1)] # cur p2 is starting player in subgame
#                             p1wins = 0
#                             p2wins = 1
#                         else: # p1 wins
#                             minP1, maxP1, minP2, maxP2 = max_min_dict[(subhand1, subhand2)] # cur p2 is starting player in subgame
#                             p1wins = 0
#                             p2wins = 1
#                     min_subgame_p1 = max(minP1+p1wins, min_subgame_p1)
#                     max_subgame_p1 = min(maxP1+p1wins, max_subgame_p1)
#                     min_subgame_p2 = min(minP2+p2wins, min_subgame_p2)
#                     max_subgame_p2 = max(maxP2+p2wins, max_subgame_p2)
                    
#                 cur_min_p1 = min(cur_min_p1, min_subgame_p1)
#                 cur_max_p1 = max(cur_max_p1, max_subgame_p1)
#                 cur_min_p2 = max(cur_min_p2, min_subgame_p2)
#                 cur_max_p2 = min(cur_max_p2, max_subgame_p2)
#             max_min_dict[(p1_hand, p2_hand)] = cur_min_p1, cur_max_p1, cur_min_p2, cur_max_p2
#     return max_min_dict


# approx E(wins) by calculating E(guaranteed wins), E(guaranteed losses)
#* not used rn, ignore
def equity(hand, trump_suit, player_id): 
    hand_num_cards = len(hand)
    deck = wizard.deck
    ordered_my_hands = list(itertools.permutations(hand))
    for card in hand:
        deck.remove(card)
    opp_hands = list(itertools.combinations(deck, hand_num_cards))
    def cmp_cards(c2, c1, trump): # does c2 beat c1?
        return wizard.get_cmp_from_trump_and_initial_suit(trump, c1.suit)(c2, c1)
    def count_wins(p1_strat, p2_strat, trump):
        wins = 0
        # breakpoint()
        p1_starting = True
        for i in range(len(p1_strat)):
            p1_card = p1_strat[i]
            p2_card = p2_strat[i]
            if p1_starting:
                if cmp_cards(p2_card, p1_card, trump): # if p1 starts and loses
                    p1_starting = False
                else:
                    p1_starting = True
                    wins += 1
            else:
                if cmp_cards(p1_card, p2_card, trump): # if p2 starts and loses
                    p1_starting = True
                    wins += 1
                else: 
                    p1_starting = False
        return wins
    def is_valid_strat(p1_strat, p2_strat, trump, p1_start):
        p1_starting = p1_start
        for i in range(len(p1_strat)):
            p1_card = p1_strat[i]
            p2_card = p2_strat[i]
            if p1_starting:
                if not wizard.is_valid_move() : return False
                if cmp_cards(p2_card, p1_card, trump): # if p1 starts and loses
                    p1_starting = False
                else:
                    p1_starting = True
            else:
                if cmp_cards(p1_card, p2_card, trump): # if p2 starts and loses
                    p1_starting = True
                else: 
                    p1_starting = False
        return wins


    max_guaranteed_wins_list = []
    min_guaranteed_wins_list = []

    for opp_hand in opp_hands: 
        ordered_opp_hands = list(itertools.permutations(opp_hand))
        # max_wins = hand_num_cards
        # min_wins = 0
        max_guaranteed_wins = 0
        min_guaranteed_wins = hand_num_cards

        for ord_my_hand in ordered_my_hands: # I choose strat/ordering
            # valid_ordered_opp_hands = filter(lambda l: , ordered_opp_hands) # makes sure ordering  is possible under wizard rules

            cur_strat_guarantee_max = hand_num_cards
            cur_strat_guarantee_min = 0
            for ord_opp_hand in ordered_opp_hands: # opponent chooses strat/ordering
                mutual_strat_wins = count_wins(ord_my_hand, ord_opp_hand, trump_suit) if player_id == 1 else count_wins(ord_opp_hand, ord_my_hand, trump_suit)
                cur_strat_guarantee_max = min(cur_strat_guarantee_max, mutual_strat_wins)
                cur_strat_guarantee_min = max(cur_strat_guarantee_min, mutual_strat_wins)
            max_guaranteed_wins = max(max_guaranteed_wins, cur_strat_guarantee_max)
            min_guaranteed_wins = min(min_guaranteed_wins, cur_strat_guarantee_min)
        #     print("inside loop", max_guaranteed_wins)
        # print("outside loop", max_guaranteed_wins)
        max_guaranteed_wins_list.append(max_guaranteed_wins)
        min_guaranteed_wins_list.append(min_guaranteed_wins)
    
    hist_max = np.histogram(max_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    hist_min = np.histogram(min_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    print(hist_max)
    print(hist_min)
    pass

# ______________________________________________________________________________________________________________________________________________________________________________________________
# DISTR-AWARE (CARD):

def get_equities_fn_card(my_card, deck, trump):
    start_time = time.time()
    wincount_list = []
    deck = [c for c in deck] # 'deepcopy'
    deck.remove(my_card)
    opp_possible_cards = deck

    # if my_card.face == wizard.Face.JESTER or my_card.face == wizard.Face.THREE: breakpoint()
    for opp_card in opp_possible_cards:
        wincount = 0
        if cmp_cards(my_card, opp_card, trump): wincount += 1
        if not cmp_cards(opp_card, my_card, trump): wincount += 1
        wincount_list.append(wincount)

    end_time = time.time()
    print(f'get_equities_fn runtime: {end_time - start_time}\n')
    return wincount_list

# could improve by taking into account possible hands formed by card, even if card is the only clustering item
# calculate 'equity' histogram for every possible card, then k-means cluster with EMD on histograms as distance metric
def distr_abstraction_card(num_clusters: int, deck, trump_suit) -> list:
    # for player 1
    hist_list = []
    card_list = []
    print(f'current clustering algorithm on {len(deck)} possible hands\n')
    for card in deck:
        list_num_wins = get_equities_fn_card(card, deck, trump_suit)
        hist = np.histogram(list_num_wins, bins=np.arange(3), density=True)
        hist_list.append(hist[0])
        card_list.append(card)

    def EMD(hist1 : list[float], hist2 : list[float]) -> float:
        start_time = time.time()
        hist1 = np.array(hist1)
        hist2 = np.array(hist2)
        distance_matrix = np.array([[float(abs(i-j)) for j in range(len(hist1))] for i in range(len(hist1))]) # defines ground distances btween elts
        
        end_time = time.time()
        # print(f'EMD runtime: {end_time - start_time}\n')
        return emd(hist1, hist2, distance_matrix) 

    def k_means(vectors, num_clusters):
        start_time = time.time()
        clusterer = KMeansClusterer(num_clusters, EMD, initial_means=None, avoid_empty_clusters=True)
        clusters = clusterer.cluster(vectors, True, trace=True)
        
        end_time = time.time()
        print(f'k_means runtime: {end_time - start_time}\n')
        return clusters

    cluster_ids = k_means(hist_list, num_clusters) 
    clusters = [[] for i in range(num_clusters)]
    for i in range(len(card_list)): #assume hist_list order same as card_list (which should be true by how i coded it)
        cluster_id = cluster_ids[i]
        card = card_list[i]
        clusters[cluster_id].append(card)
    return clusters

if __name__== "__main__":
    hands, remaining_deck = wizard.generate_hands(2, 3, wizard.deck)
    my_hand = list(hands[0])

    # my_hand = [wizard.Card(wizard.Face.WIZARD, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART) ]
    my_hand = [wizard.Card(wizard.Face.JESTER, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART) ]
    trump = list(my_hand)[0].suit
    print(trump, my_hand)
    hand_num_cards = len(my_hand)
    # equity(my_hand, trump, 1)

    '''
    max_min_dict = {}
    max_guaranteed_wins_list = []
    min_guaranteed_wins_list = []
    deck = wizard.deck
    for card in my_hand:
        deck.remove(card)
    opp_hands = list(itertools.combinations(deck, hand_num_cards))
    for opp_hand in opp_hands:
        print(my_hand, opp_hand)
        minP1, maxP1, minP2, maxP2 = max_min_cards(tuple(my_hand), tuple(opp_hand), max_min_dict, trump)
        max_guaranteed_wins_list.append(maxP1)
        min_guaranteed_wins_list.append(minP1)
        print(maxP1, minP1)
    hist_max = np.histogram(max_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    hist_min = np.histogram(min_guaranteed_wins_list, bins=np.arange(hand_num_cards+2), density=True)
    print(hist_max)
    print(hist_min)
    breakpoint()
    '''


    # TESTS
    '''
    test_deck_1 = set([wizard.Card(wizard.Face.JESTER, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.JESTER, wizard.Suit.SPADE), wizard.Card(wizard.Face.JESTER, wizard.Suit.HEART), wizard.Card(wizard.Face.JESTER, wizard.Suit.CLUB),
                     wizard.Card(wizard.Face.WIZARD, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART), wizard.Card(wizard.Face.WIZARD, wizard.Suit.CLUB)])
    
    num_clusters = 4
    hand_num_cards = 3
    clusters = distr_abstraction(num_clusters, hand_num_cards, test_deck_1, wizard.Suit.DIAMOND)

    for i in range(len(clusters)):
        cluster = clusters[i]
        num_wizards_list = []
        for hand in cluster:
            num_wizards = 0
            for card in hand:
                if card.face == wizard.Face.WIZARD: num_wizards += 1
            num_wizards_list.append(num_wizards)
        print(f'num wizards in cluster {i+1}: {num_wizards_list}')
    '''

    '''
    test_deck_2 = set([wizard.Card(wizard.Face.TWO, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.THREE, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.TEN, wizard.Suit.HEART), wizard.Card(wizard.Face.TEN, wizard.Suit.CLUB),
                     wizard.Card(wizard.Face.WIZARD, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART), wizard.Card(wizard.Face.WIZARD, wizard.Suit.CLUB)])
    
    num_clusters = 4
    hand_num_cards = 3
    clusters = distr_abstraction(num_clusters, hand_num_cards, test_deck_2, wizard.Suit.DIAMOND)

    for i in range(len(clusters)):
        cluster = clusters[i]
        card_list = []
        for hand in cluster:
            card_list.append(list(map(str, hand)))
        print(f'hands in cluster {i+1}: {card_list}')
    '''

    '''
    test_deck_3 = set([wizard.Card(wizard.Face.TWO, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.THREE, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.FOUR, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.FIVE, wizard.Suit.DIAMOND),
                     wizard.Card(wizard.Face.TWO, wizard.Suit.HEART), wizard.Card(wizard.Face.THREE, wizard.Suit.HEART), wizard.Card(wizard.Face.FOUR, wizard.Suit.HEART), wizard.Card(wizard.Face.FIVE, wizard.Suit.HEART),
                     wizard.Card(wizard.Face.TWO, wizard.Suit.SPADE), wizard.Card(wizard.Face.THREE, wizard.Suit.SPADE), wizard.Card(wizard.Face.FOUR, wizard.Suit.SPADE), wizard.Card(wizard.Face.FIVE, wizard.Suit.SPADE),
                     wizard.Card(wizard.Face.JESTER, wizard.Suit.SPADE), wizard.Card(wizard.Face.JESTER, wizard.Suit.HEART), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART)
                     ])
    
    num_clusters = 20
    hand_num_cards = 3
    clusters = distr_abstraction(num_clusters, hand_num_cards, test_deck_3, wizard.Suit.DIAMOND)

    for i in range(len(clusters)):
        cluster = clusters[i]
        card_list = []
        for hand in cluster:
            card_list.append(list(map(str, hand)))
        print(f'hands in cluster {i+1}: {card_list}')
    '''

    '''
    test_deck_4 = set([wizard.Card(wizard.Face.TWO, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.THREE, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.JESTER, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.WIZARD, wizard.Suit.DIAMOND),
                     wizard.Card(wizard.Face.TWO, wizard.Suit.HEART), wizard.Card(wizard.Face.THREE, wizard.Suit.HEART), wizard.Card(wizard.Face.JESTER, wizard.Suit.HEART), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART),
                     ])
    
    num_clusters = 4
    hand_num_cards = 3
    clusters = distr_abstraction_card(num_clusters, test_deck_4, wizard.Suit.DIAMOND)

    for i in range(len(clusters)):
        cluster = clusters[i]
        card_list = []
        for card in cluster:
            card_list.append(str(card))
        print(f'cards in cluster {i+1}: {card_list}')
    '''

    test_deck_5 = set([wizard.Card(wizard.Face.TWO, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.THREE, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.FOUR, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.FIVE, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.SIX, wizard.Suit.DIAMOND),
                     wizard.Card(wizard.Face.TWO, wizard.Suit.HEART), wizard.Card(wizard.Face.THREE, wizard.Suit.HEART), wizard.Card(wizard.Face.FOUR, wizard.Suit.HEART), wizard.Card(wizard.Face.FIVE, wizard.Suit.HEART), wizard.Card(wizard.Face.SIX, wizard.Suit.HEART),
                     wizard.Card(wizard.Face.TWO, wizard.Suit.SPADE), wizard.Card(wizard.Face.THREE, wizard.Suit.SPADE), wizard.Card(wizard.Face.FOUR, wizard.Suit.SPADE), wizard.Card(wizard.Face.FIVE, wizard.Suit.SPADE), wizard.Card(wizard.Face.SIX, wizard.Suit.SPADE),
                     wizard.Card(wizard.Face.JESTER, wizard.Suit.SPADE), wizard.Card(wizard.Face.JESTER, wizard.Suit.HEART), wizard.Card(wizard.Face.WIZARD, wizard.Suit.SPADE), wizard.Card(wizard.Face.WIZARD, wizard.Suit.HEART)
                     ])

    num_clusters = 5
    hand_num_cards = 3
    clusters = distr_abstraction_card(num_clusters, test_deck_5, wizard.Suit.DIAMOND)

    for i in range(len(clusters)):
        cluster = clusters[i]
        card_list = []
        for card in cluster:
            card_list.append(str(card))
        print(f'cards in cluster {i+1}: {card_list}')

    breakpoint()
