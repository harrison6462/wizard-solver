# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit test for Information Set MCTS bot.

This test mimics the basic C++ tests in algorithms/is_mcts_test.cc.
"""
# pylint: disable=g-unreachable-test-method
from __future__ import annotations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import mcts
import pyspiel
from open_card_wizard import *
import random
from copy import deepcopy 
_NUM_PLAYERS=2

def keywithmaxval(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))]

def is_valid_move_all_hand(preceeding_card: Card | None, card: Card, hand: Hand) -> bool:
    return (preceeding_card is None \
                            or card.face in [Face.WIZARD, Face.JESTER] \
                            or card.suit == preceeding_card.suit \
                            or not has_card_of_suit(hand, preceeding_card.suit))



def gen_hand(card_pool, player_card_amounts):
    num = 0
    for val in player_card_amounts:
        num += val 
    sample = random.sample(list(card_pool), num)
    hands = []
    idx = 0
    for val in player_card_amounts:
        hands.append(sample[idx:idx+val])
        idx += val
    return hands


def valid_hand(state, hands):
    played_so_far: list[set[Card]] = [set() for _ in range(_NUM_PLAYERS)]
    valid = True
    for round in range(len(state.previous_tricks)):  
        for player in range(state.who_started_tricks[round], state.who_started_tricks[round] + _NUM_PLAYERS):
            curr = player % _NUM_PLAYERS
           # print(type(hands[curr]))
           # print(type(played_so_far[curr]))
            if not is_valid_move_all_hand(state.previous_tricks[round][0], state.previous_tricks[round][curr], set(hands[curr]) - played_so_far[curr]): 
                return False
            played_so_far[curr].add(state.previous_tricks[round][curr])
    return True
        
class hand_sampling(Exception):
    pass


def sample(state, player_id, num_samples):
    
    player_cards_to_see = {player_id}
    out = []
    player_card_amounts = [len(hand) for i, hand in enumerate(state.player_hands) if i not in player_cards_to_see]
    all_exposed_cards = state.get_all_exposed_cards(player_cards_to_see)
    card_pool = state.Deck - all_exposed_cards
    count = 0
    while(len(out) != num_samples):
        hand = gen_hand(card_pool, player_card_amounts)
        for player in sorted(player_cards_to_see): 
            hand.insert(player, list(deepcopy(state.player_hands[player])))
        if valid_hand(state, hand):
            out.append(hand)
        count += 1
        if(count == 1000):
            print("taking too long to find hands")
            raise hand_sampling
            
    #breakpoint()
   
    return out


def do_action(state , player_id):
    if(state._next_player != player_id):
        raise Exception('not my turn')
    
    our_current_hand = state.player_hands[player_id]

    game = pyspiel.load_game('open_python_wizard')
    actions = dict()
    
    sampled_hands = sample(state, player_id, 50)
    
    for hand in sampled_hands:
        try:
            assert len(hand[0]) == len(state.player_hands[0])
            assert len(hand[1]) == len(state.player_hands[1])
        except:
            breakpoint()
        dummy_game = game.new_initial_state(game=game, hand=deepcopy(hand), trump_card = state.trump_card, next_player=state._next_player, initial_player_hands=deepcopy(state.initial_player_hands),                                   predictions =deepcopy(state.predictions), who_started_tricks = deepcopy(state.who_started_tricks), previous_tricks=deepcopy(state.previous_tricks), current_round_cards=deepcopy(state.current_round_cards), tricks_per_player=deepcopy(state.tricks_per_player), current_winning_card=state.current_winning_card,current_winning_player=state.current_winning_player, current_lead_suit=state.current_lead_suit, metadata=state.metadata, Deck=state.Deck)
        sim_nums = 100
        rng = np.random.RandomState()
        evaluator = mcts.RandomRolloutEvaluator(1, rng)
        bot = mcts.MCTSBot(
            game,
            2, #uct_c
            sim_nums, #max_simulations
            evaluator,
            random_state=rng,
            solve=True,
            verbose=False)
       
      
        action_to_take = bot.step(dummy_game)
       
        if action_to_take in actions:
            actions[action_to_take] += 1
        else:
            actions[action_to_take] = 1
    
    return keywithmaxval(actions)


'''
game = pyspiel.load_game('open_python_wizard')
_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product([Face.JESTER, Face.KING, Face.ACE, Face.WIZARD], [Suit.CLUB, Suit.DIAMOND])))
Deck = list(_DECK)
hands = [set(Deck[:2]), set(Deck[2:4])]
trump = Deck[4]

state = game.new_initial_state(game=game, hand=hands, trump_card=trump, next_player=0)



'''


