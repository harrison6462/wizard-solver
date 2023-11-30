from open_card_wizard import *
#from wizard import *
from open_spiel.python.policy import Policy, UniformRandomPolicy
import numpy as np
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import itertools
from typing import Callable
import math
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms import mcts


UCT_C = math.sqrt(2)
_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product([Face.JESTER, Face.KING, Face.ACE, Face.WIZARD], [Suit.CLUB, Suit.DIAMOND])))



game: WizardGame = pyspiel.load_game('open_python_wizard')
DECK = list(_DECK)
hand = [{DECK[0], DECK[1]}, {DECK[2], DECK[3]}]
state = game.new_initial_state(game, hand=hand, trump_card=DECK[4], next_player=0)



max_simulations = 100
evaluator = mcts.RandomRolloutEvaluator(n_rollouts=20)
bots = [
    mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
    mcts.MCTSBot(game, UCT_C, max_simulations, evaluator),
]

v = evaluate_bots.evaluate_bots(state, bots, np.random)
assert(v[0] + v[1] == 0)
breakpoint()