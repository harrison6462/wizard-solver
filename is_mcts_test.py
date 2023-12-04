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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms import evaluate_bots
import pyspiel
from open_card_wizard import *
SEED = 12983641


game = pyspiel.load_game('open_python_wizard')
_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product([Face.JESTER, Face.KING, Face.ACE, Face.WIZARD], [Suit.CLUB, Suit.DIAMOND])))
Deck = list(_DECK)
hands = [set(Deck[:2]), set(Deck[2:4])]
trump = Deck[4]

state = game.new_initial_state(game=game, hand=hands, trump_card=trump, next_player=0)

breakpoint()






