from __future__ import annotations
# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel

from dataclasses import dataclass
from enum import Enum
from functools import reduce
import itertools
from typing import Callable
import math
from copy import copy, deepcopy
from wizard import *



#types
Hand = set[Card]

'''Conceptually, a deck is just the set of cards that we want to use in a given round.
   We could set it to be all possible cards (subsets of Faces * Suits), or whatever we want.
'''
Deck = set[Card]
'''An ordered deck represents a permutation of cards. Useful for shuffling.
'''
OrderedDeck = list[Card]

faces = [face for face in Face]
suits = [suit for suit in Suit]

card = tuple[Face, Suit]


#the possible types of actions
BetAction = int
CardAction = Card

_NUM_PLAYERS = 2
_NUM_CARDS_PER_PLAYER = 2
#wizards / jesters will have club/diamond/heart/spade variety - these will have no impact
#_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product(faces, suits)))
_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product([Face.JESTER,  Face.KING,Face.ACE, Face.WIZARD], [Suit.CLUB, Suit.DIAMOND])))
#TODO this is an important function, perhaps more attention should be drawn to it
# def card_to_int(card: Card) -> int:
#   '''Gives a numbering of the cards that is bijective in [0, ... len(_DECK)). The order is arbitrary
#   '''
#   return card.face.value * 4 + card.suit.value



max_chance_actions = 1
for i in range(_NUM_PLAYERS): max_chance_actions *= math.comb(len(_DECK)-i*_NUM_CARDS_PER_PLAYER, _NUM_CARDS_PER_PLAYER)
_GAME_TYPE = pyspiel.GameType(
    short_name="open_python_wizard",
    long_name="Open Python Wizard",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM, #TODO this is a lie
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(_DECK), #either a bet action (_NUM_CARDS_PER_PLAYER+1) or playing a card
    max_chance_outcomes=int(max_chance_actions) + len(_DECK), #TODO this overflows if it's too big
    num_players=_NUM_PLAYERS,
    min_utility=-_NUM_CARDS_PER_PLAYER*10.0,
    max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
    utility_sum=0,
    max_game_length=len(_DECK) + _NUM_PLAYERS + 2)  # for 1 round, 60 cards + 6 players + 2 chance


class OpenWizardGame(pyspiel.Game):
  """A Python version of Wizard."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self, game, hand, trump_card, next_player, initial_player_hands, predictions=[], who_started_tricks = [], previous_tricks= [], current_round_cards=[], tricks_per_player = [0 for _ in range(_NUM_PLAYERS)], current_winning_card=None,current_winning_player=None, current_lead_suit=None, metadata=0,Deck=_DECK):
    """Returns a state corresponding to the start of a game."""
    return OpenWizardState(self, hand, trump_card, next_player, initial_player_hands, predictions, who_started_tricks, previous_tricks, current_round_cards, tricks_per_player, current_winning_card,current_winning_player, current_lead_suit, metadata, Deck) 

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return WizardObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
        params)


class OpenWizardState(pyspiel.State):
  """Class representing a Wizard state.
      Concerns with this approach: there are far too many chance outcomes to be computationally feasible -
      in particular, this will invoke a computation with prod_{i=0}^(num_players-1) nCr(|deck| - i*num_cards_per_player, num_cards_per_player)
      nodes, which is absolutely horrendous.
      
      There are some quirks with how this is represented; open_spiel requires that all of the actions be given unique integer indexes for chance
      and for the players, so the scheme is as follows:
      chance actions: [0-max_chance_actions) is an index that represents how the players hands should be dealt, and 
      [max_chance_actions, max_chance_actions+len(_DECK)-(cards dealt to players) represents which card was dealt from the deck for the trump card

      And, player actions: [0, _NUM_CARDS_PER_PLAYER+1) is the bet amount, and [_NUM_CARDS_PER_PLAYER+1, _NUM_CARDS_PER_PLAYER+1 + len(_DECK)) is
      which card they played
  """

  def __init__(self, game, hand, trump_card, next_player, initial_player_hands, predictions=[], who_started_tricks = [], previous_tricks= [], current_round_cards=[], tricks_per_player = [0 for _ in range(_NUM_PLAYERS)], current_winning_card=None,current_winning_player=None, current_lead_suit=None, metadata=0,Deck=_DECK):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.player_hands: list[Hand] = hand
    self.initial_player_hands = initial_player_hands
    self.predictions = predictions
    self.who_started_tricks: list[int] = who_started_tricks
    self.previous_tricks: list[list[Card]] = previous_tricks
    self.current_round_cards: list[Card] = current_round_cards
    self.tricks_per_player = tricks_per_player
    self.current_winning_card: Card | None = current_winning_card
    self.current_winning_player: int | None = current_winning_player
    self.current_lead_suit: Suit | None = current_lead_suit
    self.trump_card = trump_card
    self._next_player = next_player
    self.metadata = metadata
    self.Deck = Deck

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
   
    return self._next_player


    
  def _legal_actions(self, player) -> list[BetAction | CardAction]:
    """Returns a list of legal actions."""
    assert player >= 0
    #player still needs to bet
    if len(self.predictions) < _NUM_PLAYERS: 
        return [i for i in range(_NUM_CARDS_PER_PLAYER+1)]
    
    #otherwise, find the first card that determined a suit (if it exists) and play a suit from that
    if self.current_lead_suit is None: 
        return sorted(map(lambda c: _NUM_CARDS_PER_PLAYER+1+card_to_int(c), self.player_hands[player]))

    return sorted(map(lambda c: card_to_int(c)+_NUM_CARDS_PER_PLAYER+1, get_all_valid_moves(self.player_hands[player], Card(Face.TWO, self.current_lead_suit))))



  
  def _apply_action(self, action):
    """Applies the specified action to the state."""
   
    if len(self.predictions) < _NUM_PLAYERS:
        self.predictions.append(action)
        self._next_player += 1
        if self._next_player == _NUM_PLAYERS:
          #find the first index of the highest said number, that's who goes first
          highest_num_idx = 0
          for i in range(_NUM_PLAYERS):
            if self.predictions[i] > self.predictions[highest_num_idx]: highest_num_idx = i
          self._next_player = highest_num_idx
          self.who_started_tricks.append(self._next_player)
    else:
        action = int_to_card(action-(_NUM_CARDS_PER_PLAYER+1))
        
        try:
            assert action in self.player_hands[self._next_player]
        except: 
          breakpoint()
        self.player_hands[self._next_player].remove(action)
        #we can know for sure that this was a valid card
        cmp = get_cmp_from_trump_and_initial_suit(self.trump_card.suit if self.trump_card is not None else None, self.current_lead_suit)
        if action.face not in [Face.JESTER, Face.WIZARD] and self.current_lead_suit is None: self.current_lead_suit = action.suit
        self.current_round_cards.append(action)
        if action.face != Face.JESTER and (self.current_winning_card is None or cmp(action, self.current_winning_card)): 
          self.current_winning_card = action
          self.current_winning_player = self._next_player
        if len(self.current_round_cards) == _NUM_PLAYERS:
          if self.current_winning_player is None: self.current_winning_player = self._next_player
          self._next_player = self.current_winning_player
          self.tricks_per_player[self.current_winning_player] += 1
          self.previous_tricks.append(self.current_round_cards)
          self.current_winning_player = None
          self.current_winning_card = None
          self.current_lead_suit = None
          self.current_round_cards = []
          self.who_started_tricks.append(self._next_player)
        else: self._next_player = (self._next_player + 1) % _NUM_PLAYERS
     
  def clone(self):
    game = pyspiel.load_game('open_python_wizard')
    
    return game.new_initial_state(game=game, hand=deepcopy(self.player_hands), trump_card = self.trump_card, next_player=self._next_player, initial_player_hands=self.initial_player_hands,                                   predictions =deepcopy(self.predictions), who_started_tricks = deepcopy(self.who_started_tricks), previous_tricks=deepcopy(self.previous_tricks), current_round_cards=deepcopy(self.current_round_cards), tricks_per_player=deepcopy(self.tricks_per_player), current_winning_card=self.current_winning_card,current_winning_player=self.current_winning_player, current_lead_suit=self.current_lead_suit, metadata=self.metadata, Deck=self.Deck)
     
  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if action >= max_chance_actions:
       return f"Dealt card for trump suit: {int_to_card(action-max_chance_actions)}"
      #otherwise, it's hands for all the players
      return f'Dealt hands for all the players: {action}'
    else:
      if action <= _NUM_CARDS_PER_PLAYER: return f'Predict {action} tricks'
      return f'Play card {int_to_card(action - _NUM_CARDS_PER_PLAYER -1)}'

  def is_terminal(self):
    """Returns True if the game is over."""
    return len(self.player_hands) == _NUM_PLAYERS and len(self.predictions) == _NUM_PLAYERS and all([len(h) == 1 for h in self.player_hands])

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self.is_terminal():
      return [0. for i in range(_NUM_PLAYERS)]
    for i in range(self._next_player, self._next_player + _NUM_PLAYERS):
      self._apply_action(card_to_action(self.player_hands[i % _NUM_PLAYERS].__iter__().__next__()))
    def reward_for_player(i: int):
      if self.tricks_per_player[i] == self.predictions[i]: return 20 + 10 * self.tricks_per_player[i]
      return -10 * abs(self.tricks_per_player[i] - self.predictions[i]) 
    if _NUM_PLAYERS != 2: raise Exception('Currently, only two players are supported for 0 sum')
    r0,r1 = reward_for_player(0), reward_for_player(1)
    if r0 > r1: return [r0-r1, -(r0-r1)]
    elif r0 == r1: return [0, 0]
    else: return [-(r1-r0), (r1-r0)]
#    return [reward_for_player(i) for i in range(_NUM_PLAYERS)]
 

  def __str__(self):
    return f'Player acting: {self._next_player}\n Predictions: {self.predictions}\n' \
    + f'Trump card: {str(self.trump_card)} lead suit: {self.current_lead_suit}' \
    + f'Current round: {[str(c) for c in self.current_round_cards]}' + f'Hands: {list(map(lambda h: list(map(lambda c: str(c), h)), self.player_hands))} \n' \
    + f'History: {[[str(c) for c in t] for t in self.previous_tricks]}' + f'Initial hands: {[[str(c) for c in t] for t in self.initial_player_hands]}'


class OpenWizardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  """It's not too clear from the documentation what exactly a state is, but
  from what I can tell, the main thing is keeping track of a tensor and dict such that string_from
  can act as an identifer for the infoset for what the game looks like from the perspective of this player,
  and we encode this information in a tensor with perfect recall."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))] #all their games 1-hot encode the player
    #i think they try to make the game state maximally 1-hot encoded for good compression and speed
   
      
    
    if iig_obs_type.public_info:
      pieces.append(("player0_cards", len(_DECK), (len(_DECK),))) #1-hot encode what cards are in our hand
      pieces.append(("player1_cards", len(_DECK), (len(_DECK),)))
      pieces.append(("predictions", _NUM_PLAYERS*(_NUM_CARDS_PER_PLAYER+1), (_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER+1))) #one-hot encoding of prediction for player i
      #one-hot encoding of cards played by each player from the start of each trick, over all tricks (the player who started round i can be deduced
      #inductively by knowing that P0 starts round 0 and keeping track of who won round i-1)
      pieces.append(("trump_card", len(_DECK), (len(_DECK), )))
      pieces.append(("played_cards", len(_DECK) * _NUM_CARDS_PER_PLAYER * _NUM_PLAYERS, (_NUM_CARDS_PER_PLAYER, len(_DECK) * _NUM_PLAYERS))) 
      pieces.append(('metadata', WizardState.MAXIMUM_METADATA_SIZE, (WizardState.MAXIMUM_METADATA_SIZE, )))
        
    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state: WizardState, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1

   

    if 'player0_cards' in self.dict and player < len(state.player_hands):
      for card in state.player_hands[0]:
        self.dict["player0_cards"][card_to_int(card)] = 1

    if 'player1_cards' in self.dict and player < len(state.player_hands):
      for card in state.player_hands[1]:
        self.dict["player1_cards"][card_to_int(card)] = 1


    if "predictions" in self.dict: 
      for i in range(min(_NUM_PLAYERS, len(state.predictions))):
        self.dict["predictions"][i][state.predictions[i]] = 1
    if "played_cards" in self.dict:
      for i in range(len(state.previous_tricks)):
        for card in state.previous_tricks[i]:
          self.dict['played_cards'][i][card_to_int(card)] = 1
      for card in state.current_round_cards:
        self.dict['played_cards'][len(state.previous_tricks)][card_to_int(card)] = 1
    if 'trump_card' in self.dict and state.trump_card is not None:
      self.dict['trump_card'][card_to_int(state.trump_card)] = 1
    if 'metadata' in self.dict:
      for i, c in enumerate(str(bin(state.metadata))[2:]):
        self.dict['metadata'][i] = c


  def string_from(self, state: WizardState, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "player0_cards" in self.dict and len(state.player_hands) > player:
      pieces.append(f"cards in hand: {str(list(map(lambda s: str(s), state.player_hands[0])))}")
    if "player1_cards" in self.dict and len(state.player_hands) > player:
      pieces.append(f"cards in hand: {str(list(map(lambda s: str(s), state.player_hands[1])))}")
    if "predictions" in self.dict:
      pieces.append(f"predictions[{state.predictions}]")
    if "played_cards" in self.dict and len(state.current_round_cards) > 0 or len(state.previous_tricks) > 0:
      for i in range(len(state.previous_tricks)):
        pieces.append(f'tr: {i+1}')
        pieces.append(' '.join(str(card) for card in state.previous_tricks[i]))
      pieces.append('curr:')
      pieces.append(" ".join(str(card) for card in state.current_round_cards))
    if 'metadata' in self.dict:
      pieces.append(f'metadata: {state.metadata}')
    return " ".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, OpenWizardGame)