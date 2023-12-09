from __future__ import annotations
from copy import copy, deepcopy
import random
import wizard

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

@dataclass(frozen=True)
class AbstractedCard:
    bucket: int
    def __str__(self):
        #open_spiel requires all actions to have distinct names
        return f'B{self.bucket}'
    
    def __eq__(self, other):
      if not isinstance(other, AbstractedCard): return False
      return self.bucket == other.bucket

    def __lt__(self, other):
      if not isinstance(other, AbstractedCard): return False
      return self.bucket < other.bucket
    

@dataclass(frozen=True)
class ClusterBucket:
  strength: int
  suit: wizard.Suit | None #None is for Wizard and Jester buckets
  cards: set[wizard.Card]

#TODO finish making clusters
#we require that all cards within a cluster have the same suit and all clusters are nonempy
clusters: list[ClusterBucket] = [ClusterBucket(100, None, set([wizard.Card(wizard.Face.WIZARD, wizard.Suit.CLUB), wizard.Card(wizard.Face.WIZARD, wizard.Suit.DIAMOND)])), 
                            ClusterBucket(0, None, set([wizard.Card(wizard.Face.JESTER, wizard.Suit.CLUB), wizard.Card(wizard.Face.JESTER, wizard.Suit.DIAMOND)])), 
                            ClusterBucket(12, wizard.Suit.DIAMOND, set([wizard.Card(wizard.Face.KING, wizard.Suit.CLUB), wizard.Card(wizard.Face.QUEEN, wizard.Suit.DIAMOND)])), 
                            ClusterBucket(12, wizard.Suit.CLUB, set([wizard.Card(wizard.Face.KING, wizard.Suit.DIAMOND), wizard.Card(wizard.Face.QUEEN, wizard.Suit.CLUB)])),
                            ClusterBucket(13, wizard.Suit.DIAMOND, set([wizard.Card(wizard.Face.ACE, wizard.Suit.DIAMOND)])), 
                            ClusterBucket(13, wizard.Suit.CLUB, set([wizard.Card(wizard.Face.KING, wizard.Suit.CLUB), wizard.Card(wizard.Face.ACE, wizard.Suit.CLUB)]))]
_WIZARD_BUCKET = 0
_JESTER_BUCKET = 1

card_to_cluster = {}
for i, cluster in enumerate(clusters): 
  for card in cluster.cards:
    card_to_cluster[card] = i

#I don't like this, but there could be multiple cards from the same bucket so this is no longer a set
AbstractedHand = list[AbstractedCard]

def has_card_of_suit(hand: AbstractedHand, suit: wizard.Suit) -> bool:
    return any(map(lambda card: clusters[card.bucket].suit == suit, hand))

def is_valid_move(preceeding_card: AbstractedCard | None, card: AbstractedCard, hand: AbstractedHand) -> bool:
    return card in hand and (preceeding_card is None \
                            or clusters[card.bucket].suit is not None \
                            or clusters[card.bucket].suit == clusters[preceeding_card.bucket].suit \
                            or not has_card_of_suit(hand, clusters[preceeding_card.bucket].suit))

def get_all_valid_moves(hand: AbstractedHand, preceeding_card: AbstractedCard | None):
    return set(filter(lambda c: is_valid_move(preceeding_card, c, hand), hand))

_NUM_PLAYERS = wizard._NUM_PLAYERS
_NUM_CARDS_PER_PLAYER = wizard._NUM_CARDS_PER_PLAYER

#wizards / jesters will have club/diamond/heart/spade variety - these will have no impact
#_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product(faces, suits)))

def generate_all_possible_clustered_hands(num_cards_per_player: list[int]):
  def helper(curr_player: int, curr_hands: list[list], curr_cluster: int, remaining_per_cluster: list[int]):
    if curr_cluster == len(clusters):
      return []
    if remaining_per_cluster[curr_cluster] == 0: 
      return helper(curr_player, curr_hands, curr_cluster+1, remaining_per_cluster)
    if len(curr_hands[-1]) == num_cards_per_player[curr_player]:
      if len(num_cards_per_player) == curr_player+1:
        return [deepcopy(curr_hands)]
      else: return helper(curr_player + 1, deepcopy(curr_hands) + [[]], 0, remaining_per_cluster)
    res = []
    res.extend(helper(curr_player, curr_hands, curr_cluster+1, remaining_per_cluster))
    for i in range(1, min(num_cards_per_player[curr_player] - len(curr_hands[-1]), remaining_per_cluster[curr_cluster])+1):
      remaining_per_cluster[curr_cluster] -= i
      new_curr_hands = deepcopy(curr_hands)
      new_curr_hands[-1].extend([AbstractedCard(curr_cluster) for _ in range(i)])
      res.extend(helper(curr_player, new_curr_hands, curr_cluster+1, remaining_per_cluster))
      remaining_per_cluster[curr_cluster] += i
    return res
  return helper(0, [[]], 0, [len(cluster.cards) for cluster in clusters])


all_possible_clustered_hands = generate_all_possible_clustered_hands([_NUM_CARDS_PER_PLAYER for _ in range(_NUM_PLAYERS)])

max_chance_actions = len(all_possible_clustered_hands)
_GAME_TYPE = pyspiel.GameType(
    short_name="python_abstracted_wizard",
    long_name="Python Abstracted Wizard",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
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
    num_distinct_actions=_NUM_CARDS_PER_PLAYER+1+len(clusters), #either a bet action (_NUM_CARDS_PER_PLAYER+1) or playing a card
    max_chance_outcomes=int(max_chance_actions) + len(clusters), #TODO this overflows if it's too big
    num_players=_NUM_PLAYERS,
    min_utility=-_NUM_CARDS_PER_PLAYER*10.0,
    max_utility=20.0+10*_NUM_CARDS_PER_PLAYER,
    utility_sum=0,
    max_game_length=100)  # for 1 round, 60 cards + 6 players + 2 chance

class AbstractWizardGame(pyspiel.Game):
  """A Python version of Wizard."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return AbstractWizardState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return AbstractWizardObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
        params)

class AbstractWizardState(pyspiel.State):
  """Class representing a Wizard state.
      Concerns with this approach: there are far too many chance outcomes to be computationally feasible -
      in particular, thi 
      
      
      
      s will invoke a computation with prod_{i=0}^(num_players-1) nCr(|deck| - i*num_cards_per_player, num_cards_per_player)
      nodes, which is absolutely horrendous.
      
      There are some quirks with how this is represented; open_spiel requires that all of the actions be given unique integer indexes for chance
      and for the players, so the scheme is as follows:
      chance actions: [0-max_chance_actions) is an index that represents how the players hands should be dealt, and 
      [max_chance_actions, max_chance_actions+len(_DECK)-(cards dealt to players) represents which card was dealt from the deck for the trump card

      And, player actions: [0, _NUM_CARDS_PER_PLAYER+1) is the bet amount, and [_NUM_CARDS_PER_PLAYER+1, _NUM_CARDS_PER_PLAYER+1 + len(_DECK)) is
      which card they played
  """ 
  #this is a hack to allow subgames to have the same "game state" be in different "game states" - metadata = 0 is the normal game
  MAXIMUM_METADATA_SIZE = 16
  
  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.player_hands: list[AbstractedHand] = []
    self.initial_player_hands: list[AbstractedHand] = [] #needed to know the history of who started with what
    self.predictions = []
    self.who_started_tricks: list[int] = []
    self.previous_tricks: list[list[AbstractedCard]] = []
    self.current_round_cards: list[AbstractedCard] = []
    self.tricks_per_player = [0 for _ in range(_NUM_PLAYERS)]
    self.current_winning_card: AbstractedCard | None = None
    self.current_winning_player: int | None = None
    self.current_lead_suit: wizard.Suit | None = None
    self.trump_card: AbstractedCard | None = None
    self._next_player = 0
    self.played_unclear_cards = False #true if players played from same bucket and winner needs to be decided
   
    self.metadata = 0

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self.is_terminal():
      return pyspiel.PlayerId.TERMINAL
    elif self.is_chance_node():
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def is_chance_node(self):
    return len(self.player_hands) < _NUM_PLAYERS or (self.trump_card is None and _NUM_PLAYERS * _NUM_CARDS_PER_PLAYER < sum([len(cluster.cards) for cluster in clusters]))
    
  def _legal_actions(self, player) -> list[int]:
    """Returns a list of legal actions."""
    assert player >= 0
    #player still needs to bet
    if len(self.predictions) < _NUM_PLAYERS: 
      return [i for i in range(_NUM_CARDS_PER_PLAYER+1)]
    #otherwise, find the first card that determined a suit (if it exists) and play a suit from that
    if self.current_lead_suit is None: return sorted(set([c.bucket for c in self.player_hands[player]]))
    
    #otherwise, it's just our list of legal moves
    return sorted([c.bucket for c in get_all_valid_moves(self.player_hands[player], self.current_winning_card)])

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    if len(self.player_hands) < _NUM_PLAYERS:
      #make the player hands
      outcomes = range(len(all_possible_clustered_hands)) #generate_all_possible_hands(_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER, _DECK)
      p = 1.0 / len(outcomes)
      #because this needs to go to the C++ API, we can only pass ints, so pass an int and we'll use thathand
      return [(o, p) for o in outcomes]
      # return AllPossibleHandsList(_DECK - set().union(*self.player_hands), _NUM_CARDS_PER_PLAYER)
    else:
      # outcomes = sorted(map(lambda c: max_chance_actions + card_to_int(c), _DECK - set().union(*self.player_hands)))
      outcomes = list(range(len(clusters)))
      amt_in_cluster = [len(cluster.cards) for cluster in clusters]
      for hand in self.player_hands:
        for c in hand: amt_in_cluster[c.bucket] -= 1
      p = 1 / sum(amt_in_cluster)
      return list(filter(lambda a: a[1] > 0, [(o, p * amt_in_cluster[o]) for o in range(len(clusters))]))

  def get_all_exposed_cards(self, player_cards_to_see: set) -> list[AbstractedCard]: return set().union(*[[self.trump_card], *self.previous_tricks, self.current_round_cards, *[self.player_hands[player] for player in player_cards_to_see]])

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      #either a chance node is dealing the hands to players or specifying the trump card
      if len(self.player_hands) == _NUM_PLAYERS:
        self.trump_card = AbstractedCard(action)
      else: #otherwise, action is an index into the list of "combinations" objects, so we need to map it to a setf
        self.player_hands = deepcopy(all_possible_clustered_hands[action])
        self.initial_player_hands = deepcopy(self.player_hands)
    else:
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
        action = AbstractedCard(action)
        assert action in self.player_hands[self._next_player]
      
        self.player_hands[self._next_player].remove(action)
        strength, suit, cards = clusters[action.bucket].strength, clusters[action.bucket].suit, clusters[action.bucket].cards
        #we can know for sure that this was a valid card
        if suit is not None and self.current_lead_suit is None: self.current_lead_suit = suit
        self.current_round_cards.append(action)

        #this will only work for 2 player
        if self.current_winning_card is None or (self.current_winning_card.bucket != _WIZARD_BUCKET and action.bucket == _WIZARD_BUCKET) \
          or (self.current_winning_card.bucket == _JESTER_BUCKET and action.bucket == _JESTER_BUCKET) \
          or (suit == self.current_lead_suit and strength > clusters[self.current_winning_card.bucket].strength \
          
           or (self.trump_card is not None and self.trump_card.bucket not in [_JESTER_BUCKET, _WIZARD_BUCKET] \
            and suit == clusters[self.trump_card.bucket].suit)):
          self.current_winning_card = action
          self.current_winning_player = self._next_player
        elif clusters[self.current_winning_card.bucket] == action.bucket and clusters[self.current_winning_card.bucket].suit is not None:
          #decide the winner arbitrarily
          #TODO does this need to be in a separate chance node?
          if random.random() < 0.5:
            self.current_winning_player = 0
          else:
            self.current_winning_player = 1

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
     
  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if action >= max_chance_actions:
       return f"Dealt card for trump suit: {action}"
      #otherwise, it's hands for all the players
      return f'Dealt hands for all the players: {action}'
    else:
      if len(self.predictions) < _NUM_PLAYERS: return f'Predict {action} tricks'
      return f'Play card {action}'

  def is_terminal(self):
    """Returns True if the game is over."""
    return len(self.player_hands) == _NUM_PLAYERS and len(self.predictions) == _NUM_PLAYERS and all([len(h) == 1 for h in self.player_hands])

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self.is_terminal():
      return [0. for i in range(_NUM_PLAYERS)]
    if len(self.player_hands[0]) > 0:
      for i in range(self._next_player, self._next_player + _NUM_PLAYERS):
        self._apply_action(self.player_hands[i % _NUM_PLAYERS].__iter__().__next__().bucket)
    
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

class AbstractWizardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  """It's not too clear from the documentation what exactly a state is, but
  from what I can tell, the main thing is keeping track of a tensor and dict such that string_from
  can act as an identifer for the infoset for what the game looks like from the perspective of this player,
  and we encode this information in a tensor with perfect recall."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor.
      Currently, params lets you just specify arbitrary metadata to append to the observer

      We allow params to either be nothing, or be a dictionary with an integer "metadata", to
      store an enumerated value state (ex., useful for stepping through actions in a subgame where
      we need to be able to differentiate infosets in the gadget part vs. the subgame part). 
      If the params has metadata, we expect the state to have a metadata attribute
    """
    # Determine which observation pieces we want to include.
    pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))] #all their games 1-hot encode the player
    #i think they try to make the game state maximally 1-hot encoded for good compression and speed
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_cards", len(clusters) * max([len(cluster.cards) for cluster in clusters]), (len(clusters), max([len(cluster.cards) for cluster in clusters])))) #1-hot encode what cards are in our hand
    if iig_obs_type.public_info:
      pieces.append(("predictions", _NUM_PLAYERS*(_NUM_CARDS_PER_PLAYER+1), (_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER+1))) #one-hot encoding of prediction for player i
      #one-hot encoding of cards played by each player from the start of each trick, over all tricks (the player who started round i can be deduced
      #inductively by knowing that P0 starts round 0 and keeping track of who won round i-1)
      pieces.append(("trump_card", len(clusters), (len(clusters), )))
      pieces.append(("played_cards", len(clusters) * _NUM_CARDS_PER_PLAYER * _NUM_PLAYERS, (_NUM_CARDS_PER_PLAYER, len(clusters) * _NUM_PLAYERS))) 
      pieces.append(('metadata', AbstractWizardState.MAXIMUM_METADATA_SIZE, (AbstractWizardState.MAXIMUM_METADATA_SIZE, )))
    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state: AbstractWizardState, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1
    if "private_cards" in self.dict and player < len(state.player_hands):
      seen_per_cluster = [0 for i in range(len(clusters))]
      for card in state.player_hands[player]:
        self.dict["private_cards"][seen_per_cluster[card.bucket]][seen_per_cluster[card.bucket]] = 1
        seen_per_cluster[card.bucket] += 1
    if "predictions" in self.dict:
      for i in range(min(_NUM_PLAYERS, len(state.predictions))):
        self.dict["predictions"][i][state.predictions[i]] = 1
    if "played_cards" in self.dict:
      for i in range(len(state.previous_tricks)):
        for card in state.previous_tricks[i]:
          self.dict['played_cards'][i][card.bucket] = 1
      for card in state.current_round_cards:
        self.dict['played_cards'][len(state.previous_tricks)][card.bucket] = 1
    if 'trump_card' in self.dict and state.trump_card is not None:
      self.dict['trump_card'][state.trump_card.bucket] = 1
    if 'metadata' in self.dict:
      for i, c in enumerate(str(bin(state.metadata))[2:]):
        self.dict['metadata'][i] = c

  def string_from(self, state: AbstractWizardState, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_cards" in self.dict and len(state.player_hands) > player:
      pieces.append(f"cards in hand: {sorted(map(lambda s: str(s), state.player_hands[player]))}")
    if 'trump_card' in self.dict:
      pieces.append(f'trump card: {str(state.trump_card)}')  
    if "predictions" in self.dict:
      pieces.append(f"predictions: {state.predictions}")
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

pyspiel.register_game(_GAME_TYPE, AbstractWizardGame)

game: AbstractWizardGame = pyspiel.load_game('python_abstracted_wizard')

def map_wizard_state_to_abstracted_wizard_state(state: wizard.WizardState):
  ret=  game.new_initial_state()
  ret.player_hands = [sorted([AbstractedCard(card_to_cluster[card]) for card in hand]) for hand in state.player_hands]
  ret.initial_player_hands = [sorted([AbstractedCard(card_to_cluster[card]) for card in hand]) for hand in state.initial_player_hands] #needed to know the history of who started with what
  ret.predictions = copy(state.predictions)
  ret.who_started_tricks = copy(state.who_started_tricks)
  ret.previous_tricks = [[AbstractedCard(card_to_cluster[card]) for card in hand] for hand in state.previous_tricks]
  ret.current_round_cards = [AbstractedCard(card_to_cluster[card]) for card in state.current_round_cards]
  ret.tricks_per_player = copy(state.tricks_per_player)
  ret.current_winning_card = AbstractedCard(card_to_cluster[state.current_winning_card]) if state.current_winning_card is not None else None
  ret.current_winning_player = state.current_winning_player
  ret.current_lead_suit = state.current_lead_suit
  ret.trump_card = AbstractedCard(card_to_cluster[state.trump_card])
  ret._next_player = state._next_player
  return ret

def map_abstract_action_to_wizard_action(state: wizard.WizardState, action: int) -> int:
  if len(state.predictions) < _NUM_PLAYERS:
    return action
  else:
    #action corresponds to the bucket we're playing from - pick some card from that bucket
    #play all the ones we have with equal probabiliy
    applicable_cards = []
    for card in state.player_hands[state._next_player]:
      if card_to_cluster[card] == action:
        applicable_cards.append(wizard.card_to_action(card))
    assert len(applicable_cards) > 0
    return np.random.choice(applicable_cards)