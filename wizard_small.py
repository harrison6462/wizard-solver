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

#* WIZARD ON SIMPLIFIED GAME: 3 suits, faces 1-5, 2w, 2j
#* 19 card deck

import enum

import numpy as np

import pyspiel

from dataclasses import dataclass
from enum import Enum
from functools import reduce
import itertools
from typing import Callable
import math

class Face(Enum):
    JESTER = 0
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    WIZARD = 6

class Suit(Enum):
    DIAMOND = 1
    HEART = 2
    SPADE = 3

face_to_str = ['R', '2', '3', '4', '5', '6', 'W']
str_to_face: Callable[[str], Face | None] = lambda t: Face(face_to_str.index(t)) if t in face_to_str else None
suit_to_str = ['D', 'H', 'S']
str_to_suit: Callable[[str], Suit | None] = lambda t: Suit(suit_to_str.index(t)) if t in suit_to_str else None

def str_to_card(t: str) -> Card | None:
    try:
        face, suit = str_to_face(t[0]), str_to_suit(t[1])
        if face is None or suit is None: return None
        return Card(face, suit)
    except:
        return None

@dataclass(frozen=True)
class Card:
    face: Face
    suit: Suit | None

    def __str__(self):
        #open_spiel requires all actions to have distinct names
        #if self.face == Face.JESTER or self.face == Face.WIZARD: return face_to_str[self.face.value]
        
        return f'{face_to_str[self.face.value]}{suit_to_str[self.suit.value]}'

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

def get_cmp_from_trump_and_initial_suit(trump: Suit | None, initial_suit: Suit | None):
    '''Returns a comparison function that is true iff c1 beats c2, where c2 is the previous winning card this hand
       with the given trump suit and initial suit (first suit played this round)
    '''
    def cmp(c1: Card, c2: Card) -> bool:
        if c2.face == Face.WIZARD: return False
        elif c1.face == c2.face == Face.JESTER: return True
        elif c1.face == Face.WIZARD: return True
        elif c2.face == Face.JESTER: return False

        return (c1.suit == trump and (c2.suit != trump or c1.face.value > c2.face.value)) \
            or (c1.suit == initial_suit and (c2.suit != trump and (c2.suit != initial_suit or c1.face.value > c2.face.value))) \
            or (c2.suit != trump and c2.suit != initial_suit and c1.face.value > c2.face.value)
    return cmp

def has_card_of_suit(hand: Hand, suit: Suit) -> bool:
    return any(map(lambda card: card.suit == suit and card.face not in [Face.WIZARD, Face.JESTER], hand))

def is_valid_move(preceeding_card: Card | None, card: Card, hand: Hand) -> bool:
    return card in hand and (preceeding_card is None \
                            or card.face in [Face.WIZARD, Face.JESTER] \
                            or card.suit == preceeding_card.suit \
                            or not has_card_of_suit(hand, preceeding_card.suit))

def get_all_valid_moves(hand: Hand, preceeding_card: Card):
    return set(filter(lambda c: is_valid_move(preceeding_card, c, hand), hand))

def generate_hands(num_players: int, num_cards_per_player: int, deck: Deck) -> tuple[list[Hand], OrderedDeck]:
    assert num_players * num_cards_per_player <= len(deck)
    permutation = np.random.permutation(list(deck))
    return [set(permutation[num_cards_per_player * i: num_cards_per_player * (i+1)]) for i in range(num_players)], permutation[num_cards_per_player * num_players:]

def generate_all_possible_hands(num_players: int, num_cards_per_player: int, deck: Deck):
  from itertools import combinations
  assert num_players * num_cards_per_player <= len(deck)
  all_choices = map(lambda s: [list(s)], combinations(deck, num_cards_per_player))
  for _ in range(num_players-1):
    new_all_choices = []
    for choice in all_choices:
      all_cards_in_choice = set().union(*map(lambda s: set(s), choice))
      deck_without_choices = _DECK.difference(all_cards_in_choice)
      new_all_choices.extend([choice + [list(combo)] for combo in combinations(deck_without_choices,num_cards_per_player)])
    all_choices = new_all_choices
  return all_choices

#the possible types of actions
BetAction = int
CardAction = Card

_NUM_PLAYERS = 2
_NUM_CARDS_PER_PLAYER = 3
#wizards / jesters will have club/diamond/heart/spade variety - these will have no impact
#_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product(faces, suits)))
_DECK = frozenset(map(lambda face_and_suit: Card(*face_and_suit), itertools.product([Face.JESTER, Face.KING, Face.ACE, Face.WIZARD], [Suit.CLUB, Suit.DIAMOND])))
#TODO this is an important function, perhaps more attention should be drawn to it
# def card_to_int(card: Card) -> int:
#   '''Gives a numbering of the cards that is bijective in [0, ... len(_DECK)). The order is arbitrary
#   '''
#   return card.face.value * 4 + card.suit.value

# TODO come back and figure out what these functions are
'''
def card_to_int(card: Card) -> int:
  # Use this to play with a deck of only jesters, kings-wizards with suits D and S

  if card.face == Face.JESTER: return card.suit.value
  return (card.face.value * 2 + card.suit.value) - (Face.KING.value * 2) + 2

def int_to_card(i: int) -> Card:
  if i < 2: return Card(Face.JESTER, suits[i])
  return Card(faces[((i-2)//2+Face.KING.value)], suits[i%2])
'''

max_chance_actions = 1
for i in range(_NUM_PLAYERS): max_chance_actions *= math.comb(len(_DECK)-i*_NUM_CARDS_PER_PLAYER, _NUM_CARDS_PER_PLAYER)
_GAME_TYPE = pyspiel.GameType(
    short_name="python_wizard",
    long_name="Python Wizard",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
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
    utility_sum=None,
    max_game_length= (_NUM_CARDS_PER_PLAYER+1) * _NUM_PLAYERS + 2) # for 1 round, (num_cards_per_player) tricks and (num_players) cards per trick. Then two chance actions to deal cards[wait is the initial deal one chance action or two?] and (num_players) bets
    # for 1 round, 60 cards + 6 players + 2 chance
    # ??? bro what ^?

class WizardGame(pyspiel.Game):
  """A Python version of Wizard."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return WizardState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return WizardObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), #TODO this was False before...
        params)


class WizardState(pyspiel.State):
  """Class representing a Wizard state.
      Concerns with this approach: there are far too many chance outcomes to be computationally feasible -
      in particular, this will invoke a computation with prod_{i=0}^(num_players-1) nCr(|deck| - i*num_cards_per_player, num_cards_per_player)
      nodes, which is absolutely horrendous.

      #* Ok true but if we start with suit isomorphisms, we already decrease by a factor of 4*3! = 24 right? since 4 suits could be trump and the other three could be in any of 6 orderings? -cynthia
      #* Like that alone should be enough to make the initial combo generation feasible, then we run abstraction algo and try to solve simplified game?
      
      There are some quirks with how this is represented; open_spiel requires that all of the actions be given unique integer indexes for chance
      and for the players, so the scheme is as follows:
      chance actions: [0-max_chance_actions) is an index that represents how the players hands should be dealt, and 
      [max_chance_actions, max_chance_actions+len(_DECK)-(cards dealt to players) represents which card was dealt from the deck for the trump card
      #* each i in range(max_chance_actions) corresponds to one possible distribution of cards to all players
      #* each i in [max_chance_actions, max_chance_actions+len(_DECK)-(cards dealt to players)) represents one possible card drawn from remaining deck 
      # ! kinda sus because different cards will be available depending on the initial chance deal? Or does it represent the 'position' of the card drawn? but then the deck would need to be a list and i think it's a set rn?

      And, player actions: [0, _NUM_CARDS_PER_PLAYER+1) is the bet amount, and [_NUM_CARDS_PER_PLAYER+1, _NUM_CARDS_PER_PLAYER+1 + len(_DECK)) is
      which card they played
  """

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.player_hands: list[Hand] = []
    self.predictions = []
    self.previous_tricks: list[list[Card]] = []
    self.current_round_cards: list[Card] = []
    self.tricks_per_player = [0 for _ in range(_NUM_PLAYERS)]
    self.current_winning_card: Card | None = None
    self.current_winning_player: int | None = None
    self.current_lead_suit: Suit | None = None
    self.trump_card = None
    self._game_over = False
    self._next_player = 0

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self.is_chance_node():
      return pyspiel.PlayerId.CHANCE
    else:
      return self._next_player

  def is_chance_node(self):
    return len(self.player_hands) == 0 or (len(self.predictions) == _NUM_PLAYERS and self.trump_card is None and _NUM_PLAYERS * _NUM_CARDS_PER_PLAYER < len(_DECK))
    
  def _legal_actions(self, player) -> list[BetAction | CardAction]:
    """Returns a list of legal actions."""
    assert player >= 0
    #player still needs to bet
    if len(self.predictions) < _NUM_PLAYERS: 
      return [i for i in range(_NUM_CARDS_PER_PLAYER+1)]
    #otherwise, find the first card that determined a suit (if it exists) and play a suit from that
    if self.current_lead_suit is None: return sorted(map(lambda c: _NUM_CARDS_PER_PLAYER+1+card_to_int(c), self.player_hands[player]))
    
    #otherwise, it's just our list of legal moves
    return sorted(map(lambda c: card_to_int(c)+_NUM_CARDS_PER_PLAYER+1, get_all_valid_moves(self.player_hands[player], Card(Face.TWO, self.current_lead_suit))))

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    if len(self.player_hands) == 0:
      #make the player hands
      outcomes = range(max_chance_actions) #generate_all_possible_hands(_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER, _DECK)
    else:
      outcomes = sorted(map(lambda c: max_chance_actions + card_to_int(c), _DECK - set().union(*self.player_hands)))
    p = 1.0 / len(outcomes)
    #because this needs to go to the C++ API, we can only pass ints, so pass an int and we'll use thathand
    return [(o, p) for o in outcomes]

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():
      #either a chance node is dealing the hands to players or specifying the trump card
      if action >= max_chance_actions:
        self.trump_card = int_to_card(action - max_chance_actions)
      else: #otherwise, action is an index into the list of "combinations" objects, so we need to map it to a setf
        all_possible_hands = generate_all_possible_hands(_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER, _DECK) #TODO this could be made wayyyyyyyyyyy more efficient by knowing how to correspond an int to a combo
        self.player_hands = list(map(lambda s: set(s), all_possible_hands[action]))
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
      else:
        action = int_to_card(action-(_NUM_CARDS_PER_PLAYER+1))
        assert action in self.player_hands[self._next_player]
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
        else: self._next_player = (self._next_player + 1) % _NUM_PLAYERS
        #if none of the players have cards left, the game is over
        if all(map(lambda hand: len(hand) == 0, self.player_hands)):
          self._game_over = True

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if action >= max_chance_actions:
       return f"Dealt card for trump suit: {int_to_card(action-max_chance_actions)}"
      #otherwise, it's hands for all the players
      return f'Dealt hands for all the players: {action}'
    else:
      if action <= _NUM_PLAYERS: return f'Player {player} predicted {action}'
      return f'Player {player} played card {int_to_card(action - _NUM_CARDS_PER_PLAYER -1)}'

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    if not self._game_over:
      return [0. for i in range(_NUM_PLAYERS)]
    def reward_for_player(i: int):
      if self.tricks_per_player[i] == self.predictions[i]: return 20 + 10 * self.tricks_per_player[i]
      return -10 * abs(self.tricks_per_player[i] - self.predictions[i]) 
    return [reward_for_player(i) for i in range(_NUM_PLAYERS)]

  def __str__(self):
    return f'Player acting: {self._next_player}\n Predictions: {self.predictions}\n' \
    + f'Trump suit: {self.trump_card.suit if self.trump_card else None} lead suit: {self.current_lead_suit}' \
    + f'Current round: {self.current_round_cards}' + f'Hands: {list(map(lambda h: list(h), self.player_hands))} \n' \
    + f'History: {list(map(lambda t: str(t), self.previous_tricks))}'

class WizardObserver:
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
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_cards", len(_DECK), (len(_DECK),))) #1-hot encode what cards are in our hand
    if iig_obs_type.public_info:
      pieces.append(("predictions", _NUM_PLAYERS*(_NUM_CARDS_PER_PLAYER+1), (_NUM_PLAYERS, _NUM_CARDS_PER_PLAYER+1))) #one-hot encoding of prediction for player i
      #one-hot encoding of cards played by each player from the start of each trick, over all tricks (the player who started round i can be deduced
      #inductively by knowing that P0 starts round 0 and keeping track of who won round i-1)
      pieces.append(("played_cards", len(_DECK) * _NUM_CARDS_PER_PLAYER * _NUM_PLAYERS, (_NUM_CARDS_PER_PLAYER, len(_DECK) * _NUM_PLAYERS))) 
        
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
    if "private_cards" in self.dict and player < len(state.player_hands):
      for card in state.player_hands[player]:
        self.dict["private_cards"][card_to_int(card)] = 1
    if "predictions" in self.dict:
      for i in range(min(_NUM_PLAYERS, len(state.predictions))):
        self.dict["predictions"][i][state.predictions[i]] = 1
    if "played_cards" in self.dict:
      for i in range(len(state.previous_tricks)):
        for card in state.previous_tricks[i]:
          self.dict['played_cards'][i][card_to_int(card)] = 1
      for card in state.current_round_cards:
        self.dict['played_cards'][len(state.previous_tricks)][card_to_int(card)] = 1
    
  def string_from(self, state: WizardState, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_cards" in self.dict and len(state.player_hands) > player:
      pieces.append(f"cards in hand: {str(list(map(lambda s: str(s), state.player_hands[player])))}")
    if "predictions" in self.dict:
      pieces.append(f"predictions[{state.predictions}]")
    if "played_cards" in self.dict and len(state.current_round_cards) > 0 or len(state.previous_tricks) > 0:
      for i in range(len(state.previous_tricks)):
        pieces.append('tr: '.join(str(card) for card in state.previous_tricks[i]))
      pieces.append("curr: ".join(str(card) for card in state.current_round_cards))
    return " ".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, WizardGame)