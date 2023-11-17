from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import reduce
import itertools
from typing import Callable

import numpy as np

class Face(Enum):
    JESTER = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    WIZARD = 15

class Suit(Enum):
    CLUB = 0
    DIAMOND = 1
    HEART = 2
    SPADE = 3

face_to_str = [None, 'R', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'W']
str_to_face: Callable[[str], Face | None] = lambda t: Face(face_to_str.index(t)) if t in face_to_str else None
suit_to_str = ['C', 'D', 'H', 'S']
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
        if self.face == Face.JESTER or self.face == Face.WIZARD: return face_to_str[self.face.value]
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

#wizards / jesters will have club/diamond/heart/spade variety - these will have no impact
deck = set(map(lambda face_and_suit: Card(*face_and_suit), itertools.product(faces, suits)))
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

#fn of hand, num players, num cards per player, previous calls -> int
InputPredictionFunction = Callable[[Hand, int, int, list[int]], int]
#function of Hand, trump suit, player num, num players, num cards per player, curent cards played this round, history of previous rounds
CardPlaceFunction = Callable[[Hand, Suit, int, int, int, list[Card], list[tuple[int, list[Card]]]], Card]

def manual_input_prediction_fn(hand: Hand, num_players: int, num_cards_per_player: int, previous_calls: list[int]) -> int:
    while True:
        try:
            return int(input(f'Player {len(previous_calls)} input how many tricks you want to take '))
        except Exception as e:
            print(f'Got error {e} when inputting data, try again')

def manual_card_selection_fn(hand: Hand, trump_suit: Suit, player: int, num_players: int, num_cards_per_players: int, cards_this_trick: list[Card], round_history: list[tuple[int, list[Card]]]) -> Card:
    initial_card = cards_this_trick[0] if len(cards_this_trick) > 0 else None
    while True:
        card_to_place = str_to_card(input(f'Player {player}, what card to place? Your hand is {str(hand)} '))
        if card_to_place is not None and is_valid_move(initial_card, card_to_place, hand):
            return card_to_place
        print(f'Placing card {card_to_place} is illegal, input another card')
        card_to_place = None



def play_round(num_players: int, num_cards_per_player: int, deck: Deck, input_prediction_fns: list[InputPredictionFunction], card_place_fns: list[CardPlaceFunction]) -> list[int]:
    hands, remaining_deck = generate_hands(num_players, num_cards_per_player, deck)
    print(f'Player hands: {hands}')
    
    trump_suit = remaining_deck[0].suit if len(remaining_deck) > 0 else None

    print(f'The trump suit is {trump_suit}')

    predicted_tricks = []
    for i in range(num_players): predicted_tricks.append(input_prediction_fns[i](hands[i], num_players, num_cards_per_player, predicted_tricks))
    
    tricks_taken = [0 for _ in range(num_players)]
    #tuple of first player to play, the cards played in order
    previous_cards_placed: list[tuple[int, list[Card]]] = []
    prev_round_winner = 0
    for round_ in range(num_cards_per_player):
        initial_card = None
        winning_card, winning_player = None, None
        start_player = prev_round_winner
        cards_this_trick: list[Card] = []
        for offset_from_start in range(num_players):
            #this assumes player 0 always starts the orbit
            player = (start_player + offset_from_start) % num_players

            card_to_place = card_place_fns[player](hands[player], trump_suit, player, num_players, num_cards_per_player, cards_this_trick, previous_cards_placed)

            hands[player].remove(card_to_place)
            cards_this_trick.append(card_to_place)

            if initial_card is None:
                if winning_card is None and card_to_place.face != Face.JESTER:
                    winning_card, winning_player = card_to_place, player
                if card_to_place.face != Face.WIZARD and card_to_place.face != Face.JESTER:
                    initial_card = card_to_place
            elif winning_card is None:
                if card_to_place.face != Face.JESTER:
                    winning_card, winning_player = card_to_place, player
            elif get_cmp_from_trump_and_initial_suit(trump_suit, initial_card.suit)(card_to_place, winning_card):
                winning_card, winning_player = card_to_place, player

        #edge case where everyone plays jester - last player wins
        #? I thought it would still be first player wins?
        if winning_card is None: winning_card, winning_player = Card(Face.JESTER, Suit.DIAMOND), (round_+num_players-1)%num_players
        print(f'Player {winning_player} placed {winning_card} to take the trick')
        tricks_taken[winning_player] += 1
        previous_cards_placed.append((prev_round_winner, cards_this_trick))
        prev_round_winner = winning_player

    def calculate_score_for_player(i): return 20 + 10 * tricks_taken[i] if tricks_taken[i] == predicted_tricks[i] else -10 * abs(predicted_tricks[i] - tricks_taken[i])
    
    return list(map(calculate_score_for_player, range(num_players)))



if __name__ == '__main__':
    print(play_round(2,20,deck, [manual_input_prediction_fn, manual_input_prediction_fn], [manual_card_selection_fn, manual_card_selection_fn]))