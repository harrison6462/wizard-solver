from wizard import Card, Face, Suit, str_to_card, get_cmp_from_trump_and_initial_suit, get_all_valid_moves, is_valid_move
import unittest

class TestCardMethods(unittest.TestCase):

    def test_str_to_card(self):
        self.assertEqual(str_to_card('TH'), Card(Face.TEN, Suit.HEART))
        self.assertEqual(str_to_card('RS'), Card(Face.JESTER, Suit.SPADE))
        self.assertEqual(str_to_card('WD'), Card(Face.WIZARD, Suit.DIAMOND))
    
    def test_cmp_from_trump_and_initial_suit(self):
        cmp = get_cmp_from_trump_and_initial_suit(None, None)
        self.assertTrue(cmp(Card(Face.WIZARD, Suit.DIAMOND), Card(Face.ACE, Suit.DIAMOND)))
        self.assertTrue(cmp(Card(Face.ACE, Suit.DIAMOND), Card(Face.KING, Suit.DIAMOND)))
        #we want WIZARD > WIZARD false
        self.assertFalse(cmp(Card(Face.WIZARD, Suit.DIAMOND), Card(Face.WIZARD, Suit.SPADE)))
        #we want JESTER > JESTER to be true, as it's true temporally
        self.assertTrue(cmp(Card(Face.JESTER, Suit.DIAMOND), Card(Face.JESTER, Suit.SPADE)))
        cmp = get_cmp_from_trump_and_initial_suit(Suit.SPADE, None)
        self.assertTrue(cmp(Card(Face.WIZARD, Suit.DIAMOND), Card(Face.ACE, Suit.SPADE)))
        self.assertFalse(cmp(Card(Face.ACE, Suit.DIAMOND), Card(Face.TWO, Suit.SPADE)))        
        cmp = get_cmp_from_trump_and_initial_suit(Suit.SPADE, Suit.DIAMOND)
        self.assertFalse(cmp(Card(Face.ACE, Suit.DIAMOND), Card(Face.TWO, Suit.SPADE)))
        self.assertFalse(cmp(Card(Face.ACE, Suit.HEART), Card(Face.TWO, Suit.DIAMOND)))       
        self.assertTrue(cmp(Card(Face.TWO, Suit.SPADE), Card(Face.THREE, Suit.DIAMOND)))

    def test_get_all_valid_moves(self):
        hand = set([Card(Face.ACE, Suit.HEART), Card(Face.ACE, Suit.DIAMOND), Card(Face.WIZARD, Suit.DIAMOND)])
        self.assertSetEqual(get_all_valid_moves(hand, Card(Face.KING, Suit.CLUB)), hand)
        self.assertSetEqual(get_all_valid_moves(hand, Card(Face.KING, Suit.HEART)), set([Card(Face.ACE, Suit.HEART), Card(Face.WIZARD, Suit.DIAMOND)]))

if __name__ == '__main__':
    unittest.main()