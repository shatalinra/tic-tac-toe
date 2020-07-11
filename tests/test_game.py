import unittest
import game
from unittest.mock import Mock

class TestGame(unittest.TestCase):
    def setUp(self):
        self.player1 = Mock()
        self.player2 = Mock()
        self.game = game.Game(self.player1, self.player2)

    def test_init(self):
        self.assertEqual(self.game.length(), 0)
        self.assertEqual(self.game.winner(), -1)
        self.assertEqual(len(self.game.history().moves()), 0)
        self.assertEqual(self.game.history().outcome(), -1)

    def test_cross_won(self):
        self.player1.choose_move.side_effect = [(0, 0), (1, 0), (2, 0)]
        self.player2.choose_move.side_effect = [(0, 1), (0, 2)]

        self.game.run()
        self.assertEqual(self.game.length(), 5)
        self.assertEqual(self.game.winner(), 1)
        self.assertEqual(len(self.game.history().moves()), 5)
        self.assertEqual(self.game.history().outcome(), 1)

    def test_nought_won(self):
        self.player1.choose_move.side_effect = [(0, 0), (1, 0), (0, 2)]
        self.player2.choose_move.side_effect = [(0, 1), (1, 1), (2, 1)]

        self.game.run()
        self.assertEqual(self.game.length(), 6)
        self.assertEqual(self.game.winner(), 2)
        self.assertEqual(len(self.game.history().moves()), 6)
        self.assertEqual(self.game.history().outcome(), 2)

    def test_draw(self):
        self.player1.choose_move.side_effect = [(0, 0), (0, 1), (1, 2), (2, 0), (2, 1)]
        self.player2.choose_move.side_effect = [(0, 2), (1, 0), (1, 1), (2, 2)]

        self.game.run()
        self.assertEqual(self.game.length(), 9)
        self.assertEqual(self.game.winner(), 0)
        self.assertEqual(len(self.game.history().moves()), 9)
        self.assertEqual(self.game.history().outcome(), 0)

# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()

