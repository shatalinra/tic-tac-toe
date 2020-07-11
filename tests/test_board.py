import unittest
import board

class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = board.Board()

    def test_init(self):
        self.assertEqual(len(self.board.state()), 9)
        for cell in self.board.state():
            self.assertEqual(cell, 0)
        self.assertEqual(self.board.winner(), -1)
        self.assertEqual(len(self.board.possible_moves()), 9)

    def test_ongoing_game(self):
        self.board.mark((0, 0), 1)
        self.board.mark((0, 1), 2)
        self.board.mark((1, 0), 1)
        self.assertEqual(self.board.state(), [1, 2, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(self.board.winner(), -1)
        self.assertEqual(len(self.board.possible_moves()), 6)
        for move in self.board.possible_moves():
            self.assertNotEqual(move, (0, 0))
            self.assertNotEqual(move, (0, 1))
            self.assertNotEqual(move, (1, 0))

    def test_cross_won(self):
        self.board.mark((0, 0), 1)
        self.board.mark((0, 1), 2)
        self.board.mark((1, 0), 1)
        self.board.mark((0, 2), 2)
        self.board.mark((2, 0), 1)
        self.assertEqual(self.board.state(), [1, 2, 2, 1, 0, 0, 1, 0, 0])
        self.assertEqual(self.board.winner(), 1)
        self.assertEqual(len(self.board.possible_moves()), 4)
        for move in self.board.possible_moves():
            self.assertNotEqual(move, (0, 0))
            self.assertNotEqual(move, (0, 1))
            self.assertNotEqual(move, (1, 0))
            self.assertNotEqual(move, (0, 2))
            self.assertNotEqual(move, (2, 0))

    def test_nought_won(self):
        self.board.mark((0, 0), 1)
        self.board.mark((0, 1), 2)
        self.board.mark((1, 0), 1)
        self.board.mark((1, 1), 2)
        self.board.mark((0, 2), 1)
        self.board.mark((2, 1), 2)
        self.assertEqual(self.board.state(), [1, 2, 1, 1, 2, 0, 0, 2, 0])
        self.assertEqual(self.board.winner(), 2)
        self.assertEqual(len(self.board.possible_moves()), 3)
        for move in self.board.possible_moves():
            self.assertNotEqual(move, (0, 0))
            self.assertNotEqual(move, (0, 1))
            self.assertNotEqual(move, (1, 0))
            self.assertNotEqual(move, (1, 1))
            self.assertNotEqual(move, (0, 2))
            self.assertNotEqual(move, (2, 1))

    def test_draw(self):
        self.board.mark((0, 0), 1)
        self.board.mark((0, 2), 2)
        self.board.mark((0, 1), 1)
        self.board.mark((1, 0), 2)
        self.board.mark((1, 2), 1)
        self.board.mark((1, 1), 2)
        self.board.mark((2, 0), 1)
        self.board.mark((2, 2), 2)
        self.board.mark((2, 1), 1)

        self.assertEqual(self.board.state(), [1, 1, 2, 2, 2, 1, 1, 1, 2])
        self.assertEqual(self.board.winner(), 0)
        self.assertEqual(len(self.board.possible_moves()), 0)

# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()
