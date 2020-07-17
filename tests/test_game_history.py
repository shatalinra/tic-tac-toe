import unittest
import game_history

class TestGameHistory(unittest.TestCase):
    def setUp(self):
        self.history = game_history.GameHistory()

    def test_init(self):
        self.assertEqual(len(self.history.moves()), 0)
        self.assertEqual(len(self.history.states()), 1)
        self.assertEqual(self.history.outcome(), -1)

    def test_equivalent_games(self):
        self.history.record_move((0, 0))
        self.history.record_move((0, 2))
        self.history.record_move((2, 0))
        self.history.record_outcome(1)

        same_games = self.history.equivalent_games()

        original_state = [1, 0, 2, 0, 0, 0, 1, 0, 0]
        rotated90_state = [1, 0, 1, 0, 0, 0, 0, 0, 2]
        rotated180_state = [0, 0, 1, 0, 0, 0, 2, 0, 1]
        rotated270_state = [2, 0, 0, 0, 0, 0, 1, 0, 1]
        horizontal_reflection_state = [2, 0, 1, 0, 0, 0, 0, 0, 1]
        vertical_reflection_state = [1, 0, 0, 0, 0, 0, 1, 0, 2]
        transposed_state = [1, 0, 1, 0, 0, 0, 2, 0, 0]

        self.assertEqual(same_games[0].states()[-1], rotated90_state)
        self.assertEqual(same_games[1].states()[-1], rotated180_state)
        self.assertEqual(same_games[2].states()[-1], rotated270_state)
        self.assertEqual(same_games[3].states()[-1], horizontal_reflection_state)
        self.assertEqual(same_games[4].states()[-1], vertical_reflection_state)
        self.assertEqual(same_games[5].states()[-1], transposed_state)


    def test_full_dense_feature(self):
        self.history.record_move((0, 0))
        self.history.record_move((0, 1))
        self.history.record_move((0, 2))
        self.history.record_move((1, 0))
        self.history.record_move((1, 1))
        self.history.record_move((1, 2))
        self.history.record_move((2, 0))
        self.history.record_move((2, 1))
        self.history.record_move((2, 2))
        self.history.record_outcome(1)
        feature = self.history.dense_feature()

        parsed_history = game_history.GameHistory()
        parsed_history.parse_dense_feature(feature)

        self.assertEqual(self.history.states(), parsed_history.states())
        self.assertEqual(self.history.moves(), parsed_history.moves())
        self.assertEqual(self.history.outcome(), parsed_history.outcome())

    def test_full_sparse_feature(self):
        self.history.record_move((0, 0))
        self.history.record_move((0, 1))
        self.history.record_move((0, 2))
        self.history.record_move((1, 0))
        self.history.record_move((1, 1))
        self.history.record_move((1, 2))
        self.history.record_move((2, 0))
        self.history.record_move((2, 1))
        self.history.record_move((2, 2))
        self.history.record_outcome(1)
        feature = self.history.sparse_feature()

        parsed_history = game_history.GameHistory()
        parsed_history.parse_sparse_feature(feature)

        self.assertEqual(self.history.states(), parsed_history.states())
        self.assertEqual(self.history.moves(), parsed_history.moves())
        self.assertEqual(self.history.outcome(), parsed_history.outcome())

    def test_padded_sparse_feature(self):
        self.history.record_move((0, 0))
        self.history.record_move((0, 1))
        self.history.record_move((0, 2))
        self.history.record_move((1, 0))
        self.history.record_move((1, 1))
        self.history.record_outcome(1)
        feature = self.history.sparse_feature()

        parsed_history = game_history.GameHistory()
        parsed_history.parse_sparse_feature(feature)

        self.assertEqual(self.history.states(), parsed_history.states())
        self.assertEqual(self.history.moves(), parsed_history.moves())
        self.assertEqual(self.history.outcome(), parsed_history.outcome())

# Executing the tests in the above test case class
if __name__ == "__main__":
    unittest.main()


