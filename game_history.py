import board
import random

# it is hard to create good representation for a variable-length game with alternating moves and no cell overtaking
# generally it is better to use binary arrays but neural network would have to learn complicated game rules
# this class allows to abstract game history in various ways needed for algorithms
class GameHistory:
    def __init__(self):
        self._moves = []
        self._outcome = -1

    def record_move(self, move):
        self._moves.append(move)

    def record_outcome(self, outcome):
        self._outcome = outcome

    def moves(self):
        return self._moves

    def length(self):
        return len(self._moves)

    def outcome(self):
        return self._outcome

    def states(self):
        simulation_board = board.Board()
        game_recap = [simulation_board.state()]
        for i, move in enumerate(self._moves):
            simulation_board.mark(move, 1 + i % 2)
            game_recap.append(simulation_board.state())
        return game_recap

    # dense feature which makes it hard for network to break the rules
    # feature would have fixed size of 10 pretending game always contains 9 turns
    # if cell was not taken it would be treated as taken on some random turn
    # it serves as source of entropy for data imputation by autoencoders
    def dense_feature(self):
        feature = [self._outcome] + [None] * 9
        for i, move in enumerate(self._moves):
            feature[1 + move[0] * 3 + move[1]] = i

        empty_cells_index = [i for i in range(1, 10) if feature[i] == None]
        random.shuffle(empty_cells_index)
        last_turn = len(self._moves)
        for index in empty_cells_index:
            feature[index] = last_turn
            last_turn += 1

        # feature normalization
        feature[0] = feature[0] - 1
        for i in range(9):
            feature[i + 1] = feature[i + 1] / 4 - 1

        return feature

    def parse_dense_feature(self, feature):
         # we would not require output to be exact integers but instead just follow increasing order
        moves = []
        values = feature[1:]
        indexes = sorted(range(len(values)), key=lambda k: values[k])
        self._moves = [(int(index / 3), index % 3) for index in indexes]

        # binarize outcome, no need to require exact integer
        self._outcome = 0 if feature[0] < -0.5 else 1 if feature[0] < 0.5 else 2
