import board
import random

class InvalidLength(Exception):
    pass

class GameRulesViolation(Exception):
    pass

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

    # dense feature which makes it impossible for network to break the rules
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

    # sparse feature should be more friendly for neural network but not all possible values are valid under game rules
    # in order to create fixed length feature required numbers of empty boards states would be added at start of game
    def sparse_feature(self):
        feature = [self.outcome(), self.length()]
        states = self.states()
        while len(states) < 10:
            states.insert(0, [0] * 9)

        # transform each state in binary arrays of empty spaces, crosses and noughts to inflate input space
        for state in states:
            feature += [1 if cell == 0 else 0 for cell in state]
            feature += [1 if cell == 1 else 0 for cell in state]
            feature += [1 if cell == 2 else 0 for cell in state]

        # feature normalization
        feature[0] = float(feature[0]) / 2
        feature[1] = float(feature[1]) / 9
        return feature

    def parse_sparse_feature(self, feature):
        # sparse feature can be inconsisted in many ways, start by binarizing its values
        self._outcome = 0 if feature[0] < 0.25 else 1 if feature[0] < 0.75 else 2
        length = round(feature[1] * 9)

        # now fill state by winner-takes-all approach
        states = []
        for i in range(0, 10):
            state = [0] * 9
            for j in range(0, 9):
                empty_score = feature[2 + i * 3 * 9 + j]
                cross_score = feature[2 + i * 3 * 9 + 9 + j]
                nought_score = feature[2 + i * 3 * 9 + 2*9 + j]
                score_list = (empty_score, cross_score, nought_score)
                max_score = max(score_list)
                state[j] = score_list.index(max_score)
            states.append(state)

        # steps before were lenient on sparse feature values but now we need to boil it down to moves
        # first check how much of empty boards at start of the game
        empty_states = 0
        for state in states:
            if sum(state) > 0: break
            empty_states += 1
        if empty_states + length != 10:
            raise InvalidLength("Empty states should pad sparse feature to a fixed size")

        # just skip empty states, parse actual moves but ignore impossible ones
        self._moves = []
        simulation_board = board.Board()
        current_player = 1
        states = states[empty_states:]
        for state in states:
            moves_intersection = []
            for move in simulation_board.possible_moves():
                index = move[0] * 3 + move[1]
                if state[index] == current_player:
                    moves_intersection.append(move)

            if len(moves_intersection) != 1:
                raise GameRulesViolation("Invalid number of moves on single turn")

            selected_move = moves_intersection[0]
            self._moves.append(selected_move)
            simulation_board.mark(selected_move, current_player)
            current_player = 1 if current_player == 2 else 2

 