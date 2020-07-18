def print_board_state(board_state):
    state = [board_state[:3], board_state[3:6], board_state[6:]]
    symbols = [' ', 'X', 'O']
    separator = ('-------------')

    print(separator)
    for row in state:
        line = '| '
        for cell in row:
            line += symbols[cell] + ' | '
        print(line)
        print(separator)

class Board:
    def __init__(self):
        # each cell on 3x3 board can be taken by one of the players
        # 0 means empty cell
        # 1 means cell is taken by player with X
        # 2 means cell is taken by player with O
        # board always starts empty
        self._state = [[0,0,0], [0,0,0], [0,0,0]]

    def state(self):
        return self._state[0] + self._state[1] + self._state[2]

    def possible_moves(self):
        moves = []
        for i, row in enumerate(self._state):
            for j, cell in enumerate(row):
                if cell == 0:
                    moves.append((i, j))
        return moves

    def mark(self, cell_index, player):
        if self._state[cell_index[0]][cell_index[1]] != 0:
            raise AssertionError("Cell should be empty to make a move")

        self._state[cell_index[0]][cell_index[1]] = player

    def winner(self):
        # create list of all possible winning combinations
        lines = [[(0, 0), (0, 1), (0, 2)], # horizontal
                 [(1, 0), (1, 1), (1, 2)],
                 [(2, 0), (2, 1), (2, 2)],
                 [(0, 0), (1, 0), (2, 0)], # vertical
                 [(0, 1), (1, 1), (2, 1)],
                 [(0, 2), (1, 2), (2, 2)],
                 [(0, 0), (1, 1), (2, 2)], # diagonal
                 [(0, 2), (1, 1), (2, 0)]]

        # now we just check if any of these combinations taken by one of the players
        emptyCells = False
        for line in lines:
            # instead of long nested conditionals it is better to just find min-max and compare them
            for n, (i, j) in enumerate(line):
                if n == 0:
                    minimum = maximum = self._state[i][j]
                else:
                    minimum = min(minimum, self._state[i][j])
                    maximum = max(maximum, self._state[i][j])

            if minimum == 0:
                emptyCells = True
            elif minimum == maximum:
                # line is completely take by one of the players
                return minimum

        # if none of combinations is taken, it depends on existence of empty cells
        return -1 if emptyCells else 0