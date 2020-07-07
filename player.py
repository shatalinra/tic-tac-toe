import random

class dumb_player:
    def __init__(self):
        random.seed()

    def makeMove(self, board, own_mark):
        possible_moves = board.possible_moves()
        random_index = random.randint(0, len(possible_moves) - 1)
        choosen_move = possible_moves[random_index]
        board.mark(choosen_move, own_mark)
