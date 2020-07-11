import random

class Player:
    def __init__(self):
        random.seed()

    def choose_move(self, possible_moves, own_mark, history):
        random_index = random.randint(0, len(possible_moves) - 1)
        return possible_moves[random_index]

