import random
import pickle
import copy

class Player:
    def __init__(self):
        self._scores = {}

    def _encode_board_state(self, state):
        # each cell on board can be in three states and board have 9 cells
        # so total number of possible board states is 3^9 = 19683
        # it would be a lot easier if we just encode each state as binary number
        code = 0
        for i,cell in enumerate(state):
            if cell != 0:
                triplet = (0b10 if cell == 1 else 0b11) << (2 * i)
                code |= triplet
        return code

    def _equivalent_states(self, state):
        # due to rotation and reflection invariance there is 7 equivalent states for each state
        # to derive them it is easir to just draw board with index on paper and rotate it
        rotated90 = [state[2], state[5], state[8], state[1], state[4], state[7], state[0], state[3], state[6]]
        rotated180 = [state[8], state[7], state[6], state[5], state[4], state[3], state[2], state[1], state[0]]
        rotated270 = [state[6], state[3], state[0], state[7], state[4], state[1], state[8], state[5], state[2]]
        flipped_horizontal = [state[2], state[1], state[0], state[5], state[4], state[3], state[8], state[7], state[6]]
        flipped_vertical = [state[6], state[7], state[8], state[3], state[4], state[5], state[0], state[1], state[2]]
        flipped_diagonal = [state[0], state[3], state[6], state[1], state[4], state[7], state[2], state[5], state[8]]
        return [state, rotated90, rotated180, rotated270, flipped_horizontal, flipped_vertical, flipped_diagonal]

    def load(self, path):
        with open(path, "rb") as in_file:
            self._scores = pickle.load(in_file)

    def train(self, games, save_path):
        # simply count how much each state was involved in certain games outcome
        for game in games:
            for state in game.states():
                for equivalent_state in self._equivalent_states(state):
                    code = self._encode_board_state(equivalent_state)
                    if code not in self._scores:
                        self._scores[code] = [0, 0, 0]
                    self._scores[code][game.outcome()] += 1

        with open(save_path, "wb") as outfile:
            pickle.dump(self._scores, outfile)
                        

    def choose_move(self, possible_moves, own_mark, history):
        best_move = None
        best_score = 0
        for move in possible_moves:
            future = history.states()[-1]
            future[move[0] * 3 + move[1]] = own_mark
            future_code = self._encode_board_state(future)
            if future_code not in self._scores: continue
            draw_score = self._scores[future_code][0]
            win_score = self._scores[future_code][own_mark]
            loss_score = self._scores[future_code][2 if own_mark == 1 else 1]
            win_move_score = win_score - loss_score
            draw_move_score = draw_score - loss_score
            if win_move_score > best_score:
                best_score = win_move_score
                best_move = move
            if draw_move_score > best_score:
                best_score = draw_move_score
                best_move = move

        if not best_move:
            random_index = random.randint(0, len(possible_moves) - 1)
            return possible_moves[random_index]
        else:
            return best_move
