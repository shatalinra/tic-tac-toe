import random
import copy
import torch

class dumb_player:
    def __init__(self):
        random.seed()

    def make_move(self, board, own_mark):
        possible_moves = board.possible_moves()
        random_index = random.randint(0, len(possible_moves) - 1)
        choosen_move = possible_moves[random_index]
        board.mark(choosen_move, own_mark)

class probabilistic_player:
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


    def train(self, games):
        # simply count how much each state was involved in certain games outcome
        for game in games:
            for state in game.history():
                for equivalent_state in self._equivalent_states(state):
                    code = self._encode_board_state(equivalent_state)
                    if code not in self._scores:
                        self._scores[code] = [0, 0, 0]
                    self._scores[code][game.winner()] += 1
                        

    def make_move(self, board, own_mark):
        possible_moves = board.possible_moves()
        best_move = None
        best_score = 0
        for move in possible_moves:
            future = copy.deepcopy(board)
            future.mark(move, own_mark)
            future_code = self._encode_board_state(future.state())
            draw_score = self._scores[future_code][0]
            win_score = self._scores[future_code][own_mark]
            loss_score = self._scores[future_code][2 if own_mark == 1 else 1]
            win_move_score = win_score - loss_score
            draw_move_score = draw_score - loss_score
            if win_move_score > best_score:
                best_score = win_move_score
                best_move = move
            elif draw_move_score > best_score:
                best_score = draw_move_score
                best_move = move

        if not best_move:
            random_index = random.randint(0, len(possible_moves) - 1)
            choosen_move = possible_moves[random_index]
            board.mark(choosen_move, own_mark)
        else:
            board.mark(best_move, own_mark)

class nn_player:
    def __init__(self, model):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._model = torch.nn.Sequential(
            torch.nn.Linear(9, 200),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(200, 125),
            torch.nn.ReLU(),
            torch.nn.Linear(125, 75),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(75, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 3),
        ).to(self._device)

    def train(self, games):
        states = []
        outcomes = []
        for game in games:
            for state in game.history():
                states.append(state)
                outcomes.append(game.winner())

        # 80% of data will be used for training and 20% for testing
        training_data_size = int(len(states) * 0.8);
        training_states = torch.tensor(states[:training_data_size]).float()
        training_outcomes = torch.tensor(outcomes[:training_data_size])
        testing_states = torch.tensor(states[training_data_size:]).float().to(self._device)
        testing_outcomes = torch.tensor(outcomes[training_data_size:]).to(self._device)
        dataset = torch.utils.data.TensorDataset(training_states, training_outcomes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, pin_memory=True)

        print(self._model)
        print("Model parameters count:", sum(p.numel() for p in self._model.parameters() if p.requires_grad))

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(self._model.parameters())
        print("\nTraining on", self._device, "on", len(training_states), "board states")
        for t in range(300):
             for i, (x, y) in enumerate(dataloader, 0):
                # Forward pass: compute predicted y by passing x to the model.
                predicted_outcomes = self._model(x.to(self._device))

                loss = loss_fn(predicted_outcomes, y.to(self._device))

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                optimizer.step()

             # Verify model
             predicted_outcomes = self._model(testing_states)
             loss = loss_fn(predicted_outcomes, testing_outcomes.to(self._device))
             if t % 10 == 0:
                    print(t, loss.item())

    def make_move(self, board, own_mark):
        possible_moves = board.possible_moves()
        best_move = None
        best_score = float("-inf")
        for move in possible_moves:
            future = copy.deepcopy(board)
            future.mark(move, own_mark)
            input = torch.tensor(future.state()).float()
            prediction = self._model(input)
            prediction = torch.nn.Softmax(0)(prediction)
            draw_score = prediction[0]
            win_score = prediction[own_mark]
            loss_score = prediction[2 if own_mark == 1 else 1]
            win_move_score = win_score - loss_score
            draw_move_score = draw_score - loss_score
            if win_move_score > best_score:
                best_score = win_move_score
                best_move = move
            elif draw_move_score > best_score:
                best_score = draw_move_score
                best_move = move
         
        board.mark(best_move, own_mark)


