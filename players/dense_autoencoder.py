import random
import copy
import torch
import json
import game_history

class Player:
    def __init__(self):
        self._model = None
        self._cell_count = 9
        self._max_game_length = 9
        self.reset_metrics()

    def reset_metrics(self):
        self._random_moves = 0
        self._all_moves = 0

    def random_moves_ratio(self):
        return self._random_moves / self._all_moves

    def load(self, path):
        model = torch.load(path)
        model.eval()
        self._model = model

    def train(self, games, save_path):
        # tried ReLU but model is shallow, so tanh adds more non-linearity and decreases loss
        # tried mini batching but it actually increases loss without finding new minimum
        # tried adding new layer for better generalization but it increases loss
        # seems that dense feature just does not work

        training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.Sequential(
                torch.nn.Linear(1 + 9, 9),
                torch.nn.Tanh(),
                torch.nn.Linear(9, 8),
                torch.nn.Tanh(),
                torch.nn.Linear(8, 7),
                torch.nn.Tanh(),
                torch.nn.Linear(7, 8),
                torch.nn.Tanh(),
                torch.nn.Linear(8, 9),
                torch.nn.Tanh(),
                torch.nn.Linear(9, 1 + 9)
        ).to(training_device)
        data = [game.dense_feature() for game in games]

        # 80% of data will be used for training and 20% for testing
        training_data_size = int(len(data) * 0.8);
        training_data = torch.tensor(data[:training_data_size]).float().to(training_device)
        testing_data = torch.tensor(data[training_data_size:]).float().to(training_device)

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters())
        print("\nTraining using", training_device, "on", training_data_size, "games")
        for t in range(10000):
            output_games = model(training_data)

            loss = loss_fn(output_games, training_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Verify model
            output_games = model(testing_data)
            loss = loss_fn(output_games, testing_data)
            if t % 100 == 0:
                print(t, loss.item())

        # output how much games are actually reconstructed right
        move_errors = 0
        outcome_errors = 0
        valid_reconstruction = 0
        all_data = torch.tensor(data).float().to(training_device)
        output_games = model(all_data).cpu()
        for i, game in enumerate(games):
            reconstructed_game = game_history.GameHistory()
            reconstructed_game.parse_dense_feature(output_games[i])
            outcome_mismatch = reconstructed_game.outcome() != game.outcome()
            move_mismatch = reconstructed_game.moves()[:game.length()] != game.moves()

            if outcome_mismatch:
                outcome_errors += 1
            if move_mismatch:
                move_errors += 1
            if not (outcome_mismatch or move_mismatch):
                valid_reconstruction += 1

        print("Testing reconstruction")
        print("Outcome errors:", 100*outcome_errors /len(all_data), "%")
        print("Move errors:", 100*move_errors /len(all_data), "%")
        print("Valid reconstruction:", 100*valid_reconstruction /len(all_data), "%\n")

        self._model = model.cpu()
        torch.save(self._model, save_path)

    def choose_move(self, possible_moves, own_mark, history):
        current_turn = history.length()

        # we would use model to impute future provided desired outcome, game length and immediate history
        self._all_moves += 1
        incomplete_game = copy.deepcopy(history)
        incomplete_game.record_outcome(1)
        input = torch.tensor(incomplete_game.dense_feature()).float()
        output = self._model(input)
        reconstructed_game = game_history.GameHistory()
        reconstructed_game.parse_dense_feature(output.tolist())

        # there is no guarantee that reconstructed moves are actually possible
        choosen_move = reconstructed_game.moves()[current_turn]
        if choosen_move in possible_moves:
            return choosen_move
        
        # if no winning move was found let us move to forcing a draw
        incomplete_game = copy.deepcopy(history)
        incomplete_game.record_outcome(0)
        input = torch.tensor(incomplete_game.dense_feature()).float()
        output = self._model(input)
        reconstructed_game = game_history.GameHistory()
        reconstructed_game.parse_dense_feature(output.tolist())

        # there is no guarantee that reconstructed moves are actually possible
        choosen_move = reconstructed_game.moves()[current_turn]
        if choosen_move in possible_moves:
            return choosen_move

        # at this point network failed to provide possible move, just use random
        self._random_moves += 1
        random_index = random.randint(0, len(possible_moves) - 1)
        return possible_moves[random_index]