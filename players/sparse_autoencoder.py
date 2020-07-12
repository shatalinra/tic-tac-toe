
import random
import copy
import torch
import json
import game_history

class SparseAutoencoder(torch.nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()

        # there are 8 meaningul line combinations, output would be around 10*3*8 = 240
        self.encoder_conv = torch.nn.Conv1d(1, 8, 9, 9)
        #self.encoder_conv_activation = torch.nn.Tanh()
        #$self.encoder_linear1 = torch.nn.Linear(242, 212)
        #self.encoder_linear1_activation = torch.nn.Tanh()
        #self.encoder_linear2 = torch.nn.Linear(128, 64)
        #self.encoder_linear3 = torch.nn.Linear(64, 32)
        #self.encoder_linear4 = torch.nn.Linear(32, 16)
        #self.encoder_linear5 = torch.nn.Linear(16, 5)
        #self.decoder_linear1 = torch.nn.Linear(5, 16)
        #self.decoder_linear2 = torch.nn.Linear(16, 32)
        #self.decoder_linear3 = torch.nn.Linear(32, 64)
        #self.decoder_linear4 = torch.nn.Linear(64, 128)
        #self.decoder_linear5 = torch.nn.Linear(212, 242)
        #self.decoder_linear5_activation = torch.nn.Tanh()

        # Torch documentation don't describe how channel reduction performed at all
        # Google does not helped also and source code is quite scary
        # using fixed values input and analyzing output I figured out how it works:
        # 1. each input channel undergoes transposed convolution which is described on tons of sites and books
        # 2. results for each channels are summed together including bias and pushed to output
        # 3. if you have multiple output channels they just use different set of kernels and produce different output values
        self.decoder_conv = torch.nn.ConvTranspose1d(8, 1, 9, 9)
        # output should be binary, so using sigmoid would help
        self.decover_conv_activation = torch.nn.Sigmoid()

        # input feature consists from 2 metadata values and 10*3*9=270 state values
        self._metadata_index = torch.tensor(range(0, 2))
        self._states_index = torch.tensor(range(2, 272))
        self._state_codes_index = torch.tensor(range(2, 242))

    def _apply(self, fn):
        super(SparseAutoencoder, self)._apply(fn)
        self._metadata_index = fn(self._metadata_index)
        self._states_index = fn(self._states_index)
        self._state_codes_index = fn(self._state_codes_index)
        return self
        
    def forward(self, data):
        data_view = data.view(1,-1) if data.dim() == 1 else data
        metadata = data_view.index_select(1, self._metadata_index)
        states = data_view.index_select(1, self._states_index)

        view = states.view(-1,1,270)
        state_features = self.encoder_conv(view)
        #state_features = self.encoder_conv_activation(state_features)
        view = state_features.view(-1,240)
        
        x = torch.cat((metadata, view), 1)
        #x = self.encoder_linear1(x)
        #x = self.encoder_linear1_activation(x)
        #x = self.encoder_linear2(x).clamp(min=0)
        #x = self.encoder_linear3(x).clamp(min=0)
        #x = self.encoder_linear4(x).clamp(min=0)
        #x = self.encoder_linear5(x).clamp(min=0)
        #x = self.decoder_linear1(x).clamp(min=0)
        #x = self.decoder_linear2(x).clamp(min=0)
        #x = self.decoder_linear3(x).clamp(min=0)
        #x = self.decoder_linear4(x).clamp(min=0)
        #x = self.decoder_linear5(x)
        #x = self.decoder_linear5_activation(x)

        out_metadata = x.index_select(1, self._metadata_index)
        state_codes = x.index_select(1, self._state_codes_index)

        view = state_codes.view(-1,8,30)
        out_states = self.decoder_conv(view)
        out_states = self.decover_conv_activation(out_states)
        view = out_states.view(-1,270)

        return torch.cat((out_metadata, view), 1)

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

        training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SparseAutoencoder().to(training_device)
        data = [game.sparse_feature() for game in games]

        # 80% of data will be used for training and 20% for testing
        training_data_size = int(len(data) * 0.8);
        training_data = torch.tensor(data[:training_data_size]).float().to(training_device)
        testing_data = torch.tensor(data[training_data_size:]).float().to(training_device)
        batch_size = training_data_size
        #batch_size = 200

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters())
        print("\nTraining using", training_device, "on", training_data_size, "games")
        for t in range(10000):

            # shuffle data between epochs
            if batch_size != training_data_size:
                permutation = torch.randperm(training_data.size()[0]).to(training_device)
                training_data = training_data[permutation]

            for base_index in range(0, training_data_size, batch_size):
                # Forward pass: compute predicted y by passing x to the model.
                batch = training_data[base_index:base_index + batch_size]
                output_games = model(batch)

                #loss = loss_fn(output_games, gpu_data)
                loss = loss_fn(output_games, batch)

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

            # Check model perfomance
            if t % 100 == 0:
                #output_games = model(training_data)
                #training_loss = loss_fn(output_games, training_data)

                output_games = model(testing_data)
                validation_loss = loss_fn(output_games, testing_data)

                #print("Epoch", t,"training_loss", training_loss.item(), "validation_loss", validation_loss.item())
                print("Epoch", t, "validation loss", validation_loss.item())

        # output how much games are actually reconstructed right
        move_errors = 0
        parse_errors = 0
        outcome_errors = 0
        valid_reconstruction = 0
        all_data = torch.tensor(data).float().to(training_device)
        output_games = model(all_data).cpu()
        for i, game in enumerate(games):
            diff = [a_i - b_i for a_i, b_i in zip(all_data[i].tolist(), output_games[i].tolist())]
            reconstructed_game = game_history.GameHistory()
            try:
                reconstructed_game.parse_sparse_feature(output_games[i].tolist())
            except:
                parse_errors += 1 
                continue

            outcome_mismatch = reconstructed_game.outcome() != game.outcome()
            move_mismatch = reconstructed_game.moves()[:game.length()] != game.moves()

            if outcome_mismatch:
                outcome_errors += 1
            if move_mismatch:
                move_errors += 1
            if not (outcome_mismatch or move_mismatch):
                valid_reconstruction += 1

        print("Testing reconstruction")
        print("Ill-formed features:", 100*parse_errors /len(all_data), "%")
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
        input = torch.tensor(incomplete_game.sparse_feature()).float()
        output = self._model(input)
        reconstructed_game = game_history.GameHistory()
        try:
            reconstructed_game.parse_sparse_feature(output[0].tolist())
            # there is no guarantee that reconstructed moves are actually possible
            choosen_move = reconstructed_game.moves()[current_turn]
            if choosen_move in possible_moves:
                return choosen_move
        except:
            pass
        
        # if no winning move was found let us move to forcing a draw
        incomplete_game = copy.deepcopy(history)
        incomplete_game.record_outcome(0)
        input = torch.tensor(incomplete_game.sparse_feature()).float()
        output = self._model(input)
        reconstructed_game = game_history.GameHistory()
        try:
            reconstructed_game.parse_sparse_feature(output[0].tolist())
             # there is no guarantee that reconstructed moves are actually possible
            choosen_move = reconstructed_game.moves()[current_turn]
            if choosen_move in possible_moves:
                return choosen_move
        except:
            pass

        # at this point network failed to provide possible move, just use random
        self._random_moves += 1
        random_index = random.randint(0, len(possible_moves) - 1)
        return possible_moves[random_index]