import random
import copy
import torch
import json
import game_history
from torch.optim import lr_scheduler

class QualityEstimator(torch.nn.Module):
    def __init__(self):
        super(QualityEstimator, self).__init__()

        # there are 8 meaningul line combinations, output would be 8 channels x 3 values
        self.conv1 = torch.nn.Conv1d(1, 8, 9, 9)
        self.conv1_activation = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(24, 18)
        self.linear1_activation = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(18, 12)
        self.linear2_activation = torch.nn.Tanh()
        self.linear3 = torch.nn.Linear(12, 6)
        self.linear3_activation = torch.nn.Tanh()
        self.linear4 = torch.nn.Linear(6, 3)
        
    def forward(self, data):
        x = data.view(1,-1) if data.dim() == 1 else data

        x = x.view(-1,1,27)
        x = self.conv1(x)
        x = self.conv1_activation(x)

        x = x.view(-1,24)
        x = self.linear1(x)
        x = self.linear1_activation(x)
        x = self.linear2(x)
        x = self.linear2_activation(x)
        x = self.linear3(x)
        x = self.linear3_activation(x)
        x = self.linear4(x)

        return x

class Player:
    def __init__(self):
        self._model = None

    def load(self, path):
        model = torch.load(path)
        model.eval()
        self._model = model

    def unroll(self, game_state):
        feature = [1 if cell == 0 else 0 for cell in game_state]
        feature += [1 if cell == 1 else 0 for cell in game_state]
        feature += [1 if cell == 2 else 0 for cell in game_state]
        return feature

    def train(self, games, save_path):
        training_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = QualityEstimator().to(training_device)
        states = [self.unroll(state) for game in games for state in game.states()[1:]]
        labels = [game.outcome() for game in games for state in game.states()[1:]]

        # 80% of data will be used for training and 20% for testing
        training_data_size = int(len(states) * 0.8);
        training_states = torch.tensor(states[:training_data_size]).float().to(training_device)
        training_labels = torch.tensor(labels[:training_data_size]).to(training_device)
        testing_states = torch.tensor(states[training_data_size:]).float().to(training_device)
        testing_labels = torch.tensor(labels[training_data_size:]).to(training_device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1500, 5000], 0.1)
        print("\nTraining using", training_device, "on", training_data_size, "states")
        for t in range(10000):
            # Forward pass: compute predicted y by passing x to the model.
            output_labels = model(training_states)

            loss = loss_fn(output_labels, training_labels)

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

            scheduler.step()

            # Print current loss
            if t % 100 == 0:
                output_labels = model(testing_states)
                validation_loss = loss_fn(output_labels, testing_labels)
                print("Epoch", t, "validation loss", validation_loss.item())

        self._model = model.cpu()
        torch.save(self._model, save_path)

    def choose_move(self, possible_moves, own_mark, history):
        best_move = None
        best_score = 0
        softmax = torch.nn.Softmax(0)

        for move in possible_moves:
            future = history.states()[-1]
            future[move[0] * 3 + move[1]] = own_mark
            feature = torch.tensor(self.unroll(future)).float()
            output = self._model(feature)[0]
            output = softmax(output)

            draw_score = output[0]
            win_score = output[own_mark]
            loss_score = output[2 if own_mark == 1 else 1]
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