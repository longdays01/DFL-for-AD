import os
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import get_network, get_iterator, get_model, args_to_string, EXTENSIONS, logger_write_params, print_model
import time

import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import threading

class Network(ABC):
    def __init__(self, args):
        """
        Abstract class representing a network of worker collaborating to train a machine learning model,
        each worker has a local model and a local data iterator.
        Should implement `mix` to precise how the communication is done
        :param args: parameters defining the network
        """
        self.args = args
        self.device = args.device
        self.batch_size_train = args.bz_train
        self.batch_size_test = args.bz_test
        self.network = get_network(args.network_name, args.architecture, args.experiment)
        self.n_workers = self.network.number_of_nodes()
        self.local_steps = args.local_steps
        self.log_freq = args.log_freq
        self.fit_by_epoch = args.fit_by_epoch
        self.initial_lr = args.lr
        self.optimizer_name = args.optimizer
        self.lr_scheduler_name = args.decay
        self.max_round = args.n_rounds
        self.alpha = args.alpha
        self.beta = args.beta    # Heavy-ball momentum parameter
        self.gamma = args.gamma   # Nesterov momentum parameter        
        self.min_degree = args.min_degree
        self.best_rmse = float('inf')
        self.best_round = 0
        self.rmse_flag = False
        self.small_rmse_flag = False
        self.medium_rmse_flag = False
        # create logger
        if args.save_logg_path == "":
            self.logger_path = os.path.join("loggs", args_to_string(args), args.architecture)
        else:
            self.logger_path = args.save_logg_path
        os.makedirs(self.logger_path, exist_ok=True)
        if not args.test:
            self.logger_write_param = logger_write_params(os.path.join(self.logger_path, 'log.txt'))
        else:
            self.logger_write_param = logger_write_params(os.path.join(self.logger_path, 'test.txt'))
        self.logger_write_param.write(args.__repr__())

        self.logger_write_param.write('>>>>>>>>>> start time: ' + str(time.asctime()))
        self.time_start = time.time()
        self.time_start_update = self.time_start

        self.logger = SummaryWriter(self.logger_path)

        self.round_idx = 0  # index of the current communication round

        # get data loaders
        self.train_dir = os.path.join("data", args.experiment, args.network_name, "train")
        self.test_dir = os.path.join("data", args.experiment, args.network_name, "test")

        extension = EXTENSIONS["driving"] if "driving" in args.experiment else EXTENSIONS[args.experiment]
        self.train_path = os.path.join(self.train_dir, "train" + extension)
        self.test_path = os.path.join(self.test_dir, "test" + extension)

        print('- Loading: > %s < dataset from: %s'%(args.experiment, self.train_path))
        self.train_iterator = get_iterator(args.experiment, self.train_path, self.device, self.batch_size_test)
        print('- Loading: > %s < dataset from: %s'%(args.experiment, self.test_path))
        self.test_iterator = get_iterator(args.experiment, self.test_path, self.device, self.batch_size_test)

        self.workers_iterators = []
        train_data_size = 0
        print('>>>>>>>>>> Loading worker-datasets')
        for worker_id in range(self.n_workers):
            data_path = os.path.join(self.train_dir, str(worker_id) + extension)
            print('\t + Loading: > %s < dataset from: %s' % (args.experiment, data_path))
            self.workers_iterators.append(get_iterator(args.experiment, data_path, self.device, self.batch_size_train))
            train_data_size += len(self.workers_iterators[-1])

        self.epoch_size = int(train_data_size / self.n_workers)

        # create workers models
        self.workers_models = [get_model(args.experiment, args.model, self.device,
                                         optimizer_name=self.optimizer_name, lr_scheduler=self.lr_scheduler_name,
                                         initial_lr=self.initial_lr, epoch_size=self.epoch_size)
                               for w_i in range(self.n_workers)]

        # average model of all workers
        self.global_model = get_model(args.experiment, args.model,
                                      self.device,
                                      epoch_size=self.epoch_size)
        print_model(self.global_model.net, self.logger_write_param)

        # write initial performance
        if not self.args.test:
            self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    def write_logs(self):
        """
        write train/test loss, train/tet accuracy for average model and local models
         and intra-workers parameters variance (consensus) adn save average model
        """
        if (self.round_idx - 1) == 0:
            return None
        print('>>>>>>>>>> Evaluating')
        print('\t - train set')
        start_time = time.time()
        train_loss, train_rmse = self.global_model.evaluate_iterator(self.train_iterator)
        end_time_train = time.time()
        print('\t - test set')
        test_loss, test_rmse = self.global_model.evaluate_iterator(self.test_iterator)
        end_time_test = time.time()
        self.logger.add_scalar("Train/Loss", train_loss, self.round_idx)
        self.logger.add_scalar("Train/RMSE", train_rmse, self.round_idx)
        self.logger.add_scalar("Test/Loss", test_loss, self.round_idx)
        self.logger.add_scalar("Test/RMSE", test_rmse, self.round_idx)
        self.logger.add_scalar("Train/Time", end_time_train - start_time, self.round_idx)
        self.logger.add_scalar("Test/Time", end_time_test - end_time_train, self.round_idx)

        # write parameter variance
        average_parameter = self.global_model.get_param_tensor()

        param_tensors_by_workers = torch.zeros((average_parameter.shape[0], self.n_workers))

        for ii, model in enumerate(self.workers_models):
            param_tensors_by_workers[:, ii] = model.get_param_tensor() - average_parameter

        consensus = (param_tensors_by_workers ** 2).mean()
        self.logger.add_scalar("Consensus", consensus, self.round_idx)
        self.logger_write_param.write(f'\t Round: {self.round_idx} |Train Loss: {train_loss:.5f} |Train RMSE: {train_rmse:.5f} |Eval-train Time: {end_time_train - start_time:.3f}')
        self.logger_write_param.write(f'\t -----: {self.round_idx} |Test  Loss: {test_loss:.5f} |Test  RMSE: {test_rmse:.5f} |Eval-test  Time: {end_time_test - end_time_train:.3f}')
        self.logger_write_param.write(f'\t -----: Time: {time.time() - self.time_start_update:.3f}')
        self.logger_write_param.write(f'\t -----: Total Time: {time.time() - self.time_start:.3f}')

        self.time_start_update = time.time()

        if test_rmse < 0.1:
            self.rmse_flag = True
            if test_rmse < 0.9:
                self.medium_rmse_flag = True
                if test_rmse < 0.07:
                    self.small_rmse_flag = True

            if test_rmse < self.best_rmse:
                self.logger_write_param.write(f'\t -----: Best RMSE: {test_rmse:.5f}')
                self.best_rmse = test_rmse
                self.best_round = self.round_idx
                self.save_models(round=self.round_idx)

            else:
                self.logger_write_param.write(f'\t -----: Reload model from round: {self.best_round}') 
                self.load_models(round=self.best_round)

        # if not self.args.test:
        #     self.save_models(round=self.round_idx)

    # def save_models(self, round):
    #     round_path = os.path.join(self.logger_path, 'round_%s' % round)
    #     os.makedirs(round_path, exist_ok=True)
    #     path_global = round_path + '/model_global.pth'
    #     model_dict = {
    #         'round': round,
    #         'model_state': self.global_model.net.state_dict()
    #     }
    #     torch.save(model_dict, path_global)
    #     for i in range(self.n_workers):
    #         path_silo = round_path + '/model_silo_%s.pth' % i
    #         model_dict = {
    #             'epoch': round,
    #             'model_state': self.workers_models[i].net.state_dict()
    #         }
    #         torch.save(model_dict, path_silo)

    # def load_models(self, round):
    #     self.round_idx = round
    #     round_path = os.path.join(self.logger_path, 'round_%s' % round)
    #     path_global = round_path + '/model_global.pth'
    #     print('loading %s' % path_global)
    #     model_data = torch.load(path_global)
    #     self.global_model.net.load_state_dict(model_data.get('model_state', model_data))
    #     for i in range(self.n_workers):
    #         path_silo = round_path + '/model_silo_%s.pth' % i
    #         print('loading %s' % path_silo)
    #         model_data = torch.load(path_silo)
    #         self.workers_models[i].net.load_state_dict(model_data.get('model_state', model_data))


    def save_models(self, round):
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        os.makedirs(round_path, exist_ok=True)
        path_global = round_path + '/model_global.pth'
        model_dict = {
            'round': round,
            'model_state': self.global_model.net.state_dict(),
            'optimizer_state': self.global_model.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'alpha': self.alpha,
            'round_idx': self.round_idx
        }
        torch.save(model_dict, path_global)
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            model_dict = {
                'epoch': round,
                'model_state': self.workers_models[i].net.state_dict(),
                'optimizer_state': self.workers_models[i].optimizer.state_dict()
            }
            torch.save(model_dict, path_silo)

    def load_models(self, round):
        self.round_idx = round
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        path_global = round_path + '/model_global.pth'
        print('loading %s' % path_global)
        model_data = torch.load(path_global)
        self.global_model.net.load_state_dict(model_data['model_state'])
        self.global_model.optimizer.load_state_dict(model_data['optimizer_state'])
        self.scheduler.load_state_dict(model_data['scheduler_state'])
        self.alpha = model_data['alpha']
        self.round_idx = model_data['round_idx']
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            print('loading %s' % path_silo)
            model_data = torch.load(path_silo)
            self.workers_models[i].net.load_state_dict(model_data['model_state'])
            self.workers_models[i].optimizer.load_state_dict(model_data['optimizer_state'])




    def plot_results(self):
        def plot():
            plt.figure(figsize=(10, 5))
            plt.plot(self.rounds, self.train_losses, color='r', label='Train Loss')
            plt.plot(self.rounds, self.test_losses, color='g', label='Test Loss')
            plt.xlabel('Communication Rounds')
            plt.ylabel('Loss')
            plt.title('Training and Testing Loss over Rounds')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.logger_path, 'losses_plot_%s.png' % self.round_idx))
            plt.close()

        plot_thread = threading.Thread(target=plot)
        plot_thread.start()

    def save_evaluation_results(self):
        with open(os.path.join(self.logger_path, 'evaluation_results.txt'), 'w') as f:
            for result in self.evaluation_results:
                f.write(f"Round: {result['round']}, Avg Train Loss: {result['avg_train_loss']:.5f}, Avg Train RMSE: {result['avg_train_rmse']:.5f}, Avg Test Loss: {result['avg_test_loss']:.5f}, Avg Test RMSE: {result['avg_test_rmse']:.5f}, Evaluation Time: {result['evaluation_time']:.2f}s\n")

class Peer2PeerNetwork(Network):
    def mix(self, write_results=True):
        """
        :param write_results:
        Mix local model parameters in a gossip fashion
        """        
        # update workers
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()
            self.write_logs()

        # mix models
        for param_idx, param in enumerate(self.global_model.net.parameters()):
            temp_workers_param_list = [torch.zeros(param.shape).to(self.device) for _ in range(self.n_workers)]
            for worker_id, model in enumerate(self.workers_models):
                for neighbour in self.network.neighbors(worker_id):
                    coeff = self.network.get_edge_data(worker_id, neighbour)["weight"]
                    temp_workers_param_list[worker_id] += \
                        coeff * list(self.workers_models[neighbour].net.parameters())[param_idx].data.clone()

            for worker_id, model in enumerate(self.workers_models):
                for param_idx_, param_ in enumerate(model.net.parameters()):
                    if param_idx_ == param_idx:
                        param_.data = temp_workers_param_list[worker_id].clone()

        self.round_idx += 1
    
    def save_models(self, round):
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        os.makedirs(round_path, exist_ok=True)
        path_global = round_path + '/model_global.pth'
        model_dict = {
            'round': round,
            'model_state': self.global_model.net.state_dict(),
            'optimizer_state': self.global_model.optimizer.state_dict(),
            'round_idx': self.round_idx
        }
        torch.save(model_dict, path_global)
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            model_dict = {
                'epoch': round,
                'model_state': self.workers_models[i].net.state_dict(),
                'optimizer_state': self.workers_models[i].optimizer.state_dict()
            }
            torch.save(model_dict, path_silo)

    def load_models(self, round):
        self.round_idx = round
        round_path = os.path.join(self.logger_path, 'round_%s' % round)
        path_global = round_path + '/model_global.pth'
        print('loading %s' % path_global)
        model_data = torch.load(path_global)
        self.global_model.net.load_state_dict(model_data['model_state'])
        self.global_model.optimizer.load_state_dict(model_data['optimizer_state'])
        self.round_idx = model_data['round_idx']
        for i in range(self.n_workers):
            path_silo = round_path + '/model_silo_%s.pth' % i
            print('loading %s' % path_silo)
            model_data = torch.load(path_silo)
            self.workers_models[i].net.load_state_dict(model_data['model_state'])
            self.workers_models[i].optimizer.load_state_dict(model_data['optimizer_state'])

class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
    def __call__(self, val_loss, model, round, network):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, round, network)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, round, network)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, round, network):
        # torch.save(model.state_dict(), 'checkpoint.pt')
        round_path = os.path.join(network.logger_path, 'round_%s' % round)
        os.makedirs(round_path, exist_ok=True)
        path_global = round_path + '/model_global.pth'
        model_dict = {
            'round': round,
            'model_state': model.net.state_dict()
        }
        torch.save(model_dict, path_global)

class ExponentialDecayScheduler:
    def __init__(self, initial_lr, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        return self.initial_lr * (self.decay_rate ** (self.current_step / self.decay_steps))

    def state_dict(self):
        return {
            'initial_lr': self.initial_lr,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        self.initial_lr = state_dict['initial_lr']
        self.decay_rate = state_dict['decay_rate']
        self.decay_steps = state_dict['decay_steps']
        self.current_step = state_dict['current_step']


class Peer2PeerNetworkABP(Network):
    """
    Mix local model parameters in a gossip fashion over time-varying communication graphs.
    """           
    def __init__(self, args):        
        super(Peer2PeerNetworkABP, self).__init__(args)
        self.train_losses = [] 
        self.test_losses = []
        self.rounds = []  
        self.evaluation_results = []  
        self.k = args.local_steps  # Number of local iterations

        # self.alpha = 0.00002  # Step size/learning rate
        # self.beta = 0.0000    # Heavy-ball momentum parameter
        # self.gamma = 0.0000001   # Nesterov momentum parameter
        self.max_grad_norm = 1.0  # Maximum norm value for gradient clipping

        self.scheduler = ExponentialDecayScheduler(self.alpha, decay_rate=0.99, decay_steps=3000)
        
        self.stop_criterion = False

        self.s = []
        self.y = []
        self.x_prev = []
        self.x_diff_prev = [[torch.zeros_like(param) for param in model.net.parameters()] for model in self.workers_models]
        # self.schedulers = []  # Store schedulers for each worker

        # self.early_stopping = EarlyStopping(patience=20, delta=0.001)

        for worker_id, model in enumerate(self.workers_models):
            s_worker = []
            y_worker = []
            x_prev_worker = []

            # Initialize s and x_prev with the model parameters
            for param in model.net.parameters():
                s_worker.append(param.clone().detach().to(self.device))
                x_prev_worker.append(param.clone().detach().to(self.device))
            self.s.append(s_worker)
            self.x_prev.append(x_prev_worker)
            self.x_diff_prev
            # Compute the initial gradients for y
            model.net.to(self.device)
            model.net.zero_grad()
            x, y = next(iter(self.workers_iterators[worker_id]))
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
            predictions = model.net(x)
            loss = model.criterion(predictions, y)
            loss.backward()

            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.net.parameters(), self.max_grad_norm)

            for param in model.net.parameters():
                y_worker.append(param.grad.clone().detach().to(self.device) if param.grad is not None else torch.zeros_like(param).to(self.device))
            self.y.append(y_worker)

            # # Create and store a scheduler for each worker's optimizer
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, 'min', patience=10, factor=0.5)
            # self.schedulers.append(scheduler)

    def generate_comm_matrices(self, n_workers, min_degree, additional_edges):
        rng = np.random.default_rng()

        adj_matrix_A = self.getAdjMat(rng, n_workers, min_degree=min_degree, additional_edges=additional_edges)
        adj_matrix_B = self.getAdjMat(rng, n_workers, min_degree=min_degree, additional_edges=additional_edges)
        
        A = self.getArow(adj_matrix_A)
        B = self.getBcol(adj_matrix_B)
        
        return A, B

    def local_updates(self, worker_id):
        model = self.workers_models[worker_id]
        model.net.to(self.device)
        for _ in range(self.k):
        #     model.net.zero_grad()
        #     x, y = next(iter(self.workers_iterators[worker_id]))
        #     x = x.to(self.device, dtype=torch.float)
        #     y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
        #     predictions = model.net(x)
        #     loss = model.criterion(predictions, y)
        #     loss.backward()
        #     model.optimizer.step()
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                    n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

    def mix(self, write_results=True): 
        gradients_list = []
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            self.local_updates(worker_id)  # Perform local updates for k iterations

            # Get gradients after local updates
            model.net.zero_grad()
            x, y = next(iter(self.workers_iterators[worker_id]))
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
            predictions = model.net(x)
            loss = model.criterion(predictions, y)
            loss.backward()
            
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.net.parameters(), self.max_grad_norm)

            gradients = []
            for param in model.net.parameters():
                gradients.append(param.grad.clone().detach().to(self.device) if param.grad is not None else torch.zeros_like(param).to(self.device))
            gradients_list.append(gradients)        

        # if self.rmse_flag: self.log_freq = 16
        # if self.medium_rmse_flag: self.log_freq = 8
        # if self.small_rmse_flag: self.log_freq = 1

        # write logs
        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            for param_idx, param in enumerate(self.global_model.net.parameters()):
                param.data.fill_(0.)
                for worker_model in self.workers_models:
                    param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()
            self.write_logs()

        A, B = self.generate_comm_matrices(n_workers=self.n_workers, min_degree=self.min_degree, additional_edges=5)

        x_new = [[torch.zeros_like(param) for param in model.net.parameters()] for model in self.workers_models]
        s_new = [[torch.zeros_like(param) for param in model.net.parameters()] for model in self.workers_models]
        y_new = [[torch.zeros_like(param) for param in model.net.parameters()] for model in self.workers_models]
        # x_diff_new = [[torch.zeros_like(param) for param in model.net.parameters()] for model in self.workers_models]
        new_gradients_list = []

        for param_idx in range(len(list(self.workers_models[0].net.parameters()))):
            for worker_id, model in enumerate(self.workers_models):
                # Update x
                x_new[worker_id][param_idx] = torch.zeros_like(list(model.net.parameters())[param_idx])
                for j in range(self.n_workers):
                    x_new[worker_id][param_idx] += A[worker_id, j] * self.s[j][param_idx]
                x_new[worker_id][param_idx] -= self.alpha * self.y[worker_id][param_idx]
                # x_new[worker_id][param_idx] += self.beta / self.gamma * (self.s[worker_id][param_idx] - self.x_prev[worker_id][param_idx])
                x_new[worker_id][param_idx] += self.beta * self.x_diff_prev[worker_id][param_idx]               
                self.x_diff_prev[worker_id][param_idx] = x_new[worker_id][param_idx] - self.x_prev[worker_id][param_idx]

        # Update s
        for param_idx in range(len(list(self.workers_models[0].net.parameters()))):
            for worker_id, model in enumerate(self.workers_models):
                s_new[worker_id][param_idx] = x_new[worker_id][param_idx] + self.gamma * self.x_diff_prev[worker_id][param_idx]
                self.x_prev[worker_id][param_idx] = x_new[worker_id][param_idx]
                self.s[worker_id][param_idx] = s_new[worker_id][param_idx]

        # Compute new gradients for s_new
        for worker_id, model in enumerate(self.workers_models):
            for param_idx, param in enumerate(model.net.parameters()):
                param.data = s_new[worker_id][param_idx].data
            model.net.to(self.device)
            # Get new gradients after update
            model.net.zero_grad()
            x, y = next(iter(self.workers_iterators[worker_id]))
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
            predictions = model.net(x)
            loss = model.criterion(predictions, y)
            loss.backward()

            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.net.parameters(), self.max_grad_norm)
            

            new_gradients = []
            for param in model.net.parameters():
                new_gradients.append(param.grad.clone().detach().to(self.device) if param.grad is not None else torch.zeros_like(param).to(self.device))
            new_gradients_list.append(new_gradients)

        for param_idx in range(len(list(self.workers_models[0].net.parameters()))):
            for worker_id, model in enumerate(self.workers_models):
                # Calculate the gradient difference
                diff_grad = new_gradients_list[worker_id][param_idx] - gradients_list[worker_id][param_idx]

                # print(f"Worker {worker_id} - Param {param_idx} - Shape of gradient: {gradients_list[worker_id][param_idx].shape}")
                # print(f"Worker {worker_id} - Param {param_idx} - Shape of new gradient: {new_gradients_list[worker_id][param_idx].shape}")

                # Update y
                y_new[worker_id][param_idx] = torch.zeros_like(self.y[worker_id][param_idx])
                for j in range(self.n_workers):
                    y_new[worker_id][param_idx] += B[j, worker_id] * self.y[j][param_idx]
                y_new[worker_id][param_idx] += diff_grad
                
                # Update stored values
                self.y[worker_id][param_idx] = y_new[worker_id][param_idx]
                # Plot results after evaluation

        # if ((self.round_idx-1) % 1000 == 0 and self.round_idx!= 0 and not self.args.test) or self.round_idx == self.max_round:
        #     # if not self.args.test:
        #     self.save_models(round=self.round_idx)
        #     self.plot_results()

        # Check early stopping criteria
        # val_loss, val_rmse = self.global_model.evaluate_iterator(self.test_iterator)
        # self.early_stopping(val_loss, self.global_model, self.round_idx, self)
        # if self.early_stopping.early_stop:
        #     print("Early stopping")
        #     return

        # Update learning rate schedulers
        # for scheduler in self.schedulers:
        #     scheduler.step()

        # Update alpha using the scheduler
        # self.alpha = self.scheduler.step()
        self.round_idx += 1


    # Generate the Adjacency matrix
    def getAdjMat(self, rng, n, min_degree, additional_edges=5):
        Adj = np.zeros((n, n))
        if n == 1:
            Adj[0][0] = 1
            return Adj
        rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1610925)))

        # Ensure each node has a self-loop
        np.fill_diagonal(Adj, 1)

        # Add initial random edges to ensure strong connectivity
        for i in range(n):
            for _ in range(min_degree):
                j = rng.randint(0, n)
                if i != j:
                    Adj[i][j] = 1

        # Ensure the graph is strongly connected
        while not self.is_strongly_connected(Adj):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                Adj[i][j] = 1

        # Add additional random edges to increase density
        for _ in range(additional_edges):
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j:
                Adj[i][j] = 1

        return Adj

    # Check if the graph is strongly connected
    def is_strongly_connected(self, adj_matrix):
        graph = csr_matrix(adj_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=True, connection='strong')
        return n_components == 1

    # Get the Weight matrix A (row-stochastic)
    def getArow(self, Adj):
        A = Adj.copy()
        n = len(Adj)
        for i in range(n):
            sumRow = np.sum(Adj[i])
            A[i] = Adj[i] / sumRow
        return A

    # Get the Weight matrix B (column-stochastic)
    def getBcol(self, Adj):
        AdjT = Adj.T
        BT = AdjT.copy()
        n = len(AdjT)
        for i in range(n):
            sumRow = np.sum(AdjT[i])
            BT[i] = AdjT[i] / sumRow
        B = BT.T
        return B

class CentralizedNetwork(Network):
    def mix(self, write_results=True):
        """
        :param write_results:
        All the local models are averaged, and the average model is re-assigned to each work
        """
        for worker_id, model in enumerate(self.workers_models):
            model.net.to(self.device)
            if self.fit_by_epoch:
                model.fit_iterator(train_iterator=self.workers_iterators[worker_id],
                                   n_epochs=self.local_steps, verbose=0)
            else:
                model.fit_batches(iterator=self.workers_iterators[worker_id], n_steps=self.local_steps)

        for param_idx, param in enumerate(self.global_model.net.parameters()):
            param.data.fill_(0.)
            for worker_model in self.workers_models:
                param.data += (1 / self.n_workers) * list(worker_model.net.parameters())[param_idx].data.clone()

        for ii, model in enumerate(self.workers_models):
            for param_idx, param in enumerate(model.net.parameters()):
                param.data = list(self.global_model.net.parameters())[param_idx].data.clone()

        self.round_idx += 1

        if ((self.round_idx - 1) % self.log_freq == 0) and write_results:
            self.write_logs()