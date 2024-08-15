import os
from torch.multiprocessing import Process
import torch
# import optuna

from utils.args import parse_args
from utils.utils import loggs_to_json, args_to_string
from communication import Peer2PeerNetwork, Peer2PeerNetworkABP, CentralizedNetwork, LocalNetwork
import time

def objective(trial, args):
    # Use the initial command-line arguments
    # Suggest additional hyperparameters to tune, starting from the provided values
    args.alpha = trial.suggest_loguniform('alpha', args.alpha / 10, args.alpha * 10)
    args.beta = trial.suggest_uniform('beta', 0.0, 1.0)
    # args.gamma = trial.suggest_uniform('gamma', 0.0, 1.0)
    args.lr = trial.suggest_loguniform('lr', args.lr / 10, args.lr * 10)
    args.bz_train = trial.suggest_categorical('bz_train', [16, 32, 64, 128])
    args.local_steps = trial.suggest_int('local_steps', 1, 10)
    args.min_degree = trial.suggest_int('min_degree', 2, 10)

    # Choose the network type based on the parsed argument
    if args.network_type == "Peer2PeerNetwork":
        network = Peer2PeerNetwork(args)
    elif args.network_type == "Peer2PeerNetworkABP":
        network = Peer2PeerNetworkABP(args)
    elif args.network_type == "CentralizedNetwork":
        network = CentralizedNetwork(args)
    elif args.network_type == "LocalNetwork":
        network = LocalNetwork(args)        
    else:
        raise ValueError(f"Unsupported network type: {args.network_type}")

    start = time.time()
    for k in range(args.n_rounds):
        network.mix()
        if k % 100 == 0:
            end = time.time()
            round_time = (end - start)
            print(f'Round: {k} |Train Time: {round_time:.3f}')
            start = time.time()
        if len(network.best_rmse_arr) >= 2 and abs(network.best_rmse_arr[-1] - network.best_rmse_arr[-2]) < args.threshold:
            break
    network.write_logs()

    # Return validation metric to be minimized
    validation_rmse = network.best_rmse  # Assuming best RMSE is tracked
    return validation_rmse

if __name__ == "__main__":
    torch.manual_seed(1204)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    if args.tuning:
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, args), n_trials=50)  # Run 50 trials

        # Print the best hyperparameters and corresponding validation RMSE
        print("Best hyperparameters: ", study.best_params)
        print("Best validation RMSE: ", study.best_value)
    else: 
        total_train_time = 0.0
        print("Run experiment in sequential setting..")

        # Choose the network type based on the parsed argument
        if args.network_type == "Peer2PeerNetwork":
            network = Peer2PeerNetwork(args)
        elif args.network_type == "Peer2PeerNetworkABP":
            network = Peer2PeerNetworkABP(args)
        elif args.network_type == "CentralizedNetwork":
            network = CentralizedNetwork(args)
        elif args.network_type == "LocalNetwork":
            network = LocalNetwork(args)                   
        else:
            raise ValueError(f"Unsupported network type: {args.network_type}")

        start = time.time()
        for k in range(args.n_rounds):
            network.mix()
            if k % 100 == 0:
                end = time.time()
                round_time = (end - start)
                print(f'Round: {k} |Train Time: {round_time:.3f}')
                start = time.time()
            if (len(network.best_rmse_arr) >= 2 and abs(network.best_rmse_arr[-1]-network.best_rmse_arr[-2]) < args.threshold):
                break
        network.write_logs()