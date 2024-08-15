import os
import argparse
from datetime import datetime
from utils.utils import get_network
import pytz  # Import the pytz library

def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment',
        choices=['driving_carla', 'driving_gazebo', 'driving_udacity'],
        help='name of experiment, possible: driving_carla, driving_gazebo',
        type=str)
    parser.add_argument(
        '--network_name',
        choices=['gaia', 'amazon_us'],
        help='name of the network; possible: gaia, amazon_us',
        type=str
    )
    parser.add_argument(
        '--architecture',
        choices=['ring'],
        help='architecture to use, possible: ring',
        default='ring'
    )
    parser.add_argument(
        '--model',
        choices=['FADNet_plus', 'ADTVNet', 'AttentionADTVNet', 'InceptionNet', 'MobileNet', 'VGG16', 'RandomNet', 'ConstantNet', 'DAVE2', 'AttDAVE2', 'ResNet8'],
        help='model to use, possible: FADNet_plus, ADTVNet, AttentionADTVNet',
        default='ADTVNet'
    )
    parser.add_argument(
        "--random_ring_proba",
        type=float,
        help="the probability of using a random ring at each step; only used if architecture is ring",
        default=0.5
    )
    parser.add_argument(
        '--fit_by_epoch',
        help='if chosen each local step corresponds to one epoch,'
             ' otherwise each local step corresponds to one gradient step',
        action='store_true'
    )
    parser.add_argument(
        '--reload',
        help='if chosen reload and retrain',
        action='store_true'
    )
    parser.add_argument(
        '--tuning',
        help='find best parameters',
        action='store_true'
    )            
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds;',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--bz_train',
        help='batch_size train;',
        type=int,
        default=32
    )
    parser.add_argument(
        '--num_cpus',
        help='number of cpu cores;',
        type=int,
        default=0
    )    
    parser.add_argument(
        '--n_workers',
        help='number of worker nodes;',
        type=int,
        default=11
    )    
    parser.add_argument(
        '--poisson_rate',
        help='poisson rate for number of disconnected nodes;',
        type=int,
        default=0
    )        
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication;',
        type=int,
        default=1
    )
    parser.add_argument(
        '--log_freq',
        help='number of local steps before communication;',
        type=int,
        default=100
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or gpu;',
        type=str,
        default='cuda'
    )
    parser.add_argument(
        '--optimizer',
        help='optimizer to be used for the training;',
        type=str,
        default="adam"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help='learning rate',
        default=1e-3
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help='threshold criteria',
        default=1e-3
    )
    parser.add_argument(
        "--decay",
        help='learning rate decay scheme to be used;'
             ' possible are "cyclic", "sqrt", "linear" and "constant"(no learning rate decay);'
             'default is "cyclic"',
        type=str,
        default="constant"
    )
    parser.add_argument(
        '--bz_test',
        help='batch_size test;',
        type=int,
        default=1
    )
    parser.add_argument(
        "--test",
        help="if only evaluating test set",
        action='store_true'
    )
    parser.add_argument(
        '--save_logg_path',
        help='path to save logg and models',
        type=str,
        default="results"
    )
    parser.add_argument(
        '--alpha',
        help='step size/learning rate of individual agent',
        type=float,
        default=0.00001
    )
    parser.add_argument(
        '--beta',
        help='heavy-ball momentum parameter',
        type=float,
        default=0.000001
    )
    parser.add_argument(
        '--gamma',
        help='nesterov momentum parameter',
        type=float,
        default=0.0000001
    )        
    parser.add_argument(
        '--min_degree',
        help='minimum degree of the communication matrix',
        type=int,
        default=5
    )
    parser.add_argument(
        '--network_type',
        choices=['Peer2PeerNetwork', 'Peer2PeerNetworkABP', 'CentralizedNetwork', 'LocalNetwork'],
        help='Type of network to use; possible: Peer2PeerNetwork, Peer2PeerNetworkABP, CentralizedNetwork, LocalNetwork',
        default='Peer2PeerNetworkABP'
    )
    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    network = get_network(args.network_name, args.architecture, args.experiment)
    args.num_workers = network.number_of_nodes()

    # Get the current time in Arizona timezone
    arizona_tz = pytz.timezone('US/Arizona')
    timestamp = datetime.now(arizona_tz).strftime("%Y%m%d-%H%M%S")
    
    # Create dynamic results folder name with Arizona timestamp
    args.save_logg_path = f"{timestamp}_{args.model}_{args.network_type}_{args.experiment}_a{args.alpha}_b{args.beta}_g{args.gamma}_lr{args.lr}_md{args.min_degree}_ls{args.local_steps}_bz{args.bz_train}_n{args.n_workers}_nrounds{args.n_rounds}_poisson{args.poisson_rate}_threshold{args.threshold}"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(args.save_logg_path):
        os.makedirs(args.save_logg_path)

    return args