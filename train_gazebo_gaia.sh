#!/usr/bin/env bash
python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet --n_rounds 3000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 1 --lr 0.00008 --decay sqrt

python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet --n_rounds 3000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 1 --lr 0.0001 --alpha 0.00001 --min_degree 3 --decay sqrt

