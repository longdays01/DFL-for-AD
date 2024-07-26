#!/usr/bin/env bash
python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet --n_rounds 3000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 1 --lr 0.00008 --decay sqrt

python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet_plus --n_rounds 5000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 5 --lr 0.0001 --alpha 0.0000002 --min_degree 6 --decay cyclic

python main.py driving_gazebo --beta 0.0000001 --gamma 0.0000001 --network_name gaia --architecture ring --model FADNet_plus --n_rounds 5000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 5 --lr 0.0001 --alpha 0.0000002 --min_degree 6 --decay cyclic


