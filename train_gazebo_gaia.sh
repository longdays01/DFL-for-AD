#!/usr/bin/env bash
python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet --n_rounds 3000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 1 --lr 0.00008 --decay sqrt

python main.py driving_gazebo --network_name gaia --architecture ring --model FADNet_plus --n_rounds 5000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 5 --lr 0.0001 --alpha 0.0000002 --min_degree 6 --decay cyclic

python main.py driving_carla --beta 0.0000001 --gamma 0.0000001 --network_name gaia --architecture ring --model FADNet_plus --n_rounds 5000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 5 --lr 0.0001 --alpha 0.0000002 --min_degree 6 --decay cyclic
python main.py driving_gazebo --n_workers 22 --beta 0.0000001 --gamma 0.0000001 --network_name gaia --architecture ring --model FADNet_plus --n_rounds 5000 --bz_train 32 --bz_test 32 --device cuda --log_freq 20 --local_steps 5 --lr 0.0001 --alpha 0.0000002 --min_degree 6 --decay cyclic

# CARLA
python main.py driving_carla --network_name gaia --decay cyclic --architecture ring --device cuda --log_freq 20 --n_rounds 5000 --bz_train 32 --bz_test 32 --local_steps 5 --lr 0.0001 --min_degree 6 --beta 0.0000001 --gamma 0.0000000 --alpha 0.0000002 --model ADTVNet

#GAZEBO
python main.py driving_gazebo --network_name gaia --decay cyclic --architecture ring --device cuda --log_freq 20 --n_rounds 5000 --bz_train 32 --bz_test 32 --local_steps 5 --lr 0.0001 --min_degree 6 --beta 0.0000001 --gamma 0.0000000 --alpha 0.0000002 --model ADTVNet
