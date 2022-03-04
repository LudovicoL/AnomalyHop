#!/bin/bash
if [ -z $1 ]; then
    printf "Error! Choose an option.\n1) Aitex\n2) MvTecAD\n3)BTAD"
elif [[ $1 == 1 ]]; then
    python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 -d "aitex"
elif [[ $1 == 2 ]]; then

    # 1 - carpet 
    python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 --class_names carpet -d "mvtec"

    # 2 - grid
    python ./src/main.py --kernel 5 5 5 --num_comp 5 5 5 --layer_of_use 1 2 3 --distance_measure self_ref --hop_weights 0.2 0.2 0.2 --class_names grid -d "mvtec"

    # 3 - leather
    python ./src/main.py --kernel 5 5 3 2 2 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.3 0.7 0.3 0.3 --class_names leather -d "mvtec"

    # 4 - tile
    python ./src/main.py --kernel 4 6 6 4 2 --num_comp 3 2 5 5 2 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.3 0.1 0.6 0.4 0.2 --class_names tile -d "mvtec"

    # 5 - wood
    python ./src/main.py --kernel 3 7 3 3 2 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.6 0.0 0.0 0.2 0.6 --class_names wood -d "mvtec"

    # 6 - bottle
    python ./src/main.py --kernel 7 5 2 2 2 --num_comp 2 3 3 3 3 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.0 0.1 0.0 0.0 0.0 --class_names bottle -d "mvtec"

    # 7 - cable
    python ./src/main.py --kernel 3 4 4 2 5 --num_comp 2 2 4 3 5 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.4 0.5 0.2 0.1 0.7 --class_names cable -d "mvtec"

    # 8 - capsule
    python ./src/main.py --kernel 4 5 5 --num_comp 4 5 3 --layer_of_use 1 2 3 --distance_measure loc_gaussian --hop_weights 0.0 0.1 0.1 --class_names capsule -d "mvtec"

    # 9 - hazel nut
    python ./src/main.py --kernel 3 3 2 4 3 --num_comp 5 2 4 4 3 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.7 0.0 0.7 0.4 0.0 --class_names hazelnut -d "mvtec"

    # 10 - metal nut
    python ./src/main.py --kernel 7 3 5 4 2 --num_comp 2 3 4 5 4 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.1 0.7 0.1 0.5 0.2 --class_names metal_nut -d "mvtec"

    # 11 - pill
    python ./src/main.py --kernel 3 2 7 2 --num_comp 5 2 3 2 --layer_of_use 1 2 3 4 --distance_measure loc_gaussian --hop_weights 0.6 0.1 0.3 0.3 --class_names pill -d "mvtec"

    # 12 - screw
    python ./src/main.py --kernel 2 2 7 5 --num_comp 3 4 4 2 --layer_of_use 1 2 3 4 --distance_measure self_ref --hop_weights 0.7 0.3 0.6 0.3 --class_names screw -d "mvtec"

    # 13 - toothbrush
    python ./src/main.py --kernel 3 3 --num_comp 3 3 --layer_of_use 1 2 --distance_measure loc_gaussian --hop_weights 0.1 0.1 --class_names toothbrush -d "mvtec"

    # 14 - transistor
    python ./src/main.py --kernel 5 7 3 5 4 --num_comp 4 5 3 5 5 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.4 0.0 0.1 0.2 0.4 --class_names transistor -d "mvtec"

    # 15 - zipper
    python ./src/main.py --kernel 5 5 3 2 2 --num_comp 5 5 2 4 4 --layer_of_use 1 2 3 4 5 --distance_measure loc_gaussian --hop_weights 0.2 0.3 0.7 0.3 0.3 --class_names zipper -d "mvtec"

elif [[ $1 == 3 ]]; then
    python ./src/main.py --kernel 7 6 3 2 4 --num_comp 4 4 4 4 4 --layer_of_use 1 2 3 4 5 --distance_measure glo_gaussian --hop_weights 0.2 0.2 0.4 0.5 0.1 -d "btad"
fi