#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse

def parse_args():
    
    parser = argparse.ArgumentParser('AnomalyHop')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--kernel', nargs='+', help='kernel sizes as a list')
    parser.add_argument('--num_comp', nargs='+', help='number of components kept for each stage')
    parser.add_argument('--distance_measure', type=str, choices=['self_ref','loc_gaussian', 'glo_gaussian'])
    parser.add_argument('--layer_of_use', nargs='+', help='layers output used to compute gaussian')
    parser.add_argument('--hop_weights', nargs='+', help='weights for each hop')
    parser.add_argument('--class_names', nargs='+', help='classes for evaluation')
    parser.add_argument("-d", "--dataset", default="aitex", help="Choose the dataset: \"aitex\", \"mvtec\", \"btad\".")
    parser.add_argument("-t", '--telegram', type=bool, default=True, help="Send notification on Telegram.")
    parser.add_argument("-r", '--resize', default=False, action="store_true", help="Resize AITEX dataset.")
    return parser.parse_args()

