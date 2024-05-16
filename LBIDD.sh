#!/bin/bash

python alt_train.py -sr -d LBIDD -w PS -e 300 -hk 0.001 -rs 0.01 -lr 0.00001 -rb 0.001 -ehd 100 -eod 100 -wd 0.01 -b 5000 -edp 0.145 -odp 0.145 -ood 2 -imbf mmd2_rbf
