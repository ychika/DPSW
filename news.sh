#!/bin/bash

#python alt_train.py -sr -d news -w PS -e 300 -hk 0.01 -rs 0.1 -lr 0.00001 -rb 0.001 -ehd 100 -eod 100 -wd 0.01 -b 1000 -edp 0.145 -odp 0.145 -ood 2 -imbf mmd2_rbf
python alt_train.py -sr -d news -w PS-norm -e 300 -hk 0.01 -rs 0.1 -lr 0.00001 -rb 0.001 -ehd 100 -eod 100 -wd 0.01 -b 1000 -edp 0.145 -odp 0.145 -ood 2 -imbf mmd2_rbf
