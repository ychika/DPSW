#!/bin/bash

#python alt_train.py -sr -d synthH -di 5 -w IPW -e 300 -hk 0.001 -rs 0.01 -lr 0.0001 -lrp 0.0001 -rb 0.001 -ehd 3 -eod 3 -phd 3 -wd 0.01 -b 5000 -edp 0. -odp 0. -ood 2 -imbf mmd2_rbf
python alt_train.py -sr -d synthH -di 5 -w PS -e 300 -hk 0.001 -rs 0.01 -lr 0.0001 -lrp 0.001 -rb 0.001 -ehd 3 -eod 3 -phd 3 -wd 0.01 -b 5000 -edp 0. -odp 0. -ood 2 -imbf mmd2_rbf


