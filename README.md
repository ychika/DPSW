# Differentiable Pareto-Smoothed Weighting (DPSW) 

Weighted representation learning with differentiable Pareto-smoothed weights for CATE estimation from high-dimensional observational data.

### Reference:

`Yoichi Chikahara, Kansei Ushiyama. "Differentiable Pareto-Smoothed Weighting for High-Dimensional Heterogeneous Treatment Effect Estimation", UAI, 2024.`

[arXiv preprint](https://arxiv.org/abs/2404.17483)

# Dependencies

In our experiments, we used Python 3.10.9 on MacOS.

Use pip to install the required python packages:

```
pip install -r requirements.txt
```

Download and copy [fast-soft-sort](https://github.com/google-research/fast-soft-sort) folder:

```
cp -r (path-to-downloaded-folder)/fast-soft-sort ./
```

# Data

## Semi-synthetic datasets:

### News dataset: 
https://www.fredjo.com/files/NEWS_csv.zip

Save each csv file in ./data/NEWS_csv/csv/....csv

### LBIDD dataset:
https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/blob/master/data/LBIDD/README.md

Save it as .npz file: ./data/LBIDD/LBIDD.npz

## Synthetic datasets:

For instance, prepare datasets with number of features d = 15 by

```
python data_generator.py 5 5 5 0 1. 1. 4
```

Save them in ./data/synthH/5_5_5/...

Note: '5_5_5' denotes the dimensionality of each of the three feature representations (d/3 = 5 when d = 15).

# Run codes using .sh files

```
./synthH.sh
```

```
./news.sh
```

```
./LBIDD.sh
```

If you would like to change weighting schemes, use -w option in each .sh file:

- Proposed method (DPSW and DPSW Norm):
```
-w PS
```
```
-w PS-norm
```

- DRCFR (DRCFR, DRCFR Norm, and DRCFR Trunc):
```
-w IPW
```
```
-w norm
```
```
-w trunc
```

See config.py for the other options.

# Baseline (PSW)

We also offer an implementation of baseline method PSW, non-differentiable Pareto-smoothed weighting, which separately learns the propensity score model and the other models.

## Dependencies

Install [arviz](https://pypi.org/project/arviz/) package, which includes the implementation of Pareto smoothing 

```
pip install arviz
```

Then prepare the Pareto smoothed IPW weights for each dataset

```
python test_PSW.py -d news
```

Finally train the feature representations and outcome prediction models by
```
python fixedprop_train.py -w PS-fixed -d news .... 
```

# Licence

This repository is licensed under NTT License. See  "NTTSoftwareLicenseAgreement.txt" for details.
