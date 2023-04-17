# NOTIP: Non-parametric True Discovery Proportion control for brain imaging

This repository contains the code to reproduce all experiments of the Notip paper (https://arxiv.org/abs/2204.10572). The scripts directory contains a script per figure.
Note that the first time you run one of those scripts, the fMRI data fetching from Neurovault will take place, which takes a significant amount of time. This only needs to be done once.

### Installing dependencies

```
python -m pip install -r requirements.txt
```

### Reproducing figures

To reproduce any figure, i.e. figure 2, run:

```
python figure_2.py
```
This will display the corresponding figure as well as save it in ../figures

### Parallelization

To speed up any script using parallelization on CPU cores, use:

```
python figure_2.py 6
```

With any number of CPU cores instead of 6.
