# Maximum-Knowledge Differential Privacy: Reproducibility Repository
This repository contains the code and supplementary material accompanying the paper.

## Repository Structure

```text
.
├── proofs/               # Detailed mathematical derivations
└── experiments/
    ├── mean              # Experiments for the mean
    └── dp_multiq/        # Experiments for quantiles
```

## Supplementary Material
The proofs/ directory contains detailed derivations omitted from the paper because of page limitations.

## Code
The purpose of this repository is to provide a fully reproducible implementation of the experiments and analyses presented in the paper.

### Quantiles
To ensure fair comparisons, our implementation is built upon the Google Research dp_multiq framework. The proposed Maximum-Knowledge Quantile mechanism has been integrated into the same experimental pipeline, allowing all methods to be evaluated using identical datasets, privacy budgets, and evaluation procedures.

To run the experiments for quantiles:
- First download the Goodreads dataset from Kaggle into the /dp_multiq directory: https://www.kaggle.com/jealousleopard/goodreadsbooks.
- Save it as "books.csv" in this directory. Then use

> cd ..
> python -m dp_multiq.run_experiment

### Mean
To run the experiments for the mean:
> cd experiments/mean
> python run_experiment
