# Intelligent Maintenance with MDP & Reinforcement Learning

A degradation-aware maintenance decision framework using **NASA C-MAPSS**, **Markov Decision Processes (MDP)**, **Value Iteration**, and **Q-Learning**.

> Academic project developed by **Ialisson Roque da Silva**, Electronics Engineering, UFRPE/UACSA (2026).

## Overview

The project transforms multivariate degradation trajectories into a sequential maintenance decision problem.

```text
NASA C-MAPSS
      |
      v
Sensor preprocessing
      |
      v
Operating-regime normalization
      |
      v
Sensor selection
      |
      v
PCA / Health Index
      |
      v
Discrete degradation states
      |
      v
Empirical transition model
      |
      v
MDP maintenance environment
   /               \
  v                 v
Value Iteration   Q-Learning
   \               /
    v             v
      Maintenance policy
```

The goal is not simply to predict failure. The goal is to use degradation information to decide **what action to take before failure**.

## Important dataset note

C-MAPSS is a **simulated turbofan degradation dataset**. It is not treated here as a physical production-line dataset.

The degradation trajectories are used as an empirical source for the natural deterioration dynamics of an abstract production-equipment maintenance environment. Maintenance actions, their effects, and their utility/cost values are modeling assumptions because C-MAPSS does not contain historical maintenance decisions.

## Methodology

The main pipeline is:

1. Load FD001-FD004 training/test trajectories.
2. Compute training RUL as the distance to the final observed cycle.
3. Normalize sensors according to operating conditions.
4. Rank sensors by absolute Spearman association with RUL.
5. Select the 10 most informative sensors.
6. Use PCA to build a one-dimensional Health Index.
7. Discretize the Health Index into degradation states.
8. Estimate natural transition probabilities from consecutive observations.
9. Build an MDP with production, reduced-load, preventive-maintenance, and failure utilities.
10. Compare a rule-based baseline with Value Iteration and Q-Learning.

## MDP

### States

| State | Interpretation |
|---|---|
| S0 | Healthy |
| S1 | Moderate degradation |
| S2 | Severe degradation |
| S3 | Critical degradation |
| S4 | Failure |

### Actions

- `operate`
- `reduced_load`
- `maintenance`

### Base experimental utilities

| Event/action | Utility |
|---|---:|
| Operate | +10 |
| Reduced load | +6 |
| Preventive maintenance | -25 |
| Failure / corrective repair | -100 |

These are **experimental utility units**, not monetary values and not values supplied by NASA.

The base discount factor is `gamma = 0.95`.

## Health Index

After operating-regime normalization, the Health Index showed a strong monotonic association with RUL across all four subsets.

| Dataset | Spearman HI x RUL | PC1 explained variance |
|---|---:|---:|
| FD001 | ~0.77 | ~0.75 |
| FD002 | ~0.78 | ~0.57 |
| FD003 | ~0.84 | ~0.63 |
| FD004 | ~0.77 | ~0.58 |

RUL is used to prepare and validate the indicator. The decision agent does **not** receive the true RUL as its state.

## Learned policy

Value Iteration and Q-Learning converged to the same operational policy in the base scenario:

| State | Action |
|---|---|
| Healthy | Operate |
| Moderate | Operate |
| Severe | Operate |
| Critical | Maintenance |

The reduced-load action was not selected in the base scenario.

The rule-based baseline is more conservative and performs maintenance from the severe state onward.

## Results

Discounted return from the healthy initial state:

| Dataset | Baseline | Value Iteration | Q-Learning |
|---|---:|---:|---:|
| FD001 | 195.88 | 198.57 | 198.57 |
| FD002 | 193.51 | 197.52 | 197.52 |
| FD003 | 197.02 | 198.25 | 198.25 |
| FD004 | 196.22 | 197.11 | 197.11 |

The optimized policy improves the modeled discounted objective in all four subsets.

However, the long-run simulation also exposes an important trade-off: waiting longer before maintenance reduces preventive interventions but can increase failure exposure. For example, in FD004 the baseline produced about **102.2 maintenance actions per 10,000 steps**, while the optimized policy produced about **57.6 maintenance actions** and **18.8 failures per 10,000 steps**.

Therefore, **higher modeled return does not automatically imply higher operational safety**.

## Repository structure

```text
intelligent-maintenance-mdp/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── mdp_cmapss_analysis.ipynb
├── src/
│   └── projeto_mdp.py
├── results/
│   ├── resultados.csv
│   ├── resumo_datasets.csv
│   └── figures/
└── docs/
    └── technical_report_pt-BR.pdf
```

## Running the notebook

### Google Colab

Mount Google Drive and point `DATA_DIR` to the folder containing the C-MAPSS files.

Example:

```python
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
DATA_DIR = Path("/content/drive/MyDrive/Colab Notebooks/Projeto")
```

Then run the notebook from top to bottom.

### Local Jupyter

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Activate it, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

Place the C-MAPSS files in the expected data directory or update `DATA_DIR`.

## Reproducibility

The experiments use fixed random seeds where stochastic procedures are involved. Q-Learning uses an epsilon-greedy exploration strategy and the base experiment uses 30,000 episodes with 60 steps per episode.

## Limitations

- C-MAPSS is simulated.
- The underlying system is a turbofan model, not a production line.
- The dataset does not contain historical maintenance actions.
- Maintenance effects and utilities are modeled assumptions.
- The Health Index depends on preprocessing and dimensionality-reduction choices.
- A real industrial deployment would require calibration using real costs, constraints, maintenance records, and safety requirements.

## Technologies

Python, NumPy, Pandas, SciPy, scikit-learn, PCA, K-Means, Matplotlib, Markov Decision Processes, Value Iteration, Q-Learning, Jupyter Notebook.

## Author

**Ialisson Roque da Silva**  
Electronics Engineering — UFRPE/UACSA

## License

The source code in this repository is released under the MIT License. Dataset licensing and redistribution terms remain those of the original data provider.
