# Data

This repository does not redistribute the raw NASA C-MAPSS files.

Place the following files in this directory (or change `DATA_DIR` in the notebook):

- `train_FD001.txt` ... `train_FD004.txt`
- `test_FD001.txt` ... `test_FD004.txt`
- `RUL_FD001.txt` ... `RUL_FD004.txt`

The notebook also accepts test filenames with the `(1)` suffix used by some downloads.

## Important

C-MAPSS is a simulated turbofan degradation dataset. In this project it is used as a source of degradation trajectories for an abstract maintenance decision environment; it is **not** presented as a physical production-line dataset.
