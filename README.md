kaggle-facebook-bot
===================

**This repository contains a Jupyter notebook that outlines my approach for the Kaggle Facebook Recruiting IV contest.**

The .csv-files of predictions can be generated as follows:

1. Download and extract the data from https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data. Place the files `train.csv`, `test.csv` and `bids.csv` into the `data`-directory of this repository. Place the file `sampleSubmission.csv` into the `submissions`-directory of this repository.

2. Run the `facebook_notebook.ipynb`-notebook. This should generate the submission file`facebook_submission.csv` into the `submissions`-directory.

**Libraries:**

The basic scientific Python libraries + XGBoost

**Running time/Hardware:**

Runs in about 15 minutes on a fairly high-powered desktop (i7-4790) with 16 gb of RAM. Can clog up the ram on smaller machines.

**Update Feb 21st 2016**

- Added a `Model interpretation`-section to the notebook
- Added a `hyperopt_xgb.py`-script that shows how hyperparameters can be optimized using a grid search.
  - The script generates a file `hyperopt_xgb.csv` in the root of the repository which displays a selection of hyperparameters and the corresponding cross-validated `AUC`-score.
  - Running the script requires two additional dependencies: the `hyperopt`- and `pymongo`-libraries.
