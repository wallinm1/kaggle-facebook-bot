kaggle-facebook-bot
===================

**This repository contains a Jupyter notebook that outlines my approach for the Kaggle Facebook Recruiting IV contest.**

The .csv-files of predictions can be generated as follows:

1. Download and extract the data from https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data. Place the files `train`, `test` and `bids` into the `data`-directory of this repository. Place the file `sampleSubmission.csv` into the `submissions`-directory of this repository.

2. Run the `facebook_notebook.ipynb`-notebook. This should generate the submission file`facebook_submission.csv` into the `submissions`-directory.

**Libraries:**

The basic scientific Python libraries + XGBoost 

**Running times/Hardware:**

Runs in about 15 minutes on a fairly high-powered desktop (i7-4790) with 16 gb of RAM. Can clog up the ram on smaller machines.