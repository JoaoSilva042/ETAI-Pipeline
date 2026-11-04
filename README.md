# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.



## Name and Number

Name: João Guilherme Teixeira Pinto Santos Silva
Number: 20260492


## Result Comparison Week 02

Logistic regression scores:
  Train accuracy: 0.679
  Test accuracy:  0.676
  Gap (train - test): +0.004

Decision Three scores:
  Train accuracy: 0.829
  Test accuracy:  0.631
  Gap (train - test): +0.198

The best model is the Logitic regression because it had the best test accuracy and the decision three is overfitting


## Week 03 - Data Quality and Preprocessing

 Checked missing values, invalid values and redundant columns. Also added data cleaning, imputation and different encoding and scaling options. The main improvement is that we can handle missing and inconsistent data instead of simply dropping rows.

## Week 04 - Cross-Validation

I added 5-fold stratified cross-validation and separated 20% of the dataset as a locked test set.
Cross-validation gives a more reliable estimate than using only one split. Comparing train and validation scores also helps us detect overfitting


## Week 05 - Hyperparameter Tuning
 Optuna with 30 trials and nested cross-validation was inserted to tune the Decision Tree

Decision Tree before tuning:
  -Train accuracy: 0.695
  -Validation accuracy: 0.609
  -Gap (train - validation): +0.086

Decision Tree after tuning:
  -Train accuracy: 0.684
  -Validation accuracy: 0.675
  -Gap (train - validation): +0.009

The tuned Decision Tree has better validation accuracy and a much smaller gap. This suggests that tuning reduced overfitting and improved generalization.

## Week 06 - Ensemble Learning
Inserted Random Forest, Oblique Random Forest and AdaBoost, their tuning search spaces, and a corrected paired statistical test. Oblique Random Forest was tested with 30 Optuna trials.

Oblique Random Forest before tuning:
  -Train accuracy: 0.759
  -Validation accuracy: 0.643
  -Gap (train - validation): +0.116

Oblique Random Forest after tuning:
  -Train accuracy: 0.691
  -Validation accuracy: 0.681
  -Gap (train - validation): +0.010

The tuned Oblique Random Forest has better validation accuracy and less overfitting. It also performed slightly better than the tuned Decision Tree (0.681 vs 0.675), but the difference was not statistically significant in our corrected paired test.


## Week 07 - Gradient Boosting and Model Evaluation

Added Gradient Boosting, Histogram Gradient Boosting and Gaussian Naive Bayes. The pipeline now calculates multiple evaluation metrics, uses ROC-AUC for Optuna tuning and saves experiment comparisons in `results/runs.csv`.


Gradient Boosting:
- ROC-AUC: 0.728
- Accuracy: 0.682
- Recall: 0.559
- F1-score: 0.613

Histogram Gradient Boosting:
- ROC-AUC: 0.728
- Accuracy: 0.685
- Recall: 0.567
- F1-score: 0.618

Gaussian Naive Bayes:
- ROC-AUC: 0.703
- Accuracy: 0.616
- Recall: 0.240
- F1-score: 0.361

Both Gradient Boosting models achieved almost identical ROC-AUC, but Histogram Gradient Boosting had slightly better accuracy, recall and F1-score. Naive Bayes had higher precision (0.727) but much lower recall, meaning it missed more positive cases. These results show why accuracy or ROC-AUC alone is not enough to compare models. The locked test set was not evaluated.

## Week 08 - XGBoost and Feature Selection

This week, I added XGBoost and a VoteSelector that combines Random Forest, Gradient Boosting, and Logistic Regression (L1) to select features. I also added `min_votes` to Optuna's tuning.

Results (30 trials, nested 5-fold CV, ROC-AUC):
- Without feature selection: 0.727 ± 0.016
- With feature selection: 0.725 ± 0.013
- Corrected paired t-test: p ≈ 0.448
- Selected features: 4 out of 10 

Feature selection reduced the number of features from 10 to 4, but the ROC-AUC was slightly worse. The difference wasn't statistically significant, so there's no clear benefit from using feature selection here, especially considering the extra training time. For now, I'd stick with XGBoost without feature selection.

