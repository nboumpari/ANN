# ANN
# Credit Risk Classification with Artificial Neural Networks

A feedforward neural network (ANN) built with scikit-learn to classify loan applicants as **Good** or **Bad** borrowers - and score future applicants using a profit-optimal decision threshold derived from the bank's payoff matrix.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Payoff Matrix & Cutoff Threshold](#payoff-matrix--cutoff-threshold)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

Banks face an asymmetric risk problem when issuing loans: approving a bad borrower is far more costly than rejecting a good one. A standard classifier that simply maximises accuracy ignores this asymmetry entirely.

This project addresses that problem in two stages:

1. **Build a well-tuned ANN** to estimate the probability that an applicant is a good borrower.
2. **Apply a profit-optimal cutoff threshold** derived from the bank's payoff matrix, so that the model's grant/reject decision reflects the real financial consequences - not just classification accuracy.

---

## Dataset

| File | Description |
|---|---|
| `History.csv` | 1,000 past loan applicants with known outcomes (`Good_Bad` label) |
| `Future.csv` | Unlabelled applicants to be scored by the trained model |

The data includes a mix of ordinal features (`checking`, `savings`, `employed`, `credit_cards_other_banks`) and nominal features (`coapp`, `history`, `housing`, `job`, `marital`, `purpose`, `property`).

---

## Payoff Matrix & Cutoff Threshold

### The Payoff Matrix

The bank's financial outcomes for each combination of prediction and reality are:

|  | **Predict: Grant Loan** | **Predict: Reject Loan** |
|---|---|---|
| **Actual: Bad Borrower** | −1000 | 0 |
| **Actual: Good Borrower** | +200 | −200 |

**Reading the matrix:**
- Granting a loan to a **bad borrower** costs the bank **−1000** (loss from default).
- Rejecting a **bad borrower** costs nothing (**0**) - the correct decision.
- Granting a loan to a **good borrower** earns **+200** (interest/fee profit).
- Rejecting a **good borrower** costs **−200** - the bank loses the profit it would have earned.

### Deriving the Cutoff Threshold

The cutoff is the probability *p* at which the bank is **indifferent** between granting and rejecting a loan - i.e., the point where the expected payoff of both actions is equal.

Let *p* = P(Good Borrower) for a given applicant.

**Expected payoff of granting the loan:**

$$E[\text{Grant}] = p \times 200 + (1 - p) \times (-1000)$$

**Expected payoff of rejecting the loan:**

$$E[\text{Reject}] = p \times (-200) + (1 - p) \times 0$$

**Setting them equal to find the indifference point:**

$$p \times 200 + (1 - p) \times (-1000) = p \times (-200) + (1 - p) \times 0$$

$$200p - 1000 + 1000p = -200p$$

$$1200p - 1000 = -200p$$

$$1400p = 1000$$

$$p^* = \frac{1000}{1400} \approx 0.2857$$

### Interpreting the Result

The optimal cutoff is **p* ≈ 0.2857** (≈ 28.57%).

> **Grant the loan if P(Good Borrower) ≥ 0.2857, otherwise reject.**

This threshold is intentionally low because the cost of incorrectly rejecting a good borrower (−200) is much smaller than the cost of incorrectly granting a loan to a bad borrower (−1000). The bank should therefore be willing to grant loans even to applicants it is not very confident about - as long as the expected profit still outweighs the expected loss.

This is a classic application of **Bayes-optimal decision theory**: rather than using a naive 0.5 cutoff, we shift the threshold to reflect the true asymmetry of the costs.

---

## Project Structure

```
.
├── ANN.ipynb           # Main notebook: preprocessing, architecture search, training, scoring
├── History.csv         # Historical labelled data (1,000 applicants)
├── Future.csv          # Future unlabelled applicants to score
└── README.md
```

---

## Methodology

### 1. Data Preprocessing

- **Ordinal encoding** for features with a natural order: `checking`, `savings`, `employed`, `credit_cards_other_banks`.
- **One-hot encoding** for nominal features with no meaningful order: `coapp`, `history`, `housing`, `job`, `marital`, `purpose`, `property`.
- Data is split **70 / 15 / 15** (train / validation / test), **stratified** on the target label to preserve class proportions.

### 2. Feature Scaling

All features are standardised using `StandardScaler` (zero mean, unit variance). The scaler is **fit only on the training set** and then applied to validation and test sets to prevent data leakage.

This step is essential for ANNs: without it, features with large numeric ranges (e.g., loan amount) would dominate gradient updates and slow or distort training.

### 3. Architecture Search

A grid search over single hidden layers of 2 to 40 neurons (in steps of 2) is performed. Each network is trained for a fixed 500 iterations (no early stopping, to ensure fair comparison). The architecture with the **lowest validation MSE** is selected.

### 4. Final Model Training

The best architecture is retrained with:
- `max_iter = 1000` - more room to converge.
- `early_stopping = True` - an internal 10% validation split halts training if no improvement is observed for 20 consecutive epochs, preventing overfitting.
- Activation: `ReLU` | Solver: `Adam`

### 5. Cutoff Application & Scoring

The model outputs a probability `p = P(Good Borrower)`. The profit-optimal cutoff **p* = 0.2857** (derived above) is applied:

```python
cut_off = 0.2857
y_pred = (model.predict_proba(X_test)[:, 1] >= cut_off).astype(int)
```

### 6. Scoring Future Applicants

The same preprocessing pipeline (encoders + scaler) is applied to `Future.csv` and the trained model assigns each applicant a probability score and a grant/reject recommendation.

---

## Results

| Metric | Value |
|---|---|
| Evaluation threshold | 0.2857 (profit-optimal) |
| Test MSE | Reported in notebook |
| AUC-ROC | Reported in notebook |

The ROC curve plots the True Positive Rate (bad borrowers correctly identified) against the False Positive Rate (good borrowers incorrectly flagged) across all possible thresholds - providing a threshold-independent view of model discrimination ability.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook ANN.ipynb
```

Make sure `History.csv` and `Future.csv` are in the same directory as the notebook before running.

---

## Dependencies

```
pandas
numpy
matplotlib
scikit-learn
jupyter
```

Install all at once:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

---

## Key Takeaway

Using the raw 0.5 default threshold ignores the bank's actual financial incentives. By grounding the decision boundary in the payoff matrix, the model becomes a **profit-maximising tool** rather than just an accuracy metric - a critical distinction in real-world credit risk management.
