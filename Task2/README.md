# 📨 Task 2 — Spam Email Detection (Logistic Regression)

**Dataset:** [`max.ge/aiml_midterm/b_oikashvili2024_894531_csv`](https://max.ge/aiml_midterm/b_oikashvili2024_894531_csv)
**Repository copy:** [`data/b_oikashvili2024_894531.csv`](./data/b_oikashvili2024_894531.csv)

This project implements a **console-based Python program** that classifies email messages as **spam** or **legitimate** using a **logistic regression model**.
The workflow covers **data preprocessing**, **model training (70 % / 30 % split)**, **evaluation**, **email text parsing**, and **visualization** of results.

**GitHub repository (exam project):**
👉 [github.com/bekarius/cyber_ai_exam](https://github.com/bekarius/cyber_ai_exam)

---

## 2.1  Dataset Upload  (1 pt)

The provided dataset file was uploaded to the repository in
`data/b_oikashvili2024_894531.csv`

This CSV contains extracted numerical features such as:

* number of words,
* number of links,
* capitalized word count,
* spam trigger word count,
  and a binary label column (`is_spam`) for classification.

---

## 2.2  Model Training  (2 + 1 + 2 + 1 + 1 pts)

### 🧠 Command

```bash
python3 app.py train \
  --data data/b_oikashvili2024_894531.csv \
  --model models/logreg.joblib
```

### 📂 Source Code References

| Component          | File                                     | Description                                                   |
| :----------------- | :--------------------------------------- | :------------------------------------------------------------ |
| Main entry point   | [`app.py`](./app.py)                     | Parses CLI arguments and controls workflow                    |
| Training logic     | [`src/train.py`](./src/train.py)         | Fits logistic regression on 70 % of data                      |
| Utilities          | [`src/utils.py`](./src/utils.py)         | Loads CSV and detects target column                           |
| Feature extraction | [`src/features.py`](./src/features.py)   | Extracts `words`, `links`, `capital_words`, `spam_word_count` |
| Evaluation         | [`src/evaluate.py`](./src/evaluate.py)   | Computes accuracy & confusion matrix                          |
| Visualization      | [`src/visualize.py`](./src/visualize.py) | Produces class balance & feature scatter plots                |

---

### 🔍 Data Loading & Processing

* Automatically detects the label column (`is_spam` / `Label`).
* Strips spaces from column headers.
* Replaces missing values with 0.
* Keeps numeric columns:

  ```
  ['words', 'links', 'capital_words', 'spam_word_count']
  ```
* Encodes target as 0 (legitimate) / 1 (spam).
* Performs stratified train/test split (70 % / 30 %) for balanced representation.

---

### ⚙️ Model

Uses `LogisticRegression(max_iter=1000, solver='lbfgs')`
to learn how feature values relate to the probability of spam.

---

### 📊 Coefficients & Interpretation

| Feature             | Description                                            | Weight (β) |
| :------------------ | :----------------------------------------------------- | :--------: |
| **links**           | Number of URLs in the email body                       |   +0.879   |
| **spam_word_count** | Number of spam trigger terms (“win”, “offer”, “bonus”) |   +0.706   |
| **capital_words**   | Count of ALL-CAPS tokens                               |   +0.416   |
| **words**           | Total word count (normalizer term)                     |   +0.008   |

Intercept = −8.9508
A higher linear sum predicts higher spam probability.

---

## 2.3  Model Evaluation on Hold-out Data  (1 + 2 pts)

### Command

```bash
python3 app.py eval \
  --data data/b_oikashvili2024_894531.csv \
  --model models/logreg.joblib
```

### Output

```
Accuracy: 0.9580
Confusion Matrix (rows = true, cols = pred)
[[1214   37]
 [ 68 1181]]
```

*Accuracy ≈ 95.8 % on unseen data.*

**Visualization:** The confusion matrix image (`figs/confusion_matrix.png`)
is generated via `src/evaluate.py` and matplotlib.

---

## 2.4  Email Parsing and Prediction  (3 pts)

### Command

```bash
python3 app.py predict \
  --model models/logreg.joblib \
  --data data/b_oikashvili2024_894531.csv \
  --email data/sample_email.txt
```

### Process

1. Reads raw text from a .txt file.
2. Calls `src/features.py` to extract identical features as training data.
3. Passes vector to trained logistic regression.
4. Prints `Probability(spam)` and final classification based on threshold = 0.5.

---

## 2.5  Manual Spam Email Example  (1 pt)

```
URGENT: YOU WON A BONUS PRIZE!
Confirm now to claim your REWARD:
http://promo.example/claim  http://offer.example/win

Limited-time CASH bonus. Click to ACTIVATE your account and get $$$ fast.
This message is EXCLUSIVE — do not share.
```

**Result**

```
Probability(spam) = 0.5874 → class = spam
```

**Explanation:**
Contains multiple URLs, capital letters, and common spam trigger words.
All highly weighted positive features → classified as spam.

---

## 2.6  Manual Legitimate Email Example  (1 pt)

```
Hi team,
Please find attached the meeting notes and the agenda for tomorrow.
We will review Q2 metrics and open action items.
Thanks,
Beka
```

**Result**

```
Probability(spam) = 0.0002 → class = legitimate
```

**Explanation:**
Formal language, no links or spam keywords — model correctly identifies it as legitimate.

---

## 2.7  Visualizations  (4 pts)

### Command

```bash
python3 app.py viz \
  --data data/b_oikashvili2024_894531.csv \
  --outdir figs
```

### Figures Generated

| File                            | Description                                                                                                      |
| :------------------------------ | :--------------------------------------------------------------------------------------------------------------- |
| **`figs/class_balance.png`**    | Bar chart of spam vs legitimate counts (showing balanced dataset).                                               |
| **`figs/top2_scatter.png`**     | Scatter of top-2 correlated features (`links` vs `spam_word_count`) colored by label — reveals clear separation. |
| **`figs/confusion_matrix.png`** | Heatmap of true vs predicted classes — bright diagonal indicates high accuracy.                                  |

All visualizations are saved in `figs/` and linked in this report.



### 🔗 References to Project Files

| Type              | File / Path                                                                                                                                                          |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source code       | [`app.py`](./app.py)                                                                                                                                                 |
| Model             | [`models/logreg.joblib`](./models/logreg.joblib)                                                                                                                     |
| Dataset           | [`data/b_oikashvili2024_894531.csv`](./data/b_oikashvili2024_894531.csv)                                                                                             |
| Spam sample       | [`data/sample_spam.txt`](./data/sample_spam.txt)                                                                                                                     |
| Legitimate sample | [`data/sample_legit.txt`](./data/sample_legit.txt)                                                                                                                   |
| Visualizations    | [`figs/class_balance.png`](./figs/class_balance.png), [`figs/top2_scatter.png`](./figs/top2_scatter.png), [`figs/confusion_matrix.png`](./figs/confusion_matrix.png) |

---