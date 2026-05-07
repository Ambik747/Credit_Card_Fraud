# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## What's This Project About?

Ever wondered how your bank knows when your card is being used fraudulently — even before you do?

This project tries to answer exactly that. I built a machine learning pipeline that looks at real-world credit card transaction data and learns to spot the ones that look suspicious. It's not just about training a model — it's about understanding *why* fraud happens, *when* it tends to occur, and *how* to catch it reliably despite the fact that fraudulent transactions are incredibly rare (less than 1% of all transactions).

The dataset has **339,607 transactions**, out of which only about **1,770 are fraud (~0.52%)**. That tiny fraction is what makes this problem both interesting and tricky.

---

## What's in This Repo?

```text
credit-card-fraud-detection/
│
├── Credit_Card_Fraud.ipynb       ← The main notebook — all the analysis and modeling lives here
├── fraud_detection_model.pkl     ← The best trained model, ready to use
├── fraud_scaler.pkl              ← The scaler used during training (needed for inference)
├── fraud_metadata.pkl            ← Stores the optimal threshold and feature names
├── README.md                     ← You're reading it!
└── .gitignore                    ← Keeps the large CSV out of the repo
```

> The dataset isn't included here because of its size.
> You can grab it from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — just download and drop it in the project folder as `credit_card_fraud.csv`.

---

## Exploratory Data Analysis — What Did I Find?

Before jumping into models, I spent time really *understanding* the data. Here's a summary of the 8 analyses I ran:

| # | What I Looked At |
|---|-----------------|
| 1 | How many fraud vs. non-fraud transactions are there? |
| 2 | Do fraudulent transactions tend to be larger amounts? |
| 3 | Which shopping categories have the most fraud? |
| 4 | What time of day (and day of week) does fraud spike? |
| 5 | Which US states see the most fraud? |
| 6 | Are older or younger customers more targeted? |
| 7 | Are there specific merchants that are fraud hotspots? |
| 8 | How do numeric features correlate with fraud? |

### The Most Interesting Takeaways

- **Online shopping categories** (`shopping_net`, `misc_net`) had the highest fraud rates — not surprising given the rise of card-not-present fraud
- Fraud happens most often **between midnight and 3 AM** — when people are asleep and less likely to catch it immediately
- **Transactions made far from the cardholder's location** are much more likely to be fraud (this inspired the Haversine distance feature!)
- Fraudulent transactions tend to involve **higher-than-average amounts**

---

## Feature Engineering — Making the Data Smarter

Raw data rarely tells the full story. Here's what I added or transformed:

- **Time features** — Extracted `hour`, `day_of_week`, and `month` from the transaction timestamp
- **Distance** — Calculated the real-world distance (in km) between the cardholder's home and the merchant using the Haversine formula
- **Cleaned up** irrelevant personal info — columns like name, address, and card number don't help the model and could introduce bias, so I dropped them
- **Encoded** categorical columns like `category`, `state`, and `job` into numbers the model can work with
- **Scaled** all numeric features so no single feature dominates the others

---

## Modeling — Three Different Approaches

I trained three models, each handling the class imbalance problem a bit differently:

| Model | How It Handles Imbalance |
|-------|--------------------------|
| Logistic Regression | Penalizes misclassifying fraud more heavily (`class_weight='balanced'`) |
| Random Forest | Same weighted approach, but more powerful with ensembles |
| XGBoost | Uses `scale_pos_weight` to natively upweight the fraud class |

For Logistic Regression and Random Forest, I also used **SMOTE** — a technique that *synthetically generates* new fraud examples to balance the training set, rather than just throwing away non-fraud data.

---

## How Did the Models Do?

| Model | ROC-AUC | Fraud F1 | Fraud Recall |
|-------|---------|----------|--------------|
| XGBoost | **~0.97** | High | High |
| Random Forest | ~0.96 | High | High |
| Logistic Regression | ~0.88 | Moderate | Moderate |

One important thing I did: **threshold tuning**. By default, most models predict fraud when the probability is above 0.5. But for fraud detection, you often want to be more aggressive — catch more fraud even if it means a few more false alarms. I found the optimal threshold for each model by maximizing the F1-score on the fraud class specifically.

---

## Tools & Libraries Used

| Tool | What I Used It For |
|------|--------------------|
| `pandas`, `numpy` | Wrangling the data |
| `matplotlib`, `seaborn` | All the charts and visualizations |
| `scikit-learn` | Preprocessing, training, and evaluation |
| `xgboost` | The best-performing model |
| `imbalanced-learn` | SMOTE for handling class imbalance |
| `joblib` | Saving and loading the trained model |

---

## Want to Run This Yourself?

**1. Clone the repo**

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install the dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

**3. Download the dataset**

Get `credit_card_fraud.csv` from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place it in the project folder.

**4. Open the notebook and run all cells**

```bash
jupyter notebook Credit_Card_Fraud.ipynb
```

---

## Using the Saved Model

After running the notebook, you'll have a trained model saved locally. Here's how to load it and make a prediction:

```python
import joblib

model    = joblib.load('fraud_detection_model.pkl')
scaler   = joblib.load('fraud_scaler.pkl')
metadata = joblib.load('fraud_metadata.pkl')

# Scale your input features the same way as during training
features = scaler.transform([your_feature_array])

# Get fraud probability and apply the tuned threshold
prob = model.predict_proba(features)[:, 1]
prediction = (prob >= metadata['threshold']).astype(int)

print("Fraud detected!" if prediction[0] == 1 else "Looks legit")
```

---

## What I'd Do Next

This project has a solid foundation, but there's always more to explore:

- [ ] Try **LightGBM** — it's even faster than XGBoost on large datasets
- [ ] Add **SHAP values** to explain *why* the model flags a transaction as fraud
- [ ] Run **hyperparameter tuning** to squeeze out more performance
- [ ] Wrap the model in a **Flask / FastAPI REST API** for real-time scoring
- [ ] Build a simple **Streamlit dashboard** to visualize fraud patterns interactively

---

## About Me

Hi, I'm **Ambik Mitra** , a data enthusiast who loves turning messy real-world data into meaningful insights.
Feel free to connect or check out my other projects!

[GitHub](https://github.com/Ambik747)

---

## License

This project is open-source under the [MIT License](LICENSE). Feel free to use, fork, or build on it!
