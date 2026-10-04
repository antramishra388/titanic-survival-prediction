# Titanic Survival Prediction

Predicting which passengers survived the Titanic using machine learning (Python, pandas, scikit-learn).

## What I did
- Explored the data and handled missing values (Age filled with median, Cabin dropped because ~77% was missing)
- Converted text columns to numbers (Sex mapped to 0/1, Embarked one-hot encoded)
- Built a new feature, Title (Mr, Mrs, Miss, Master), extracted from passenger names
- Trained Logistic Regression and Random Forest models and compared them with 5-fold cross-validation

## Results
| Model | 5-fold CV accuracy |
|---|---|
| Logistic Regression | 0.826 |
| Random Forest | 0.802 |

Baseline (always predicting "died") is about 62%.

## Key findings
- Sex was the strongest signal: ~74% of women survived vs ~19% of men
- Passenger class mattered: 63% survival in 1st class vs 24% in 3rd
- Title captured something Sex alone couldn't: boys ("Master") survived at 58% vs 16% for adult men

## What I learned
- A single train/test split can be misleading; cross-validation changed which model looked better
- A simpler model can match a complex one on a small dataset

## Data
Kaggle: Titanic - Machine Learning from Disaster (train.csv)
