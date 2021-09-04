# Credit Card Customer Default
This repo hosts a Machine Learning pipeline for analysis of credit card customer default.

For more details, please refer to the [Kaggle Kernel](https://www.kaggle.com/arashshamseddini/kaggle-customer-default/edit).

In this analysis, a classification problem of predicting whether a credit card client will default or not is addressed.

## Dataset
The dataset in this study is the [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) in which there are 30,000 examples and 24 features, and the goal is to estimate whether a person will default (fail to pay) their credit card bills; this column is labeled `default.payment.next.month` in the data.

Below is a sneak peek at the dataset.
![dataset](./Images/dataset.png)

## Modeling
In this work, the following classifiers have been used and the final [recall](https://en.wikipedia.org/wiki/Precision_and_recall) and [f1](https://en.wikipedia.org/wiki/F-score) scores are compared as follows. 

- Baseline Classifier ([DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html))

- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) (Don't fall for the name, this is a classifier not a regressor)

- [RandomForest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

- [XGBoost](https://xgboost.readthedocs.io/en/latest/)

- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)


![scores](./Images/scores.png)

