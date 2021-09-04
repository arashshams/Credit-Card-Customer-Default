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

## [Hyperparameter Optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

Attempt has been made to optimize the **RandomForest** and **XGBoost** classifiers. The scores are listed in below table which does not show significant improvement. Looks like the vanilla LightGBM scores are better that the optimized models. 

![hyper](./Images/hyperparameter_optimization.png)

Of course, since this is more like an open ended study, there is the chance that given a deeper dive into hyperparameter optimization, better scores from the models would not be far from reach. However, this is not the main focus of this work.

## Interpretation & Feature Importances
In this section of the analysis, we would like to answer the following question:

**Which features have the most and the least impact on the target?**

First of all, we will be continuing with `LightGBM` as the best performing model. Second, we will use the [SHAP](https://github.com/slundberg/shap) approach to explain the importance of each feature. In other words:

- How small or big a feature can affect the target

- The direction towards which the feature pushes the target (class `0` or class `1`)

Technically SHAP uses the encoded features. Therefore, it would help to take a look at these features.

![encoded](./Images/encoded_features.png)

Below table shows the calculated SHAP coefficients for these encoded features.

![coeff](./Images/SHAP_coeff.png)

The results can further be explained by **SHAP Dependence** plot as well as **SHAP Summary** plot.

**SHAP Dependence Plot**

As an example for one of the features, `BILL_AMT1`:

![dependence](./Images/dependence_plot.png)

The plot above shows the effect of `BILL_AMT1` feature on the prediction. Here, the x-axis represents values of the feature `BILL_AMT1` and the y-axis is the SHAP value for that feature, which represents how much knowing that feature's value changes the output. Obviously, higher values of `BILL_AMT1` result in lower SHAP values for class "0" of the target. Also, the color corresponds to a second feature (`LIMIT_BA`L) that may have an interaction effect with `BILL_AMT1`.

**SHAP Summary Plot**

![summary](./Images/summary_plot.png)


The plot shows the most important features for predicting the class. It also shows the direction of how it is going to drive the prediction. Higher SHAP value means positive association with class `0` of the target as we are using SHAP values for class `0`. As an example, higher value of `x5_2` feature has a high negative impact on the prediction of the target as class `0`, or higher value of `LIMIT_BAL` feature has a high positive impact on the prediction of the target as class `0`.

## Results on the Test Set

Using `LightGBM` as the best performing model, we can score the model on the test set and get the pertinent scores.

The overall summary of the models' scores are indicated in the following diagram.

![scores_plot](./Images/scores_plot.png)

## Conclusions

Since the project is open-ended, there is still room for improvement. We can go much deeper in **Preprocessing**, **Feature Selection**, **Feature Engineering** and **Hyperparameter Optimization**. We can also investigate the performance of other classifiers as well and even make attempts to tune them to get better validation scores. We can summarize the concluding remarks as below:

- As expected `DummyClassifier` has the lowest performance compared to other models and the fact that we have class imbalance within the target makes the results from `DummyClassifier` even more unreliable.
- For our case, a linear model like `LogisticRegression` has an average performance where tree_based models such as `RandomForest` and `XGBoost` have scores lower than `LogisticRegression`. However, `LightGBM` performs relatively better than linear `LogisticRegression`.
- Signs of improvement in validation scores are observed with hyperparameter optimization. This is another field which, as mentioned earlier, has the potential to better the scores.


