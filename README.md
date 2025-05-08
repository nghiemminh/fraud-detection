# Fraud Detection 

The objective of this project is to detect fraudulent transactions using various machine learning models. Given the highly imbalanced nature (0.17%) of the dataset, the models are evaluated based on their Recall (with a minimum target of 0.9), F2, and financial impacts. This ensures that they are sensitive to identifying frauds, while minimizing financial impacts.

### Data Description 

[The dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains transactions made by credit cards in September 2013 by European cardholders.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the data does not contain the original features and background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount.

Given the class imbalance ratio, the author recommends measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

### Model Architecture and Results

- **Neural Network:** Trained a simple neural network with 3 dense layer, 2 two drop out, using pr_auc as metric. Achieved 0.65 PR-AUC.
- **CatBoost:** Trained a CatBoost with 1500 iterations, and max_depth = 6. Achieved 0.81 PR-AUC, 0.12 F2, and 0.03 Precision. 
- **Random Forest:** Trained a Random Forest with 500 estimators, max depth = 10, and max_features = squared. Achieved 0.8 PR-AUC, 0.14 F2, and 0.03 Precision.
- **XGBoost:** Trained a XGBoost classifier with 1000 estimators, each has max_depth = 6. Achieved **0.8 PR-AUC, 0.18 F2 score**, 0.04 Precision, and minimize the financial impacts of misclassification. 
  
---> XGBoost is the best model, combined both highest perforance **(0.8 PR-AUC)** with lowest financial impacts.

### Dependencies
Choose the lastest versions of any of the dependencies below: 
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)
- [catboost](https://catboost.ai/)
