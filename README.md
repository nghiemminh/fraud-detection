# Fraud Detection 

This repo analyzes the pattern of fraud transaction and build fraud detection models. The dataset for this repo is a [real transaction dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) of European cardholders in 2 days. The dataset contains nearly 300.000 transaction, and is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. The data contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and background information are not available. 

Here are different models (required all models get minimum 0.9 Recall):

- Neural Network: Trained a simple neural network with 3 dense layer, 2 two drop out, using pr_auc as metric. Achieved 0.65 PR-AUC.
- XGBoost: Trained a XGBoost classifier with 1000 estimators, each has max_depth = 6. Achieved **0.8 PR-AUC, 0.18 F2 score**, 0.04 Precision, and minimize the financial impacts of misclassification. 
- CatBoost: Trained a CatBoost with 1500 iterations, and max_depth = 6. Achieved 0.81 PR-AUC, 0.12 F2, and 0.03 Precision. 
- Random Forest: Trained a Random Forest with 500 estimators, max depth = 10, and max_features = squared. Achieved 0.8 PR-AUC, 0.14 F2, and 0.03 Precision.
  
---> XGBoost is the best model, combined both highest perforance with lowest financial impacts.

### Dependencies
Choose the lastest versions of any of the dependencies below: 
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [tensorflow](https://www.tensorflow.org/)
