# Instacart competition

This is the repository for the Kaggle Instacart Market Basket Analysis competition.

Link to competition [https://www.kaggle.com/c/instacart-market-basket-analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)

## Repository

To not kill my RAM and speed up trail/error process i split up the process into feature generation filese hat generates features that are saved as .csv-files. Thes csv-files are later used by a prediction file

### Feature generation files

This repository contains the following files for feature generation:
- product_features.py
- user_features.py
- userXproduct_features.py
- userXproduct_features_2.py
- order_features.py

### Prediction

The prediction of market baskets is run using:
- xgboost_starter.py


