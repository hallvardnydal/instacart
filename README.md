# Instacart competition

This is the repository for the Kaggle Instacart Market Basket Analysis competition.

Link to competition [https://www.kaggle.com/c/instacart-market-basket-analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)

## Repository

To not kill my RAM and speed up trail/error process i split up the process into feature generation files (e.g. product_features.py, user_features.py) that generates features that are saved as .csv-files. These csv-files are later used by a prediction file (prediction.py) to generate a prediction for the market baskets in the test set

### Feature generation files

This repository contains the following files for feature generation:
- product_features.py
- user_features.py
- userXproduct_features.py
- order_features.py

### Prediction

The prediction of market baskets is run using:
- prediction.py

### Other
Note that the data files (downloaded from Kaggle) is stored in the data folder, but that these are to large to be uploaded to github.
