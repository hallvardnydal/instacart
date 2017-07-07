import numpy as np
import pandas as pd
import os

data_dir = "data"
feature_dir = "features"

########################################################################
### Load data
########################################################################

print('loading prior')
priors = pd.read_csv(os.path.join(data_dir,'order_products__prior.csv'), dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(os.path.join(data_dir,'orders.csv'), dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

########################################################################
### Compute features
########################################################################

print('generating features')

def days_to_last_order(x):
    x = np.nan_to_num(x.values)
    x = np.cumsum(x)
    x = np.abs(x-x.max())
    return x

orders['days_to_last_order'] = orders.groupby('user_id')['days_since_prior_order'].transform(days_to_last_order))

# Add order info to priors
priors = priors.merge(orders,on="order_id",how="left")
priors['user_product_id'] = priors['user_id'].map(str) + "_" + priors['product_id'].map(str)

print(priors)

priors = priors.merge(orders[['order_id','days_to_last_order']],on="order_id",how="left")

user_product_group = priors.groupby('user_product_id')

def func(x):
    print(x)
    exit()

userXproduct = priors.groupby('user_product_id')['days_to_last_order'].apply(func).to_frame()
userXproduct.columns = ['UP_days_to_last_order']
userXproduct['user_product_id'] = userXproduct.index

day_data['UP_order_frequency_days'] = priors.groupby('user_product_id')['days_to_last_order'].apply((x.max()-x.min())/x.size)

print('writing features to csv')
userXproduct.to_csv(os.path.join(feature_dir,'userXproduct_features_2.csv'), index=False)
