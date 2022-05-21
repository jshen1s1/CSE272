# Basic Requirements 
import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy import sparse

M_train = sparse.load_npz('parsed_data/train_data.npz')
M_test = sparse.load_npz('parsed_data/test_data.npz')


# calculate cosine similarity matrix
user_similarity = cosine_similarity(M_train)
np.fill_diagonal(user_similarity, 0)

item_similarity = cosine_similarity(M_train.transpose())
np.fill_diagonal(item_similarity, 0)

# predict the user rating with collaborative filtering
def predict(rating, similarity, t='user'):
    if t == 'user':
        mean_user_rating = np.true_divide(rating.sum(1),(rating!=0).sum(1))
        rating_diff = (rating - mean_user_rating)
        x = similarity.dot(rating_diff)
        y = np.abs(similarity).sum(axis=1)[:, None]
        pred = mean_user_rating + x / y
    elif t == 'item':
        pred = rating.dot(similarity) / np.abs(similarity).sum(axis=1)
    return pred

user_pred = predict(M_train, user_similarity, 'user')
item_pred = predict(M_train, item_similarity, 'item')

# calculate MAE
def MAE(pred, truth, t='user'):
    if t == 'user':
        pred = pred[truth.nonzero()].flatten()
        truth = truth[truth.nonzero()].flatten()
    elif t == 'item': 
        pred = pred[truth.nonzero()][:, np.newaxis].transpose()
        truth = truth[truth.nonzero()].flatten()
    return mean_absolute_error(pred, truth)

print('User CF MAE:' + str(MAE(user_pred, M_test, 'user')))
#out: User CF MAE:4.058589748963497
print('Item CF MAE:' + str(MAE(item_pred, M_test, 'item')))
#out: Item CF MAE:4.032211335797178

# calculate RMSE
def RMSE(pred, truth, t='user'):
    if t == 'user':
        pred = pred[truth.nonzero()].flatten()
        truth = truth[truth.nonzero()].flatten()
    elif t == 'item': 
        pred = pred[truth.nonzero()][:, np.newaxis].transpose()
        truth = truth[truth.nonzero()].flatten()
    return math.sqrt(mean_squared_error(pred, truth))

print('User CF RMSE:' + str(RMSE(user_pred, M_test, 'user')))
#out: User CF RMSE:4.217341187777234
print('Item CF RMSE:' + str(RMSE(item_pred, M_test, 'item')))
#out: Item CF RMSE:4.209659891755404