import numpy as np
import math
import json
import sklearn
import gzip
import random
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

# stores information from dataset
user_id = []
item_id = []
ratings = []
all_data = []

# load data from json file
with gzip.open("../../reviews_Video_Games_5.json.gz", "rb") as a_file:
    reviews = a_file.readlines()
    for review in reviews:
        jd = json.loads(review)
        #review = review.decode('UTF-8').strip()
        #item_index = review.index('\"asin\"')
        #rating_index = review.index('\"overall\"')
        #item_index_end = review.find('\", ', review.find('\", ') + 1)
        uID = jd['reviewerID']
        iID = jd['asin']
        r = float(jd['overall'])
        user_id.append(uID)
        item_id.append(iID)
        ratings.append(r)
        all_data.append([uID, iID, r])

# take unique user and item id
user_id =set(user_id)
item_id = set(item_id)
user_id = list(user_id)
item_id = list(item_id)

n_ratings = len(ratings)
n_items = len(item_id)
n_users = len(user_id)

#print(f"Number of ratings: {n_ratings}")
#print(f"Number of unique gameIDs: {n_items}")
#print(f"Number of unique users: {n_users}")

# map user to ID, item to ID
id2user = {}
user2id = {}
id2item = {}
item2id = {}
for i,j in enumerate(user_id):
    id2user[i]=j
    user2id[j]=i
for i,j in enumerate(item_id):
    id2item[i]=j
    item2id[j]=i    
for i in range(n_users):
    user_id[i] = user2id[user_id[i]]
for i in range(n_items):
    item_id[i] = item2id[item_id[i]]
for i in range(len(all_data)):
    old = all_data[i]
    new = [user2id[old[0]],item2id[old[1]],old[2]]
    all_data[i] = new

# for each user, split data into trainning and testing
train_data = []
test_data = []
for i in user_id:
    gather_i = []
    for j in all_data:
        if j[0]==i:
            gather_i.append(j)
    random.shuffle(gather_i)
    select_num = int(len(gather_i)*0.8)
    train_data.extend(gather_i[:select_num])
    test_data.extend(gather_i[select_num:])


# create user-item matrix using csr_matrix
def create_matrix(data):
    user_index = [row[0] for row in data]
    item_index = [row[1] for row in data]
    rating = [row[2] for row in data]

    matrix = csr_matrix((rating, (user_index, item_index)), shape=(n_users, n_items))
    return matrix
    

M_train = create_matrix(train_data)
M_test = create_matrix(test_data)

save_npz('parsed_data/train_data.npz', M_train)
save_npz('parsed_data/test_data.npz', M_test)