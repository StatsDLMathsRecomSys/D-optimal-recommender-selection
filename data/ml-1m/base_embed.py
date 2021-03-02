import os 
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.preprocessing import LabelEncoder
from acgan.recommender import SVDRecommender

# import gensim
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data_path='.'
names = ['uidx', 'iidx', 'rating', 'ts']
dtype = {'uidx':int, 'iidx':int, 'rating':float, 'ts':float}
data_df = pd.read_csv(os.path.join(data_path, 'ratings.dat'), 
sep='::', 
names=names,
dtype=dtype)

data_df['rating'] = (data_df['rating'] >= 4).astype(np.float32)

user_num, item_num = data_df.uidx.max() + 1, data_df.iidx.max() + 1
print(user_num, item_num)

uidx_encoder = LabelEncoder()
iidx_encoder = LabelEncoder()
data_df.uidx = uidx_encoder.fit_transform(data_df.uidx)
data_df.iidx = iidx_encoder.fit_transform(data_df.iidx)

user_num, item_num = data_df.uidx.max() + 1, data_df.iidx.max() + 1

print(len(uidx_encoder.classes_))
print(user_num, item_num)
data_df.to_feather(os.path.join(data_path, 'ratings.feather'))

def movie():
    # feature engineering for movies
    train_corpus = []
    with open(os.path.join(data_path, 'movies.dat'), encoding='ISO-8859-1') as f:
        for line in f:
            iidx_raw, title_raw, genre_raw = line.strip().split('::')
            try:
                iidx = int(iidx_encoder.transform([int(iidx_raw)])[0])
            except ValueError:
                continue
            genre_str = genre_raw.strip().replace('|', ' ')
            doc = simple_preprocess(title_raw + genre_str)
            train_corpus.append(TaggedDocument(doc, [iidx]))

    model = Doc2Vec(vector_size=50, min_count=2, epochs=20)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    context_embedding = np.zeros((item_num, 50))
    for i in range(item_num):
        context_embedding[i, :] = model.docvecs.vectors_docs[i]
    np.save(os.path.join(data_path,'item_feat.npy'), context_embedding)


def user():
    # feature engineering for users
    # UserID::Gender::Age::Occupation::Zip-code
    gender_set = set()
    occu_set = set()
    zipcode_set = set()
    age_set = set()
    data = []
    max_idx = 0
    with open(os.path.join(data_path, 'users.dat'), encoding='ISO-8859-1') as f:
        for line in f:
            puidx, gender, age, occupation, zipcode = line.strip().split('::')
            uidx = uidx_encoder.transform([int(puidx)])[0]
            
            max_idx = max(max_idx, uidx)
            # assert(uidx < user_num)
            gender_set.add(gender)
            age_set.add(age)
            occu_set.add(occupation)
            zipcode_set.add(zipcode)
            data.append([uidx, gender, age, occupation, zipcode])
    print(max_idx)
    gender_encoder = LabelEncoder().fit(list(gender_set))
    gender_feat = np.zeros((user_num, len(gender_set)))

    age_encoder = LabelEncoder().fit(list(age_set))
    age_feat = np.zeros((user_num, len(age_set)))

    occu_encoder = LabelEncoder().fit(list(occu_set))
    occu_feat = np.zeros((user_num, len(occu_set)))

    zipcode_encoder = LabelEncoder().fit(list(zipcode_set))
    zipcode_feat = np.zeros((user_num, len(zipcode_set)))

    for uidx, gender, age, occupation, zipcode in data:
        gender = gender_encoder.transform([gender])[0]
        gender_feat[uidx, gender] += 1

        age = age_encoder.transform([age])[0]
        age_feat[uidx, age] += 1

        occupation = occu_encoder.transform([occupation])[0]
        occu_feat[uidx, age] += 1

        zipcode = zipcode_encoder.transform([zipcode])[0]
        zipcode_feat[uidx, zipcode] += 1

    user_context = np.concatenate([gender_feat, age_feat, occu_feat], axis=1)
    print(f'user_context shape: {user_context.shape}')
    np.save(os.path.join(data_path,'user_feat.npy'), user_context)

movie()

user()




