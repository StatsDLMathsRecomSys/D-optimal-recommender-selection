from typing import cast
import os 
import numpy as np
import pandas as pd
from gensim.parsing import preprocess_string
from sklearn.preprocessing import LabelEncoder

data_path = '.'
rating = pd.read_feather(os.path.join(data_path, 'ratings.feather'))
print(rating.iidx.min())
user_num, item_num = rating.uidx.max() + 1, rating.iidx.max() + 1

word_set = set()
genre_set = set()
data = []
with open(os.path.join(data_path, 'movies.dat'), encoding = "ISO-8859-1") as f:
    for line in f:
        iidx_raw, title_raw, genre_raw = line.strip().split('::')
        iidx = int(iidx_raw)
        title_feat = preprocess_string(genre_raw)
        word_set.update(title_feat)
        genre_list = genre_raw.strip().split('|')
        genre_set.update(genre_list)

        data.append((iidx, title_feat, genre_list))

word_encoder = LabelEncoder().fit(list(word_set))
genre_encoder = LabelEncoder().fit(list(genre_set))

bow_title = np.zeros((item_num, len(word_set)))
bow_genre = np.zeros((item_num, len(genre_set)))
for iidx, word_list, genre_list in data:
    word_idx_list = word_encoder.transform(word_list)
    genre_idx_list = genre_encoder.transform(genre_list)
    bow_title[iidx, word_idx_list] += 1
    bow_genre[iidx, genre_idx_list] += 1
movie_context = np.concatenate([bow_title, bow_genre], axis=1)
np.save('item_feat.npy')

