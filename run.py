from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional, Union, cast
import json
import time
import random
import argparse

from tqdm import tqdm # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy import sparse as sp # type: ignore
from matplotlib import pyplot as plt # type: ignore

from acgan.module import PopularModel
from acgan.recommender import PopRecommenderV2, SVDRecommenderV2, RandRecommender, UserBasedKnn, ContextItemKnn

import uni_bandit
from simulation import run_basic, run_lin_contextual, run_optimal_contextual

#
parser = argparse.ArgumentParser('System Bandit Simulation')
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--topk', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--explore_step', type=int, default=500)
parser.add_argument('--feat_map', type=str, default='onehot_context', choices=['onehot', 'context', 'armed_context', 'onehot_context'])
parser.add_argument('--algo', type=str, default='lin_ct', choices=['base', 'e_greedy', 'thomson', 'lin_ct', 'optimal'])
args = parser.parse_args()
#
np.set_printoptions(precision=4)

print(vars(args))
#
user_embed = np.load('./data/ml-1m/user_feat.npy')
item_embed = np.load('./data/ml-1m/item_feat.npy')
data_df = pd.read_feather('./data/ml-1m/ratings.feather')
user_num, item_num = data_df.uidx.max() + 1, data_df.iidx.max() + 1
assert(max(data_df.rating) == 1)
assert(min(data_df.rating) == 0)
print(f'average overall rating: {data_df.rating.mean()}')
print(user_num, item_num)


sv_recom = SVDRecommenderV2(user_num, item_num, args.dim)
pop_recom = PopRecommenderV2(user_num, item_num)
uknn_recom = UserBasedKnn(user_num, item_num)
item_knn_recom = ContextItemKnn(user_num, item_num, item_embed)

recom_model_list = [item_knn_recom, pop_recom, sv_recom]

num_arms = len(recom_model_list)
bandit_ins = uni_bandit.bandit.Bandit(recom_model_list)

#feature_map = feature.ConcatContext(num_arms, user_embed, item_embed)

feature_map: uni_bandit.feature.FeatureMap
if args.feat_map == 'onehot':
    feature_map = uni_bandit.feature.ArmOneHot(num_arms)
elif args.feat_map == 'context':
    feature_map = uni_bandit.feature.ConcatContext(num_arms, user_embed, item_embed)
elif args.feat_map == 'armed_context':
    feature_map = uni_bandit.feature.ArmedConcatContext(num_arms, user_embed, item_embed)
elif args.feat_map == 'onehot_context':
    feature_map = uni_bandit.feature.ArmOneHotWithContext(num_arms, user_embed, item_embed)


bandit_algo: uni_bandit.bandit.BanditAlgorithm
if args.algo == 'base':
    bandit_algo = uni_bandit.bandit.RandomBandit(num_arms)
elif args.algo == 'e_greedy':
    bandit_algo = uni_bandit.bandit.EpsilonGreedy(num_arms, args.epsilon)
elif args.algo == 'thomson':
    bandit_algo = uni_bandit.bandit.ThomsonSampling(num_arms)
elif args.algo == 'lin_ct':
    bandit_algo = uni_bandit.bandit.Contextual(num_arms, feature_map.dim)
elif args.algo == 'optimal':
    bandit_algo = uni_bandit.bandit.OptimalDesign(num_arms, feature_map.dim)
else:
    raise ValueError('no known algorithms')


env = uni_bandit.env.Environment(user_num, item_num, data_df)
if args.algo == 'lin_ct':
    epoch_record = run_lin_contextual(env, bandit_algo, bandit_ins, feature_map, args)
elif args.algo == 'optimal':
    bandit_algo = cast(uni_bandit.bandit.OptimalDesign, bandit_algo)
    epoch_record = run_optimal_contextual(env, bandit_algo, bandit_ins, feature_map, args)
else:
    epoch_record = run_basic(env, bandit_algo, bandit_ins, args)  

record_name = f'{bandit_algo.__class__.__name__}_{time.time()}'
with open(f'output/{record_name}.json', 'w') as f:
    json.dump(epoch_record, f)

plt.plot(epoch_record['epoch'], epoch_record['cumu_recall'])
plt.savefig(f'{record_name}_plt.jpg')