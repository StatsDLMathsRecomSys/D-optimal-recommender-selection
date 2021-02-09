from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional, Union, cast
import random 
from argparse import Namespace

import numpy as np # type: ignore
import pandas as pd # type: ignore

class EpochIter:
    def __init__(self, env: Environment, shuffle: bool):
        self.env = env
        self.user_lists = list(env.user_full_hist_dict.keys())
        self.ptr = 0
        self.shuffle = shuffle

    def __next__(self) -> Tuple[int, List[int]]:
        if self.ptr < len(self.user_lists):
            user = self.user_lists[self.ptr]
            self.ptr += 1
        else:
            self.env._update_recall()
            raise StopIteration()
        return user, self.env.user_recall_dict[user]

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.user_lists)
        return self

    def __len__(self):
        return len(self.user_lists)

class Environment:
    def __init__(self, user_num: int, item_num: int, data_df: pd.DataFrame, init_ratio: float = 0.05):
        # sample initial data and build user history profile.
        self.user_num = user_num
        self.item_num = item_num
        self.item_candidate = list(range(item_num))
        self.data_df = data_df
        self.user_full_hist_dict = data_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
        self.curr_df = data_df.sample(n=int(data_df.shape[0] * 0.05))
        self.init_test_relevant_size = self.data_df[self.data_df.rating > 0].shape[0] - self.curr_df.shape[0]

        self.user_curr_hist_dict = self.curr_df.groupby('uidx').apply(lambda x: set(x.iidx)).to_dict()
        self.user_curr_reject_dict: Dict[int, Set[int]] = {}

        self.rating_dict = {(uidx, iidx):rating for uidx, iidx, rating in zip(data_df.uidx, data_df.iidx, data_df.rating)}
        assert(len(self.rating_dict) == len(data_df))

        print(f'avg_user_hist_length: {np.mean([len(v) for v in self.user_full_hist_dict.values()])}')
        print(f'avg_user_init_hist_length: {np.mean([len(v) for v in self.user_curr_hist_dict.values()])}')

        print(self.curr_df.groupby('uidx').count().iidx.mean())
        print('build initial recall set')
        self.user_recall_dict: Dict[int, List[int]] = {}
        for uidx in self.user_full_hist_dict.keys():
            self.user_recall_dict[uidx] = self.item_candidate.copy()
        print('--initial filter')
        for uidx in self.user_recall_dict.keys():
            past_set = self.user_curr_hist_dict.get(uidx, set())
            self.user_recall_dict[uidx] = [x for x in self.user_recall_dict[uidx] if x not in past_set]

    def get_epoch(self, shuffle: bool = True):
        return EpochIter(self, shuffle)

    def action(self, uidx: int, recommendations: List[int]) -> float:
        num_match = 0
        for item in recommendations:
            assert(item not in self.user_curr_hist_dict.get(uidx, []))
            assert(item not in self.user_curr_reject_dict.get(uidx, []))
            if item in self.user_full_hist_dict[uidx]:
                # mark as positive if the user has positive feedbacks in the test and add to history
                num_match += self.rating_dict[(uidx, item)]
                if uidx not in self.user_curr_hist_dict:
                    self.user_curr_hist_dict[uidx] = set()
                self.user_curr_hist_dict[uidx].add(item)
            else:
                # include to the user's reject history so that dont recommend it again
                if uidx not in self.user_curr_reject_dict:
                    self.user_curr_reject_dict[uidx] = set()
                self.user_curr_reject_dict[uidx].add(item)
        reward = float(num_match > 0)
        return reward

    def _update_recall(self):
        #update the user recall set
        print('filter recall at the end of epoch')
        for uidx in self.user_recall_dict.keys():
            reject_set = self.user_curr_reject_dict.get(uidx, set())
            past_set = self.user_curr_hist_dict.get(uidx, set())
            self.user_recall_dict[uidx] = [x for x in self.user_recall_dict[uidx] if x not in reject_set and x not in past_set]
        avg_recall_length = np.mean([len(v) for v in self.user_recall_dict.values()])

    def get_train(self):
        #re-build the data-frame
        u_list, i_list = [], []
        for uidx, item_set in self.user_curr_hist_dict.items():
            u_list.extend([uidx] * len(item_set))
            i_list.extend(list(item_set))
        new_df = pd.DataFrame({'uidx': u_list, 'iidx': i_list, 
        'rating': np.ones(len(u_list)), 'ts': np.ones(len(u_list))})
        return new_df
