from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional, Union, cast
import random 
from argparse import Namespace

import numpy as np # type: ignore
import pandas as pd # type: ignore

from uni_bandit.bandit import Bandit, BanditData, BanditAlgorithm, OptimalDesign
from uni_bandit.env import Environment
from uni_bandit.feature import FeatureMap


def run_basic(env: Environment, algo: BanditAlgorithm, bandit_ins: Bandit, args: Namespace): 
    num_epochs=args.num_epochs
    top_k=args.topk
    epoch_record: Dict[str, Any] = {'epoch':[], 'cumu_recall': []}
    epoch_record['recommenders'] = [x.__class__.__name__ for x in bandit_ins.recom_list]
    epoch_record['bandit_algorithm'] = algo.__class__.__name__

    for epoch in range(num_epochs):
        # train the recommendation models
        new_train_data = env.get_train()
        for recom in bandit_ins.recom_list:
            recom.fit(new_train_data)

        data: BanditData = []
        for uidx, recall_set in env.get_epoch():
            arm = algo.predict()
            recommendations = bandit_ins.get_arm(arm).recommend(uidx, recall_set, top_k)
            reward = env.action(uidx, recommendations)
            data.append((arm, reward, None))
        algo.update(data)
        algo.record_metric(data)

        cumu_recall = algo.total_reward_per_arm.sum() / env.init_test_relevant_size
        print(f'epoch: {epoch}, cumulative recall: {cumu_recall}, arm_cnt: {algo.total_cnt_per_arm}, curr_arm: {algo.curr_cnt_per_arm}')
        epoch_record['epoch'].append(epoch)
        epoch_record['cumu_recall'].append(cumu_recall)
    return epoch_record

def run_lin_contextual(env: Environment, algo: BanditAlgorithm, bandit_ins: Bandit, feat_map: FeatureMap , args: Namespace): 
    num_epochs=args.num_epochs
    top_k=args.topk
    epoch_record: Dict[str, Any] = {'epoch':[], 'cumu_recall': []}
    epoch_record['recommenders'] = [x.__class__.__name__ for x in bandit_ins.recom_list]
    epoch_record['bandit_algorithm'] = algo.__class__.__name__

    est_list = np.zeros(algo.num_arms)
    feat_mat = np.zeros((algo.num_arms, feat_map.dim))
    recommendations_mat: List[List[int]] = [[] for _ in range(algo.num_arms)]

    for epoch in range(num_epochs):
        # train the recommendation models
        new_train_data = env.get_train()
        for recom in bandit_ins.recom_list:
            recom.fit(new_train_data)

        data: BanditData = []
        for uidx, recall_set in env.get_epoch():
            for arm in range(algo.num_arms):
                recommendations_mat[arm] = bandit_ins.get_arm(arm).recommend(uidx, recall_set, top_k)
                feature = feat_map(arm, uidx, recommendations_mat[arm])
                est_list[arm] = algo.predict(feature)
                feat_mat[arm, :] = feature

            max_value = est_list.max()
            best_arms = algo.arm_arr[est_list == max_value]
            if (best_arms).sum() > 1:
                #break ties randomly
                arm = best_arms[np.random.randint(0, len(best_arms))]
            else:
                arm = best_arms[0]
            #recommendations = bandit.get_arm(arm).recommend(uidx, recall_set, top_k)
            reward = env.action(uidx, recommendations_mat[arm])
            data.append((arm, reward, feat_mat[arm, :]))
        algo.update(data)
        algo.record_metric(data)

        cumu_recall = algo.total_reward_per_arm.sum() / env.init_test_relevant_size
        print(f'epoch: {epoch}, cumulative recall: {cumu_recall}, arm_cnt: {algo.total_cnt_per_arm}, curr_arm: {algo.curr_cnt_per_arm}')
        epoch_record['epoch'].append(epoch)
        epoch_record['cumu_recall'].append(cumu_recall)
    return epoch_record


def run_optimal_contextual(env: Environment, algo: OptimalDesign, bandit_ins: Bandit, feat_map: FeatureMap, args: Namespace):
    num_epochs:int =args.num_epochs
    top_k: int=args.topk
    explore_step: int=args.explore_step
    epoch_record: Dict[str, Any] = {'epoch':[], 'cumu_recall': []}
    epoch_record['recommenders'] = [x.__class__.__name__ for x in bandit_ins.recom_list]
    epoch_record['bandit_algorithm'] = algo.__class__.__name__

    est_list = np.zeros(algo.num_arms)
    feat_mat = np.zeros((algo.num_arms, feat_map.dim))
    recommendations_mat: List[List[int]] = [[] for _ in range(algo.num_arms)]

    for epoch in range(num_epochs):
        # train the recommendation models using latest trainning data
        new_train_data = env.get_train()
        for recom in bandit_ins.recom_list:
            recom.fit(new_train_data)

        data: BanditData = []
        explore_data: BanditData = []
        explore_cnt = 0
        wait_explore_update = True

        #algo.clear_buffer()
        for uidx, recall_set in env.get_epoch():
            # update the theta if has done enough exploration
            if explore_cnt >= explore_step and wait_explore_update:
                algo.update(explore_data)
                wait_explore_update = False
            
            # get system output
            for arm in range(algo.num_arms):
                recommendations_mat[arm] = bandit_ins.get_arm(arm).recommend(uidx, recall_set, top_k)
                feat_mat[arm, :] = feat_map(arm, uidx, recommendations_mat[arm])

            # explore or exploit
            if explore_cnt < explore_step:
                #explore to maximize the information gain
                arm = algo.explore_decision(feat_mat)
            else:
                for arm in range(algo.num_arms):
                    est_list[arm] = algo.predict(feat_mat[arm, :])
                # interact with environment, add record
                max_value = est_list.max()
                best_arms = algo.arm_arr[est_list == max_value]
                if (best_arms).sum() > 1:
                    #break ties randomly
                    arm = best_arms[np.random.randint(0, len(best_arms))]
                else:
                    arm = best_arms[0]
            #recommendations = bandit.get_arm(arm).recommend(uidx, recall_set, top_k)
            reward = env.action(uidx, recommendations_mat[arm])
            data.append((arm, reward, feat_mat[arm, :]))

            if explore_cnt < explore_step:
                explore_data.append((arm, reward, feat_mat[arm, :]))
            explore_cnt += 1
        
        # update metric at the end epoch, no need to update parameters
        algo.record_metric(data)

        cumu_recall = algo.total_reward_per_arm.sum() / env.init_test_relevant_size
        print(f'epoch: {epoch}, cumulative recall: {cumu_recall}, arm_cnt: {algo.total_cnt_per_arm}, curr_arm: {algo.curr_cnt_per_arm}')
        epoch_record['epoch'].append(epoch)
        epoch_record['cumu_recall'].append(cumu_recall)
    return epoch_record