from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional, Union, cast
import random 


import pandas as pd # type: ignore
import numpy as np # type: ignore
from scipy import sparse as sp # type: ignore

from acgan.recommender import Recommender

BanditData = List[Tuple[int, float, Any]]

class Bandit:
    def __init__(self, recom_list: List[Recommender]):
        self.recom_list = recom_list
        self.k = len(self.recom_list)

    def get_arm(self, arm: int) -> Recommender:
        return self.recom_list[arm]

class BanditAlgorithm:
    def __init__(self, num_arms: int, rng: Optional[np.random.RandomState] = None):
        self.num_arms = num_arms
        if rng is None:
            self.rng = np.random.RandomState()
        self.total_reward_per_arm = np.zeros(num_arms)
        self.total_cnt_per_arm = np.zeros(num_arms)
        self.curr_cnt_per_arm = np.zeros(num_arms)
        self.avg_reward_per_arm = np.zeros(num_arms)
        self.arm_arr = np.arange(num_arms)

    def predict(self, *args, **kwds) -> int:
        raise NotImplementedError()

    def update(self, data: BanditData):
        raise NotImplementedError()

    def record_metric(self, data: BanditData):
        self.curr_cnt_per_arm *= 0
        for arm, reward, _ in data:
            self.total_reward_per_arm[arm] += reward
            self.total_cnt_per_arm[arm] += 1
            self.curr_cnt_per_arm[arm] += 1
        valid = self.total_cnt_per_arm > 0
        self.avg_reward_per_arm[valid] = self.total_reward_per_arm[valid] / self.total_cnt_per_arm[valid]


class RandomBandit(BanditAlgorithm):
    def __init__(self, num_arms: int):
        super(RandomBandit, self).__init__(num_arms)

    def update(self, data: BanditData):
        pass

    def predict(self):
        return int(self.rng.randint(0, self.num_arms))

class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, num_arms: int, epsilon: float):
        super(EpsilonGreedy, self).__init__(num_arms)
        self.epsilon = epsilon

    def update(self, data: BanditData):
        pass

    def predict(self):
        if self.rng.rand() <= self.epsilon:
            arm = self.rng.randint(0, self.num_arms)
        else:
            max_value = self.avg_reward_per_arm.max()
            best_arms = self.arm_arr[self.avg_reward_per_arm == max_value]
            if (best_arms).sum() > 1:
                #break ties randomly
                arm = best_arms[np.random.randint(0, len(best_arms))]
            else:
                arm = best_arms[0]
        return arm 


class ThomsonSampling(BanditAlgorithm):
    def __init__(self, num_arms: int):
        super(ThomsonSampling, self).__init__(num_arms)
        self.alpha = np.ones(num_arms) * 1000
        self.beta = np.ones(num_arms)

    def update(self, data: BanditData):
        for arm, reward, _ in data:
            if reward > 0:
                self.alpha[arm] += 1
            else:
                self.beta[arm] += 1

    def predict(self):
        theta = self.rng.beta(self.alpha, self.beta)
        max_value = theta.max()
        best_arms = self.arm_arr[theta == max_value]
        if (best_arms).sum() > 1:
            #break ties randomly
            arm = best_arms[np.random.randint(0, len(best_arms))]
        else:
            arm = best_arms[0]
        return arm


class Contextual(BanditAlgorithm):
    def __init__(self, num_arms: int, feat_dim: int, lambda_: float = 0.001, ucb_weight: float = 0.00):
        super(Contextual, self).__init__(num_arms)
        self.feat_dim = feat_dim
        self.ucb_weight = ucb_weight
        self.context_cov = np.zeros((feat_dim, feat_dim))
        self.cumu_feat = np.zeros(feat_dim)
        self.theta = np.zeros(feat_dim)
        self.inv_v = np.eye(feat_dim)
        self.lambda_I = lambda_ * np.eye(feat_dim)

    def predict(self, feature):
        rest = self.theta.dot(feature) 
        if self.ucb_weight > 0.0001:
            rest += self.ucb_weight * np.sqrt(feature.dot(self.inv_v).dot(feature))
        return rest

    def update(self, data: BanditData):
        for arm, reward, feature in data:
            self.cumu_feat += feature * reward
            self.context_cov += feature.reshape(-1, 1).dot(feature.reshape(1, -1))
        #self.theta = np.linalg.solve(self.lambda_I + self.context_cov, self.cumu_feat)
        self.inv_v = np.linalg.inv(self.lambda_I + self.context_cov)
        self.theta = self.inv_v.dot(self.cumu_feat)
        

class OptimalDesign(BanditAlgorithm):
    def __init__(self, num_arms: int, feat_dim: int, lambda_: float = 0.001, ucb_weight: float = 0.00):
        super(OptimalDesign, self).__init__(num_arms)
        self.feat_dim = feat_dim
        self.ucb_weight = ucb_weight
        self.lambda_I = lambda_ * np.eye(feat_dim)
        self.context_cov = np.zeros((feat_dim, feat_dim))
        self.cumu_feat = np.zeros(feat_dim)
        self.theta = np.random.rand(feat_dim)
        self.clear_buffer()

    def predict(self, feature):
        """Predict the reward for the arm
        """
        return self.theta.dot(feature) + self.ucb_weight * np.sqrt(feature.dot(self.inv_v).dot(feature))

    def update(self, data: BanditData):
        """Update prediction perameters
        """
        #self.theta = np.linalg.solve(self.lambda_I + self.context_cov, self.cumu_feat)
        for arm, reward, feature in data:
            self.cumu_feat += feature * reward
            self.context_cov += feature.reshape(-1, 1).dot(feature.reshape(1, -1))
        self.inv_v = np.linalg.inv(self.lambda_I + self.context_cov)
        self.theta = self.inv_v.dot(self.cumu_feat)

    def clear_buffer(self):
        feat_dim = self.feat_dim
        self.explore_context_cov = np.zeros((feat_dim, feat_dim))
        self.context_cov = np.zeros((feat_dim, feat_dim))
        self.cumu_feat = np.zeros(feat_dim)
        #self.inv_v = np.eye(feat_dim)
    
    def explore_decision(self, feat_mat: np.ndarray) -> int:
        """Find the arm that maximize the log(det(V(pi)))
        """
        det_v_arr = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            feature = feat_mat[arm, :]
            det_v_arr[arm] = np.linalg.det(self.lambda_I + self.explore_context_cov + feature.reshape(-1, 1).dot(feature.reshape(1, -1)))
        
        arm = det_v_arr.argmax()
        feature = feat_mat[arm, :]
        self.explore_context_cov += feature.reshape(-1, 1).dot(feature.reshape(1, -1))
        return arm