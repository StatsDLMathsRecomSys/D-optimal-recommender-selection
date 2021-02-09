from __future__ import annotations

from typing import Dict, Any, List, Set, Tuple, Optional, Union, cast
import random 
import numpy as np # type: ignore
import pandas as pd # type: ignore

class FeatureMap:
    def __init__(self) -> None:
        self.dim = -1
    def __call__(self, arm:int, user:int, recommendations: List[int], ) -> np.ndarray:
        raise NotImplementedError()


class ArmOneHot(FeatureMap):
    def __init__(self, num_arm: int) -> None:
        self.num_arm = num_arm
        self._cand_slot = np.eye(num_arm)
        self.dim = num_arm

    def __call__(self, arm:int, user:int, recommendations: List[int], ) -> np.ndarray:
        return self._cand_slot[arm, :]

class ConcatContext(FeatureMap):
    def __init__(self, num_arms: int, user_embed: np.ndarray, item_embed: np.ndarray):
        self.num_arms = num_arms
        user_embed = user_embed[:, user_embed.std(0) > 0]
        item_embed = item_embed[:, item_embed.std(0) > 0]
        self.user_embed = (user_embed - user_embed.mean(0)) / user_embed.std(0)
        self.item_embed = (item_embed - item_embed.mean(0)) / item_embed.std(0)
        self.dim = self.item_embed.shape[1] + self.user_embed.shape[1]
        print(self.dim)

    def __call__(self, arm:int, user:int, recommendations: List[int], ) -> np.ndarray:
        # item_dim = self.item_embed.shape[1]
        if len(recommendations) > 1:
            recom_pool = self.item_embed[recommendations, :].sum(0) 
        else:
            recom_pool = self.item_embed[recommendations[0], :] 
        return np.concatenate([self.user_embed[user], recom_pool], axis=0)

class ArmedConcatContext(FeatureMap):
    def __init__(self, num_arms: int, user_embed: np.ndarray, item_embed: np.ndarray):
        self.num_arms = num_arms
        user_embed = user_embed[:, user_embed.std(0) > 0]
        item_embed = item_embed[:, item_embed.std(0) > 0]
        self.user_embed = (user_embed - user_embed.mean(0)) / user_embed.std(0)
        self.item_embed = (item_embed - item_embed.mean(0)) / item_embed.std(0)
        self.dim = num_arms * self.item_embed.shape[1] + self.user_embed.shape[1]
        print(self.dim)

    def __call__(self, arm:int, user:int, recommendations: List[int], ) -> np.ndarray:
        item_dim = self.item_embed.shape[1]
        if len(recommendations) > 1:
            recom_pool = self.item_embed[recommendations, :].sum(0) 
        else:
            recom_pool = self.item_embed[recommendations[0], :] 
        arm_anchor_feat = np.zeros(self.num_arms * item_dim)
        arm_anchor_feat[arm*item_dim:(arm + 1)*item_dim] = recom_pool
        return np.concatenate([self.user_embed[user], arm_anchor_feat], axis=0)

class ArmOneHotWithContext(FeatureMap):
    def __init__(self, num_arms: int, user_embed: np.ndarray, item_embed: np.ndarray):
        self.concat_context = ConcatContext(num_arms, user_embed, item_embed)
        self.arm_onehot = ArmOneHot(num_arms)
        self.dim = self.concat_context.dim + self.arm_onehot.dim

    def __call__(self, arm:int, user:int, recommendations: List[int], ) -> np.ndarray:
        f1 = self.concat_context(arm, user, recommendations)
        f2 = self.arm_onehot(arm, user, recommendations)
        return np.concatenate([f1, f2], axis=0)
