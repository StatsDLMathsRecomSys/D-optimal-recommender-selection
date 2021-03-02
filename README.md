# Design of the bandit

## Running code

### Install packages

```bash
pip install -r requirements.txt 
```

### Recommender

We use the recommenders implemented under [our project for adversarial counterfactual learning](https://github.com/richardruancw/Adversarial-Counterfactual-Learning-and-Evaluation-for-Recommender-System) published in NIPS 2020. 
* Step 1: clone the project to your local directory.

* Step 2: `pip install .` to install the library. 

### Item features
The data `ml-1m.zip` is under the `data` folder. We need to generate the 
movies and users features before running the simulations.

```bash
cd data & unzip ml-1m.zip
cd ml-1m
python base_embed.py # This generates base movie and user features vector
```

### Simulation
Assume you are in the project's main folder:
```bash
python run.py #This will runs all defined simulation routines defined in simulation.py
```

Optional argument:
```bash
usage: System Bandit Simulation [-h] [--dim DIM] [--topk TOPK] [--num_epochs NUM_EPOCHS] [--epsilon EPSILON] [--explore_step EXPLORE_STEP] [--feat_map {onehot,context,armed_context,onehot_context}]
                                [--algo {base,e_greedy,thomson,lin_ct,optimal}]

optional arguments:
  -h, --help            show this help message and exit
  --dim DIM
  --topk TOPK
  --num_epochs NUM_EPOCHS
  --epsilon EPSILON
  --explore_step EXPLORE_STEP
  --feat_map {onehot,context,armed_context,onehot_context}
  --algo {base,e_greedy,thomson,lin_ct,optimal}

```

## Major class

### Environment 
This class implement the simulation logics described in our paper. 
For each user, we runs the `get_epoch` method, which returns an refreshed 
simulator based on the last interaction with the user.

Example:
```python
class Environment:
    def get_epoch(self, shuffle: bool = True):
        """Return updated environment iterator"""
        return EpochIter(self, shuffle)

    def action(self, uidx: int, recommendations: List[int]) -> float:
        """Return the reward given selected arm and the recommendations"""
        pass

# Example usage
BanditData = List[Tuple[int, float, Any]]
data: BanditData = []
for uidx, recall_set in env.get_epoch():
    arm = algo.predict()
    recommendations = bandit_ins.get_arm(arm).recommend(uidx, recall_set, top_k)
    reward = env.action(uidx, recommendations)
    data.append((arm, reward, None))
algo.update(data)
algo.record_metric(data) 
```

### BanditAlgorithm

The `BanditALgorithm` implement the interfaces for any bandit algorithms evaluated
in this project.

```python
class BanditAlgorithm:
    def predict(self, *args, **kwds) -> int:
        """Return the estimated return for contextual bandit"""
        pass

    def update(self, data: BanditData):
        """Update the algorithms based on observed (action, reward, context)"""
        pass

    def record_metric(self, data: BanditData):
        """Record the cumulative performance metrics for this algorithm"""
        pass
```