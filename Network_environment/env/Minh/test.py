import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
from SinglePacketRoutingEnv import SinglePacketRoutingEnv
from DijkstraAgent import DijkstraAgent


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    nodes2 = ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15",
              "n16",
              "n17", "n18", "n19", "n20", "n21", "n22", "n23", "n24", "n25"]
    links2 = [["n15", "n16", 1], ["n15", "n17", 1], ["n15", "n13", 10], ["n14", "n13", 1], ["n15", "n18", 10],
              ["n13", "n7", 10], ["n10", "n7", 1], ["n9", "n7", 1], ["n11", "n7", 1], ["n8", "n7", 1], ["n7", "n12", 1],
              ["n7", "n18", 1], ["n18", "n19", 1], ["n7", "n4", 1], ["n4", "n5", 1], ["n4", "n6", 1], ["n4", "n3", 10],
              ["n1", "n3", 1], ["n4", "n20", 1], ["n1", "n2", 1], ["n20", "n21", 1], ["n20", "n22", 1],
              ["n20", "n23", 1],
              ["n20", "n24", 1], ["n20", "n25", 1], ["n2", "n3", 10], ["n20", "n2", 10], ["n0", "n2", 1],
              ["n18", "n20", 1]]

    env = SinglePacketRoutingEnv(nodes=nodes2, edges=links2)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    print("Learning")
    model = ACKTR(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=5000)

    print("Predicting")
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        reward += rewards
        env.render()
    print(reward)
    #dijkstra_agent = DijkstraAgent(env=env, nodes=nodes2, edges=links2)