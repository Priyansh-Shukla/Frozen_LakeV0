### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from lake_envs import *

np.set_printoptions(precision=3)


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    v_temp = np.zeros(nS)
    value_function = np.zeros(nS)
    for i in range(1000):
        for s in range(nS):
            for prob, nex, rew, term in P[s][policy[s]]:
                v_temp[s] = v_temp[s] + rew + prob * gamma * value_function[nex]
        if np.max(abs(value_function - v_temp)) < tol:
            break
        value_function = np.copy(v_temp)
        v_temp = np.zeros(nS)
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    new_policy = np.zeros(nS, dtype='int')
    for s in range(nS):
        q = np.zeros(nA)
        for a in range(nA):
            for prob, nex, rew, term in P[s][a]:
                q[a] = q[a] + prob * (rew + gamma * value_from_policy[nex])
        new_policy[s] = np.argmax(q)
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    v_new = np.zeros(nS)
    pol_new = np.zeros(nS)

    num = 1

    for i in range(1000):
        v_new = policy_evaluation(P, nS, nA, policy, gamma)
        pol_new = policy_improvement(P, nS, nA, v_new, policy, gamma)
        val_history.append(v_new)

        if np.all(pol_new == policy):
            num = num - 1
            if num == 0:
                break

        policy = np.copy(pol_new)
        value_function = np.copy(v_new)

    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3, max_iteration=1000):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    for i in range(max_iteration):
        v_temp = np.zeros(nS)
        for s in range(nS):
            q = np.zeros(nA)
            for a in range(nA):
                for prob, nex, rew, term in P[s][a]:
                    q[a] = q[a] + prob*(rew + gamma*value_function[nex])
            policy[s] = np.argmax(q)
            v_temp[s] = np.max(q)
            val_history.append(v_temp)
        if np.max(abs(value_function-v_temp)) < tol:
            break
        value_function = np.copy(v_temp)

    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render();

    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


def show_val_history(val_history):

    val_his = np.array(val_history)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # test data
    iterations = len(val_his)
    print(val_his[:,14])
    itr = range(1, iterations + 1)
    states_num = range(16)

    # plot test data
    for i in range(16):
        ax.plot(itr, val_his[:,i], i)

    # make labels
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Value')
    ax.set_zlabel('State (0-15)')

    plt.show()


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # comment/uncomment these lines to switch between deterministic/stochastic environments

    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    #env = gym.make("Stochastic-4x4-FrozenLake-v0")

    val_history = []
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)
    show_val_history(val_history)


    val_history = []
    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
    show_val_history(val_history)

    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make("Stochastic-4x4-FrozenLake-v0")

    val_history = []
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)
    show_val_history(val_history)


    val_history = []
    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)
    show_val_history(val_history)
