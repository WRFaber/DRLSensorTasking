from tqdm import tqdm
from DRLSensorTasking.dqn_agent import DQNAgent
from DRLSensorTasking.rso_env import RSOEnv
import numpy as np
import random

# Iterate over episodes

RSOS = 10
ERROR_PENALTY = 1
SLEW_PENALTY = 1
EFFORT_PENALTY = 3
TASKING_PERIOD = 10
ACTIONS = RSOS + 1  # Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001
MODEL_NAME = "SIMPLE_MLP"
MIN_ENTROPY = 25
RENDER = True
EPISODES = 1
# For stats

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# # Create models folder
# if not os.path.isdir("models"):
#     os.makedirs("models")

agent = DQNAgent(RSOS)
env = RSOEnv(RSOS, EFFORT_PENALTY, SLEW_PENALTY, EFFORT_PENALTY, TASKING_PERIOD)
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):

    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0

    # Reset environment and get initial state
    current_state, time = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    state_memory = current_state.reshape(1, RSOS)
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, ACTIONS - 1)

        new_state, reward, done = env.step(current_state, action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
        #     env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory(current_state, action, reward, new_state, done)
        agent.train(done, step)

        state_memory = np.vstack([state_memory, new_state.reshape(1, RSOS)])
        current_state = new_state

        # Append episode reward to a list and log stats (every given number of episodes)
        # ep_rewards.append(episode_reward)
        # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
        #         ep_rewards[-AGGREGATE_STATS_EVERY:]
        #     )
        #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        #     agent.tensorboard.update_stats(
        #         reward_avg=average_reward,
        #         reward_min=min_reward,
        #         reward_max=max_reward,
        #         epsilon=epsilon,
        #     )

    end_state_entropy = env.system_entropy(current_state)

    # Save model, but only when min reward is greater or equal a set value
    if end_state_entropy <= MIN_ENTROPY:
        agent.model.save(
            f"models/{MODEL_NAME}.model"  # __{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
        )

    if RENDER:
        env.render(state_memory)

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
