import unittest
from DRLSensorTasking.dqn_agent import DQNAgent
from DRLSensorTasking.rso_env import RSOEnv
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import csv
import tensorflow as tf

RSOS = 10
ERROR_PENALTY = 1
SLEW_PENALTY = 1
EFFORT_PENALTY = 1
TASKING_PERIOD = 10
INITIAL_ERROR_MAX = 12
ACTIONS = RSOS + 1  # Exploration settings
EPSILON_DECAY = 0.99955
MIN_EPSILON = 0.001
MODEL_NAME = "SIMPLE_MLP"
MIN_ENTROPY = 45
RENDER = False
EPISODES = 10 # Will break under 10
INITIAL_EPSILON = 0.80
INFORMATION_GAIN = 100
LOAD_MODELS = False
FILE = "models/SIMPLE_MLP.model"
# For stats


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.epsilon = INITIAL_EPSILON
        self.best_information_gain = INFORMATION_GAIN
        self.agent = DQNAgent(RSOS, LOAD_MODELS, FILE)
        self.env = RSOEnv(
            RSOS,
            EFFORT_PENALTY,
            SLEW_PENALTY,
            EFFORT_PENALTY,
            TASKING_PERIOD,
            INITIAL_ERROR_MAX,
        )
        self.rolling_ave = 100
        self.rolling_ave_memory = np.array(self.rolling_ave)

    def test_simple_dqn(self):
        for episode in range(1, EPISODES + 1):
            # # For more repetitive results
            # random.seed(1)
            # np.random.seed(1)

            # Memory fraction, used mostly when training multiple agents
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
            # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

            # # Create models folder
            # if not os.path.isdir("models"):
            #     os.makedirs("models")

            # Update tensorboard step every episode
            # agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            # Reset environment and get initial state
            current_state = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            state_memory = current_state.reshape(1, RSOS)
            initial_state_entropy = self.env.system_entropy(current_state)
            stopwatch = time.time()

            while not done:
                biasedCoinFlip = np.random.random()
                # print(
                #     "Flipping a biased coin to see if we follow the model or a random selction"
                # )
                # print(
                #     f"If tails i.e. {biasedCoinFlip} is less than {epsilon} we go with random "
                # )
                # This part stays mostly the same, the change is to query a model for Q values
                if biasedCoinFlip > self.epsilon:
                    # Get action from Q table
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, ACTIONS - 1)

                new_state, reward, done = self.env.step(current_state, action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                #     env.render()

                # Every step we update replay memory and train main network
                self.agent.update_replay_memory(
                    current_state, action, reward, new_state, done
                )
                self.agent.train(done)

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

            end_state_entropy = self.env.system_entropy(current_state)
            entropy_added = end_state_entropy - initial_state_entropy

            print(f"The current epsilon {self.epsilon}")
            print(f"The initial state entropy {initial_state_entropy}")
            print(f"The end state entropy {end_state_entropy}")
            print(f"The total entropy growth {entropy_added}")
            print(
                f"Percent complete..... {episode/EPISODES*100}% and latest episode took {time.time()-stopwatch}"
            )
            if episode == 1:
                entropy_memory = np.array(entropy_added)
            else:
                entropy_memory = np.append(entropy_memory, entropy_added)

            if episode % 10 == 0:
                rolling_ave_array = entropy_memory[-10:]
                rolling_ave = np.sum(rolling_ave_array) / np.size(rolling_ave_array)
                self.rolling_ave_memory = np.append(
                    self.rolling_ave_memory, rolling_ave
                )

            # Save model, but only when min reward is greater or equal a set value
            if self.epsilon < 0.30 and rolling_ave < self.rolling_ave:
                self.agent.model.save(
                    f"models/{MODEL_NAME}.model"  # __{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
                )
                self.rolling_ave = rolling_ave

            if episode % 10 == 0 and RENDER:
                pllt = self.env.render(state_memory)
                pllt.colorbar()
                pllt.show()

            if episode % 1000 == 0 and RENDER:
                self.agent.model.save(
                    f"models/{MODEL_NAME}_{episode}.model"  # __{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
                )

            # Decay epsilon
            if self.epsilon > MIN_EPSILON:
                self.epsilon *= EPSILON_DECAY
                self.epsilon = max(MIN_EPSILON, self.epsilon)
        np.savetxt("entropy.csv", entropy_memory, delimiter=",")
        np.savetxt("rolling_ave.csv", self.rolling_ave_memory, delimiter=",")

        # Simple test criteria to get it running. You can make this more complicated.
        self.assertEqual(
            self.agent.input_size, RSOS, "The input size is not set correctly"
        )

    def test_results_ave(self):
        with open("rolling_Ave.csv", newline="") as f:
            reader = csv.reader(f)
            y = []
            x = []
            index = 1
            for r in reader:
                x.append(index)
                y.append(float(r[0]))
                index += 1
        plt.plot(x[:-2], y[1:-1])
        plt.axvline(x=450, label="line at epsilon = .11", c="black")
        plt.xlabel("Episode (10s)", fontsize=12)
        plt.ylabel("Entropy Levels Added", fontsize=12)
        plt.title("Rolling Average Entropy Growth", fontsize=12, fontweight="bold")
        plt.grid()
        plt.show()
        self.assertEqual(len(x), len(y), "The input size is not set correctly")

    def test_results_entropy(self):
        with open("entropy.csv", newline="") as f:
            reader = csv.reader(f)
            y = []
            x = []
            index = 1
            for r in reader:
                x.append(index)
                y.append(float(r[0]))
                index += 1
        plt.plot(x[:-2], y[1:-1])
        plt.axvline(x=450, label="line at epsilon = .11", c="red")
        plt.xlabel("Episode (10s)", fontsize=12)
        plt.ylabel("Rolling Average Entropy Growth", fontsize=12)
        plt.title("Simple DNN", fontsize=12, fontweight="bold")
        plt.grid()
        plt.show()
        self.assertEqual(len(x), len(y), "The input size is not set correctly")

    def test_random_dqn(self):
        # random.seed(1)
        # np.random.seed(1)

        # Restarting episode - reset episode reward and step number
        episode_reward = 0

        # Reset environment and get initial state
        current_state = self.env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        state_memory = current_state.reshape(1, RSOS)
        initial_state_entropy = self.env.system_entropy(current_state)

        while not done:
            biasedCoinFlip = np.random.random()
            if biasedCoinFlip > 1:
                # Get action from Q table
                action = np.argmax(self.agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, ACTIONS - 1)

            new_state, reward, done = self.env.step(current_state, action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory and train main network
            self.agent.update_replay_memory(
                current_state, action, reward, new_state, done
            )
            self.agent.train(done)

            state_memory = np.vstack([state_memory, new_state.reshape(1, RSOS)])
            current_state = new_state

        end_state_entropy = self.env.system_entropy(current_state)
        entropy_added = end_state_entropy - initial_state_entropy

        print(f"The initial state entropy {initial_state_entropy}")
        print(f"The end state entropy {end_state_entropy}")
        print(f"The total entropy growth {entropy_added}")

        plt = self.env.render(state_memory)
        plt.colorbar()
        plt.xlabel("Objects", fontsize=10)
        plt.ylabel("Time", fontsize=10)
        plt.bar_label = "Uncertainty"
        plt.title("Single Night Random", fontsize=10, fontweight="bold")
        plt.show()
        self.assertEqual(
            self.agent.input_size, RSOS, "The input size is not set correctly"
        )

    # Note: Before running please change the LOAD_MODELS variable to true and point
    # the FILE variable to the model you would like to load
    def test_trained_dqn(self):
        for i in range(0, 10):
            # random.seed(1)
            # np.random.seed(1)

            # Restarting episode - reset episode reward and step number
            episode_reward = 0

            # Reset environment and get initial state
            current_state = self.env.reset()

            # Reset flag and start iterating until episode ends
            done = False
            state_memory = current_state.reshape(1, RSOS)
            initial_state_entropy = self.env.system_entropy(current_state)

            while not done:
                biasedCoinFlip = np.random.random()
                if biasedCoinFlip > 0.0:
                    # Get action from Q table
                    action = np.argmax(self.agent.get_qs(current_state))
                else:
                    # Get random action
                    action = np.random.randint(0, ACTIONS - 1)

                new_state, reward, done = self.env.step(current_state, action)

                # Transform new continous state to new discrete state and count reward
                episode_reward += reward

                # Every step we update replay memory and train main network
                self.agent.update_replay_memory(
                    current_state, action, reward, new_state, done
                )
                self.agent.train(done)

                state_memory = np.vstack([state_memory, new_state.reshape(1, RSOS)])
                current_state = new_state

            end_state_entropy = self.env.system_entropy(current_state)
            entropy_added = end_state_entropy - initial_state_entropy

            print(f"The initial state entropy {initial_state_entropy}")
            print(f"The end state entropy {end_state_entropy}")
            print(f"The total entropy growth {entropy_added}")

        plt = self.env.render(state_memory)
        plt.colorbar()
        plt.xlabel("Objects", fontsize=10)
        plt.ylabel("Time", fontsize=10)
        plt.bar_label = "Uncertainty"
        plt.title("Single Night Trained", fontsize=10, fontweight="bold")
        plt.show()
        self.assertEqual(
            self.agent.input_size, RSOS, "The input size is not set correctly"
        )

    # def test_plot_model(self):
    #     tf.keras.utils.plot_model(
    #         self.agent.model,
    #         to_file="model.png",
    #         show_shapes=False,
    #         show_dtype=False,
    #         show_layer_names=True,
    #         rankdir="TB",
    #         expand_nested=False,
    #         dpi=96,
    #     )
    #     self.assertEqual(0, 0, "QUICK CHECK")
