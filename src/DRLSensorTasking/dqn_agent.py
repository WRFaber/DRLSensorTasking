import numpy as np
import tensorflow as tf
from tensorflow import keras

from collections import deque
import time
import random

MODEL_NAME = "SIMPLE_MLP"
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 1_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 12  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


class DQNAgent:
    def __init__(self, size, load, file):

        self.input_size = size

        self.output_size = size + 1

        if load:
            self.model = tf.keras.models.load_model(file)
        else:
            self.model = self.create_model()

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Target model allows me to retain a stable prediction without updating
        # the weights so I have less variation in the prediction while training
        # the target model will be updated periodically with the model weights
        # # from the model in training
        self.target_model = self.create_model()

        self.target_model.set_weights(self.model.get_weights())

        # # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(
        #     log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time()))
        # )
        # Replay Memory for retaining a semblance of batching in DQN
        self.replay_memory_state = deque(maxlen=100)

        self.replay_memory_action = deque(maxlen=100)

        self.replay_memory_reward = deque(maxlen=100)

        self.replay_memory_posterior = deque(maxlen=100)

        self.replay_memory_done = deque(maxlen=100)

    def create_model(self):
        model = tf.keras.models.Sequential(name=MODEL_NAME)
        model.add(
            tf.keras.layers.Dense(
                12, input_shape=(self.input_size,), activation="relu", batch_size=None
            )
        )
        model.add(tf.keras.layers.Dense(8, activation="relu"))
        model.add(tf.keras.layers.Dense(self.output_size, activation="relu"))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
        return model

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, input_state):
        return self.model.predict(input_state)

    # Adds step's data to a memory replay array
    # (input space, action, reward, new input space, done)
    def update_replay_memory(self, current_state, action, reward, posterior, done):
        self.replay_memory_state.append(current_state)
        self.replay_memory_action.append(action)
        self.replay_memory_reward.append(reward)
        self.replay_memory_posterior.append(posterior)
        self.replay_memory_done.append(done)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory_state) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch_index = random.sample(
            range(0, len(self.replay_memory_state) - 1), MINIBATCH_SIZE
        )

        # Now we need to enumerate our batches
        for index in minibatch_index:

            # Get current states from minibatch, then query NN model for Q values
            current_state = self.replay_memory_state[index]
            current_qs = self.model.predict(
                current_state
            )  # TODO check if this should be target model

            action = self.replay_memory_action[index]
            reward = self.replay_memory_reward[index]

            # Get future states from minibatch, then query NN model for Q values
            # When using target network, query it, otherwise main network should be queried
            new_current_state = self.replay_memory_posterior[index]
            future_qs = self.target_model.predict(new_current_state)

            done = self.replay_memory_done[index]

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here # TODO: Check this does not need some weird normalization
            if not done:
                max_future_q = np.max(future_qs)
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs[0][action] = new_q

            # And append to our training data
            if minibatch_index.index(index) == 0:
                X = current_state
                y = current_qs
            else:
                X = np.append(X, current_state, axis=0)
                y = np.append(y, current_qs, axis=0)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            X,
            y,
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
