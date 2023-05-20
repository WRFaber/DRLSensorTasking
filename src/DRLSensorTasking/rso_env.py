import numpy as np
import matplotlib.pyplot as plt


class RSOEnv:
    def __init__(
        self,
        size,
        error_penalty,
        slew_penalty,
        effort_penalty,
        tasking_period,
        initial_error_max,
    ):
        self.size = size
        self.error_penalty = error_penalty
        self.slew_penalty = slew_penalty
        self.effort_penalty = effort_penalty
        self.tasking_period = tasking_period
        self.intial_error_max = initial_error_max
        self.initial_state = np.random.random_integers(
            0, self.intial_error_max, (1, self.size)
        )
        self.initial_time = 0

        self.current_time = self.initial_time

    def action(self, q):
        return np.where(q == max(q))

    def render(self, evolving_env):
        plt.imshow(evolving_env, cmap="Blues", interpolation="nearest")
        return plt

    def reset(self):
        self.initial_state = np.random.random_integers(
            0, self.intial_error_max, (1, self.size)
        )
        self.initial_time = 0

        self.current_time = self.initial_time
        return self.initial_state

    def system_entropy(self, current_state):
        return np.sum(current_state)

    def step(self, current_state, action):
        posterior_state = current_state + self.error_penalty
        # Complicate reward to include cool things like minimizing effort based on how far the sensor has to look
        # or other things that make it more interesting than just a simple greedy algorithm
        if action + 1 > np.size(current_state):
            reward = 0
        else:
            previous_task = np.where(current_state == 0)
            if previous_task[0].size == 0:
                slew = 0
            else:
                slew = abs(previous_task[1][0] - action)
            reward = (
                current_state[0][action]
                - self.effort_penalty
                - slew * self.slew_penalty
            )
            posterior_state[0][action] = 0

        self.current_time += 1
        if self.current_time == self.tasking_period:
            done = True
        else:
            done = False

        return posterior_state, reward, done
