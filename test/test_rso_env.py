import unittest
from DRLSensorTasking.rso_env import RSOEnv
import numpy as np

RSOS = 10
ERROR_PENALTY = 1
SLEW_PENALTY = 1
EFFORT_PENALTY = 3
TASKING_PERIOD = 10
ACTIONS = RSOS + 1


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = RSOEnv(10, 1, 1, 3, 10, 7)
        self.done = False

    def test_taking_step(self):
        action = 3
        current_state = np.random.random_integers(0, 12, (1, 10))
        finalState, reward, done = self.env.step(current_state, action)
        self.assertEqual(
            finalState[0][3], 0.0, "Reward is not consitent with step action"
        )
        self.assertEqual(done, False, "Reward is not consitent with step action")

    def test_full_episode(self):
        current_state = np.random.random_integers(0, 12, (1, 10))
        print(len(current_state))
        state_memory = current_state.reshape(1, 10)
        while self.done == False:
            # q = np.random.random_integers(0, 10, 11)
            # print(len(q))
            action = np.random.randint(RSOS)
            current_state, reward, done = self.env.step(current_state, action)
            print(current_state)
            state_memory = np.vstack([state_memory, current_state.reshape(1, 10)])
            plt = self.env.render(state_memory)
            plt.show()
            self.done = done
        self.assertEqual(
            self.env.current_time, 10, "Reward is not consitent with step action"
        )

    def test_imshow(self):
        current_state = np.random.random_integers(0, 10, 10)
        print(current_state)
        plt = self.env.render(current_state.reshape(1, 10))
        plt.show()
