import unittest
from DRLSensorTasking.dqn_agent import DQNAgent
import numpy as np
import tensorflow as tf

SIZE = 10


class TestDQNAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = DQNAgent(10, False, "")

    def test_dqn_input_size(self):
        self.assertEqual(
            self.agent.input_size, SIZE, "The input size is not set correctly"
        )

    def test_dqn_output_size(self):
        self.assertEqual(
            self.agent.output_size, SIZE + 1, "The input size is not set correctly"
        )

    def test_model_output(self):
        input = tf.ones((1, 10))
        print(input)
        output = self.agent.get_qs(input)
        self.assertEqual(np.size(output), 11, "The model is not functioning properly")

    def test_model_output_with_batch(self):
        val = np.ones((1, 10))
        for i in range(3):
            print(i)
            val = np.append(val, np.ones((1, 10)), axis=0)
        print(val)
        output = self.agent.get_qs(val)
        self.assertEqual(np.size(output), 44, "The model is not functioning properly")
