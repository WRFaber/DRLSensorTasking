from collections import deque
import numpy as np
import unittest
import random


class TestDeque(unittest.TestCase):
    def setUp(self) -> None:
        self.que = deque(maxlen=10)
        self.index = False

    def test_building_simple_que(self):
        while self.index <= 12:
            self.que.append(1)
            self.index += 1
        queTwo = self.que
        queTwo.append(2)
        self.assertEqual(
            len(self.que), 10, "Queue length does not meet max length requirement"
        )
        self.assertEqual(len(queTwo), 10, "Queue does not append appropriately")

    def test_complex_deque(self):
        self.que.append((1, 1, 1))
        while self.index <= 12:
            last = len(self.que)
            newElement1 = self.que[last - 1][0] + 1
            newElement2 = self.que[last - 1][1] + 1
            newElement3 = self.que[last - 1][2] + 1
            self.que.append((newElement1, newElement2, newElement3))
            self.index += 1
        samp = random.sample(self.que, 4)
        # indeces = random.sample(range(0, len(self.que) - 1), 5)
        # for i in indeces:
        #     newx = self.que[i][0]
        #     newy = self.que[i][1]
        # newque = []
        # for transition in samp:
        #     newque.append(transition[0])
        self.assertEqual(
            len(self.que), 10, "Queue length does not meet max length requirement"
        )
        self.assertEqual(
            len(samp), 4, "Queue length does not meet max length requirement"
        )
