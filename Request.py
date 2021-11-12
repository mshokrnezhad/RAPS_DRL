import math
import matplotlib.pyplot as plt
import numpy as np
rnd = np.random


class Request:
    def __init__(self, NUM_REQUESTS, SEED=4, CAPACITY_REQUIREMENT_LB=5, CAPACITY_REQUIREMENT_UB=10,
                 BW_REQUIREMENT_LB=5, BW_REQUIREMENT_UB=10, DLY_REQUIREMENT_LB=10, DLY_REQUIREMENT_UB=20,
                 BURST_SIZE_LB=5, BURST_SIZE_UB=6):
        rnd.seed(SEED)
        self.NUM_REQUESTS = NUM_REQUESTS
        self.REQUESTS = np.arange(NUM_REQUESTS)
        self.CAPACITY_REQUIREMENT_LB = CAPACITY_REQUIREMENT_LB
        self.CAPACITY_REQUIREMENT_UB = CAPACITY_REQUIREMENT_UB
        self.BW_REQUIREMENT_LB = BW_REQUIREMENT_LB
        self.BW_REQUIREMENT_UB = BW_REQUIREMENT_UB
        self.DLY_REQUIREMENT_LB = DLY_REQUIREMENT_LB
        self.DLY_REQUIREMENT_UB = DLY_REQUIREMENT_UB
        self.BURST_SIZE_LB = BURST_SIZE_LB
        self.BURST_SIZE_UB = BURST_SIZE_UB
        self.CAPACITY_REQUIREMENTS = self.initialize_capacity_requirements()
        self.BW_REQUIREMENTS = self.initialize_bw_requirements()
        self.DELAY_REQUIREMENTS = self.initialize_delay_requirements()
        self.BURST_SIZES = self.initialize_burst_sizes()
        self.STATE = self.get_state()

    def initialize_capacity_requirements(self):
        capacity_requirements = np.array([rnd.randint(self.CAPACITY_REQUIREMENT_LB, self.CAPACITY_REQUIREMENT_UB)
                                          for i in self.REQUESTS])

        return capacity_requirements

    def initialize_bw_requirements(self):
        bw_requirements = np.array([rnd.randint(self.BW_REQUIREMENT_LB, self.BW_REQUIREMENT_UB)
                                    for i in self.REQUESTS])

        return bw_requirements

    def initialize_delay_requirements(self):
        delay_requirements = np.array([rnd.randint(self.DLY_REQUIREMENT_LB, self.DLY_REQUIREMENT_UB)
                                       for i in self.REQUESTS])

        return delay_requirements

    def initialize_burst_sizes(self):
        burst_sizes = np.array([rnd.randint(self.BURST_SIZE_LB, self.BURST_SIZE_UB) for i in self.REQUESTS])

        return burst_sizes

    def get_state(self):
        arr1 = np.concatenate((self.CAPACITY_REQUIREMENTS, self.BW_REQUIREMENTS))
        arr2 = np.concatenate((arr1, self.DELAY_REQUIREMENTS))
        arr3 = np.concatenate((arr2, self.BURST_SIZES))

        return arr3

    def update_state(self, action):
        self.REQUESTS = np.delete(self.REQUESTS, action["req_id"])
        self.CAPACITY_REQUIREMENTS[action["req_id"]] = 0
        self.BW_REQUIREMENTS[action["req_id"]] = 0
        self.DELAY_REQUIREMENTS[action["req_id"]] = 0
        self.BURST_SIZES[action["req_id"]] = 0