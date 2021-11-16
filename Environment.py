from Network import Network
from Request import Request
from Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services
from CPLEX import CPLEX
import numpy as np


class Environment:
    def __init__(self, NUM_NODES=9, NUM_REQUESTS=2, NUM_SERVICES=3):
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.net_obj = Network(NUM_NODES, 4)
        self.req_obj = Request(NUM_REQUESTS, 4)
        self.srv_obj = Service(NUM_SERVICES, 4)
        self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj, self.req_obj, 0)
        self.REQUESTED_SERVICES = assign_requests_to_services(self.srv_obj, self.req_obj, 1)
        self.model_obj = CPLEX(self.net_obj, self.req_obj, self.srv_obj, self.REQUESTED_SERVICES,
                               self.REQUESTS_ENTRY_NODES)

    def get_state(self):
        net_state = self.net_obj.get_state()
        req_state = self.req_obj.get_state()
        env_state = np.concatenate((req_state, net_state))

        return env_state

    def step(self, action):
        result = self.model_obj.solve(action)
        optimum_result = self.model_obj.solve({})

        if result["done"]:
            reward = 0
        else:
            reward = (1 - ((result["OF"] - optimum_result["OF"]) / result["OF"])) * 10000
            self.update_state(action, result)

        resulted_state = self.get_state()

        return resulted_state, int(reward), result["done"] or len(self.req_obj.REQUESTS) == 0, result["info"]

    def update_state(self, action, result):
        self.net_obj.update_state(action, result, self.req_obj)
        self.req_obj.update_state(action)

    def reset(self, SEED):
        self.req_obj = Request(self.NUM_REQUESTS, SEED)
        self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj, self.req_obj, SEED)
