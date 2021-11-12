from Network import Network
from Request import Request
from Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services
from CPLEX import CPLEX
import numpy as np


class Environment:
    def __init__(self, NUM_NODES=9, NUM_REQUESTS=2, NUM_SERVICES=3):
        self.SEED = 4
        self.net_obj = Network(NUM_NODES, self.SEED)
        self.req_obj = Request(NUM_REQUESTS, self.SEED)
        self.srv_obj = Service(NUM_SERVICES, self.SEED)
        self.SEED = 0
        self.REQUESTS_ENTRY_NODES = specify_requests_entry_nodes(self.net_obj, self.req_obj, self.SEED)
        self.SEED = 1
        self.REQUESTED_SERVICES = assign_requests_to_services(self.srv_obj, self.req_obj, self.SEED)
        self.model_obj = CPLEX(self.net_obj, self.req_obj, self.srv_obj, self.REQUESTED_SERVICES,
                               self.REQUESTS_ENTRY_NODES)

    def get_state(self):
        net_state = self.net_obj.get_state()
        req_state = self.req_obj.get_state()
        env_state = np.concatenate((net_state, req_state))

        return env_state

    def step(self, action):
        result = self.model_obj.solve(action)
        optimum_result = self.model_obj.solve({})

        self.update_state(action, result)
        resulted_state = self.get_state()

        if result["done"]:
            reward = 0
        else:
            reward = (1 - ((result["OF"] - optimum_result["OF"]) / result["OF"])) * 10000

        return resulted_state, int(reward), result["done"], result["info"]

    def update_state(self, action, result):
        self.net_obj.update_state(action, result, self.req_obj)
        self.req_obj.update_state(action)
