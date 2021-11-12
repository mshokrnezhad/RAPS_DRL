from Environment import Environment
from Functions import parse_state
from DNN import DNN
import numpy as np

NUM_NODES = 9
NUM_REQUESTS = 2
NUM_SERVICES = 3

env_obj = Environment(NUM_NODES, NUM_REQUESTS, NUM_SERVICES)
active_requests = env_obj.req_obj.REQUESTS
selected_request = np.random.choice(active_requests)
selected_request_array = np.array([1 if i == selected_request else 0 for i in range(NUM_REQUESTS)])
env_state = env_obj.get_state()
state = np.concatenate((selected_request_array, env_state))

parse_state(state, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, env_obj)

action = {"req_id": 0, "node_id": 5}
resulted_state, reward, done, info = env_obj.step(action)
# print(env_obj.req_obj.REQUESTS)
# print(reward)







