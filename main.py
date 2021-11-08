from Network import Network
from Request import Request
from Service import Service
from Functions import specify_requests_entry_nodes, assign_requests_to_services
from CPLEX import CPLEX

seed = 4
net_obj = Network(9, seed)
req_obj = Request(4, seed)
srv_obj = Service(3, seed)
seed = 0
requests_entry_nodes = specify_requests_entry_nodes(net_obj, req_obj, seed)
seed = 1
requested_services = assign_requests_to_services(srv_obj, req_obj, seed)

model_obj = CPLEX(net_obj, req_obj, srv_obj, requested_services, requests_entry_nodes)

optimum_value = model_obj.solve({})
action = {"req_id": 1, "node": 8}
action_value = model_obj.solve(action)

if optimum_value == -1 or action_value == -1:
    reward = 0
else:
    reward = (1 - ((action_value - optimum_value) / action_value)) * 10000

print(int(reward))






