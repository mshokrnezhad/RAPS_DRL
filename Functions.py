# from docplex.mp.model import Model
import numpy as np

rnd = np.random


def specify_requests_entry_nodes(net_obj, req_obj, seed):
    rnd.seed(seed)
    return np.array([rnd.choice(net_obj.FIRST_TIER_NODES) for i in req_obj.REQUESTS])


def assign_requests_to_services(srv_obj, req_obj, seed):
    rnd.seed(seed)
    return np.array([rnd.choice(srv_obj.SERVICES) for i in req_obj.REQUESTS])


def parse_state(state, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, env_obj):

    print("selected request array:")
    print(state[0:NUM_REQUESTS].astype(int))




    print(state)


"""
def solver(net_obj, req_obj, srv_obj, requested_services, requests_entry_nodes):
    M = 10 ** 6
    OF_WEIGHTS = [1, 100, 1, 1]
    Z = [(s, j) for s in srv_obj.SERVICES for j in net_obj.NODES]
    G = [(i, j) for i in req_obj.REQUESTS for j in net_obj.NODES]
    F = [(i, (j, m)) for i in req_obj.REQUESTS for (j, m) in net_obj.LINKS_LIST]
    P = [(i, n) for i in req_obj.REQUESTS for n in net_obj.PRIORITIES]
    PP = [(i, (j, m), n) for i in req_obj.REQUESTS
          for (j, m) in net_obj.LINKS_LIST for n in net_obj.PRIORITIES]
    PL = [((j, m), n) for (j, m) in net_obj.LINKS_LIST for n in net_obj.PRIORITIES]
    LINK_DELAYS = net_obj.LINK_DELAYS_DICT
    epsilon = 0.001

    # print("Solving the problem is started...")
    mdl = Model('RASP')
    z = mdl.binary_var_dict(Z, name='z')
    g = mdl.binary_var_dict(G, name='g')
    g_sup = mdl.binary_var_dict(G, name='g_sup')
    req_flw = mdl.continuous_var_dict(PP, lb=0, name='req_flw')
    res_flw = mdl.continuous_var_dict(PP, lb=0, name='res_flw')
    flw = mdl.continuous_var_dict([(j, m) for j in net_obj.NODES for m in net_obj.NODES
                                   if (j, m) in net_obj.LINKS_LIST], lb=0, name='flw')
    p = mdl.binary_var_dict(P, name='p')
    req_pp = mdl.binary_var_dict(PP, name='req_pp')
    res_pp = mdl.binary_var_dict(PP, name='res_pp')
    req_d = mdl.continuous_var_dict([i for i in req_obj.REQUESTS], name='req_d')
    res_d = mdl.continuous_var_dict([i for i in req_obj.REQUESTS], name='res_d')
    d = mdl.continuous_var_dict([i for i in req_obj.REQUESTS], name='d')

    mdl.minimize(mdl.sum(z[s, j] for s, j in Z) * OF_WEIGHTS[0] +
                 mdl.sum(mdl.sum(g[i, j] for i in req_obj.REQUESTS) * net_obj.DC_COSTS[j]
                         for j in net_obj.NODES) * OF_WEIGHTS[1] +
                 mdl.sum(mdl.sum(flw[j, m]) * net_obj.LINK_COSTS_DICT[j, m]
                         for (j, m) in net_obj.LINKS_LIST) * OF_WEIGHTS[2] +
                 mdl.sum(d[i] for i in req_obj.REQUESTS) * OF_WEIGHTS[3])

    mdl.add_constraints(mdl.sum(g[i, j] for j in net_obj.NODES) == 1 for i in req_obj.REQUESTS)
    mdl.add_constraints(g[i, j] <= z[s, j] for i in req_obj.REQUESTS for j in net_obj.NODES
                        for s in srv_obj.SERVICES if s == requested_services[i])
    mdl.add_constraints(mdl.sum(g[i, j] * req_obj.CAPACITY_REQUIREMENTS[i]
                                for i in req_obj.REQUESTS) <= net_obj.DC_CAPACITIES[j] for j in net_obj.NODES)
    mdl.add_constraints(g_sup[i, j] == 1 - g[i, j]
                        for i in req_obj.REQUESTS for j in net_obj.NODES)

    mdl.add_constraints(mdl.sum(p[i, n] for n in net_obj.PRIORITIES) == 1 for i in req_obj.REQUESTS)

    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in net_obj.PRIORITIES
                                for j in net_obj.NODES for m in net_obj.NODES
                                if j == requests_entry_nodes[i] and (j, m) in net_obj.LINKS_LIST)
                        >= req_obj.BW_REQUIREMENTS[i] for i in req_obj.REQUESTS)
    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in net_obj.PRIORITIES
                                for m in net_obj.NODES for j in net_obj.NODES
                                if j != requests_entry_nodes[i] and (j, m) in net_obj.LINKS_LIST)
                        >= 0 for i in req_obj.REQUESTS)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g[i, j], mdl.sum(req_flw[i, (m, j), n]
                                                                            for n in net_obj.PRIORITIES
                                                                            for m in net_obj.NODES
                                                                            if (m, j) in net_obj.LINKS_LIST)
                                                           >= req_obj.BW_REQUIREMENTS[i])
                                  for i in req_obj.REQUESTS for j in net_obj.NODES)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                           mdl.sum(req_flw[i, (m, j), n]
                                                                   for n in net_obj.PRIORITIES
                                                                   for m in net_obj.NODES
                                                                   if (m, j) in net_obj.LINKS_LIST) >= 0)
                                  for i in req_obj.REQUESTS for j in net_obj.NODES)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j], mdl.sum(req_flw[i, (x, j), n]
                                                                                for n in net_obj.PRIORITIES
                                                                                for x in net_obj.NODES
                                                                                if (x, j) in net_obj.LINKS_LIST) ==
                                                           mdl.sum(req_flw[i, (j, m), n] for n in net_obj.PRIORITIES
                                                                   for m in net_obj.NODES
                                                                   if (j, m) in net_obj.LINKS_LIST))
                                  for i in req_obj.REQUESTS for j in net_obj.NODES if j != requests_entry_nodes[i])
    mdl.add_constraints(req_flw[i, (j, m), n] <= p[i, n] * M for i in req_obj.REQUESTS for n in net_obj.PRIORITIES
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)

    mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in net_obj.PRIORITIES
                                for m in net_obj.NODES for j in net_obj.NODES
                                if j == requests_entry_nodes[i] if (m, j) in net_obj.LINKS_LIST)
                        >= req_obj.BW_REQUIREMENTS[i] for i in req_obj.REQUESTS)
    mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in net_obj.PRIORITIES
                                for m in net_obj.NODES for j in net_obj.NODES
                                if j != requests_entry_nodes[i] if (m, j) in net_obj.LINKS_LIST)
                        >= 0 for i in req_obj.REQUESTS)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g[i, j],
                                                           mdl.sum(res_flw[i, (j, m), n]
                                                                   for n in net_obj.PRIORITIES
                                                                   for m in net_obj.NODES
                                                                   if (j, m) in net_obj.LINKS_LIST) >=
                                                           req_obj.BW_REQUIREMENTS[i])
                                  for i in req_obj.REQUESTS for j in net_obj.NODES)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                           mdl.sum(res_flw[i, (j, m), n]
                                                                   for n in net_obj.PRIORITIES
                                                                   for m in net_obj.NODES
                                                                   if (j, m) in net_obj.LINKS_LIST) >= 0)
                                  for i in req_obj.REQUESTS for j in net_obj.NODES)
    mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                           mdl.sum(res_flw[i, (j, x), n]
                                                                   for n in net_obj.PRIORITIES
                                                                   for x in net_obj.NODES
                                                                   if (j, x) in net_obj.LINKS_LIST) ==
                                                           mdl.sum(res_flw[i, (m, j), n]
                                                                   for n in net_obj.PRIORITIES
                                                                   for m in net_obj.NODES
                                                                   if (m, j) in net_obj.LINKS_LIST))
                                  for i in req_obj.REQUESTS for j in net_obj.NODES if j != requests_entry_nodes[i])
    mdl.add_constraints(res_flw[i, (j, m), n] <= p[i, n] * M for i in req_obj.REQUESTS for n in net_obj.PRIORITIES
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)

    mdl.add_constraints(flw[j, m] == mdl.sum(req_flw[i, (j, m), n] for n in net_obj.PRIORITIES
                                             for i in req_obj.REQUESTS) +
                        mdl.sum(res_flw[i, (j, m), n] for n in net_obj.PRIORITIES
                                for i in req_obj.REQUESTS) for j in net_obj.NODES
                        for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)
    mdl.add_constraints(flw[j, m] + flw[m, j] <= net_obj.LINK_BWS_DICT[j, m] for j in net_obj.NODES
                        for m in net_obj.NODES if j < m and (j, m) in net_obj.LINKS_LIST)

    mdl.add_constraints(req_pp[i, (j, m), n] >= req_flw[i, (j, m), n] / net_obj.LINK_BWS_DICT[j, m]
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST
                        for i in req_obj.REQUESTS for n in net_obj.PRIORITIES)
    mdl.add_constraints(req_pp[i, (j, m), n] <= req_flw[i, (j, m), n] for j in net_obj.NODES
                        for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST for i in req_obj.REQUESTS
                        for n in net_obj.PRIORITIES)

    mdl.add_constraints(res_pp[i, (j, m), n] >= res_flw[i, (j, m), n] / net_obj.LINK_BWS_DICT[j, m]
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST
                        for i in req_obj.REQUESTS for n in net_obj.PRIORITIES)
    mdl.add_constraints(res_pp[i, (j, m), n] <= res_flw[i, (j, m), n] for j in net_obj.NODES
                        for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST for i in req_obj.REQUESTS
                        for n in net_obj.PRIORITIES)

    mdl.add_constraints(req_d[i] == mdl.sum(mdl.sum(req_pp[i, (j, m), n] * LINK_DELAYS[(j, m), n]
                                                    for n in net_obj.PRIORITIES)
                                            for j in net_obj.NODES for m in net_obj.NODES
                                            if (j, m) in net_obj.LINKS_LIST) for i in req_obj.REQUESTS)
    mdl.add_constraints(res_d[i] == mdl.sum(mdl.sum(res_pp[i, (j, m), n] * LINK_DELAYS[(j, m), n]
                                                    for n in net_obj.PRIORITIES)
                                            for j in net_obj.NODES for m in net_obj.NODES
                                            if (j, m) in net_obj.LINKS_LIST) for i in req_obj.REQUESTS)
    mdl.add_constraints(d[i] == req_d[i] + res_d[i] + mdl.sum(g[i, j] * net_obj.PACKET_SIZE / (
            net_obj.DC_CAPACITIES[j] + epsilon) for j in net_obj.NODES) for i in req_obj.REQUESTS)

    mdl.add_constraints(mdl.sum((req_pp[i, (j, m), n]) * req_obj.BURST_SIZES[i] for i in req_obj.REQUESTS)
                        <= net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[n] for n in net_obj.PRIORITIES
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)
    mdl.add_constraints(mdl.sum((res_pp[i, (j, m), n]) * req_obj.BURST_SIZES[i] for i in req_obj.REQUESTS)
                        <= net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[n] for n in net_obj.PRIORITIES
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)
    mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] + res_flw[i, (j, m), n] + req_flw[i, (m, j), n] +
                                res_flw[i, (m, j), n] for i in req_obj.REQUESTS) <=
                        net_obj.LINK_BWS_LIMIT_PER_PRIORITY
                        [(j, m), n] for n in net_obj.PRIORITIES
                        for j in net_obj.NODES for m in net_obj.NODES if (j, m) in net_obj.LINKS_LIST)

    # mdl.parameters.timelimit = 60
    # mdl.log_output = True
    solution = mdl.solve()

    try:
        print(solution.get_objective_value())
    except:
        print("no solution is available!")
"""
