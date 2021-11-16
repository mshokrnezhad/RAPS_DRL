from docplex.mp.model import Model
import numpy as np


class CPLEX:

    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES
        self.M = 10 ** 6
        self.OF_WEIGHTS = [1, 100, 1, 1]
        self.Z = [(s, j) for s in srv_obj.SERVICES for j in net_obj.NODES]
        self.G = [(i, j) for i in req_obj.REQUESTS for j in net_obj.NODES]
        self.F = [(i, (j, m)) for i in req_obj.REQUESTS for (j, m) in net_obj.LINKS_LIST]
        self.P = [(i, n) for i in req_obj.REQUESTS for n in net_obj.PRIORITIES]
        self.PP = [(i, (j, m), n) for i in req_obj.REQUESTS
                   for (j, m) in net_obj.LINKS_LIST for n in net_obj.PRIORITIES]
        self.PL = [((j, m), n) for (j, m) in net_obj.LINKS_LIST for n in net_obj.PRIORITIES]
        self.LINK_DELAYS = net_obj.LINK_DELAYS_DICT
        self.EPSILON = 0.001

    def initialize_model(self):
        mdl = Model('RASP')
        z = mdl.binary_var_dict(self.Z, name='z')
        g = mdl.binary_var_dict(self.G, name='g')
        g_sup = mdl.binary_var_dict(self.G, name='g_sup')
        req_flw = mdl.continuous_var_dict(self.PP, lb=0, name='req_flw')
        res_flw = mdl.continuous_var_dict(self.PP, lb=0, name='res_flw')
        flw = mdl.continuous_var_dict([(j, m) for j in self.net_obj.NODES for m in self.net_obj.NODES
                                       if (j, m) in self.net_obj.LINKS_LIST], lb=0, name='flw')
        p = mdl.binary_var_dict(self.P, name='p')
        req_pp = mdl.binary_var_dict(self.PP, name='req_pp')
        res_pp = mdl.binary_var_dict(self.PP, name='res_pp')
        req_d = mdl.continuous_var_dict([i for i in self.req_obj.REQUESTS], name='req_d')
        res_d = mdl.continuous_var_dict([i for i in self.req_obj.REQUESTS], name='res_d')
        d = mdl.continuous_var_dict([i for i in self.req_obj.REQUESTS], name='d')

        return mdl, z, g, g_sup, req_flw, res_flw, flw, p, req_pp, res_pp, req_d, res_d, d

    def define_model(self, mdl, z, g, g_sup, req_flw, res_flw, flw, p, req_pp, res_pp, req_d, res_d, d):
        mdl.minimize(mdl.sum(z[s, j] for s, j in self.Z) * self.OF_WEIGHTS[0] +
                     mdl.sum(mdl.sum(g[i, j] for i in self.req_obj.REQUESTS) * self.net_obj.DC_COSTS[j]
                             for j in self.net_obj.NODES) * self.OF_WEIGHTS[1] +
                     mdl.sum(mdl.sum(flw[j, m]) * self.net_obj.LINK_COSTS_DICT[j, m]
                             for (j, m) in self.net_obj.LINKS_LIST) * self.OF_WEIGHTS[2] +
                     mdl.sum(d[i] for i in self.req_obj.REQUESTS) * self.OF_WEIGHTS[3])

        mdl.add_constraints(mdl.sum(g[i, j] for j in self.net_obj.NODES) == 1 for i in self.req_obj.REQUESTS)
        mdl.add_constraints(g[i, j] <= z[s, j] for i in self.req_obj.REQUESTS for j in self.net_obj.NODES
                            for s in self.srv_obj.SERVICES if s == self.REQUESTED_SERVICES[i])
        mdl.add_constraints(mdl.sum(g[i, j] * self.req_obj.CAPACITY_REQUIREMENTS[i]
                                    for i in self.req_obj.REQUESTS) <= self.net_obj.DC_CAPACITIES[j]
                            for j in self.net_obj.NODES)
        mdl.add_constraints(g_sup[i, j] == 1 - g[i, j]
                            for i in self.req_obj.REQUESTS for j in self.net_obj.NODES)

        mdl.add_constraints(mdl.sum(p[i, n] for n in self.net_obj.PRIORITIES) == 1 for i in self.req_obj.REQUESTS)

        mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in self.net_obj.PRIORITIES
                                    for j in self.net_obj.NODES for m in self.net_obj.NODES
                                    if j == self.REQUESTS_ENTRY_NODES[i] and (j, m) in self.net_obj.LINKS_LIST)
                            >= self.req_obj.BW_REQUIREMENTS[i] for i in self.req_obj.REQUESTS)
        mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] for n in self.net_obj.PRIORITIES
                                    for m in self.net_obj.NODES for j in self.net_obj.NODES
                                    if j != self.REQUESTS_ENTRY_NODES[i] and (j, m) in self.net_obj.LINKS_LIST)
                            >= 0 for i in self.req_obj.REQUESTS)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g[i, j], mdl.sum(req_flw[i, (m, j), n]
                                                                                for n in self.net_obj.PRIORITIES
                                                                                for m in self.net_obj.NODES
                                                                                if (m, j) in self.net_obj.LINKS_LIST)
                                                               >= self.req_obj.BW_REQUIREMENTS[i])
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                               mdl.sum(req_flw[i, (m, j), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for m in self.net_obj.NODES
                                                                       if (m, j) in self.net_obj.LINKS_LIST) >= 0)
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                               mdl.sum(req_flw[i, (x, j), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for x in self.net_obj.NODES
                                                                       if (x, j) in self.net_obj.LINKS_LIST) ==
                                                               mdl.sum(req_flw[i, (j, m), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for m in self.net_obj.NODES
                                                                       if (j, m) in self.net_obj.LINKS_LIST))
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES
                                      if j != self.REQUESTS_ENTRY_NODES[i])
        mdl.add_constraints(req_flw[i, (j, m), n] <= p[i, n] * self.M for i in self.req_obj.REQUESTS
                            for n in self.net_obj.PRIORITIES
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST)

        mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in self.net_obj.PRIORITIES
                                    for m in self.net_obj.NODES for j in self.net_obj.NODES
                                    if j == self.REQUESTS_ENTRY_NODES[i] if (m, j) in self.net_obj.LINKS_LIST)
                            >= self.req_obj.BW_REQUIREMENTS[i] for i in self.req_obj.REQUESTS)
        mdl.add_constraints(mdl.sum(res_flw[i, (m, j), n] for n in self.net_obj.PRIORITIES
                                    for m in self.net_obj.NODES for j in self.net_obj.NODES
                                    if j != self.REQUESTS_ENTRY_NODES[i] if (m, j) in self.net_obj.LINKS_LIST)
                            >= 0 for i in self.req_obj.REQUESTS)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g[i, j],
                                                               mdl.sum(res_flw[i, (j, m), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for m in self.net_obj.NODES
                                                                       if (j, m) in self.net_obj.LINKS_LIST) >=
                                                               self.req_obj.BW_REQUIREMENTS[i])
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                               mdl.sum(res_flw[i, (j, m), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for m in self.net_obj.NODES
                                                                       if (j, m) in self.net_obj.LINKS_LIST) >= 0)
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES)
        mdl.add_indicator_constraints(mdl.indicator_constraint(g_sup[i, j],
                                                               mdl.sum(res_flw[i, (j, x), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for x in self.net_obj.NODES
                                                                       if (j, x) in self.net_obj.LINKS_LIST) ==
                                                               mdl.sum(res_flw[i, (m, j), n]
                                                                       for n in self.net_obj.PRIORITIES
                                                                       for m in self.net_obj.NODES
                                                                       if (m, j) in self.net_obj.LINKS_LIST))
                                      for i in self.req_obj.REQUESTS for j in self.net_obj.NODES
                                      if j != self.REQUESTS_ENTRY_NODES[i])
        mdl.add_constraints(res_flw[i, (j, m), n] <= p[i, n] * self.M for i in self.req_obj.REQUESTS
                            for n in self.net_obj.PRIORITIES
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST)

        mdl.add_constraints(flw[j, m] == mdl.sum(req_flw[i, (j, m), n] for n in self.net_obj.PRIORITIES
                                                 for i in self.req_obj.REQUESTS) +
                            mdl.sum(res_flw[i, (j, m), n] for n in self.net_obj.PRIORITIES
                                    for i in self.req_obj.REQUESTS) for j in self.net_obj.NODES
                            for m in self.net_obj.NODES if (j, m) in self.net_obj.LINKS_LIST)
        mdl.add_constraints(flw[j, m] + flw[m, j] <= self.net_obj.LINK_BWS_DICT[j, m] for j in self.net_obj.NODES
                            for m in self.net_obj.NODES if j < m and (j, m) in self.net_obj.LINKS_LIST)

        mdl.add_constraints(req_pp[i, (j, m), n] >= req_flw[i, (j, m), n] / self.net_obj.LINK_BWS_DICT[j, m]
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST
                            for i in self.req_obj.REQUESTS for n in self.net_obj.PRIORITIES)
        mdl.add_constraints(req_pp[i, (j, m), n] <= req_flw[i, (j, m), n] for j in self.net_obj.NODES
                            for m in self.net_obj.NODES if (j, m) in self.net_obj.LINKS_LIST
                            for i in self.req_obj.REQUESTS
                            for n in self.net_obj.PRIORITIES)

        mdl.add_constraints(res_pp[i, (j, m), n] >= res_flw[i, (j, m), n] / self.net_obj.LINK_BWS_DICT[j, m]
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST
                            for i in self.req_obj.REQUESTS for n in self.net_obj.PRIORITIES)
        mdl.add_constraints(res_pp[i, (j, m), n] <= res_flw[i, (j, m), n] for j in self.net_obj.NODES
                            for m in self.net_obj.NODES if (j, m) in self.net_obj.LINKS_LIST
                            for i in self.req_obj.REQUESTS
                            for n in self.net_obj.PRIORITIES)

        mdl.add_constraints(req_d[i] == mdl.sum(mdl.sum(req_pp[i, (j, m), n] * self.LINK_DELAYS[(j, m), n]
                                                        for n in self.net_obj.PRIORITIES)
                                                for j in self.net_obj.NODES for m in self.net_obj.NODES
                                                if (j, m) in self.net_obj.LINKS_LIST) for i in self.req_obj.REQUESTS)
        mdl.add_constraints(res_d[i] == mdl.sum(mdl.sum(res_pp[i, (j, m), n] * self.LINK_DELAYS[(j, m), n]
                                                        for n in self.net_obj.PRIORITIES)
                                                for j in self.net_obj.NODES for m in self.net_obj.NODES
                                                if (j, m) in self.net_obj.LINKS_LIST) for i in self.req_obj.REQUESTS)
        mdl.add_constraints(d[i] == req_d[i] + res_d[i] + mdl.sum(g[i, j] * self.net_obj.PACKET_SIZE / (
                self.net_obj.DC_CAPACITIES[j] + self.EPSILON) for j in self.net_obj.NODES)
                            for i in self.req_obj.REQUESTS)

        mdl.add_constraints(mdl.sum((req_pp[i, (j, m), n]) * self.req_obj.BURST_SIZES[i]
                                    for i in self.req_obj.REQUESTS)
                            <= self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[n] for n in self.net_obj.PRIORITIES
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST)
        mdl.add_constraints(mdl.sum((res_pp[i, (j, m), n]) * self.req_obj.BURST_SIZES[i]
                                    for i in self.req_obj.REQUESTS)
                            <= self.net_obj.BURST_SIZE_LIMIT_PER_PRIORITY[n] for n in self.net_obj.PRIORITIES
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST)
        mdl.add_constraints(mdl.sum(req_flw[i, (j, m), n] + res_flw[i, (j, m), n] + req_flw[i, (m, j), n] +
                                    res_flw[i, (m, j), n] for i in self.req_obj.REQUESTS) <=
                            self.net_obj.LINK_BWS_LIMIT_PER_PRIORITY
                            [(j, m), n] for n in self.net_obj.PRIORITIES
                            for j in self.net_obj.NODES for m in self.net_obj.NODES
                            if (j, m) in self.net_obj.LINKS_LIST)

        return mdl

    def add_action_constraint(self, mdl, g, action):
        mdl.add_constraint(g[action["req_id"], action["node_id"]] == 1)

        return mdl

    def solve(self, action):
        mdl, z, g, g_sup, req_flw, res_flw, flw, p, req_pp, res_pp, req_d, res_d, d = self.initialize_model()
        mdl = self.define_model(mdl, z, g, g_sup, req_flw, res_flw, flw, p, req_pp, res_pp, req_d, res_d, d)
        if len(action) != 0:
            mdl = self.add_action_constraint(mdl, g, action)

        # mdl.parameters.timelimit = 60
        # mdl.log_output = True
        solution = mdl.solve()
        # print("\n")
        # print(solution)
        # print("\n")

        try:
            parsed_solution = {}
            status = str(mdl.solve_details.status)
            raw_p = np.array([[1 if p[i, n].solution_value > 0 else 0
                               for i in range(self.req_obj.NUM_REQUESTS)] for n in self.net_obj.PRIORITIES])
            raw_p = np.array([raw_p[:, i].argmax() for i in range(self.req_obj.NUM_REQUESTS)])
            parsed_p = {}
            parsed_req_flw = {}
            parsed_res_flw = {}
            for i in self.req_obj.REQUESTS:
                priority = raw_p[i]
                assigned_req_links = [(j, m) for (j, m) in self.net_obj.LINKS_LIST
                                               if req_flw[i, (j, m), priority].solution_value > 0]
                assigned_res_links = [(j, m) for (j, m) in self.net_obj.LINKS_LIST
                                               if res_flw[i, (j, m), priority].solution_value > 0]
                parsed_p[i] = priority
                parsed_req_flw[i] = assigned_req_links
                parsed_res_flw[i] = assigned_res_links

            parsed_solution["p"] = parsed_p
            parsed_solution["req_flw"] = parsed_req_flw
            parsed_solution["res_flw"] = parsed_res_flw
            parsed_solution["info"] = status
            parsed_solution["OF"] = solution.get_objective_value()
            parsed_solution["done"] = False

            return parsed_solution

        except:
            status = str(mdl.solve_details.status)

            parsed_solution["p"] = {}
            parsed_solution["req_flw"] = {}
            parsed_solution["res_flw"] = {}
            parsed_solution["info"] = status
            parsed_solution["OF"] = 0
            parsed_solution["done"] = True

            return parsed_solution


"""
class CPLEX:

    def __init__(self, net_obj, req_obj, srv_obj, REQUESTED_SERVICES, REQUESTS_ENTRY_NODES):
        self.net_obj = net_obj
        self.req_obj = req_obj
        self.srv_obj = srv_obj
        self.REQUESTED_SERVICES = REQUESTED_SERVICES
        self.REQUESTS_ENTRY_NODES = REQUESTS_ENTRY_NODES

    def solve(self, action):
        if len(action) != 0:
            return 18845.409
        else:
            return 17745.409
"""
