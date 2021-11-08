import math
import numpy as np

rnd = np.random


class Network:
    def __init__(self, NUM_NODES, SEED=4, NUM_TIERS=3, TIER_HEIGHT=100, TIER_WIDTH=20, DC_CAPACITY_UNIT=50,
                 DC_COST_UNIT=50, LINK_BW_LB=100, LINK_BW_UB=150, LINK_COST_LB=10, LINK_COST_UB=20,
                 NUM_PRIORITY_LEVELS=1, BURST_SIZE_LIMIT=50, PACKET_SIZE=10):

        rnd.seed(SEED)
        self.NUM_NODES = NUM_NODES
        self.NUM_TIERS = NUM_TIERS
        self.TIER_HEIGHT = TIER_HEIGHT
        self.TIER_WIDTH = TIER_WIDTH
        self.DC_CAPACITY_UNIT = DC_CAPACITY_UNIT
        self.DC_COST_UNIT = DC_COST_UNIT
        self.LINK_BW_LB = LINK_BW_LB
        self.LINK_BW_UB = LINK_BW_UB
        self.LINK_COST_LB = LINK_COST_LB
        self.LINK_COST_UB = LINK_COST_UB
        self.BURST_SIZE_LIMIT = BURST_SIZE_LIMIT
        self.PACKET_SIZE = PACKET_SIZE
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.PRIORITIES = np.linspace(0, NUM_PRIORITY_LEVELS, NUM_PRIORITY_LEVELS + 1).astype(int)
        self.NODES = np.arange(NUM_NODES)
        self.X_LOCS, self.Y_LOCS = self.initialize_coordinates()
        self.DISTANCES = self.initialize_distances()
        self.DC_CAPACITIES = self.initialize_dc_capacities()
        self.DC_COSTS = self.initialize_dc_costs()
        self.LINKS_LIST = self.initialize_links_list()
        self.LINKS_MATRIX = self.initialize_links_matrix()
        self.LINK_BWS_DICT, self.LINK_BWS_MATRIX = self.initialize_link_bws()
        self.LINK_COSTS_DICT, self.LINK_COSTS_MATRIX = self.initialize_link_costs()
        self.BURST_SIZE_LIMIT_PER_PRIORITY = self.compute_burst_size_limit_per_priority()
        self.LINK_BWS_LIMIT_PER_PRIORITY = self.compute_link_bws_limit_per_priority()
        self.BURST_SIZE_CUM_LIMIT_PER_PRIORITY = self.compute_burst_size_cum_limit_per_priority()
        self.LINK_BWS_CUM_LIMIT_PER_PRIORITY = self.compute_link_bws_cum_limit_per_priority()
        self.LINK_DELAYS_DICT, self.LINK_DELAYS_MATRIX = self.initialize_link_delays()
        self.FIRST_TIER_NODES = self.get_first_tier_nodes()
        self.STATE = self.get_state()

    def get_tier_num(self, i):

        tier_size = math.ceil(self.NUM_NODES / self.NUM_TIERS)

        for t in range(self.NUM_TIERS):
            if t * tier_size <= i <= (t + 1) * tier_size:
                tier_num = t

        return tier_num

    def initialize_coordinates(self):

        X_LOCS = np.array(
            [rnd.randint(self.get_tier_num(i) * self.TIER_WIDTH, (self.get_tier_num(i) + 1) * self.TIER_WIDTH) for i in
             self.NODES])
        Y_LOCS = np.random.randint(0, self.TIER_HEIGHT, self.NUM_NODES)

        return X_LOCS, Y_LOCS

    def initialize_distances(self):

        distances = np.array([[np.hypot(self.X_LOCS[i] - self.X_LOCS[j], self.Y_LOCS[i] - self.Y_LOCS[j])
                               for j in self.NODES] for i in self.NODES])
        distances = distances.astype(int)

        return distances

    def initialize_dc_capacities(self):

        dc_capacities = np.array([rnd.randint(self.get_tier_num(i) * self.DC_CAPACITY_UNIT, (self.get_tier_num(i) + 1)
                                              * self.DC_CAPACITY_UNIT) for i in self.NODES])

        return dc_capacities

    def initialize_dc_costs(self):

        dc_costs = np.array([rnd.randint((self.NUM_TIERS - self.get_tier_num(i) - 1) * self.DC_COST_UNIT,
                                         (self.NUM_TIERS - self.get_tier_num(i)) * self.DC_COST_UNIT)
                             for i in self.NODES])

        return dc_costs

    def initialize_links_list(self):

        links_list = []

        for i in self.NODES:
            for j in self.NODES:
                if i != j and self.is_j_neighbor_of_i(i, j):
                    if (i, j) not in links_list:
                        links_list.append((i, j))
                    if (j, i) not in links_list:
                        links_list.append((j, i))

        return links_list

    def is_j_neighbor_of_i(self, i, j):

        if abs(self.get_tier_num(i) - self.get_tier_num(j)) <= 1:
            close_neighbors = {k: self.DISTANCES[i, k] for k in self.NODES if self.get_tier_num(k) ==
                               self.get_tier_num(j) and k != i}
            if j == min(close_neighbors, key=close_neighbors.get):
                return True
            else:
                return False
        else:
            return False

    def initialize_links_matrix(self):

        links_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES))

        for (i, j) in self.LINKS_LIST:
            links_matrix[i, j] = 1

        return links_matrix

    def initialize_link_bws(self):

        link_bws_dict = {}
        link_bws_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES))

        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS_LIST:
                    rnd_bw = rnd.randint(self.LINK_BW_LB, self.LINK_BW_UB)
                    link_bws_dict[(i, j)] = rnd_bw
                    link_bws_dict[(j, i)] = rnd_bw
                    link_bws_matrix[i, j] = rnd_bw
                    link_bws_matrix[j, i] = rnd_bw

        link_bws_matrix = link_bws_matrix.astype(int)

        return link_bws_dict, link_bws_matrix

    def initialize_link_costs(self):

        link_costs_dict = {}
        link_costs_matrix = np.zeros((self.NUM_NODES, self.NUM_NODES))

        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS_LIST:
                    rnd_cost = rnd.randint(self.LINK_COST_LB, self.LINK_COST_UB)
                    link_costs_dict[(i, j)] = rnd_cost
                    link_costs_dict[(j, i)] = rnd_cost
                    link_costs_matrix[i, j] = rnd_cost
                    link_costs_matrix[j, i] = rnd_cost

        link_costs_matrix = link_costs_matrix.astype(int)

        return link_costs_dict, link_costs_matrix

    def compute_burst_size_limit_per_priority(self):

        # burst_size_limit_per_priority = [(self.NUM_PRIORITY_LEVELS + 1 - i) * self.BURST_SIZE_LIMIT if i > 0 else 0
        #                                  for i in self.PRIORITIES]
        burst_size_limit_per_priority = np.array([((self.NUM_PRIORITY_LEVELS + 1 - i) /
                                                   np.array(self.PRIORITIES).sum()) * self.BURST_SIZE_LIMIT
                                                  if i > 0 else 0 for i in self.PRIORITIES])

        return burst_size_limit_per_priority.astype(int)

    def compute_link_bws_limit_per_priority(self):

        link_bws_limit_per_priority = {}

        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS_LIST:
                    for n in self.PRIORITIES:
                        if n > 0:
                            link_bws_limit_per_priority[(i, j), n] = \
                                ((self.NUM_PRIORITY_LEVELS + 1 - n) / np.array(self.PRIORITIES).sum()) \
                                * self.LINK_BWS_DICT[i, j]
                            link_bws_limit_per_priority[(j, i), n] = \
                                ((self.NUM_PRIORITY_LEVELS + 1 - n) / np.array(self.PRIORITIES).sum()) \
                                * self.LINK_BWS_DICT[i, j]
                        else:
                            link_bws_limit_per_priority[(i, j), n] = 0
                            link_bws_limit_per_priority[(j, i), n] = 0

        return link_bws_limit_per_priority

    def compute_burst_size_cum_limit_per_priority(self):

        burst_size_cum_limit_per_priority = [np.array(self.BURST_SIZE_LIMIT_PER_PRIORITY[:i + 1]).sum()
                                             for i in self.PRIORITIES]

        return np.array(burst_size_cum_limit_per_priority)

    def compute_link_bws_cum_limit_per_priority(self):

        link_bws_cum_limit_per_priority = {}

        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS_LIST:
                    array = []
                    for n in self.PRIORITIES:
                        array.append(
                            self.LINK_BWS_LIMIT_PER_PRIORITY[(i, j), n])
                    for n in self.PRIORITIES:
                        link_bws_cum_limit_per_priority[(i, j), n] = np.array(array)[:n].sum()
                        link_bws_cum_limit_per_priority[(j, i), n] = np.array(array)[:n].sum()

        return link_bws_cum_limit_per_priority

    def initialize_link_delays(self):

        link_delays_dict = {}
        link_delays_matrix = np.ones((self.NUM_PRIORITY_LEVELS + 1, self.NUM_NODES, self.NUM_NODES)) * 1000

        for i in self.NODES:
            for j in self.NODES:
                if i > j and (i, j) in self.LINKS_LIST:
                    for n in self.PRIORITIES:
                        if n > 0:
                            delay = (self.BURST_SIZE_CUM_LIMIT_PER_PRIORITY[n] + self.PACKET_SIZE) / (
                                    self.LINK_BWS_DICT[i, j] - self.LINK_BWS_CUM_LIMIT_PER_PRIORITY[(i, j), n])
                            + self.PACKET_SIZE / self.LINK_BWS_DICT[i, j]
                            link_delays_dict[(i, j), n] = round(delay, 3)
                            link_delays_dict[(j, i), n] = round(delay, 3)
                            link_delays_matrix[n, i, j] = round(delay, 3)
                            link_delays_matrix[n, j, i] = round(delay, 3)
                        else:
                            link_delays_dict[(i, j), n] = 1000
                            link_delays_dict[(j, i), n] = 1000

        return link_delays_dict, link_delays_matrix

    def get_state(self):

        arr1 = np.concatenate((self.DC_CAPACITIES, self.DC_COSTS))
        arr2 = np.concatenate((arr1, self.LINK_BWS_MATRIX.reshape(1, -1)[0]))
        arr3 = np.concatenate((arr2, self.LINK_COSTS_MATRIX.reshape(1, -1)[0]))
        arr4 = np.concatenate((arr3, self.LINK_DELAYS_MATRIX[1:][0].reshape(1, -1)[0]))

        return arr4

    def get_first_tier_nodes(self):

        return np.array([i for i in self.NODES if self.get_tier_num(i) == 0])