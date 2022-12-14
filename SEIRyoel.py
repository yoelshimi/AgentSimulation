import plotting
import numpy as np
import future
from ndlib.models.DiffusionModel import DiffusionModel
import time
import collections


def inf_prob(beta, weights):
    p = 1
    for w in weights:
        p = p * (1-beta*w)
    return 1 - p

def infection_time(lmda):
    if lmda == np.inf:
        return np.inf
    return np.random.exponential(lmda)

class SEIRQModel(DiffusionModel):

    def __init__(self, graph):

        super(self.__class__, self).__init__(graph)

        self.name = "SEIRQ"

        self.available_statuses = {
            "Susceptible": 0,
            "Exposed": 2,
            "Infected": 1,
            "Removed": 3,
            "Quarantine": 4,
            "Hospital": 5,
            "Dead": 6
        }
        self.parameters = {
            "model": {
                "alpha": {
                    "descr": "Incubation period",
                    "range": [0, 1],
                    "optional": False},
                "beta": {
                    "descr": "Infection rate",
                    "range": [0, 1],
                    "optional": False},
                "gamma": {
                    "descr": "Recovery rate",
                    "range": [0, 1],
                    "optional": False
                },
                "quarantine": {
                    "descr": "Probability of being quarantined upon infection",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "q_time": {
                    "descr": "time to go into quarantine",
                    "range": [0, 1e6],
                    "optional": True,
                    "default": 1
                },
                "tp_rate": {
                    "descr": "Whether if the infection rate depends on the number of infected neighbors",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                },
                "is_sensitive": {
                    "descr": "whether sensitivity is measured",
                    "range": [0, 1],
                    "optional": True,
                    "default": False
                },
                "is_believer": {
                    "descr": "whether belief in fake news is measured",
                    "range": [0, 1],
                    "optional": True,
                    "default": False
                },
                "hosp_rate": {
                    "descr": "The de-hospitalization rate of the disease",
                    "optional": True,
                    "default": 0
                },
                "prob_d": {
                    "descr": "The probability of death from disease",
                    "optional": True,
                    "default": 0
                },
                "prob_h": {
                    "descr": "The probability of hospitalization from disease",
                    "optional": True,
                    "default": 0
                },
                "sim_out": {
                    "descr": "output file name",
                    "optional": True,
                    "default": "simulation output"
                },
                "beta_l": {
                    "descr": "list of beta factors for ",
                    "optional": True,
                    "default": "simulation output"
                }
            },
            "nodes": {},
            "edges": {},
        }

        self.progress = np.full((len(graph.nodes), 1), np.inf)  # determines time until I move state
        self.infector = {}  # who infected me
        self.inf_time = {}  # when I was infected.
        self.Dead = np.zeros((2, 2), dtype='float')  # A counter of the number of sensitive people to be infected.
        self.Sick = np.zeros((2, 2), dtype='float')
        self.Hosp = np.zeros((2, 2), dtype='float')


    def yoel_iteration_bunch(self, bunch_size, node_status=True):
        """
        Execute a bunch of model iterations

        :param bunch_size: the number of iterations to execute
        :param node_status: if the incremental node status has to be returned.

        :return: a list containing for each iteration a dictionary {"iteration": iteration_id, "status": dictionary_node_to_status}
        """
        system_status = []
        it = 0
        its = self.iteration({}, node_status)
        system_status.append(its)
        while (it < bunch_size-1) or (sum([its['node_count'][a] for a in [1,5]]) >= 1 and it < bunch_size * 4):
            its = self.iteration(node_status=node_status, last_its=its)
            system_status.append(its)
            if it % 100 == 0:
                print(it)
            it += 1
        its = self.iteration({}, node_status=True, is_final=True)
        system_status.append(its)
        return system_status

    def showInfectiousCondition(self):
        # find nodes that interacted with the virus --- they will be "R"
        res, infs = plotting.getSensitiveFromGraph(self.graph, self.infector, True)
        towrite = [
            f"\n**-------------------------**",
            f"\n {time.asctime(time.localtime(time.time()))}",
            f"\npopulation: {res} ",
            f"\nsick divisions:",
            f"\n 'snb': {self.Sick[0][0]}, 'sb': {self.Sick[0][1]}, 'nsnb': {self.Sick[1][0]}, 'nsb': {self.Sick[1][1]}",
            f"\nhospital divisions:",
            f"\n 'snb': {self.Hosp[0][0]}, 'sb': {self.Hosp[0][1]}, 'nsnb': {self.Hosp[1][0]}, 'nsb': {self.Hosp[1][1]}",
            f"\nDead divisions:",
            f"\n 'snb': {self.Dead[0][0]}, 'sb': {self.Dead[0][1]}, 'nsnb': {self.Dead[1][0]}, 'nsb': {self.Dead[1][1]}",
            f"\ninfectors SB: {infs} "
            ]

        f = open(self.params["model"]['sim_out']+"_sb.txt", "a")
        f.writelines(towrite)
        f.close()
        print(towrite)

    def status_delta(self, actual_status, last_its={}, eventNodes = []):
        """
        Compute the point-to-point variations for each status w.r.t. the previous system configuration

        :param actual_status: the actual simulation status
        :return: node that have changed their statuses (dictionary status->nodes),
                 count of actual nodes per status (dictionary status->node count),
                 delta of nodes per status w.r.t the previous configuration (dictionary status->delta)
        """
        actual_status_count = {}
        delta = {}
        for n, v in future.utils.iteritems(self.status):
            if v != actual_status[n]:
                delta[n] = actual_status[n]

        if False and eventNodes and last_its:
            for st in list(self.available_statuses.values()):
                actual_status_count[st] = last_its['node_count'][st]
            for n in eventNodes:
                last_state = self.status[n]
                actual_status_count[last_state] -= 1
                new_state = actual_status[n]
                actual_status_count[new_state] += 1
        else:
            for st in list(self.available_statuses.values()):
                actual_status_count[st] = len([x for x in actual_status.values() if x == st])

        if last_its:
            old_status_count = last_its['node_count']
        else:
            old_status_count = {}
            for st in list(self.available_statuses.values()):
                old_status_count[st] = len([x for x in self.status if self.status[x] == st])

        status_delta = {st: actual_status_count[st] - old_status_count[st] for st in actual_status_count}
        #  if [len([x for x in actual_status.values() if x == y]) for y in range(len(self.available_statuses))] != [actual_status_count[x] for x in range(len(self.available_statuses))]:
        #      print("error")
        if any([x < 0 for x in actual_status_count.values()]):
            print("error")
        return delta, actual_status_count, status_delta

    def iteration(self, last_its, node_status=True, is_final=False):
        #  self.clean_initial_status(list(self.available_statuses.values()))

        actual_status = self.status.copy()
        #  {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.actual_iteration += 1

            #  update people initially sick
            od = collections.OrderedDict(sorted(self.initial_status.items()))
            for u in np.nonzero(list(od.values()))[0]:
                self.infector[u] = -1
                self.inf_time[u] = 0
                self.progress[u] = 0
                if self.initial_status[u] == 1:
                    actual_status[u] = 2  # we make him exposed with 0 time to infected.
                elif self.initial_status[u] == 2:
                    actual_status[u] = 0  # we make him susceptible with 0 time to go

            delta, node_count, status_delta = self.status_delta(actual_status)
            self.status = actual_status
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        self.progress -= 1

        eventNodes = list(np.flatnonzero([self.progress < 0]))
        if eventNodes:
            # order them so the first node is first.
            if len(eventNodes) > 1:
                eventNodes = [eventNodes[i] for i in np.fromiter(np.argsort(self.progress[eventNodes], 0), dtype="int32")]
            for u in eventNodes:
                actual_status = self.run_node(actual_status=actual_status, u=u)
        if len(eventNodes) != 0 or not last_its:
            #  there has been an update, or the first or last run.
            delta, node_count, status_delta = self.status_delta(actual_status, eventNodes=list(eventNodes), last_its=last_its)
        else:
            #  print(self.actual_iteration)
            delta = {}
            node_count = last_its['node_count']
            status_delta = {x: 0 for x in self.available_statuses.values()}
        self.status = actual_status
        self.actual_iteration += 1

        if node_status and is_final:
            print("final iter reached!")
            tau_IB = {}
            i_times = self.inf_time.copy()
            for u in self.graph.nodes:
                if actual_status[u] in (3, 6):  # recovered or dead (sick should be empty)
                    #  if tau_IB.get(u) is None:
                        #  tau_IB[u] = set()
                    #  tau_IB[u].add((u, self.inf_time[u]))
                    # when did I infect people
                    s = self.infector[u]  # who infected me
                    if s not in (float('inf'), -1):
                        # when did infector of u infect people.
                        try:
                            tau_IB[s].add((u, self.inf_time[u]))
                        except KeyError:
                            tau_IB[s] = {(u, self.inf_time[u])}
            if self.params['model']['is_sensitive']:
                self.showInfectiousCondition()

            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy(),
                    "tau_IB": tau_IB, "i_times": i_times,"Sick": self.Sick, "Hosp": self.Hosp, "Dead": self.Dead}
        else:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

    def run_node(self, actual_status, u, is_recursive=False):
        u_status = self.status[u]
        if is_recursive:
            u_status = actual_status[u]
        u_progress = self.progress[u].copy()
        if u_progress <= 0 and (u_status not in (3,6)):
            if u_status == 0:  # Susceptible
                actual_status[u] = 2  # Exposed
                self.progress[u] = np.random.exponential(self.params['model']['alpha'])

            if u_status == 2:  # Exposed becomes infected
                actual_status[u] = 1  # Infected
                self.inf_time[u] = self.actual_iteration
                self.progress[u] = np.random.exponential(self.params['model']['gamma'])

                for v in self.graph.neighbors(u):
                    # update susceptible neighbors
                    if self.status[v] == 0:
                        (s_ind, b_ind) = self.SB_partition(u, 2)
                        (s_ind1, b_ind1) = self.SB_partition(v, 0)
                        b_factor = self.parameters
                        # some will quarantine
                        next_inf_time = infection_time(self.params['model']['beta'] / self.graph.edges[u, v]['weight'])
                        # if my new time is faster than current time, and still within my infectious window
                        if (next_inf_time < self.progress[v]) and (self.progress[u] > next_inf_time):
                            if np.random.random() < self.params['model']['quarantine']:
                                self.progress[v] = self.params['model']['q_time']
                                actual_status[v] = 4  # quarantined
                            else:
                                self.progress[v] = next_inf_time

                            if u_progress + next_inf_time < 0:
                                #  immediately run next node, to allow "shortcutting" before next iteration.
                                #  if u infects neighbor, and neighbor changes phase before tick finishes.
                                actual_status[v] = 2
                                self.progress[v] = np.random.exponential(self.params['model']['alpha'])
                                # if alpha is zero, it can skip straight to new infections before iteration ends.
                                print("recursive skip: " +str(v)+" infected by: "+str(u))
                                actual_status = self.run_node(actual_status, v, is_recursive=True)
                            #  self.inf_time[v] = self.actual_iteration + next_inf_time

                            self.infector[v] = u

            #  INFECTED
            elif u_status == 1:
                # finished lifetime
                if self.params['model']['is_sensitive'] and self.params['model']['is_believer']:
                    (s_ind, b_ind) = self.SB_partition(u, u_status)
                    if np.random.rand() < self.params['model']['prob_h'][s_ind]:
                        actual_status[u] = 5  # hospital
                        self.progress[u] = np.random.exponential(self.params['model']['hosp_rate'])
                    else:
                        actual_status[u] = 3  # Removed
                        self.progress[u] = np.inf

                else:
                    actual_status[u] = 3  # Removed, not sensitive.
                    self.progress[u] = np.inf

            #  quarantine
            elif u_status == 4:
                actual_status[u] = 3  # Removed
                self.progress[u] = np.inf

            #  hospital
            elif u_status == 5:
                s_ind = 0
                b_ind = 0  # default: SNB
                if self.params['model']['is_sensitive'] and self.params['model']['is_believer']:
                    (s_ind, b_ind) = self.SB_partition(u, u_status)

                if np.random.rand() < self.params['model']['prob_d'][s_ind]:
                    #  dead
                    actual_status[u] = 6  # Dead
                    self.Dead[s_ind][b_ind] += 1
                    self.progress[u] = np.inf
                else:
                    actual_status[u] = 3  # Removed
                    self.progress[u] = np.inf

        #  elif u_status == 3:
        # do nothing
        #  elif u_status == 6:
        # do nothing
        return actual_status

    def SB_partition(self, u, u_status):
        # assume SB is considered.
        if self.graph.nodes[u]['is_believer']:
            b_ind = 1
        else:
            b_ind = 0
        if self.graph.nodes[u]['is_sensitive']:
            s_ind = 0
        else:
            s_ind = 1
        if u_status == 6:
            self.Dead[s_ind][b_ind] += 1
        if u_status == 5:
            self.Hosp[s_ind][b_ind] += 1
        if u_status == 1:
            self.Sick[s_ind][b_ind] += 1
        return (s_ind, b_ind)

    def event_iteration(self, last_its, node_status=True, is_final=False):
        actual_status = last_its.copy()

        if self.actual_iteration == 0:
            self.actual_iteration += 1

            #  update people initially sick
            od = collections.OrderedDict(sorted(self.initial_status.items()))
            for u in np.nonzero(list(od.values()))[0]:
                self.progress[u] = 0
                if self.initial_status[u] == 1:
                    actual_status[u] = 2  # we make him exposed with 0 time to infected.
                elif self.initial_status[u] == 2:
                    actual_status[u] = 0  # we make him susceptible with 0 time to go

            delta, node_count, status_delta = self.status_delta(actual_status)
            self.status = actual_status
            self.internalTime = 0
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        self.internalTime += 1

        eventNodes = list(np.flatnonzero([self.progress < 0]))
        for u in eventNodes:
            actual_status = self.run_node(actual_status=actual_status, u=u)
        if len(eventNodes) != 0 or not last_its:
            #  there has been an update, or the first or last run.
            delta, node_count, status_delta = self.status_delta(actual_status, eventNodes=eventNodes, last_its=last_its)
        else:
            #  print(self.actual_iteration)
            delta = {}
            node_count = last_its['node_count']
            status_delta = {x: 0 for x in self.available_statuses.values()}
        self.status = actual_status
        self.actual_iteration += 1

        if node_status and is_final:
            tau_IB = {}
            i_times = self.inf_time.copy()
            for u in self.graph.nodes:
                if actual_status[u] in (3, 6):  # recovered or dead (sick should be empty)
                    #  if tau_IB.get(u) is None:
                        #  tau_IB[u] = set()
                    #  tau_IB[u].add((u, self.inf_time[u]))
                    # when did I infect people
                    s = self.infector[u]  # who infected me
                    if s not in (float('inf'), -1):
                        # when did infector of u infect people.
                        try:
                            tau_IB[s].add((u, self.inf_time[u]))
                        except KeyError:
                            tau_IB[s] = {(u, self.inf_time[u])}
            if self.params['model']['is_sensitive']:
                self.showInfectiousCondition()

            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy(),
                    "tau_IB": tau_IB, "i_times": i_times,"Sick": self.Sick, "Hosp": self.Hosp, "Dead": self.Dead}
        else:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}