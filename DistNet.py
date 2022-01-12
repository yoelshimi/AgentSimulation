import numpy as np
import random
import networkx as nx

class Person:
    def __init__(self, number, age, contact_list, contact_weights):
        self.number = number
        self.age = age
        self.contact_list = contact_list
        self.contact_weights = contact_weights

generate_contacts()


def generate_from_dist():
    return

def generate_people(age_dist, num_people,age_range, bin_size):
    population = []

    age_pop = [[] for _ in range(np.ceil(age_range / bin_size))]

    for id in range(num_people):
        age = generate_from_dist(age_dist)
        age_bin = np.floor(age/bin_size)
        new_person = Person(age=age,[],[])
        population.append(new_person)
        age_pop[age_bin].append(new_person)


    for p in population:
        friends = get_connections_ages(connection_dist)
        for f in friends:
            p.contact_list.append(f.number)
            f.contact_list.append(p.number)
            p.contact_weights
