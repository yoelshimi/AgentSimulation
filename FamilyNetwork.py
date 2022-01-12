import numpy as np
import random
import plotting
import RumorSpreading


class Person:
    def __init__(self, number, typ, age, fmly, wrk, schl, connectList, connectWeightList, status, belief=False, suscept=False):
        self.number = number
        self.age = age
        self.type = typ
        self.family = fmly
        self.work = wrk
        self.school = schl
        self.connectList = connectList
        self.connectWeightList = connectWeightList
        self.status = status
        self.is_believer = belief
        self.is_sensitive = suscept

    def my_func(self):
        print("Hello my name is " + self.number)
        print(vars(self))


class Family:
    def __init__(self, number, size, member_list, out_weight):
        self.number = number
        self.size = size
        self.member_list = member_list
        self.out_weight = out_weight


class Work:
    def __init__(self, number, size, member_list, out_weight):
        self.number = number
        self.size = size
        self.member_list = member_list
        self.out_weight = out_weight


class School:
    def __init__(self, number, size, member_list, out_weight):
        self.number = number
        self.size = size
        self.member_list = member_list
        self.out_weight = out_weight


# def generate_family(size, infection, employment):
def get_from_dist(dist):
    # gets array of distribution
    dist = dist / sum(dist)
    x = random.random()
    s = dist[0]
    y = 0
    while s < x:
        y += 1
        s += dist[y]
    return y


def get_from_shifted_dist(dist, dist_x):
    val = get_from_dist(dist)
    val = val / len(dist) * (dist_x[1]-dist_x[0]) + dist_x[0]
    return val


def get_ages(num_parents,num_kids,age_distributions):

    if num_parents == 1 and num_kids == 0:
        #  return [get_from_dist(age_distributions["age_dist_pdf"])]
        return [get_from_dist(age_distributions["single_age_dist"]) - 10]
    elif num_parents == 2 and num_kids == 0:
        mum_age = get_from_dist(age_distributions["couple_no_kids"] )
        partner_age = mum_age + round(get_from_shifted_dist(age_distributions["partner_age_delta"], age_distributions["partner_age_delta_x"])-5)
        return [mum_age, partner_age]
    elif num_parents > 2 and num_kids == 0:
        return [get_from_dist(age_distributions["age_dist_pdf"]) for _ in range(num_parents)]
    elif num_parents >= 1 and num_kids >= 1:
        ages = []
        mum_age = get_from_dist(age_distributions["mums_age"])
        if num_kids > 1:
            while mum_age >= 40:
                mum_age = get_from_dist(age_distributions["mums_age"])
        ages.append(mum_age)

        if num_parents > 1:
            ages.append(mum_age + round(get_from_shifted_dist(age_distributions["partner_age_delta"], age_distributions["partner_age_delta_x"])))

        for _ in range(num_parents - 2):
            ages.append(get_from_dist(age_distributions["age_dist_pdf"]))
        kids_ages = [mum_age - get_from_dist(age_distributions["kids_age_delta"]) for _ in range(num_kids)]
        #  kids_ages = [get_from_dist(age_distributions["kids_age_dist"]) for _ in range(num_kids)]

        """
        while max(kids_ages) > 20:
            loc = np.argmax(kids_ages)
            kids_ages[loc] = random.randrange(18)
        while min(kids_ages) < 0:
            loc = np.argmin(kids_ages)
            kids_ages[loc] = mum_age - get_from_dist(age_distributions["kids_age_delta"])
            
            """
        ages += kids_ages
        return ages
    else:
        print("family age error!")
        return []


def get_ages_basic(num_parents, num_kids, distributions):
    ages = []
    [ages.append(get_from_dist(distributions["adults_x"])) for _ in range(num_parents)]
    [ages.append(get_from_dist(distributions["kids_x"])) for _ in range(num_kids)]
    return ages


def mean_dist(dist):
    dist = dist / sum(dist)
    return sum(range(len(dist)) * dist)


def generate_work_school(n_works, s_work, i_work,
                         n_schools, s_school, i_school):
    works = []
    for work_no in range(n_works):
        works.append(Work(work_no, s_work, [], i_work))

    schools = []
    for school_no in range(n_schools):
        schools.append(School(school_no, s_school, [], i_school))

    return works, schools


def generate(number_families, size_family_dist, age_distributions, mean_adults, employment, size_work,
             size_school, random_connection, plot, infection_school=0, infection_family=0,
             infection_work=0, infection_random=0, BelieverSusceptibleCorr=0, beta_l=None,
             num_children=0, number_schools=1,
             ):

    if beta_l is None:
        beta_l = [0, 0, 0]
    number_works = int(np.ceil(number_families * mean_adults * employment / size_work))

    families = []

    people = []

    num_people = len(people)

    works, schools = generate_work_school(number_works, size_work,
                                          infection_work, number_schools, size_school, infection_school)

    for family_no in range(number_families):

        family_id_list = []
        # parents
        parents, kids = divmod(get_from_dist(size_family_dist), 6)
        parents += 1
        #  ages = get_ages_basic(parents, kids, age_distributions)
        ages = get_ages(num_parents=parents, num_kids=kids, age_distributions=age_distributions)
        ages = [min(ages[f], 100) for f in range(len(ages))]
        ages = [max(ages[f], 0) for f in range(len(ages))]

        if plot:
            print(ages)
        for j in range(parents):
            work = -1
            if np.random.random(1) < employment:
                work = random.randrange(number_works)
                works[work].member_list.append(num_people)
            adult = Person(num_people, 'adult', ages[j], family_no, work, -1, [], [], 'healthy')
            people.append(adult)
            family_id_list.append(num_people)
            num_people += 1

        # kids
        num_children += kids
        for j in range(kids):
            school_no = len(schools) - 1

            if len(schools[school_no].member_list) > size_school:
                school_no += 1
                schools.append(School(school_no, size_school, [], infection_school))

            # if school_no > 1 and random.random() < 0.3 and schools[school_no - 1] < size_school:
            #    school_no = school_no - 1

            child = Person(num_people, 'child', ages[j], family_no, -1, school_no, [], [], 'healthy')
            people.append(child)
            family_id_list.append(num_people)
            schools[child.school].member_list.append(num_people)

            num_people += 1

        family = Family(family_no, len(family_id_list), family_id_list, infection_family)
        families.append(family)
        for member_id in family.member_list:
            member = people[member_id]
            member.connectList = np.asarray(family_id_list)
            member.connectWeightList = np.ones(len(family_id_list)) * infection_family

    # randomly pick out adults and send them to school

    temp = random.sample(range(number_families), len(schools))
    teachers = [families[school_no].member_list[0] for school_no in temp]
    for (person_id, school) in zip(teachers, schools):
        person = people[person_id]
        if person.work != -1:
            works[person.work].member_list.remove(person.number)
        person.work = -1
        person.school = school.number
        school.member_list.append(person.number)

    for person in people:
        # update work
        # [print(f.member_list) for f in families]
        colleagues = []
        if person.type == 'adult' and person.work != -1:

            colleagues = works[person.work].member_list
            person.connectList = np.append(person.connectList, colleagues)
            person.connectWeightList = np.append(person.connectWeightList, np.ones(len(colleagues),'float') * infection_work)

        if person.school != -1:

            colleagues = schools[person.school].member_list
            person.connectList = np.append(person.connectList, colleagues)
            person.connectWeightList = np.append(person.connectWeightList, np.ones(len(colleagues),'float') * infection_school)

        for _ in range(int(np.round(random_connection/2))):
            buddy_no = random.randrange(num_people)
            # update connections
            person.connectList = np.append(person.connectList, buddy_no)
            person.connectWeightList = np.append(person.connectWeightList, infection_random)
            people[buddy_no].connectList = np.append(people[buddy_no].connectList, person.number)
            people[buddy_no].connectWeightList = np.append(people[buddy_no].connectWeightList, infection_work)

    if plot:
        for person in people:
            print([person.number, person.type, person.family, len(person.connectList), sum(person.connectWeightList),
                   person.work, person.school])

    #  Adj_Matrix = np.zeros((num_people+1, num_people+1), 'int8')
    list_edges = []
    # list_edges.append(('source', 'target', 'weight'))

    avg_weights = 0
    for person in people:
        num = person.number
        for (neighbor, weight) in zip(person.connectList, person.connectWeightList):
            if num != neighbor:
                #  Adj_Matrix[num, neighbor] = weight
                list_edges.append((num, neighbor, weight))
                #  avg_weights += weight
        #   Adj_Matrix[num, num] = 0
    ages = [p.age for p in people]
    ages = [p.age for p in people]
    types =[1 if p.type=='child' else 0 for p in people]
    if plot:
        plotting.plotAges(ages)

    # to guarantee that a person gets conneceted via strongest connection
    list_edges.reverse()

    G = RumorSpreading.Graphify(people, list_edges, BelieverSusceptibleCorr, beta_l=beta_l)

    degrees = np.asarray(G.degree)
    degrees = degrees[:, 1]
    avg_weights = np.mean(np.asarray(G.degree(weight='weight'))[:, 1]) / np.mean(degrees)
    # avg_weights / (2 * len(G.edges))

    print(f"people: {num_people} children: {num_children} ")
    print(f"schools: {len(schools)} num offices: {number_works}")
    print(f"graph density: {sum(degrees)/(num_people*(num_people-1))}")

    return G, num_people, avg_weights, degrees
