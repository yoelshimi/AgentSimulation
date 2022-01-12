import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt

# Network topology
n = 500
p = 0.045
G = nx.erdos_renyi_graph(n, p)

degrees = []
# some properties
print("node degree clustering")
for v in nx.nodes(G):
    degrees.append(nx.degree(G, v))
    print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# print the adjacency list
#for line in nx.generate_adjlist(G):
#    print(line)

nx.draw(G)
plt.show()

_ = plt.hist(degrees, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram of degrees")
plt.show()


# Model selection
model = ep.SEIRModel(G)

# Model Configuration
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.01)
cfg.add_model_parameter('gamma', 0.005)
cfg.add_model_parameter('alpha', 0.05)
cfg.add_model_parameter("fraction_infected", 0.05)
model.set_initial_status(cfg)

# Simulation execution
iterations = model.iteration_bunch(200)
trends = model.build_trends(iterations)

viz = DiffusionTrend(model, trends)
p3 = viz.plot(width=400, height=400)
vm.add_plot(p3)