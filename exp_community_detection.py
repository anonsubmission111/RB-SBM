import numpy as np
import sys
import model_gibbs as model_file
from sklearn.metrics import normalized_mutual_info_score

sys.path.append('./Data/cora')
sys.path.append('./Data/citeseer')
import read_cora as rcr
import read_citeseer as rcs


# Read the data
adj_mat, node_features, ground_truth = rcr.read_cora('./Data/cora')
# adj_mat, node_features, ground_truth = rcs.read_citeseer('./Data/citeseer')
ground_truth = np.asarray(ground_truth)


# Some useful variables
num_nodes = adj_mat.shape[0]
num_communities = np.max(ground_truth) + 1
num_features = node_features.shape[1]
batch_size = min(256, num_nodes)


print('Number of nodes:', num_nodes)
print('Number of attributes:', num_features)
print('Number of edges:', adj_mat.sum())
print('P(edge):', adj_mat.sum() / (num_nodes * (num_nodes - 1)))
print('Number of communities:', num_communities)

# Get the edges
indices = np.nonzero(adj_mat)
edges = []
for i in range(indices[0].shape[0]):
    edges.append((indices[0][i], indices[1][i]))

# Compute prior for B matrix
alphas = np.ones((num_communities, num_communities))
betas = np.ones((num_communities, num_communities)) + \
        10 * (np.ones((num_communities, num_communities)) - np.eye(num_communities))


dump_file = open('./dump_cora_gibbs.txt', 'w')
for iter in range(50):
    # Set up the model
    model = model_file.RBMSBM(num_nodes, num_communities, num_features, alphas, betas)

    # Train the model
    for i in range(1000):
        # Get the communities
        communities = np.argmax(model.q, axis=1)

        # Show the scores
        if i % 20 == 0:
            print('Trial:', iter + 1, 'NMI:', normalized_mutual_info_score(ground_truth, communities))
            dump_file.write(str(iter + 1) + '\t' + str(normalized_mutual_info_score(ground_truth, communities)) + '\n')

        # Run the E-Step
        model.variational_e_step(edges, adj_mat, node_features,
                                 indices=np.random.randint(0, num_nodes, size=batch_size).tolist(),
                                 lamda=1 - 0.7 * (1 - i/float(1000)))

        # Run the M-step
        for _ in range(10):
            model.variational_m_step(node_features, chain_length=10, num_samples=100, lr=1, momentum=0.0,
                                     use_persistence=True)

dump_file.close()
