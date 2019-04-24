import numpy as np
import sys
import model_link_pred as model_file
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from scipy.sparse import csr_matrix

sys.path.append('./Data/cora')
sys.path.append('./Data/citeseer')
import read_cora as rcr
import read_citeseer as rcs


# Read the data
# adj_mat, node_features, ground_truth = rcr.read_cora('./Data/cora')
adj_mat, node_features, ground_truth = rcs.read_citeseer('./Data/citeseer')
ground_truth = np.asarray(ground_truth)
adj_mat = adj_mat.todense()


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
indices_negative = np.nonzero(np.ones(adj_mat.shape) - adj_mat)
num_missing = int(0.2 * indices[0].shape[0])
labels = np.concatenate([np.ones((1, num_missing)), np.zeros((1, num_missing))], axis=1).squeeze()

# Compute prior for B matrix
alphas = np.ones((num_communities, num_communities))
betas = np.ones((num_communities, num_communities)) + \
        10 * (np.ones((num_communities, num_communities)) - np.eye(num_communities))


dump_file = open('./dump_citeseer_gibbs_link.txt', 'w')
for iter in range(50):
    # Inject missing links such that edges and non-edges are equally likely to be missing
    positive = list(range(indices[0].shape[0]))
    np.random.shuffle(positive)
    missing_edges = [(indices[0][i], indices[1][i]) for i in positive[:num_missing]]

    negative = list(range(indices_negative[0].shape[0]))
    np.random.shuffle(negative)
    missing_edges += [(indices_negative[0][i], indices_negative[1][i]) for i in negative[:num_missing]]

    # Calculate the observed adjacency matrix
    A_unk = np.zeros((num_nodes, num_nodes))
    for i, j in missing_edges:
        A_unk[i, j] = 1
    A_obs = csr_matrix(np.multiply(adj_mat, (np.ones(A_unk.shape) - A_unk)))
    A_unk = csr_matrix(A_unk)

    # Get the edges
    indices_obs = np.nonzero(A_obs)
    edges = []
    for i in range(indices_obs[0].shape[0]):
        edges.append((indices_obs[0][i], indices_obs[1][i]))

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
        model.variational_e_step(edges, missing_edges, A_obs, A_unk, node_features,
                                 indices=np.random.randint(0, num_nodes, size=batch_size).tolist(),
                                 lamda=1 - 0.7 * (1 - i/float(1000)))

        # Run the M-step
        for _ in range(10):
            model.variational_m_step(node_features, chain_length=10, num_samples=100, lr=1, momentum=0.0,
                                     use_persistence=True)

    probs = model.predict(missing_edges)
    auc = roc_auc_score(labels, probs)
    print('AUC:', auc)
    dump_file.write(str(iter + 1) + '\tAUC\t' + str(auc) + '\n')

dump_file.close()
