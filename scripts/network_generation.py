# Importing Libraries
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
plt.style.use('ggplot')

class diffusion_network:
    """
    This class will generate a network based on a population's demographic information or an existing edgelist.
    Arguments:
    ===========
    demographics_df: pd.Dataframe
        Denotes the demographic attributes to generate a dataframe on. Can have as many columns as needed
    edgelist: pd.Dataframe
        A two-column dataframe denoting the network edgelist
    num_nodes: int
        How many nodes that network generated from demographics should have. Must be equal to the number of rows in demographics_df.
    edges_per_node: int
        How many edges each node should have when it enters the network.
    n_communities: int
        The number of communities in the network.
    community_strength: int
        Denotes the increased likelihood of a node connecting to an in-community node relative to out-community node.
        For example, community_strength=100 denotes that every node is 100x more likely to connect to a node in its same community compared to a node not in its community.
    """
    def __init__(self, demographics_df=None,edgelist=None,num_nodes:int=5000,n_communities:int=100,community_strength:int=1000):
        # Save the basic network attributes of the network
        self.num_nodes = num_nodes
        self.n_communities = n_communities
        self.community_strength = community_strength


        # Initialize the network structure either from an edgelist or a population's demographic information
        if demographics_df is None and edgelist is None: 
            raise Exception("Specify how the network should be generated")
        elif demographics_df is not None and edgelist is not None:
            raise Exception("Please only input demographic information or and edgelist, not both.")
        elif demographics_df is not None:
            self.G = self.generate_network_from_demographic(demographics_df,num_nodes,n_communities,community_strength)
        elif edgelist is not None:
            self.G = self.import_edgelist(edgelist)

        """Initialize the node attributes for diffusion"""
        # Generate and validate network adjacency matrix
        self.adj_matrix = nx.to_numpy_array(self.G)
        assert self.adj_matrix.shape[0] == self.adj_matrix.shape[1]

        # Normalize adjacency matrix to get the mean influence weight of each node
        row_sums = self.adj_matrix.sum(axis=1, keepdims=True)
        self.normalized_adj_matrix = self.adj_matrix / row_sums
        
        # Check that the network structure has an eventual stable state (i.e. all eigenvalues >= 1 are not complex numbers)
        self.eigvals, self.eigvecs = np.linalg.eig(self.normalized_adj_matrix) 
        self.signf_eigvals = np.abs(self.eigvals) > 0.999
        isreal_eigvals = np.isreal(self.eigvals)
        if not np.array_equal(np.logical_and(self.signf_eigvals,isreal_eigvals), self.signf_eigvals):
            raise Exception("This network has no stable state. Please regenerate the network.")

    def generate_network_from_demographic(self,attitudes_df:pd.DataFrame,num_nodes,n_communities,community_strength,diagnostic_plots=True):
        """Generate a Bianconi-Barabási model with Homophily"""
        # Create communities based on vaccination attitudes
        cluster_model = KMeans(n_clusters=n_communities, random_state=100)
        cluster_model.fit(attitudes_df)
        community = cluster_model.labels_

        # Save community assignments to a table for later use
        self.node_community = pd.Series(community)
        self.node_community.index.name = 'id'
        self.node_community.name = 'community'

        # Generate similarity matrix based on if nodes are in same community for later use
        demographic_similarity = np.ones((len(community),len(community)))
        for i in range(0,len(community)):
            for j in range(0,len(community)):
                if community[i] == community[j]:
                    # increased likelihood of an edge if in same community
                    demographic_similarity[i,j] = community_strength
        
        # Map community to a list of node numbers for later use
        community = pd.DataFrame(community,columns=['community']).reset_index()
        comm_to_node = community.groupby('community')['index'].agg(list).to_dict()
    
        def bb_network(N, m,demographic_similarity): # Generate a Bianconi–Barabási (BB) network
            # 1. Start with undirected graph
            G = nx.Graph()

            # 2. Create a fully-connected clique (of size m + 1) for each community
            for community in comm_to_node:
                G.add_nodes_from(comm_to_node[community][:(m+1)])
                G.add_edges_from(list(itertools.combinations(comm_to_node[community][:(m + 1)], 2)))

            connected_nodes = G.number_of_nodes()
            i = m
            while connected_nodes < N:
                i += 1
                for community in comm_to_node: # 3. add one node from a community then move on to next community
                    if i < len(comm_to_node[community]):
                        connected_nodes += 1
                        node_to_add = comm_to_node[community][i]

                        # 4. Connect the new node to m different nodes, weighted by their similarity and number of existing ties.
                        possible_neighbors = list(G.nodes)
                        G.add_node(node_to_add)

                        weight = [demographic_similarity[node_to_add,j] * G.degree(j) for j in possible_neighbors]
                        p = np.array(weight)/np.sum(np.array(weight))
                        new_neighbors = np.random.choice(possible_neighbors, size=m, replace=False, p=p)

                        for j in new_neighbors:
                            G.add_edge(node_to_add, j)
            return G   
        G = bb_network(num_nodes,3,demographic_similarity) # Generate BB network

        # Print Network Diagnostic Information:
        print('\033[92m Network Generation Successful')
        print('\033[92m Network Fully Connected:',nx.is_connected(G))
        print('\033[92m Number of Nodes:',G.number_of_nodes())
        print("\033[92m Number of Edges:",G.number_of_edges())
        print("\033[92m Average Clustering:",nx.average_clustering(G))
        print("\033[92m Modularity:",nx.community.modularity(G,comm_to_node.values()))

        if diagnostic_plots: # Plot Network Degree Distribution
            degree_dist = list(dict(G.degree).values())
            generate_log_bin = lambda data: np.logspace(np.log10(np.min(data)), np.log10(np.max(data)), 15)
            plt.hist(degree_dist,bins=generate_log_bin(degree_dist))
            plt.title('Degree distribution of network')
            plt.xlabel('Degree centrality')
            plt.ylabel('Frequency')
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('figures/network_degree_dist.png')
            plt.show()

            # Plot Network Path Length
            path_length_dict = dict(nx.shortest_path_length(G))

            # Unwrap nested values in dictionary
            path_length_dist = np.array([list(path_length_dict[i].values()) for i in path_length_dict]).flatten()

            # Execute Plot
            plt.hist(path_length_dist)
            plt.title('Network Shortest Path Length Between Nodes')
            plt.xlabel('Path Length')
            plt.ylabel('Frequency')
            plt.savefig('figures/network_path_dist.png')
            plt.show()
        return G

    """Implement Importing Functions to Import and Export to Nodelist"""
    def import_edgelist(self,edgelist:pd.DataFrame):
        return nx.from_pandas_edgelist(edgelist)
    
    def export_edgelist(self,filename:str='edgelist.csv'):
        edgelist = nx.to_pandas_edgelist(self.G)
        edgelist.columns = ['source', 'target']
        edgelist.to_csv(filename,index=False)
        return None

    """Implement Diffusion of Ideas Along Network"""
    def fj_diffusion(self,node_values):
        L = nx.laplacian_matrix(self.G)
        I = np.identity(L.shape[0])
        A = np.linalg.inv(I + L)
        return np.round(np.dot(A,node_values))