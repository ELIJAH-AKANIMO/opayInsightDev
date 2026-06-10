import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx

print('Loading enriched full data (script)...')
df = pd.read_csv('data/enriched_full.csv')
print('Rows:', len(df))

# build graph
print('Building graph...')
edges = df[['cc_num','merchant']].drop_duplicates().values.tolist()
G = nx.Graph()
G.add_edges_from(edges)
print('Nodes:', G.number_of_nodes(), 'Edges:', G.number_of_edges())

# compute measures
print('Computing PageRank...')
pr = nx.pagerank(G, alpha=0.85)
print('Computing eigenvector centrality (numpy)...')
evc = nx.eigenvector_centrality_numpy(G)
print('Computing clustering coefficient...')
clust = nx.clustering(G)
print('Computing core number...')
core = nx.core_number(G)

# compute node-level fraud rates
print('Computing per-node fraud rates...')
card_fraud = df.groupby('cc_num')['is_fraud'].mean().to_dict()
merch_fraud = df.groupby('merchant')['is_fraud'].mean().to_dict()
node_fraud = {}
for node in G.nodes():
    if node in card_fraud:
        node_fraud[node] = card_fraud[node]
    elif node in merch_fraud:
        node_fraud[node] = merch_fraud[node]
    else:
        node_fraud[node] = 0.0

# neighbor fraud rate = mean node_fraud over neighbors
print('Computing neighbor fraud rates...')
neighbor_fraud = {}
for i, node in enumerate(G.nodes()):
    nbrs = list(G.neighbors(node))
    if not nbrs:
        neighbor_fraud[node] = 0.0
    else:
        vals = [node_fraud.get(n,0.0) for n in nbrs]
        neighbor_fraud[node] = float(np.mean(vals))
    if (i+1) % 500 == 0:
        print(f'Processed {i+1} nodes')

# map back to df
print('Mapping features to transactions...')
df['node_pagerank_card'] = df['cc_num'].map(lambda x: pr.get(x,0.0))
df['node_pagerank_merch'] = df['merchant'].map(lambda x: pr.get(x,0.0))
df['node_eig_card'] = df['cc_num'].map(lambda x: evc.get(x,0.0))
df['node_eig_merch'] = df['merchant'].map(lambda x: evc.get(x,0.0))
df['node_clust_card'] = df['cc_num'].map(lambda x: clust.get(x,0.0))
df['node_clust_merch'] = df['merchant'].map(lambda x: clust.get(x,0.0))
df['node_core_card'] = df['cc_num'].map(lambda x: core.get(x,0))
df['node_core_merch'] = df['merchant'].map(lambda x: core.get(x,0))
df['node_neighbor_fraud_card'] = df['cc_num'].map(lambda x: neighbor_fraud.get(x,0.0))
df['node_neighbor_fraud_merch'] = df['merchant'].map(lambda x: neighbor_fraud.get(x,0.0))

# Save enriched with structural features
Path('data').mkdir(exist_ok=True)
df.to_csv('data/enriched_full_graph_struct.csv', index=False)
print('Saved data/enriched_full_graph_struct.csv')

# Save node-level dicts
import joblib
Path('models').mkdir(exist_ok=True)
joblib.dump({'pr': pr, 'evc': evc, 'clust': clust, 'core': core, 'neighbor_fraud': neighbor_fraud}, 'models/graph_struct_dicts.pkl')
print('Saved models/graph_struct_dicts.pkl')

print('Done')