import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add nodes
nodes = {
    'seg': 'Image Segmentation',
    'sim': 'Similarity Principle',
    'dis': 'Discontinuity Principle',
    'point': 'Point',
    'line': 'Line',
    'edge': 'Edge',
    'reg': 'Region based',
    'clus': 'Clustering',
    'thres': 'Thresholding'
}

# Add nodes to graph
for node_id, label in nodes.items():
    G.add_node(node_id, label=label)

# Add edges
edges = [
    ('seg', 'sim'),
    ('seg', 'dis'),
    ('sim', 'reg'),
    ('sim', 'clus'),
    ('sim', 'thres'),
    ('dis', 'point'),
    ('dis', 'line'),
    ('dis', 'edge')
]
G.add_edges_from(edges)

# Set up the plot
plt.figure(figsize=(12, 8))

# Define positions for nodes
pos = {
    'seg': (0, 0),
    'sim': (-2, -1),
    'dis': (2, -1),
    'point': (-3, -2),
    'line': (2, -2),
    'edge': (4, -2),
    'reg': (-3, -3),
    'clus': (0, -3),
    'thres': (3, -3)
}

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', 
                      node_shape='s', alpha=0.8)
nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, width=2)
nx.draw_networkx_labels(G, pos, {n: G.nodes[n]['label'] for n in G.nodes()}, 
                       font_size=10, font_weight='bold')

# Remove axis
plt.axis('off')

# Add title
plt.title('Overview of Image Segmentation Principles and Methods', 
          fontsize=14, pad=20)

# Save the figure
plt.tight_layout()
plt.savefig('segmentation_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Diagram saved as 'segmentation_diagram.png'") 