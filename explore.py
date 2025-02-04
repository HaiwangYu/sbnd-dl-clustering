import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

from scipy.spatial import ConvexHull

def visualize_polyhedrons_plotly(blobs):
    """
    Visualize polyhedrons using plotly.

    Args:
        blobs (numpy.ndarray): 2D array representing the polyhedrons.
    """
    fig = go.Figure()

    for blob in blobs:
        num_vertices = int(round(blob[1]))
        vertices = blob[2:2+num_vertices*3].reshape(num_vertices, 3)
        
        print(f'bf: {vertices}')

        hull = ConvexHull(vertices)
        sorted_vertices = vertices[hull.vertices]
        print(f'af: {sorted_vertices}')

        fig.add_trace(go.Mesh3d(
            x=sorted_vertices[:, 0],
            y=sorted_vertices[:, 1],
            z=sorted_vertices[:, 2],
            color='lightblue',
            opacity=1
        ))
        break

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()

def visualize_polyhedrons(blobs):
    """
    Visualize polyhedrons using matplotlib.
    
    Args:
        blobs (numpy.ndarray): 2D array representing the polyhedrons.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for blob in blobs:
        num_vertices = int(round(blob[1]))
        vertices = blob[2:2+num_vertices*3].reshape(num_vertices, 3)
        hull = ConvexHull(vertices)
        sorted_vertices = vertices[hull.vertices]
        ax.plot_trisurf(sorted_vertices[:, 0], sorted_vertices[:, 1], sorted_vertices[:, 2])
        break

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_graph_3d(x, edge_index, node_labels=None, node_size=2, edge_width=1):
   """
   Visualize a graph in 3D using plotly.
   
   Args:
       x (torch.Tensor): Node features/positions (N x 3)
       edge_index (torch.Tensor): Edge indices (2 x E)
       node_labels (list, optional): Labels for nodes. Default: None
       node_size (int): Size of nodes. Default: 10
       edge_width (int): Width of edges. Default: 2
   """
   import plotly.graph_objects as go

   if node_labels is None:
       node_labels = [f'Node {i}' for i in range(len(x))]

   # Prepare edge coordinates
   edges_x, edges_y, edges_z = [], [], []
   for edge in edge_index.t():
       start_node = x[edge[0]]
       end_node = x[edge[1]]
       edges_x.extend([start_node[0], end_node[0], None])
       edges_y.extend([start_node[1], end_node[1], None])
       edges_z.extend([start_node[2], end_node[2], None])

   fig = go.Figure(data=[
       # Nodes
       go.Scatter3d(
           x=x[:,0], y=x[:,1], z=x[:,2],
           mode='markers+text',
           marker=dict(size=node_size),
           # text=node_labels,
           name='Nodes'
       ),
       # Edges
       go.Scatter3d(
           x=edges_x, y=edges_y, z=edges_z,
           mode='lines',
           line=dict(color='black', width=edge_width),
           name='Edges'
       )
   ])

   fig.update_layout(
       showlegend=True,
       scene=dict(
           xaxis_title='X',
           yaxis_title='Y',
           zaxis_title='Z',
           camera=dict(
               eye=dict(x=1, y=0, z=0)
           )
       )
   )
   
   return fig

import numpy as np
import torch
from torch_geometric.data import Data

# Load the file
data = np.load('graph2json.npz')

# Print the content
print(data.files)
for file in data.files:
    print(file)
    print(data[file].shape)
    print(data[file])

# Load the file
data = np.load('graph2json.npz')

# Extract 'apoints' and 'aedges'
ablobs = data['blobs']
apoints = data['points']
aedges = data['ppedges']
# apoints = data['ablobs']
# aedges = np.array([[0,1]])
# print(f'apoints: {apoints.shape}')
# print(f'aedges: {aedges.shape}')
visualize_polyhedrons_plotly(ablobs)
visualize_polyhedrons(ablobs)
exit()

# Convert to torch tensors
apoints = torch.from_numpy(apoints)
aedges = torch.from_numpy(aedges).long()[:, :2].t()
print(aedges)

# Create graph data object
data = Data(x=apoints, edge_index=aedges)

# Print the graph data
print(data)
fig = visualize_graph_3d(data.x, data.edge_index, node_size=2, edge_width=1)
fig.show()

