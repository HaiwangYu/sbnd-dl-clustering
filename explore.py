import torch
from torch_geometric.data import Data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go

from scipy.spatial import ConvexHull

def visualize_polyhedrons(blobs):
    """
    Visualize polyhedrons using plotly.

    Args:
        blobs (numpy.ndarray): 2D array representing the polyhedrons.
    """
    fig = go.Figure()

    nblobs = 0
    for blob in blobs:
        # print(f'blob: {blob}')
        # exit()
        num_vertices = int(round(blob[1]))
        # print(f'num_vertices: {num_vertices}')
        vertices = blob[2:2+num_vertices*3].reshape(num_vertices, 3)
        if vertices.shape[0] == 0:
            # print('Empty vertices')
            continue
        # print(f'vertices: {vertices}')
        
        hull = ConvexHull(vertices)
        sorted_vertices = vertices[hull.vertices]

        # Create the convex volume
        x, y, z = sorted_vertices[:, 0], sorted_vertices[:, 1], sorted_vertices[:, 2]
        convex_volume = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            color='lightblue',
            opacity=0.2
        )
        fig.add_trace(convex_volume)
        nblobs += 1
        # if nblobs > 0:
        #     exit()

    # fig.update_layout(
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z'
    #     )
    # )
    return fig

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
    nskip = 10
    fig = go.Figure(data=[
        # Nodes
        go.Scatter3d(
            x=x[::nskip,0], y=x[::nskip,1], z=x[::nskip,2],
            mode='markers+text',
            marker=dict(size=node_size),
            # text=node_labels,
            name='Nodes'
        ),
        # Edges
        go.Scatter3d(
           x=edges_x[::], y=edges_y[::], z=edges_z[::],
           mode='lines',
           line=dict(color='black', width=edge_width),
           name='Edges'
        )
    ])

    # fig.update_layout(
    #     showlegend=True,
    #     scene=dict(
    #         xaxis_title='X',
    #         yaxis_title='Y',
    #         zaxis_title='Z',
    #         camera=dict(
    #             up=dict(x=0, y=1, z=0),    # Y is up (vertical)
    #             eye=dict(x=0, y=0, z=2),  # Looking along the X-axis
    #         )
    #     )
    # )
    
    return fig

if __name__ == '__main__':
    import numpy as np
    import torch
    from torch_geometric.data import Data
    import argparse


    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='../wcp-porting-img/dunefd/groupingcrm0-0groupingcrm0-0-1-1-1.npz',
                        help='Path to the input file')
    args = parser.parse_args()

    # Load the file
    data = np.load(args.input_file)

    # Print the content
    print(data.files)
    for file in data.files:
        print(file)
        print(data[file].shape)
        print(data[file])

    # Extract 'apoints' and 'aedges'
    ablobs = data['blobs']
    apoints = data['points']
    aedges = data['ppedges']
    # apoints = data['ablobs']
    # aedges = np.array([[0,1]])
    # print(f'apoints: {apoints.shape}')
    # print(f'aedges: {aedges.shape}')
    # fig = visualize_polyhedrons(ablobs)
    # fig.show()

    # Convert to torch tensors
    apoints = torch.from_numpy(apoints)
    aedges = torch.from_numpy(aedges).long()[:, :2].t()
    # print(aedges)

    # Create graph data object
    data = Data(x=apoints, edge_index=aedges)

    # Print the graph data
    print(data)
    fig = visualize_graph_3d(data.x, data.edge_index, node_size=2, edge_width=1)
    fig.show()
    exit()


    # Visualize polyhedrons
    fig_polyhedrons = visualize_polyhedrons(ablobs)

    # Visualize graph in 3D
    fig_graph_3d = visualize_graph_3d(data.x, data.edge_index, node_size=2, edge_width=1)

    # Overlay the figures
    fig_polyhedrons.add_trace(fig_graph_3d.data[0])
    fig_polyhedrons.add_trace(fig_graph_3d.data[1])

    # Show the overlayed figure

    fig_polyhedrons.update_layout(scene=dict(aspectmode='data'))
    fig_polyhedrons.show()
