
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cudaq
from typing import List
from ipywidgets import interact, widgets

def create_qaoa_visualization(sampleGraph):
    """
    This function is used to visualize the QAOA circuit and its results.
    It uses the interactive widgets from the ipywidgets library to create a
    user interface for the visualization.
    """
    # Set variables
    seed = 123
    cudaq.set_random_seed(seed)
    shots = 1
    layer_count = 1
    parameter_count = 2 * layer_count  # Each layer of the QAOA kernel contains 2 parameters
    #%matplotlib inline
    # Problem parameters from the graph
    nodes = list(sampleGraph.nodes())
    qubit_count = len(nodes)  # The number of qubits we'll need is the same as the number of vertices in our graph
    qubit_src = []
    qubit_tgt = []
    for u, v in nx.edges(sampleGraph):
        qubit_src.append(nodes.index(u))
        qubit_tgt.append(nodes.index(v))

    # Function to update the graph based on parameter values
    def update_graph(param1, param2):
        initial_parameters = [param1, param2]
        # Sample the circuit using the initial parameters
        counts = cudaq.sample(kernel_qaoa, qubit_count, layer_count, qubit_src, qubit_tgt, initial_parameters, shots_count=shots)
        #result = str(counts)
        result = str(counts.most_probable())

        # Map the resulting bitstring from the qubit measurements to a list of integer values
        graphColors = [int(i) for i in result]
        nodes_sorted = sorted(list(nx.nodes(sampleGraph)))

        # Compute the cut value associated with resulting coloring
        sampleGraphMaxCut = 0
        cut_edges = []
        for u, v in sampleGraph.edges():
            indexu = nodes_sorted.index(u)
            indexv = nodes_sorted.index(v)
            if graphColors[indexu] != graphColors[indexv]:
                sampleGraphMaxCut += 1
                cut_edges.append((u, v))

        # Compute the partitioning associated with resulting coloring
        cut0 = []
        cut1 = []
        for u in sampleGraph.nodes():
            indexu = nodes_sorted.index(u)
            if graphColors[indexu] == 0:
                cut0.append(u)
                sampleGraph.nodes[u]['color'] = 0
            else:
                cut1.append(u)
                sampleGraph.nodes[u]['color'] = 1

    
    
        # Draw the graph identifying the max cut
        max_cut_color_map = ['#8C8C8C' if sampleGraph.nodes[u]['color'] == 0 else '#76B900' for u in sampleGraph]
        # Set the position of the nodes so that the graphs is easily recognizable each time it's plotted.
        pos = nx.spring_layout(sampleGraph, seed=311)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(
            sampleGraph,
            pos,
            edgelist=cut_edges,
            width=8,
            alpha=0.5,
            edge_color='#76B900',
        )
        nx.draw(sampleGraph, with_labels=True, pos=pos, node_color=max_cut_color_map)
        plt.title('QAOA Max-Cut Visualization: Changing parameters α1 and β1 in a one-layer QAOA circuit')
        plt.show()
        print('The cut found using the parameters α1 and β1 in the QAOA circuit is highlighted in green.')
        print('The cut value for the this coloring is', sampleGraphMaxCut)
    



    
    # auxilary functions to generate data for the visualization
    @cudaq.kernel
    def kernel_qaoa(qubit_count :int, layer_count: int, edges_src: List[int], edges_tgt: List[int], thetas : List[float]):
        """Build the QAOA circuit for max cut of the graph with given edges and nodes
            Parameters
            ----------
            qubit_count: int
                Number of qubits in the circuit, which is the same as the number of nodes in our graph
            layer_count : int
                Number of layers in the QAOA kernel
            edges_src: List[int]
                List of the first (source) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
            edges_tgt: List[int]
                List of the second (target) node listed in each edge of the graph, when the edges of the graph are listed as pairs of nodes
            thetas: List[float]
                Free variables to be optimized
        """
        # Let's allocate the qubits
        qreg = cudaq.qvector(qubit_count)

        # And then place the qubits in superposition
        h(qreg)

        # Each layer has two components: the problem kernel and the mixer
        for i in range(layer_count):
            # Add the problem kernel to each layer
            for edge in range(len(edges_src)):
                qubitu = edges_src[edge]
                qubitv = edges_tgt[edge]
                x(qreg[qubitu])  # Apply the problem kernel for the edge (qubitu, qubitv)
                #qaoaProblem(qreg[qubitu], qreg[qubitv], thetas[i])
            # Add the mixer kernel to each layer
            #for j in range(qubit_count):
                #qaoaMixer(qreg[j],thetas[i+layer_count])
    
    
    

    # Create sliders for the initial parameters
    param1_slider = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=np.random.uniform(-np.pi, np.pi), description='α1')
    param2_slider = widgets.FloatSlider(min=-np.pi, max=np.pi, step=0.1, value=np.random.uniform(-np.pi, np.pi),description='β1')

    # Create the interactive widget
    interact(update_graph, param1=param1_slider, param2=param2_slider)


