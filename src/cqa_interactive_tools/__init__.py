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



# Bloch Sphere Visualization Widget

import ipywidgets as widgets
from ipywidgets import interactive_output, HBox, VBox
from IPython.display import display, clear_output
import cudaq
import numpy as np

# Function to plot Bloch sphere and visualize quantum states
def bloch_sphere_visualization():
    # Kernel to initialize a qubit and set it to the state |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)e^{iφ}|1⟩
    @cudaq.kernel
    def state_kernel(theta: float, phi: float):
        qubit = cudaq.qubit()
        ry(theta, qubit)
        rz(phi, qubit)

    # Function to update the Bloch sphere plot based on the slider values
    def update_bloch_sphere(theta, phi):
        with output_bloch:
            clear_output(wait=True)  # Clear previous output
            state = cudaq.get_state(state_kernel, theta, phi)
            bloch_sphere = cudaq.add_to_bloch_sphere(state)
            cudaq.show(bloch_sphere)

    # Function to update the circuit diagram based on the slider values
    def update_circuit_diagram(theta, phi):
        with output_circuit:
            clear_output(wait=True)  # Clear previous output
            print(cudaq.draw(state_kernel, theta, phi))

    # Create interactive sliders for angles
    slider_theta = widgets.FloatSlider(min=0, max=2*np.pi, step=0.01, value=0, description='θ:', continuous_update=False)
    slider_phi = widgets.FloatSlider(min=0, max=2*np.pi, step=0.01, value=0, description='φ:', continuous_update=False)

    # Add a new title to the widget
    title = widgets.HTML(value="<h3>Visualize the State |ψ⟩ on the Bloch Sphere with Varying Angles θ and φ (in radians)</h3>")

    # Display the title and sliders
    display(title)

    # Create output widgets
    output_bloch = widgets.Output()
    output_circuit = widgets.Output()

    def update_all(theta, phi):
        update_bloch_sphere(theta, phi)
        update_circuit_diagram(theta, phi)

    interactive_plot = interactive_output(update_all, {'theta': slider_theta, 'phi': slider_phi})

    # Arrange sliders horizontally
    slider_box = HBox([slider_theta, slider_phi])

    # Display the sliders and the interactive plots side by side
    display(VBox([slider_box, HBox([output_bloch, output_circuit])]))
import matplotlib.pyplot as plt
import cudaq
from ipywidgets import interactive_output, widgets, HBox, VBox
import numpy as np

def create_rotation_visualization():
    # Execute this function to enable the interactive widget that shows the Bloch sphere and circuit diagram of
    # a qubit that is rotated about the x, y, and z axes by given angles.

    # Kernel to initialize a qubit in the zero ket state and rotate it about the x, y, and z axis by given angles
    @cudaq.kernel
    def rotation_kernel(theta_x: float, theta_y: float, theta_z: float):
        qubit = cudaq.qubit()
        rx(theta_x, qubit)
        ry(theta_y, qubit)
        rz(theta_z, qubit)

    # Function to update the Bloch sphere plot based on the slider values
    def update_bloch_sphere(theta_x, theta_y, theta_z):
        state = cudaq.get_state(rotation_kernel, theta_x, theta_y, theta_z)
        bloch_sphere = cudaq.add_to_bloch_sphere(state)
        cudaq.show(bloch_sphere)

    # Function to update the circuit diagram based on the slider values
    def update_circuit_diagram(theta_x, theta_y, theta_z):
        # Draw the circuit diagram
        print(cudaq.draw(rotation_kernel, theta_x, theta_y, theta_z))

    # Create interactive sliders for angles
    slider_x = widgets.FloatSlider(min=0, max=2*np.pi, step=0.01, value=0, description='θx:', continuous_update=False)
    slider_y = widgets.FloatSlider(min=0, max=2*np.pi, step=0.01, value=0, description='θy:', continuous_update=False)
    slider_z = widgets.FloatSlider(min=0, max=2*np.pi, step=0.01, value=0, description='θz:', continuous_update=False)

    # Add a new title to the widget
    title = widgets.HTML(value="<h3>Visualize the Final Qubit State on the Bloch Sphere after Rotating about the X, Y and Z Axis with Varying Angles (θx, θy, θz in radians)</h3>")

    # Display the title and sliders
    display(title)

    # Create interactive outputs
    output_bloch = widgets.Output()
    output_circuit = widgets.Output()

    def update_all(theta_x, theta_y, theta_z):
        with output_bloch:
            output_bloch.clear_output(wait=True)
            update_bloch_sphere(theta_x, theta_y, theta_z)
        with output_circuit:
            output_circuit.clear_output(wait=True)
            update_circuit_diagram(theta_x, theta_y, theta_z)

    interactive_plot = interactive_output(update_all, {'theta_x': slider_x, 'theta_y': slider_y, 'theta_z': slider_z})

    # Arrange sliders horizontally
    slider_box = HBox([slider_x, slider_y, slider_z])

    # Display the sliders and the interactive plots side by side
    display(VBox([slider_box, HBox([output_bloch, output_circuit])]))
