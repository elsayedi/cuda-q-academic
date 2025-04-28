
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
