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
