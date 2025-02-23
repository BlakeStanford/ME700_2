# ME700_2

Tutorial: Using the 3D Frames Solver
Overview
In this tutorial, you will learn how to:

Define the frame geometry: Create nodes and an element to represent a cantilever beam.
Assign element properties: Specify material and geometric properties (e.g., Young's modulus, moments of inertia, etc.).
Apply loads and boundary conditions: Apply a force at the free node and fix all degrees of freedom at the support.
Solve the system: Assemble the global stiffness matrix, solve for nodal displacements, and compute reaction forces.
Visualize and interpret results: Print displacement and reaction outputs and optionally plot the deformed shape.
Note: This example assumes you have already implemented (or imported) the Node, Element, and FrameSolver classes as shown in our previous implementation.

Prerequisites
Make sure you have Python 3.11.11 installed along with these libraries:

NumPy: For numerical operations.
SciPy: For advanced linear algebra (if needed).
Matplotlib: For plotting the deformed shape.
unittest: For testing (optional but recommended).
You can install the necessary packages using pip:

bash
Copy
pip install numpy scipy matplotlib
Step 1: Define the Frame Geometry
A frame is defined by its nodes (points in 3D space) and elements (the beams connecting the nodes). In our cantilever beam example, we have two nodes:

Node 0: Located at the origin (fixed support).
Node 1: Located 1.0 meter along the X-axis (free end).
Create the nodes and the element that connects them.

python
Copy
from direct_stiffness_solver import Node, Element, FrameSolver  # Assuming the solver code is in this module
import numpy as np

# Define nodes (node IDs are zero-indexed)
node0 = Node(0, [0, 0, 0])      # Fixed support at the origin
node1 = Node(1, [1.0, 0, 0])    # Free end at 1 meter along the X-axis
nodes = [node0, node1]

# Define element properties (SI units)
properties = {
    'E': 210e9,      # Young's modulus in Pascals
    'nu': 0.3,       # Poisson's ratio
    'I_y': 8.1e-6,   # Second moment of area about the local z-axis (for bending in Y)
    'I_z': 4.05e-6,  # Second moment of area about the local y-axis (for bending in Z)
    'I_rho': 1e-5,   # Torsional constant (polar moment)
    'A': 0.01        # Cross-sectional area in m²
}

# For the local coordinate system, we need a vector for the local z-axis.
# Here, we simply use the global Z-axis.
element = Element(0, node0, node1, properties, local_z=[0, 0, 1])
elements = [element]
Step 2: Specify Loads and Boundary Conditions
Loads
For our cantilever beam, we apply a downward vertical force of 1000 N at Node 1.
Each node’s load vector has 6 components:

Translations: [F<sub>x</sub>, F<sub>y</sub>, F<sub>z</sub>]
Rotations: [M<sub>rx</sub>, M<sub>ry</sub>, M<sub>rz</sub>]
Boundary Conditions
Node 0 (the support) is fully fixed, meaning all 6 degrees of freedom are constrained.

python
Copy
# Define nodal loads:
# For Node 1, apply a force of -1000 N in the Y-direction.
loads = {
    1: [0, -1000.0, 0, 0, 0, 0]
}

# Define boundary conditions:
# At Node 0, fix all degrees of freedom: [0, 1, 2, 3, 4, 5]
boundary_conditions = {
    0: [0, 1, 2, 3, 4, 5]
}
Step 3: Assemble and Solve the System
Instantiate the FrameSolver with the nodes, elements, loads, and boundary conditions. Then solve the system to obtain nodal displacements and reactions.

python
Copy
# Create the FrameSolver instance
solver = FrameSolver(nodes, elements, loads, boundary_conditions)

# Solve for nodal displacements and reaction forces
displacements, reactions = solver.solve()

# Print the nodal displacements
print("Nodal Displacements (translations in meters and rotations in radians):")
for node in nodes:
    idx = 6 * node.id
    disp = displacements[idx:idx+6]
    print(f"Node {node.id}: {disp}")

# Print the reaction forces and moments at the fixed node (Node 0)
print("\nReaction Forces and Moments at Fixed Node (Node 0):")
print(f"Node 0: {reactions[0:6]}")
What to Expect:
Node 0: As it is fully constrained, its displacement should be all zeros. The reaction vector at Node 0 should have a force approximately balancing the applied load (about +1000 N in the Y-direction) and appropriate moments due to beam bending.
Node 1: Will exhibit nonzero displacements. In particular, the vertical displacement (the second entry in the vector) represents the deflection caused by the 1000 N load.
Step 4: Visualize the Deformed Shape
To help interpret the results, you can visualize the deformed frame. The solver’s plot_deformed_shape method displays both the undeformed (dashed blue) and deformed (solid red) configurations.

python
Copy
# Optionally, visualize the deformed shape.
# The 'scale' parameter amplifies the displacements for better visualization.
solver.plot_deformed_shape(scale=1000)
A plot window will appear showing:

Blue dashed lines: The original (undeformed) configuration.
Red solid lines: The deformed configuration, which should clearly indicate the deflection under load.
Running the Tutorial
You can save the code snippets above in a script (for example, tutorial.py). Make sure the module containing your FrameSolver, Node, and Element classes is accessible. Run the script using:

bash
Copy
python tutorial.py
Watch the terminal output for displacement and reaction force values and view the plot to see the deformed shape.

Conclusion
In this tutorial, you learned how to:

Define the geometry of a 3D frame (using nodes and elements).
Specify element properties, loads, and boundary conditions.
Assemble the global stiffness matrix and solve the system for displacements and reaction forces.
Visualize the deformed shape to understand the structural response.
This example of a cantilever beam is a basic starting point. You can extend this approach to analyze more complex frames by adding additional nodes, elements, and load cases. Experiment with different parameters to see how the structure responds under various conditions.

Happy coding and structural analysis!

