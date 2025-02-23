"""
Direct Stiffness Method 3D Frames Solver

This module implements a 3D frame solver using the direct stiffness method.
It expects the following inputs:
  - Frame geometry: node locations and element connectivity.
  - Element section properties: Young's modulus (E), Poisson's ratio (nu),
    second moments of area (I_y, I_z), torsional constant (I_rho), and (optionally)
    cross-sectional area (A; default = 1.0 if not provided).
  - Nodal loading: forces and moments applied at nodes (each node has 6 DOFs).
  - Boundary conditions: specified supported (constrained) DOFs at nodes.

The solver outputs:
  - Nodal displacements (translations and rotations).
  - Reaction forces and moments at the constrained nodes.

The implementation uses common libraries such as NumPy, SciPy, and Matplotlib.
Test Driven Development (TDD) practices are followed by including unit tests.
"""

import numpy as np
from numpy.linalg import norm
import scipy.linalg
import matplotlib.pyplot as plt
import unittest

# =============================================================================
# Data Classes for Node and Element
# =============================================================================

class Node:
    """
    Class representing a node in the frame.
    
    Attributes:
        id (int): Node identifier (should be zero-indexed).
        coord (np.ndarray): 3D coordinates of the node.
    """
    def __init__(self, id: int, coord):
        self.id = id
        self.coord = np.array(coord, dtype=float)

class Element:
    """
    Class representing a frame element.
    
    Attributes:
        id (int): Element identifier.
        node_start (Node): Start node of the element.
        node_end (Node): End node of the element.
        properties (dict): Section properties containing:
            - 'E': Young's modulus.
            - 'nu': Poisson's ratio.
            - 'I_y': Second moment of area about local y-axis.
            - 'I_z': Second moment of area about local z-axis.
            - 'I_rho': Torsional constant (polar moment).
            - 'A': Cross-sectional area (optional, default = 1.0).
        local_z (np.ndarray): A vector indicating the desired orientation of the local z-axis.
                              (It need not be perpendicular to the element axis; it is used to define the local coordinate system.)
    """
    def __init__(self, id: int, node_start: Node, node_end: Node, properties: dict, local_z):
        self.id = id
        self.node_start = node_start
        self.node_end = node_end
        self.properties = properties.copy()
        # Set default cross-sectional area if not provided.
        if 'A' not in self.properties:
            self.properties['A'] = 1.0
        self.local_z = np.array(local_z, dtype=float)

# =============================================================================
# Frame Solver Class
# =============================================================================

class FrameSolver:
    """
    Solver for 3D frames using the Direct Stiffness Method.
    
    The solver performs the following steps:
      1. Assemble the global stiffness matrix.
      2. Assemble the global load vector.
      3. Apply boundary conditions.
      4. Solve for nodal displacements.
      5. Compute reaction forces at constrained DOFs.
    """
    def __init__(self, nodes, elements, loads, boundary_conditions):
        """
        Initializes the solver.
        
        Parameters:
            nodes (list of Node): List of nodes.
            elements (list of Element): List of elements.
            loads (dict): Dictionary mapping node id to a 6-component load vector.
            boundary_conditions (dict): Dictionary mapping node id to a list of constrained DOF indices (0: u, 1: v, 2: w, 3: rx, 4: ry, 5: rz).
        """
        self.nodes = {node.id: node for node in nodes}
        self.elements = elements
        self.loads = loads  # {node_id: np.array([F_u, F_v, F_w, M_rx, M_ry, M_rz])}
        self.boundary_conditions = boundary_conditions  # {node_id: [dof indices that are fixed]}
        self.n_nodes = len(nodes)
        self.ndof = 6 * self.n_nodes  # Each node has 6 DOF.
        
        # Global stiffness matrix, load vector, displacement vector, and reaction vector.
        self.K_global = np.zeros((self.ndof, self.ndof))
        self.F_global = np.zeros(self.ndof)
        self.U = np.zeros(self.ndof)
        self.reactions = np.zeros(self.ndof)
        
        self._assemble_load_vector()
        self._assemble_global_stiffness()
    
    # -------------------------------
    # Static Methods for Element Matrices
    # -------------------------------
    
    @staticmethod
    def _local_stiffness_matrix(E, A, I_y, I_z, I_rho, L, nu):
        """
        Computes the 12x12 local stiffness matrix for a 3D frame element.
        
        The DOF ordering at each node is assumed as:
          [u (axial), v, w, rx, ry, rz]
        where:
          - u: displacement along local x (axis of element)
          - v: displacement along local y
          - w: displacement along local z
          - rx: rotation about local x (torsion)
          - ry: rotation about local y (bending in local z)
          - rz: rotation about local z (bending in local y)
          
        Parameters:
            E (float): Young's modulus.
            A (float): Cross-sectional area.
            I_y (float): Second moment of area about local y-axis.
            I_z (float): Second moment of area about local z-axis.
            I_rho (float): Torsional constant.
            L (float): Length of the element.
            nu (float): Poisson's ratio (used to compute shear modulus for torsion).
            
        Returns:
            k_local (np.ndarray): 12x12 local stiffness matrix.
        """
        k = np.zeros((12, 12))
        G = E / (2 * (1 + nu))  # Shear modulus
        
        # Axial stiffness
        EA_L = E * A / L
        k[0, 0] =  EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] =  EA_L
        
        # Torsional stiffness
        GJ_L = G * I_rho / L
        k[3, 3] =  GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[9, 9] =  GJ_L
        
        # Bending about local z (affecting DOFs: v (1) and rz (5))
        factor = E * I_z
        k[1, 1]    =  12 * factor / L**3
        k[1, 5]    =   6 * factor / L**2
        k[1, 7]    = -12 * factor / L**3
        k[1, 11]   =   6 * factor / L**2
        k[5, 1]    =   6 * factor / L**2
        k[5, 5]    =   4 * factor / L
        k[5, 7]    =  -6 * factor / L**2
        k[5, 11]   =   2 * factor / L
        k[7, 1]    = -12 * factor / L**3
        k[7, 7]    =  12 * factor / L**3
        k[7, 5]    =  -6 * factor / L**2
        k[7, 11]   =  -6 * factor / L**2
        k[11, 1]   =   6 * factor / L**2
        k[11, 5]   =   2 * factor / L
        k[11, 7]   =  -6 * factor / L**2
        k[11, 11]  =   4 * factor / L
        
        # Bending about local y (affecting DOFs: w (2) and ry (4))
        factor = E * I_y
        k[2, 2]    =  12 * factor / L**3
        k[2, 4]    =  -6 * factor / L**2
        k[2, 8]    = -12 * factor / L**3
        k[2, 10]   =  -6 * factor / L**2
        k[4, 2]    =  -6 * factor / L**2
        k[4, 4]    =   4 * factor / L
        k[4, 8]    =   6 * factor / L**2
        k[4, 10]   =   2 * factor / L
        k[8, 2]    = -12 * factor / L**3
        k[8, 8]    =  12 * factor / L**3
        k[8, 4]    =   6 * factor / L**2
        k[8, 10]   =   6 * factor / L**2
        k[10, 2]   =  -6 * factor / L**2
        k[10, 4]   =   2 * factor / L
        k[10, 8]   =   6 * factor / L**2
        k[10, 10]  =   4 * factor / L
        
        return k

    @staticmethod
    def _transformation_matrix(node_start: Node, node_end: Node, given_local_z):
        """
        Computes the 12x12 transformation matrix that transforms an element stiffness matrix 
        from local to global coordinates.
        
        The local coordinate system is defined as:
          - local x: along the element (from node_start to node_end).
          - local z: determined from the provided vector (given_local_z) and then adjusted to be perpendicular to local x.
          - local y: computed to complete a right-handed coordinate system.
          
        Parameters:
            node_start (Node): Start node.
            node_end (Node): End node.
            given_local_z (array-like): Provided vector for local z-axis orientation.
        
        Returns:
            T (np.ndarray): 12x12 transformation matrix.
        """
        # Compute local x-axis
        delta = node_end.coord - node_start.coord
        L = norm(delta)
        x_local = delta / L

        # Ensure given_local_z is a numpy array and not parallel to x_local
        given_local_z = np.array(given_local_z, dtype=float)
        # Compute local y as the cross product of given_local_z and x_local.
        local_y = np.cross(given_local_z, x_local)
        if norm(local_y) < 1e-8:
            raise ValueError("The provided local_z vector is parallel to the element axis.")
        local_y = local_y / norm(local_y)
        # Recompute local z to ensure orthonormality.
        local_z = np.cross(x_local, local_y)
        
        # Build the 3x3 rotation matrix R: columns are the local axes in global coordinates.
        R = np.column_stack((x_local, local_y, local_z))
        
        # Build the 12x12 transformation matrix (block diagonal with four 3x3 blocks).
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*(i+1), 3*i:3*(i+1)] = R
        return T

    # -------------------------------
    # Assembly and Solver Methods
    # -------------------------------
    
    def _assemble_load_vector(self):
        """
        Assembles the global load vector from the nodal loads.
        """
        for node_id, load in self.loads.items():
            index = 6 * node_id
            self.F_global[index:index+6] = np.array(load, dtype=float)
    
    def _assemble_global_stiffness(self):
        """
        Assembles the global stiffness matrix by looping over each element,
        computing its global stiffness contribution, and adding it to the global matrix.
        """
        for element in self.elements:
            i = element.node_start.id
            j = element.node_end.id
            
            # Compute element length and transformation matrix.
            delta = element.node_end.coord - element.node_start.coord
            L = norm(delta)
            T = self._transformation_matrix(element.node_start, element.node_end, element.local_z)
            
            # Retrieve element properties.
            props = element.properties
            E = props['E']
            nu = props['nu']
            I_y = props['I_y']
            I_z = props['I_z']
            I_rho = props['I_rho']
            A = props['A']
            
            # Compute the local stiffness matrix.
            k_local = self._local_stiffness_matrix(E, A, I_y, I_z, I_rho, L, nu)
            # Transform to global coordinates.
            k_global_elem = T.T @ k_local @ T
            
            # Assemble into the global stiffness matrix.
            dof_indices = np.hstack([np.arange(6*i, 6*i+6), np.arange(6*j, 6*j+6)])
            for a in range(12):
                for b in range(12):
                    self.K_global[dof_indices[a], dof_indices[b]] += k_global_elem[a, b]
    
    def solve(self):
        """
        Applies boundary conditions, solves for nodal displacements, and computes reactions.
        
        Returns:
            U (np.ndarray): Global displacement vector.
            reactions (np.ndarray): Reaction forces/moments at constrained DOFs.
        """
        # Identify constrained (fixed) and free DOFs.
        fixed_dofs = []
        for node_id, dofs in self.boundary_conditions.items():
            fixed_dofs.extend([6*node_id + dof for dof in dofs])
        fixed_dofs = np.array(sorted(fixed_dofs))
        all_dofs = np.arange(self.ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # Partition the stiffness matrix and load vector.
        K_ff = self.K_global[np.ix_(free_dofs, free_dofs)]
        F_f = self.F_global[free_dofs]
        
        # Solve for free displacements.
        U_free = np.linalg.solve(K_ff, F_f)
        self.U[free_dofs] = U_free
        
        # Compute reaction forces at fixed DOFs.
        self.reactions = self.K_global @ self.U - self.F_global
        
        return self.U, self.reactions

    # -------------------------------
    # (Optional) Plotting Functionality
    # -------------------------------
    
    def plot_deformed_shape(self, scale=1.0):
        """
        Plots the deformed shape of the frame (for visualization purposes).
        
        Parameters:
            scale (float): Scaling factor for displacements.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot undeformed structure
        for element in self.elements:
            x = [element.node_start.coord[0], element.node_end.coord[0]]
            y = [element.node_start.coord[1], element.node_end.coord[1]]
            z = [element.node_start.coord[2], element.node_end.coord[2]]
            ax.plot(x, y, z, 'b--', label='Undeformed' if element.id==0 else "")
        
        # Compute deformed nodal coordinates.
        deformed_coords = {}
        for node in self.nodes.values():
            index = 6 * node.id
            displacement = self.U[index:index+3]
            deformed_coords[node.id] = node.coord + scale * displacement
        
        # Plot deformed structure.
        for element in self.elements:
            n1 = deformed_coords[element.node_start.id]
            n2 = deformed_coords[element.node_end.id]
            x = [n1[0], n2[0]]
            y = [n1[1], n2[1]]
            z = [n1[2], n2[2]]
            ax.plot(x, y, z, 'r-', label='Deformed' if element.id==0 else "")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.title('Deformed vs. Undeformed Shape')
        plt.show()

# =============================================================================
# Unit Tests using TDD
# =============================================================================

class TestFrameSolver(unittest.TestCase):
    def setUp(self):
        """
        Sets up a simple cantilever beam example:
          - Two nodes: Node 0 fixed at the origin, Node 1 at (L,0,0).
          - One element connecting them.
          - A vertical load applied at Node 1.
        """
        L = 1.0  # Length of the beam in meters.
        E = 210e9  # Young's modulus in Pascals.
        nu = 0.3
        I_y = 8.1e-6  # m^4 (bending about local z, for deflection in y)
        I_z = 4.05e-6  # m^4 (bending about local y, for deflection in z)
        I_rho = 1e-5  # m^4 (torsional constant)
        A = 0.01   # m^2 (cross-sectional area)
        
        # Define nodes (using zero-indexed ids).
        self.node0 = Node(0, [0, 0, 0])
        self.node1 = Node(1, [L, 0, 0])
        nodes = [self.node0, self.node1]
        
        # Define an element.
        properties = {'E': E, 'nu': nu, 'I_y': I_y, 'I_z': I_z, 'I_rho': I_rho, 'A': A}
        # For this example, choose local z arbitrarily (e.g., global Z-axis)
        element = Element(0, self.node0, self.node1, properties, local_z=[0, 0, 1])
        elements = [element]
        
        # Define loads: apply a downward force (in global Y) at node 1.
        # Note: In the local coordinate system, if the beam is horizontal along X and local y is vertical, then
        # a global load in the Y-direction corresponds to the local v direction.
        loads = {1: [0, -1000.0, 0, 0, 0, 0]}  # Force of -1000 N in the Y direction at node 1.
        
        # Define boundary conditions: Node 0 is fully fixed.
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        
        self.solver = FrameSolver(nodes, elements, loads, boundary_conditions)
    
    def test_global_stiffness_symmetry(self):
        """Test that the assembled global stiffness matrix is symmetric."""
        K = self.solver.K_global
        self.assertTrue(np.allclose(K, K.T, atol=1e-8), "Global stiffness matrix is not symmetric.")
    
    def test_displacement_vector_size(self):
        """Test that the displacement vector has the correct size."""
        U, _ = self.solver.solve()
        self.assertEqual(len(U), self.solver.ndof, "Displacement vector size mismatch.")
    
    def test_nonzero_displacement_under_load(self):
        """Test that a load leads to nonzero displacements at the free node."""
        U, _ = self.solver.solve()
        # Node 1 is free; its displacement indices are 6 to 11.
        U_node1 = U[6:12]
        self.assertFalse(np.allclose(U_node1, np.zeros(6)), "Displacements at the loaded node should not be zero.")
    
    def test_reaction_equilibrium(self):
        """Test that the computed reactions at the fixed node balance the applied load."""
        U, reactions = self.solver.solve()
        # For the fixed node (node 0), check that the reaction in the Y-direction is approximately +1000 N.
        reaction_node0 = reactions[0:6]
        # Because of sign conventions, the reaction should counteract the applied load.
        self.assertAlmostEqual(reaction_node0[1], 1000.0, delta=50.0, msg="Reaction force in Y at fixed node is unexpected.")
    
    def test_transformation_matrix_orthonormality(self):
        """Test that the computed rotation matrix (from the transformation matrix) is orthonormal."""
        T = FrameSolver._transformation_matrix(self.node0, self.node1, [0, 0, 1])
        # Extract one of the 3x3 blocks (they are identical).
        R = T[0:3, 0:3]
        I3 = np.eye(3)
        self.assertTrue(np.allclose(R.T @ R, I3, atol=1e-8), "Rotation matrix is not orthonormal.")

# =============================================================================
# Main Execution (Run Tests)
# =============================================================================

if __name__ == "__main__":
    unittest.main()
