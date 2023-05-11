import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh
import trimesh

mesh = trimesh.load('/home/chi-onn/fyp_ws/src/meshes/bowl.STL')
n_points_desired = 3000
mesh_simplified = mesh.simplify_quadric_decimation(n_points_desired)
trimesh.exchange.export.export_mesh(mesh_simplified, '/home/chi-onn/fyp_ws/src/meshes/simplified_bowl.STL', file_type='stl')


# Load the STL file
your_mesh = trimesh.load('/home/chi-onn/fyp_ws/src/meshes/simplified_bowl.STL')



# Get the vertices and faces of the mesh
vertices = your_mesh.vertices
faces = your_mesh.faces

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, color = (0, 1, 0),alpha=0.3)

# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()