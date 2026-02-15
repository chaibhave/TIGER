from ExodusReader import ExodusReader

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib
import numpy as np
from time import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from plotting_utils import setup_publication_style, get_scientific_colormap, save_figure

#SETUP PUBLICATION-QUALITY PLOTTING STYLE
setup_publication_style(use_latex=False)

#EXODUS FILE FOR RENDERING
#ANY CHARACTER(S) CAN BE PLACED IN PLACE OF THE *, EG. 2D/grain_growth_2D_graintracker_out.e.1921.0000 or 2D/grain_growth_2D_graintracker_out.e-s001
filenames = '3D/grain_growth_3D_out.e*'                             #Star represents all files following this template

#GENERATE FIGURE WINDOW AND SUBPLOT AXIS
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#READ EXODUS FILE SERIES WITH ExodusReader (USING CONTEXT MANAGER)
with ExodusReader(filenames) as MF:
    #GET A LIST OF SIMULATION TIME POINTS
    times = MF.global_times

    #GET X,Y,Z AND C (UNIQUE GRAINS VALUES) AT CURRENT TIME STEP
    x,y,z,c = MF.get_data_at_time('unique_grains',MF.global_times[-1])               #Read coordinates and variable value --> Will be parallelized in future

    #CREATE A COLORMAP OBJECT (cw) THAT CONVERTS VARIABLE VALUE c INTO A RGBA VALUE
    c_min = np.amin(c)
    c_max = np.amax(c)
    #USING BATLOW COLORMAP (PERCEPTUALLY UNIFORM ALTERNATIVE TO HSV)
    #NOTE: HSV is not perceptually uniform and can be misleading. Batlow provides
    #a perceptually uniform rainbow alternative that is better for scientific visualization.
    cmap_scientific = get_scientific_colormap('batlow')
    cw = matplotlib.cm.ScalarMappable(cmap=cmap_scientific)
    cw.set_array([c_min,c_max])

    #CONVERT VARIABLE VALUES TO RGBA VALUES
    C = cw.to_rgba(c)

    #GENERATE CORNER POINT COORDINATES FOR THE QUAD8 MESH POLYGONS
    coords = np.dstack([x,z,y])

    #REPEAT THE COLORING ARRAY 3 TIMES, ONCE FOR EACH SURFACE
    colors = np.repeat(C,3,axis=0)
    #GENERATE THE 3 SIDES FOR EACH POLYGON, AND ASSIGN THE CELL VALUE OF THE POLYGON TO THE sides --> WE ONLY RENDER 3 SIDES OF EACH CUBE IN THE POLYGON
    surfaces = np.ndarray((coords.shape[0]*3,4,3))
    surfaces[0::3]=coords[:,[0,1,2,3],:]
    surfaces[1::3]=coords[:,[3,2,6,7],:]
    surfaces[2::3]=coords[:,[4,7,3,0],:]

    #CREATE A Poly3DCollection FROM OUR SURFACES
    # Add edge colors to make voxel structure visible
    P = Poly3DCollection(surfaces, facecolors=colors, edgecolors='k', linewidths=0.1, alpha=0.95)

    #PLOT THE POLY3DCOLLECTION ON OUR AXIS
    collection = ax.add_collection3d(P)

#FIGURE FORMATTING SETTINGS
ax.set_xlim([0,1000])                                                                   #You can use x and y arrays for setting this, but usually it is easier to manually set
ax.set_ylim([0,1000])
ax.set_zlim([0,1000])

#MAKE AXIS SPINES BLACK AND VISIBLE TO SHOW CUBE SHAPE
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.set_linewidth(2)
ax.yaxis.pane.set_linewidth(2)
ax.zaxis.pane.set_linewidth(2)

#SET PANE COLORS TO TRANSPARENT FOR VISIBILITY
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

#MAKE GRID LINES VISIBLE
ax.grid(True, linewidth=0.5, alpha=0.3)

#MAKE ALL AXIS LINES BLACK AND THICK
ax.xaxis.line.set_color('black')
ax.yaxis.line.set_color('black')
ax.zaxis.line.set_color('black')
ax.xaxis.line.set_linewidth(2)
ax.yaxis.line.set_linewidth(2)
ax.zaxis.line.set_linewidth(2)

#SET TICK PARAMETERS TO MAKE THEM VISIBLE
ax.tick_params(axis='x', colors='black', width=2)
ax.tick_params(axis='y', colors='black', width=2)
ax.tick_params(axis='z', colors='black', width=2)

#DRAW COMPLETE WIREFRAME BOX TO SHOW CUBE BOUNDARY
#Define the 12 edges of the cube
xmin, xmax = 0, 1000
ymin, ymax = 0, 1000
zmin, zmax = 0, 1000

#Bottom face (z=zmin)
ax.plot([xmin,xmax], [ymin,ymin], [zmin,zmin], 'k-', linewidth=2)
ax.plot([xmax,xmax], [ymin,ymax], [zmin,zmin], 'k-', linewidth=2)
ax.plot([xmax,xmin], [ymax,ymax], [zmin,zmin], 'k-', linewidth=2)
ax.plot([xmin,xmin], [ymax,ymin], [zmin,zmin], 'k-', linewidth=2)

#Top face (z=zmax)
ax.plot([xmin,xmax], [ymin,ymin], [zmax,zmax], 'k-', linewidth=2)
ax.plot([xmax,xmax], [ymin,ymax], [zmax,zmax], 'k-', linewidth=2)
ax.plot([xmax,xmin], [ymax,ymax], [zmax,zmax], 'k-', linewidth=2)
ax.plot([xmin,xmin], [ymax,ymin], [zmax,zmax], 'k-', linewidth=2)

#Vertical edges
ax.plot([xmin,xmin], [ymin,ymin], [zmin,zmax], 'k-', linewidth=2)
ax.plot([xmax,xmax], [ymin,ymin], [zmin,zmax], 'k-', linewidth=2)
ax.plot([xmax,xmax], [ymax,ymax], [zmin,zmax], 'k-', linewidth=2)
ax.plot([xmin,xmin], [ymax,ymax], [zmin,zmax], 'k-', linewidth=2)

#SET VIEWING ANGLE FOR REPRODUCIBILITY
#NOTE: Default view is elevation=30, azimuth=-60. Adjust these for desired perspective.
#ax.view_init(elev=30, azim=-60)  # Uncomment and adjust as needed

#CREATE COLORBAR FROM OUR COLORBAR OBJECT CW
fig.colorbar(cw, ax=ax, label="Unique Grains")

#SAVE FIGURE USING save_figure FOR CONSISTENT OUTPUT
save_figure(fig, '3D/3d_render', formats=['png', 'pdf'], dpi=500, transparent=True)

#USE PLT.SHOW FOR AN INTERACTIVE 3D DISPLAY OF IMAGE. THIS CAN BE A BIT SLOW TO MANIPULATE DEPENDING ON EXODUS FILE SIZE
plt.show()
