'''
    Author: Callum James Gill
    Email: cjg772@student.bham.ac.uk
    Creation date: 22/12/19
    Date submitted: 24/1/20
    Description: Plots all the billiard maps and phase plots
    
    Note: this may take some time to run. On my machine, a 2017 macbook pro,
    it had a runtime of 116 seconds.
'''
import os
import numpy as np
import BilliardClasses as BC
from matplotlib import pyplot as plt

### GLOBAL VARIABLES ###
# All variables below are constants
CSV_DIR = r"Csv files/" # Folder to read .csv files from.
# In this form to allow it to work on Mac
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_DIR = os.path.join(CURRENT_DIR, CSV_DIR)
# Folder to save the plots in
FIG_DIR = 'Plot images/'
FIG_DIR = os.path.join(CURRENT_DIR, FIG_DIR)
# Number of collisions to calculate
NUM_COLL = 200

### FUNCTIONS ###
def printPositionPlot(board_list, title):
    '''
        Plots the balls x and y positions w.r.t. the board,
        i.e. plots the billiard map.

        Parameters:

            board_list : list of 2D numpy arrays
                A list of <BilliardBoard> objects

            title : string
                the name of the board

        Returns:
            None
    '''
    # Name for the file to save the figure to
    filename = "Billiard map plot for %s.jpeg" % title
    filename = os.path.join(FIG_DIR, filename)
    title = "Billiard maps for the %s board" % title
    # Generate basic plot properties
    fig, ax = plt.subplots(nrows=1, ncols=len(board_list),
                           sharex=True, sharey=True)
    fig.suptitle(title, fontsize=14, y=1)
    for i in range(len(board_list)):
        # Retrieve all relevant data
        collision_points = board_list[i].collision_points
        # Set up the ax variables
        ax[i].set_xlabel(r'$x$')
        ax[i].set_ylabel(r'$y$')
        ax[i].axis('equal')
        # Draw the board
        board_list[i].drawBoard(ax[i], "Board")
        # Draws the lines between collision points, i.e.
        ax[i].plot(collision_points[0], collision_points[1],
          color="b", linewidth=0.5, label="Ball's trajectory")
        ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, 1))
        
    fig.tight_layout()
    fig.savefig(filename, format='jpeg', quality=95, dpi=600)

def printPhasePlots(positions_list, velocities_list, angles, main_plot, title):
    '''
        Plots all the phase plots for the billiard map onto a single
        figure.

        Parameters:
            positions_list : list of 2D numpy arrays
                A list of 2D numpy arrays which contain all the collision 
                points with x, y components

            velocities_list : list of 2D numpy arrays
                A list of 2D numpy arrays which contain all the velocities at 
                each collision with v_x, v_y components

            angles_list : list of 1D numpy arrays
                A list of 1D numpy arrays which contain all the angles to the 
                normal of the board at each collision. Measured in radians.
                
            main_plot : tuple of ints (i, j)
                Index of the subplot to use as an individual plot. i is the row
                and j is the column

            title : string
                the name of the board
            
        Returns:
            None
    '''
    # Name for the file to save the 4 subplot figure to
    filename = "Phase plots for %s.jpeg" % title
    filename = os.path.join(FIG_DIR, filename)
    filename_main = "Main phase plot for %s.jpeg" % title
    filename_main = os.path.join(FIG_DIR, filename_main)
    # Generate basic plot properties
    fig, ax = plt.subplots(2, 2)
    # Generate the individual plot
    index_row, index_col = main_plot
    fig_main, ax_main = plt.subplots()
    ax_main.plot()
    ax_main.set_title("Phase portrait for %s" % title)
    x_labels = [r'$x$ ($m$)', r'$y$ ($m$)']
    y_labels = [r'$v_x$ ($ms^{-1}$)', r'$v_y$ ($ms^{-1}$)']
    # Retrieve and compute all relevant data
    title = "Phase portraits for the " + title + " board"
    fig.suptitle(title, fontsize=14, y = 1)
    for i in range(len(positions_list)):
        collision_points = positions_list[i]
        velocities = velocities_list[i]
        # Checks cases where the shapes of the arrays don't match by 1 integer
        # These cases are for when corner hits occur, i.e. less collisions.
        # Therefore the final collision is stored for plotting the billiard map
        # but not the velocity. So the final collision point is removed
        coll_shape = collision_points.shape
        vel_shape = velocities.shape
        if coll_shape[1] != vel_shape[1]:
            collision_points = collision_points[:, :coll_shape[1]-1]
        # Plot the points for the particular trajectory
        for j in range(ax[0].size):
            data = (collision_points[j], velocities[j])
            # Labels
            ax[0, j].grid()
            ax[0, j].set_xlabel(x_labels[j])
            ax[0, j].set_ylabel(y_labels[j])
            # Draws the points in the phase space. Each new board has a 
            # different colour.
            ax[0, j].plot(*data, marker='.',
              markersize = 0.05, linestyle = 'none')
            if index_row == 0 and index_col == j:
                # Plot the final individual plot
                ax_main.plot(*data, marker='.',
                             markersize = 0.5, linestyle = 'none')
                ax_main.set_xlabel(x_labels[j])
                ax_main.set_ylabel(y_labels[j])
        # For the Third plot, it plots theta vs. the reflection angle (called
        # phi here), where theta is the angle the balls position vector makes 
        # with the positive x-axis and the reflection angle is the angle 
        # subtended by the balls velocity vector and the normal vector to the 
        # board boundary.
        theta = np.arctan2(collision_points[1], collision_points[0])
        data = (theta, angles[i])
        ax[1, 0].grid()
        ax[1, 0].set_xlabel(r'$\theta$ (rad)')
        ax[1, 0].set_ylabel(r'$\phi$ (rad)')
        ax[1, 0].plot(*data, marker='.', markersize = 0.05, linestyle = 'none')
        if index_row == 1 and index_col == 0:
            # Plot the final individual plot
            ax_main.plot(*data, marker='.',
                         markersize = 0.5, linestyle = 'none')
            ax_main.set_xlabel(r'$\theta$ (rad)')
            ax_main.set_ylabel(r'$\phi$ (rad)')
        # Final plot is a parameterised phase space. This parameterises the 
        # ball's position as a distance in polar coordinates, r, and plots it
        # against the reflection angle.
        r = np.sqrt(collision_points[0] ** 2 + collision_points[1] ** 2)
        data = (r, angles[i])
        ax[1, 1].grid()
        ax[1, 1].set_xlabel(r'$r$ (m)')
        ax[1, 1].set_ylabel(r'$\phi$ (rad)')
        ax[1, 1].plot(*data, marker='.', markersize = 0.05, linestyle = 'none')
        if index_row == 1 and index_col == 1:
            # Plot the final individual plot
            ax_main.plot(*data, marker='.',
                         markersize = 0.5, linestyle = 'none')
            ax_main.set_xlabel(r'$r$ (m)')
            ax_main.set_ylabel(r'$\phi$ (rad)')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.7) #<- Adjust vertical height between subplots
    fig.savefig(filename, format = 'jpeg', quality = 95, dpi=600)
    fig_main.savefig(filename_main, format = 'jpeg', quality = 95, dpi=600)
    plt.close(fig) # <- frees up memory
    plt.close(fig_main) # <- frees up memory
    

def retrieveCsvData(board, type_of_data):
    '''
        Retrieves the data from a .csv file and stores it as a numpy array.
        Returns a list of numpy arrays.

        Parameters:
            board : string
                the name of the board

            type_of_data : string
                the type of data to retrieve, e.g. 'Collisions', 'Angles', 
                'Velocities'
            
        Returns:
            list_of_nparray : list of numpy arrays
                list of numpy arrays where each array corresponds to a single 
                board object
    '''
    list_of_nparray = []
    i = 0 # File number; is increased for checking each file
    # Max number of data files to check will be 500. If there is less then the 
    # loop is broken
    while i < 500:
        # For i = 0, the file doesn't have a number
        if i > 0:
            file_path_1 = board + '/' + board + ' Phase '
            file_path_2 = type_of_data + ' ' + str(i)
        else:
            file_path_1 = board + '/' + board
            file_path_2 = ' Phase ' + type_of_data
        file_path = file_path_1 + file_path_2 + '.csv'
        full_path = os.path.join(CSV_DIR, file_path)
        if os.path.isfile(full_path):
            array = np.genfromtxt(full_path, delimiter = ",")
            list_of_nparray.append(array)
        else:
            # Max num files found before i reaches a value of 500
            break
        i += 1
    return list_of_nparray

### MAIN ###

print('0%') # <- Print programs progress

## SQUARE ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
square_start_position = np.array([[0.2], [0]])
# Vertices of the square, going from 
square_vertices = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
# Parameters of the square board class to use
sq_para_1 = (NUM_COLL, square_vertices, square_start_position, np.pi/8)
sq_para_2 = (NUM_COLL, square_vertices, square_start_position, 3 * np.pi/8)
square_board_1 = BC.PolygonBilliardBoard(*sq_para_1)
square_board_2 = BC.PolygonBilliardBoard(*sq_para_2)
square_boards = [square_board_1, square_board_2]
printPositionPlot(square_boards, 'Square')
# Phase plot #
# Retrieve the data from the .csv file
sq_phase_coll = retrieveCsvData('Square', 'Collisions')
sq_phase_vel = retrieveCsvData('Square', 'Velocities')
sq_phase_angles = retrieveCsvData('Square', 'Angles')
printPhasePlots(sq_phase_coll, sq_phase_vel, sq_phase_angles, (1, 0), 'square')

print('10%') # <- Print programs progress

## TRIANGLE ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
tri_start_position = np.zeros((2, 1))
# Parameters of the triangle board class to use
tri_para_1 = (NUM_COLL, 'triangle', tri_start_position, np.pi/8)
tri_para_2 = (NUM_COLL, 'triangle', tri_start_position, 3 * np.pi/8)
tri_board_1 = BC.PolygonBilliardBoard(*tri_para_1)
tri_board_2 = BC.PolygonBilliardBoard(*tri_para_2)
tri_boards = [tri_board_1, tri_board_2]
printPositionPlot(tri_boards, 'Triangle')
# Phase plot #
# Retrieve the data from the .csv file
tri_phase_coll = retrieveCsvData('Triangle', 'Collisions')
tri_phase_vel = retrieveCsvData('Triangle', 'Velocities')
tri_phase_angles = retrieveCsvData('Triangle', 'Angles')
printPhasePlots(tri_phase_coll, tri_phase_vel, tri_phase_angles,
                (1, 0), 'triangle')

print('20%') # <- Print programs progress

## CIRCLE ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
circ_start_position = np.array([[-0.5], [0]])
# Parameters of the circle board class to use
circ_para_1 = (NUM_COLL, (1, 1), circ_start_position, np.pi/8)
circ_para_2 = (NUM_COLL, (1, 1), circ_start_position, 3 * np.pi/8)
circ_board_1 = BC.EllipticalBilliardBoard(*circ_para_1)
circ_board_2 = BC.EllipticalBilliardBoard(*circ_para_2)
circ_boards = [circ_board_1, circ_board_2]
printPositionPlot(circ_boards, 'Circle')
# Phase plot #
# Retrieve the data from the .csv file
circ_phase_coll = retrieveCsvData('Circle', 'Collisions')
circ_phase_vel = retrieveCsvData('Circle', 'Velocities')
circ_phase_angles = retrieveCsvData('Circle', 'Angles')
printPhasePlots(circ_phase_coll, circ_phase_vel, circ_phase_angles,
                (0, 0), 'Circle')

print('30%') # <- Print programs progress

## ELLIPSE ##

# Billiard map #
# Two billiard maps, same starting launch angle but different start positions
ellipse_start_position_1 = np.array([[-0.1], [0]])
ellipse_start_position_2 = np.array([[-1.9], [0]])
# Parameters of the ellipse board class to use
ellipse_para_1 = (NUM_COLL, (2, 1), ellipse_start_position_1, np.pi/8)
ellipse_para_2 = (NUM_COLL, (2, 1), ellipse_start_position_2, np.pi/8)
ellipse_board_1 = BC.EllipticalBilliardBoard(*ellipse_para_1)
ellipse_board_2 = BC.EllipticalBilliardBoard(*ellipse_para_2)
ellipse_boards = [ellipse_board_1, ellipse_board_2]
printPositionPlot(ellipse_boards, 'Ellipse')
# Phase plot #
# Retrieve the data from the .csv file
ellipse_phase_coll = retrieveCsvData('Ellipse', 'Collisions')
ellipse_phase_vel = retrieveCsvData('Ellipse', 'Velocities')
ellipse_phase_angles = retrieveCsvData('Ellipse', 'Angles')
printPhasePlots(ellipse_phase_coll, ellipse_phase_vel, ellipse_phase_angles,
                (0, 1), 'Ellipse')

print('40%') # <- Print programs progress

## HYPERBOLIC ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
hb_start = np.zeros((2, 1))
# Parameters of the hyperbolic board class to use
hb_para_1 = (NUM_COLL, (1, 1), hb_start, np.pi/8)
hb_para_2 = (NUM_COLL, (1, 1), hb_start, 3*np.pi/8)
hb_board_1 = BC.HyperbolicBilliardBoard(*hb_para_1)
hb_board_2 = BC.HyperbolicBilliardBoard(*hb_para_2)
hb_boards = [hb_board_1, hb_board_2]
printPositionPlot(hb_boards, 'Hyperbolic')
# Phase plot #
# Retrieve the data from the .csv file
hb_phase_coll = retrieveCsvData('Hyperbolic', 'Collisions')
hb_phase_vel = retrieveCsvData('Hyperbolic', 'Velocities')
hb_phase_angles = retrieveCsvData('Hyperbolic', 'Angles')
printPhasePlots(hb_phase_coll, hb_phase_vel, hb_phase_angles,
                (1, 1), 'Hyperbolic')

print('50%') # <- Print programs progress

## STADIUM ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
stad_start = np.array([[0.5], [0]])
# Parameters of the hyperbolic board class to use
stad_para_1 = (NUM_COLL, (1, 1, 0.5), stad_start, np.pi/8)
stad_para_2 = (NUM_COLL, (1, 1, 0.5), stad_start, 3*np.pi/8)
stad_board_1 = BC.StadiumBilliardBoard(*stad_para_1)
stad_board_2 = BC.StadiumBilliardBoard(*stad_para_2)
stad_boards = [stad_board_1, stad_board_2]
printPositionPlot(stad_boards, 'Stadium')
# Phase plot #
# Retrieve the data from the .csv file
stad_phase_coll = retrieveCsvData('Stadium', 'Collisions')
stad_phase_vel = retrieveCsvData('Stadium', 'Velocities')
stad_phase_angles = retrieveCsvData('Stadium', 'Angles')
printPhasePlots(stad_phase_coll, stad_phase_vel, stad_phase_angles,
                (0, 0), 'Stadium')

print('60%') # <- Print programs progress

## BUNIMOVICH ##

# Billiard map #
# Two billiard maps, same starting position but different launch angles
bunimovich_start = np.array([[0.5], [0]])
# Parameters of the hyperbolic board class to use
bunimovich_para_1 = (NUM_COLL, (1, 1, -0.25), bunimovich_start, np.pi/8)
bunimovich_para_2 = (NUM_COLL, (1, 1, -0.25), bunimovich_start, 3*np.pi/8)
bunimovich_board_1 = BC.BunimovichBilliardBoard(*bunimovich_para_1)
bunimovich_board_2 = BC.BunimovichBilliardBoard(*bunimovich_para_2)
bunimovich_boards = [bunimovich_board_1, bunimovich_board_2]
printPositionPlot(bunimovich_boards, 'Bunimovich')
# Phase plot #
# Retrieve the data from the .csv file
bunimovich_phase_coll = retrieveCsvData('Bunimovich', 'Collisions')
bunimovich_phase_vel = retrieveCsvData('Bunimovich', 'Velocities')
bunimovich_phase_angles = retrieveCsvData('Bunimovich', 'Angles')
printPhasePlots(bunimovich_phase_coll, bunimovich_phase_vel,
                bunimovich_phase_angles, (1, 1), 'Bunimovich')

print('70%') # <- Print programs progress

## MUSHROOM ##

# Billiard map #
# Two billiard maps, different starting positions and launch angles
mush_start_1 = np.array([[-0.9], [0.1]])
mush_start_2 = np.array([[0], [-0.7]])
# Parameters of the hyperbolic board class to use
mush_para_1 = (NUM_COLL, (1, 1, 1, 3), mush_start_1, np.pi/8)
mush_para_2 = (NUM_COLL, (1, 1, 1, 3), mush_start_2, np.pi/9)
mush_board_1 = BC.MushroomBilliardBoard(*mush_para_1)
mush_board_2 = BC.MushroomBilliardBoard(*mush_para_2)
mush_boards = [mush_board_1, mush_board_2]
printPositionPlot(mush_boards, 'Mushroom')
# Phase plot #
# Retrieve the data from the .csv file
mush_phase_coll = retrieveCsvData('Mushroom', 'Collisions')
mush_phase_vel = retrieveCsvData('Mushroom', 'Velocities')
mush_phase_angles = retrieveCsvData('Mushroom', 'Angles')
printPhasePlots(mush_phase_coll, mush_phase_vel, mush_phase_angles,
                (0, 1), 'Mushroom')

print('80%') # <- Print programs progress

## LORENTZ GAS ##

# Billiard map #
# Two billiard maps, same starting positions and different launch angles
lg_start = np.array([[-1.2], [0]])
# Parameters of the hyperbolic board class to use
lg_para_1 = (NUM_COLL, None, lg_start, np.pi/8)
lg_para_2 = (NUM_COLL, None, lg_start, np.pi/9)
lg_board_1 = BC.LorentzGasBilliardBoard(*lg_para_1)
lg_board_2 = BC.LorentzGasBilliardBoard(*lg_para_2)
lg_boards = [lg_board_1, lg_board_2]
printPositionPlot(lg_boards, 'Lorentz Gas')
# Phase plot #
# Retrieve the data from the .csv file
lg_phase_coll = retrieveCsvData('Lorentz Gas', 'Collisions')
lg_phase_vel = retrieveCsvData('Lorentz Gas', 'Velocities')
lg_phase_angles = retrieveCsvData('Lorentz Gas', 'Angles')
printPhasePlots(lg_phase_coll, lg_phase_vel, lg_phase_angles, 
                (0, 1), 'Lorentz Gas')

print('90%') # <- Print programs progress

## HYPERBOLIC LORENTZ GAS ##

# Billiard map #
# Two billiard maps, same starting positions and different launch angles
hlg_start = np.array([[0], [1.5]])
# Parameters of the hyperbolic board class to use
hlg_para_1 = (NUM_COLL, None, hlg_start, np.pi/8)
hlg_para_2 = (NUM_COLL, None, hlg_start, np.pi/9)
hlg_board_1 = BC.HyperbolicLorentzGasBilliardBoard(*hlg_para_1)
hlg_board_2 = BC.HyperbolicLorentzGasBilliardBoard(*hlg_para_2)
hlg_boards = [hlg_board_1, hlg_board_2]
printPositionPlot(hlg_boards, 'Hyperbolic Lorentz Gas')
# Phase plot #
# Retrieve the data from the .csv file
hlg_phase_coll = retrieveCsvData('Hyperbolic Lorentz Gas', 'Collisions')
hlg_phase_vel = retrieveCsvData('Hyperbolic Lorentz Gas', 'Velocities')
hlg_phase_angles = retrieveCsvData('Hyperbolic Lorentz Gas', 'Angles')
printPhasePlots(hlg_phase_coll, hlg_phase_vel, hlg_phase_angles,
                (0, 1), 'Hyperbolic Lorentz Gas')

#plt.show() #<- Commented out to allow plt.close(fig) to free up memory

print('100%') # <- Print programs progress