'''
    Author: Callum James Gill
    Email: cjg772@student.bham.ac.uk
    Creation date: 18/1/20
    Date submitted: 24/1/20
    Description: Produce all of the data used in the phase plots and stores 
        them in individual csv files. This is built into numpy and is much 
        easier to use to store the multi-dimensional arrays.
    
    NOTE: Since nested for loops are imployed which contain multiple class 
    method calls from instatiating the board classes within the inner loop,
    the running of the code in this file takes a very long time. 
    For my machine, a MacBook Pro 2017 model, this took 240 seconds.
'''

import os
import numpy as np
import BilliardClasses as BC

### GLOBAL VARIABLES ###
# All uppercase variables below are constants
CSV_DIR = r"Csv files/" # <- Folder to write .txt files too.
# <- This form to allow it to work on Mac
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_DIR = os.path.join(CURRENT_DIR, CSV_DIR)
# Number of collisions to calculate
NUM_COLL = 200
# 10 angles between 0 and 2pi, linearly spaced
START_ANGLES = np.linspace(0, 2*np.pi, 10, endpoint = False)
# Start position for each board
START_POS = np.zeros((2, 50))
# Change the x values to linear spaced values
START_POS[0] = np.linspace(-0.9, 0.9, 50)

### FUNCTIONS ###
def saveArrayToCsv(boards, board_name):
    '''
        Saves the arrays storing a boards collision points, velocities, and 
        angles to inidividual .csv files.

        Parameters:
            boards : list of <BilliardBoard> object or <BilliardBoard> object
                list of billiard board objects to save the data to
                a .txt file from. If its a <BilliardBoard> object then it shall
                be converted to a 1 element list.

            board_name: string
                the board name, e.g. for a square board it would be
                "Square" and for an ellipse "Ellipse" etc.

        Returns:
            None
    '''
    # Set the filepath
    f_path = board_name + "/" + board_name + " Phase"
    # Convert the boards variable to a list if only an boards object has 
    # been passed
    if not isinstance(boards, list):
        boards = [boards]
    # Iterate over all the boards
    for i in range(len(boards)):
        board = boards[i] # Get board object from list
        # Directionary name to save to, if first file to save then don't 
        # include a number
        if i > 0:
            collision_points_path = f_path + " Collisions " + str(i) + '.csv'
            velocities_path = f_path + " Velocities " + str(i) + '.csv'
        else:
            collision_points_path = f_path + " Collisions.csv"
            velocities_path = f_path + " Velocities.csv"
        # Turn path into a readable path for the OS
        collision_points_path = os.path.join(CSV_DIR, collision_points_path)
        velocities_path = os.path.join(CSV_DIR, velocities_path)
        # Retrieve the collision points and velocities arrays
        collision_points = board.collision_points
        velocities = board.velocities
        # Save the arrays to .txt files
        np.savetxt(collision_points_path, collision_points, delimiter=',')
        np.savetxt(velocities_path, velocities, delimiter=',')
        # Retrieve the angles to the normal array
        angles = board.angles_to_normal
        # Directionary name to save to, if first file to save then don't 
        # include a number
        if i > 0:
            angles_path = f_path + " Angles " + str(i) + '.csv'
        else:
            angles_path = f_path + " Angles.csv"
        # Turn path into a readable path for the OS
        angles_path = os.path.join(CSV_DIR, angles_path)
        np.savetxt(angles_path, angles, delimiter=',')


### MAIN ###
print('0%') # <- Print programs progress

## SQUARE ##
# Generate the square board classes
square_boards = [] #<- stores each of the square board classes
for i in range(START_ANGLES.size):
    # Square won't work for angles 0, pi/2, pi, 3pi/2 and 2pi
    # If statement has following form for readability
    if START_ANGLES[i] == 0 or START_ANGLES[i] == np.pi/2:
        continue
    elif START_ANGLES[i] == np.pi or START_ANGLES[i] == 3*np.pi/2:
        continue
    for j in range(START_POS[0].size):
        start_position = START_POS[:, j].reshape(2, 1)
        sq_para = (NUM_COLL, 'square', start_position, START_ANGLES[i])
        square_board = BC.PolygonBilliardBoard(*sq_para)
        square_boards.append(square_board)
        # Class has been stored in list, so destroy the local copy
        del square_board
saveArrayToCsv(square_boards, 'Square')

print('10%') # <- Print programs progress

## Triangle ##
# Generate the triangle board classes
triangle_boards = [] #<- stores each of the triangle board classes
tri_start = START_POS * 0.5
for i in range(START_ANGLES.size):
    for j in range(tri_start[0].size):
        start_position = tri_start[:, j].reshape(2, 1)
        tri_para = (NUM_COLL, 'triangle', start_position, START_ANGLES[i])
        triangle_board = BC.PolygonBilliardBoard(*tri_para)
        triangle_boards.append(triangle_board)
        # Class has been stored in list, so destroy the local copy
        del triangle_board
saveArrayToCsv(triangle_boards, 'Triangle')

print('20%') # <- Print programs progress

## Circle ##
# Generate the circle board classes
circle_boards = [] #<- stores each of the circle board classes
for i in range(START_ANGLES.size):
    for j in range(START_POS[0].size):
        start_position = START_POS[:, j].reshape(2, 1)
        circle_para = (NUM_COLL, (1, 1), start_position, START_ANGLES[i])
        circle_board = BC.EllipticalBilliardBoard(*circle_para)
        circle_boards.append(circle_board)
        # Class has been stored in list, so destroy the local copy
        del circle_board
saveArrayToCsv(circle_boards, 'Circle')

print('30%') # <- Print programs progress

## Ellipse ##
# Generate the circle board classes
ell_boards = [] #<- stores each of the circle board classes
ell_start = START_POS * 2
for i in range(START_ANGLES.size):
    for j in range(ell_start[0].size):
        start_position = ell_start[:, j].reshape(2, 1)
        ell_para = (NUM_COLL, (2, 1), start_position, START_ANGLES[i])
        ell_board = BC.EllipticalBilliardBoard(*ell_para)
        ell_boards.append(ell_board)
        # Class has been stored in list, so destroy the local copy
        del ell_board
saveArrayToCsv(ell_boards, 'Ellipse')

print('40%') # <- Print programs progress

## Hyperbolic ##
# Generate the hyperbolic board classes
hb_boards = [] #<- stores each of the hyperbolic board classes
# Removes first, second and last angles as they don't produce any points
hb_angles = START_ANGLES[2:START_ANGLES.size]
for i in range(hb_angles.size):
    for j in range(START_POS[0].size):
        start_position = START_POS[:, j].reshape(2, 1)
        hb_para = (NUM_COLL, (1, 1), start_position, hb_angles[i])
        hb_board = BC.HyperbolicBilliardBoard(*hb_para)
        hb_boards.append(hb_board)
        # Class has been stored in list, so destroy the local copy
        del hb_board
saveArrayToCsv(hb_boards, 'Hyperbolic')

print('50%') # <- Print programs progress

## Stadium ##
# Generate the stadium board classes
stad_boards = [] #<- stores each of the stadium board classes
# Moves the starting positions to just above the x-axis to avoid divide by zero
# errors. Also doesn't seem to work for theta = 0 rad; divide by zero error
# encountered in StadiumBilliardBoard class and I'm not too sure how to resolve
# it so I'm going to emit it.
stad_start = START_POS
stad_start[1] = stad_start[1] + 0.1
stad_angles = START_ANGLES[1:]
for i in range(stad_angles.size):
    for j in range(stad_start[0].size):
        start_position = stad_start[:, j].reshape(2, 1)
        stad_para = (NUM_COLL, (1, 1, 0.5), start_position, stad_angles[i])
        stad_board = BC.StadiumBilliardBoard(*stad_para)
        stad_boards.append(stad_board)
        # Class has been stored in list, so destroy the local copy
        del stad_board
saveArrayToCsv(stad_boards, 'Stadium')

print('60%') # <- Print programs progress

## Bunimovich ##
# Generate the Bunimovich board classes
bunimovich_boards = [] #<- stores each of the Bunimovich board classes
# Moves the starting positions to just above the x-axis to avoid divide by zero
# errors. Not sure why this is occuring but the phase plots can still be
# produced nicely this way
bunimovich_start = START_POS
bunimovich_start[1] = bunimovich_start[1] + 0.1
for i in range(START_ANGLES.size):
    for j in range(bunimovich_start[0].size):
        start_position = bunimovich_start[:, j].reshape(2, 1)
        bunimovich_para = (NUM_COLL, (1, 1, -0.25),
                           start_position, START_ANGLES[i])
        bunimovich_board = BC.BunimovichBilliardBoard(*bunimovich_para)
        bunimovich_boards.append(bunimovich_board)
        # Class has been stored in list, so destroy the local copy
        del bunimovich_board
saveArrayToCsv(bunimovich_boards, 'Bunimovich')

print('70%') # <- Print programs progress

## Mushroom ##
# Generate the Mushroom board classes
mush_boards = [] #<- stores each of the Mushroom board classes
# Move the start to just above the x axis as the mushroom cap is centred on
# the origin
mush_start = START_POS
mush_start[1] = mush_start[1] + 0.1
for i in range(START_ANGLES.size):
    for j in range(mush_start[0].size):
        start_position = mush_start[:, j].reshape(2, 1)
        mush_para = (NUM_COLL, (1, 1, 1, 3), start_position, START_ANGLES[i])
        mush_board = BC.MushroomBilliardBoard(*mush_para)
        mush_boards.append(mush_board)
        # Class has been stored in list, so destroy the local copy
        del mush_board
saveArrayToCsv(mush_boards, 'Mushroom')

print('80%') # <- Print programs progress

## Lorentz Gas ##
# Generate the Lorentz gas board classes
lg_boards = [] #<- stores each of the Lorentz gas board classes
# Split the x positions between -2 and -1, and 1 and 2 since the circular 
# cutout through the centre of the board is a circle
lg_start_1 = lg_start_2 = np.zeros((2, 25))
lg_start_1[0] = np.linspace(-1.9, -1.1, 25)
lg_start_2[0] = np.linspace(1.1, 1.9, 25)
lg_start = np.concatenate((lg_start_1, lg_start_2), axis = 1)
# theta = 0 produces a divide by zero error in the tangent to ball's velocity 
# vector calculation
lg_angles = START_ANGLES[1:]
for i in range(lg_angles.size):
    for j in range(lg_start[0].size):
        start_position = lg_start[:, j].reshape(2, 1)
        lg_para = (NUM_COLL, None, start_position, lg_angles[i])
        lg_board = BC.LorentzGasBilliardBoard(*lg_para)
        lg_boards.append(lg_board)
        # Class has been stored in list, so destroy the local copy
        del lg_board
saveArrayToCsv(lg_boards, 'Lorentz Gas')

print('90%') # <- Print programs progress

## Hyperbolic Lorentz Gas ##
# Generate the hyperbolic Lorentz gas board classes
hlg_boards = [] #<- stores each of the hyperbolic Lorentz gas board classes
# Split the x positions between -2 and -1, and 1 and 2 since the cutout
# through the centre of the board is a hyperbolae
hlg_start_1 = hlg_start_2 = np.zeros((2, 25))
hlg_start_1[0] = np.linspace(-1.9, -1.1, 25)
hlg_start_2[0] = np.linspace(1.1, 1.9, 25)
hlg_start = np.concatenate((hlg_start_1, hlg_start_2), axis = 1)
# theta = 0 and theta = pi produces a divide by zero error in the tangent to
# ball's velocity vector calculation
hlg_angles = np.concatenate((START_ANGLES[1:5], START_ANGLES[6:]))
for i in range(hlg_angles.size):
    for j in range(hlg_start[0].size):
        start_position = hlg_start[:, j].reshape(2, 1)
        hlg_para = (NUM_COLL, None, start_position, hlg_angles[i])
        hlg_board = BC.HyperbolicLorentzGasBilliardBoard(*hlg_para)
        hlg_boards.append(hlg_board)
        # Class has been stored in list, so destroy the local copy
        del hlg_board
saveArrayToCsv(hlg_boards, 'Hyperbolic Lorentz Gas')

print('100%') # <- Print programs progress