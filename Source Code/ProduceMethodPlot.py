'''
    Author: Callum James Gill
    Email: cjg772@student.bham.ac.uk
    Creation date: 22/12/19
    Date submitted: 24/1/20
    Description: Produces a single plot for the board shapes. The plot is then
        used in the report as an illustration of the board shapes being
        investigated.
'''
import BilliardClasses as BC
from matplotlib import pyplot as plt

# Data creation
# Each board has no collision points to calculate
num_coll = 0
# Vertices of the square, going anti-clockwise around the sqaure
square_vertices = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
square = BC.PolygonBilliardBoard(num_coll, square_vertices)
triangle = BC.PolygonBilliardBoard(num_coll, 'triangle')
circle = BC.EllipticalBilliardBoard(num_coll) #<- default is circle
ellipse = BC.EllipticalBilliardBoard(num_coll, (2, 1))
hyperbolic = BC.HyperbolicBilliardBoard(num_coll) # Use default values
stadium = BC.StadiumBilliardBoard(num_coll) # Use default values
bunimovich = BC.BunimovichBilliardBoard(num_coll) # Use default values
mushroom = BC.MushroomBilliardBoard(num_coll) # Use default values
lorentz_gas = BC.LorentzGasBilliardBoard(num_coll)
hyperbolic_lorentz_gas = BC.HyperbolicLorentzGasBilliardBoard(num_coll)
# List containing pairs of board class and their titles to iterate over
board_list = [(square, 'Square'), (triangle, 'Triangle'), (circle, 'Circle'),
              (ellipse, 'Ellipse'), (hyperbolic, 'Hyperbolic'),
              (stadium, 'Stadium'), (bunimovich, 'Bunimovich'),
              (mushroom, 'Mushroom'), (lorentz_gas, 'Lorentz_gas'),
              (hyperbolic_lorentz_gas, 'Hyperbolic Lorentz gas')]

# Plot every board on 1 figure
fig, axes = plt.subplots(5, 2, sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0, 'wspace': 0})
i = 0 # For indexing over the columns of ax
j = 0 # For indexing over the rows of ax
for board, title in board_list:
    ax = axes[j, i]
    ax.set_title(title)
    ax.axis('off')
    board.drawBoard(ax)
    if i == 1:
        j += 1 # Next row
        i = 0 # Reset column index
    else:
        i += 1
fig.set_size_inches(10, 25)
plt.savefig('allBoards.jpeg', format='jpeg', quality=95, dpi=600)
plt.show()
