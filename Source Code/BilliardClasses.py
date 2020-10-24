'''
    Author: Callum James Gill
    Email: cjg772@student.bham.ac.uk
    Creation date: 29/11/19
    Date submitted: 24/1/20
    Description: Contains the billard ball and boards classes.
'''

import numpy as np
from random import uniform as rand
from random import randint

#### BILLIARD BALL CLASS ####

class BilliardBall():
    '''
        Class which defines the ball in the billiard system.
    '''

    ## CONSTRUCTOR ##

    def __init__(self, start, angle = None):
        '''
            Billard ball constructor.

            Note: should never be overridden

            Parameters:
                start : numpy array; 2 x 1 matrix
                    starting point vector defined as a 2x1 column matrix

                angle : None (default) or float (optional)
                    starting angle between 0 and 2 pi
            
            Returns:
                None
        '''
        if angle is not None:
            launch_ang = angle
        else:
            # random angle between -pi and pi
            launch_ang = rand(-np.pi, np.pi)
        # vector defining x, y position of start position
        self.start = start
        # vector defining the direction vector where ang = 0 is the positive x 
        # direction (right)
        self.direction = np.array([[np.cos(launch_ang)], [np.sin(launch_ang)]])
    
    ## PUBLIC METHODS ##

    def changeDirection(self, new_direction):
        '''
            Changes the direction vector of the billiard ball.

            Parameters:
                new_direction : numpy array
                    new direction vector defined as a column matrix of 
                    shape 2x1
            
            Returns:
                None
        '''
        self.direction = new_direction

    def changeStart(self, new_start):
        '''
            Changes the direction vector of the billiard ball.

            Parameters:
                new_start : numpy array
                    starting point vector defined as a column matrix of shape 
                    2x1

            Returns:
                None
        '''
        self.start = new_start

#### BILLIARD BOARD CLASSES ####

### SUPERCLASS (PARENT) ###

class BilliardBoard():
    '''
        Parent class which defines the 2D billiard board the ball moves within.
        All other billiard board classes ultimately inherit from this class.

        Note: this class should typically never be directly used in the API, 
        its simply to allow other billiard classes to inherit and override 
        methods and attributes.
    '''

    ## CONSTRUCTOR ##

    def __init__(self, num_iter, additional_parameters = None,
                 start_pos = None, start_angle = None):
        '''
            Billard domain constructor.

            Note: Shouldn't be overridden.

            Parameters:
                num_iter : int
                    number of iterations to end the analysis at
                
                additional_parameters : single item or tuple (optional)
                    the additional parameter to use in initalising or a tuple 
                    which contains multiple additional parameters.
                    Default is None.

                start_pos : None (Default) or tuple or numpy array
                    (x, y) to be converted into a vector
                
                start_angle : None (Default) or float
                    The starting angle between 0 and 2pi.
            
            Returns:
                None
        '''
        # Assigns any additional paramaters for child classes
        self._assignAdditionalAttributes(additional_parameters)
        # Generates the parameters needed to initalise a billiard ball object
        if start_pos is not None:
            if type(start_pos) is tuple:
                start = np.array(start_pos).reshape(2, 1)
            elif isinstance(start_pos, np.ndarray):
                start = start_pos
            else:
                message_1 = 'start_pos must be a tuple (x, y) '
                message_2 = 'or numpy array [[x], [y]]'
                message = message_1 + message_2
                raise TypeError(message)
        else:
            start = self._generateBallStartingPoint()
        # Instatitate the ball
        ball = BilliardBall(start, start_angle)
        # Initialises the list of angles used in plotting the phase space plots
        self.angles_to_normal = []
        # Array of velocity vectors
        self.velocities = None
        # The array of collision points. 1st row is the x values and 2nd the y 
        # values.
        # if statement used purely for cases where no collision points are to
        # be calculated
        if num_iter > 0:
            self.collision_points = self._generateCollisionPoints(ball, 
                                                                  num_iter)
        else:
            self.collision_points = None

    ## PUBLIC METHODS ##
    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Never used directly, hence the pass statement
        pass

    ## PRIVATE METHODS ##

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : single item or tuple
                    tuple which contains the other parameters used in 
                    constructor
            
            Returns:
                None
        '''
        # Not used by default
        # Will be overriden if additional parameters are assigned (which most
        # certainly will be the case).
        pass

    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Always overriden so the ball can be unbounded in its position
        # These numbers really don't matter
        x = rand(-999, 999)
        y = rand(-999, 999)
        return np.array([[x], [y]])
    
    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : numpy array shape (2, end)
                    numpy array with 2 rows, 1st row is x values and 2nd is y.
                    
                    Might not have the same number of columns as end because it
                    will be determined by the number of collision points, which
                    could be less if the calculation is terminated, e.g. at a 
                    corner of the board
        '''
        # Simply returns an empty array; method will always be overwritten
        collision_points = np.empty((2, end))
        return collision_points

    def _changeBallVectorLine(self, ball, direction_boundary, collision_point):
        '''
            Calculates the reflected ball vector line and changes the balls
            vector line.

            Do not override

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                
                direction_boundary: 2x1 numpy array
                    column matrix for the boundary direction vector
                
                collision_point: tuple
                    contains (x, y); coordinates of the collision point
            
            Returns:
                None
        '''
        # Add the velocity vector to the array of column vectors
        if self.velocities is None:
            self.velocities = ball.direction
        else:
            self.velocities = np.concatenate((self.velocities, ball.direction), 
                                             axis = 1)
        # Normal direction vector to the boundary
        normal_to_b = np.array([[direction_boundary[1][0]],
                                       [-1*direction_boundary[0][0]]])
        # Normalise the normal
        normal_to_b = normal_to_b / np.linalg.norm(normal_to_b)
        # Calculates the angle with the normal and boundary from the dot 
        # product with the reflected vector (both are normalised vectors)
        # Need positive values of each vector element for correct calculation
        normal_dot_ball = np.dot(normal_to_b.flatten(),
                                 ball.direction.flatten())
        normal_angle = np.arccos(normal_dot_ball)
        # Append the angles to the list of angles
        self.angles_to_normal.append(normal_angle)
        # dot product of balls current direction and normal vector
        # Reflected ball direction vector
        reflect_ball_dir = ball.direction - 2*normal_dot_ball*normal_to_b
        # Normalise the reflected ball direction (if not so all ready)
        reflect_ball_dir = reflect_ball_dir / np.linalg.norm(reflect_ball_dir)
        ball.changeDirection(reflect_ball_dir)
        ball.changeStart(collision_point)

### SUBCLASSES (CHILD) ###

## POLYGON ##

class PolygonBilliardBoard(BilliardBoard):
    '''
        Class which defines the 2D polygon billiard board the ball moves 
        within.

        Child class of BilliardBoard.
    '''

    ## PUBLIC METHODS ##

    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Loop over all vertices
        legend_labels = [legend_label]
        # Used to prevent multiple labels in legend
        legend_labels.extend([None] * len(self.vertices))
        for i in range(len(self.vertices)):
            point_1 = self.vertices[i]
            # Deals with the issue of the final side
            if i+1 == len(self.vertices):
                point_2 = self.vertices[0]
            else:
                point_2 = self.vertices[i+1]
            xs = [point_1[0][0], point_2[0][0]]
            ys = [point_1[1][0], point_2[1][0]]
            ax.plot(xs, ys, color='k', linewidth=1, label=legend_labels[i])

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly selects a point along the intercepts of the corners of the 
        # polygon with the origin
        # Randomly chooses the direction vector scalar between 0.1 and 0.9
        # can't be 0 or 1
        lambda_vertex = rand(0.1, 0.9)
        vertex_index = randint(0, len(self.vertices)-1)
        start = lambda_vertex * self.vertices[vertex_index]
        return start

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : None (Default) or single item (optional)
                    if no arguments given then default values of the below will 
                    be used, 

                        vertices : list of tuples  (optional)
                            Contains a list of tuples where each tuple contains 
                            the coordinate of a vertex of the board. The order 
                            should be such that the first to last vertices go 
                            anti-clockwise around the whole polygon.

                        shape : or string (optional)
                            String which indicates which regular polygon to use 
                            for the board. The possible options are:
                                triangle, square, pentagon, hexagon, 
                                heptagon, octagon, nonagon, decagon.
                            Default is 'triangle'. This default value is only 
                            used if the vertices parameter isn't given.
            
            Returns:
                None
        '''
        # Assigns the vertices in the order in anti-clockwise direction
        self.vertices = []
        vertices = None # Default
        if additional_parameters is None:
            shape = 'triangle'
        elif type(additional_parameters) is list:
            # If list is given then its the vertices
            vertices = additional_parameters
        elif type(additional_parameters) is str:
            # If string is given then it's the regular ploygon to plot
            shape = additional_parameters
        # Assigns the largest and smallest x, y for use in the limits method
        # for plotting the trajectory plot
        self._largest_x=self._smallest_x=self._largest_y=self._smallest_y=0
        # Checks the vertices for the largest and smallest x, y values
        for vertex in self.vertices:
            x = vertex[0][0]
            y = vertex[1][0]
            if x < self._smallest_x:
                self._smallest_x = x
            elif x > self._largest_x:
                self._largest_x = x
            if y < self._smallest_y:
                self._smallest_y = y
            elif y > self._largest_y:
                self._largest_y = y
        # Generates the vertices if none given, otherwise convert the vertex 
        # tuple pairs from list into a numpy array
        if vertices is None:
            shapes = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 
                      'heptagon': 7, 'octagon': 8, 'nonagon': 9, 'decagon': 10}
            if shape in shapes:
                n = shapes[shape] # Number of sides
                s = n # Side length of the polygon
                next_vertices = [] # Used to store the next vertices
                theta = np.pi * (n-2) / (2*n) # 1/2 interior angle (radians)
                R = s / (2 * np.sin(np.pi / n)) # Circumradius
                #Angle subtended by the lines from the origin to two vertices
                argument = s * np.sin(theta) / R
                phi = np.arcsin(argument)
                if n == 3:
                    # Converts the triangle angle from 60 to 120 degrees
                    phi = np.pi - phi
                new_phi = phi
                first_vertex = np.array([[0], [R]]) # First vertex always here
                # Rounding is done as the more significant figures there are,
                # then errors in calculating tangents and normals occur.
                self.vertices.append(np.round(first_vertex, 2))
                # Angle to calculate the direction vector, 
                # measured from positive x axis
                i = 1
                while i < n:
                    angle = np.pi / 2 - new_phi
                    cosine = np.cos(angle)
                    sine = np.sin(angle)
                    direction_vector = np.array([[cosine], [sine]])
                    # Vertex always a circumradius away from centre of polygon
                    vertex = R * direction_vector
                    # Round the answer before appending
                    next_vertices.append(np.round(vertex, 2))
                    new_phi += phi
                    i += 1
                next_vertices.reverse()
                self.vertices.extend(next_vertices)
            else:
                keys = list(shapes.keys())
                s = " "
                keys_string = s.join(keys)
                message = "Shape must be one of the following: " + keys_string
                raise ValueError(message)
        else:
            for vertex in vertices:
                vertex_vec = np.array([[vertex[0]], [vertex[1]]])
                self.vertices.append(vertex_vec)

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            # Correct intercept returned
            line_tup = self._calculateLineIntercept(ball.start, ball.direction)
            collision_point, boundary_direction, end_inter = line_tup
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.append(collision_points, collision_point,
                                             axis = 1)
            if end_inter:
                print("Corner hit, stopping...")
                break
            # Change the vector line of the ball
            self._changeBallVectorLine(ball, boundary_direction,
                                       collision_point)
            i += 1
        return collision_points

    # NEW METHODS #

    def _calculateLineIntercept(self, point, direction, no_end = False):
        '''
            Calculates where along each of the boundaries the billiard collides
            via a parameter t which should be between 0 and 1. If it's not
            in this range then it doesn't intercept that particular boundary.

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction : 2x1 numpy array
                    column matrix for the balls direction vector

                no_end : bool (optional)
                    if true then don't return the direction scalar, otherwise 
                    (default) return it.
            
            Returns:
                (collision_point, boundary_direction, end) : tuple of 2 numpy 
                    arrays and a boolean
                    
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
                    end : bool
                        True if the collision calculations should end, 
                        False if not.
        '''
        # Calculates the intercepts
        # It checks the computed values of t for each boundary and deduces
        # which boundary is the correct one to use. It will sometimes calculate 
        # the previous collision point as an intercept, so it will ignore this 
        # and continue
        end = False # Default value for the end parameter
        coll_point=bound_dir=correct_coll=correct_bound_dir=np.empty((2, 1))
        correct_lambda = None
        for i in range(len(self.vertices)):
            point_1 = self.vertices[i]
            if i+1 == len(self.vertices):
                point_2 = self.vertices[0]
            else:
                point_2 = self.vertices[i+1]
            # Calculates the boundary direction vector
            bound_dir = point_2 - point_1
            # Choose correct parameters to calculate lambda and t with
            if bound_dir[0][0] == 0:
                ratio = bound_dir[0][0] / bound_dir[1][0]
                A = direction[0][0]
                B = direction[1][0]
                C = point_1[0][0]
                D = point[0][0]
                E = point[1][0]
                F = point_1[1][0]
                G = 1 / bound_dir[1][0]
            else:
                ratio = bound_dir[1][0] / bound_dir[0][0]
                A = direction[1][0]
                B = direction[0][0]
                C = point_1[1][0]
                D = point[1][0]
                E = point[0][0]
                F = point_1[0][0]
                G = 1 / bound_dir[0][0]
            # Calculates the balls direction scalar
            lambda_ball = (C - D + ratio * (E - F)) / (A - ratio * B)
            if lambda_ball < 0:
                continue
            # Calculates the boundaries directions scalar
            t = G * (E + lambda_ball * B - F)
            coll_point = point_1 + t * bound_dir
            # Makes sure the correct ball direction has been calculated, i.e.
                # the balls vector direction scalar is positive and t is
                # between 0 and 1
            if t >= 0 and t <= 1:
                # Checks whether the balls current position has been calculated
                # Rounding is used to avoid slight differences in precision
                if np.array_equal(np.round(point, 6), np.round(coll_point, 6)):
                    continue
                # This is needed for weirder shapes, such as the mushroom 
                # billiard board
                if correct_lambda is not None:
                    if lambda_ball > correct_lambda:
                        continue
                correct_coll = coll_point
                correct_bound_dir = bound_dir
                correct_lambda = lambda_ball
                if (t <= 0.0001 and t >= 0)  or (t <= 1 and t >= 0.9999):
                    # Here the ball should stop, effectively so close to the 
                    # corner it's at the corner
                    end = True
        if no_end:
            tup = (correct_coll, correct_bound_dir)
        else:
            tup = (correct_coll, correct_bound_dir, end)
        return tup

## ELlIPSE ##

class EllipticalBilliardBoard(BilliardBoard):
    '''
        Class which defines the 2D elliptical billiard board the ball moves
        within.

        Child class of BilliardBoard.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Calculate all angles and trig funcs
        angles = np.linspace(0, 2*np.pi, 1000)
        cosine = np.cos(angles)
        sine = np. sin(angles)
        # Calculate the radius of the ellipse at theta
        rs = self.a*self.b/np.sqrt((self.b*cosine)**2+(self.a*sine)**2)
        # x and y values
        xs = self.centre[0][0] + rs * cosine
        ys = self.centre[1][0] + rs * sine
        # Draw the board
        ax.plot(xs, ys, color = 'k', linewidth = 1, label = legend_label)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        tol = 0.2
        # For the choice in y given x since there are two y's for every x
        sign = randint(0, 1)
        x_max = self.a + self.centre[0][0] - tol
        x_min = self.centre[0][0] - self.a + tol
        x = rand(x_min, x_max)
        y = self.centre[1][0] + sign * self.b * (x - self.centre[0][0])/self.a
        return np.array([[x], [y]])

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : tuple or list
                    list or tuple which contains the other parameters used in 
                    constructor. These are either: 
                    
                    [(width_x, width_y), (centre_x, centre_y)] 
                    
                    or
                    
                    (width_x, width_y)

                    width_x and width_y are both int or floats
                    centre_x and centre_y are both int or floats
                    (the centre parameters are more useful for piecewise use,
                    so they shouldn't be passed when using this object 
                    directly)
            
            Returns:
                None
        '''
        # Checks if the additional_parameters tuple has a first element thats 
        # a tuple or not
        # In this case then only the semi-major and semi-minor axes have been 
        # given.
        # Assigns the centre of the circle or ellipse
        self.centre = np.array([[0], [0]]) # <- Default centre
        if additional_parameters is None:
            # When no additional parameters are given, use default values 
            #  (that of a circle centred on origin)
            self.a = self.b = 1
        elif type(additional_parameters) is list:
            # Assigns the radius and foci of the circle/ellipse
            self.a = additional_parameters[0][0]
            self.b = additional_parameters[0][1]
            # Assigns the centre of the circle or ellipse
            centre_x, centre_y = additional_parameters[1]
            self.centre = np.array([[centre_x], [centre_y]])
        else:
            # Assigns the radius and foci of the circle/ellipse
            self.a = additional_parameters[0]
            self.b = additional_parameters[1]
        # Raises a ValueError if the width and height of the circle/ellipse
        # aren't greater than 0.
        if self.a <= 0:
            message = str(self.a) + " must be greater than 0!"
            raise ValueError(message)
        elif self.b <= 0:
            message = str(self.b) + " must be greater than 0!"
            raise ValueError(message)

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            # Get the collision point and direction of the tangent to the 
            # ellipse at that point
            intercept_tup = self._calculateEllipseIntercept(ball.start,
                                                            ball.direction)
            collision_point, boundary_direction = intercept_tup
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball,boundary_direction,collision_point)
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            i += 1
        return collision_points

    def _calculateEllipseIntercept(self, point, direction,
                                   closest_inter=False, return_lamba=False):
        '''
            Calculates where along the ellipse the ball's vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction : 2x1 numpy array
                    column matrix for the balls direction vector

                closest_inter : Bool (optional)
                    If False (default) then return the intercept furtherest 
                    away, otherwise return the closer one
                
                return_lambda : Bool (optional)
                    If False (default) then don't return the direction scalar 
                    lambda, otherwise it should be returned.
            
            Returns:
                (collision_point, boundary_direction) : tuple of 2 numpy arrays 
                and a boolean
                
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                        
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
        '''
        # Calculates the intercepts
        lambda_ball = None #<- initalised here
        # It checks the computed values of lambda for each boundary point and 
        # deduces which boundary is the correct one to use.
        # Variables dx, dy, x0, y0 needed to calculate lambda_ball (the 
        # direction scalar of the balls vector line)
        # Theta needed to calculate the tangent to boundary and the radius
        dx = direction[0][0]
        dy = direction[1][0]
        x0 = point[0][0]
        y0 = point[1][0]
        xc = self.centre[0][0]
        yc = self.centre[1][0]
        # A, B, C are the coefficients of the quadratic equation of lambda_ball
        A = (self.b * dx) ** 2 + (self.a * dy) ** 2
        B = 2 * (self.b ** 2 * dx * (x0 - xc)  + self.a ** 2 * dy * (y0 - yc))
        c_1 = self.b ** 2 * (x0 ** 2 + xc ** 2 - 2 * x0 * xc)
        c_2 = self.a ** 2 * (y0 ** 2 + yc ** 2 - 2 * y0 * yc)
        c_3 = (self.a * self.b) ** 2
        C = c_1 + c_2 - c_3
        # Calculates lambda_ball
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            # This is for when lines don't intercept the ellipse
            collision_point = boundary_direction = None
        else:
            lambda_ball1 = (-1*B + np.sqrt(discriminant)) / (2*A)
            lambda_ball2 = (-1*B - np.sqrt(discriminant)) / (2*A)
            # If one of the lambda_ball's is less than or equal to 0 then that 
            # corresponds to the wrong direction or the current collision point
            if lambda_ball1 > lambda_ball2:
                if closest_inter:
                    lambda_ball = lambda_ball2
                else:
                    lambda_ball = lambda_ball1
            else:
                if closest_inter:
                    lambda_ball = lambda_ball1
                else:
                    lambda_ball = lambda_ball2
            collision_point = point + lambda_ball * direction
            tangent_direct_x = -1*self.a**2 / (collision_point[0][0]-xc)
            tangent_direct_y = self.b**2 / (collision_point[1][0]-yc)
            boundary_direction = np.array([[tangent_direct_x],
                                           [tangent_direct_y]])
            if closest_inter:
                boundary_direction *= -1
        if return_lamba:
            tup = (collision_point, boundary_direction, lambda_ball)
        else:
            tup = (collision_point, boundary_direction)
        return tup

## HYPERBOLIC ##
class HyperbolicBilliardBoard(BilliardBoard):
    '''
        Class which defines the 2D convex hyperbolic billiard board the ball 
        moves within.

        Child class of BilliardBoard.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # x and y values for top and bottom
        xs_tb = np.linspace(self.min_x, self.max_x)
        ys_t = self.c * np.sqrt(1 + (xs_tb / self.d) ** 2)
        ys_b = -1 * ys_t
        # x and y for left and right
        ys_lr = np.linspace(self.min_y, self.max_y)
        xs_r = self.a * np.sqrt(1 + (ys_lr / self.b) ** 2)
        xs_l = -1 * xs_r
        # Join arrays together
        xs = np.concatenate((xs_tb, xs_r[::-1], xs_tb[::-1], xs_l))
        ys = np.concatenate((ys_t, ys_lr[::-1], ys_b[::-1], ys_lr))
        # Draw the board
        ax.plot(xs, ys, color = 'k', linewidth = 1, label = legend_label)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        tol = 0.05
        # All lists are in the following order: top, bottom, left, right
        possible_starts = [] # List to append to and randomly choose from
        # All possible max and min angles
        min_max_list = [(np.arctan2(self.max_y, self.min_x),
                         np.arctan2(self.max_y, self.max_x)),
                    (np.arctan2(self.min_y, self.max_x),
                     np.arctan2(self.min_y, self.min_x)),
                    (2*np.pi + np.arctan2(self.min_y, self.min_x),
                     np.arctan2(self.max_y, self.min_x)),
                    (np.arctan2(self.max_y, self.max_x),
                     np.arctan2(self.min_y, self.max_x))]
        for i in range(len(min_max_list)):
            max_theta, min_theta = min_max_list[i]
            angle = rand(min_theta, max_theta)
            sine = np.sin(angle)
            cosine = np.cos(angle)
            if i < 2:
                r = self.c*self.d/np.sqrt((self.d*sine)**2-(self.c*cosine)**2)
            else:
                r = self.a*self.b/np.sqrt((self.b*cosine)**2-(self.a*sine)**2)
            rand_r = rand(0, r-tol)
            xy = rand_r * np.array([[cosine], [sine]])
            possible_starts.append(xy)
        rand_index = randint(0, 3)
        return possible_starts[rand_index]

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : tuple or list
                    list or tuple which contains the other parameters used in 
                    the constructor.
                    
                    These are either: (width_x, width_y)

                    width_x and width_y are both int or floats
            
            Returns:
                None
        '''
        # Checks if the additional_parameters tuple has a first element thats 
        # a tuple or not
        # In this case then only the semi-major and semi-minor axes have been 
        # given
        if additional_parameters is None:
            # Assigns the radius and foci of the hyperbola
            self.a = self.b = 1
        else:
            # Assigns the radius and foci of the hyperbola
            self.a, self.b = additional_parameters
        # c and d are the top and bottom hyperbolae widths and heights
        self.c = 0.5 * self.b
        self.d = 0.75 * self.a
        # Raises a ValueError if the width and height of the hyperbola
        # aren't greater than 0.
        if self.a <= 0:
            message = str(self.a) + "must be greater than 0!"
            raise ValueError(message)
        elif self.b <= 0:
            message = str(self.b) + "must be greater than 0!"
            raise ValueError(message)
        # max and min y's and x's
        denominator = ((self.d * self.b) ** 2 - (self.a * self.c) ** 2)
        numerator = (self.b * self.c) ** 2 * (self.d ** 2 + self.a ** 2)
        self.max_y = np.sqrt(numerator / denominator)
        self.min_y = -1 * self.max_y
        self.max_x = np.sqrt(self.a ** 2 * (1 + (self.max_y / self.b ) ** 2))
        self.min_x = -1 * self.max_x

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            # Get the collision point and direction of the tangent to the 
            # hyperbola at that point
            intercept_tup = self._calculateHyperbolicIntercept(ball.start,
                                                               ball.direction)
            collision_point, boundary_direction = intercept_tup
            if boundary_direction is None:
                # Corner has been hit so end loop.
                break
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball, boundary_direction,
                                       collision_point)
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            i += 1
        return collision_points

    # NEW METHODS #

    def _calculateHyperbolicIntercept(self, point, direction,
                                      closest_inter=False, return_lamba=False):
        '''
            Calculates where along the convex hyperbolic surface the ball's 
            vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line

                direction: 2x1 numpy array
                    column matrix for the balls direction vector
                
                closest_inter : Bool (optional)
                    If False (default) then return the intercept furtherest 
                    away, otherwise return the closer one
                
                return_lambda : Bool (optional)
                    If False (default) then don't return the direction scalar 
                    lambda, otherwise it should be returned.
            
            Returns:
                Default:
                    (collision_point, boundary_direction) : tuple of 2 numpy 
                    arrays and a boolean

                        collision point : column vector numpy array (2x1)
                            position vector for the collision point

                        boundary_direction : column vector numpy array (2x1)
                            direciton vector for the boundary
                
                If return_lambda = True:
                    (collision_point, boundary_direction, lambda_ball) : tuple 
                    of 2 numpy arrays and a boolean

                        collision point : column vector numpy array (2x1)
                            position vector for the collision point

                        boundary_direction : column vector numpy array (2x1)
                            direciton vector for the boundary

                        lambda_ball : float
                            scalar multiple to the ball's direction vector to 
                            take it from the starting point to the collision 
                            point.                        
        '''
        # Initalise the variables
        intercept = (None, None)
        # Calculate both possible intercepts
        l_r_intercept = self._leftRightIntercept(point,direction,closest_inter)
        t_b_intercept = self._topBottomIntercept(point,direction,closest_inter)
        if t_b_intercept[0] is None and l_r_intercept[0] is not None:
            intercept = l_r_intercept
        elif t_b_intercept[0] is not None and l_r_intercept[0] is None:
            intercept = t_b_intercept
        elif t_b_intercept[0] is not None and l_r_intercept[0] is not None:
            # Choose the smallest lambda
            if t_b_intercept[2] < l_r_intercept[2]:
                if t_b_intercept[2] < 0:
                    # For when the collision points are on the outside of the
                    # hyperbolae
                    intercept = l_r_intercept
                else:
                    intercept = t_b_intercept
            else:
                if l_r_intercept[2] < 0:
                    # For when the collision points are on the outside of the
                    # hyperbolae
                    intercept = t_b_intercept
                else:
                    intercept = l_r_intercept
        if not return_lamba and intercept[0] is not None:
            intercept = intercept[:2]
        return intercept

    def _topBottomIntercept(self, point, direction, closest_inter):
        '''
            Calculates where along the top or/and bottom hyperbolae the ball's
            vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction : 2x1 numpy array
                    column matrix for the balls direction vector

                closest_inter : Bool
                    If False then return the intercept furtherest away,
                    otherwise return the closer one
            
            Returns:
                (collision_point, boundary_direction, lambda_ball) : tuple of 
                2 numpy arrays or None
                
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point

                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary

                    lambda_ball : float
                        direction scalar for the direction vector
        '''
        # Initalise the collision point and boundary direction vectors as None
        collision_point = boundary_direction = lambda_ball = None
        # It checks the computed values of lambda for each intercept on either
        # t_b_intercept and deduces which boundary is the correct one to use.
        # Variables dx, dy, x0, y0 needed to calculate lambda_ball (the 
        # direction scalar of the balls vector line)
        # Theta needed to calculate the tangent to boundary and the radius
        dx = direction[0][0]
        dy = direction[1][0]
        x0 = point[0][0]
        y0 = point[1][0]
        # A, B, C are the coefficients of the quadratic equation of lambda_ball
        # for top and bottom
        A = (self.d * dy) ** 2 - (self.c * dx) ** 2
        B = 2 * (self.d ** 2 * dy * y0 - self.c ** 2 * dx * x0)
        C = (self.d * y0) ** 2 - (self.c * x0) ** 2 - (self.c * self.d) ** 2
        # Calculates lambda_ball for top and bottom
        discriminant = B ** 2 - 4 * A * C
        if discriminant >= 0:
            # The two possible lambda's
            lambda_ball1 = (-1*B + np.sqrt(discriminant)) / (2*A)
            lambda_ball2 = (-1*B - np.sqrt(discriminant)) / (2*A)
            # Calculate the possible intercepts
            intercept_1 = point + lambda_ball1 * direction
            intercept_2 = point + lambda_ball2 * direction
            # Check if each collision point is valid
            if intercept_1[0,0] < self.min_x or intercept_1[0,0] > self.max_x:
                # Outside boundaries
                intercept_1 = None
            if intercept_2[0,0] < self.min_x or intercept_2[0,0] > self.max_x:
                # Outside boundaries
                intercept_2 = None
            # Check which one is correct if at least one of the intercepts 
            # exists
            if intercept_1 is not None and intercept_2 is not None:
                # If both the intercepts have y's greater than zero then its 
                # on the top intercept
                if intercept_1[1, 0] > 0 and intercept_2[1, 0] > 0:
                    # when the collision points are outisde the hyperbolae
                    if closest_inter:
                        if np.array_equal(np.round(point, 6),
                                          np.round(intercept_1, 6)):
                            collision_point = intercept_2
                            lambda_ball = lambda_ball2
                        elif np.array_equal(np.round(point, 6),
                                            np.round(intercept_2, 6)):
                            collision_point = intercept_1
                            lambda_ball = lambda_ball1
                    else:
                        # If neither is the start point, choose the one with 
                        # the smaller lambda
                        if not np.array_equal(np.round(point, 6),
                                              np.round(intercept_1, 6)):
                            if not np.array_equal(np.round(point, 6),
                                                  np.round(intercept_2, 6)):
                                if lambda_ball1 < lambda_ball2:
                                    collision_point = intercept_1
                                    lambda_ball = lambda_ball1
                                else:
                                    collision_point = intercept_2
                                    lambda_ball = lambda_ball2
                elif intercept_1[1, 0] < 0 and intercept_2[1, 0] < 0:
                    # If both the intercepts have y's less than zero then its
                    # on the bottom intercept when the collision points are 
                    # outisde the hyperbolae
                    if closest_inter:
                        if np.array_equal(np.round(point, 6),
                                          np.round(intercept_1, 6)):
                            collision_point = intercept_2
                            lambda_ball = lambda_ball2
                        elif np.array_equal(np.round(point, 6),
                                            np.round(intercept_2, 6)):
                            collision_point = intercept_1
                            lambda_ball = lambda_ball1
                    else:
                        # If neither is the start point, choose the one with 
                        # the smaller lambda
                        if not np.array_equal(np.round(point, 6),
                                              np.round(intercept_1, 6)):
                            if not np.array_equal(np.round(point, 6),
                                                  np.round(intercept_2, 6)):
                                if lambda_ball1 < lambda_ball2:
                                    collision_point = intercept_1
                                    lambda_ball = lambda_ball1
                                else:
                                    collision_point = intercept_2
                                    lambda_ball = lambda_ball2
                else:
                    if lambda_ball1 > lambda_ball2 and not closest_inter:
                        collision_point = intercept_1
                        lambda_ball = lambda_ball1
                    else:
                        collision_point = intercept_2
                        lambda_ball = lambda_ball2
            elif intercept_1 is not None:
                collision_point = intercept_1
                lambda_ball = lambda_ball1
            elif intercept_2 is not None:
                collision_point = intercept_2
                lambda_ball = lambda_ball2
            # Now check to see if the collision point isn't none in order to 
            # calculate the tangent
            if collision_point is not None:
                if np.array_equal(np.round(point, 6),
                                  np.round(collision_point, 6)):
                    collision_point = None
                    lambda_ball = None
                else:
                    tangent_direct_x = self.d ** 2 / collision_point[0][0]
                    tangent_direct_y = self.c ** 2 / collision_point[1][0]
                    boundary_direction = np.array([[tangent_direct_x],
                                                   [tangent_direct_y]])
        return (collision_point, boundary_direction, lambda_ball)

    def _leftRightIntercept(self, point, direction, closest_inter):
        '''
            Calculates where along the left or/and right hyperbolae the ball's 
            vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction : 2x1 numpy array
                    column matrix for the balls direction vector

                closest_inter : Bool
                    If False then return the intercept furtherest away,
                    otherwise return the closer one
            
            Returns:
                (collision_point, boundary_direction, lambda_ball) : tuple of 
                2 numpy arrays or None
                
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
                    lambda_ball : float
                        direction scalar for the direction vector
        '''
        # Initalise the collision point and boundary direction vectors as None
        collision_point = boundary_direction = lambda_ball = None
        # It checks the computed values of lambda for each intercept on either
        # the top or bottom or both and deduces which boundary is the correct 
        # one to use.
        # Variables dx, dy, x0, y0 needed to calculate lambda_ball 
        # (the direction scalar of the balls vector line)
        # Theta needed to calculate the tangent to boundary and the radius
        dx = direction[0][0]
        dy = direction[1][0]
        x0 = point[0][0]
        y0 = point[1][0]
        # A, B, C are the coefficients of the quadratic equation of lambda_ball
        # for left and right
        A = (self.b * dx) ** 2 - (self.a * dy) ** 2
        B = 2 * (self.b ** 2 * dx * x0 - self.a ** 2 * dy * y0)
        C = (self.b * x0) ** 2 - (self.a * y0) ** 2 - (self.a * self.b) ** 2
        # Calculates lambda_ball for top and bottom
        discriminant = B ** 2 - 4 * A * C
        if discriminant >= 0:
            # The two possible lambda's
            lambda_ball1 = (-1*B + np.sqrt(discriminant)) / (2*A)
            lambda_ball2 = (-1*B - np.sqrt(discriminant)) / (2*A)
            # Calculate the possible intercepts
            intercept_1 = point + lambda_ball1 * direction
            intercept_2 = point + lambda_ball2 * direction
            # Check if each collision point is valid
            if intercept_1[1,0] < self.min_y or intercept_1[1,0] > self.max_y:
                # Outside boundaries
                intercept_1 = None
            if intercept_2[1,0] < self.min_y or intercept_2[1,0] > self.max_y:
                # Outside boundaries
                intercept_2 = None
            # Check which one is correct if at least one of the intercepts 
            # exists
            if intercept_1 is not None and intercept_2 is not None:
                # If both the intercepts have x's greater than zero then its 
                # on the right intercept
                if intercept_1[0, 0] > 0 and intercept_2[0, 0] > 0:
                    # when the collision points are outisde the hyperbolae
                    if closest_inter:
                        if np.array_equal(np.round(point, 6),
                                          np.round(intercept_1, 6)):
                            collision_point = intercept_2
                            lambda_ball = lambda_ball2
                        elif np.array_equal(np.round(point, 6),
                                            np.round(intercept_2, 6)):
                            collision_point = intercept_1
                            lambda_ball = lambda_ball1
                    else:
                        # If neither is the start point, choose the one with
                        # the smaller lambda
                        if not np.array_equal(np.round(point, 6),
                                              np.round(intercept_1, 6)):
                            if not np.array_equal(np.round(point, 6),
                                                  np.round(intercept_2, 6)):
                                if lambda_ball1 < lambda_ball2:
                                    collision_point = intercept_1
                                    lambda_ball = lambda_ball1
                                else:
                                    collision_point = intercept_2
                                    lambda_ball = lambda_ball2
                elif intercept_1[0, 0] < 0 and intercept_2[0, 0] < 0:
                    # If both the intercepts have x's less than zero then 
                    # its on the left intercept when the collision points are 
                    # outisde the hyperbolae
                    if closest_inter:
                        if np.array_equal(np.round(point, 6),
                                          np.round(intercept_1, 6)):
                            collision_point = intercept_2
                            lambda_ball = lambda_ball2
                        elif np.array_equal(np.round(point, 6),
                                            np.round(intercept_2, 6)):
                            collision_point = intercept_1
                            lambda_ball = lambda_ball1
                    else:
                        # If neither is the start point, choose the one with 
                        # the smaller lambda
                        if not np.array_equal(np.round(point, 6),
                                              np.round(intercept_1, 6)):
                            if not np.array_equal(np.round(point, 6),
                                                  np.round(intercept_2, 6)):
                                if lambda_ball1 < lambda_ball2:
                                    collision_point = intercept_1
                                    lambda_ball = lambda_ball1
                                else:
                                    collision_point = intercept_2
                                    lambda_ball = lambda_ball2
                else:
                    if lambda_ball1 > lambda_ball2 and not closest_inter:
                        collision_point = intercept_1
                        lambda_ball = lambda_ball1
                    else:
                        collision_point = intercept_2
                        lambda_ball = lambda_ball2
            elif intercept_1 is not None:
                collision_point = intercept_1
                lambda_ball = lambda_ball1
            elif intercept_2 is not None:
                collision_point = intercept_2
                lambda_ball = lambda_ball2
            # Now check to see if the collision point isn't none in order to 
            # calculate the tangent and a final check on whether its the start 
            # point
            if collision_point is not None:
                if np.array_equal(np.round(point, 6),
                                  np.round(collision_point, 6)):
                    collision_point = None
                    lambda_ball = None
                else:
                    tangent_direct_x = self.a**2 / collision_point[0][0]
                    tangent_direct_y = self.b**2 / collision_point[1][0]
                    boundary_direction = np.array([[tangent_direct_x],
                                                   [tangent_direct_y]])
        return (collision_point, boundary_direction, lambda_ball)

### SUBSUBCLASSES (GRANDCHILDREN) ###

## BUNIMOVICH BILLIARDS ##

# The stadium is technically BunimovichBilliardBoardCase1
class StadiumBilliardBoard(PolygonBilliardBoard, EllipticalBilliardBoard):
    '''
        Class that defines the 2D stadium billiard board the ball moves within.
        This is the first case of a Bunimovich billiard board being analysed.

        Child class of both PolygonBilliardBoard and EllipticalBilliardBoard.
        It inherits the collision point calculator methods from both
        of these classes. This is because its a board of a rectangular section 
        with elliptical caps.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Calculate all angles and trig funcs
        angles = np.linspace(-1*np.pi / 2, np.pi / 2, 1000)
        cosine = np.cos(angles)
        sine = np. sin(angles)
        # Calculate the radius of the ellipse at theta
        rs = self.a*self.b/np.sqrt((self.b*cosine)**2+(self.a*sine)**2)
        # x and y values
        x_cap_one = self.cap_left_centre[0][0] - rs * cosine
        y_cap_one = self.cap_left_centre[1][0] + rs * sine
        x_cap_two = self.cap_right_centre[0][0] + rs * cosine
        y_cap_two = self.cap_right_centre[1][0] + rs * sine
        x_rect = np.linspace(-1*self.width/2, self.width/2, 1000)
        y_rect_top = np.ones(x_rect.shape) * self.height / 2
        y_rect_bottom = -1 * y_rect_top
        # Join the arrays together
        xs = np.concatenate((x_cap_one, x_rect, x_cap_two[::-1], x_rect))
        ys = np.concatenate((y_cap_one, y_rect_top, y_cap_two[::-1],
                             y_rect_bottom))
        # Draw the board
        ax.plot(xs, ys, color = 'k', linewidth = 1, label = legend_label)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly chooses a point within the caps or the rectangular section
        possible_starts = []
        cap_centres = [self.cap_left_centre, self.cap_right_centre]
        for i in range(3):
            if i < len(cap_centres):
                # For circular caps
                tol = 0.2
                # Sign is for the choice in y given x since there are two y's 
                # for every x
                sign = randint(0, 1)
                if i == 0:
                    x_max = 0
                    x_min = cap_centres[i][0][0] - self.cap_width + tol
                else:
                    x_max = self.cap_width + cap_centres[i][0][0] - tol
                    x_min = 0
                x_cap = rand(x_min, x_max)
                y_c1 = (x_cap-cap_centres[i][0][0])/self.cap_width
                y_cap = cap_centres[i][1][0] + sign * self.height * y_c1
                possible_starts.append(np.array([[x_cap], [y_cap]]))
            else:
                # For rectangular section
                x_rect = rand(-1*self.width / 2, self.width / 2)
                y_rect = rand(-1*self.height / 2, self.height / 2)
                possible_starts.append(np.array([[x_rect], [y_rect]]))
        rand_index = randint(0, len(possible_starts)-1)
        return possible_starts[rand_index]

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : tuple
                    (width, height, cap_width)

                width, height and cap_width are all int or floats.
                The width and height determine the rectangular section size and
                cap_width determines the circular/elliptical cap width
            
            Returns:
                None
        '''
        if additional_parameters is None:
            # Default values
            self.width = self.height = 1
            self.cap_width = 0.5
        else:
            self.width, self.height, self.cap_width = additional_parameters
            # Raises a ValueError if the width height and cap width aren't
            # greater than 0.
            if self.width <= 0:
                message = str(self.width) + "must be greater than 0!"
                raise ValueError(message)
            elif self.height <= 0:
                message = str(self.height) + "must be greater than 0!"
                raise ValueError(message)
            elif self.cap_width <= 0:
                message = str(self.cap_width) + "must be greater than 0!"
                raise ValueError(message)
        self.cap_left_centre = np.array([[-1*self.width / 2], [0]])
        self.cap_right_centre = np.array([[self.width / 2], [0]])
        self.a = self.cap_width # Needed for the circular intercepts
        self.b = self.height / 2 # Needed for the circular intercepts
        top_right = np.array([[self.width/2], [self.height/2]])
        top_left = np.array([[-1*self.width/2], [self.height/2]])
        bottom_right = np.array([[self.width/2], [-1*self.height/2]])
        bottom_left = np.array([[-1*self.width/2], [-1*self.height/2]])
        self.vertices = [top_right, top_left, bottom_left, bottom_right]

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            start = ball.start
            direction = ball.direction
            left_cap_intercept = self._leftCapIntercept(start, direction)
            right_cap_intercept = self._rightCapIntercept(start, direction)
            rectangle_intercept = self._rectangleIntercept(start, direction)
            # Tuples used to iterate over efficiently with a for loop
            # Last element string is used to determine which conditional check 
            # to use
            left_tup = (left_cap_intercept, self.cap_left_centre[0,0], 'left')
            right_tup = (right_cap_intercept, self.cap_right_centre[0, 0],
                         'right')
            rect_tup = (rectangle_intercept,
                        (self.cap_left_centre[0, 0],
                         self.cap_right_centre[0, 0]),
                         'rect')
            tup_list = [left_tup, right_tup, rect_tup]
            # Declare the variables to be written in the loop
            collision_point = boundary_direction = np.empty((2, 1))
            # Iterate over each tuple in the list to find correct collision 
            # point
            for tup in tup_list:
                inter, centre, string = tup
                if inter[0] is None:
                    # Keep looping as no intercept
                    continue
                # Rounding needed for precision purposes, numpy will compare 
                # differences at a very small decimal places
                is_start = np.array_equal(np.round(inter[0], 6),
                                          np.round(start, 6))
                # Continue the loop if the intercept is the starting location
                if is_start:
                    continue
                # Checks which intercept is being dealt with
                if string == 'left':
                    if inter[0][0, 0] > centre:
                        continue
                elif string == 'right':
                    if inter[0][0, 0] < centre:
                        continue
                elif string == 'rect':
                    if (inter[0][0,0]==centre[0] or inter[0][0,0]==centre[1]):
                        if inter[0][1, 0] < self.vertices[0][1, 0] or inter[0][1, 0] > self.vertices[2][1, 0]:
                            inter = self._rectangleIntercept(inter[0], direction)
                        else:
                            continue
                collision_point, boundary_direction = inter
                break #<- if its the correct collision point don't keep looping

            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball, boundary_direction,
                                       collision_point)
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            i += 1
        return collision_points

    # NEW METHODS #

    def _leftCapIntercept(self, point, direction):
        '''
            Calculates where along the left cap the ball's vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction: 2x1 numpy array
                    column matrix for the balls direction vector
            
            Returns:
                (collision_point, boundary_direction) : tuple of 2 numpy arrays 
                    and a boolean.
                    
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
        '''
        # Overrides the inherrited ellipse centre to calculate the intercept of 
        # the left cap
        self.centre = self.cap_left_centre
        # The inheritted _calculateIntecept() method is from 
        # EllipticalBilliardBall
        return self._calculateEllipseIntercept(point, direction)

    def _rightCapIntercept(self, point, direction):
        '''
            Calculates where along the right cap the ball's vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction: 2x1 numpy array
                    column matrix for the balls direction vector
            
            Returns:
                (collision_point, boundary_direction) : tuple of 2 numpy arrays 
                and a boolean
                
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
        '''
        # Overrides the inherrited ellipse centre attribute to calculate the 
        # intercept of the left cap
        self.centre = self.cap_right_centre
        # The inheritted _calculateEllipseIntecept() method is from
        # EllipticalBilliardBall
        return self._calculateEllipseIntercept(point, direction)

    def _rectangleIntercept(self, point, direction):
        '''
            Calculates where in the rectangle the ball's vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction: 2x1 numpy array
                    column matrix for the balls direction vector
            
            Returns:
                (collision_point, boundary_direction) : tuple of 2 numpy arrays 
                and a boolean
                
                    collision point : column vector numpy array (2x1)
                        position vector for the collision point
                    boundary_direction : column vector numpy array (2x1)
                        direciton vector for the boundary
        '''
        # The inheritted _calculateLineIntecept() method is from 
        # PolygonBilliardBoard
        # Don't return the final element
        return self._calculateLineIntercept(point, direction)[:2]

# This is technically BunimovichBilliardBoardCase2

class BunimovichBilliardBoard(EllipticalBilliardBoard, PolygonBilliardBoard):
    '''
        Class which defines the 2D Bunimovich billiard board the ball moves 
        within.
        
        This is the second case of a Bunimovich billiard board being analysed.

        Child class of both PolygonBilliardBoard and EllipticalBilliardBoard.
        It inherits the collision point calculator methods from both
        of these classes. This is because its a board of a circular section 
        with a chord cutout in its lower half
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Calculate the max and min angles
        min_x = self.vertices[0][0, 0]
        max_x = self.vertices[1][0, 0]
        min_theta = np.arctan2(self.chord_height, min_x)
        max_theta = np.arctan2(self.chord_height, max_x)
        # Calculate all angles and trig funcs
        if self.chord_height == 0:
            angles = np.linspace(np.pi, 0, 1000)
        elif self.chord_height < 0:
            angles = np.linspace(min_theta+2*np.pi, max_theta, 1000)
        else:
            angles = np.linspace(min_theta, max_theta, 1000)
        cosine = np.cos(angles)
        sine = np. sin(angles)
        # Calculate the radius of the circle/ellipse at theta
        rs = self.a*self.b/np.sqrt((self.b*cosine)**2+(self.a*sine)**2)
        # x and y values for circle/ellipse
        x_ellipse = rs * cosine
        y_ellipse = rs * sine
        # Join the arrays together
        xs = np.concatenate((x_ellipse, self.vertices[0][0]))
        ys = np.concatenate((y_ellipse, self.vertices[0][1]))
        # Draw the board
        ax.plot(xs, ys, color = 'k', linewidth = 1, label = legend_label)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly chooses a point in the circle/ellipse with chord cutout
        while True:
            # Loop until correct starting point found, on average that should 
            # be higher the lower
            # the chord.
            # Ellipse ball starting point generator
            start = super()._generateBallStartingPoint()
            if start[1, 0] > self.chord_height:
                break
        return start

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : tuple
                    (width_x, width_y, chord_height)

                width_x, width_y and chord_height are all int or floats.
                The width_x and width_y determine the circle/ellipse size and
                chord_height determines the location of the chord which cuts 
                the cicle/ellipse. The chord height should determine the value
                of y to cut the ellipse at.
            
            Returns:
                None
        '''
        if additional_parameters is None:
            # Default values
            self.a = self.b = 1
            self.chord_height = -0.25
        else:
            self.a, self.b, self.chord_height = additional_parameters
            # Raises a ValueError if the width's aren't
            # greater than 0 and if chord_height is greater than width_y
            if self.a <= 0:
                message = str(self.a) + "must be greater than 0!"
                raise ValueError(message)
            elif self.b <= 0:
                message = str(self.b) + "must be greater than 0!"
                raise ValueError(message)
            elif np.abs(self.chord_height) >= self.b:
                message_1 = "cannot be larger than the ellipse height!"
                message = str(self.chord_height) + message_1
                raise ValueError(message)
        self.centre = np.array([[0], [0]])
        x = self.a * np.sqrt(1 - (self.chord_height / self.b) ** 2)
        bottom_left = np.array([[-1*x], [self.chord_height]])
        bottom_right = np.array([[x], [self.chord_height]])
        self.vertices = [bottom_left, bottom_right]

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:

                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on

                end : int
                    number of iterations to end the analysis at
            Returns:

                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            start = ball.start
            direction = ball.direction
            ellipse_intercept = self._calculateEllipseIntercept(start,
                                                                direction)
            if ellipse_intercept[0][1, 0] > self.chord_height:
                collision_point, boundary_direction = ellipse_intercept
            else:
                intercept_tup = self._calculateLineIntercept(start, direction)
                collision_point, boundary_direction, stop = intercept_tup
                if stop:
                    print("Corner hit, stopping...")
                    break
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball,boundary_direction,collision_point)
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            i += 1
        return collision_points

## LORENTZ GAS ##

class LorentzGasBilliardBoard(PolygonBilliardBoard, EllipticalBilliardBoard):
    '''
        Class which defines the 2D Lorentz gas billiard board the ball moves 
        within.

        Child class of both PolygonBilliardBoard and EllipticalBilliardBoard.
        It inherits the collision point calculator methods from both
        of these classes. This is because its a board of a rectangular section 
        with circular disk removed from the centre.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Calculate all angles and trig funcs
        angles = np.linspace(0, 2 * np.pi, 1000)
        cosine = np.cos(angles)
        sine = np. sin(angles)
        # Draw the circle cut out
        ax.plot(cosine, sine, color = 'k', linewidth = 1, label = legend_label)
        # Draw the square
        # Loop over all vertices
        for i in range(len(self.vertices)):
            point_1 = self.vertices[i]
            # Deals with the issue of the final side
            if i+1 == len(self.vertices):
                point_2 = self.vertices[0]
            else:
                point_2 = self.vertices[i+1]
            xs = [point_1[0][0], point_2[0][0]]
            ys = [point_1[1][0], point_2[1][0]]
            ax.plot(xs, ys, color = 'k', linewidth = 1)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly chooses a point within the rectangle but outside of the 
        # cicle cut out
        tol = 0.05 #<- tollerance either way of the max and min xy values
        theta_rand = rand(0, 2*np.pi)
        direction = np.array([[np.cos(theta_rand)], [np.sin(theta_rand)]])
        start = np.zeros((2, 1))
        square_intercept = self._calculateLineIntercept(start, direction)[0]
        circle_intercept = self._calculateEllipseIntercept(start, direction)[0]
        x_square = square_intercept[0, 0]
        y_square = square_intercept[1, 0]
        x_circ = circle_intercept[0, 0]
        y_circ = circle_intercept[1, 0]
        x = rand(x_circ+tol, x_square-tol)
        y = rand(y_circ+tol, y_square-tol)
        return np.array([[x], [y]])

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : None
                    Lorentz gas isn't modifiable
            
            Returns:
                None
        '''
        if additional_parameters is not None:
            # No additional parameters are needed. This is a special case 
            # billiard board
            message = "Too many parameters!"
            raise TypeError(message)
        self.a = 1 # Defines the max x circular centre disk radius
        self.b = 1 # Defines the max y circular centre disk radius
        self.centre = np.array([[0], [0]]) # Centre at zero
        top_right = np.array([[2], [2]])
        top_left = np.array([[-2], [2]])
        bottom_right = np.array([[2], [-2]])
        bottom_left = np.array([[-2], [-2]])
        self.vertices = [top_right, top_left, bottom_left, bottom_right]

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        prev_string = ''
        while i < end:
            start = ball.start
            direction = ball.direction
            if i == 0:
                # For the starting point
                circle_intercept = self._calculateEllipseIntercept(start,
                                                                   direction,
                                                                   True, True)
                square_intercept = self._calculateLineIntercept(start,
                                                                direction)
                if circle_intercept[0] is None:
                    # If the circle intercept doesn't exist then it can only be 
                    # the sqaure
                    intercept = square_intercept[:2]
                    prev_string = 'square'
                elif circle_intercept[2] > 0:
                    # If the direction scalar for the circle is wrong then its
                    # the square
                    intercept = square_intercept[:2]
                    prev_string = 'square'
                else:
                    # In every other case its the circle
                    intercept = circle_intercept[:2]
                    prev_string = 'circle'
            else:
                if prev_string == 'circle':
                    square_intercept = self._calculateLineIntercept(start,
                                                                    direction)
                    if square_intercept[2]:
                        # Corner hit, so break the loop
                        print("Corner hit, stopping...")
                        break
                    intercept = square_intercept[:2]
                    prev_string = 'square'
                else:
                    circ_inter = self._calculateEllipseIntercept(start,
                                                                 direction,
                                                                 True)
                    square_intercept = self._calculateLineIntercept(start,
                                                                    direction)
                    if circ_inter[0] is None or prev_string == 'circle':
                        if square_intercept[2]:
                            # Corner hit, so break the loop
                            print("Corner hit, stopping...")
                            break
                        intercept = square_intercept[:2]
                        prev_string = 'square'
                    else:
                        intercept = circ_inter
                        prev_string = 'circle'
            collision_point, boundary_direction = intercept
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball, boundary_direction,
                                       collision_point)
            i += 1
        return collision_points

## MUSHROOM ##
class MushroomBilliardBoard(PolygonBilliardBoard, EllipticalBilliardBoard):
    '''
        Class which defines the 2D Mushroom billiard board the ball moves
        within.

        Child class of both PolygonBilliardBoard and EllipticalBilliardBoard.
        It inherits the collision point calculator methods from both
        of these classes. This is because its a board of a elliptical head with
        top hat base.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Calculate all angles and trig funcs
        angles = np.linspace(0, np.pi, 1000)
        cosine = np.cos(angles)
        sine = np. sin(angles)
        # Calculate the radius of the ellipse at theta
        rs = self.a*self.b/np.sqrt((self.b*cosine)**2+(self.a*sine)**2)
        # Draw the circle cut out
        ax.plot(rs * cosine, rs * sine, color = 'k',
                linewidth = 1, label = legend_label)
        # Draw the square
        # Loop over all non virtual vertices
        for i in range(2, len(self.vertices)-1):
            point_1 = self.vertices[i]
            # Deals with the issue of the final side
            point_2 = self.vertices[i+1]
            xs = [point_1[0][0], point_2[0][0]]
            ys = [point_1[1][0], point_2[1][0]]
            ax.plot(xs, ys, color = 'k', linewidth = 1)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly chooses a point within the rectangle but outside of the 
        # cicle cut out
        tol = 0.05 #<- tollerance either way of the max and min xy values
        # Calculate the possible ellipse start point
        theta_rand = rand(0, np.pi)
        cosine = np.cos(theta_rand)
        sine = np.sin(theta_rand)
        # Calculate the radius of the ellipse at theta, with the tolerance
        # included
        rs = self.a*self.b/np.sqrt((self.b*cosine)**2+(self.a*sine)**2)-tol
        rand_rs = rand(0, rs)
        x_ellipse = rand_rs * cosine
        y_ellipse = rand_rs * sine
        if y_ellipse <= 0:
            y_ellipse = tol
        ellipse_start = np.array([[x_ellipse], [y_ellipse]])
        # Calculate the possible rectangle start
        rect_x = rand(self.vertices[3][0, 0]+tol, self.vertices[6][0, 0]-tol)
        rect_y = rand(self.vertices[3][1, 0], self.vertices[4][1, 0]+tol)
        rect_start = np.array([[rect_x], [rect_y]])
        # Store the random ellipse and rectangle start points to randomly pick
        possible_starts = [ellipse_start, rect_start]
        rand_index = randint(0, 1) # Random index to choose the start location
        return possible_starts[rand_index]

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:

                additional_parameters : tuple
                    (width, height, cap_height, cap_width)

                width, height and cap_width are all int or floats.

                The width determines the width of the mushroom stem, 
                height determines the height of the mushroom stem,
                cap_height determines the circular/elliptical cap height,
                cap_width determines the cicular/elliptical cap width.
            
            Returns:
                None
        '''
        if additional_parameters is None:
            width = 1 
            height = 1
            cap_width = 3
            cap_height = 1
        else:
            width, height, cap_height, cap_width = additional_parameters
            # Raise ValueErrors if each parameter is of the incorrect value
            message = None
            if width <= 0:
                message = 'Width of rectangular base must be greater than 0!'
            elif height <= 0:
                message = 'Height of rectangular base must be greater than 0!'
            elif cap_height <= 0:
                message = 'Height of mushroom head must be greater than 0!'
            elif cap_width <= 0:
                message = 'Width of mushroom head must be greater than 0!'
            elif width >= cap_width:
                message = "Mushroom width must be greater than the stem width!"
            if message is not None:
                raise ValueError(message)
        self.a = cap_width / 2 # Defines the max x mushroom head radius
        self.b = cap_height # Defines the max y elliptical mushroom head radius
        self.centre = np.array([[0], [0]]) # Centre at zero
        # Defines the 
        top_right = np.array([[cap_width / 2], [0]])
        top_mid_right = np.array([[width / 2], [0]])
        bottom_right = np.array([[width / 2], [-1 * height]])
        bottom_left = np.array([[-1 * width / 2], [-1 * height]])
        top_mid_left = np.array([[-1 * width / 2], [0]])
        top_left = np.array([[-1* cap_width / 2], [0]])
        # Virtual vertices used to close the polygon section for calculations
        virtual_left = np.array([[-1* cap_width / 2], [cap_height]])
        virtual_right = np.array([[cap_width / 2], [cap_height]])
        self.vertices = [virtual_right, virtual_left, top_left, top_mid_left,
                         bottom_left, bottom_right, top_mid_right, top_right]

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            start = ball.start
            direction = ball.direction
            mushroom_base_inter = self._rect_intercept(start, direction)
            mushroom_head_inter = self._calculateEllipseIntercept(start,
                                                                  direction)
            # If the base intercept doesn't exist, its the head intercept
            if mushroom_base_inter[0] is None:
                intercept = mushroom_head_inter
            else:
                if mushroom_base_inter[2]:
                    print("Corner hit, stopping...")
                    break
                intercept = mushroom_base_inter[:2]
            collision_point, boundary_direction = intercept
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball,boundary_direction,collision_point)
            i += 1
        return collision_points

    # New Methods #
    def _rect_intercept(self, point, direction):
        '''
            Calculates where along the ellipse the ball's vector intercepts

            Parameters:
                point : 2x1 numpy array
                    column matrix for the balls position along its vector
                    line
                
                direction : 2x1 numpy array
                    column matrix for the balls direction vector
            
            Returns:
                (collision_point, boundary_direction, end) : tuple of 2 numpy 
                arrays and a boolean
                
                    collision point : column vector numpy array (2x1) or None 
                    (when not a valid intercept)
                        position vector for the collision point
                        
                    boundary_direction : column vector numpy array (2x1) or
                    None (when not a valid intercept)
                        direciton vector for the boundary
                        
                    end : boolean or None (when not a valid intercept)
                        If True, end the calculations.
        '''
        # Unpack vertices tuple to get the max and min y values
        virtual_left, top_left = self.vertices[1:3]
        rectangle_intercept = self._calculateLineIntercept(point, direction)
        y_intercept = rectangle_intercept[0][1, 0]
        if y_intercept <= virtual_left[1, 0] and y_intercept > top_left[1, 0]:
            # Not a valid intercept if along the virtual sides of the polygon
            rectangle_intercept = (None,) * 3
        return rectangle_intercept

## 'HYPERBOLIC LORENTZ GAS' ##
# This is a type of Wojtkowski Billiard Board
class HyperbolicLorentzGasBilliardBoard(PolygonBilliardBoard,
                                        HyperbolicBilliardBoard):
    '''
        Class which defines the 2D `Hyperbolic Lorentz gas' billiard board the 
        ball moves within.

        Child class of both PolygonBilliardBoard and EllipticalBilliardBoard.
        It inherits the collision point calculator methods from both
        of these classes. This is because its a board of a rectangular section 
        with circular disk removed from the centre.
    '''

    ## PUBLIC METHODS ##
    
    # OVERRIDDEN METHODS #

    def drawBoard(self, ax, legend_label = None):
        '''
            Draws the billiard board

            Parameters:
                ax : matplotlib ax object
                    the ax object to draw on

                legend_label : string or None (optional)
                    the label for the legend
        '''
        # Draw the hyperbolae cut out
        # x and y values for top and bottom
        xs_tb = np.linspace(self.min_x, self.max_x)
        ys_t = self.c * np.sqrt(1 + (xs_tb / self.d) ** 2)
        ys_b = -1 * ys_t
        # x and y for left and right
        ys_lr = np.linspace(self.min_y, self.max_y)
        xs_r = self.a * np.sqrt(1 + (ys_lr / self.b) ** 2)
        xs_l = -1 * xs_r
        # Join arrays together
        xs = np.concatenate((xs_tb, xs_r[::-1], xs_tb[::-1], xs_l))
        ys = np.concatenate((ys_t, ys_lr[::-1], ys_b[::-1], ys_lr))
        # Draw the board
        ax.plot(xs, ys, color = 'k', linewidth = 1, label = legend_label)
        # Draw the square
        # Loop over all vertices
        for i in range(len(self.vertices)):
            point_1 = self.vertices[i]
            # Deals with the issue of the final side
            if i+1 == len(self.vertices):
                point_2 = self.vertices[0]
            else:
                point_2 = self.vertices[i+1]
            xs = [point_1[0][0], point_2[0][0]]
            ys = [point_1[1][0], point_2[1][0]]
            ax.plot(xs, ys, color = 'k', linewidth = 1)

    ## PRIVATE METHODS ##

    # OVERRIDDEN METHODS #
    
    def _generateBallStartingPoint(self):
        '''
            Generates the starting vector of the billiard ball based
            on the geometry of the board.

            Parameters:
                None
            
            Returns:
                start : numpy array; 2 x 1 matrix; column matrix of shape 2x1
                    starting point vector for the ball
        '''
        # Randomly chooses a point within the rectangle but outside of the 
        # cicle cut out
        tol = 0.05 #<- tollerance either way of the max and min xy values
        theta_rand = rand(0, 2*np.pi)
        direction = np.array([[np.cos(theta_rand)], [np.sin(theta_rand)]])
        start = np.zeros((2, 1))
        square_intercept = self._calculateLineIntercept(start, direction)[0]
        hyperbolae_intercept = self._calculateHyperbolicIntercept(start,
                                                                  direction)[0]
        x_square = square_intercept[0, 0]
        y_square = square_intercept[1, 0]
        x_hyp = hyperbolae_intercept[0, 0]
        y_hyp = hyperbolae_intercept[1, 0]
        x = rand(x_hyp+tol, x_square-tol)
        y = rand(y_hyp+tol, y_square-tol)
        return np.array([[x], [y]])

    def _assignAdditionalAttributes(self, additional_parameters):
        ''' 
            Used to assign additional attributes to the class

            Parameters:
                additional_parameters : None
                    Hyperbolic lorentz gas cannot be altered
            
            Returns:
                None
        '''
        if additional_parameters is not None:
            # No additional parameters are needed. This is a special case 
            # billiard board
            message = "There cannot be any additional parameters!"
            raise TypeError(message)
        self.a = self.b = 1
        # c and d are the top and bottom hyperbolae widths and heights
        self.c = 0.5 * self.b
        self.d = 0.75 * self.a
        top_right = np.array([[2], [2]])
        top_left = np.array([[-2], [2]])
        bottom_right = np.array([[2], [-2]])
        bottom_left = np.array([[-2], [-2]])
        self.vertices = [top_right, top_left, bottom_left, bottom_right]
        # max and min y's and x's
        denominator = ((self.d * self.b) ** 2 - (self.a * self.c) ** 2)
        numerator = (self.b * self.c) ** 2 * (self.d ** 2 + self.a ** 2)
        self.max_y = np.sqrt(numerator / denominator)
        self.min_y = -1 * self.max_y
        self.max_x = np.sqrt(self.a ** 2 * (1 + (self.max_y / self.b ) ** 2))
        self.min_x = -1 * self.max_x 

    def _generateCollisionPoints(self, ball, end):
        '''
            Generates the collision points for the ball.

            Parameters:
                ball : a <BilliardBall> object
                    the billiard ball object to perform the analysis on
                end : int
                    number of iterations to end the analysis at
            Returns:
                collision_points : list of tuples
                    returns a list of collision points up to the end value
        '''
        collision_points = None
        i = 0
        while i < end:
            start = ball.start
            direction = ball.direction
            hyperbolae_inter = self._calculateHyperbolicIntercept(start,
                                                                  direction,
                                                                  True,
                                                                  True)
            square_intercept = self._calculateLineIntercept(start, direction)
            if hyperbolae_inter[0] is None:
                # If the hyperbola intercept doesn't exist then it can only be 
                # the rectangle
                intercept = square_intercept[:2]
            elif hyperbolae_inter[2] < 0:
                # If the direction scalar for the hyperbola is wrong then its
                # the square
                intercept = square_intercept[:2]
            else:
                # In every other case its the hyperbola
                intercept = hyperbolae_inter[:2]
            collision_point, boundary_direction = intercept
            if collision_points is None:
                collision_points = collision_point
            else:
                # Join the collision point column vectors together
                collision_points = np.concatenate((collision_points,
                                                   collision_point), axis = 1)
            # Change the vector line of the ball and its current direction
            self._changeBallVectorLine(ball, boundary_direction,
                                       collision_point)
            i += 1
        return collision_points