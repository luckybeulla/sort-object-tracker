import cv2
import numpy as np


def calculate_hogs(grid, cell_size=8, stride=1, block_size=2, num_bins=9):
    '''
    Input is a cropped image with width:height of 1:2.

    If the detection window (grid) is size 64 x 128, cells are
    8 x 8, stride length is 1, block size is 2 x 2 cells, and there
    are 9 orientation bins, then there are (64 x 128)/8 = 8 x 16
    cells. Since blocks are 2 x 2 cells, there are 
    (8 - 1) x (16 - 1) = 7 x 15 = 105 2 x 2 blocks. There are 
    9 * 4 = 36 features per block, so 36 * 105 = 3780 features
    total. 

    In general, output is a
    (bin_size * block_size^2) * ((width/cell_size - 1) * (height/cell_size -1)) x 1
    NumPy array.
    '''
    blocks = []
    fvs = {}
    for y in range(0, len(grid)-block_size+1):
        for x in range(0, len(grid[0]), stride):
            # Calculate feature vector for each cell
            fvs[(x,y)] = calculate_feature_vector()
            


def calculate_hog(cell):
    # Cell is a square

    grad_mags = []
    grad_angles = []

    for y in range(1, len(cell)-1):
        for x in range(1, len(cell[0]-1)):
            # Calculate gradient magnitude and orientation of pixel at (x, y)
            xgrad = abs(cell[y, x+1]-cell[y, x-1])
            ygrad = abs(cell[y+1, x]-cell[y-1, x])
            grad_mags.append(np.sqrt(xgrad**2 + ygrad**2))
            grad_angles.append(np.degrees(np.arctan(ygrad/xgrad))) # np.arctan is in radians

    return grad_mags, grad_angles


def calculate_feature_vector(cell):
    grad_mags, grad_angles = calculate_hog(cell)

    fv = {}

    for i in range(len(grad_angles)):
        if (grad_angles[i]//20)*20 in fv.keys():
            fv[(grad_angles[i]//20)*20]+=grad_mags[i]

        else:
            fv[(grad_angles[i]//20)*20] = grad_mags[i]

    return fv