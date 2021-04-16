import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import data
from skimage import io
from skimage import filters
from scipy import ndimage
from skimage.feature import match_template

def convolution(img_matrix, fltr_matrix):
    """
    Given an image and a filter, return the convolution of
    the image with the filter. Assumptions, filter is a square n x n
    with odd dimensions.
    """

    # flip the filter vertically and horizontaly.
    
    fltr_matrix = fltr_matrix[::-1,::-1]

    fltr_shape = fltr_matrix.shape[0]
    padding = fltr_shape - 1
    
    # zero pad the original image to do the convolution.
    
    padded = np.zeros((img_matrix.shape[0] + padding, img_matrix.shape[1] + padding))
    # copy in the image matrix between the padding in the matrix
    for row in range(padding//2, padded.shape[0] - (padding//2)):
        for col in range(padding//2, padded.shape[1] - (padding//2)):
            padded[row,col] = img_matrix[row - (padding//2), col - (padding//2)]

    # loop through all the pixels and apply filter to padded matrix, place in output matrix
    # output matrix same size as the input matrix
    result = np.zeros(img_matrix.shape)

    # let's fill the result!
    for row in range(0, result.shape[0]):
        for col in range(0, result.shape[1]):
            # get the padded image section
            img_section = padded[row:(row + fltr_shape),col:(col + fltr_shape)]
            # do the convolution calculation using the dot product by flattening the matrices
            convolution_result = img_section.flatten().dot(fltr_matrix.flatten())
            result[row, col] = convolution_result
    return result

waldo = io.imread("waldo.png")
filtr = ndimage.gaussian_filter(waldo, sigma=1, order=0)
#io.imshow(filtr)
#io.show()

def separable(fltr, image):
    """
    1) check to see if filter is separable
    2) perform a faster convolution with the given image

    NOTE: assuming that filter and image are in matrix form
    
    """
    #flip the filter
    new_fltr = fltr[::-1,::-1]
    # check to see if filter is separable
    U, S, V = np.linalg.svd(new_fltr)

    # Assuming that the 2nd elemnt is close to zero, gives room for
    # python rounding errors 
    if (round(S[1], 15) != 0):
        return "Filter is not separable"
    
    # function is separable, continue
    sigma = S[0]
    # The vertical and horizontal filters from first part
    vertical = math.sqrt(sigma) * np.asmatrix(V[0])
    horizontal = math.sqrt(sigma) * np.asmatrix(U[:,0])
    
    # Using python built in function, first convolve image with
    # horizontal filter, then with the vertical filter.

    horizontal_output = ndimage.convolve(image, horizontal)

    return ndimage.convolve(horizontal_output, vertical.T)

example = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])

def laplachian(sigma):
    # size of the kernel will be 5 x 5 matrix
    h_x = np.zeros((5,5))
    # using the second partial derivative of the Gaussian with respect to x, we get:
    for row in range(5):
        for col in range(5):
            power = ((row - 2)**2 - (sigma ** 2)) * np.exp(-((row - 2) ** 2 + (col - 2) ** 2) / (2 * sigma ** 2))
            h_x[row,col] = power / (2 * np.pi * sigma ** 6)

    h_y = np.zeros((5,5))
    # using the second partial derivative of the Gaussian with respect to y, we get:
    for row in range(5):
        for col in range(5):
            power = ((col - 2)**2 - (sigma ** 2)) * np.exp(-((row - 2) ** 2 + (col - 2)** 2) / (2 * sigma ** 2))
            h_x[row,col] = power / (2 * np.pi * sigma ** 6)
    # adding the two to get the Laplacian of Gaussians
    return h_y + h_x

def gaussian(img, sigma):
    img_matrix = io.imread(img, as_gray=True)
    gauss = np.zeros((9,9))
    for row in range(3):
        for col in range(3):
            power =  np.exp(-((row - 4) ** 2 + (col - 4) ** 2) / (2 * sigma ** 2))
            gauss[row,col] = power / (2 * np.pi * sigma ** 2)
          
    filtr = ndimage.convolve(img_matrix, gauss)
    io.imshow(filtr)
    io.show()

def magnitude_gradient(img):
    img_matrix = io.imread(img,as_gray=True)
    
    # Using the Prewitt filter for the horizontal gradient
    horizontal = ndimage.convolve(img_matrix, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))

    # Using the Prewitt filter for the vertical gradient
    vertical = ndimage.convolve(img_matrix, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
    final = np.zeros(img_matrix.shape)                            
    # square rooting the sum of squares of vertical and horizontals                          
    for row in range(final.shape[0]):
        for col in range(final.shape[1]):
            final[row,col] = math.sqrt(vertical[row,col] ** 2 + horizontal[row,col] ** 2)
    #io.imshow(final)
    #io.show()
    return final        

def grid_matching():
    # using function from part a to compute the gradient magnitude of the images
    img_gradient = magnitude_gradient("waldo.png")
    fltr_gradient = magnitude_gradient("template.png")
    #return img_gradient
    # using template matching to find result
    result = match_template(img_gradient, fltr_gradient)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x,y = ij[::-1]
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8, 3))

    ax1.imshow(fltr_gradient)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(img_gradient)
    ax2.set_axis_off()
    ax2.set_title('waldo')
    # highlight matched region
    xwaldo, ywaldo = fltr_gradient.shape
    rect = plt.Rectangle((x, y), ywaldo, xwaldo, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()

def canny_edge(img):
    # read the image
    img_matrix = io.imread(img,as_gray=True)

    # Apply the Gaussian filter to reduce noise
    img_matrix = ndimage.gaussian_filter(img_matrix, sigma=1, order=0)
    
    # get the gradient magnitude
    # as Q3A
    horizontal = ndimage.convolve(img_matrix, np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))

    vertical = ndimage.convolve(img_matrix, np.array([[1,1,1],[0,0,0],[-1,-1,-1]]))
    gradient = np.zeros(img_matrix.shape)
    
    # square rooting the sum of squares of vertical and horizontals
    for row in range(gradient.shape[0]):
        for col in range(gradient.shape[1]):
            gradient[row,col] = math.sqrt(vertical[row,col] ** 2 + horizontal[row,col] ** 2)
            
    # get the angles for each pixel
    angles = np.zeros(img_matrix.shape)
    
    for row in range(angles.shape[0]):
        for col in range(angles.shape[1]):
            current_angle = np.arctan2(vertical[row,col],horizontal[row,col])
            # convert to degrees for simplicity
            current_angle = current_angle * 180 / np.pi
            if current_angle < 0:
                current_angle += 180
            angles[row, col] = current_angle        
    # apply non maximum suppression
    non_max = np.zeros(img_matrix.shape)
    for row in range(1,non_max.shape[0]-1):
        for col in range(1,non_max.shape[1]-1):
            # Find the edge direction
            direction = 45 * round(angles[row,col] / 45)
            # save the neighbor edge strengths
            if (direction == 0) or (direction == 180):
                left = gradient[row, col-1]
                right = gradient[row, col+1]
            elif (direction == 45):
                left = gradient[row + 1, col - 1]
                right = gradient[row - 1, col + 1]
            elif (direction == 90):
                left = gradient[row - 1, col]
                right = gradient[row + 1, col]
            else:
                left = gradient[row - 1, col - 1]
                right = gradient[row + 1, col + 1]

            # compare edge strengths of current pixel with neighbors in gradient direction
            if (left < gradient[row,col] and right < gradient[row,col]):
                non_max[row,col] = gradient[row,col]
            else:
                non_max[row,col] = 0
                 
    io.imshow(non_max)
    io.show()       
    
    

"""
dx = 0.1
dy = 0.1
x = np.arange(-10, 10, dx)
y = np.arange(-10, 10, dy)
x2d, y2d = np.meshgrid(x, y)
kernel_x = (x2d**2 - (3 ** 2)) * np.exp(-(x2d ** 2 + y2d ** 2) / (2 * 3 ** 2))
kernel_x = kernel_x / (2 * np.pi * 3 ** 6)
kernel_y = (y2d**2 - (3 ** 2)) * np.exp(-(x2d ** 2 + y2d ** 2) / (2 * 3 ** 2))
kernel_y = kernel_y / (2 * np.pi * 3 ** 6)
final = kernel_x + kernel_y
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x2d, y2d, kernel_2d)
plt.show()
"""
