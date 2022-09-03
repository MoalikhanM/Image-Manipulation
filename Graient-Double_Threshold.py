from PIL import Image
import math
import numpy as np

def odd(n):
    # find the next odd number of the given number by formula below
    return round(n/2)*2+1

# create 1d gaussian filter
def gauss1d(sigma):
    # firstly we compute the length of our 1d array
    length = odd(sigma*6)
    # then we create an array with values representing how far
    # current element from the center 
    # so we use arange with max distances as boundaries 
    # and step 1
    fil = np.arange(-(length-1)//2, (length+1)//2, 1)
    # then we use map function and gaussion formula as lambda to create
    # not normalized 1d array from previous array
    notNormalizedArray = list(map(lambda x: math.exp(-x**2/(2*sigma**2)), fil))
    # we normalize an array so the sum is equal to 1
    normalizedArray = np.array(notNormalizedArray)/sum(notNormalizedArray)
    return normalizedArray

# create 2d gaussian filter
def gauss2d(sigma):
    # 2d gaussian filter is created by outer product
    # of 1d gaussian filters with the same value of sigma
    array1d = gauss1d(sigma)
    array2d = np.outer(array1d, array1d)
    return array2d

# this function performs convolution to the given array by the given filter
def convolve2d(array, filter):
    # reverse the filter for convolution
    filter = np.fliplr(filter)
    filter = np.flipud(filter)
    # firstly we take the sizes of filter, original image,
    # and new image without padding
    filterSize = len(filter)
    originalSizeX, originalSizeY = array.shape
    # to find new size we substract the width of padding from two sides
    sizeWithoutPaddingX = originalSizeX - (filterSize//2)*2 
    sizeWithoutPaddingY = originalSizeY - (filterSize//2)*2 
    # create empty 2d array for new image
    newImg = np.zeros((sizeWithoutPaddingX, sizeWithoutPaddingY))
    # iterate through each neighborhood
    for i in range(sizeWithoutPaddingX):
        for j in range(sizeWithoutPaddingY):
            # take only the neighborhood needed right now
            neighborhood = array[i:i+filterSize, j:j+filterSize]
            # perform convolution and save the result into new image
            newImg[i,j] = np.sum(np.multiply(neighborhood, filter))
    # after convolution is done we add padding 
    # by creating new 2d array of zeros with original size
    newImgWithPadding = np.zeros((originalSizeX, originalSizeY))
    # we put the convolved image inside the new 2d array, so 
    # the borders became padding zeros needed to keep the size
    newImgWithPadding[filterSize//2:-filterSize//2+1, filterSize//2:-filterSize//2+1] = newImg
    return newImgWithPadding

# this function performs convolution by using 
# gaussian 2d filter from given sigma value
def gaussconvolve2d(array, sigma):
    return convolve2d(array, gauss2d(sigma))

#connect two images to compare
def connectImages(im1, im2):
    # create new image with extended width
    result = Image.new('RGB', (im1.width + im2.width, im1.height))
    # put images together
    result.paste(im1, (0, 0))
    result.paste(im2, (im1.width, 0))
    #result.show()
    return result

# this function applies gaussconvolve2d() function
# on the picture of the iguana
def part1():
    # we open the iguana picture, transform it into
    # grayscale and then convert it into numpy array
    im = Image.open('iguana.bmp')
    imGrey = im.convert('L')
    
    imArray = np.asarray(imGrey)
    # dtype of numpy array should be float32
    imArray = imArray.astype('float32')
    # we perform convolution on the picture with sigma = 1.6
    imNewArray = gaussconvolve2d(imArray, 1.6)
    # convert values into unit8 so we can save it
    imNewArray = imNewArray.astype('uint8')
    # create picture from new array and save it
    imNew = Image.fromarray(imNewArray)
    imNew.save('iguanaNew.bmp','BMP')
    # show new image next to the original
    connectImages(im, imNew).save('Iguanas.bmp')
    return imNewArray
    
def sobel_filters(img):
    # create filters
    filterX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    filterY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # create two arrays for different directions
    newX = convolve2d(img, filterX)
    newY = convolve2d(img, filterY)
    
    # calculate the gradient and map it into 0-255
    G = np.hypot(newX, newY)
    G *= 255.0/G.max()
    # calculate angles
    theta = np.arctan2(newX, newY)
    
    # create image from gradient magnitudes
    imNewArray = G.astype('uint8')
    imNew = Image.fromarray(imNewArray)
    imNew.save('G.bmp','BMP')
    return (G, theta)

# this part applies sobel filters and takes G and theta from function
def part2(greyIm):
    G, theta = sobel_filters(greyIm)
    return G, theta

def non_max_suppression(G, theta):
    # the shape og new image
    rows, cols = G.shape
    result = np.zeros((rows,cols))
    
    # convert the angle to degrees and make them all positive
    theta = theta * 180.0 / np.pi
    theta[ theta<0 ] += 180

    # go through each internal pixel to check is it local max or not
    for i in range(1 ,rows-1):
        for j in range(1, cols-1):
            # take neighbor pixels according to the angle
            # for angle 0
            if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                pix1 = G[i, j+1]
                pix2 = G[i, j-1]
            # for angle 45
            elif (22.5 <= theta[i,j] < 67.5):
                pix1 = G[i+1, j-1]
                pix2 = G[i-1, j+1]
            #for angle 90
            elif (67.5 <= theta[i,j] < 112.5):
                pix1 = G[i+1, j]
                pix2 = G[i-1, j]
            #for angle 135
            elif (112.5 <= theta[i,j] < 157.5):
                pix1 = G[i-1, j-1]
                pix2 = G[i+1, j+1]
                
            # check if this pixel is local maximum
            if (G[i,j] >= pix1) and (G[i,j] >= pix2):
                result[i,j] = G[i,j]
            else:
                result[i,j] = 0
    
    # save the final result in the image
    imNewArray = result.astype('uint8')
    imNew = Image.fromarray(imNewArray)
    imNew.save('Non_max.bmp','BMP')
    return result

# this part applies non max suppresion on last image we got
def part3(G, theta):
    return non_max_suppression(G, theta)

def double_thresholding(img):
    # calculate threshold values by formulas given
    diff = img.max() - img.min()
    high = img.min() + diff* 0.15
    low = img.min() + diff*0.03
    
    # create array for the result filled with zeros
    result = np.zeros(img.shape)
 
    # determine the positions of strong pixels and weak pixels
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))

    # assign pixels according to the class
    result[strong_i, strong_j] = np.float32(255)
    result[weak_i, weak_j] = np.float32(80)
    
    # save the final result in the image
    imNewArray = result.astype('uint8')
    imNew = Image.fromarray(imNewArray)
    imNew.save('Double_Threshold.bmp','BMP')
    
    return result

# this part applies double thresholding on non max suppressed image
def part4(img):
    return double_thresholding(img)
    
# Depth First Search
def DFS(img, i, j, weak, strong):
    rows, cols = img.shape
    # if the parameters out of boundaries we finish
    if(i<=0 or i>=rows or j<=0 or j>=cols):
        return
    # if the pixel is weak we make it strong and perform DFS on its neighbors
    if(img[i][j]==weak):
        img[i][j]=strong
        final[i][j] = strong
        DFS(img, i-1, j, weak, strong)
        DFS(img, i, j-1, weak, strong)
        DFS(img, i-1, j-1, weak, strong)
        DFS(img, i+1, j, weak, strong)
        DFS(img, i, j+1, weak, strong)
        DFS(img, i+1, j+1, weak, strong)
        DFS(img, i-1, j+1, weak, strong)
        DFS(img, i+1, j-1, weak, strong)

def hysteresis(img):
    # take shape to go through pixels
    rows, cols = img.shape
    # strong and weak values to compare
    strong = np.float32(255)
    weak = np.float32(80)
    # going through internal pixels
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # if the pixel is strong we perform DFS on its neighbors
            if (img[i][j] == strong):
                final[i][j] = strong
                DFS(img, i-1, j, weak, strong)
                DFS(img, i, j-1, weak, strong)
                DFS(img, i-1, j-1, weak, strong)
                DFS(img, i+1, j, weak, strong)
                DFS(img, i, j+1, weak, strong)
                DFS(img, i+1, j+1, weak, strong)
                DFS(img, i-1, j+1, weak, strong)
                DFS(img, i+1, j-1, weak, strong)
                    
    # save the final result in the image
    imNewArray = final.astype('uint8')
    imNew = Image.fromarray(imNewArray)
    imNew.save('Final.bmp','BMP')
    
    return img

def part5(img):
    return hysteresis(img)


greyIm = part1()
G, theta = part2(greyIm)
non_max_suppres_img = part3(G, theta)
double_threshold_img = part4(non_max_suppres_img)
global final
final = np.zeros(double_threshold_img.shape)
final_image = part5(double_threshold_img)

