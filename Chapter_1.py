#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:29:53 2019

@author: jonas.palacionis
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import color, viewer, exposure, img_as_float, data
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from scipy.ndimage import affine_transform, zoom
from scipy import misc
from skimage import color
from skimage import data
from imageio import imsave
from matplotlib import cm
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from scipy import stats



#import an image and display its parameters and show it
im = Image.open('/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot.png')
print(im.width, im.height, im.mode, im.format, type(im))
im.show()

#using the PIL function convert() to convert it to grayscale image
im_g = im.convert('L')  # convert to RGB color imgae to a grayscale image
im_g.save('/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_gray.png')
Image.open('/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_gray.png')

#using matplotlib imread() to read an image in a floating-point numpy ndarray

im = mpimg.imread('/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.png')
print(im.shape, im.dtype, type(im))
plt.figure(figsize=(10,10))
plt.imshow(im)
plt.axis('off')
plt.show()


# making the image darker by setting all of the pixel values below 0.5 to 0 and then saving the numpy ndarray to disk
im_1 = im
im_1[im_1 < 0.5] = 0
plt.imshow(im_1)
plt.axis('off')
plt.tight_layout()
plt.savefig('/Users/jonas.palacionis/Desktop/Image Processing/Images/hill_darker.png')
im = mpimg.imread('/Users/jonas.palacionis/Desktop/Image Processing/Images/hill_darker.png')
plt.figure(figsize = (10,10))
plt.imshow(im)
plt.axis('off')
plt.tight_layout()
plt.show()

#using different methods of interpolation with the imshow()
im = mpimg.imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/lena_small.jpg")  # read the image from disk as a numpy ndarray
methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9.3, 6),subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(im, interpolation=interp_method, cmap='viridis')
    ax.set_title(str(interp_method))
plt.tight_layout()
plt.show()


#reading, saving, and displaying an image using sckit-image
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.png")
print(im.shape, im.dtype, type(im))
hsv = color.rgb2hsv(im)
hsv[:,:,1] = 0.9
im1 = color.hsv2rgb(hsv)
imsave("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_saturation.png", im1)
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_saturation.png")
plt.axis('off')
imshow(im)
show()


#using imageio to read an image
im = imageio.imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/veg.jpg")
print(type(im), im.shape, im.dtype)
plt.imshow(im)
plt.axis('off')
plt.show()


#converting from one file to another
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_saturation.png")
print(im.mode)
im.save("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_saturation.jpg")
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.png")
print(im.mode)
im.convert('RGB').save("/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.jpg")
print(im.mode)


#converting from one image mode into another
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.png", as_gray = True)
print(im.shape)
plt.imshow(im)

    
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/Ishihara.png")
img_g = color.rgb2gray(im)
plt.subplot(121)
plt.imshow(im,cmap = 'gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_g, cmap = 'gray')
plt.axis('off')
plt.show()

#converting from one color space into another
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.png")
im_hsv = color.rgb2hsv(im)
plt.gray()
plt.figure(figsize = (10, 8))
plt.subplot(221), plt.imshow(im_hsv[...,0]), plt.title('h', size = 20)
plt.axis('off')
plt.subplot(222), plt.imshow(im_hsv[...,1]), plt.title('s', size = 20)
plt.axis('off')
plt.subplot(223), plt.imshow(im_hsv[...,2]), plt.title('v', size = 20)
plt.axis('off')
plt.subplot(224), plt.imshow(im), plt.title('normal', size = 20)
plt.axis('off')
plt.show()


#converting image datq structures from PIL to numpy ndarray ( to be consumed by scikit-image )
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/flowers.png")
im = np.array(im)
imshow(im)
plt.axis('off')
show()

#from numpy to PIL
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/flowers.png")
im = Image.fromarray(im)
im.show()


#image manipulations with numpy array slicing
lena = mpimg.imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/lena.jpg")
#print(lena[0,40])
#print(lena[10:13, 20:23, 0:1])
lx, ly, _ = lena.shape
X, Y = np.ogrid[0:lx, 0: ly]
mask = ( X - lx /2 ) ** 2 + ( Y - ly /2) ** 2 > lx * ly / 4
lena = np.array(Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/lena.jpg"))
lena[mask,:] = 0
plt.figure(figsize=(10,10))
plt.imshow(lena)
plt.axis('off')


#image morphing using cross-dissolving

im1 = mpimg.imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/messi.jpg") / 255 # scale RGB values in [0, 1]
im2 = mpimg.imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/ronaldo.jpg") / 255
i = 1
plt.figure(figsize=(18,15))
for alpha in np.linspace(0,1,20):
    plt.subplot(4,5,i)
    plt.imshow((1-alpha)*im1 + alpha*im2)
    plt.axis('off')
    i += 1
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()


#image manipulations with PIL
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.jpg")
print(im.width, im.height, im.mode, im.format)
im_c = im.crop((175,75,320,200)) # cropping rectangle using left, top, right, bottom points
im_c.show()

#resizing to a bigger image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/clock.jpg")
print(im.width, im.height)
im.show()
im_large = im.resize((im.width*5, im.height*5), Image.BILINEAR) # bi-linear interpolation
im_large.show()

#resizing to a smaller image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/vic.png")
print(im.width, im.height)
im_small = im.resize((im.width//5, im.height//5), Image.ANTIALIAS)
im_small.show()

#negating an image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.jpg")
im_t = im.point(lambda x: 255 - x)
im_t.show()


#converting an image into grayscale
im_g = im.convert('L')
im_g.show()


#log gray-level transformations
im_g.point(lambda x: 255*np.log(1+x/225)).show()

#power-law transformation
im_g.point(lambda x: 255*(x/255)**0.6).show()

#geometric transformations
im.transpose(Image.FLIP_LEFT_RIGHT).show()
im_45 = im.rotate(45).show()

#applying an affine transformation on an image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.jpg")
im.transform((int(1.4*im.width), im.height), Image.AFFINE, data = (1, -0.5, 0, 0, 1, 0)).show()


#perspective transformation
params = [1, 0.1, 0, -0.1, 0.5, 0, -0.005, -0.001]
im1 = im.transform((im.width//3, im.height), Image.PERSPECTIVE, params, Image.BICUBIC)
im1.show()

#changing pixel values of an image
#choose 5000 random locations inside image
im1 = im.copy()
n = 5000
x, y = np.random.randint(0, im.width, n), np.random.randint(0, im.height, n)
for(x, y) in zip(x, y):
    im1.putpixel((x, y), ((0, 0, 0) if np.random.rand()< 0.5 else (255, 255, 255)))
im1.show()

#drawing on an image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.jpg")
draw = ImageDraw.Draw(im)
draw.ellipse((125, 125, 200, 250), fill = (255, 255, 255, 128))
del draw
im.show()

#adding text on an image
draw = ImageDraw.Draw(im)
font = ImageFont.truetype("arial.ttf", 25)
draw.text((10, 5),'welcome to image processing with python', font = font)
del draw
im.show()

#adding a thumbnal to an image
im = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot_copy.jpg")
im_thumbnail = im.copy()
size = (36, 36)
im_thumbnail.thumbnail((100,100), Image.ANTIALIAS)


#computing the basic statistics of an image
s = stat.Stat(im)
print(s.extrema)

#plotting the histogram of pixel valyes for the RGB channels of an image
pl = im.histogram()
plt.bar(range(256),pl[:256],color='r',alpha = 0.5)
plt.bar(range(256),pl[256:2*256],color='g',alpha = 0.4)
plt.bar(range(256),pl[2*256:],color='b',alpha = 0.3)
plt.show()

#separating the RGB channles of an image
ch_r, ch_g, ch_b = im.split()
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.imshow(ch_r, cmap=plt.cm.Reds)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(ch_g, cmap=plt.cm.Greens)
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(ch_b, cmap=plt.cm.Blues)
plt.axis('off')
plt.tight_layout()
plt.show()

#combining multiple channels of an image
im = Image.merge('RGB', (ch_b, ch_g, ch_r))
im.show()

#a - blending 2 images
im1 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot.png")
im2 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.png")
im1 = im1.convert('RGBA')
im2 = im2.resize((im1.width,im1.height),Image.BILINEAR)
im = Image.blend(im1, im2, alpha = 0.5).show()

#superimposing two image
im1 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot.png")
im2 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/hill.png").convert('RGB').resize((im1.width, im1.height))
multiply(im1, im2).show()

#adding two images
add(im1, im2).show()

#computing the difference between two images
im1 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/goal1.png")
im2 = Image.open("/Users/jonas.palacionis/Desktop/Image Processing/Images/goal2.png")
im = difference(im1, im2)
im.show()
im.save("/Users/jonas.palacionis/Desktop/Image Processing/Images/difference_goal.png")

plt.subplot(311)
plt.imshow(im1)
plt.axis('off')
plt.subplot(312)
plt.imshow(im2)
plt.axis('off')
plt.subplot(313)
plt.imshow(im), plt.axis('off')
plt.show()
im.show()

# applying swirl transform with scikit-image
im = imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot.png")
swirled = swirl(im, rotation = 0, strength = 15, radius = 200)
plt.imshow(swirled)
plt.axis('off')
plt.show()

#adding random gaussian noise to images
im = img_as_float(imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/parrot.png"))
plt.figure(figsize=(15,12))
sigmas = [0.1, 0.25, 0.5, 1]
for i in range (4):
    noisy = random_noise(im, var = sigmas[i]**2)
    plt.subplot(2,2,i+1)
    plt.imshow(noisy)
    
    
#drawing contour lines for an image using matplotlib
im = rgb2gray(imread("/Users/jonas.palacionis/Desktop/Image Processing/Images/einstein.jpg"))
plt.figure(figsize=(20,6))
plt.subplot(131), plt.imshow(im, cmap='gray'),plt.title('Original Image', size = 20)
plt.subplot(132), plt.contour(np.flipud(im), colors='k', levels=np.logspace(-15, 15, 100))
plt.title('Image Contour Lines', size = 20)
plt.subplot(133), plt.title('Image Filled Contour', size = 20)
plt.contourf(np.flipud(im), cmap = 'inferno')
plt.show()

