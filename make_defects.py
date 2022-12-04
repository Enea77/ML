#from PIL import Image
import numpy as np
#from skimage import io,transform
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.filters import gaussian

def plot_images(images,columns=4,rows=4,cmap='jet',size=(16,16),Labels=[]):
  fig, axes = plt.subplots(nrows=rows, ncols=columns,figsize=size)
  if Labels == []:
    Labels = np.arange(len(images))

  # find minimum of minima & maximum of maxima
  minmin = np.min(images)
  maxmax = np.max(images)

  for i in range(rows):
    for j in range(columns):
      im1 = axes[i][j].imshow(images[i*columns+j], vmin=minmin, vmax=maxmax,cmap=cmap)
      axes[i][j].set_title(Labels[i*columns+j])


  # add space for colour bar
  fig.subplots_adjust(right=0.85)
  cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
  fig.colorbar(im1, cax=cbar_ax)

def make_antiSr_defect(images,defect_size=7):
    """Replaces a Sr column with a Ti column in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        img_cut = img[defect_size:-defect_size,defect_size:-defect_size]
        xmax, ymax = np.unravel_index(np.argmax(gaussian(img_cut,2), axis=None), img_cut.shape)
        for i in range(xmax,xmax+2*defect_size+1):
            for j in range(ymax,ymax+2*defect_size+1):
                if i in range(0,64) and j in range(0,64):
                    img[i][j] = Ti_SPOT[i-xmax][j-ymax]
        a.append(img)
    return np.array(a)

def make_antiTi_defect(images,defect_size=7):
    """replaces a Ti column with a Sr column in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        xmax, ymax = np.unravel_index(np.argmax(gaussian(img,2), axis=None), img.shape)
        if xmax < 3*defect_size: 
            shift_x = xmax+defect_size
        else:
            shift_x = xmax-3*defect_size
        if ymax < 3*defect_size:
            shift_y = ymax+defect_size
        else:
            shift_y = ymax-3*defect_size
        for i in range(shift_x,shift_x+2*defect_size+1):
            for j in range(shift_y,shift_y+2*defect_size+1):
                try:
                    img[i][j] = Sr_SPOT[i-shift_x][j-shift_y]
                except:
                    pass
        a.append(img)
    return np.array(a)

def make_TiVacancy_defect(images,defect_size=7):
    """replaces a Ti column with an empty column in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        xmax, ymax = np.unravel_index(np.argmax(img, axis=None), img.shape)
        if xmax < 3*defect_size: 
            shift_x = xmax+defect_size
        else:
            shift_x = xmax-3*defect_size
        if ymax < 3*defect_size:
            shift_y = ymax+defect_size
        else:
            shift_y = ymax-3*defect_size
        for i in range(shift_x,shift_x+2*defect_size+1):
            for j in range(shift_y,shift_y+2*defect_size+1):
                try:
                    img[i][j] = DARK_SPOT[i-shift_x][j-shift_y]
                except:
                    pass
        a.append(img)
    return np.array(a)

def make_TiFMag_defect(images,defect_size=7):
    """Shifts a Ti column by 'defect_size' pixels in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        xmax, ymax = np.unravel_index(np.argmax(img, axis=None), img.shape)
        if xmax < 3*defect_size: 
            shift_x = xmax+defect_size
        else:
            shift_x = xmax-3*defect_size
        if ymax < 3*defect_size:
            shift_y = ymax+defect_size
        else:
            shift_y = ymax-3*defect_size
        for i in range(shift_x,shift_x+2*defect_size+1):
            for j in range(shift_y,shift_y+2*defect_size+1):
                try:
                    img[i][j] = DARK_SPOT[i-shift_x][j-shift_y]
                except:
                    pass
        for i in range(shift_x,shift_x+2*defect_size+1):
            for j in range(shift_y+defect_size,shift_y+3*defect_size+1):
                try:
                    img[i][j] = Ti_SPOT[i-shift_x][j-shift_y-defect_size]
                except:
                    pass    
        a.append(img)
    return np.array(a)

def make_SrFMag_defect(images,defect_size=7):
    """Shifts a Sr column by 'defect_size' pixels in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        img_cut = img[defect_size:-defect_size,defect_size:-defect_size]
        xmax, ymax = np.unravel_index(np.argmax(gaussian(img_cut,2), axis=None), img_cut.shape)
        for i in range(xmax,xmax+2*defect_size+1):
            for j in range(ymax,ymax+2*defect_size+1):
                if i in range(0,64) and j in range(0,64):
                    img[i][j] = DARK_SPOT[i-xmax][j-ymax]
        for i in range(xmax,xmax+2*defect_size+1):
            for j in range(ymax+defect_size,ymax+3*defect_size+1):
                if i in range(0,64) and j in range(0,64):
                    img[i][j] = Sr_SPOT[i-xmax][j-ymax-defect_size]
        a.append(img)
    return np.array(a)

def make_SrVacancy_defect(images,defect_size=7):
    """replaces a Sr column with an empty column in each image of the
    input array 'images'. defect_size representes half the width in pixels 
    of a column in the inout images"""
    a = []
    for image in images:
        img = np.copy(image)
        img_cut = img[defect_size:-defect_size,defect_size:-defect_size]
        xmax, ymax = np.unravel_index(np.argmax(gaussian(img_cut,2), axis=None), img_cut.shape)
        for i in range(xmax,xmax+2*defect_size+1):
            for j in range(ymax,ymax+2*defect_size+1):
                if i in range(0,64) and j in range(0,64):
                    img[i][j] = DARK_SPOT[i-xmax][j-ymax]
        a.append(img)
    return np.array(a)

#get bulk images to modify
exp_clean = np.load("raw_testing_set_20111206DF2_64.npy")
print("bulk set shape: ",exp_clean.shape) #check that the size is as desired

#get point defects to use
Ti_SPOT = np.load("raw_Ti.npy")
#plt.imshow(Ti_SPOT)
#plt.title("Ti column")
Sr_SPOT = np.load("raw_Sr.npy")
#plt.imshow(Sr_SPOT)
#plt.title("Sr column")
DARK_SPOT = np.load("raw_Vacancy.npy")
#plt.imshow(DARK_SPOT)
#plt.title("Vacancy column")

#make defect images from bulk images array
antiSr_defect = make_antiSr_defect(exp_clean)
antiTi_defect = make_antiTi_defect(exp_clean)
TiVacancy_defect = make_TiVacancy_defect(exp_clean,)
SrVacancy_defect = make_SrVacancy_defect(exp_clean)
SrFMag_defect = make_SrFMag_defect(exp_clean)
TiFMag_defect = make_TiFMag_defect(exp_clean,7)

#save the np arrays with the created defect images
np.save("raw_TiFMag_defect_64.npy",TiFMag_defect)
np.save("raw_SrFMag_defect_64.npy",SrFMag_defect)
np.save("raw_SrVacancy_defect_64.npy",SrVacancy_defect)
np.save("raw_TiVacancy_defect_64.npy",TiVacancy_defect)
np.save("raw_antiSr_defect_64.npy",antiSr_defect)
np.save("raw_antiTi_defect_64.npy",antiTi_defect)

#show some of the results
images_to_plot = np.concatenate((exp_clean[:4],SrVacancy_defect[:4],antiSr_defect[:4],SrFMag_defect[:4]))
plot_images(images_to_plot,4,4,size=(12,12))
plt.show()

images_to_plot = np.concatenate((exp_clean[:4],TiVacancy_defect[:4],antiTi_defect[:4],TiFMag_defect[:4]))
plot_images(images_to_plot,4,4,size=(12,12))
plt.show()

