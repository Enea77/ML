import tensorflow as tf
import numpy as np
from skimage import io,transform
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.filters import gaussian

def gaussian_blur(img,sigma):
  """Returns the Gaussian blurred version of the image 'img' with a sigma value of 'sigma'"""
  return np.array(gaussian(img,(sigma,sigma)))

def gaussian_blur_arr(images,sigma):
  """Applies the function gaussian_blur to all images in the set 'images'"""
  a = []
  for img in images:
    a.append(gaussian_blur(img,sigma))
  return np.array(a)

def norm_max_pixel(images):
  """Normalizes each image in the array 'images' such that the pixel intensities are within a range of 0 to 1"""
  a = []
  for img in images:
    img = img/np.max(img)
    a.append(img)
  return np.array(a)

def preprocess_images(images, size, sigma):
  """Preprocesses each image in the array 'images' applying the proper blurring, normalization, and shape to each image"""
  images = gaussian_blur_arr(images,sigma)
  images = norm_max_pixel(images)
  images = images.reshape((images.shape[0], size, size, 1))
  return images.astype('float32') #np.where(images > .5, 1.0, 0.0).astype('float32')

def get_predictions_from_model(model,images):
  """Returns an array of images 'prd' containing the prediction of each image in the input array 'images' from the CVAE model 'model'"""
  prd = []
  for i in range(len(images)):
    prd.append(model.predict(images[i:i+1])[0,:,:,0])
  return np.array(prd)

def get_difference(images,prd):
  """Returns an array of images containing the difference of each image in the input array 'images' with its respective prediction
  of the model in the input array 'prd'"""
  a = []
  for i in range(len(images)):
    d = images[i,:,:,0] - prd[i]
    #d = np.absolute(d)
    a.append(d)
  return np.array(a)

def get_heatmap(images):
  "set all pixels with an absolute value lower than MAX_PIXEL to 0"
  return np.where(abs(images) < MAX_PIXEL,0,images)

def get_avg_near_max(images):
  """Returns the average pixel intesity in an area of about a column size around the pixel with largest absolute value"""
  a = []
  for img in images:
    xmax, ymax = np.unravel_index(np.argmax(abs(img), axis=None), img.shape)
    s = 0
    c = 0
    for i in range(-CLMN_SIZE,CLMN_SIZE+1):
      for j in range(-CLMN_SIZE,CLMN_SIZE+1):
        try:
          s += img[xmax+i][ymax+j]
          c += 1
        except:
          pass
    a.append(s/c)
  return np.array(a)

def crop_images(images, size=64):
    """Reads a list of image files 'images' and crops it in smaller sections of size 'size' with overlaps"""
    arr = []
    half_size = int(size/2)
    for image in images:
      img = io.imread(image)
      width = len(img[0])
      height = len(img)
      print(width, height)
      try:
          img = img[:,:,0]
      except:
          pass
      for j in range(0,height,half_size):
        for i in range(0,width,half_size):
          if i+size <= width and j+size <= height:
            arr.append(img[j:j+size,i:i+size])
        if j+size <= height: 
          arr.append(img[j:j+size,-size:])
      for i in range(0,width,half_size):
        if i+size <= width: 
          arr.append(img[-size:,i:i+size])
      arr.append(img[-size:,-size:])           
    return np.array(arr)

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

SIZE = 64 #set the size in pixels of the training samples
SIGMA = 2 #Set the Gaussian Blurring sigma value
CLMN_SIZE = 6 #half the width of a column in pixels
MAX_PIXEL = 0.2 #Describes the largest absolute value of pixel intensity in a bulk difference image

epochs = 200 #Number of epochs on which to train the CVAE
latent_dim = 20 #Latent dimentions of the CVAE
image_file = "20111206DF" #Image file used to obtain the training sets
path = "/content/drive/Shareddrives/ML_Project/"#Path to train image file

#Load model
model = tf.keras.models.load_model(path+'model_'+image_file.strip(".jpg")+'_Size{}_SIGMA{}_epochs{}_latentdim{}'.format(SIZE,SIGMA,epochs,latent_dim),compile=False)

#Crop image file used in training to get testing set in sections of size 'SIZE'
files = [path+"20111206DF.jpg"]
testing_set = crop_images(files,SIZE)

#Crop image files that you want to check for anomalies in sections of size 'SIZE'
files = ["20111206DF.jpg"]
data_to_test = crop_images(files,SIZE)

#Preprocess the images
bulk_images = preprocess_images(testing_set,SIZE,SIGMA)
data_to_test_images = preprocess_images(data_to_test,SIZE,SIGMA)

#Get the model predictions
bulk_prd = get_predictions_from_model(model,bulk_images)
data_to_test_prd = get_predictions_from_model(model,data_to_test_images)

#Get the Average Near Maximum distribution of the bulk testing set
bulk_ANM = get_avg_near_max(get_heatmap(get_difference(bulk_images,bulk_prd)))

#Set the bounds to identify a bulk sample from the Average Near Maximum distribution of the bulk testing set
limit_coefficient = 1.5 
MIN_ANM = bulk_ANM.mean() - limit_coefficient*bulk_ANM.std()
MAX_ANM = bulk_ANM.mean() + limit_coefficient*bulk_ANM.std()
if bulk_ANM.std() == 0:
  MIN_ANM = bulk_ANM.mean() -0.1
  MAX_ANM = bulk_ANM.mean() +0.1

#Get the Average Near Maximum distribution of the data_to_test
data_to_test_ANM = get_avg_near_max(get_heatmap(get_difference(data_to_test_images,data_to_test_prd)))

#Lable each image in the data_to_test as an Anomaly (True) or Bulk (False)
data_to_test_labels = np.logical_and(data_to_test_ANM > MIN_ANM, data_to_test_ANM < MAX_ANM)