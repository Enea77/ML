import tensorflow as tf
import numpy as np
from skimage import io,transform
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.filters import gaussian
import time
from IPython import display

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""
 
  def __init__(self, latent_dim):
    reduced_size = int(SIZE/8)
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(SIZE, SIZE, 1)),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )
 
    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=reduced_size*reduced_size*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(reduced_size, reduced_size, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )
 
  @tf.function
  def sample(self, eps=None, apply_sigmoid=True):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid)
 
  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar
 
  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=(latent_dim,))
    #eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean
 
  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def call(self,image):
    #shape = image.shape
    #shape = tf.convert_to_tensor(shape)
    #image = tf.reshape(image, shape)
    mean, logvar = self.encode(image)
    z = self.reparameterize(mean, logvar)
    predictions = self.sample(z)
    return predictions

optimizer = tf.keras.optimizers.Adam(1e-4)
 
 
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)
 
 
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)
 
 
@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.
 
  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def infrance(model,image):
  mean, logvar = model.encode(image)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  return predictions

def generate_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))
 
  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
  plt.show()
 
  fig = plt.figure(figsize=(4, 4))
 
  for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, 0], cmap='gray')
    plt.axis('off')
  plt.show()

def predict(model, inp_image, apply_sigmoid=True):
  mean, logvar = model.encode(inp_image)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z,apply_sigmoid)
  return predictions[0,:,:,0]

def get_testing_training_sets(image='20111206DF2', size=64,n_images=10):
    """Reads an image file 'image' and crops it in 2*n_images samaller sections of size 'size'. Sections from the 
    top half are added in the training set while those from the bottom half are added to the testing set. 
    Trainign and testing sets are then returned as numpy arrays"""
    training = []
    testing = []
    img = io.imread(image)
    width = len(img[0])
    height = len(img)
    try:
        img = img[:,:,0]
    except:
        pass
    for i in range(n_images):
        shift_y = np.random.randint(height/2,height-size)
        shift_x = np.random.randint(0,width-size)
        training.append(img[shift_y:shift_y+size,shift_x:shift_x+size])
        shift_y = np.random.randint(0,height/2-size)
        shift_x = np.random.randint(0,width-size)
        testing.append(img[shift_y:shift_y+size,shift_x:shift_x+size])
    return np.array(training), np.array(testing)

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
  images = gaussian_blur_arr(images,sigma)
  images = norm_max_pixel(images)
  images = images.reshape((images.shape[0], size, size, 1))
  return images.astype('float32') #np.where(images > .5, 1.0, 0.0).astype('float32')

####################################################################################
####################################################################################
####################################################################################
####################################################################################

SIZE = 64 #set the size in pixels of the training samples
SIGMA = 2 #Set the Gaussian Blurring sigma value
epochs = 200 #Number of epochs on which to train the CVAE
latent_dim = 20 #Latent dimentions of the CVAE
image_file = "20111206DF2" #Image file used to obtain the training sets
path = "/"#Path to image file
training_set, testing_set = get_testing_training_sets(path+image_file,SIZE,100) 


#Uncomment the lines below to use the provided training and testing sets
'''
image_file = "20111206DF2"
mix_training_set = np.load("/content/drive/Shareddrives/ML_Project/raw_training_set_"+image_file+"_64.npy")
np.random.shuffle(mix_training_set)
mix_training_set.shape

bulk_test = np.load("/content/drive/Shareddrives/ML_Project/raw_testing_set_"+image_file+"_64.npy")[:100]
bulk_test.shape
'''

train_images = preprocess_images(training_set,SIZE,SIGMA)
test_images = preprocess_images(testing_set,SIZE,SIGMA)

train_size = 16
batch_size = 16
test_size = 5

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

num_examples_to_generate = 16
 
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

#Call the CVAE and set the desired size of the samples
model = CVAE(latent_dim)
model.compute_output_shape(input_shape=(1,SIZE, SIZE,1))

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_images(model, 0, test_sample) #rerun this cell after 50 epochs steps
 
for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()
 
  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_images(model, epoch, test_sample)

model.save(path+image_file.strip(".jpg")+'_Size{}_SIGMA{}_epochs{}_latentdim{}'.format(SIZE,SIGMA,epochs,latent_dim))