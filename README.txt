CVAE_training.py - Code for training the CVAE and saving the trained model. It requires a bulk sample, which is then divided into 
smaller sections to obtain a training set (to train the CVAE) and a testing set (to test the behavior of the CVAE on bulks samples
for future comparisons to distinguish bulk or anomalies). Parameters like SIZE, SIGMA, epochs, and latent_dim can be modified by the 
user. By default, they are set to the optimal values that we found after various tests.

CVAE_testing.py - Code we used to obtain the plots shown in the paper. It reads sets of various anomalies we created manually to 
show the efficiency of this method in finding point defects. 

analysis_example.py - Provides a sample code for the reader to apply our method. After running CVAE_training.py with your bulk image,
use analysis_exaple.py to look for anomalies in other image/images of the same crystal. It will provide a list of booleans labeling 
if an anomaly was found in a section of the image/images inputted