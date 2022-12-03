<h1> Convolutional Neural Networks for Anomaly Detection in Scanning Transmission Electron Microscopy  </h1>
<h4> Enea Prifti, James P. Buban, Arashdeep Thind, Robert F Klie </h4>
<h4> University of Illinois Chicago, Department of Physics, 845 W Taylor Street, Chicago, IL 60607 </h4>

The codes in this repository are used in the paper mentioned above to detect anomalies in the pattern of a crystal lattice <br />
using a Convolutional Variational Autoencoder trained to reproduce that pattern. <br />

`CVAE_training.py` - Code for training the CVAE and saving the trained model. It requires a bulk sample, which is then divided into <br />
smaller sections to obtain a training set (to train the CVAE) and a testing set (to test the behavior of the CVAE on bulks samples <br />
for future comparisons to distinguish bulk or anomalies). Parameters like `SIZE`, `SIGMA`, `epochs`, and `latent_dim` can be modified by the <br />
user. By default, they are set to the optimal values that we found after various tests. <br />
<br />
`CVAE_testing.py` - Code we used to obtain the plots shown in the paper. It reads sets of various anomalies we created manually to <br />
show the efficiency of this method in finding point defects. <br />

`analysis_example.py` - Provides a sample code for the reader to apply our method. After running CVAE_training.py with your bulk image, <br />
use analysis_exaple.py to look for anomalies in other image/images of the same crystal. It will provide a list of booleans labeling  <br />
if an anomaly was found in a section of the image/images inputted <br />
