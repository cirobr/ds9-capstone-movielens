## edX Movielens Project

### Important: Installation instructions

This code has been tested on a specific Anaconda environment created on a Ubuntu 20.04 Linux Mate machine. All necessary scripts can be downloaded from the student's GitHub:

* https://github.com/cirobr/ds9-capstone-movielens.git

Next, the user should download and install Anaconda:

* https://www.anaconda.com/products/individual-d

Now, it is time to create an environment named "r-gpu" with the help of two command lines typed on terminal:

* conda create --name r-gpu python=3.9 notebook r-base=4.1 r-essentials r-e1071 r-irkernel r-varhandle r-foreach r-doparallel r-reticulate r-keras r-tfdatasets

* Rscript install-keras-gpu.R

Lastly, the below script executed on RStudio runs the code provided by edX to generate the datasets, and stores the files "edx.csv" and "validation.csv" on a sub-folder "./dat", which is also created in the process:

* code-preset.R

At this point, the code is ready for execution on RStudio, through this script:

* code-movielens.R
