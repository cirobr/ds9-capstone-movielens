if(!require(reticulate)) install.packages("reticulate", repos = "http://cran.us.r-project.org")
reticulate::use_condaenv("r-gpu", required = TRUE)      # conda env for running tf and keras

if(!require(keras)) install.packages("keras-gpu", repos = "http://cran.us.r-project.org")
keras::install_keras(tensorflow = "gpu")
