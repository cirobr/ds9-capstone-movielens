install.packages("reticulate")
reticulate::use_condaenv("teste", required = TRUE)      # conda env for running tf and keras
install.packages("keras-gpu")
library(keras)
install_keras(tensorflow = "gpu")
