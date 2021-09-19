# suppress warnings
oldw <- getOption("warn")
options(warn = -1)

# environment
print("setup environment")

# one-function libraries
# library(stringi)     # used as stringi::stri_sub()

# libraries
library(ggplot2)
library(lubridate)
library(tidyverse)
library(caret)
library(foreach)                            # parallel computing

# keras environment
library(reticulate)                         # interface R and Python
use_condaenv("teste", required = TRUE)      # conda env for running tf and keras
library(keras)
library(tfdatasets)

# global variables
numberOfDigits <- 8
options(digits = numberOfDigits)
proportionTestSet <- 0.20
numberOfEpochs    <- 20                    # keras training parameter

# error function
errRMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# function: difference between timestamps in days
daysBetweenTimestamps <- function(x,y){
  difftime(as_date(as_datetime(x)), 
           as_date(as_datetime(y)), 
           units = c("days")) %>% 
    as.numeric()
}

# clean memory
if(exists("validation")) {rm(validation)}

# read dataset from csv
print("pre-process edx")
if(!exists("edx")) {edx <- read_csv(file = "./dat/edx.csv") %>% as_tibble()}
head(edx)

# move ratings to first column
edx2 <- edx %>% select(-c(rating))
edx2 <- cbind(rating = edx$rating, edx2)

# extract yearOfRelease and timestampYear
edx2 <- edx2 %>% 
  select(-c(genres)) %>%
  mutate(yearOfRelease = as.numeric(stringi::stri_sub(edx$title[1], -5, -2)),
         timestampYear = year(as_datetime(timestamp)),
         yearsFromRelease = timestampYear - yearOfRelease) %>%
  select(-c(title, yearOfRelease, timestampYear)) 

# extract firstUserRating
dfFirstUserRating <- edx2 %>% group_by(userId) %>%
  select(userId, timestamp) %>%
  summarize(firstUserRating = min(timestamp))

edx2 <- left_join(edx2, dfFirstUserRating)

# extract firstMovieRating
dfFirstMovieRating <- edx2 %>% group_by(movieId) %>%
  select(movieId, timestamp) %>%
  summarize(firstMovieRating = min(timestamp))

edx2 <- left_join(edx2, dfFirstMovieRating)

# extract daysFromFirstUserRating and daysFromFirstMovieRating
edx2 <- edx2 %>% mutate(daysFromFirstUserRating  = daysBetweenTimestamps(timestamp, firstUserRating),
                        daysFromFirstMovieRating = daysBetweenTimestamps(timestamp, firstMovieRating)) %>%
  select(-c(timestamp, firstUserRating,firstMovieRating))

# extract movie genres as predictors
genres_names <- strsplit(edx$genres, "|", fixed = TRUE) %>%
  unlist() %>%
  unique()

fn <- function(element_vector){
  as.numeric(grepl(element_vector, vector))
}

vector <- edx$genres
df <- sapply(genres_names, fn) %>% as_tibble()

# remove hiphen from predictor names
colnames(df)[7]  <- "SciFi"
colnames(df)[16] <- "FilmNoir"
colnames(df)[20] <- "NoGenre"

edx2 <- bind_cols(edx2, df)
head(edx2)

# cleanup memory
rm(df, edx, genres_names, vector, fn)

# split train and test sets
print("split edx in trainset/testset")
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx2$rating, 
                                  times = 1, 
                                  p = proportionTestSet,
                                  list = FALSE)
test_set <- edx2 %>% slice(test_index)
train_set <- edx2 %>% slice(-test_index)

# remove movies and users from testset that are not present on trainset
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# check for stratification of train / test split
p1 <- train_set %>%
  group_by(rating) %>%
  summarize(qty = n()) %>%
  mutate(split = 'train_set')

p2 <- test_set %>%
  group_by(rating) %>%
  summarize(qty = n()) %>%
  mutate(split = 'test_set')

p <- bind_rows(p1, p2) %>% group_by(split)
p %>% ggplot(aes(rating, qty, fill = split)) +
  geom_bar(stat="identity", position = "dodge") +
  ggtitle("Stratification of Test_set / Train_set split")

# save datasets
train_set %>% write_csv(file = "./dat/train.csv")
test_set  %>% write_csv(file = "./dat/test.csv")
head(train_set)
head(test_set)

# cleanup memory
rm(edx2, test_index)
rm(p, p1, p2)

# predict by global average
print("bias predictions")
mu <- mean(train_set$rating)
predicted <- mu
err <- RMSE(test_set$rating, predicted)
rmse_results <- tibble(model = "naiveAvg",
                       error = err)

# add movie bias effect
df <- train_set %>%
  group_by(movieId) %>%
  summarize(deltaRating = mean(rating - mu))

dfBiasMovie <- left_join(train_set, df) %>%
  select(rating, movieId, deltaRating) %>%
  group_by(movieId) %>%
  summarize(biasMovie = mean(deltaRating))
head(dfBiasMovie)

df <- left_join(test_set, dfBiasMovie) %>%
  select(rating, movieId, biasMovie)
predicted = mu + df$biasMovie

err <- RMSE(test_set$rating, predicted)
rmse_results <- bind_rows(rmse_results,
                          tibble(model ="movieBias",
                                 error = err))

# add user bias effect
df <- train_set %>%
  left_join(dfBiasMovie) %>%
  group_by(userId) %>%
  summarize(deltaRating = mean(rating - mu - biasMovie))

dfBiasUser <- left_join(train_set, df) %>%
  select(rating, userId, movieId, deltaRating) %>%
  group_by(userId) %>%
  summarize(biasUser = mean(deltaRating))
head(dfBiasUser)

df <- left_join(test_set, dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  select(rating, movieId, userId, biasMovie, biasUser)
predicted = mu + df$biasMovie + df$biasUser

err <- RMSE(test_set$rating, predicted)
rmse_results <- bind_rows(rmse_results,
                          tibble(model ="userBias",
                                 error = err))

# show RMSE results
rmse_results

# cleanup memory
rm(df, predicted)

# prepare trainset
print("pre-process trainset")
df_train <- train_set %>%
  select(rating:daysFromFirstMovieRating) %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(deltaRating = (rating - mu - biasMovie - biasUser),
         .before = rating) %>%
  select(-c(rating, userId, movieId, biasMovie, biasUser))

# remove predictors with small variance
nzv <- df_train %>%
  select(-deltaRating) %>%
  nearZeroVar(foreach = TRUE, allowParallel = TRUE)

removedPredictors <- colnames(df_train[,nzv])
df_train <- df_train %>% select(-all_of(removedPredictors))

# scale predictors
spec <- feature_spec(df_train, deltaRating ~ . ) %>% 
  step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>% 
  fit()
spec

# prepare testset
print("pre-process testset")
df_test <- test_set %>%
  select(rating:daysFromFirstMovieRating) %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(deltaRating = (rating - mu - biasMovie - biasUser),
         .before = rating) %>%
  select(-c(rating, userId, movieId, biasMovie, biasUser))

df_test <- df_test %>% select(-all_of(removedPredictors))

# wrap the model in a function
print("build keras model")
build_model <- function() {
  # create model
  input <- layer_input_from_dataset(df_train %>% select(-deltaRating))
  
  output <- input %>% 
    layer_dense_features(dense_features(spec)) %>% 
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1) 
  
  model <- keras_model(input, output)
  summary(model)
  
  # compile model
  model %>% 
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )
  
  model
}

# train the model
### remark:
### the whole block from below can be commented if a pre-trained model is used

print("train keras model")

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)

early_stop <- callback_early_stopping(monitor = "val_loss",
                                      min_delta = 1e-5,
                                      patience = 5)

model <- build_model()

history <- model %>% fit(
  x = df_train %>% select(-deltaRating),
  y = df_train$deltaRating,
  epochs = numberOfEpochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop, print_dot_callback)
)

plot(history)

# save model
# print("save model")
# model %>% save_model_tf("./model/", overwrite = FALSE)

# load pre-built model
# model2 <- load_model_tf("./model/")

# predict
print("predict testset results")
p <- model %>% predict(df_test %>% select(-deltaRating))
p <- p[ , 1]

df <- test_set %>%
  select(userId, movieId) %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(predicted = mu + biasMovie + biasUser + p)

# calculate error metrics
err <- errRMSE(test_set$rating, df$predicted)

rmse_results <- bind_rows(rmse_results,
                          tibble(model ="keras",
                                 error = err))

# show RMSE results
rmse_results

# clean memory
rm(df, df_train, df_test, p)

# validation
# read dataset from csv
print("predict validation results")
validation <- read_csv(file = "./dat/validation.csv") %>% as_tibble()
head(validation)

validation <- validation %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

df_val <- validation %>%
  select(-c(genres)) %>%
  mutate(yearOfRelease = as.numeric(stringi::stri_sub(validation$title[1], -5, -2)),
         timestampYear = year(as_datetime(timestamp)),
         yearsFromRelease = timestampYear - yearOfRelease) %>%
  select(-c(rating, title, yearOfRelease, timestampYear)) %>%
  left_join(dfFirstMovieRating) %>%
  left_join(dfFirstUserRating) %>%
  mutate(daysFromFirstUserRating  = daysBetweenTimestamps(timestamp, firstUserRating),
         daysFromFirstMovieRating = daysBetweenTimestamps(timestamp, firstMovieRating)) %>%
  select(-c(userId, movieId, timestamp, firstUserRating,firstMovieRating))
head(df_val)

p <- model %>% predict(df_val)
p <- p[ , 1]

df <- validation %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(predicted = mu + biasMovie + biasUser + p)

err <- errRMSE(validation$rating, df$predicted)

rmse_results <- bind_rows(rmse_results,
                          tibble(model ="validation",
                                 error = err))

# show RMSE results
print("project results")
rmse_results
err

# clean memory
rm(df, df_test, df_val)
rm(train_set, test_set, validation)

# restore warnings
print("job done")
options(warn = oldw)
