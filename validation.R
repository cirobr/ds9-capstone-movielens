# final step: validation
# read dataset from csv
print("predict validation results")
validation <- read_csv(file = "./dat/validation.csv") %>% as_tibble()
head(validation)

# prepare validation dataset
df_val <- validation %>%
  select(-c(rating))
df_val <- cbind(rating = validation$rating, df_val)

df_val <- df_val %>%
  select(-c(genres)) %>%
  mutate(yearOfRelease = as.numeric(stringi::stri_sub(validation$title[1], -5, -2)),
         timestampYear = year(as_datetime(timestamp)),
         yearsFromRelease = timestampYear - yearOfRelease) %>%
  select(-c(title, yearOfRelease, timestampYear)) %>%
  left_join(dfFirstUserRating) %>%
  left_join(dfFirstMovieRating) %>%
  mutate(daysFromFirstUserRating  = daysBetweenTimestamps(timestamp, firstUserRating),
         daysFromFirstMovieRating = daysBetweenTimestamps(timestamp, firstMovieRating)) %>%
  select(-c(timestamp, firstUserRating,firstMovieRating))

genresVector <- validation$genres
df <- sapply(genresNames, extractGenresNames) %>% as_tibble()
colnames(df)[7]  <- "SciFi"
colnames(df)[16] <- "FilmNoir"
colnames(df)[20] <- "NoGenre"

df_val <- bind_cols(df_val, df)
df_val <- df_val %>% 
  select(-all_of(removedPredictors)) %>%
  as_tibble()

df_val <- df_val %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(deltaRating = (rating - mu - biasMovie - biasUser),
         .before = rating) %>%
  select(-c(rating, userId, movieId, biasMovie, biasUser))

# predict
neuralNetPrediction <- model %>% predict(df_val %>% select(-deltaRating))
neuralNetPrediction <- neuralNetPrediction[ , 1]

df <- validation %>%
  select(userId, movieId) %>%
  left_join(dfBiasMovie) %>%
  left_join(dfBiasUser) %>%
  mutate(predicted = mu + biasMovie + biasUser + neuralNetPrediction)
validRows <- !is.na(df$predicted)

# calculate error metrics
err <- errRMSE(validation$rating[validRows], df$predicted[validRows])

rmse_results <- bind_rows(rmse_results,
                          tibble(model ="fullModel validation",
                                 error = err))
rmse_results

# clean memory
rm(df, df_val, validation)
