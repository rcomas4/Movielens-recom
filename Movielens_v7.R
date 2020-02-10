################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# CODE MADE BY ROGER COMAS (FEBRUARY 2020)
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# Load libraries
library(tidyverse)
library(caret)
library(stringr)
library(lubridate)
library(Matrix)

# Create RMSE function
RMSE <- function(real_ratings, predicted_ratings){
  sqrt(mean((real_ratings - predicted_ratings)^2))
}

### Data analysis ###

head(edx)

# Create a column with the year of the movie
pattern <- "\\(([^()]+)\\)$"
edx <- edx %>% mutate(year = str_extract(title,pattern))
edx$year <- substring(edx$year, 2, nchar(edx$year)-1)
validation <- validation %>% mutate(year = str_extract(title,pattern))
validation$year <- substring(validation$year, 2, nchar(validation$year)-1)

# Summary of the dataset
summary(edx)

# Number of uniques users and movies
edx %>%
  summarize(num_users = n_distinct(userId), 
            num_movies = n_distinct(movieId))

# Plot of the number of ratings per movie
edx %>% group_by(movieId) %>% 
  summarize(n_ratings = n()) %>% 
  ggplot(aes(n_ratings)) + 
  geom_histogram(binwidth = 50 , color = "black") + 
  xlab("Ratings") +
  ylab("Number of movies") +
  ggtitle("Histogram of the ratings per movies")

# Plot mean rating per movie
edx %>% group_by(movieId) %>% 
  summarize(mean = mean(rating)) %>% 
  ggplot(aes(mean)) + 
  geom_histogram(bins = 30 , color = "black") + 
  xlab("Rating") +
  ylab("Number of movies") +
  ggtitle("Mean of rating per movie")

# top_10title is a data frame which contains the TOP 10 titles with more ratings.
top_10title <- edx %>%
  group_by(title) %>%
  summarize(count=n()) %>%
  top_n(10,count) %>%
  arrange(desc(count))

# Plot mean movie rating per user
edx %>%
  group_by(userId) %>%
  summarize(mean = mean(rating)) %>%
  ggplot(aes(mean)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings per user") 

# Plot timestamp and the mean rating
edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Timestamp, time unit : week")

# Mean movie ratings per year
edx %>%
  group_by(year) %>%
  summarize(mean = mean(rating)) %>%
  ggplot(aes(year, mean, fill = year)) + 
  geom_point()+
  ggtitle("Mean rating of the movies per year") +
  xlab("Year") +
  ylab("Mean rating") +
  scale_x_discrete(breaks = c(seq(1915,2008, 10)))


# Mean rating per genre
edx %>%
  group_by(genres) %>%
  filter(n()>100000)%>%
  summarize( mean = mean(rating)) %>%
  arrange(mean) %>%
  ggplot(aes(genres, mean, fill = genres)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10), legend.position = "none")

#------------------------------------------------------------------------------------------------------------
# Modelling
#-------------------------------------------------
-----------------------------------------------------------
### Modelling ###

# First simple model
mu <- edx %>% 
  summarize(mu = mean(rating))

rmse_mu <- RMSE(validation$rating, mu[[1]])
rmse_mu

rmse_results <- data.frame(model = "simple",
                           RMSE = rmse_mu)
rmse_results

#Add first bias
# Movie effect

b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_i) %>%
  mutate(pred = mu[[1]] + b_i)

rmse_movie <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie",
                                     RMSE = rmse_movie))
rmse_results

#User effects
b_u <- edx %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_u) %>%
  mutate(pred = mu[[1]] + b_u)

rmse_user <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "user",
                                     RMSE = rmse_user))
rmse_results

# Year effects
b_y <- edx %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - mu[[1]]))

predicted_ratings <- validation %>% 
  left_join(b_y) %>%
  mutate(pred = mu[[1]] + b_y)

rmse_year <- RMSE(validation$rating, predicted_ratings$pred)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "year",
                                     RMSE = rmse_year))
rmse_results

# Genre effects
b_g <- edx %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_g) %>%
  mutate(pred = mu[[1]] + b_g)

rmse_genre <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "genre",
                                     RMSE = rmse_genre))
rmse_results

#Date effects

b_d <- edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - mu[[1]]))

predicted_ratings <- validation %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_d) %>%
  mutate(pred = mu[[1]] + b_d)

rmse_date <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "date",
                                     RMSE = rmse_date))
rmse_results

#The best model is with the movie bias

#Add second bias

# Movie + user effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  mutate(pred = mu [[1]] + b_i + b_u)

rmse_iu <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user",
                                     RMSE = rmse_iu))
rmse_results

# Movie + year effect
b_y <- edx %>%
  left_join(b_i) %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - b_i - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_y) %>%
  mutate(pred = mu[[1]] + b_i + b_y)

rmse_yu <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+year",
                                     RMSE = rmse_yu))
rmse_results

# Movie + genre effect
b_g <- edx %>%
  left_join(b_i) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_g) %>%
  mutate(pred = mu [[1]] + b_g + b_i)

rmse_gu <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+genre",
                                     RMSE = rmse_gu))
rmse_results

# Movie + date effect
b_d <- edx %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - mu[[1]]))

predicted_ratings <- validation %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_d) %>%
  mutate(pred = mu[[1]] + b_i + b_d)

rmse_du <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+date",
                                     RMSE = rmse_du))
rmse_results

#The best model is the movie+user biases
#Add a third bias

# Movie + user + year effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_y <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - b_i - b_u - mu[[1]]))
  
  
predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_y) %>%
  mutate(pred = mu [[1]] + b_i + b_u + b_y)

rmse_iuy <- RMSE(validation$rating, predicted_ratings$pred)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+year",
                                     RMSE = rmse_iuy))
rmse_results

#Movie + user + genre effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_g <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - b_u - mu[[1]]))


predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  mutate(pred = mu [[1]] + b_i + b_u + b_g)

rmse_iug <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+genre",
                                     RMSE = rmse_iug))
rmse_results

# Movie + user + date effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_d <- edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - b_i - b_u - mu[[1]]))


predicted_ratings <- validation %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_d) %>%
  mutate(pred = mu [[1]] + b_i + b_u + b_d)

rmse_iud <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+date",
                                     RMSE = rmse_iud))
rmse_results

#The best model is the movie+user+genre biases
#Add a 4th bias

# Movie + user + genre + year effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_g <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - b_u - mu[[1]]))

b_y <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - b_i - b_u - b_g - mu[[1]]))

predicted_ratings <- validation %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  left_join(b_y) %>%
  mutate(pred = mu [[1]] + b_i + b_u + b_g + b_y)

rmse_iugy <- RMSE(validation$rating, predicted_ratings$pred)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+genre+year",
                                     RMSE = rmse_iugy))
rmse_results

# Movie + user + genre + date effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_g <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - b_u - mu[[1]]))

b_d <- edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - b_i - b_u - b_g  - mu[[1]]))

predicted_ratings <- validation %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  left_join(b_d) %>%
  mutate(pred = mu [[1]] + b_i + b_u + b_g +b_d)

rmse_iugd <- RMSE(validation$rating, predicted_ratings$pred)

rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+genre+date",
                                     RMSE = rmse_iugd))
rmse_results

#The best model is the movie+user+genre+date effect
#Add the 5th effect

# Movie + user + genre + year + date effect
b_u <- edx %>%
  left_join(b_i) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu[[1]]))

b_g <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - b_i - b_u - mu[[1]]))

b_y <- edx %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  group_by(year) %>%
  summarize(b_y = mean(rating - b_i - b_u - b_g  - mu[[1]]))

b_d <- edx %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  left_join(b_y) %>%
  group_by(date) %>%
  summarize(b_d = mean(rating - b_i - b_u - b_g - b_y - mu[[1]]))


predicted_ratings <- validation %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(b_i) %>%
  left_join(b_u) %>%
  left_join(b_g) %>%
  left_join(b_y) %>%
  left_join(b_d) %>% 
  mutate(pred = mu [[1]] + b_i + b_u + b_g + b_y + b_d)

rmse_iugyd <- RMSE(validation$rating, predicted_ratings$pred)
rmse_results <- bind_rows(rmse_results,
                          data.frame(model= "movie+user+genre+year+date",
                                     RMSE = rmse_iugyd))
rmse_results

#Show the best model of all created

best_model <- rmse_results[which.min(rmse_results$RMSE),]
best_model


#Regularization of the best model
lambdas <- seq(4, 5, 0.05)
rmses <- sapply(lambdas, function(l){
  
  b_u <- edx %>%
    left_join(b_i) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu[[1]])/(n()+l))
  
  b_g <- edx %>%
    left_join(b_i) %>%
    left_join(b_u) %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu[[1]])/(n()+l))
  
  b_y <- edx %>%
    left_join(b_i) %>%
    left_join(b_u) %>%
    left_join(b_g) %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_g  - mu[[1]])/(n()+l))
  
  b_d <- edx %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    left_join(b_i) %>%
    left_join(b_u) %>%
    left_join(b_g) %>%
    left_join(b_y) %>%
    group_by(date) %>%
    summarize(b_d = sum(rating - b_i - b_u - b_g - b_y - mu[[1]])/(n()+l))
  
  
  predicted_ratings <- validation %>%
    mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
    left_join(b_i) %>%
    left_join(b_u) %>%
    left_join(b_g) %>%
    left_join(b_y) %>%
    left_join(b_d) %>% 
    mutate(pred = mu [[1]] + b_i + b_u + b_g + b_y + b_d)
  
  predicted_ratings$pred <- ifelse(predicted_ratings$pred > 5, 5, predicted_ratings$pred)
  predicted_ratings$pred <- ifelse(predicted_ratings$pred < 0, 0 , predicted_ratings$pred)
  
  return(RMSE(validation$rating, predicted_ratings$pred))
})


# Plot rmses vs lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses)  


# The optimal lambda                                                             
lambda <- lambdas[which.min(rmses)]
lambda

rmse_regularized <- min(rmses)
rmse_regularized
