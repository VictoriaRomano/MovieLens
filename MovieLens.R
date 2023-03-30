##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem",repos = "http://cran.us.r-project.org" )

library(tidyverse)
library(caret)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
# remove from memory unnecessary information
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Check the number of columns and rows in edx set
dim(edx)

#Have a look of first rows of edx set and names of columns
head(edx)

#Check the number of different movies
n_distinct(edx$movieId)

# Check the number of different users
n_distinct(edx$userId)

# Check if we have some NA
sum(is.na(edx))

# Check rating distribution
edx %>%
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = rating, y = count)) +
  geom_bar(stat = "identity") +
  xlab("Rating") + ggtitle("Distribution of ratings") 

#Check if some movies were rated more often than others
# to br we assign a vector of minor break values
br <-unique(as.numeric(1:10 %o% 10 ^ (0:3)))
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 40, color = "blue") + 
  scale_x_log10(breaks = scales::breaks_log(10),minor_breaks = br) + 
  ggtitle("Number of ratings per movie") +
  ylab("Number of movies")+
  xlab("Number or ratings per movie")

#Check if some users rated movies more often than others and how numbers of ratings
#distributed between users
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 40, color = "red") + 
  scale_x_log10(breaks = scales::breaks_log(10),minor_breaks = br ) + 
  ggtitle("Number of users with certain number of ratings")+
  ylab("Number of users")+
  xlab("Number of ratings per user")


#Create the test set and train set from edx set
test_index_1 <- createDataPartition(y = edx$rating, times = 1,
                                    p = 0.2, list = FALSE)
train_set <- edx[-test_index_1,]
temp_test_set <- edx[test_index_1,]

#Make sure we do not include movies and users in the test set that do not appear in the train set
test_set <- temp_test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from the testing set back into the training set set
removed <- anti_join(temp_test_set, test_set)
train_set <- rbind(train_set, removed)

#Function for calculating RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Compute the average rating for all users across all movies
mu <- mean(train_set$rating)
mu
#Compute  RMSE if we assign all predicted ratings to the average rating
naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse

#Make tibble with RMSE results
model_result <- tibble(Model = "Just the average",
                       RMSE = naive_rmse)
model_result

#Check if there are some relationship between number of movie ratings and average rating of the movie
num_rate <-test_set %>% 
  group_by(movieId) %>%
  summarize(n = n(),
            rating = mean(rating))
head(num_rate)
num_rate %>% ggplot(aes(n, rating)) + geom_point() + geom_hline(yintercept = 3.51, col = "red")+
  ggtitle("Relationship between how often movie was rated and average rating")

# Check if there are some relationship between how many movies user rated and
#average rating this user gives to movies
user_rate <- test_set%>% 
  group_by(userId) %>%
  summarize(n = n(),
            rating = mean(rating))
head(user_rate)
user_rate %>% ggplot(aes(n, rating)) + geom_point() + geom_hline(yintercept = 3.51, col = "blue")+
  ggtitle("Relationship between how many movies user rated and his average rating")

# Movie effect model 

#calculate b_i (a difference between average movie rating and average rating across all movies) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
# Calculate predicted rating by adding movie effect b_i to average rating across all movies
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred
head(predicted_ratings) # just for checking that we got what we want

# calculate movie model RMSE
model_movie_rmse <- RMSE(predicted_ratings, test_set$rating)
model_movie_rmse

# Add movie model RMSE to the table
model_result <- bind_rows(model_result, tibble(Model = "Movie effect",
                                               RMSE = model_movie_rmse))
model_result

# check what result it will give, if we calculate b-i(movie effect) in the case when movie has
# more than m ratings, and if less, we assign the predicted rating to average rating across all movies 

m <- seq(0, 10, 1)

rmses <- sapply(m, function(m){
  movie_avgs_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = ifelse(n()>m,mean(rating - mu),0))
  
  predicted_ratings <- test_set %>% left_join(movie_avgs_reg, by='movieId') %>%
    mutate(pred = mu + b_i) %>%
    .$pred  
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(m, rmses)  
m[which.min(rmses)]

# Movie-user model

# calculate b_u(the difference between average rating the user gives and average rating across all movies
# and b_i)
movie_user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
# Calculate predicted ratings, using b_u
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(movie_user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
# Calculate RMSE for movie-user model
model_movie_user_rmse <- RMSE(predicted_ratings, test_set$rating)
model_movie_user_rmse

#Add RMSE results of movie_user model to results table
model_result <- bind_rows(model_result, tibble(Model = "Movie-user effect",
                                               RMSE = model_movie_user_rmse))
model_result

#check, if we will ignore users who rated less than z movies
z<- seq(0,10,1)
rmses <- sapply(z, function(z){
  movie_user_avgs_reg <- train_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = ifelse(n() > z, mean(rating - mu - b_i),0))
  predicted_ratings <- test_set %>% 
    left_join(movie_avgs, by='movieId') %>%
    left_join(movie_user_avgs_reg, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(z, rmses)  
z[which.min(rmses)]


#Recommender system using matrix factorization

set.seed(5, sample.kind = "Rounding")
# Return data format for train_set
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId,
                                          rating = rating))

#Return data format for test set
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, 
                                        rating = rating))

#Return an object of class "RecoSys"
recommendation_system <- Reco()

#Train the model with default parameters
recommendation_system$train(train_reco)

#Calculate predicted ratings with default model
predicted_ratings <-  recommendation_system$predict(test_reco, out_memory())

#Calculate RMSE of default matrix factorization model
reco_model <- RMSE(predicted_ratings, test_set$rating)
reco_model

#Add RMSE results of reco_model to the results table
model_result <- bind_rows(model_result, tibble(Model = "Matrix fact",
                                               RMSE = reco_model))
model_result

# Tune the model parameters, using cross validation - takes time(around 20 min)
set.seed(5, sample.kind = "Rounding")
tuning <- recommendation_system$tune(train_reco, opts = list(dim = c(20L, 30L),
                                                             lrate = c(0.05, 0.1),
                                                             nthread  = 6,
                                                             niter = 10))
# Check which tuning parameters work the best                                                                   
tuning$min

#Train the model using tuned parameters
recommendation_system$train(train_reco, opts = c(tuning$min,
                                                 nthread = 4,
                                                 niter = 25))
# Calculate the predicted ratings using tuned model
predicted_ratings <-  recommendation_system$predict(test_reco, out_memory())

#Calculate RMSE of tuned model
reco_model_tuned <- RMSE(predicted_ratings, test_set$rating)
reco_model_tuned

#Add results of tuned recosystem model to the results table
model_result <- bind_rows(model_result, tibble(Model = "Matrix fact tuned",
                                               RMSE = reco_model_tuned))
model_result

## Check the results on Final holdout test set using tuned recosystem model##

#Prepare data format for the recosystem model
final_test_reco <- with(final_holdout_test,data_memory(user_index = userId, 
                                                       item_index = movieId, 
                                                       rating= rating))
# Compute predicted ratings
final_predicted_ratings <- recommendation_system$predict(final_test_reco, out_memory())

# Compute RMSE
reco_model_final_rmse <- RMSE(final_predicted_ratings, final_holdout_test$rating)
reco_model_final_rmse







