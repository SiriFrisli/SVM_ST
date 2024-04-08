packages <- c("tidyverse", "tidymodels", "readxl", "recipes", "textrecipes", "tune", 
              "e1071", "tidytext", "caret", "ROSE", "spacyr", "textdata", "lubridate",
              "glmnet")

for (x in packages) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
  } else {
    library(x, character.only = TRUE)
  }
}

################################################################################
# Need to load and prepare the dataset first. 

covid_links <- readRDS("D:/Data/covid_misinfo_links_only.RDS") # only misinfo links
covid_all <- readRDS("D:/Data/covid_relevant_nort_domain.RDS") # all, no RT's

covid_all <- covid_all |>
  select(id, text, conversation_id, created_at)

covid_links <- covid_links |>
  select(-dom_url)

# I want to extract samples from the larger df to use in the training process
# For this I'll do a sample and remove and obs that is also present in the smaller df

# preparing a month/year column
covid_all$date <- as.POSIXct(covid_all$created_at, format = "%Y-%m-%dT%H:%M:%S")
covid_all$date <- format(as.Date(covid_all$date), "%Y-%m")

# Creating a new df with only the first instance of every unique conversation ID, 
# to make sure that I don't end up with a tweet in the middle of a conversation with no context
covid_all <- covid_all[with(covid_all, order(date)), ]
first_tweet <- covid_all[match(unique(covid_all$conversation_id), covid_all$conversation_id), ]

set.seed(1234)
covid_sample <- first_tweet |>
  sample_n(4400)

covid_sample |> # just making sure that every month is present within this data
  count(date)

# checking to see if there's any overlap in observations between the two df's covid_sample and covid_links
match <- covid_links |>
  semi_join(covid_sample, by = "id") # 69 matches needs to be removed

covid_sample <- covid_sample |>
  anti_join(covid_links, by = "id")

# preparing the covid_sample df by removing some columns and adding a label
covid_sample <- covid_sample |>
  select(-c(created_at, date)) |>
  cbind(label = 0)

covid <- rbind(covid_sample, covid_links)

# saving the df
# saveRDS(covid, "D:/Data/Training samples/covid_links_manual_labelled.RDS")

################################################################################
covid <- readRDS("D:/Data/Training samples/covid_links_manual_labelled.RDS")

stopwords <- read_xlsx("~/INORK/INORK_R/Processing/stopwords.xlsx")
custom_words <- stopwords |>
  pull(word)

covid <- covid |>
  select(tweet = text, label, id)

# Some preprocessing
removeURL <- function(tweet) {
  return(gsub("http\\S+", "", tweet))
}

removeUsernames <- function(tweet) {
  return(gsub("@[a-z,A-Z,_]*[0-9]*[a-z,A-Z,_]*[0-9]*", "", tweet))
}

covid$tweet <- apply(covid["tweet"], 1, removeURL)
covid$tweet <- apply(covid["tweet"], 1, removeUsernames)

covid$label <- as.factor(covid$label) # Outcome variable needs to be factor
covid$tweet <- tolower(covid$tweet)
covid$tweet <- gsub("[[:punct:]]", " ", covid$tweet)
# covid$tweet <- str_replace_all(covid$tweet, "[0-9]", "")

################################################################################
set.seed(1234)
covid_split <- initial_split(covid, prop = 0.8, strata = label)
train <- training(covid_split)
test <- testing(covid_split)

spacy_initialize(model = "nb_core_news_sm")
# glove27b <- embedding_glove27b(dimensions = 200, manual_download = TRUE) 

################################################################################
## TRAINING SVM MODEL

covid_recipe <- recipe(label~tweet, data = covid) |>
  step_tokenize(tweet, engine = "spacyr") |>
  # step_date(date, features = c("year", "month")) |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_tokenfilter(tweet, min_times = 2, max_tokens = 1000) |>
  step_tfidf(tweet) |>
  step_normalize(all_numeric_predictors()) |>
  prep()

new_train <- bake(covid_recipe, new_data = train)
new_test <- bake(covid_recipe, new_data = test)

################################################################################
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 probability = TRUE)

test_svm <- predict(svm_model, new_test)
test_svm <- bind_cols(new_test, test_svm)
test_svm |>
  accuracy(truth = label, estimate = ...1002) # 0.9

cm_svm_test <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm_test$byClass["F1"] # 0.904

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 55 FP, 118 FN

################################################################################
## Tuning time
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0,0.1, by = 0.01), 
                     cost = seq(0.1,1, by = 0.1), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.01
svm_tune$best.parameters$cost # 1

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0,0.01, by = 0.002), 
                       cost = c(1, 10, 100, 500, 1000), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 
svm_tune_2$best.parameters$cost # 

################################################################################
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.01,
                 cost = 1,
                 probability = TRUE)

test_svm <- predict(svm_model, new_test)
test_svm <- bind_cols(new_test, test_svm)
test_svm |>
  accuracy(truth = label, estimate = ...1002) # 0.605

cm_svm_test <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm_test$byClass["F1"] # 0.7136

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 16 FP, 667 FN
