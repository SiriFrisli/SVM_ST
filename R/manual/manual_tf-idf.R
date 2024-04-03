packages <- c("tidyverse", "tidymodels", "readxl", "recipes", "textrecipes", 
              "e1071", "tidytext", "caret", "ROSE", "spacyr", "textdata", "lubridate")

for (x in packages) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x)
  } else {
    library(x, character.only = TRUE)
  }
}

################################################################################
covid <- read_xlsx("D:/Data/Training samples/misinformation_labeled.xlsx")

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
covid |>
  count(label) # the classes are pretty imbalanced, 79/973

covid_os <- ovun.sample(label~., data = covid, method = "both", p = 0.2, seed = 1234)$data
match <- subset(covid, !(covid$id %in% covid_os$id))
match <- match |>
  filter(label == 1)
covid_os <- rbind(covid_os, match)

covid_os |>
  count(label) # a bit better now, 221/832

################################################################################
set.seed(1234)
covid_split <- initial_split(covid_os, prop = 0.8, strata = label)
train <- training(covid_split)
test <- testing(covid_split)

spacy_initialize(model = "nb_core_news_sm")
# glove27b <- embedding_glove27b(dimensions = 200, manual_download = TRUE) 

covid_recipe <- recipe(label~tweet, data = covid_os) |>
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
  accuracy(truth = label, estimate = ...1002) # 0.920

cm_svm_test <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm_test$byClass["F1"] # 0.952

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 17 FN, 0 FP

################################################################################
## Tuning time
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0,0.1, by = 0.01), 
                     cost = seq(0.1,1, by = 0.1), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.01
svm_tune$best.parameters$cost # 0.8

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0.001,0.01, by = 0.001), 
                       cost = seq(0.7,9, by = 0.05), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.02
svm_tune_2$best.parameters$cost # 0.55

################################################################################
## Training the finished model
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.02,
                 cost = 0.55,
                 probability = TRUE)

model_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

cm_svm <- confusionMatrix(table(new_test$label, model_svm$...1002)) 
cm_svm$byClass["F1"] # 0.981

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 7 FN, 0 FP

################################################################################
## Loading the full data set and cleaning it
covid_df <- readRDS("D:/Data/covid_relevant_filtered.RDS")

match <- subset(covid, (covid$id %in% covid_df$id))
covid_df <- covid_df |>
  anti_join(match, by = "id")

covid_full <- covid_df |>
  sample_n(50000, seed = 1234)

# Removing urls and usernames
removeURL <- function(tweet) {
  return(gsub("http\\S+", "", tweet))
}

removeUsernames <- function(tweet) {
  return(gsub("@[a-z,A-Z,_]*[0-9]*[a-z,A-Z,_]*[0-9]*", "", tweet))
}

covid_full$text <- apply(covid_full["text"], 2, removeURL)
covid_full$text <- apply(covid_full["text"], 2, removeUsernames)

covid_full$text <- tolower(covid_full$text )
covid_full$text <- gsub("[[:punct:]]", " ", covid_full$text)
covid_full$text <- str_replace_all(covid_full$text , "[0-9]", "")

covid_full <- covid_full |>
  rename(tweet = text)

################################################################################
## Using the sVM to predict on the larger data
covid_full_baked <- bake(covid_recipe, new_data = covid_full)

covid_full_pred <- predict(svm_model, covid_full_baked, probability = TRUE)
covid_full_pred_probs <- attr(covid_full_pred, "probabilities")
covid_full_pred_probs <- covid_full_pred_probs |>
  as.data.frame() |>
  select(not_misinfo = "0", misinfo = "1")

covid_full_pred_df <- bind_cols(covid_full, covid_full_pred_probs)
covid_full_pred_df <- covid_full_pred_df |>
  filter(not_misinfo > 0.999 | misinfo > 0.9)

covid_full_pred_df_label <- covid_full_pred_df |>
  mutate(label = case_when(
    misinfo > 0.9 ~ "1",
    not_misinfo > 0.999 ~ "0"
  ))

covid_full_pred_df_label <- covid_full_pred_df_label |>
  select(tweet, label, id, conversation_id, date)

covid_predicted <- full_join(covid_full_pred_df_label, covid)

saveRDS(covid_predicted, "D:/Data/Training samples/misinformation_labeled_2.RDS")

################################################################################
## Training a new model on the larger data

covid <- readRDS("D:/Data/Training samples/misinformation_labeled_2.RDS")

covid |>
  count(label) # the classes are pretty imbalanced

covid_os <- ovun.sample(label~., data = covid, method = "both", p = 0.5, seed = 1234)$data
# match <- subset(covid, !(covid$id %in% covid_os$id))
# covid_os <- rbind(covid_os, match)

covid_os |>
  count(label) # a bit better now

################################################################################
set.seed(1234)
covid_split <- initial_split(covid_os, prop = 0.8, strata = label)
train <- training(covid_split)
test <- testing(covid_split)

spacy_initialize(model = "nb_core_news_sm")
# glove27b <- embedding_glove27b(dimensions = 200, manual_download = TRUE) 

covid_recipe <- recipe(label~tweet, data = covid_os) |>
  step_tokenize(tweet, engine = "spacyr") |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_tokenfilter(tweet, min_times = 2, max_tokens = 1000) |>
  step_tfidf(tweet) |>
  step_normalize(all_predictors()) |>
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
  accuracy(truth = label, estimate = ...1002) # 0.993

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 0 FN, 5 FP

################################################################################
## Tuning time
set.seed(1234) 
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0,0.1, by = 0.01), 
                     cost = seq(0.1,1, by = 0.1), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.02
svm_tune$best.parameters$cost # 0.3

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0,0.02, by = 0.005), 
                       cost = seq(0.2,0.4, by = 0.05), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.02
svm_tune_2$best.parameters$cost # 0.3

################################################################################
## Training the new model

svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.02,
                 cost = 0.3,
                 probability = TRUE)

test_svm <- predict(svm_model, new_test)
test_svm <- bind_cols(new_test, test_svm)
test_svm |>
  accuracy(truth = label, estimate = ...1002) # 0.999

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 1 false positive, 0 false negatives

################################################################################
## Loading the full data set and cleaning it
covid_df <- readRDS("D:/Data/covid_relevant_filtered.RDS")

match <- subset(covid, (covid$id %in% covid_df$id))
covid_df <- covid_df |>
  anti_join(match, by = "id")

covid_full <- covid_df |>
  sample_n(50000, seed = 4321)

# Removing urls and usernames
removeURL <- function(tweet) {
  return(gsub("http\\S+", "", tweet))
}

removeUsernames <- function(tweet) {
  return(gsub("@[a-z,A-Z,_]*[0-9]*[a-z,A-Z,_]*[0-9]*", "", tweet))
}

covid_full$text <- apply(covid_full["text"], 2, removeURL)
covid_full$text <- apply(covid_full["text"], 2, removeUsernames)

covid_full$text <- tolower(covid_full$text )
covid_full$text <- gsub("[[:punct:]]", " ", covid_full$text)
covid_full$text <- str_replace_all(covid_full$text , "[0-9]", "")

covid_full <- covid_full |>
  rename(tweet = text)

################################################################################
## Using the sVM to predict on the larger data
covid_full_baked <- bake(covid_recipe, new_data = covid_full)

covid_full_pred <- predict(svm_model, covid_full_baked, probability = TRUE)
covid_full_pred_probs <- attr(covid_full_pred, "probabilities")
covid_full_pred_probs <- covid_full_pred_probs |>
  as.data.frame() |>
  select(not_misinfo = "0", misinfo = "1")

covid_full_pred_df <- bind_cols(covid_full, covid_full_pred_probs)
covid_full_pred_df <- covid_full_pred_df |>
  filter(not_misinfo > 0.9999 | misinfo > 0.9)

covid_full_pred_df_label <- covid_full_pred_df |>
  mutate(label = case_when(
    misinfo > 0.9 ~ "1",
    not_misinfo > 0.9999 ~ "0"
  ))

covid_full_pred_df_label <- covid_full_pred_df_label |>
  select(tweet, label, id, conversation_id, date)

covid_predicted <- full_join(covid_full_pred_df_label, covid)
covid_predicted$date <- as.character(covid_predicted$date)

saveRDS(covid_predicted, "D:/Data/Training samples/misinformation_labeled_3.RDS")

################################################################################
## Training a new model on the larger data

covid <- readRDS("D:/Data/Training samples/misinformation_labeled_3.RDS")

covid |>
  count(label) # the classes are pretty imbalanced

covid_os <- ovun.sample(label~., data = covid, method = "both", p = 0.1, seed = 4321)$data
# match <- subset(covid, !(covid$id %in% covid_os$id))
# covid_os <- rbind(covid_os, match)

covid_os |>
  count(label) # a bit better now

################################################################################
set.seed(134)
covid_split <- initial_split(covid_os, prop = 0.8, strata = label)
train <- training(covid_split)
test <- testing(covid_split)

spacy_initialize(model = "nb_core_news_sm")
# glove27b <- embedding_glove27b(dimensions = 200, manual_download = TRUE) 

covid_recipe <- recipe(label~tweet, data = covid_os) |>
  step_tokenize(tweet, engine = "spacyr") |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_tokenfilter(tweet, min_times = 2, max_tokens = 1000) |>
  step_tfidf(tweet) |>
  step_normalize(all_predictors()) |>
  prep()

new_train <- bake(covid_recipe, new_data = train)
new_test <- bake(covid_recipe, new_data = test)

################################################################################
set.seed(12)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 probability = TRUE)

test_svm <- predict(svm_model, new_test)
test_svm <- bind_cols(new_test, test_svm)
test_svm |>
  accuracy(truth = label, estimate = ...1002) # 0.982

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 22 FN, 0 FP

################################################################################
## Tuning time
set.seed(123) 
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0,0.1, by = 0.01), 
                     cost = seq(0.1,1, by = 0.1), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.04
svm_tune$best.parameters$cost # 0.9

set.seed(4321)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0.03,0.05, by = 0.005), 
                       cost = seq(0.8,1, by = 0.05), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.035
svm_tune_2$best.parameters$cost # 0.95

################################################################################
## Training the new model

svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.035,
                 cost = 0.95,
                 probability = TRUE)

test_svm <- predict(svm_model, new_test)
test_svm <- bind_cols(new_test, test_svm)
test_svm |>
  accuracy(truth = label, estimate = ...1002) # 0.986

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 0 FP, 17 FN
