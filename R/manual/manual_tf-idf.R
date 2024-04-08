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

################################################################################
## TRAINING SVM MODEL

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
                       gamma = seq(0.005,0.02, by = 0.005), 
                       cost = seq(0.7,0.9, by = 0.05), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.01
svm_tune_2$best.parameters$cost # 0.8

################################################################################
## Training the finished model
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.01,
                 cost = 0.8,
                 probability = TRUE)

model_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

svm_pred <- predict(svm_model, new_test)
svm_pred <- bind_cols(test, svm_pred)

cm_svm <- confusionMatrix(table(new_test$label, model_svm$...1002)) 
cm_svm$byClass["F1"] # 0.9681159 
cm_svm$byClass["Precision"] # 1
cm_svm$byClass["Recall"] # 0.9382022 

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 11 FN, 0 FP

################################################################################
## TRAINING RANDOM FOREST
control <- trainControl(method = "cv",
                        number = 5,
                        search = "grid")

set.seed(1234)
rf_default <- train(label~., 
                    data = new_train, 
                    method = "rf",
                    metric = "Accuracy", 
                    maximize = TRUE,
                    trControl = control)
rf_default # mtry = 44 gives acc = 0.9596

tunegrid <- expand.grid(.mtry = (35:55))
rf_grid <- train(label~., 
                 data = new_train, 
                 method = "rf",
                 metric = "Accuracy", 
                 maximize = TRUE,
                 tuneGrid = tunegrid)

print(rf_grid) # mtry = 49 gives acc = 9499

best_mtry <- 49
best_grid <- expand.grid(.mtry = best_mtry)

store_maxtrees <- list()
for (ntree in c(300, 400, 500, 600, 700, 800, 900, 1000)) {
  set.seed(1234)
  rf_maxtrees <- train(label~.,
                       data = new_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = best_grid,
                       trControl = control,
                       importance = TRUE,
                       nodesize = 10,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree) # the max accuracy doesn't change, but the min acc is highest at 400 tree

rf_model <- train(label~.,
                  data = new_train,
                  method = "rf",
                  metric = "Accuracy",
                  tuneGrid = best_grid,
                  trControl = control,
                  importance = TRUE,
                  nodesize = 1,
                  ntree = 400)

model_rf <- new_test |>
  bind_cols(predict(rf_model, new_test))

rf_pred <- predict(rf_model, new_test)
rf_pred <- bind_cols(test, rf_pred)

cm_rf <- confusionMatrix(table(new_test$label, model_rf$...1002)) 
cm_rf$byClass["F1"] # 0.9681159 
cm_rf$byClass["Precision"] # 1
cm_rf$byClass["Recall"] # 0.9382022 

new_test |>
  bind_cols(predict(rf_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 11 FN, 0 FP

################################################################################
## TRAINING LOGISTIC REGRESSION

# creating a new recipe, no prep
reg_recipe <- recipe(label~tweet, data=covid_os) |>
  step_tokenize(tweet, engine = "spacyr") |>
  # step_date(date, features = c("year", "month")) |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_tokenfilter(tweet, min_times = 2, max_tokens = 1000) |>
  step_tfidf(tweet)

## Ridge penalty
ridge_spec <- logistic_reg(penalty = 0.01, mixture = 0) |>
  set_mode("classification") |>
  set_engine("glmnet")

set.seed(1234)
ridge_fit <- workflow() |>
  add_recipe(reg_recipe) |>
  add_model(ridge_spec) |>
  fit(data = train)

ridge_preds <- predict(ridge_fit, test)
ridge_preds <- bind_cols(test, ridge_preds)

ridge_preds |>
  accuracy(truth = label, estimate = .pred_class) # 0.844

test |>
  bind_cols(predict(ridge_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 1 FP, 32 FN

## LASSO penalty
lasso_spec <- logistic_reg(penalty = 0.01, mixture = 1) |>
  set_mode("classification") |>
  set_engine("glmnet")

set.seed(1234)
lasso_fit <- workflow() |>
  add_recipe(reg_recipe) |>
  add_model(lasso_spec) |>
  fit(data = train)

lasso_preds <- predict(lasso_fit, test)
lasso_preds <- bind_cols(test, lasso_preds)

lasso_preds |>
  accuracy(truth = label, estimate = .pred_class) # 0.91

test |>
  bind_cols(predict(lasso_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 6 FP, 13 FN

## Tuning LASSO
set.seed(1234)
train_boot <- bootstraps(train, strata = label)

lasso_tune_spec <- logistic_reg(penalty = tune(), mixture = 1) |>
  set_mode("classification") |>
  set_engine("glmnet")

lambda_grid <- grid_regular(penalty(), levels = 50)

lasso_tune_wf <- workflow() |>
  add_recipe(reg_recipe) |>
  add_model(lasso_tune_spec)

set.seed(1234)
lasso_grid <- tune_grid(lasso_tune_wf,
                        resamples = train_boot,
                        grid = lambda_grid)

lasso_grid|>
  collect_metrics()

best_acc <- lasso_grid |>
  select_best("accuracy")

final_lasso <- finalize_workflow(
  lasso_tune_wf,
  best_acc) |>
  fit(data = train)

lasso_tune_preds <- test |>
  bind_cols(predict(final_lasso, test))

cm_lasso <- confusionMatrix(table(test$label, lasso_tune_preds$.pred_class)) 
cm_lasso$byClass["F1"] # 0.961
cm_lasso$byClass["Precision"] # 0.958
cm_lasso$byClass["Recall"] # 0.964

test |>
  bind_cols(predict(final_lasso, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 7 FP, 6 FN

