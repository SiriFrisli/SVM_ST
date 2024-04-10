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
covid_manual <- read_xlsx("E:/Data/Training samples/misinformation_labeled.xlsx")
covid_links <- readRDS("E:/Data/covid_misinfo_links_only.RDS")

covid_links <- covid_links |>
  select(-dom_url)

covid_manual <- covid_manual |>
  select(-c(claim, description, fact_check, date))

covid <- rbind(covid_links, covid_manual)

stopwords <- read_xlsx("~/INORK/Processing/stopwords.xlsx")
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
  count(label) # the classes are pretty imbalanced, 0=973, 1=4398

covid_os <- ovun.sample(label~., data = covid, method = "both", p = 0.2, seed = 1234)$data
match <- subset(covid, !(covid$id %in% covid_os$id))
match <- match |>
  filter(label == 0)
covid_os <- rbind(covid_os, match)

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
cm_svm <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm$byClass["F1"] # 0.962

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 13 FN, 54 FP

################################################################################
## Tuning time
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0.09,0.11, by = 0.005), 
                     cost = seq(0.8,1, by = 0.05), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.09
svm_tune$best.parameters$cost # 0.8

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0.01,0.09, by = 0.005), 
                     cost = seq(0.01,0.8, by = 0.05), 
                     kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.01
svm_tune_2$best.parameters$cost # 0.71

set.seed(1234)
svm_tune_3 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = 10^(-3:3), 
                       cost = c(0.01, 0.1, 1, 10, 100, 1000), 
                       kernel = "radial")

svm_tune_3$best.parameters$gamma # 0.001
svm_tune_3$best.parameters$cost # 1000

set.seed(1234)
svm_tune_4 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0.000, 0.001, by = 0.0002), 
                       cost = c(500, 800, 1000, 1200, 1500), 
                       kernel = "radial")

svm_tune_3$best.parameters$gamma # 0.001
svm_tune_3$best.parameters$cost # 1000

################################################################################
## Training the finished model
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.001,
                 cost = 1000,
                 probability = TRUE)

model_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

cm_svm <- confusionMatrix(table(new_test$label, model_svm$...1002)) 
cm_svm$byClass["F1"] # 0.9687
cm_svm$byClass["Recall"] # 0.9631336 
cm_svm$byClass["Precision"] # 0.974359 

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 22 FN, 32 FP

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
rf_default # mtry = 1000 gives acc = 0.9804, and mtry = 44 gives acc = 0.9802

tunegrid <- expand.grid(.mtry = (40:50))
rf_grid <- train(label~., 
                 data = new_train, 
                 method = "rf",
                 metric = "Accuracy", 
                 maximize = TRUE,
                 tuneGrid = tunegrid)

print(rf_grid) #

best_mtry <- 50 # acc = 0.97656
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
summary(results_tree) # 400 acc = 0.9846

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
cm_rf$byClass["F1"] # 0.9871
cm_rf$byClass["Precision"] # 0.9825
cm_rf$byClass["Recall"] # 0.99176

new_test |>
  bind_cols(predict(rf_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 15 FN, 7 FP


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
  accuracy(truth = label, estimate = .pred_class) # 0.963

test |>
  bind_cols(predict(ridge_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 15 FN, 27 FP

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
  accuracy(truth = label, estimate = .pred_class) # 0.968

test |>
  bind_cols(predict(lasso_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 26 FN, 10 FP

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
cm_lasso$byClass["F1"] # 0.9859155 
cm_lasso$byClass["Precision"] # 0.979021 
cm_lasso$byClass["Recall"] # 0.9929078 

test |>
  bind_cols(predict(final_lasso, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 18 FN, 6 FP

