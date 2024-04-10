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
covid_links <- readRDS("D:/Data/covid_misinfo_links_only.RDS")

covid_links <- covid_links |>
  select(-dom_url)

covid <- covid |>
  select(-c(claim, description, fact_check, date))

covid_full <- rbind(covid_links, covid)
covid <- covid_full

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
  count(label) # 973/4398

covid_os <- ovun.sample(label~., data = covid, method = "both", p = 0.2, seed = 1234)$data
match <- subset(covid, !(covid$id %in% covid_os$id))
match <- match |>
  filter(label == 0)
covid_os <- rbind(covid_os, match)

covid_os |>
  count(label) # 1401/4287

################################################################################
set.seed(1234)
covid_split <- initial_split(covid_os, prop = 0.8, strata = label)
train <- training(covid_split)
test <- testing(covid_split)

spacy_initialize(model = "nb_core_news_sm")
no_we <- read.table("C:/Users/sirifris.ADA/OneDrive - OsloMet/Dokumenter/no.wiki.bpe.vs50000.d200.w2v.txt", header = FALSE, sep = " ", quote = "", encoding = "UTF-8")
no_we <- no_we |>
  as_tibble()
no_we$V1 <- gsub("▁", "", no_we$V1)

################################################################################
## TRAINING SVM MODEL
covid_recipe <- recipe(label~tweet, data = covid_os) |>
  step_tokenize(tweet, engine = "spacyr") |>
  # step_date(date, features = c("year", "month")) |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_word_embeddings(tweet, embeddings = no_we) |>
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
  accuracy(truth = label, estimate = ...202) # 0.846

cm_svm_test <- confusionMatrix(table(new_test$label, test_svm$...202)) 
cm_svm_test$byClass["F1"] # 0.9061662 

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 13 FN, 162 FP

################################################################################
## Tuning time
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = 10^(-3:3), 
                     cost = c(0.01, 0.1, 1, 10, 100, 1000),  
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.01
svm_tune$best.parameters$cost # 10

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0,0.01, by = 0.002), 
                       cost = seq(5,15, by = 1), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.006
svm_tune_2$best.parameters$cost # 8

################################################################################
## Training the finished model
set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.006,
                 cost = 8,
                 probability = TRUE)

model_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

svm_pred <- predict(svm_model, new_test)
svm_pred <- bind_cols(test, svm_pred)

cm_svm <- confusionMatrix(table(new_test$label, model_svm$...202)) 
cm_svm$byClass["F1"] # 0.9101124
cm_svm$byClass["Precision"] # 0.9440559
cm_svm$byClass["Recall"] # 0.8785249

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 48 FN, 11 FP

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
rf_default # mtry = 101, acc = 0.8645878

tunegrid <- expand.grid(.mtry = 90:110)
rf_grid <- train(label~., 
                 data = new_train, 
                 method = "rf",
                 metric = "Accuracy", 
                 maximize = TRUE,
                 tuneGrid = tunegrid)

print(rf_grid) # 

best_mtry <- 103 # acc = 0.8513829
best_grid <- expand.grid(.mtry = best_mtry)

store_maxtrees <- list()
for (ntree in c(50, 100, 200, 400, 500, 1000)) {
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
summary(results_tree) # 200, acc = 0.8756876  

rf_model <- train(label~.,
                  data = new_train,
                  method = "rf",
                  metric = "Accuracy",
                  tuneGrid = best_grid,
                  trControl = control,
                  importance = TRUE,
                  nodesize = 1,
                  ntree = 200)

model_rf <- new_test |>
  bind_cols(predict(rf_model, new_test))

rf_pred <- predict(rf_model, new_test)
rf_pred <- bind_cols(test, rf_pred)

cm_rf <- confusionMatrix(table(new_test$label, model_rf$...202)) 
cm_rf$byClass["F1"] # 0.9264305 
cm_rf$byClass["Precision"] # 0.990676 
cm_rf$byClass["Recall"] # 0.8700102 

new_test |>
  bind_cols(predict(rf_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 8 FN, 127 FP

################################################################################
## TRAINING LOGISTIC REGRESSION

# creating a new recipe, no prep
reg_recipe <- recipe(label~tweet, data=covid_os) |>
  step_tokenize(tweet, engine = "spacyr") |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_word_embeddings(tweet, embeddings = no_we)

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
  accuracy(truth = label, estimate = .pred_class) # 0.795

test |>
  bind_cols(predict(ridge_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 55 FN, 178 FP

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
  accuracy(truth = label, estimate = .pred_class) # 0.787

test |>
  bind_cols(predict(lasso_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 22 FN, 221 FP

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

lasso_grid |>
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
cm_lasso$byClass["F1"] # 0.87708  
cm_lasso$byClass["Precision"] # 0.9522145 
cm_lasso$byClass["Recall"] # 0.8129353 

test |>
  bind_cols(predict(final_lasso, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 41 FN, 188 FP

## Tuning ridge
set.seed(1234)
train_boot <- bootstraps(train, strata = label)

ridge_tune_spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_mode("classification") |>
  set_engine("glmnet")

lambda_grid <- grid_regular(penalty(), levels = 50)

ridge_tune_wf <- workflow() |>
  add_recipe(reg_recipe) |>
  add_model(ridge_tune_spec)

set.seed(1234)
ridge_grid <- tune_grid(ridge_tune_wf,
                        resamples = train_boot,
                        grid = lambda_grid)

ridge_grid |>
  collect_metrics()

best_acc <- ridge_grid |>
  select_best("accuracy")

final_ridge <- finalize_workflow(
  ridge_tune_wf,
  best_acc) |>
  fit(data = train)

ridge_tune_preds <- test |>
  bind_cols(predict(final_ridge, test))

cm_ridge <- confusionMatrix(table(test$label, ridge_tune_preds$.pred_class)) 
cm_ridge$byClass["F1"] # 0.872074  
cm_ridge$byClass["Precision"] # 0.9335664   
cm_ridge$byClass["Recall"] # 0.8181818  

test |>
  bind_cols(predict(final_ridge, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 57 FN, 178 FP
