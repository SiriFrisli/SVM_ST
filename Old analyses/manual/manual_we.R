packages <- c("tidyverse", "tidymodels", "readxl", "recipes", "textrecipes", "tune", 
              "e1071", "tidytext", "caret", "ROSE", "spacyr", "textdata", "lubridate",
              "glmnet", "kernlab", "randomForest")

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

covid$label <- case_when(
  covid$label == 0 ~ "non.misinfo",
  covid$label == 1 ~ "misinfo"
)

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

train |>
  count(label) # the classes are pretty imbalanced, 64/777

# train_os <- ovun.sample(label~., data = train, method = "both", p = 0.2, seed = 1234)$data
# match <- subset(train, !(train_os$id %in% train_os$id))
# match <- match |>
#   filter(label == 1)
# train_os <- rbind(train_os, match)
# 
# train_os |>
#   count(label) # a bit better now, 175/666

################################################################################
spacy_initialize(model = "nb_core_news_sm")
no_we <- read.table("C:/Users/sirifris.ADA/OneDrive - OsloMet/Dokumenter/no.wiki.bpe.vs50000.d200.w2v.txt", header = FALSE, sep = " ", quote = "", encoding = "UTF-8")
no_we <- no_we |>
  as_tibble()
no_we$V1 <- gsub("▁", "", no_we$V1)

################################################################################
## TRAINING SVM MODEL
covid_recipe <- recipe(label~tweet, data = covid) |>
  step_tokenize(tweet, engine = "spacyr") |>
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
  accuracy(truth = label, estimate = ...202) # 0.929

cm_svm_test <- confusionMatrix(table(new_test$label, test_svm$...202)) 
cm_svm_test$byClass["F1"] # 0.963145  

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 0 FP, 15 FN (0 TP)

################################################################################
## Tuning time

ctrl <- tune.control(sampling = "cross",
                     cross = 5,
                     performances = TRUE,
                     best.model = TRUE)

set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = 10^(-3:3),
                     class.weights = c("misinfo" = 8.219, "non.misinfo" = 0.677),  
                     kernel = "radial",
                     tunecontrol = ctrl)

svm_tune$best.parameters$gamma # 0.001
svm_tune$best.parameters$class.weights # 1

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0,0.001, by = 0.0001), 
                       cost = seq(1,10, by = 1), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0
svm_tune_2$best.parameters$cost # 1

################################################################################
## Training the finished model

set.seed(1234)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 class.weights = c("misinfo" = 8.219, "non.misinfo" = 0.677),
                 probability = TRUE)

model_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

svm_pred <- predict(svm_model, new_test)
svm_pred <- bind_cols(test, svm_pred)

cm_svm <- confusionMatrix(table(new_test$label, model_svm$...202)) 
cm_svm$byClass["F1"] # 0.4166667   
cm_svm$byClass["Precision"] # 0.3333333 
cm_svm$byClass["Recall"] # 0.5555556 

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 0 FP, 15 FN (no TP)

################################################################################
## TRAINING RANDOM FOREST
ctrl <- trainControl(method = "repeatedcv",
                     number = 5, 
                     repeats = 2,
                     summaryFunction = prSummary,
                     classProbs = TRUE,
                     verboseIter = TRUE,
                     search = "grid")

model_weights <- ifelse(new_train$label == "non.misinfo",
                        1052/(table(new_train$label)[1] * 2),
                        1052/(table(new_train$label)[2] * 2))

rf_model <- train(label~.,
                  data = new_train,
                  method = "rf",
                  weights = model_weights,
                  metric = "F",
                  trControl = ctrl)

set.seed(1234)
rf_default <- train(label~., 
                    data = new_train, 
                    method = "rf",
                    metric = "Accuracy", 
                    maximize = TRUE,
                    trControl = control)
rf_default # mtry = 2, acc = 0.9239071    

tunegrid <- expand.grid(.mtry = 20:50)
rf_grid2 <- train(label~., 
                  data = new_train, 
                  method = "rf",
                  metric = "Accuracy", 
                  maximize = TRUE,
                  tuneGrid = tunegrid)

print(rf_grid2) # 49 acc = 0.9206235

best_mtry <- 49
best_grid <- expand.grid(.mtry = best_mtry)

store_maxtrees <- list()
for (ntree in c(50, 100, 200, 400, 500, 1000)) {
  set.seed(1234)
  rf_maxtrees <- train(label~.,
                       data = new_train,
                       method = "rf",                  
                       weights = model_weights,
                       metric = "F",
                       tuneGrid = best_grid,
                       trControl = control,
                       importance = TRUE,
                       nodesize = 10,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree) # 200, acc = 0.9761905    

rf_model <- train(label~.,
                  data = new_train,
                  method = "rf",
                  metric = "F",
                  weights = model_weights,
                  tuneGrid = best_grid,
                  trControl = ctrl,
                  importance = TRUE,
                  nodesize = 1,
                  ntree = 200)

model_rf <- new_test |>
  bind_cols(predict(rf_model, new_test))

rf_pred <- predict(rf_model, new_test)
rf_pred <- bind_cols(test, rf_pred)

cm_rf <- confusionMatrix(table(new_test$label, model_rf$...202)) 
cm_rf$byClass["F1"] # 0.963145   
cm_rf$byClass["Precision"] # 1 
cm_rf$byClass["Recall"] # 0.92891   

new_test |>
  bind_cols(predict(rf_model, new_test)) |>
  conf_mat(truth = label, estimate = ...202) |>
  autoplot(type = "heatmap") # 0 FP, 15 FN

cw <- c(0.677, 8.219)
rf_model <- randomForest(formula = label~.,
                         data = new_train,
                         classwt = cw)
ss <- c(64,64)
cw <- c(0.677, 8.219)
rf_model_samp <- randomForest(formula = label~.,
                              data = new_train,
                              sampsize = ss,
                              classwt = cw)

################################################################################
## TRAINING LOGISTIC REGRESSION

# creating a new recipe, no prep
reg_recipe <- recipe(label~tweet, data=covid) |>
  step_tokenize(tweet, engine = "spacyr") |>
  step_stopwords(tweet, language = "no", keep = FALSE, 
                 stopword_source = "snowball", 
                 custom_stopword_source = custom_words) |>
  step_lemma(tweet) |>
  step_
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
  accuracy(truth = label, estimate = .pred_class) # 0.924

test |>
  bind_cols(predict(ridge_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 4 FP, 13 FN

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
  accuracy(truth = label, estimate = .pred_class) # 0.929

test |>
  bind_cols(predict(lasso_fit, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 0 FP, 15 FN

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
cm_lasso$byClass["F1"] # 0.963145   
cm_lasso$byClass["Precision"] # 1  
cm_lasso$byClass["Recall"] # 0.92891   

test |>
  bind_cols(predict(final_lasso, test)) |>
  conf_mat(truth = label, estimate = .pred_class) |>
  autoplot(type = 'heatmap') # 20 FP, 8 FN
