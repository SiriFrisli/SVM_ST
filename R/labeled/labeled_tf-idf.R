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
covid <- read_xlsx("E:/Data/Training samples/misinformation_labeled.xlsx")

covid <- covid |>
  select(id, tweet = text, conversation_id, date, label)

stopwords <- read_xlsx("~/INORK/Processing/stopwords.xlsx")
custom_words <- stopwords |>
  pull(word)

################################################################################
# Some preprocessing
covid$label <- as.factor(covid$label) # Outcome variable needs to be factor
covid$tweet <- str_replace_all(covid$tweet, "USERNAME", "")
covid$tweet <- tolower(covid$tweet)
covid$tweet <- gsub("[[:punct:]]", " ", covid$tweet)
covid$tweet <- str_replace_all(covid$tweet, "[0-9]", "")

removeURL <- function(tweet) {
  return(gsub("http\\S+", "", tweet))
}
covid$tweet <- apply(covid["tweet"], 1, removeURL)


################################################################################
covid |>
  count(label) # the classes are pretty imbalanced

covid_os <- ovun.sample(label~., data = covid, method = "over", p = 0.3, seed = 1234)$data
match <- subset(covid, !(covid$id %in% covid_os$id))
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

test_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

cm_svm <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm$byClass["F1"] # 0.947

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 21 FN, 0 FP

################################################################################
## Tuning time
set.seed(1234)
svm_tune <- tune.svm(x = label ~., 
                     data = new_train,
                     gamma = seq(0,0.1, by = 0.01), 
                     cost = seq(0.1,1, by = 0.1), 
                     kernel = "radial")

svm_tune$best.parameters$gamma # 0.02
svm_tune$best.parameters$cost # 0.8

set.seed(1234)
svm_tune_2 <- tune.svm(x = label ~., 
                       data = new_train,
                       gamma = seq(0.015,0.025, by = 0.005), 
                       cost = seq(0.75,0.85, by = 0.05), 
                       kernel = "radial")

svm_tune_2$best.parameters$gamma # 0.02
svm_tune_2$best.parameters$cost # 0.8

################################################################################
## Training the finished model
set.seed(4321)
svm_model <- svm(formula = label~.,
                 data = new_train,
                 type = "C-classification",
                 kernel = "radial",
                 cross = 5,
                 gamma = 0.02,
                 cost = 0.8,
                 probability = TRUE)

test_svm <- new_test |>
  bind_cols(predict(svm_model, new_test))

cm_svm <- confusionMatrix(table(new_test$label, test_svm$...1002)) 
cm_svm$byClass["F1"] # 0.966
cm_svm$byClass["Precision"] # 1
cm_svm$byClass["Recall"] # 0.935

new_test |>
  bind_cols(predict(svm_model, new_test)) |>
  conf_mat(truth = label, estimate = ...1002) |>
  autoplot(type = "heatmap") # 13 FN, 0 FP

################################################################################
## RANDOM FOREST
control <- trainControl(method = "cv",
                        number = 5,
                        search = "grid")

set.seed(1234)
rf_default <- train(label~., 
                    data = new_train, 
                    method = "rf",
                    metric = "Accuracy", 
                    trControl = control)
rf_default
# Best acc at 0.9527 with mtry = 4

grids <- expand.grid(.mtry = c(seq(from = 35, to = 50, by = 1)))
set.seed(1234)
rf_mtry <- train(
  label~., 
  data = new_train, 
  method = "rf",
  metric = "Accuracy", 
  trControl = control,
  tuneGrid = grids, 
  importance = TRUE,
  nodesize = 15, # Increasing the nodesize to make it go a bit faster
  ntree = 500
)

rf_mtry
# acc = 0.954 at mtry = 42, the increased nodesize decreases accuracy a bit, 

# Trying with a lower nodesize
best_mtry <- rf_mtry$bestTune$mtry 
best_grid <- expand.grid(.mtry = best_mtry)

set.seed(1234)
rf_mtry_2 <- train(
  label~., 
  data = new_train, 
  method = "rf",
  metric = "Accuracy", 
  trControl = control,
  tuneGrid = best_grid, 
  importance = TRUE,
  nodesize = 5,
  ntree = 500
)
rf_mtry_2
# decreasing the nodesize to 5 increases the accuracy only slightly to 0.957, 
# but increases the time spent fitting the model by quite a lot, sticking to 
# a nodesize of 15 seems to be the best. 

# time to test for best maxnodes
store_maxnode <- list()
for (maxnodes in c(1:30)) {
  set.seed(1234)
  rf_maxnode <- train(label~.,
                      data = new_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = best_grid,
                      trControl = control,
                      importance = TRUE,
                      nodesize = 15,
                      maxnodes = maxnodes,
                      ntree = 500)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)
# 57 maxnodes acc = 0.811, better to just not specify it?


store_maxtrees <- list()
for (ntree in c(300, 350, 400, 450, 500, 550, 600, 800, 900, 1000)) {
  set.seed(1234)
  rf_maxtrees <- train(label~.,
                       data = new_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = best_grid,
                       trControl = control,
                       importance = TRUE,
                       nodesize = 15,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)
# No change inn acc after 350 trees, acc = 0.967

# Final model:
grid <- c(21)
grid <- grid |>
  as.data.frame()
colnames(grid)[1] <- ".mtry"

control <- trainControl(method = "cv",
                        number = 5,
                        search = "grid")
set.seed(1234)
final_rf <- train(
  label~., 
  data = new_train, 
  method = "rf",
  metric = "Accuracy", 
  trControl = control,
  tuneGrid = grid, 
  importance = TRUE,
  nodesize = 15, 
  ntree = 350
)

final_rf
# acc = 0.951
