#install.packages("nnet")
#install.packages("readr")
library(nnet) ## for multinomial
library(readr) ## read csv
# train and test file path are hard coded as of now
train_df <- data.frame(readr::read_csv("/home/vidwath/Downloads/freelancer/GlogmanSachs/data/train.csv"))
test_df <- data.frame(readr::read_csv("/home/vidwath/Downloads/freelancer/GlogmanSachs/data/test.csv",col_names = F))
train_df[] <- lapply(train_df, as.factor) # converting all columns to factors

colnames(test_df) <- colnames(train_df)[1:6] # assigning test data with column names
test_df[] <- lapply(test_df,as.factor) # converting test data to factor


#' Metrics
#'
#' @param cm confusion matrix to be passed
#'
#' @return
#' @export
#'
#' @examples
metrics <- function(cm){
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  diag = diag(cm) # number of correctly classified instances per class
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes

  accuracy = sum(diag) / n
  precision = diag / colsums
  recall = diag / rowsums
  f1 = 2 * precision * recall / (precision + recall)
  print("F1 score ")
  print(f1)
}

##### Multinomial Regression #####


library(caret)
set.seed(100)
index_v <- createDataPartition(train_df$popularity, p = 0.65, list = FALSE)
training <- train_df[index_v, ]
test <- train_df[-index_v, ]

#' Title
#'
#' @param training training data frame
#' @param test test data frame
#' @param prod_df production data frame
#' @param output_file output file name
#'
#' @return
#' @export
#'
#' @examples
multinomial_model <- function(training,test,prod_df = test_df,output_file="multinom_result.csv"){
  multinomModel <- multinom(popularity ~ ., data=training,maxit=500)

  predicted_tr_class <- predict (multinomModel, training)
  cm = as.matrix(table(training$popularity,predicted_tr_class))
  print("Training Result:")
  metrics(cm)
  predicted_te_class <- predict (multinomModel, test)
  cm = as.matrix(table(test$popularity,predicted_te_class))
  print("Test results: ")
  metrics(cm)

  predicted_te_class <- predict (multinomModel, prod_df)
  write.csv(pred_prod,output_file,row.names = F)
}







###### Random Forest ###########
library(randomForest)
#' random forest modelling
#'
#' @param training trainign data frame
#' @param test test data frame
#' @param prod_df  production data frame
#' @param output_file output file name
#'
#' @return
#' @export
#'
#' @examples
random_forest_model <- function(training,test,prod_df,output_file="randomforest_result.csv"){
  ## Build random forest model
  random_model <- randomForest::randomForest(popularity~.,data = training,method = "class",
                                            mtry = 20,ntree = 5000,proximity=TRUE,corr.bias=TRUE,
                                            maxnodes=1000)
  predict_tr_cart = predict(random_model,newdata = training,type ="class")
  cm = as.matrix(table(training$popularity,predict_tr_cart))
  print("Metrics details for training")
  metrics(cm)

  predict_cart = predict(random_model,newdata = test,type ="class")
  cm = as.matrix(table(test$popularity,predict_cart))
  print("Metrics details for test")
  metrics(cm)

  ## Run on the production data
  predicted_class <- predict(random_model,prod_df,type = "class")
  write.csv(predicted_class,output_file,row.names = F)
}







###---------------- XGboost------------------------##




library("xgboost")  # the main algorithm
library("archdata") # for the sample dataset
library("caret")    # for the confusionmatrix() function (also needs e1071 package)
library("dplyr")    # for some data preperation
#' Title
#'
#' @param train_df train data frame
#' @param test_df production data frame
#' @param cv.nfold cross validation folds
#' @param nround number of rounds on xgboost
#' @param output_file result file name
#'
#' @return
#' @export
#'
#' @examples
data_processing_xgboost <- function(train_df,test_df,cv.nfold=5,nround=50,output_file="predicted_xgboost.csv"){
  train_df[] <- lapply(train_df, as.numeric)
  train_index <- sample(1:nrow(train_df), nrow(train_df)*0.75) # split percentage
  train_df$popularity <- train_df$popularity - 1
  # Full data set
  data_variables <- as.matrix(train_df[,-7])
  data_label <- train_df[,"popularity"]
  data_matrix <- xgb.DMatrix(data = as.matrix(train_df), label = data_label)
  # split train data and make xgb.DMatrix
  train_data   <- data_variables[train_index,]
  train_label  <- (data_label[train_index])
  train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
  # split test data and make xgb.DMatrix
  test_data  <- data_variables[-train_index,]
  test_label <- data_label[-train_index]
  test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

  numberOfClasses <- length(unique(train_df$popularity))
  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses)

  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model <- xgb.cv(params = xgb_params,
                     data = train_matrix,
                     nrounds = nround,
                     nfold = cv.nfold,
                     verbose = FALSE,
                     prediction = TRUE)

  OOF_prediction <- data.frame(cv_model$pred) %>%
    mutate(max_prob = max.col(., ties.method = "last"),
           label = train_label + 1)
  print("Training Summary details")
  print(confusionMatrix(factor(OOF_prediction$label),
                  factor(OOF_prediction$max_prob),
                  mode = "everything"))

  ## get the best cross validated model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = nround)

  # Predict on test set
  test_matrix <- xgb.DMatrix(data = test_data, label = test_label)
  test_pred <- predict(bst_model, newdata = test_matrix)
  test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                            ncol=length(test_pred)/numberOfClasses) %>%
    t() %>%
    data.frame() %>%
    mutate(label = test_label + 1,
           max_prob = max.col(., "last"))
  # confusion matrix of test set
  print("Summary stat on test data")
  print(confusionMatrix(factor(test_prediction$label),
                  factor(test_prediction$max_prob),
                  mode = "everything"))

  test_df[] <- lapply(test_df, as.numeric)
  prod_matrix <- xgb.DMatrix(data = data.matrix(test_df)) # matrix ready for test data
  prod_pred <- predict(bst_model, newdata = prod_matrix)
  pred <- matrix(prod_pred, ncol=numberOfClasses, byrow=TRUE)
  pred_labels <- max.col(pred) # get the predicted class
  print("Summary of predicted class")
  print(table(pred_labels))
  write.csv(pred_labels,output_file,row.names = F)

}


