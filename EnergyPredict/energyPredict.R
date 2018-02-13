library(readr)
train_df <- data.frame(read_csv("/home/vidwath/Downloads/freelancer/EnergyPredict/ae5e3a76-0-energy/train.csv"))
train_df$Observation <- NULL
train_df$Energy <- as.numeric(train_df$Energy)
library(caret)
set.seed(100)
index_v <- createDataPartition(train_df$Energy, p = 0.65, list = FALSE)
training <- train_df[index_v, ]
test <- train_df[-index_v, ]

lm_mdl <- lm(Energy~.,data=training)

tested_obs <- round(predict(lm_mdl,test))
setdiff(tested_obs,test$Energy)

library(randomForest)
rdm_mdl <- randomForest::randomForest(Energy~.,training,mtry=10,ntree=500)
tested_obs <- round(predict(rdm_mdl,test))

install.packages("rnn")
library(rnn)
rnn_mdl <- rnn::trainr(Y = as.matrix(train_df$Energy),X = as.matrix(train_df[,-25]),learningrate = 0.1,seq_to_seq_unsync = TRUE,network_type = "lstm",numepochs = 5)
tested_obs <- predictr(rnn_mdl,as.matrix(test[,-25]),real_output = T)

#### Production#######
test_df <- read_csv("/home/vidwath/Downloads/freelancer/EnergyPredict/ae5e3a76-0-energy/test.csv")
Observation <- test_df$Observation
Energy <- round(predict(lm_mdl,test_df))
predicted_result <- cbind(Observation,Energy)
write.csv(predicted_result,"/home/vidwath/Downloads/freelancer/EnergyPredict/ae5e3a76-0-energy/pred_result.csv",row.names = F)


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

