#importing datasets
test <- read.csv(file.choose(), header = T)
train <- read.csv(file.choose(), header = T)

# install packages
install.packages("readr")
install.packages("ggplot2")
install.packages("caret")
install.packages("Matrix")
install.packages("xgboost")

# load required packages
library(readr)
library(ggplot2)
library(caret)
library(Matrix)
library(xgboost)

#separate label(digit drawn) by the user from train dataset.
X <- train[,-1]
Y <- train[,1]
trainlabel <- train[,1]


# xgboost parameters

PARAM <- list(
  # General Parameters
  booster            = "gbtree",          
  silent             = 0,                 
  # Booster Parameters
  eta                = 0.05,              
  gamma              = 0,                 
  max_depth          = 5,                
  min_child_weight   = 1,                 
  subsample          = 0.70,             
  colsample_bytree   = 0.95,             
  num_parallel_tree  = 1,                 
  lambda             = 0,                
  lambda_bias        = 0,                 
  alpha              = 0,                
  # Task Parameters
  objective          = "multi:softmax",   
  num_class          = 10,                
  base_score         = 0.5,               
  eval_metric        = "merror"           
)

# convert TRAIN dataframe into a design matrix
TRAIN_SMM <- sparse.model.matrix(Y ~ ., data = X)
TRAIN_XGB <- xgb.DMatrix(data = TRAIN_SMM, label = Y)

# set seed
set.seed(1)

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = TRAIN_XGB, 
                   nrounds     = 400 , 
                   verbose     = 2,
                   watchlist   = list(TRAIN_SMM = TRAIN_XGB)
)

# attach a predictions vector to the test dataset
test$label <- 0

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("LABEL")
TEST_SMM <- sparse.model.matrix(LABEL ~ ., data = TEST)
PRED <- predict(MODEL, TEST_SMM)

# create submission file
SUBMIT <- data.frame(ImageId = c(1:length(PRED)), Label = PRED)
write_csv(SUBMIT, "submission.csv")

######
 testsub = train[35001:42000,]
 trainsub= train[1:35000,]
 
#Method 2
install.packages("drat", repos = "https://cran.rstudio.com")
drat::addRepo("dmlc")
install.packages("mxnet")
library(mxnet)
 
 
 
 





