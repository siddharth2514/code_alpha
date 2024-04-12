% 1. Library 

library(caret)
library(dplyr)
library(e1071)
library(MASS)
library(MLmetrics)
library(readr)
library(rpart)
library(UBL)


% 2. Read Data

train <- read.csv("train.csv")
test <- read.csv("test.csv")


% 3. Remove Bank Client ID Feature

train$X = NULL
test$X = NULL


% 4. Missing Value(s) Inspection

train[,!complete.cases(train)]
test[,!complete.cases(train)]


%5. Data Types Conversion

train$jumlah_kartu = as.numeric(train$jumlah_kartu)
train$outstanding = as.numeric(train$outstanding)
train$skor_delikuensi = as.numeric(train$skor_delikuensi)
train$flag_kredit_macet = as.factor(as.numeric(train$flag_kredit_macet))

test$jumlah_kartu = as.numeric(test$jumlah_kartu)
test$outstanding = as.numeric(test$outstanding)
test$skor_delikuensi = as.numeric(test$skor_delikuensi)


% 6. Standardize All Numerical Feature(s)
indeks_train <- sapply(train, is.numeric)
train[indeks_train] <- lapply(train[indeks_train], scale)
rm(indeks_train)

indeks_test <- sapply(test, is.numeric)
test[indeks_test] <- lapply(test[indeks_test], scale)
rm(indeks_test)

% 7. Apply [S]ynthetic [M]inority [O]versampling [TE]chnique for Nominal and Numerical Feature(s)

set.seed(2020)
% pick between HEOM and HVDM
balanced_train = SmoteClassif(flag_kredit_macet~., train, C.perc="balance", dist="HEOM")


% 8. Split Train Data into Learning Data and Validation Data

set.seed(2020)
samp = sample(1:nrow(balanced_train), nrow(balanced_train)*0.7)
learning = balanced_train[samp,]
validation = balanced_train[-samp,]


% 9. Construct Logistic Regression Model

set.seed(2020)
logmodel = glm(flag_kredit_macet~., learning, family = "binomial")
prob_logpred = predict(logmodel,validation,type = "response")
logpred = ifelse(prob_logpred > 0.5, "1", "0") 
table(logpred,validation$flag_kredit_macet)
F1_Score(y_pred = logpred, y_true = validation$flag_kredit_macet, positive = "1")


% 10. Construct Support Vector Machine Model

set.seed(2020)
svm_model <- svm(flag_kredit_macet ~ ., data=learning)
#summary(svm_model)
svm_pred <- predict(svm_model,validation)
table(svm_pred,validation$flag_kredit_macet)
F1_Score(y_pred = svm_pred, y_true = validation$flag_kredit_macet, positive = "1")


% 11. Construct Decision Tree Model

set.seed(2020)
tree_model = rpart(flag_kredit_macet ~ ., data = learning, method = 'class')
tree_pred = as.data.frame(predict(tree_model,validation))
tree_pred = tree_pred %>%
            mutate(decision = ifelse( tree_pred$`0` < tree_pred$`1` , 1, 0))
tree_pred = tree_pred$decision 
table(tree_pred,validation$flag_kredit_macet)
F1_Score(y_pred = tree_pred, y_true = validation$flag_kredit_macet, positive = "1")


% 12. Construct Nearest Neighbour Model

set.seed(2020)
trCtrl = trainControl(method = "repeatedcv", number = 10, repeats = 3 )
nnmodel = train(flag_kredit_macet~.,data=learning,method = 'knn',tuneLength = 20, trControl = trCtrl)
nnpred = predict(nnmodel,validation)
table(nnpred,validation$flag_kredit_macet)
F1_Score(y_pred = nnpred, y_true = validation$flag_kredit_macet, positive = "1")


% 13. Predicting "Default" Feature on Test Data

set.seed(2020)
test_prob_logpred = predict(logmodel,test,type = "response")
test_logpred = ifelse(test_prob_logpred > 0.5, "1", "0") # ini yang digunakan

test_svm_pred <- predict(svm_model,test) # ini yang digunakan

test_tree_pred = as.data.frame(predict(tree_model,test))
test_tree_pred = test_tree_pred %>%
            mutate(decision = ifelse( test_tree_pred$`0` < test_tree_pred$`1` , 1, 0))
test_tree_pred = test_tree_pred$decision # ini yang digunakan

test_nnpred = predict(nnmodel,test)


% 14. Best Model Selection (Best model, logically, means model with highest F1_Score)

table(test_logpred)

table(test_svm_pred)

table(test_tree_pred)

table(test_nnpred)


% 15. Clear Console and All Variables in The Global Environment

rm(list=ls())
cat("\014")  


