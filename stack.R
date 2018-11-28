setwd('D:\\Auction_master_kr')
list.files()

options(scipen = 10)
data<- read.csv('Auction_master_train.csv',fileEncoding="UTF-8")
data<- data[-1522,]
reg<- read.csv('Auction_regist.csv',fileEncoding="UTF-8")
rent<- read.csv('Auction_rent.csv',fileEncoding="UTF-8")
result<- read.csv('Auction_result.csv',fileEncoding="UTF-8")
test<- read.csv('Auction_master_test.csv',fileEncoding="UTF-8")
#<-----------------------------------------------------------------------------------------------
library(dplyr)

#층수가 다른 아파트가 존재. 수작업 필수 
library(stringr)
floor1<- str_extract(str_extract(data$addr_etc, '[0-9][0-9]층'), '[0-9][0-9]')
floor2<- str_extract(str_extract(data$addr_etc, '[0-9]층'), '[0-9]')

floor1[is.na(floor1)]<- floor2[is.na(floor1)]

idx<- which(!str_detect(data$addr_etc, '층'))

data$Current_floor<- floor1
data[idx,'Current_floor'] <- c(13,4,14,4)
data$Current_floor<- as.numeric(data$Current_floor)


#test
floor1<- str_extract(str_extract(test$addr_etc, '[0-9][0-9]층'), '[0-9][0-9]')
floor2<- str_extract(str_extract(test$addr_etc, '[0-9]층'), '[0-9]')

floor1[is.na(floor1)]<- floor2[is.na(floor1)]

idx<- which(!str_detect(test$addr_etc, '층'))

test$Current_floor<- floor1
test[idx,'Current_floor'] <- c(3,13,6,1)
test$Current_floor<- as.numeric(test$Current_floor)


#Claim_price 0인 데이터 수정
idx<- which(data$Claim_price <= 1000)
data[idx,'Claim_price'] <- mean(data[-idx,'Claim_price'])

idx<- which(test$Claim_price <= 1000)
test[idx,'Claim_price'] <- mean(test[-idx,'Claim_price'])

#Auction_count
#facor형식으로 처리 
plot(log(data$Total_appraisal_price), log(data$Hammer_price)
     ,col = data$Auction_count     )
abline(a=0,b=1, col=2, lwd=3)

data$Auction_count<- ifelse(data$Auction_count > 2 , 3, data$Auction_count)
data$Auction_count<- as.factor(data$Auction_count)

test$Auction_count<- ifelse(test$Auction_count > 2 , 3, test$Auction_count)
test$Auction_count<- as.factor(test$Auction_count)

#Total_land_gross_area
hist(data$Total_land_gross_area, breaks=1000)

plot(data$Total_land_gross_area, data$Hammer_price) #극단값을 제거하지 말고, 평균으로 대체 
idx<- which.max(data$Total_land_gross_area)
data[idx,'Total_land_gross_area']<- mean(data[-idx, 'Total_land_gross_area'])

idx<- which(data$Total_land_gross_area <= 1)
data[idx,'Total_land_gross_area']<- mean(data[-idx,'Total_land_gross_area'])

idx<- which(test$Total_land_gross_area <= 1)
test[idx,'Total_land_gross_area']<- mean(test[-idx,'Total_land_gross_area'])

#Total_land_auction_area
idx<- which(data$Total_land_auction_area <= 1)
data[idx,'Total_land_auction_area']<- mean(data[-idx,'Total_land_auction_area'])

idx<- which(test$Total_land_auction_area <= 1)
test[idx,'Total_land_auction_area']<- mean(test[-idx,'Total_land_auction_area'])

#Total_building_area
idx<- which(data$Total_building_area <= 1)
data[idx,'Total_building_area']<- mean(data[-idx,'Total_building_area'])

idx<- which(test$Total_building_area <= 1)
test[idx,'Total_building_area']<- mean(test[-idx,'Total_building_area'])


#reg추가
temp<- reg%>%group_by(Auction_key)%>%summarize(Regist_price = max(Regist_price))#, Rent_monthly_price = max(Rent_monthly_price)

idx <-which(temp$Regist_price == 0)
temp<- as.data.frame(temp)
temp[idx,'Regist_price']<- mean(temp[-idx,'Regist_price'])

data<- data%>%left_join(temp, by= 'Auction_key')
data[is.na(data)]<- mean(data$Regist_price, na.rm=T)
test<- test%>%left_join(temp, by= 'Auction_key')
test[is.na(test)]<- mean(test$Regist_price, na.rm=T)


#rent추가
temp<- rent%>%group_by(Auctiuon_key)%>%summarize(Rent_deposit = max(Rent_deposit))#, Rent_monthly_price = max(Rent_monthly_price)
colnames(temp)[1]<- 'Auction_key'

idx <-which(temp$Rent_deposit == 0)
temp<- as.data.frame(temp)
temp[idx,'Rent_deposit']<- mean(temp[-idx,'Rent_deposit'])

data<- data%>%left_join(temp, by= 'Auction_key')
data[is.na(data)]<- mean(data$Rent_deposit, na.rm=T)

test<- test%>%left_join(temp, by= 'Auction_key')
test[is.na(test)]<- mean(test$Rent_deposit, na.rm=T)


gang_nam <- c('강서구', '양천구', '구로구','영등포구','금천구','관악구','동작구','서초구','강남구','송파구','강동구')
gang_buk <- c('마포구','용산구','성동구','광진구','서대문구','중구','동대문구','중랑구','종로구','은평구','성북구','강북구','도봉구','노원구')

data$han_river<- 0
for(i in 1:nrow(data)){
  
  if(data[i,'addr_si'] %in% gang_nam){
    data[i,'han_river'] <- '강남'
  }else if(data[i,'addr_si'] %in% gang_buk){
    data[i,'han_river'] <- '강북'
  }else{
    data[i,'han_river'] <- '부산'
  }
}
data$han_river <- as.factor(data$han_river)

test$han_river<- 0
for(i in 1:nrow(test)){
  
  if(test[i,'addr_si'] %in% gang_nam){
    test[i,'han_river'] <- '강남'
  }else if(test[i,'addr_si'] %in% gang_buk){
    test[i,'han_river'] <- '강북'
  }else{
    test[i,'han_river'] <- '부산'
  }
}
test$han_river <- as.factor(test$han_river)




#유찰 0, 1, 2회이상
data$Auction_miscarriage_count<- ifelse(data$Auction_miscarriage_count >= 2 , 2, data$Auction_miscarriage_count)
data$Auction_miscarriage_count<- as.factor(data$Auction_miscarriage_count)

test$Auction_miscarriage_count<- ifelse(test$Auction_miscarriage_count >= 2 , 2, test$Auction_miscarriage_count)
test$Auction_miscarriage_count<- as.factor(test$Auction_miscarriage_count)

# data$year_construct<- as.numeric(as.Date(data$Preserve_regist_date)- as.Date('2018-01-01'))
# test$year_construct<- as.numeric(as.Date(test$Preserve_regist_date)- as.Date('2018-01-01'))

#타겟 변수 로그 화
data$Hammer_price <- log(data$Hammer_price)
#새로운 변수 추가
data$appraisal_div_Min<- log(data$Total_appraisal_price) - log(data$Minimum_sales_price)
data$appraisal_mul_Min<- log(data$Total_appraisal_price) + log(data$Minimum_sales_price)
data$Total_building_auction_area <-log(data$Total_building_auction_area)

test$appraisal_div_Min<- log(test$Total_appraisal_price) - log(test$Minimum_sales_price)
test$appraisal_mul_Min<- log(test$Total_appraisal_price) + log(test$Minimum_sales_price)
test$Total_building_auction_area <-log(test$Total_building_auction_area)

data$date_diff<- as.numeric(as.Date(data$Final_auction_date) - as.Date(data$Appraisal_date))
test$date_diff<- as.numeric(as.Date(test$Final_auction_date) - as.Date(test$Appraisal_date))

#파생변수 생성
plot(data$Total_land_auction_area * data$Current_floor / data$Total_floor, log(data$Hammer_price))
plot(test$Total_land_auction_area *  test$Current_floor / test$Total_floor, log(test$Total_appraisal_price))

data$masaka<- data$Total_land_auction_area * data$Current_floor / data$Total_floor
test$masaka<- test$Total_land_auction_area *  test$Current_floor / test$Total_floor

#data에서 필요없는 정보들 지우기
data<-subset(data, select = -c(Auction_key,Auction_class,Close_result,Final_result,addr_li,addr_bunji1, addr_bunji2,road_bunji1, road_bunji2,addr_etc,road_name
                               ,Appraisal_company,Creditor,Specific,addr_san,addr_dong,addr_si,Appraisal_date
                               ,First_auction_date,Close_date,Preserve_regist_date,Total_land_real_area
                               ,Final_auction_date,Share_auction_YorN,Bid_class,Share_auction_YorN,Apartment_usage
                               ,point.y, point.x,addr_do))
#
test<-subset(test, select = -c(Auction_key,Auction_class,Close_result,Final_result,addr_li,addr_bunji1, addr_bunji2,road_bunji1, road_bunji2,addr_etc,road_name
                               ,Appraisal_company,Creditor,Specific,addr_san,addr_dong,addr_si,Appraisal_date
                               ,First_auction_date,Close_date,Preserve_regist_date,Total_land_real_area
                               ,Final_auction_date,Share_auction_YorN,Bid_class,Share_auction_YorN,Apartment_usage
                               ,point.y, point.x,addr_do))

train<- data

temp_train<- subset(train, select= -c(Hammer_price,appraisal_div_Min))
temp_test<- subset(test, select= -c(Hammer_price,appraisal_div_Min))
#box-cox Transform 나중에 추가하기.
library(car)
for(i in seq_len(ncol(temp_train))){
  if(sapply(temp_train, class)[i] %in% c('numeric','integer')){
    PT<- powerTransform(temp_train[,i])
    if(PT$lambda <= 0){
      temp_train[,i]<- log(temp_train[,i])
    }else{
      temp_train[,i]<- log(temp_train[,i], base = PT$lambda)
      temp_test[,i]<- log(temp_test[,i], base = PT$lambda)
    }
  }
}
train_<- cbind(temp_train, train$appraisal_div_Min); colnames(train_)[ncol(train_)] = 'appraisal_div_Min'
train<- cbind(train_, train$Hammer_price); colnames(train)[ncol(train)] = 'Hammer_price'
test_<- cbind(temp_test, test$appraisal_div_Min); colnames(test_)[ncol(test_)] = 'appraisal_div_Min'
test<- cbind(test_, test$Hammer_price); colnames(test)[ncol(test)] = 'Hammer_price'

# dummify the data
library(caret)
factor_idx<-which(sapply(train, class) == 'factor')
dmy <- dummyVars(" ~ .", data = train[,factor_idx])
trsf <- data.frame(predict(dmy, newdata = train[,factor_idx]))
train<- train[,-factor_idx]
train<- cbind(train, trsf)

factor_idx<-which(sapply(test, class) == 'factor')
dmy <- dummyVars(" ~ .", data = test[,factor_idx])
trsf <- data.frame(predict(dmy, newdata = test[,factor_idx]))
test<- test[,-factor_idx]
test<- cbind(test, trsf)

train_mat <- model.matrix(data = train, Hammer_price ~ .)[, -1]
test_mat <- model.matrix(data = test, Hammer_price ~ .)[, -1]


library(Boruta)

set.seed(7)
bor.result <- Boruta(train_mat, train$Hammer_price, doTrace = 1)
getSelectedAttributes(bor.result)
plot(bor.result)

bor.result$finalDecision

# Turn off scientific notation
options(scipen=10)

#======================================================================================================
# Load packages
library(data.table)
library(caret)
library(mltools)  # For generating CV folds and one-hot-encoding
library(glmnet)
#train<- data.table(train)
#test<- data.table(test)

#k_fold<- folds(train$Hammer_price, nfolds=5, stratified=TRUE, seed=2018)

# Build folds for cross validation and stacking
#train$FoldID <- k_fold
#======================================================================================================
# Ridge
#
# Do a grid search for k = 1, 2, ... 30 by cross validating model using folds 1-5
# I.e. [test=f1, train=(f2, f3, f4, f5)], [test=f2, train=(f1, f3, f4, f5)], ...
RMSE<- function(y=train$Hammer_price, y_pred){
  sqrt(mean(( (exp(y) - exp(y_pred))^2 )))
}


#1. KNN
tr <- trainControl( method = 'cv', number = 10)

library(e1071)
library(class)
knn.Model <- train(
  Hammer_price ~ ., 
  data = train,
  method = 'knn',
  metric = 'RMSE',
  tuneGrid = data.frame(.k = 1:40),
  trControl = tr)


RMSE(y_pred = predict(knn.Model))

knn <- as.numeric(as.character(knn(train_mat, test_mat, train$Hammer_price, k = knn.Model$bestTune)))
  
#2. randomForest
library(ranger)

set.seed(7)
rf <- ranger(Hammer_price ~ .,
             data = train,
             num.trees = 2000)
randomForest <- predict(rf$forest, test)
randomForest <- randomForest$predictions
RMSE(y_pred = predict(rf$forest, train))

#3.  Regression with L1 Regularization
library(glmnet)

set.seed(7)
logit_L1.cv <- cv.glmnet(x = train_mat, y = train$Hammer_price,alpha = 1)
logit.L1 <- predict(logit_L1.cv, test_mat, s = logit_L1.cv$lambda.min, type = 'response')
RMSE(y_pred = predict(logit_L1.cv, train_mat, s = logit_L1.cv$lambda.min, type = 'response'))

#4. Regression with L2 Regularization
set.seed(7)
logit_L2.cv <- cv.glmnet(x = subset(train,select=-Hammer_price), y = train$Hammer_price, alpha = 0)
logit.L2 <- predict(logit_L2.cv, test_mat, s = logit_L2.cv$lambda.min, type = 'response')
RMSE(y_pred = predict(logit_L2.cv, train_mat, s = logit_L2.cv$lambda.min, type = 'response'))
#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(xgb)

write.csv(final, 'xgb_20181128.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------

#5.Xgboost
library(xgboost)

xgb.grid <- expand.grid(nrounds = c(180, 200, 220),
                        eta = c(0.01, 0.03, 0.05),
                        max_depth = c(3, 5, 7),
                        gamma = 0,
                        colsample_bytree = c(0.6, 0.8, 1),
                        min_child_weight = c(0.8, 0.9, 1),
                        subsample = 1
)

set.seed(7)

xgbTrain <- train(x = train_mat,
                  y = train$Hammer_price,
                  trControl = tr,
                  tuneGrid = xgb.grid,
                  method = "xgbTree"
)

xgb.dacon <- xgboost(params = xgbTrain$bestTune,
                       nrounds = xgbTrain$bestTune[1, 1],
                       data = train_mat,
                       label = train$Hammer_price,
                       verbose = FALSE)
xgb <- predict(xgb.dacon, test_mat)
RMSE(y_pred = predict(xgb.dacon, train_mat))
#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(xgb)

write.csv(final, 'xgb_20181128.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------

cat("Accuracy(kNN):", RMSE(y_pred = predict(knn.Model, train_mat)),
    "\nAccuracy(Random Forest):", RMSE(y_pred = predict(rf,train_mat)$predictions),
    "\nAccuracy(Logistic L1):", RMSE(y_pred = predict(logit_L1.cv, train_mat)),
    "\nAccuracy(Logistic L2):", RMSE(y_pred = predict(logit_L2.cv, train_mat)),
    "\nAccuracy(Xgboost):",RMSE(y_pred = predict(xgb.dacon, train_mat)) )

#cor확인
pred_data_frame<- data.frame(kNN = predict(knn.Model, test_mat)
                             , rf= randomForest
                             , L1 = as.vector(logit.L1)
                             , L2 = as.vector(logit.L2)
                             , xgb = predict(xgb.dacon, test_mat))
library(psych)
pairs.panels(pred_data_frame)

#data수가 맞지 않아서 6-fold로 진행
folded_train <- train %>%
  mutate(foldID = rep(1:6, each = nrow(train)/6)) %>%
  select(foldID, everything())

folded_mat <- model.matrix(data = folded_train, Hammer_price ~ .)[, -1]



### Initiating ###
k_NN <- NULL
rf <- NULL
logitL1 <- NULL
logitL2 <- NULL
XGB <- NULL

for(targetFold in 1:6){
  trainFold <- filter(folded_train, foldID != targetFold) %>% select(-foldID)
  trainSurvived <- trainFold$Hammer_price
  testFold <- filter(folded_train, foldID == targetFold) %>% select(-foldID)
  testSurvived <- testFold$Hammer_price
  
  fold_train_mat <- folded_mat[folded_mat[, 1] != targetFold, -1]
  fold_test_mat <- folded_mat[folded_mat[, 1] == targetFold, -1]
  
  ### kNN ###
  temp <- knn(fold_train_mat, fold_test_mat, trainSurvived, k = knn.Model$bestTune)
  k_NN <- c(k_NN, temp)
  

  ### Random Forest ###
  set.seed(7)
  RF <- ranger(Hammer_price ~ ., data = trainFold, num.trees = 2000)
  temp <- predict(RF$forest, testFold)
  temp <- temp$predictions
  rf <- c(temp, rf)
  
  ### Logistic Regression with L1 Regularization ###
  logit_L1 <- glmnet(x = fold_train_mat, y = trainSurvived, alpha = 1, lambda = logit_L1.cv$lambda.min)
  temp <- predict(logit_L1, fold_test_mat, type = 'response')
  temp <- as.vector(temp)
  logitL1 <- c(temp, logitL1)
  
  ### Logistic Regression with L2 Regularization ###
  logit_L2 <- glmnet(x = fold_train_mat, y = trainSurvived, alpha = 0, lambda = logit_L2.cv$lambda.min)
  temp <- predict(logit_L2, fold_test_mat, type = 'response')
  temp <- as.vector(temp)
  logitL2 <- c(temp, logitL2)
  
  ### XGBoost ###
  xgb.dacon <- xgboost(params = xgbTrain$bestTune,
                         nrounds = xgbTrain$bestTune[1, 1],
                         data = fold_train_mat,
                         label = trainSurvived,
                         verbose = FALSE)
  temp <- predict(xgb.dacon, fold_test_mat)
  XGB <- c(temp, XGB)
}



meta_train <- cbind(train_mat,
                    #kNN = k_NN,
                    RF = rf,
                    LogitL1 = logitL1,
                    LogitL2 = logitL2,
                    XGB = XGB)

meta_test <- cbind(test_mat,
                   #kNN = knn,
                   RF = randomForest,
                   LogitL1 = as.vector(logit.L1),
                   LogitL2 = as.vector(logit.L2),
                   XGB = xgb)


#Model Stacking

xgb.grid <- expand.grid(nrounds = c(180, 200, 220),
                        eta = c(0.01, 0.03, 0.05),
                        max_depth = c(3, 5, 7),
                        gamma = 0,
                        colsample_bytree = c(0.6, 0.8, 1),
                        min_child_weight = c(0.8, 0.9, 1),
                        subsample = 1
)

set.seed(7)

xgbTrain <- train(x =  meta_train,
                  y = train$Hammer_price,
                  trControl = tr,
                  tuneGrid = xgb.grid,
                  method = "xgbTree"
)

meta.xgb.cv <- xgboost(params = xgbTrain$bestTune,
                     nrounds = xgbTrain$bestTune[1, 1],
                     data = meta_train,
                     label = train$Hammer_price,
                     verbose = FALSE)

meta.pred <- predict(meta.xgb.cv, meta_test)

#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(meta.pred)

write.csv(final, 'meta_xgb_20181128.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------



#제출1: RF
stack_level1_test<- data.frame(lm = predict(fit1,test), rf= predict(rf.Model,test), knn = predict(knn.Model,test))
stack_level1_test<- cbind(test, stack_level1_test)

stack_level2_test<- data.frame(level1_RF=predict(rf.Model2,stack_level1_test), level1_gbm= predict(gbm.Model,stack_level1_test) )
stack_level2_test<- cbind(stack_level1_test,stack_level2_test)

pred<- predict(rf.Model2,stack_level2_test)

pairs.panels(stack_level2_test[,25:29])
#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(pred)

write.csv(final, 'stack_RF_20181128.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------


RidgeCV <- list()
RidgeCV[["Features"]] <- setdiff(colnames(train), c('Hammer_price','FoldID'))
RidgeCV[["ParamGrid"]] <- CJ(.lambda= seq(0, 0.025,length= 100) , Score=NA_real_)
RidgeCV[["BestScore"]] <- 100 # Sum of square error, low is best

# Loop through each set of parameters
for(i in seq_len(nrow(RidgeCV[["ParamGrid"]]))){
  
  # Get the ith set of parameters
  params <- RidgeCV[["ParamGrid"]][i]
  
  # Build an empty vector to store scores from each train/test fold
  scores <- numeric()
  
  # Build an empty list to store predictions from each train/test fold
  predsList <- list()
  
  # Loop through each test fold, fit model to training folds and make predictions on test fold
  for(foldID in 1:5){
    
    # Build the train/test folds
    testFold <- train[J(FoldID=foldID), on="FoldID"]
    trainFolds <- train[!J(FoldID=foldID), on="FoldID"]  # Exclude fold i from trainFolds
    
    # Make X, Y
    y <- trainFolds$Hammer_price
    x <- trainFolds[, RidgeCV$Features, with=FALSE]%>% data.matrix()
    #x <- apply(x,2,function(x)scale(x,center = FALSE))
    
    # Train the model & make predictions
    ridge <- glmnet(x,y , alpha = 0, lambda = params$.lambda)
    
    test_x<- testFold[, RidgeCV$Features, with=FALSE]%>%data.matrix()
    #test_x <- apply(test_x,2,function(x)scale(x,center = FALSE))
    testFold[, Pred := predict(ridge, test_x)]
    predsList <- c(predsList, list(testFold[, list(FoldID, Pred)]))
    
    # Evaluate predictions (Square error) and append score to scores vector
    score <- sum( (testFold$Hammer_price - testFold$Pred)^2 )
    scores <- c(scores, score)
  }
  # Measure the overall score. If best, tell RidgeCV
  score <- mean(scores)
  
  # Insert the score into ParamGrid
  RidgeCV[["ParamGrid"]][i, Score := score][]
  print(paste("Params:", paste(colnames(RidgeCV[["ParamGrid"]][i]), RidgeCV[["ParamGrid"]][i], collapse = " | ")))
  
  if(score < RidgeCV[["BestScore"]]){
    RidgeCV[["BestScores"]] <- scores
    RidgeCV[["BestScore"]] <- score
    RidgeCV[["BestParams"]] <- RidgeCV[["ParamGrid"]][i]
    RidgeCV[["BestPreds"]] <- rbindlist(predsList)
  }
}


# Plot the score for each (cost, type) pairs
RidgeCV[["ParamGrid"]]
ggplot(RidgeCV[["ParamGrid"]], aes(x= .lambda, y=Score))+geom_line()+geom_point()

#======================================================================================================
# Lasso
LassoCV <- list()
LassoCV[["Features"]] <- setdiff(colnames(train), c('Hammer_price','FoldID'))
LassoCV[["ParamGrid"]] <- CJ(.lambda= seq(0, 0.0015,length= 100) , Score=NA_real_)
LassoCV[["BestScore"]] <- 100 # Sum of square error, low is best

# Loop through each set of parameters
for(i in seq_len(nrow(LassoCV[["ParamGrid"]]))){
  
  # Get the ith set of parameters
  params <- LassoCV[["ParamGrid"]][i]
  
  # Build an empty vector to store scores from each train/test fold
  scores <- numeric()
  
  # Build an empty list to store predictions from each train/test fold
  predsList <- list()
  
  # Loop through each test fold, fit model to training folds and make predictions on test fold
  for(foldID in 1:5){
    
    # Build the train/test folds
    testFold <- train[J(FoldID=foldID), on="FoldID"]
    trainFolds <- train[!J(FoldID=foldID), on="FoldID"]  # Exclude fold i from trainFolds
    
    # Make X, Y
    y <- trainFolds$Hammer_price
    x <- trainFolds[, LassoCV$Features, with=FALSE]%>% data.matrix()
    #x <- apply(x,2,function(x)scale(x,center = FALSE))
    
    # Train the model & make predictions
    ridge <- glmnet(x,y , alpha = 1, lambda = params$.lambda)
    
    test_x<- testFold[, LassoCV$Features, with=FALSE]%>%data.matrix()
    #test_x <- apply(test_x,2,function(x)scale(x,center = FALSE))
    testFold[, Pred := predict(ridge, test_x)]
    predsList <- c(predsList, list(testFold[, list(FoldID, Pred)]))
    
    # Evaluate predictions (Square error) and append score to scores vector
    score <- sum( (testFold$Hammer_price - testFold$Pred)^2 )
    scores <- c(scores, score)
  }
  # Measure the overall score. If best, tell LassoCV
  score <- mean(scores)
  
  # Insert the score into ParamGrid
  LassoCV[["ParamGrid"]][i, Score := score][]
  print(paste("Params:", paste(colnames(LassoCV[["ParamGrid"]][i]), LassoCV[["ParamGrid"]][i], collapse = " | ")))
  
  if(score < LassoCV[["BestScore"]]){
    LassoCV[["BestScores"]] <- scores
    LassoCV[["BestScore"]] <- score
    LassoCV[["BestParams"]] <- LassoCV[["ParamGrid"]][i]
    LassoCV[["BestPreds"]] <- rbindlist(predsList)
  }
}

# Plot the score for each (cost, type) pairs
ggplot(LassoCV[["ParamGrid"]], aes(x= .lambda, y=Score))+geom_line()+geom_point()


#======================================================================================================
# Ensemble Ridge, Lasso using GBM Regression
library(gbm)
# Extract predictions
metas.ridge <- RidgeCV[["BestPreds"]]
metas.lasso <- LassoCV[["BestPreds"]]

# Insert regular predictions into train
train[metas.ridge, Meta.ridge := Pred, on="FoldID"]
train[metas.lasso, Meta.lasso := Pred, on="FoldID"]

# Cross Validation
lrCV <- list()
lrCV[["Features"]] <- setdiff(colnames(train), c("FoldID"))#, "Hammer_price", "metas.ridge", "metas.lasso"
lrCV[["ParamGrid"]] <- CJ(interaction.depth = c(1, 5, 9)
                          ,n.trees = c(500,1000,1500)
                          ,shrinkage = c(0.01,0.05)
                          ,n.minobsinnode = c(5,3,1)
                          , Score=NA_real_)
lrCV[["BestScore"]] <- 100


# Loop through each set of parameters
for(i in seq_len(nrow(lrCV[["ParamGrid"]]))){
  
  # Get the ith set of parameters
  params <- lrCV[["ParamGrid"]][i]
  
  # Build an empty vector to store scores from each train/test fold
  scores <- numeric()
  
  # Build an empty list to store predictions from each train/test fold
  predsList <- list()
  
  # Loop through each test fold, fit model to training folds and make predictions on test fold
  for(foldID in 1:5){
    
    # Build the train/test folds
    testFold <- train[J(FoldID=foldID), on="FoldID"]
    trainFolds <- train[!J(FoldID=foldID), on="FoldID"]  # Exclude fold i from trainFolds
    
    # Train the model & make predictions
    GBM <- gbm( Hammer_price ~ . ,data= subset(trainFolds, select = lrCV[["Features"]] )
                ,distribution = "gaussian"
                ,interaction.depth = lrCV[["ParamGrid"]][i]$interaction.depth
                ,n.trees = lrCV[["ParamGrid"]][i]$n.trees
                ,shrinkage = lrCV[["ParamGrid"]][i]$shrinkage
                ,n.minobsinnode = lrCV[["ParamGrid"]][i]$n.minobsinnode
                )
    testFold[, Pred := predict(GBM
                               , newdata = testFold[, lrCV$Features, with=FALSE]
                               , n.tree= lrCV[["ParamGrid"]][i]$n.trees)]
    predsList <- c(predsList, list(testFold[, list(FoldID, Pred)]))
    
    # Evaluate predictions (Square error) and append score to scores vector
    score <- sum( (testFold$Hammer_price - testFold$Pred)^2 )
    scores <- c(scores, score)
  }
  
  # Measure the overall score. If best, tell lrCV
  score <- mean(scores)
  
  # Insert the score into ParamGrid
  lrCV[["ParamGrid"]][i, Score := score][]
  print(paste("Params:", paste(colnames(lrCV[["ParamGrid"]][i]), lrCV[["ParamGrid"]][i], collapse = " | ")))
  
  if(score < lrCV[["BestScore"]]){
    lrCV[["BestScores"]] <- scores
    lrCV[["BestScore"]] <- score
    lrCV[["BestParams"]] <- lrCV[["ParamGrid"]][i]
    lrCV[["BestPreds"]] <- rbindlist(predsList)
  }
}

ggplot(lrCV[["ParamGrid"]], aes(x= interaction.depth, y=Score,color=factor(n.minobsinnode)))+geom_line()+geom_point()


RidgeCV[["BestParams"]]
LassoCV[["BestParams"]]
lrCV[["BestParams"]]


# Make X, Y
y <- train$Hammer_price
x <- subset(train, select = -Hammer_price)%>%data.matrix()
# 제출용
ridge <- glmnet(x,y , alpha = 0, lambda = RidgeCV[["BestParams"]]$.lambda)

Meta.ridge = predict(ridge, s=RidgeCV[["BestParams"]]$.lambda
               ,type="response", newx = subset(test, select=-Hammer_price)%>% data.matrix()) # coefficients

#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(pred)

write.csv(final, 'Ridge_20181126.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------


# 제출용
lasso <- glmnet(x,y , alpha = 1, lambda = LassoCV[["BestParams"]]$.lambda)

Meta.lasso = predict(lasso, s=LassoCV[["BestParams"]]$.lambda
               ,type="response", newx = subset(test, select=-Hammer_price)%>% data.matrix()) # coefficients

#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(pred)

write.csv(final, 'Lasso_20181126.csv',fileEncoding="UTF-8", row.names = FALSE)
#<------------------------------------------------------------------------
GBM <- gbm( Hammer_price ~ . ,data= subset(train, select = lrCV[["Features"]] )
            ,distribution = "gaussian"
            ,interaction.depth = lrCV[["BestParams"]]$interaction.depth
            ,n.trees = lrCV[["BestParams"]]$n.trees
            ,shrinkage = lrCV[["BestParams"]]$shrinkage
            ,n.minobsinnode = lrCV[["BestParams"]]$n.minobsinnode
)
test<- cbind(test, Meta.ridge, Meta.lasso)
colnames(test)[23:24]<- c('Meta.ridge', 'Meta.lasso')
pred_gbm<- predict(GBM , newdata = test , n.tree= lrCV[["BestParams"]]$n.trees)
#<------------------------------------------------------------------------
final<- read.csv('Auction_submission.csv',fileEncoding="UTF-8")

final[,2]<-exp(pred_gbm)

write.csv(final, 'GBM_20181126.csv',fileEncoding="UTF-8", row.names = FALSE)

train2<- train[-1113,]
fit<- lm(Hammer_price ~ . , data = train)
library(MASS)
fit1 <- lm(Hammer_price ~ ., train2)
fit2 <- lm(Hammer_price ~ 1, train2)
back<- stepAIC(fit1,direction="backward")
forward<- stepAIC(fit2,direction="forward",scope=list(upper=fit1,lower=fit2))
both<- stepAIC(fit2,direction="both",scope=list(upper=fit1,lower=fit2))
par(mfrow=c(2,2))
plot(back)
plot(forward)


summary(back)
