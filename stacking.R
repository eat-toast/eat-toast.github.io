# 자료 출처:
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

options(scipen=10)

library(data.table)
library(caret)
library(dplyr)
library(e1071)
library(class)
library(randomForest)


setwd("D:/MLPB-master/Problems/Classify Dart Throwers")
# data load
train <- fread("_Data/train.csv")
test <- fread("_Data/test.csv")

train$Competitor<- as.factor(train$Competitor)
test$Competitor<- as.factor(test$Competitor)

# train, test간 데이터는 일정한지 비교
par(mfrow=c(1,2))
plot(train$XCoord, train$YCoord, col=train$Competitor)
plot(test$XCoord, test$YCoord, col=test$Competitor)
par(mfrow=c(1,1))

# X를 Input으로 받는 알고리즘을 위해, model.matrix() 함수를 사용한다.
train_mat <- model.matrix(data = train, Competitor ~ XCoord+ YCoord)[, -1]
test_mat <- model.matrix(data = test, Competitor ~ XCoord+ YCoord)[, -1]


#### Construct Base Model ###
# 생성한 데이터를 이용해서 Level 0의 기본 모델들을 생성한다.
# 사용할 모델은 총 4개로 다음과 같다.
 
# k Nearest Neighbor
# Support Vector Machine with Polynomial Kernel
# Support Vector Machine with Radial Kernel
# Random Forest


# 각각의 모델들은 모두 10-fold Cross Validation을 이용해서 Parameter tuning을 한다.

#Base Model 1: kNN

set.seed(7)
knn.cv <- tune.knn(x = train_mat, y = factor(train$Competitor), k = seq(1, 40, by = 2),
                   tunecontrol = tune.control(sampling = "cross"), cross = 10)
knn <- knn(train_mat, test_mat, factor(train$Competitor), k = knn.cv$best.parameters[, 1])

# Base Model 2: SVM with Polynomial Kernel

set.seed(7)
poly.svm.cv <- tune.svm(x = train_mat, y = factor(train$Competitor), kernel = "polynomial",
                        degree = c(2, 3, 4), coef0 = c(0.1, 0.5, 1, 2),
                        cost = c(0.001, 0.01, 0.1, 1, 3, 5),
                        tunecontrol = tune.control(sampling = "cross"))
poly_svm <- predict(poly.svm.cv$best.model, test_mat)

# Base Model 3: SVM with Radial Kernel

set.seed(7)
radial.svm.cv <- tune.svm(x = train_mat, y = factor(train$Competitor), kernel = "radial",
                          gamma = c(0.1, 0.5, 1, 2, 3), coef0 = c(0.1, 0.5, 1, 2),
                          cost = c(0.001, 0.01, 0.1, 1, 3, 5),
                          tunecontrol = tune.control(sampling = "cross"))
radial_svm <- predict(radial.svm.cv$best.model, test_mat)


# Base Model 4: Random Forest
# 랜덤 포레스트의 경우 변수가 현재 2가지 밖에 없어서 전부 사용하기로 한다.
set.seed(7)

rf <- randomForest(train_mat, train$Competitor, mtry = 2)
random_Forest <- predict(rf, test_mat)


ACC<- function(Y, Y_hat){
  tab<- table(Y,Y_hat)
  acc = sum(diag(tab))/sum(tab)
  acc = round(acc,4)
  return(acc)
}


cat("Accuracy(kNN):", ACC(Y = test$Competitor, Y_hat = knn),
    "\nAccuracy(SVM, Polynomial):", ACC(Y = test$Competitor, Y_hat = poly_svm),
    "\nAccuracy(SVM, Radial):", ACC(Y = test$Competitor, Y_hat = radial_svm),
    "\nAccuracy(Random Forest):", ACC(Y = test$Competitor, Y_hat = random_Forest)
    )

### 시각화 ###
x<- subset(train%>%data.frame, select = c(XCoord,YCoord,Competitor))

cl <- x[,'Competitor']
data <- x[,1:2]
k <- length(unique(cl))

plot(train_mat, col = as.integer(cl), pch = as.integer(cl))


# make grid
r <- sapply(train[,c(2,3)], range)
resolution = 100
xs <- seq(r[1,1], r[2,1], length.out = resolution)
ys <- seq(r[1,2], r[2,2], length.out = resolution)
g <- cbind(rep(xs, each=resolution), rep(ys, time = resolution))
colnames(g) <- colnames(r)
g <- as.data.frame(g)
folded_g <- g %>% mutate(foldID = rep(1:5, each = nrow(g)/5)) %>% select(foldID, everything())

### 출처: http://michael.hahsler.net/SMU/EMIS7332/R/viz_classifier.html
plot(train_mat, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, main='KNN(70%)')
p_knn <- knn(train_mat, g, factor(train$Competitor), k = knn.cv$best.parameters[, 1])
z_knn <- matrix(as.integer(p_knn), nrow = resolution, byrow = TRUE)
contour(xs, ys, z_knn, add = TRUE, drawlabels = FALSE, lwd = 2, levels = (1:(k-1))+.5)

plot(train_mat, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, main='SVM_poly(86%)')
p_svm <- predict(poly.svm.cv$best.model, g)
z_svm <- matrix(as.integer(p_svm), nrow = resolution, byrow = TRUE)
contour(xs, ys, z_svm, add = TRUE, drawlabels = FALSE, lwd = 2, levels = (1:(k-1))+.5)

plot(train_mat, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, main='SVM_radial(75%)')
p_svm <- predict(radial.svm.cv$best.model, g)
z_svm <- matrix(as.integer(p_svm), nrow = resolution, byrow = TRUE)
contour(xs, ys, z_svm, add = TRUE, drawlabels = FALSE, lwd = 2, levels = (1:(k-1))+.5)

plot(train_mat, col = as.integer(cl)+1L, pch = as.integer(cl)+1L, main='RF(86%)')
p_rf <- predict(rf, g)
z_rf <- matrix(as.integer(p_rf), nrow = resolution, byrow = TRUE)
contour(xs, ys, z_rf, add = TRUE, drawlabels = FALSE, lwd = 2, levels = (1:(k-1))+.5)


### Stacking ###
k_NN <- NULL
svm.poly <- NULL
svm.radial <- NULL
rf <- NULL

meta<- NULL


folded_train <- train %>% mutate(foldID = rep(1:5, each = nrow(train)/5)) %>% select(foldID, everything())

folded_mat <- model.matrix(data = folded_train, Competitor ~ .)[, -1]


for(targetFold in 1:5){
  trainFold <- filter(folded_train, foldID != targetFold) %>% select(-foldID)%>% select(-ID)
  trainCompetitor <- trainFold$Competitor
  
  fold_train_mat <- folded_mat[folded_mat[,'foldID'] != targetFold, -c(1,2)]
  fold_test_mat <- folded_mat[folded_mat[,'foldID'] == targetFold, -c(1,2)]
  
  ### kNN ###
  temp <- knn(fold_train_mat, fold_test_mat, trainCompetitor, k = knn.cv$best.parameters[, 1])
  folded_train[ folded_train$foldID == targetFold, 'kNN']<- temp
  
  
  ### SVM with Polynomial Kernel ###
  poly <- svm(x = fold_train_mat, y = trainCompetitor, kernel = 'polynomial',
              cost = poly.svm.cv$best.parameters[1, "cost"],
              degree = poly.svm.cv$best.parameters[1, "degree"],
              coef0 = poly.svm.cv$best.parameters[1, "coef0"])
  temp <- predict(poly, fold_test_mat)
  folded_train[ folded_train$foldID == targetFold, 'svm_poly']<- temp
  
  
  ### SVM with Radial Basis Kernel ###
  radial <- svm(x = fold_train_mat, y = trainCompetitor, kernel = 'radial',
                cost = radial.svm.cv$best.parameters[1, "cost"],
                gamma = radial.svm.cv$best.parameters[1, "gamma"],
                coef0 = radial.svm.cv$best.parameters[1, "coef0"])
  temp <- predict(radial, fold_test_mat)
  folded_train[ folded_train$foldID == targetFold, 'svm_radial']<- temp
  
  
  ### Random Forest ###
  set.seed(7)
  RF <- randomForest(x = fold_train_mat, y= trainCompetitor)
  temp <- predict(RF, fold_test_mat)
  folded_train[ folded_train$foldID == targetFold, 'rf']<- temp
  
  ### 시각화 ###
  folded_g_test<- folded_g[folded_g$foldID == targetFold,c(2,3)]
  folded_g[ folded_g$foldID == targetFold, 'kNN']<- knn(fold_train_mat, folded_g_test, trainCompetitor, k = knn.cv$best.parameters[, 1])
  folded_g[ folded_g$foldID == targetFold, 'svm_poly']<- predict(poly, folded_g_test) 
  folded_g[ folded_g$foldID == targetFold, 'svm_radial']<- predict(radial, folded_g_test)
  folded_g[ folded_g$foldID == targetFold, 'rf']<- predict(RF, folded_g_test)
}

# testFold <- filter(folded_train, foldID == targetFold) %>% select(-foldID)%>% select(-ID)
# testSurvived <- testFold$Competitor

meta_test <- data.frame(test_mat,
                        kNN = knn,
                        svm_poly = poly_svm,
                        svm_radial = radial_svm,
                        rf = random_Forest )

set.seed(7)
meta.rf<- randomForest(x = subset(folded_train, select = -c(ID,foldID,Competitor ))
                                  , y = folded_train$Competitor) 
meta.randomForest <- predict(meta.rf, meta_test)

cat("Accuracy(Stacked Model):", ACC(test$Competitor, meta.randomForest) )

plot(test_mat, col = as.integer(cl)+1L, pch = as.integer(cl)+1L
     ,main= paste('Stacking RF :',  ACC(test$Competitor, meta.randomForest) ) )
p_meta_rf <- predict(meta.rf, subset(folded_g, select= -foldID))
z_meta_rf <- matrix(as.integer(p_meta_rf), nrow = resolution, byrow = TRUE)
contour(xs, ys, z_meta_rf, add = TRUE, drawlabels = FALSE, lwd = 2, levels = (1:(k-1))+.5)

