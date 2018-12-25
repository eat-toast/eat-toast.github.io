---
title: "stacking"
date: 2018-12-24 00:33:28 -0400
categories: study
---

# 1. stacking에 관하여

## 1.1 stacking이란?
[Kaggle][kaggle-home] 상위권 랭커들이 사용하는 알고리즘으로 여러 모델들의 장점을 하나로 합해 새로운 모델을 만드는 방법이다.

<img src="/resources/staking_structure.PNG" width="650">
    (간단한 stacking의 구조)


## 1.2 왜?
  다음과 같은 가정과 장단점이 존재한다.

### 1.2.1. 가정 

    1. 모든 모델은 `mistake`를 가지고 있다. --> 완벽한 모델은 존재하지 않는다.
       아무리 Xgboost, lightGBM 등 앙상블 계열 알고리즘이 뛰어나도 알고리즘 구조상 
       놓치는 부분이 존재한다는 것을 인정한다.
       
    2. 잘 맞추는 부분을 통합 한다면 더 잘 맞추게 될 것이다.

### 1.2.2. 장점
    1. 각 알고리즘의 좋은 부분을 습득 가능
    
### 1.2.3. 단점
    1. 연산량이 증가한다

***
<img src="/resources/staking_dart.PNG" width="600">

 [원본링크][interview] 위 사진은 1.2.1 가정 1번의 예시가 잘 담겨있다.
  완벽한 알고리즘은 존재하지 않으며 각각의 장점을 살려보자!

`KNN: 중앙에서 높은 정확도  &  SVM: 가장자리에서 높은 정확도`


# 2. Stacking 설명

## 2.1 Base Model 혹은 Level 0 모델

Base Model에 해당되는 알고리즘이 딱히 정해져 있지는 않다. 

다만, 위에서 본 다트 예제 처럼 특정한 부분의 정확도가 높다면 쓸만하다.

~~전체적인 정확도가 낮더라도!!!~~

Base Model을 구성할 때, 다음 2가지 단계를 거쳐야 한다.

    1. 알고리즘을 선정 (KNN, SVM 등)
    2. Hyper Parameter 튜닝



먼저 데이터를 불러온다.
데이터는 [이곳에서][MLPB] 받을 수 있다
```R
setwd("D:/MLPB-master/Problems/Classify Dart Throwers")
# data load
train <- fread("_Data/train.csv")
test <- fread("_Data/test.csv")

# 타겟 변수 factor로 변경
train$Competitor<- as.factor(train$Competitor)
test$Competitor<- as.factor(test$Competitor)
```

## 2.1.1 KNN
 어떤 K가 (1<=k<=40) 뛰어난지 알 수 없어 Cross Validation을 통해 k를 선택한다.
 
 ```R
 #Base Model 1: kNN

set.seed(7)
knn.cv <- tune.knn(x = train_mat, y = factor(train$Competitor), k = seq(1, 40, by = 2),
                   tunecontrol = tune.control(sampling = "cross"), cross = 10)
knn <- knn(train_mat, test_mat, factor(train$Competitor), k = knn.cv$best.parameters[, 1])

 ```

## 2.1.2 SVM - poly
 마찬가지로 SVM에 대해서도 진행한다.
 
```R
# Base Model 2: SVM with Polynomial Kernel

set.seed(7)
poly.svm.cv <- tune.svm(x = train_mat, y = factor(train$Competitor), kernel = "polynomial",
                        degree = c(2, 3, 4), coef0 = c(0.1, 0.5, 1, 2),
                        cost = c(0.001, 0.01, 0.1, 1, 3, 5),
                        tunecontrol = tune.control(sampling = "cross"))
poly_svm <- predict(poly.svm.cv$best.model, test_mat)
```

## 2.1.3 Base Model 결과

    Accuracy(kNN): 0.7027 
    Accuracy(SVM, Polynomial): 0.8649
    Accuracy(SVM, Radial): 0.7568 
    Accuracy(Random Forest): 0.7838
    (SVM-Radial과 Random Forest는 아래 코드 참조)


## 2.1.4 Base Model 시각화
<img src="/resources/staking_KNN.png" width="300">
<img src="/resources/staking_SVM_ploy.png" width="300">
<img src="/resources/staking_SVM_radial.png" width="300">
<img src="/resources/staking_RF.png" width="300">
## 1.3 실습

데이터는 [이곳에서][MLPB] 받을 수 있다


```R
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

# 타겟 변수 factor로 변경
train$Competitor<- as.factor(train$Competitor)
test$Competitor<- as.factor(test$Competitor)
```








[kaggle-home]: https://www.kaggle.com/
[interview]: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
[MLPB]: https://github.com/ben519/MLPB
