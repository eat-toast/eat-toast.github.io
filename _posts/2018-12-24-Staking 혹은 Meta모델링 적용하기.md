---
title: "stacking"
date: 2018-12-24 00:33:28 -0400
categories: study
---

# 1. stacking에 관하여

## 1.1 stacking이란?
[Kaggle][kaggle-home] 상위권 랭커들이 사용하는 알고리즘으로 여러 모델들의 장점을 하나로 합해 새로운 모델을 만드는 방법이다.

<img src="/resources/staking_structure.PNG" width="600">
    (간단한 stacking의 구조)


## 1.2 왜?
 기본적으로 다음과 같은 가정을 한다.

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
```


[kaggle-home]: https://www.kaggle.com/
[interview]: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
