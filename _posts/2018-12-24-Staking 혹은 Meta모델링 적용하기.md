---
title: "stacking"
date: 2018-12-24 00:33:28 -0400
categories: study
---

# stacking이란?

[Kaggle][kaggle-home] 상위권 랭커들이 사용하는 알고리즘으로 여러 모델들의 장점을 하나로 합해 새로운 모델을 만드는 방법이다.

<img src="/resources/staking_structure.PNG" width="600">
    (간단한 stacking의 구조)


### 왜?
 기본적으로 다음과 같은 가정을 한다.

1. 모든 모델은 `mistake`를 가지고 있다. --> 완벽한 모델은 존재하지 않는다.
   아무리 Xgboost, lightGBM 등 앙상블 계열 알고리즘이 뛰어나도 알고리즘 구조상 놓치는 부분이 존재한다는 것을 인정한다.
   
2. 잘 맞추는 부분을 통합 한다면 더 잘 맞추게 될 것이다.


<img src="/resources/staking_dart.PNG" width="600">
## 1.2. 마크다운의 장-단점
### 1.2.1. 장점
	1. 간결하다.
	2. 별도의 도구없이 작성가능하다.
	3. 다양한 형태로 변환이 가능하다.
	3. 텍스트(Text)로 저장되기 때문에 용량이 적어 보관이 용이하다.
	4. 텍스트파일이기 때문에 버전관리시스템을 이용하여 변경이력을 관리할 수 있다.
	5. 지원하는 프로그램과 플랫폼이 다양하다.
### 1.2.2. 단점
	1. 표준이 없다.
	2. 표준이 없기 때문에 도구에 따라서 변환방식이나 생성물이 다르다.
	3. 모든 HTML 마크업을 대신하지 못한다.


위 사진은 1번의 예시가 잘 담겨있다.

KNN: 중앙에서 높은 정확도
SVM: 가장자리에서 높은 정확도


[원본링크][interview]


[kaggle-home]: https://www.kaggle.com/
[interview]: http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
