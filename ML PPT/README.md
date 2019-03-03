### 스터디를 진행하면서 만들었던 PPT 자료 

PDF로 올리지 않는 이유는, PT 하단 설명란에 있는 추가 설명 때문입니다.


## 1. [Promotion Impact][Promotion Impact] : <br>
  이벤트, 프로모션 진행 시 효과를 측정 할 수 있는 방법 및 패키지 

## 2. [How Dese Batch Norm Help Optimization][How Dese Batch Norm Help Optimization]: <br>
   BN(배치)는 ICS(내부 공변량)를 줄여주는 역할이라고 알려져 있었지만,<br>
   실험과 수식을 한 검증을 통해 다른 효과가 있다고 밝힘<br>
   주된 요인은 Loss landscape를 립시츠화 시켜 학습에 도움이 되게 함.   

## 3. [InstaGAN][InstaGAN]: <br>
  CycleGAN을 읽었을 때, 꽤 좋은 성능이라고 느꼈는데 그보다 발전한 논문<br>
  - 장점: unfair Data를 효과적으로 활용 <br>
  - 단점: mask 데이터가 필요 <br>
  
## 4. [Staking][Staking]<br>
  포스트로 만들었지만, PPT도 내용 추가.<br>
  - 장점: 모델의 성능을 한층 올릴 수 있다. 
  - 단점: 엄청난 연산량에 비해 오르는 수치는 미비하다. 오롯이 성능을 위해서라면 해볼만하다
  
## 5. [BEGAN][BEGAN]<br>
  2017년 8월, 알고 있는 GAN중 mode collapse에 대해 고민한 모델
  - 장점: mode collapse가 일어나는 일정 비율을 측정 할 수 있었다.
  - 단점: 어려운 수식 <br>
  <img src="/ML PPT/resouces/IOI_GAN2.gif" width="650"> <br>
   BEGAN을 이용해도 mode collapse가 일어난 모습 (ioi 멤버 얼굴 모음)

## 6. [InfoGAN][InfoGAN]<br>
  처음으로 정보이론을 알게 해준 GAN
  - 장점: 정보(information)관점에서 GAN의 모습을 볼 수 있고<br>
    & 조건부 GAN이 아니지만 변화하는 이미지 생성 가능 <br>
  - 단점: 변화를 컨트롤 할 수 없다. <br>
<img src="/ML PPT/resouces/infoGAN_박이삭_1.jpg" width="150"> <br>

## 7. [유전알고리즘 - GA][유전알고리즘] <br>
  가장 기억에 남는 알고리즘. <br>
  - 장점: 복잡한 계산없이 최적해 구하기 가능<br>
  - 단점: Obj Function을 잘 정의해야 한다. 
  
## 8. [LIME][LIME]<br>
  선형선을 띄는 다른 모델을 이용해 복잡한 ML 모델 해석<br>
  -장점: 제대로 된 모델 해석의 시초 <br>
  -단점: 비 선형적인 해석을 불가하다. SHAP와 비교시 한계점이 보인다.<br>
  [LIME 쥬피터][LIME 쥬피터]
  
  
## 9. SHAP<br>
  직접 PPT는 작성하지 않았다 <br>
  스터디원이 소개해준 논문으로 모델 해석의 끝판왕이라 생각한다.<br>
  - 장점: 빠른 모델 해석과 클래스별 변수 중요도 파악가능<br>
  - 단점: 모델 해석이지 데이터 해석은 아니다. <br>
  [binary_class 쥬피터 노트북][binary_class], [multiple_class 쥬피터 노트북][multiple_class]



[Promotion Impact]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/promotionImpact_20190222.pptx
[How Dese Batch Norm Help Optimization]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/How%20Dese%20Batch%20Norm%20Help%20Optimization_박이삭20190304.pptx
[InstaGAN]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/InstaGAN_20190114_박이삭.pptx
[Staking]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/Stacking_20181224.pptx
[BEGAN]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/BEGAN_20170803박이삭.pptx
[InfoGAN]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/infoGAN_20170713이삭.pptx
[유전알고리즘]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/유전알고리즘_20170504.pptx
[LIME]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/LIME_VS_SHAP_20181029_박이삭.pdf
[LIME 쥬피터]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/resouces/SHAP%20VS%20LIME.ipynb
[binary_class]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/resouces/SHAP_value%20-%20%20binary%20class.ipynb
[multiple_class]: https://github.com/eat-toast/eat-toast.github.io/blob/master/ML%20PPT/resouces/SHAP_value%20multi%20class.ipynb
