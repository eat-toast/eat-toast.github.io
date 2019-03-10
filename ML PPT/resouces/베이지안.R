##########################################
#베이지안 추론
#동전의 앞면과 뒷면이 나올 확률을 각각, 
# 7 : 3이라 믿는 A
# 5 : 5이라 믿는 B
# 2 : 8이라 믿는 C
#가 있을 때, 이들의 사전분포(믿음)이 실험을 하면서 어떻게 변하는지 확인해 보겠음.

#동전 던지기는 이항분포를 따르고, 이항분포의 conjugate distribution은 Beta분포.

#--> Beta(a, b) 2개의 파라미터를 가지고  각각 a:앞면, b:뒷면과 연관이 있음.

#따라서 A는 a=7, b=3
# B는 a=5, b=5
# C는 a=2, b=8


x<-rbeta(n=500, shape1=7, shape2 = 3)
# A의 상태를 나타냄.

curve(dbeta(x,(7),(3)),col = "red", xlab = "Batting Average = P", ylab = "density", lwd=5)
#붉은 선은 이항분포의 p의 분포.  ex) Pr(p=0) = 0

#가장 가능성이 높은 지역은
idx<-which.max(dbeta(x,(7),(3)))
abline(v=x[idx], col='blue', lwd=3)
text(x=0.6, y=2.7,labels = round(x[idx],2), cex=2)


person<-function(alpha=7, beta=3, n=500, step=10){
  set.seed(n)
  x<-rbeta(n=500, shape1=alpha, shape2 = beta) #초기분포(사전확률)
  
  test<-rbinom(n ,1, prob=0.5)# 동전 던지기 500회 수행
  idx<-seq(from =10, to=n, by=step)
  
  curve(dbeta(x,(alpha),(beta)),col = "red", xlab = "Batting Average", ylab = "density", xlim=c(0, 1))
  
  curve(dbeta(x,(alpha),(beta)),col = "brown", xlab = "Batting Average", ylab = "density", xlim=c(0, 1), ylim=c(0, 20), lwd=5)
  for(i in idx){
    par(new=TRUE)
    tab<-table(test[1:i]) #결과확인  
    curve(dbeta(x,(alpha+ tab[1]),(beta+ tab[2])),col = "red", xlim=c(0, 1), ylim=c(0, 20),xlab = "Batting Average", ylab = "density")
  }
  par(new=TRUE)
  alpha= alpha+ tab[1]
  beta = beta+ tab[2]
  
  curve(dbeta(x,(7+ tab[1]),(3+ tab[2])),col = "blue", xlim=c(0, 1), ylim=c(0, 20), lwd=3,xlab = "Batting Average", ylab = "density",
        main= paste('P 평균', round(alpha / (alpha + beta),2)), cex.main=5)
  legend('topright', fill=c('brown','blue'), legend = c('처음', '마지막'), cex=5)
}

png('test.png', width = 2400, height = 1200)
par(mfrow=c(3,2))
person(alpha = 7, beta=3, n=1000, step=100)
person(alpha = 5, beta=5, n=1000, step=100)
person(alpha = 2, beta=8, n=1000, step=100)
dev.off()


#####
#GMM 알고리즘

#참고자료: https://www.youtube.com/watch?v=nktiUUd6X_U
#참고자료: https://cran.r-project.org/web/packages/mclust/vignettes/mclust.html
data(iris)

X<- iris[,-5]
class<- iris[,5]

head(X)
table(class)

library(mclust)

mod1<-Mclust(X)
summary(mod1)

#MODEL BASE CLUSTERING
#Mclust: 자동적으로 그룹의 수(k)와 파라미터들을 추정한다. by BIC (bayesian Information Criterion)
#BIC: penalized form of the log likelihood. Higher means better
# 그룹의 수는 1 ~ 9개로 조정할 수 있음

# 공분산 추정은 E와 V 과정이 존재.
# E: 동일한 분산을 가정(eual Variance)
# V: 서로 다른 분산을 가정 (different variance)
# 다변량일때는 14가지 추정방법이 존재. --> 평가 메트릭이 14가지 존재.

mod2<- Mclust(X, G=3, modelNames = c('EII','EEE', 'VEV','VVV'))

##Parameters
#G: 생성할 가우시안의 수 = Cluster의 갯수
#modelNames: 
summary(mod2, parameters=TRUE)

##########################################
#View the Clusters Result

round(mod2$z,4) #k번째 클래스 확률
mod2$classification #clustering labels

mod2$parameters


##########################################
#Evaluate the Model by Labels
table(class, mod2$classification)

adjustedRandIndex(class, mod2$classification) #ARI : Adjust Rand Idx
#0 ~ 1 사잇값을 나타내며, 높을 수록 클래스당 좋은 acc를 가졌을음 뜻함.


##########################################
#Plot the clustering Result
plot(mod1, what='BIC')

plot(mod2, what='classification')
plot(mod2, what='classification', dimens = c(2,1))
plot(mod2, what='density', lwd=3) #dimens옵션 사용 불가.

