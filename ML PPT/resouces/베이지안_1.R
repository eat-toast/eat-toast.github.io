# some basis egs
one <- function(x) rep(1,length(x))
id  <- function(x) x
sq  <- function(x) x^2
x3  <- function(x) x^3
x4  <- function(x) x^4

#X를 만드는 함수. 
make.X <- function(n) {
  runif(n,-1,1)
}

a0 <- -0.3 # the true values (unknown to model)
a1 <-  0.5

sigma <- 0.2
beta  <- 1/sigma^2  # precision

make.Y <- function(xs) {
  a0 + a1*xs + rnorm(length(xs),0,sigma)
}




# uses linear regression basis (phi) by default 
compute_posterior <- function(X, Y, m_old, S_old, phi= c(one, id)) {
  Phi <- sapply(phi, function(base) base(X))  # make design matrix
  
  if(length(X)==1)  # type hack, with just 1 point, R makes a vector, not a matrix
    Phi <- t(as.matrix(Phi))                               
  
  S_new <- solve(solve(S_old) + beta * t(Phi) %*% Phi)
  m_new <- S_new %*% (solve(S_old) %*% m_old + beta * t(Phi) %*% Y)
  
  list(m=m_new, S=S_new)  # return the new updated parameters
}

alpha <- 2.0
m_0 <- c(0,0)         # we know the mean is (0,0), otherwise, center first
S_0 <- alpha*diag(2)  # relatively uninformative prior

set.seed(121) 
X <- make.X(5) # make some points
Y <- make.Y(X)

posterior_1 <- compute_posterior(X, Y, m_0, S_0)
posterior_1$m



X_new <- make.X(10) # more points are available!
Y_new <- make.Y(X_new)

posterior_2 <- compute_posterior(X_new,  Y_new, posterior_1$m, posterior_1$S)
posterior_2$m

plot(c(X,X_new),c(Y,Y_new),type="n")
legend("topleft",c("true fit","1st fit","2nd fit"), 
       col=c("green","grey","red"), lty=1, lwd=2) 
points(X    , Y    , pch=19, col="black")
points(X_new, Y_new, pch=19, col="blue")
abline(posterior_1$m, col="grey")  # old fit
abline(posterior_2$m, col="red")   # new fit
abline(c(-0.3,.5), col="green")    # target function (true parameter values)





# return the predictive distribution's mean and 95% density interval
get_predictive_vals <- function(x, m_N, S_N, phi) {
  phix <- sapply(phi, function(base) base(x))
  mean_pred <- t(m_N) %*% phix
  sd_pred  <- sqrt(1/beta + t(phix) %*% S_N %*% phix)
  
  c(mean_pred, mean_pred-2*sd_pred, mean_pred+2*sd_pred)
}

draw_predictive <- function(xs, m_N, S_N, phi) {
  vs <- rep(NA, length(xs))
  ys <- data.frame(means=vs, p2.5=vs, p97.5=vs)  # init dataframe
  
  for (i in 1:length(xs)) {  # compute predictive values for all xs
    ys[i,] <- get_predictive_vals(xs[i],m_N, S_N, phi)
  }
  
  # draw mean and 95% interval
  lines(xs, ys[,1], col="red", lwd=2)
  lines(xs, ys[,2], col="red", lty="dashed")
  lines(xs, ys[,3], col="red", lty="dashed")
}


set.seed(121) 
X <- make.X(5) # make some points
Y <- make.Y(X)

phi <- c(one,id,sq,x3) # basis for the cubic regression
m_0 <- c(0,0,0,0)      # priors
S_0 <- alpha*diag(4) 

posterior_1 <- compute_posterior(X, Y, m_0, S_0, phi=phi)
m_N <- posterior_1$m
S_N <- posterior_1$S

plot(X, Y, pch=20, ylim=c(-1.5,1), xlim=c(-1,1), ylab="y", xlab="x")
xs <- seq(-1,1,len=50)
draw_predictive(xs, m_N, S_N, phi=phi)
