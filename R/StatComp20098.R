#' @title Initializing GMM with k-means
#' @description Initializing GMM with k-means
#' @param dat training set(matrix)
#' @param k The number of Gaussian components (numeric)
#' @return a list (weight,mu,sigma)
#' @examples
#' \dontrun{
#' init <- getInitGMM(A_train,2)
#' }
#' @export
getInitGMM <- function(dat, k) {
  km <- kmeans(dat, k)
  weight <- km$size / length(dat)
  mu <- km$centers
  cluster <- km$cluster
  sigma <- numeric()
  for (i in 1:k) {
    index = which(cluster == i)
    var <- var(dat[index])
    sigma[i] <- sqrt(var)
  }
  res <- list(weight, mu, sigma)
  return(res)
}

#' @title EM algorithm
#' @description Using EM algorithm to estimate the parameters of GMM
#' @param dat training set(matrix)
#' @param k The number of Gaussian components (numeric)
#' @param init_param initial value of parameters (list)
#' @return a list of parameters of GMM  (weight,mu,sigma)
#' @examples
#' \dontrun{
#' param <- EM(A_train,2,weight,mu,sigma)
#' }
#' @export
EM <- function(dat, k, init_param) {
  weight <- init_param[1][[1]]
  mu <- init_param[2][[1]]
  sigma <- init_param[3][[1]]
  N <- length(dat)
  prob <- matrix(rep(0,N*k),N,k)
  gamma <- matrix(rep(0,N*k),N,k)
  iter = 0
  while(iter <= 200) {
    iter = iter + 1
    #E-step
    for (j in 1:k) {
      prob[,j] <- sapply(dat, dnorm, mu[j], sigma[j])
    }
    for (i in 1:N) {
      gamma[i,] <- mapply(function(x,y) x*y , weight, prob[i,])
    }
    gamma <- gamma / rowSums(gamma)
    
    old_weight <- weight
    old_mu <- mu
    old_sigma <- sigma
    #M-step
    for (j in 1:k) {
      p1 <- sum(gamma[,j]*dat)
      p2 <- sum(gamma[,j])
      weight[j] <- p2 / N
      mu[j] <- p1 / p2
      sigma[j] <- sqrt(sum(gamma[,j]*(dat-mu[j])^2) / p2)
    }
    epsilon <- 1e-4
    if (sum(abs(weight - old_weight)) < epsilon & 
        sum(abs(mu - old_mu)) < epsilon & 
        sum(abs(sigma - old_sigma)) < epsilon) {
      break
    }
  }
  return(list(weight,mu,sigma))
}

#' @title Calculate the prediction accuracy of GMM
#' @description Calculate the prediction accuracy of GMM
#' @param test_data test set(matrix)
#' @param k The number of Gaussian components (numeric)
#' @param weight GMM weights on training set (matrix)
#' @param mu GMM mu on training set (matrix)
#' @param sigma GMM sigma on training set (matrix)
#' @return  prediction accuracy on test set (numeric)
#' @examples
#' \dontrun{
#' acc <- getAcc(test,2,weight,mu,sigma)
#' }
#' @export
getAcc <- function(test_data, k, weight, mu, sigma) {
  n <- length(test_data[,1])
  predict <- matrix(rep(0,n*3),n,3)
  for (k in 1:2) {
    temp = matrix(rep(0,n*2), nrow = n, ncol = 2)
    for (j in 1:2) {
      temp[,j] = sapply(test_data[,2], dnorm, mu[k,j], sigma[k,j])
    }
    for (i in 1:n) {
      temp[i,] = mapply(function(x,y) x*y, weight[k,], temp[i,])
    }
    predict[,k] = rowSums(temp[,1:2])
  }
  predict = as.data.frame(predict)
  for (i in 1:n) {
    if (predict[i,1] >= predict[i,2]) {
      predict[i,3] = 'A'
    }else{
      predict[i,3] = 'B'
    }
  }
  acc = sum(predict[,3] == test_data[,1]) / n
  return(acc)
}

#' @title Benchmark R and Rcpp functions.
#' @name benchmarks
#' @description Use R package \code{microbenchmark} to compare the performance of C functions (\code{gibbsR} and \code{vaccR}) and Cpp functions (\code{gibbsC} and \code{vaccC}).
#' @import microbenchmark
#' @importFrom Rcpp evalCpp
#' @importFrom stats rnorm rgamma var dnorm kmeans
#' @useDynLib StatComp20098
#' @examples
#' \dontrun{
#' data(data)
#' attach(data)
#' tm1 <- microbenchmark::microbenchmark(
#'   rnR = gibbsR(100,10),
#'   rnC = gibbsC(100,10)
#' )
#' print(summary(tm1)[,c(1,3,5,6)])
#' 
#' tm2 <- microbenchmark::microbenchmark(
#'   vR = vaccR(age,female,ily),
#'   vC = vaccC(age,female,ily)
#' )
#' print(summary(tm2)[,c(1,3,5,6)])
#' }
NULL