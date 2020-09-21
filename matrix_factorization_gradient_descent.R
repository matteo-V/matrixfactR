k <- 2
N = 400
p = 500
########## Simulate some data ####################
#latents
U.true <- array(runif(N*k,0, 1/k)*runif(N*k,0,2), dim=c(N,k))
M.true <-  array(runif(p*k,0, 1/k)*runif(p*k,-1,2), dim=c(p,k))
#generate some ratings data
R <- sample(1:5, N*p, replace=T, prob = c(1,1, 2,1,1))
#add noise, this is an p X N matrix
Y <- array(rnorm(N*p, M.true %*% t(U.true), 3)+R, dim=c(p,N))
#sparsify the data
dd <- sample(0:1, N*p, replace=T, prob = c(3,1))
Y[Y*dd<1] <- 0 #use 0 to code for missing
w <- Y!=0 #indicator matrix
#do some mean noamlization
Y <- Y - apply(Y, 1, function(x) mean(x[x!=0]) )
#define index set of observed items (ratings, measures, etc.)
idx.set = which(!is.na(Y), arr.ind=T) #for unobserved movies
usrs <- idx.set[,2]
itms <- idx.set[,1]

#array
# Y <- as.matrix( read.table("~/Desktop/MISPA_HPV_normalized_data/Normalized-Table 1.tsv",sep="\t", header=T)[-1,-1] )
# Y <- t(Y)
# N = ncol(Y)
# p = nrow(Y)
# w <- 1
# k <- 2

#initial array guesses
U <- array(runif(p*k, 0,1/k), dim=c(N,k))
M <- array(runif(p*k, 0,1/k), dim=c(p,k))

#regularization parameters
lam.U= 3e-3
lam.M = 3e-3
eta = 0.00015
MAXITER = 100
TOL = 1e-3
prev.err = -1
time.then=Sys.time()
for(iter in 1:MAXITER){
  #users gradient desent step
  ##### I.E. Content Based Recommendations #######
  ## We know M (movie features) and want to learn user features


  #Figure out this vectorized implementation
  M =  M - eta *(((M%*%t(U) - Y)*w)%*%U + lam.M*(M))

  #Figure out this vectorized implementation
  U =  U - eta * (t((M %*% t(U) - Y)*w)%*%M + lam.U*sum(U))

  # for(j in usrs){
  #   rated.itms <- itms[usrs==j] #items that user j rated
  #   #gradient descent for each element of U
  #   U[j,] = U[j,] -
  #     eta * ( crossprod(x = (M[rated.itms,]%*%U[j,] - Y[rated.itms,j])
  #                       ,y = M[rated.itms,])
  #             + lam.U*U[j,] ) #regularization
  # }

  #items gradient desent step
  ##### I.E. Collaborative FIltering Algorithm #######
  ## We know M (movie features) and want to learn user features


  # for(i in itms){
  #   #users that rated item i
  #   usrs.rated = usrs[itms==i]
  #   M[i,] = M[i,] -
  #     eta * (crossprod(x = (U[usrs.rated,]%*% M[i,]-Y[i,usrs.rated])
  #                     ,y = U[usrs.rated,]) + lam.M*M[i,] )
  # }
  ################## STOP CONDITION: Convergence ###############
  #Error calculation is only for observed Y (hence the subset)
  err = sqrt( mean( ((M %*% t(U) - Y)*w)^2, na.rm = T ) )
  #If change in error is too small
  #if((prev.err - err) < TOL | err < TOL){
  #   cat(sprintf("%d\t%0.4f\t\n", iter, err))
  #   cat("Converged. Decrease tolerance for longer fitting")
  #   break #end simulation
  # }
  prev.err = err
  ################### VERBOSE OPTIONS ###################
  if(iter%%20==0||iter==1){
    time.now = Sys.time()
    rate = ((time.now-time.then)) / 60
    cat(sprintf("%d\t%0.4f\t%0.4f seconds/10 steps\n", iter, err, rate))
    time.then = time.now #reset rate counter
  }
}


ii <- c(rep(0,10), rep(1,10))
z <- cbind(ii, U)
plot(U, pch=19, col=ii+1)

#reproject left out point to dim(k) (XtX)-1 * XtY (standard regression)
solve(t(M)%*%M)%*%t(M)%*%Y[,1] #left out point, Y[,1]
