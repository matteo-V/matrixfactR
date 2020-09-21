######## Matrix Factorization via Alternating Least Squares ##########
# Original algorithm reference:
# Large-scale Parallel Collaborative Filtering for the Netflix Prize
# Yunhong Zhou, Dennis Wilkinson, Robert Schreiber and Rong Pan
# HP Labs, 1501 Page Mill Rd, Palo Alto, CA, 94304
# {yunhong.zhou, dennis#.wilkinson, rob.schreiber, rong.pan}@hp.com
#
# This script implements matrix factorization via ALS
#
# Copyright 2019. Matteo Vaiente
# This software is not guaranteed to be free of bugs.
# It is available to use and modify freely
#####################################################################

k <- 6
N <- 400
p <- 500

#true latents
U.true <- array(runif(N*k,0, 1/k)*runif(N*k,0,2), dim=c(N,k))
M.true <-  array(runif(p*k,0, 1/k)*runif(p*k,-1,2), dim=c(p,k))

#generate some ratings data
R <- sample(1:5, N*p, replace=T, prob = c(3,4, 5,1,2))

#add noise, this is an p X N matrix
Y <- array(rnorm(N*p, M.true %*% t(U.true)+R, 0.5), dim=c(p,N))

#sparsify
dd <- sample(0:1, N*p, replace=T, prob = c(5,1))
Y[dd<1] <- NA #use 0 to code for missing

#define index set of observed things
idx.set <- which(Y!=0, arr.ind=T) #for unobserved movies
usrs <- idx.set[,2]
itms <- idx.set[,1]

U2 <- array(runif(N*k, 0,1/k), dim=c(N,k))
M2 <- array(runif(p*k, 0,1/k), dim=c(p,k))
M2[,1] <- apply(M, 1, function(x) mean(x[x!=0]))

#mean normalization
#Y <- Y - apply(M, 1, function(x) mean(x[x!=0]) )

prev.error = Inf
lam.U= 2e-4
lam.M = 2e-4
TOL = 1e-4
MAXITER = 30

for(iter in 1:MAXITER){

  if(iter==1){
    time.then = Sys.time()
    cat(sprintf("Iter\tError\t\n"))
  }

  ############## ALERNATING LEAST SQUARES ####################
  if(iter%%2==0){ #update M on even steps
    for(i in itms){ #update latents for all items
    #
    #   #users that rated item i
        usrs.rated = usrs[itms==i]
    #   #Alternating least squares solution
       Ai = crossprod(rbind(U2[usrs.rated,])) + lam.M*p*diag(k)
       Vi = crossprod(rbind(U2[usrs.rated,]), Y[i, usrs.rated])
       M2[i,] = solve(Ai,Vi)
     }
  } else { #update the U matrix (Content-based filtering step)

    ##### I.E. Content Based Recommendation #######
    ## We know M (movie features) and want to learn user features
    for(j in usrs){ #update latents for all users

      #items that user j rated
      rated.itms <- itms[usrs==j]

      #Alternating least squares solution
      Aj = crossprod(M2[rated.itms,]) #+ lam.U*N*diag(k)
      Vj = crossprod(M2[rated.itms,], Y[rated.itms,j])
      U2[j,]=  solve(Aj,Vj)
    }
  }
  ################## STOP CONDITION: Convergence ###############
  #Error calculation is only for observed Y (hence the subset)
  err = mean( ((M2 %*% t(U2) - Y)[Y!=0])^2, na.rm = T )
  #If change in error is too small
  if((prev.error - err) < TOL | err < TOL){
    cat(sprintf("%d\t%0.4f\t\n", iter, err))
    cat("Converged. Decrease tolerance for longer fitting")
    break #end simulation
  }
  ################### VERBOSE OPTIONS ###################
  if(iter%%2==0||iter==1){
    time.now = Sys.time()
    rate = (time.now-time.then) / 60
    cat(sprintf("%d\t%0.4f\t%0.4f minutes/20 steps\n", iter, err, rate))
    time.then = time.now #reset rate counter
  }
}
