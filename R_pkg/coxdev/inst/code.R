## Return values from a python pickle
get_pyobj <- function(fname = '/Users/naras/GitHub/coxdev/R_pkg/coxdev/inst/out100.pkl') {
  code  <- paste(
    c(
      "import pickle",
      paste0("junk = ", sprintf("open('%s', 'rb')", fname)),
      "junk2 = pickle.load(junk)",
      "junk.close()"
    ),
    collapse = "\n")
  reticulate::py_run_string(code)
  as.list(reticulate::py$junk2)
}

dev_result  <- get_pyobj('/Users/naras/GitHub/coxdev/R_pkg/coxdev/inst/dev100.pkl')

library(survival)
library(Matrix)
library(coxdev)
d  <- readRDS("~/GitHub/coxdev/R_pkg/coxdev/inst/out100.RDS")
attach(d)
y  <- Surv(event, status)
F <- coxph(y ~ X, init=beta, weights=weights, control=coxph.control(iter.max=0), ties=ties, robust=FALSE)
score <- colSums(coxph.detail(F)$scor)
G <- score
D <- F$loglik[2]
cov <- vcov(F)
#list(-2 * G, -2 * D, cov)

#debug(make_cox_deviance)

cox_deviance  <- make_cox_deviance(event = y[, 1], status = y[, 2], weight = weights, tie_breaking = 'breslow')

eta  <- X %*% beta
#debug(cox_deviance$coxdev)

C <- cox_deviance$coxdev(X %*% beta, weights)

H  <- cox_deviance$information(X %*% beta, weights)
debug(H)
w <- H(X)
