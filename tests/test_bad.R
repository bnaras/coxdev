library(coxdev)
library(survival)
tol  <- 1e-10

all_close  <- function(a, b, rtol = 1e-05, atol = 1e-08) {
  all(abs(a - b) <= (atol + rtol * abs(b)))
}

rel_diff_norm  <- function(a, b) { a <- as.matrix(a); b <- as.matrix(b); norm(a - b, 'F') / norm(b, 'F') }

generate_problematic_test_data <- function() {
  # Set random seed for reproducibility
  set.seed(42)
  
  # Generate a problematic test case
  n <- 50
  p <- 10
  
  # Generate event times, status, and weights
  event <- rexp(n, rate = 0.5)
  status <- sample(c(0, 1), size = n, replace = TRUE, prob = c(0.3, 0.7))
  sample_weight <- runif(n, 0.5, 2.0)
  
  # Generate design matrix and coefficients
  X <- matrix(rnorm(n * p), n, p)
  beta <- rnorm(p) * 0.1
  
  list(
    event = event,
    start = NULL,  # No start times for this test
    status = status,
    sample_weight = sample_weight,
    X = X,
    beta = beta
  )
}

get_glmnet_result <- function(event,
                              status,
                              start,
                              eta,
                              weight) {
  event <- as.numeric(event)
  status <- as.numeric(status)
  weight <- as.numeric(weight)
  eta <- as.numeric(eta)
  if (length(start) == length(status)) {
    y <- survival::Surv(start, event, status)
    D_R <- glmnet:::coxnet.deviance3(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad3(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  } else {
    y <- Surv(event, status)
    D_R <- glmnet:::coxnet.deviance2(pred=eta, y=y, weight=weight, std.weights=FALSE)
    G_R <- glmnet:::coxgrad2(eta, y, weight, std.weights=FALSE, diag.hessian=TRUE)
    H_R <- attr(G_R, 'diag_hessian')
  }
  list(D = D_R, G = as.numeric(-2 * G_R), H = -2 * H_R)
}

get_coxph <- function(event,
                      status,
                      X,
                      beta,
                      sample_weight,
                      start=None,
                      ties='efron') {

  if (length(start) == length(status)) start  <- as.numeric(start)
  start <- as.numeric(start)
  status <- as.numeric(status)
  event <- as.numeric(event)
  weight <- as.numeric(sample_weight)
  if (length(start) == length(status)) {
    y <- Surv(start, event, status)
  } else {
    y <- Surv(event, status)
  }
  F <- coxph(y ~ X, init=beta, weights=sample_weight, control=coxph.control(iter.max=0), ties=ties, robust=FALSE)
  G <- colSums(coxph.detail(F)$scor)
  D <- F$loglik
  cov <- vcov(F)
  list(G = -2 * G, D = -2 * D, cov = cov)
}

check_results  <- function(data_dict, ties = c('efron', 'breslow')) {
  ties  <- match.arg(ties)

  weight <- data_dict$sample_weight
  event <- data_dict$event
  start <- data_dict$start
  status <- data_dict$status
  cox_deviance  <- make_cox_deviance(event = event,
                                     start = start,
                                     status = status,
                                     weight = weight,
                                     tie_breaking = ties)

  X  <- data_dict$X; tX  <- t(X);
  beta  <- data_dict$beta
  C <- cox_deviance$coxdev(X %*% beta, weight)
  h <- cox_deviance$information(X %*% beta, weight)
  I <- tX %*% h(X)
  message(ifelse(all_close(I, t(I)), "I is symmetric enough", "I is not symmetric"))
  new_cov  <- solve(I)

  ## Check against coxph with
  coxph_result  <- get_coxph(event = event,
                             status = status,
                             beta  = beta,
                             sample_weight = weight,
                             start = start,
                             ties = ties,
                             X = X)
  G_coxph <- coxph_result$G
  D_coxph <- coxph_result$D[1]
  cov_coxph  <- coxph_result$cov

  message(ifelse(all_close(D_coxph, C$deviance - 2 * C$loglik_sat), "Coxph Deviance matches", "Coxph Deviance mismatch"))
  message(ifelse(rel_diff_norm(G_coxph, tX %*% C$gradient) < tol, "Coxph Gradient matches", "Coxph Gradient mismatch"))
  message(ifelse(rel_diff_norm(new_cov, cov_coxph) < tol, "Coxph Cov matches", "Coxph Cov mismatch"))

  if (ties == 'breslow') {
  ## Check against glmnet also
    glmnet_result  <- get_glmnet_result(event = event,
                                        status = status,
                                        start = NA,
                                        eta = X %*% beta,
                                        weight = weight)
    G_glmnet <- glmnet_result$G
    D_glmnet <- glmnet_result$D[1]
    H_glmnet  <- glmnet_result$H

    message(ifelse(all_close(D_glmnet, C$deviance), "Glmnet Deviance matches", "Glmnet Deviance mismatch"))
    message(ifelse(rel_diff_norm(G_glmnet, C$gradient) < tol, "Glmnet Gradient matches", "Glmnet Gradient mismatch"))
    message(ifelse(rel_diff_norm(H_glmnet, C$diag_hessian) < tol, "Glmnet hessian matches", "Glmnet hessian mismatch"))
  }
}

# Generate test data instead of loading from file
test_data <- generate_problematic_test_data()
check_results(test_data)

check_results(test_data, 'breslow')

