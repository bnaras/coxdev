_preprocess <- function(start, event, status) {
  # Convert inputs to vectors
  start <- as.numeric(start)
  event <- as.numeric(event)
  status <- as.numeric(status)
  nevent <- length(status)

  # Perform stacking of arrays
  stacked_time <- c(start, event)
  stacked_status_c <- c(rep(1, nevent), 1 - status) # complement of status
  stacked_is_start <- c(rep(1, nevent), rep(0, nevent))
  stacked_index <- c(seq_len(nevent), seq_len(nevent))

  # Perform the joint sort
  order_indices <- order(stacked_time, stacked_status_c, stacked_is_start)
  sorted_time <- stacked_time[order_indices]
  sorted_status <- 1 - stacked_status_c[order_indices]
  sorted_is_start <- stacked_is_start[order_indices]
  sorted_index <- stacked_index[order_indices]

  # Initialize variables for loop
  event_count <- 0
  start_count <- 0
  event_order <- numeric(0)
  start_order <- numeric(0)
  start_map <- numeric(0)
  event_map <- numeric(0)
  first <- numeric(0)
  which_event <- -1
  first_event <- -1
  num_successive_event <- 1
  last_row <- NULL

  # Loop through sorted data
  for (i in seq_along(sorted_time)) {
    _time <- sorted_time[i]
    _status <- sorted_status[i]
    _is_start <- sorted_is_start[i]
    _index <- sorted_index[i]

    if (_is_start == 1) {
      start_order <- c(start_order, _index)
      start_map <- c(start_map, event_count)
      start_count <- start_count + 1
    } else {
      if (_status == 1) {
        if (!is.null(last_row) && _time != last_row[1]) {
          first_event <- first_event + num_successive_event
          num_successive_event <- 1
          which_event <- which_event + 1
        } else {
          num_successive_event <- num_successive_event + 1
        }
        first <- c(first, first_event)
      } else {
        first_event <- first_event + num_successive_event
        num_successive_event <- 1
        first <- c(first, first_event)
      }

      event_map <- c(event_map, start_count)
      event_order <- c(event_order, _index)
      event_count <- event_count + 1
    }
    last_row <- c(_time, _status, _is_start, _index)
  }

  # Reset start_map to original order and set to event order
  start_map_cp <- start_map
  start_map[start_order] <- start_map_cp

  _status <- status[event_order]
  _first <- first
  _start_map <- start_map[event_order]
  _event_map <- event_map

  _event <- event[event_order]
  _start <- event[start_order]

  # Compute `last`
  last <- numeric(0)
  last_event <- nevent - 1
  for (i in length(_first):1) {
    f <- _first[i]
    last <- c(last, last_event)
    if (f - (nevent - i) == 0) {
      last_event <- f - 1
    }
  }
  _last <- rev(last)

  den <- _last + 1 - _first
  _scaling <- (seq_len(nevent) - 1 - _first) / den

  # Prepare the output list
  preproc <- list(
    start = _start,
    event = _event,
    first = _first,
    last = _last,
    scaling = _scaling,
    start_map = _start_map,
    event_map = _event_map,
    status = _status
  )

  return(list(preproc, event_order, start_order))
}


make_cox_deviance <- function(event,
                              start = NA, # if NA, indicates just right censored data
                              status,
                              tie_breaking = c('efron', 'breslow'),
                              weight) {

  tie_breaking  <- match.arg(tie_braking)

  event <- as.numeric(event)
  nevent <- length(event)
  status <- as.numeric(status)
  if (is.na(start)) {
    start <- rep(-Inf, nevent)
    have_start_times <- FALSE
  } else {
    have_start_times <- TRUE
  }

  prep_result  <- _preprocess(start, event, status)
  preproc  <- prep_result[[1]]
  event_order  <- as.integer(prep_result[[2]])
  start_order  <- as.integer(prep_result[[3]])
  efron  <- (tie_breaking == 'efron') && (norm(preproc[['scaling']]) > 0)
  status <- as.numeric(preproc[['status']])
  event <- as.numeric(preproc[['event']])
  start <- as.numeric(preproc[['start']])
  first <- as.integer(preproc[['first']])
  last <- as.integer(preproc[['last']])
  scaling <- as.numeric(preproc[['scaling']])
  event_map <- as.integer(preproc[['event_map']])
  start_map <- as.integer(preproc[['start_map']])
  first_start <- first[start_map]

  if (!all(first_start == start_map)) {
    stop('first_start disagrees with start_map')
  }

  n <- length(status)

  # allocate necessary memory

  T_1_term <- numeric(n)
  T_2_term <- numeric(n)
  # event_reorder_buffers = np.zeros((3, n))
  event_reorder_buffers <- lapply(seq_len(3), numeric(n))
  # forward_cumsum_buffers = np.zeros((5, n+1))
  forward_cumsum_buffers <- lapply(seq_len(5), numeric(n + 1))
  forward_scratch_buffer <- np.zeros(n)
  # reverse_cumsum_buffers = np.zeros((4, n+1))
  reverse_cumsum_buffers <- lapply(seq_len(4), numeric(n + 1))
  # risk_sum_buffers = np.zeros((2, n))
  risk_sum_buffers <- list(numeric(n), numeric(n))
  hess_matvec_buffer <- numeric(n)
  grad_buffer <- numeric(n)
  diag_hessian_buffer <- numeric(n)
  diag_part_buffer <- numeric(n)
  w_avg_buffer <- numeric(n)
  exp_w_buffer <- numeric(n)

  function (eta, sample_weight = NULL) {
    if (is.null(sample_weight)) {
      sample_weight  <- rep(1.0, length(eta))
    } else {
      samlpe_weight  <- as.numeric(sample_weight)
    }
    loglik_sat  <- .Call('compute_sat_loglik',
                         first,
                         last,
                         sample_weight,
                         event_order,
                         status,
                         forward_cumsum_buffers[[1]],
                         PACKAGE = "coxdev")
    eta <- eta - mean(eta)
    exp_w_buffer <- sample_weight * exp(eta)

    ## The C++ code has to be modified for R lists!
    deviance  <- .Call('deviance',
                       eta,
                       sample_weight,
                       exp_w_buffer,
                       event_order,
                       start_order,
                       status,
                       first,
                       last,
                       scaling,
                       event_map,
                       start_map,
                       loglik_sat,
                       T_1_term,
                       T_2_term,
                       grad_buffer,
                       diag_hessian_buffer,
                       diag_part_buffer,
                       w_avg_buffer,
                       event_reorder_buffers,
                       risk_sum_buffers, #[0] is for coxdev, [1] is for hessian...
                       forward_cumsum_buffers,
                       forward_scratch_buffer,
                       reverse_cumsum_buffers, #[0:2] are for risk sums, [2:4] used for hessian risk*arg sums
                       have_start_times,
                       efron)

