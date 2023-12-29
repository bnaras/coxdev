#ifdef R_INTERFACE

  
  std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);
  Eigen::VectorXi argsort = Eigen::Map<const Eigen::VectorXi>(sort_order.data(), sort_order.size());

  std::cout << "Step 6" << std::endl;
  
  // Since they are all the same size, we can put them in one loop for efficiency!
  Eigen::VectorXd sorted_time(stacked_time.size()), sorted_status(stacked_status_c.size()),
    sorted_is_start(stacked_is_start.size()), sorted_index(stacked_index.size());
  for (int i = 0; i < sorted_time.size(); ++i) {
    int j = argsort(i);
    sorted_time(i) = stacked_time(j);
    sorted_status(i) = 1 - stacked_status_c(j);
    sorted_is_start(i) = stacked_is_start(j);
    sorted_index(i) = stacked_index(j);    
  }

  std::cout << "Step 7" << std::endl;  
  // do the joint sort

  int event_count = 0, start_count = 0;
  std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
  int which_event = -1, first_event = -1, num_successive_event = 1;
  double last_row_time;
  bool last_row_time_set = false;

  for (int i = 0; i < sorted_time.size(); ++i) {
    double _time = sorted_time(i); 
    int _status = sorted_status(i);
    int _is_start = sorted_is_start(i);
    int _index = sorted_index(i);
    if (_is_start == 1) { //a start time
      start_order_vec.push_back(_index);
      start_map_vec.push_back(event_count);
      start_count++;
    } else { // an event / stop time
      if (_status == 1) {
	// if it's an event and the time is same as last row 
	// it is the same event
	// else it's the next "which_event"
	// TODO: CHANGE THE COMPARISON below to _time >= last_row_time since time is sorted.
	if (last_row_time_set  && _time != last_row_time) {// # index of next `status==1` 
	  first_event += num_successive_event;
	  num_successive_event = 1;
	  which_event++;
	} else {
	  num_successive_event++;
	}
	first_vec.push_back(first_event);
      } else {
	first_event += num_successive_event;
	num_successive_event = 1;
	first_vec.push_back(first_event); // # this event time was not an failure time
      }
      event_map_vec.push_back(start_count);
      event_order_vec.push_back(_index);
      event_count++;
      last_row_time = _time;
      last_row_time_set = true;
    }
  }

  std::cout << "Step 8" << std::endl;
  
  // Except for start_order and event_order which are returned, we can probably not make copies
  // for others here.
  // Eigen::VectorXi first = Eigen::Map<Eigen::VectorXi>(first_vec.data(), first_vec.size());
  // Eigen::VectorXi start_order = Eigen::Map<Eigen::VectorXi>(start_order_vec.data(), start_order_vec.size());
  // Eigen::VectorXi event_order = Eigen::Map<Eigen::VectorXi>(event_order_vec.data(), event_order_vec.size());
  // Eigen::VectorXi start_map = Eigen::Map<Eigen::VectorXi>(start_map_vec.data(), start_map_vec.size());
  // Eigen::VectorXi event_map = Eigen::Map<Eigen::VectorXi>(event_map_vec.data(), event_map_vec.size());

  Eigen::VectorXi first(first_vec.size());
  for (size_t i = 0; i < first.size(); ++i) {
    first[i] = first_vec[i];
  }
  Eigen::VectorXi start_order(start_order_vec.size());
  for (size_t i = 0; i < start_order.size(); ++i) {
    start_order[i] = start_order_vec[i];
  }
  Eigen::VectorXi event_order(event_order_vec.size());
  for (size_t i = 0; i < event_order.size(); ++i) {
    event_order[i] = event_order_vec[i];
  }
  Eigen::VectorXi start_map(start_map_vec.size());
  for (size_t i = 0; i < start_map.size(); ++i) {
    start_map[i] = start_map_vec[i];
  }
  Eigen::VectorXi event_map(event_map_vec.size());
  for (size_t i = 0; i < event_map.size(); ++i) {
    event_map[i] = event_map_vec[i];
  }

  std::cout << "Step 9" << std::endl;
    
  // reset start_map to original order
  Eigen::VectorXi start_map_cp = start_map;
  for (int i = 0; i < start_map.size(); ++i) {
    start_map(start_order(i)) = start_map_cp(i);
  }

  std::cout << "Step 10" << std::endl;
  
  // set to event order
  Eigen::VectorXi _status(status.size());
  for (int i = 0; i < status.size(); ++i) {
    _status(i) = status(event_order(i));
  }
  std::cout << "Step 11" << std::endl;
  
  Eigen::VectorXi _first = first;
  
  Eigen::VectorXi _start_map(start_map.size());
  for (int i = 0; i < start_map.size(); ++i) {
    _start_map(i) = start_map(event_order(i));
  }

  std::cout << "Step 12" << std::endl;
  
  Eigen::VectorXi _event_map = event_map;

  Eigen::VectorXi _event(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _event(i) = event(event_order(i));
  }

  std::cout << "Step 13" << std::endl;
    
  Eigen::VectorXi _start(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _start(i) = event(start_order(i));
  }

  std::cout << "Step 14" << std::endl;
  
  std::vector<int> last_vec;
  int last_event = nevent - 1, first_size = first.size();
  for (int i = 0; i < first_size; ++i) {
    int f = _first(first_size - i - 1);
    last_vec.push_back(last_event);
    // immediately following a last event, `first` will agree with np.arange
    if (f - (nevent - 1 - i) == 0) {
      last_event = f - 1;
    }
  }
  std::cout << "Step 15" << std::endl;  
  Eigen::VectorXi last = Eigen::Map<Eigen::VectorXi>(last_vec.data(), last_vec.size());  

  std::cout << "Step 16" << std::endl;
  
  int last_size = last.size();
  Eigen::VectorXi _last(last_size);
  // Now reverse last into _last
  for (int i = 0; i < _last.size(); ++i) {
    _last(i) = last_vec[last_size - i - 1];
  }

  std::cout << "Step 17" << std::endl;
  
  Eigen::VectorXd _scaling(nevent);
  for (int i = 0; i < nevent; ++i) {
    double fi = (double) _first(i);
    _scaling(i) = ((double) i - fi) / ((double) _last(i) + 1.0 - fi);
  }
  
#ifdef PY_INTERFACE
  py::dict preproc;
  std::cout << "Step 18" << std::endl;    
  preproc["start"] = _start;
  preproc["event"] = _event;
  preproc["first"] = _first;
  preproc["last"] = _last;
  preproc["scaling"] = _scaling;
  preproc["start_map"] = _start_map;
  preproc["event_map"] = _event_map;
  preproc["status"] = _status;
  std::cout << "Step 18" << std::endl;
  
  return std::make_tuple(preproc, event_order, start_order);
#endif

#ifdef R_INTERFACE
  Rcpp::List preproc = Rcpp::List::create(
					  Rcpp::_["start"] = Rcpp::wrap(_start),
					  Rcpp::_["event"] = Rcpp::wrap(_event),
					  Rcpp::_["first"] = Rcpp::wrap(_first),
					  Rcpp::_["last"] = Rcpp::wrap(_last),
					  Rcpp::_["scaling"] = Rcpp::wrap(_scaling),
					  Rcpp::_["start_map"] = Rcpp::wrap(_start_map),
					  Rcpp::_["event_map"] = Rcpp::wrap(_event_map),
					  Rcpp::_["status"] = Rcpp::wrap(_status)
					  );
  return(Rcpp::List::create(
			    Rcpp::_["preproc"] = preproc,
			    Rcpp::_["event_order"] = Rcpp::wrap(event_order),
			    Rcpp::_["start_order"] = Rcpp::wrap(start_order)));
#endif
  
