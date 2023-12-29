#ifdef DEBUG
#include <iostream>
#endif

#ifdef PY_INTERFACE

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <algorithm> // For std::sort and other algorithms

// Map python buffers list element into an Eigen double vector
// SRC_LIST = python list, OFFSET = index offset (e.g. 0 or 1),
// DEST will be the ref downstream, TMP should be **unique** throwaway name with each invocation
#define MAP_BUFFER_LIST(SRC_LIST, OFFSET, DEST, TMP)				\
  py::array_t<double> TMP = SRC_LIST[OFFSET].cast<py::array_t<double>>();       \
  Eigen::Map<Eigen::VectorXd> DEST(TMP.mutable_data(), TMP.size());

namespace py = pybind11;
#define EIGEN_REF Eigen::Ref
#define ERROR_MSG(x) throw std::runtime_error(x)
#define BUFFER_LIST(x) py::list &x 

#endif

/**
 * Equivalent of numpy.lexsort for our case where a is stacked_is_start, b is stacked_status_c,
 * and c is stacked event time.
 */
std::vector<int> lexsort(const Eigen::VectorXi& a, const Eigen::VectorXi& b, const Eigen::VectorXd& c) {
  std::vector<int> idx(a.size());
  std::iota(idx.begin(), idx.end(), 0); // Fill idx with 0, 1, ..., a.size() - 1
  
  auto comparator = [&](int i, int j) {
    if (c[i] != c[j]) return c[i] < c[j];
    if (b[i] != b[j]) return b[i] < b[j];
    return a[i] < a[j];
  };
  
  std::sort(idx.begin(), idx.end(), comparator);
  
  return idx;
}

/**
 * Equivalent of numpy.hstack for our Eigen arrays
 */
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> hstack(const Eigen::Matrix<T, Eigen::Dynamic, 1> &a, 
					   const Eigen::Matrix<T, Eigen::Dynamic, 1> &b) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> result = a;
  result.conservativeResize(a.size() + b.size());  // Resize the result vector
  result.segment(a.size(), b.size()) = b;  // Assign values from vector b to the resized part of result
  
  return result;
}

// Return type for preprocess routine below.



/**
 * Compute various functions of the start / event / status to be used to help in computing cumsums
 * This is best done in C++ also to avoid dealing with 1-based indexing in R which can bite us in the arse
 */
#ifdef PY_INTERFACE
std::tuple<py::dict, Eigen::VectorXi, Eigen::VectorXi> preprocess(const EIGEN_REF<Eigen::VectorXd> start,
								  const EIGEN_REF<Eigen::VectorXd> event,
								  const EIGEN_REF<Eigen::VectorXi> status)
#endif
#ifdef R_INTERFACE
Rcpp::List preprocess(const EIGEN_REF<Eigen::VectorXd> start,
		      const EIGEN_REF<Eigen::VectorXd> event,
		      const EIGEN_REF<Eigen::VectorXi> status)
#endif
{
  int nevent = status.size();
  Eigen::VectorXi ones = Eigen::VectorXi::Ones(nevent);
  Eigen::VectorXi zeros = Eigen::VectorXi::Zero(nevent);
  
  // second column of stacked_array is 1-status...
  Eigen::VectorXd stacked_time = hstack(start, event);
  Eigen::VectorXi stacked_status_c = hstack(np.ones(ones, ones - status]); // complement of status
  Eigen::VectorXi stacked_is_start = hstack(ones, zeros);
  Eigen::VectorXi stacked_index = hstack(Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1), Eigen::VectorXi::LinSpaced(nevent, 0, nevent - 1));
  std::vector<int> sort_order = lexsort(stacked_is_start, stacked_status_c, stacked_time);
  Eigen::VectorXi argsort = Eigen::Map<const Eigen::VectorXi>(sort_order.data(), sort_order.size());

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

  // do the joint sort

  int event_count = 0, start_count = 0;
  std::vector<int> event_order_vec, start_order_vec, start_map_vec, event_map_vec, first_vec;
  int which_event = -1, first_event = -1, num_successive_event = 1;
  int last_row_time;
  bool last_row_time_set = false;

  for (int i = 0; i < sorted_time.size(); ++i) {
    _time = sorted_time(i); 
    _status = sorted_status(i);
    _is_started = sorted_is_start(i);
    _index = sorted_index(i);
    if (_is_start == 1) { //a start time
      start_order_vec.push_back(_index);
      start_map_vec.push_back(event_count);
      start_count++;
    } else { // an event / stop time
      if (_status == 1) {
	// if it's an event and the time is same as last row 
	// it is the same event
	// else it's the next "which_event"
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

  // Except for start_order and event_order which are returned, we can probably not make copies
  // for others here.
  Eigen::VectorXi first = Eigen::Map<Eigen::VectorXi>(first_vec.data(), first_vec.size());
  Eigen::VectorXi start_order = Eigen::Map<Eigen::VectorXi>(start_order_vec.data(), start_order_vec.size());
  Eigen::VectorXi event_order = Eigen::Map<Eigen::VectorXi>(event_order_vec.data(), event_order_vec.size());
  Eigen::VectorXi start_map = Eigen::Map<Eigen::VectorXi>(start_map_vec.data(), start_map_vec.size());
  Eigen::VectorXi event_map = Eigen::Map<Eigen::VectorXi>(event_map_vec.data(), event_map_vec.size());

  // reset start_map to original order
  Eigen::VectorXi start_map_cp = start_map;
  for (int i = 0; i < start_map.size(); ++i) {
    start_map(start_order(i)) = start_map_cp(i);
  }

  // set to event order
  Eigen::VectorXi _status(status.size());
  for (int i = 0; i < status.size(); ++i) {
    _status(i) = status(event_order(i));
  }

  Eigen::VectorXi _first = first;
  
  Eigen::VectorXi _start_map(start_map.size());
  for (int i = 0; i < start_map.size(); ++i) {
    _start_map(i) = start_map(event_order(i));
  }

  Eigen::VectorXi _event_map = event_map;

  Eigen::VectorXi _event(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _event(i) = event(event_order(i));
  }

  Eigen::VectorXi _start(event.size());
  for (int i = 0; i < event.size(); ++i) {
    _start(i) = event(start_order(i));
  }

  std::vector<int> last_vec;
  int last_event = nevent - 1, first_size = first.size();
  for (i = 0; i < _first_size; ++i) {
    int f = _first(first_size - i - 1);
    last_vec.push_back(last_event);
    // immediately following a last event, `first` will agree with np.arange
    if (f - (nevent - 1 - i) == 0) {
      last_event = f - 1;
    }
  }

  int last_vec_size = last_vec.size();
  Eigen::VectorXi _last(last_vec_size);
  for (int i = 0; i < _last.size(); ++i) {
    _last(i) = last(last_vec_size - i - 1);
  }

  Eigen::VectorXd _scaling(nevent);
  for (int i = 0; i < nevent; ++i) {
    double fi = (double) _first(i);
    _scaling(i) = ((double) i - fi) / ((double) _last(i) + 1.0 - fi);
  }

#ifdef PY_INTERFACE
  py::dict preproc;
  preproc["start"] = _start;
  preproc["event"] = _event;
  preproc["first"] = _first;
  preproc["last"] = _last;
  preproc["scaling"] = _scaling;
  preproc["start_map"] = _start_map;
  preproc["event_map"] = _event_map;
  preproc["status"] = _status;
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
  
}
