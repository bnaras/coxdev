import numpy as np
import pickle as pk
from coxc import preproc as c_preproc

def preprocess(start,
               event,
               status):
    """
    Compute various functions of the start / event / status
    to be used to help in computing cumsums
    
    This can probably stay in python, and have a separate
    implementation in R
    """
    
    start = np.asarray(start)
    event = np.asarray(event)
    status = np.asarray(status)
    nevent = status.shape[0]
    
    # second column of stacked_array is 1-status...
    stacked_time = np.hstack([start, event])
    stacked_status_c = np.hstack([np.ones(nevent, int), 1-status]) # complement of status
    stacked_is_start = np.hstack([np.ones(nevent, int), np.zeros(nevent, int)])
    stacked_index = np.hstack([np.arange(nevent), np.arange(nevent)])
    
    argsort = np.lexsort((stacked_is_start,
                          stacked_status_c,
                          stacked_time))
    sorted_time = stacked_time[argsort]
    sorted_status = 1 - stacked_status_c[argsort]
    sorted_is_start = stacked_is_start[argsort]
    sorted_index = stacked_index[argsort]
    
    # do the joint sort

    event_count, start_count = 0, 0
    event_order, start_order = [], []
    start_map, event_map = [], []
    first = []

    last_row = None
    which_event = -1
    first_event = -1
    num_successive_event = 1
    ties = {}    
    for row in zip(sorted_time,
                   sorted_status,
                   sorted_is_start,
                   sorted_index):
        (_time, _status, _is_start, _index) = row
        if _is_start == 1: # a start time
            start_order.append(_index)
            start_map.append(event_count)
            start_count += 1
        else: # an event / stop time
            if _status == 1:
                # if it's an event and the time is same as last row 
                # it is the same event
                # else it's the next "which_event"
                
                if (last_row is not None and 
                    _time != last_row[0]): # index of next `status==1`
                    first_event += num_successive_event
                    num_successive_event = 1
                    which_event += 1
                else:
                    num_successive_event += 1
                    
                first.append(first_event)
            else:
                first_event += num_successive_event
                num_successive_event = 1
                first.append(first_event) # this event time was not an failure time

            event_map.append(start_count)
            event_order.append(_index)
            event_count += 1
        last_row = row

    first = np.array(first)
    start_order = np.array(start_order, int)
    event_order = np.array(event_order, int)
    start_map = np.array(start_map, int)
    event_map = np.array(event_map, int)

    # reset start_map to original order
    start_map_cp = start_map.copy()
    start_map[start_order] = start_map_cp

    # set to event order

    _status = status[event_order]
    _first = first
    _start_map = start_map[event_order]
    _event_map = event_map

    _event = event[event_order]
    _start = event[start_order]

    # compute `last`
    #pdb.set_trace()
    last = []
    last_event = nevent-1
    for i, f in enumerate(_first[::-1]):
        last.append(last_event)
        # immediately following a last event, `first` will agree with np.arange
        if f - (nevent - 1 - i) == 0:
            last_event = f - 1        
    _last = np.array(last[::-1])

    den = _last + 1 - _first

    # XXXX
    _scaling = (np.arange(nevent) - _first) / den
    
    preproc = {'start':np.asarray(_start),
               'event':np.asarray(_event),
               'first':np.asarray(_first),
               'last':np.asarray(_last),
               'scaling':np.asarray(_scaling),
               'start_map':np.asarray(_start_map),
               'event_map':np.asarray(_event_map),
               'status':np.asarray(_status)}
    return preproc, event_order, start_order

pkl_file = '/Users/naras/GitHub/coxdev/R_pkg/coxdev/inst/out100.pkl'
f = open(pkl_file, 'rb')
res = pk.load(f)
f.close()

_pre, _eo, _so = preprocess(res['start'], res['event'], res['status'])

a = res['start']
b = res['event']
c = res['status'].astype("int32")

p1, e1, s1 = c_preproc(a, b, c)
