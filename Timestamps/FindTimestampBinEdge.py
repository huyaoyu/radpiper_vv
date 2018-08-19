#!/usr/bin/python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 20180816

import numpy as np

def generate_random_bin_edges(tsStart, tsNum, tsStepMean, tsStepSigma):
    """
    Generate random timestamps starting from tsStart. The interval between two timestamps
    follows a normal distribution with a mean value of tsStepMean and a standard deviation
    of tsStepSigma. The total number of timestamps to be generated is tsNum.

    tsStart: Positive integer.
    tsNum: Positive integer. Must larger than 2.
    tsStepMean: Positive integer.
    tsStepSigma: Non-negative integer. Must be smaller than tsStepMean.

    return: A NumPy array. All the timestamps generated. The timestamps are represeneted by 
    the int64 data type of NumPy.
    """

    # Argument check.
    assert tsStart     > 0
    assert tsNum       > 2
    assert tsStepMean  > 0
    assert tsStepSigma >= 0
    assert tsStepSigma < tsStepMean

    # timeStep sequence.
    timeSteps = tsStepSigma * np.random.randn( (int)( tsNum ) ) + tsStepMean

    timestamps = [tsStart]

    for step in timeSteps:
        timestamps.append( timestamps[-1] + step )

    bins = np.stack(( timestamps[0:-1], timestamps[1:] ), axis = 1)

    return bins

def naive_match(ts, tsArray, threshold = None):
    """
    Find the closest value in tsArray.

    ts: A floating point value.
    tsArray: A NumPy 1D array. Already in ascending order.
    """

    if ( ts < tsArray[0]):
        if ( threshold is None ):
            return -1
        else:
            if ( tsArray[0] - ts <= threshold ):
                return 0
            else:
                return -1

    if ( ts > tsArray[-1] ):
        if ( threshold is None ):
            return -1
        else:
            if ( ts - tsArray[-1] <= threshold ):
                return tsArray.size - 1
            else:
                return -1

    for i in range( tsArray.size ):
        if ( tsArray[i] == ts):
            return i
        elif ( tsArray[i] < ts ):
            continue

        if ( tsArray[i-1] < ts and ts < tsArray[i] ):
            if ( ts - tsArray[i-1] <= tsArray[i] - ts ):
                return i - 1
            else:
                return i
    
    # Did not find a index.
    return -1

if __name__ == "__main__":
    # Simple test with out any data.
    
    # Get a random set of bin edges.
    bins = generate_random_bin_edges(1000000001, 100, 10000000, 1000000)

    # Turn binEdges into floating point values.
    binsF = bins / 100.0

    # Get an array of timestamps based on binEdges.
    binNum = bins.shape[0]
    tsArray = np.linspace( binsF[0, 0], binsF[binNum - 1, 1], binNum * 10 )

    # Get the index and timestamp based on binEdges and tsArray.

    idxList = []

    for binEdge in binsF:
        bStart = binEdge[0]
        bEnd   = binEdge[1]

        # Find the starting timestamp.
        idxStart = naive_match(bStart, tsArray)
        idxEnd   = naive_match(bEnd,   tsArray)

        idxList.append((idxStart, idxEnd))
