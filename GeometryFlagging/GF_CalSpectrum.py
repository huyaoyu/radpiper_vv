#!/usr/bin/python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 20180817

import argparse
import json
import numpy as np
import os
import pandas as pd

DEFAULT_WORKING_DIR = "./data"
DEFAULT_INPUT = "input.json"

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range(n/2)]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

def read_csv_as_array(csvFilename, header = 0, omit = None):
    # Read the csv file.
    df = pd.read_csv(csvFilename, header = header)

    # Convert the data into NumPy array.
    if ( omit is None ):
        array = df.values
    else:
        array = df.values[ omit:, : ]

    print("read_csv_as_array: {} read. shape = {}.".format(csvFilename, array.shape))

    return array

def read_csv_columns_as_array(csvFilename, cols):
    # Read the csv file.
    df = pd.read_csv(csvFilename, header = None, usecols=cols, skiprows=[0])

    array = df.values

    print("read_csv_columns_as_array: {} read. shape = {}.".format(csvFilename, array.shape))

    return array 

def find_valid_diameters(raw):
    """
    raw: A 2D NumPy array. Each row of the array is an entry.

    returns: A 2D NumPy array.
    """

    # Check the dimension.
    if ( 360 != raw.shape[1] ):
        raise Exception("Wrong shape. shape = {}.".format(raw.shape))

    d = raw[:, 0:180] + raw[:, 180:]

    return d

def remove_according_to_segment(diameters, scanBins, segmentColumnIndex = 1):

    mask = scanBins[:, segmentColumnIndex] > 0.5

    maskedD  = diameters[mask, :]
    maskedSB = scanBins[mask, :]

    return maskedD, maskedSB

def mean_and_deviation(d):

    # mask = ~( np.isinf(d) | np.isnan(d) )
    mask = np.isfinite(d)
    v = d[mask]

    m = np.mean(v)
    s = np.std(v)

    return m, s

class DataBin:
    def __init__(self, diameters, scanBins, segment, forward):
        self.diameters = diameters
        self.scanBins  = scanBins
        self.segment   = segment
        self.forward   = forward

    def count_voiddiam(self, voiddiam_m):
        # Note that for NumPy, inf > any real number. Compare a real number with nan is False,
        # however, it triggers a runtime warning.

        mask = ~np.isnan( self.diameters )

        count = np.sum( self.diameters[mask] > voiddiam_m )

        return count

    def random_sample(self, n, voiddiam_m = None):
        """
        n: Positive integer. The number of samples.

        This function heavyly relys on the NumPy.random.choice() function.
        That means for a single set of sampled elements, identical ones may
        be in the sample.
        """

        if ( n <= 0 ):
            raise Exception("wrong sample number n = {}".format(n))

        # mask = ~( np.isinf(self.diameters) | np.isnan(self.diameters) )
        mask = np.isfinite( self.diameters )

        tempD = self.diameters[mask]

        if ( voiddiam_m is not None ):
            maskV = tempD <= voiddiam_m
            tempD = tempD[maskV]

        if ( n > tempD.size ):
            raise Exception("n = {} is too large. Allowed value up to {}".format(n, tempD.size))

        return np.random.choice(tempD, size = n)

def split_into_DataBins(diameters, scanBins, segmentColumnIndex = 1, forwardColumnIndex = 3):
    # Temparory arrays.
    segArray = scanBins[:, segmentColumnIndex].astype(np.int)
    forArray = scanBins[:, forwardColumnIndex].astype(np.int)
    
    # Figure out the segments.
    segments = np.unique( segArray ).astype(np.int)

    # Figure out the forward flags.
    forwardFlags = np.unique( forArray ).astype(np.int)

    dataBinList = []

    for f in forwardFlags:
        maskForArray = forArray == f
        for s in segments:
            mask = ( np.logical_and( maskForArray, segArray == s ) )

            db = DataBin( diameters[mask, :], scanBins[mask, :], s, f )
            dataBinList.append(db)

            print("Create DataBin object with s = %2d, f = %d, num = %d." % (s, f, db.diameters.shape[0]))

    return dataBinList

def find_valid_indices_by_time_frame(endTimestamp, timeSpan, timestampArray, tolerance):
    """
    endTimestamp: A two element NumPy array. Integers. [second, nanosecond].
    timeSpan: A floating point number. Seconds passed.
    timestampArray: A 2D NumPy matrix. Each row is like an endTimestamp.
    tolerance: A positive floating point number. If the staring or ending index is the first or the last
        index of timestampArray, a further check will be made. The check will test whether the 
        starting time or the ending time is in the vicinity of the first element or the last
        element of timestampArray within a range defined by tolerance.

    return: A two element list. Valid starting and ending index in timestampArray.
    """

    # Compose floating point number representations of endTimestamp and timestampArray.
    endT   = endTimestamp[0] + endTimestamp[1] / 1e9
    ts     = timestampArray[:, 0] + timestampArray[:, 1] / 1e9
    startT = endT - timeSpan

    # Find the nearest timestamp of starting point.
    idxStart = np.argmin( np.fabs( ts - startT ) )
    idxEnd   = np.argmin( np.fabs( ts -   endT ) )

    if ( idxStart == 0 and ts[0] < startT):
        if ( ts[0] - startT > tolerance ):
            raise Exception("Starting timestamp out of range.")

    if ( idxEnd == ts.size - 1 and endT > ts[-1] ):
        if ( endT - ts[-1] > tolerance):
            raise Exception("Ending timestamp out of range.")

    return [idxStart, idxEnd]

def count_against_voiddiam_m(dataBinList, voiddiam_m, void_qty):
    void_flag = np.zeros((len(dataBinList)), dtype = np.int)

    for i in range(len(dataBinList)):
        db = dataBinList[i]
        count = db.count_voiddiam(voiddiam_m)

        if ( count > void_qty ):
            void_flag[i] = 1

        print("s = %2d, f = %d, count = %d, void_flag = %d." % (db.segment, db.forward, count, void_flag[i]))

    return void_flag

def randoms_sample_on_DataBins(dataBinList, voiddiam_m, thicksample_qty, thicktry_qty):
    # Allocate chunk of memory.

    nDataBins = len( dataBinList )

    sampledMeans = np.zeros( (nDataBins, thicktry_qty), dtype = np.float )

    # Loop.
    for i in range(nDataBins):
        db = dataBinList[i]

        print("Random sample over bin (s = %2d, f = %d)." % (db.segment, db.forward))

        for j in range(thicktry_qty):
            sd = db.random_sample(thicksample_qty, voiddiam_m)
            sampledMeans[i, j] = np.mean(sd)
    
    return sampledMeans


def statistics_of_random_sampled_DataBins(dataBinList, randomSampledDiam):
    # Allocate memory.
    statistics = np.zeros((len(dataBinList), 3), dtype = np.float)
    
    for i in range( len(dataBinList) ):
        db = dataBinList[i]

        m   = np.mean( randomSampledDiam[i, :] )
        std = np.std(  randomSampledDiam[i, :] )

        statistics[i, :] = [ m, std, m - 2.0 * std ]

        print( "bin (s = %2d, f = %d) random sampled diameter. Mean = %.6e, std = %.6e, 95%% = %.6e" % \
            ( db.segment, db.forward, statistics[i, 0], statistics[i, 1], statistics[i, 2] ) )

    return statistics

def get_thick_flag(dataBinList, diamStatistics, thickdiam_m):
    thick_flag = np.zeros( (len(dataBinList)), dtype = np.int )

    for i in range( len(dataBinList) ):
        db = dataBinList[i]

        if ( thickdiam_m > diamStatistics[i, 2] ):
            thick_flag[i] = 1
        
        print("bin (s = %2d, f = %d) thick_flag = %d." % (db.segment, db.forward, thick_flag[i]))

    return thick_flag

def plain_report(params, dMean, dStd, dataBinList, void_flag, randomSampledDiam, statisticsOfRandSampledDiam, thick_flag):
    print("""
==========================================
            Test %d summary.
==========================================
    """ % (params["testNumber"]))

    print("The mean and std. deviation for all diameters which are neither Inf nor NaN are %.12e, %.12e" % (dMean, dStd))

    print("voiddiam_m      = %.12e" % (params["voiddiam_m"]))
    print("void_qty        = %d" % (params["void_qty"]))
    print("thickdiam_m     = %.12e" % (params["thickdiam_m"]))
    print("thicksample_qty = %d" % (params["thicksample_qty"]))
    print("thicktry_qty    = %d" % (params["thicktry_qty"]))

    nDataBins = len(dataBinList)

    print("segment, forward, void_flag,      rand. mean,       rand. std.,      rand. 95%%,    thick_flag")
    for i in range(nDataBins):
        db = dataBinList[i]

        print("     %2d,       %d,         %d, %.12e, %.12e, %.12e,   %d" % \
            (db.segment, db.forward, void_flag[i], \
            statisticsOfRandSampledDiam[i, 0], statisticsOfRandSampledDiam[i, 1], statisticsOfRandSampledDiam[i, 2], \
            thick_flag[i]))

if __name__ == "__main__":
    # ==================================================
    # =              Input arguments.                  =
    # ==================================================

    parser = argparse.ArgumentParser(description="Run Test 55. Re-implement geometry flagging (thickness and hole).")

    parser.add_argument("--input", help = "The filename of the input JSON file.", default = DEFAULT_WORKING_DIR + "/" + DEFAULT_INPUT)
    parser.add_argument("--voiddiam_m",\
        help = "Overwrite voiddiam_m in the input JSON file.",\
        default = -1.0, type = float)
    parser.add_argument("--void_qty",\
        help = "Overwrite void_qty in the input JSON file.",\
        default = -1, type = int)
    parser.add_argument("--thickdiam_m",\
        help = "Overwrite thickdiam_m in the input JSON file.",\
        default = -1.0, type = float)
    parser.add_argument("--thicksample_qty",\
        help = "Overwrite thicksample_qty in the input JSON file.",\
        default = -1, type = int)
    parser.add_argument("--thicktry_qty",\
        help = "Overwrite thicktry_qty in the input JSON file.",\
        default = -1, type = int)
    parser.add_argument("--write", help = "Write multiple arrays to file system.", action = "store_true", default = False)
    parser.add_argument("--s0", help = "Do not filter out segment 0.", action = "store_true", default = False)

    args = parser.parse_args()

    inputFp = open(args.input)
    params = json.load(inputFp)
    workingDir = params["workingDir"]

    if ( args.voiddiam_m > 0.0 ):
        voiddiam_m = args.voiddiam_m
    else:
        voiddiam_m = params["voiddiam_m"]

    if ( -1 != args.void_qty ):
        void_qty = args.void_qty
    else:
        void_qty = params["void_qty"]

    if ( args.thickdiam_m > 0.0 ):
        thickdiam_m = args.thickdiam_m
    else:
        thickdiam_m = params["thickdiam_m"]

    if ( -1 != args.thicksample_qty ):
        thicksample_qty = args.thicksample_qty
    else:
        thicksample_qty = params["thicksample_qty"]

    if ( -1 != args.thicktry_qty ):
        thicktry_qty = args.thicktry_qty
    else:
        thicktry_qty = params["thicktry_qty"]

    output = workingDir + "/" + params["output"]

    if ( True == args.write ):
        if ( False == os.path.isdir( output ) ):
            os.makedirs(output)

    # ==================================================
    # =                  Read files.                   =
    # ==================================================

    fnCentered = workingDir + "/" + params["centered"]
    print("Reading %s..." % (fnCentered))
    centered = read_csv_as_array(fnCentered, header = None)

    fnRplidarScan = workingDir + "/" + params["rplidarScan"]
    print("Reading %s..." % (fnRplidarScan))
    rplidarScan = read_csv_columns_as_array(fnRplidarScan, cols = [3, 4])

    fnCalSpectrum = workingDir + "/" + params["calSpectrum"]
    print("Reading %s..." % (fnCalSpectrum))
    calSpectrum = read_csv_columns_as_array(fnCalSpectrum, cols = [3, 4, 7])

    fnScanBins = workingDir + "/" + params["scanBins"]
    print("Reading %s..." % (fnScanBins))
    scanBins = read_csv_as_array(fnScanBins)

    # Check the dimensions.
    if ( centered.shape[0] != scanBins.shape[0] or \
         centered.shape[0] != rplidarScan.shape[0] ):
        raise Exception("The dimensions are not compatible.")

    # ==================================================
    # =                  Diameters.                    =
    # ==================================================

    diameters = find_valid_diameters(centered[:, 2:362])

    if ( True == args.write ):
        np.savetxt( output + "/diameters.dat", diameters, fmt = "%.6e" )

    # ==================================================
    # =        Remove all Segment = 0 rows.            =
    # ==================================================

    if ( True == args.s0 ):
        diametersM = diameters
        scanBinsM  = scanBins
    else:
        diametersM, scanBinsM = remove_according_to_segment(diameters, scanBins)

    if ( True == args.write ):
        np.savetxt( output + "/diametersM.dat", diametersM, fmt = "%.6e" )
        np.savetxt( output + "/scanBinsM.dat",   scanBinsM, fmt = "%.4f" )

    # ==================================================
    # = Mean and standard deviation of the diameters.  =
    # ==================================================

    dMean, dStd = mean_and_deviation(diametersM)

    print( "Mean and standard deviation of the diameters are (omitting the rows with segment = 0): %.6e, %.6e" % (dMean, dStd) )

    # =========================================================
    # = Split the diameters according to segment and forward. =
    # =========================================================

    print_delimeter(title = "Create data bins.")

    dataBinList = split_into_DataBins(diametersM, scanBinsM)

    # Must be only one DataBin object.

    if ( len(dataBinList) != 1 ):
        raise Exception("Too many DataBin objects. len(dataBinList) = %d." % (len(dataBinList)))

    # Find the valid timestamp indices.
    [idxStart, idxEnd] = find_valid_indices_by_time_frame(\
        calSpectrum[0, 0:2], calSpectrum[0, 2], rplidarScan, 1e-6)

    print("The valid indices based in %s are [%d, %d]." % (params["rplidarScan"], idxStart, idxEnd))

    # Slice the data from the only DataBin object.
    dataBinList[0].diameters = dataBinList[0].diameters[ idxStart:idxEnd+1, : ]
    dataBinList[0].scanBins  = dataBinList[0].scanBins[ idxStart:idxEnd+1, : ]

    # =========================================================
    # = Count the number of diameters larger than voiddiam_m. =
    # =========================================================

    print_delimeter(title = "void_flag.")

    void_flag = count_against_voiddiam_m(dataBinList, voiddiam_m, void_qty)

    # ============================================================
    # = Random sample from diameters for each segment-direction. =
    # ============================================================

    print_delimeter(title = "Random sample and mean calculation.")

    randomSampledDiam = randoms_sample_on_DataBins(dataBinList, voiddiam_m, thicksample_qty, thicktry_qty)

    # ====================================================
    # = Statistics of the random sampled mean diameters. =
    # ====================================================

    statistics = statistics_of_random_sampled_DataBins(dataBinList, randomSampledDiam)

    # ====================================================
    # =                thick_flag vector.                =
    # ====================================================

    print_delimeter(title = "thick_flag.")

    thick_flag = get_thick_flag(dataBinList, statistics, thickdiam_m)

    # ====================================================
    # =                     Report.                      =
    # ====================================================

    print_delimeter(title = "Report.")

    plain_report(params, dMean, dStd, dataBinList, void_flag, randomSampledDiam, statistics, thick_flag)

    if ( args.voiddiam_m > 0.0 ):
        print("%s (%.12e) overwritten by command line argument %s %.12e" % ("voiddiam_m", params["voiddiam_m"], "--voiddiam_m", args.voiddiam_m))

    if ( -1 != args.void_qty ):
        print("%s (%d) overwritten by command line argument %s %d" % ("void_qty", params["void_qty"], "--void_qty", args.void_qty))

    if ( args.thickdiam_m > 0.0 ):
        print("%s (%.12e) overwritten by command line argument %s %.12e" % ("thickdiam_m", params["thickdiam_m"], "--thickdiam_m", args.thickdiam_m))

    if ( -1 != args.thicksample_qty ):
        print("%s (%d) overwritten by command line argument %s %d" % ("thicksample_qty", params["thicksample_qty"], "--thicksample_qty", args.thicksample_qty))

    if ( -1 != args.thicktry_qty ):
        print("%s (%d) overwritten by command line argument %s %d" % ("thicktry_qty", params["thicktry_qty"], "--thicktry_qty", args.thicktry_qty))

    print("Done.")
