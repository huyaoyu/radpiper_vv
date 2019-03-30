#!/usr/bin/python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 20180817

import argparse
import json
import numpy as np
import os
import pandas as pd

DEBUG_FLAG = False

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

def find_radius_by_angle(radii, angles, gapWidthThreshold = 2):
    # Check the dimensions.
    
    if ( radii.shape != angles.shape ):
        raise Exception("radii({}) and angles({}) have different dimensions.".format(radii.shape, angles.shape))

    idxFiniteAngles = np.where( np.isfinite( angles ) )[0]
    finiteAngles    = angles[idxFiniteAngles]

    # Prepare two matrix.
    m0, m1 = np.meshgrid(finiteAngles, finiteAngles)

    m1       += np.pi
    mask      = m1 >= np.pi
    m1[mask] -= 2 * np.pi

    # Subtract.
    s  = m0 - m1
    sa = np.fabs(s)

    matchedRadii = np.ones_like(radii) * np.inf

    for i in range(s.shape[0]):
        # Find the closest index.
        idxfa = np.argmin( sa[i, :] )

        idx = idxFiniteAngles[idxfa]

        # if ( DEBUG_FLAG == True and idxfa == 179 ):
        #     idxfa = idxfa

        if ( s[i, idxfa] > 0 ):
            # Look to the left.
            if ( np.isfinite( angles[ idx - 1 ] ) ):
                matchedRadii[idxFiniteAngles[i]] = radii[idx]
            else:
                # Figure out the gap width.
                idxLeft = idxFiniteAngles[ idxfa - 1 ]

                if ( idxLeft > idx ):
                    # Loop back.
                    gapWidth = angles.size - idxLeft + idx - 1
                else:
                    gapWidth = idx - idxLeft - 1
                
                if ( gapWidth <= gapWidthThreshold ):
                    matchedRadii[idxFiniteAngles[i]] = radii[idx]
        elif ( s[i, idxfa] < 0 ):
            # Look to the right.
            if ( idx == angles.shape[0] - 1 ):
                newIdx = 0
            else:
                newIdx = idx + 1

            if ( np.isfinite( angles[newIdx] ) ):
                matchedRadii[idxFiniteAngles[i]] = radii[idx]
            else:
                # Figure out the gap width.
                if ( idxfa == idxFiniteAngles.size - 1 ):
                    newIdxfa = 0
                else:
                    newIdxfa = idxfa + 1

                idxRight = idxFiniteAngles[ newIdxfa ]

                if ( idxRight < idx ):
                    # Loop back.
                    gapWidth = angles.size - idx + idxRight - 1
                else:
                    gapWidth = idxRight - idx - 1
                
                if ( gapWidth <= gapWidthThreshold ):
                    matchedRadii[idxFiniteAngles[i]] = radii[idx]
        else:
            matchedRadii[idxFiniteAngles[i]] = radii[idx]

    return matchedRadii

def find_valid_diameters_by_angle(raw):
    """
    raw: A 2D NumPy array. Each row of the array is an entry.

    returns: A 2D NumPy array.
    """

    # Check the dimension.
    if ( 720 != raw.shape[1] ):
        raise Exception("Wrong shape. shape = {}.".format(raw.shape))

    d = np.zeros((raw.shape[0], 360), dtype = np.float)

    # Process row by row.
    for i in range( raw.shape[0] ):

        # if ( i == 1753 ):
        #     global DEBUG_FLAG 
        #     DEBUG_FLAG = True
        #     import ipdb; ipdb.set_trace()

        r = raw[i, :][  0:360]
        a = raw[i, :][360:720]

        mr = find_radius_by_angle(r, a, 2)

        d[i, :] = r + mr

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

def count_against_voiddiam_m(dataBinList, voiddiam_m, void_qty, divider = 1):
    void_flag = np.zeros((len(dataBinList)), dtype = np.int)

    for i in range(len(dataBinList)):
        db = dataBinList[i]
        count = db.count_voiddiam(voiddiam_m)

        count = (int)( 1.0 * count / divider + 0.5 )

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

    fnCentered = workingDir + "/" + params["rplidar_raw"]
    print("Reading %s..." % (fnCentered))
    rplidarRaw = read_csv_as_array(fnCentered, header = None)

    fnScanBins = workingDir + "/" + params["scan_bins"]
    print("Reading %s..." % (fnScanBins))
    scanBins = read_csv_as_array(fnScanBins)

    # Check the dimensions.
    if ( rplidarRaw.shape[0] != scanBins.shape[0] ):
        raise Exception("The dimensions are not compatible.")

    # ==================================================
    # =                  Diameters.                    =
    # ==================================================

    # diameters = find_valid_diameters(rplidarRaw[:, 2:362])
    diameters = find_valid_diameters_by_angle(rplidarRaw[:, 2:722])

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

    # =========================================================
    # = Count the number of diameters larger than voiddiam_m. =
    # =========================================================

    print_delimeter(title = "void_flag.")
    
    void_flag = count_against_voiddiam_m(dataBinList, voiddiam_m, void_qty, 2)

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
