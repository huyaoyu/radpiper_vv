#!/usr/bin/python

from __future__ import print_function

# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 20180818

import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit

DEFAULT_INPUT = "input.json"

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range(n/2)]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

G_G_TEMP    = np.sqrt(2.0 * np.pi)
G_HALF_TEMP = math.sqrt( -2.0 * math.log(0.5) )

def scaled_Gaussian(x, mu, sigma):
    return ( 1.0 / ( G_G_TEMP * sigma ) * np.exp( -np.power( (x-mu)/sigma, 2 ) / 2 ))

class QcSpectrumEntry(object):
    def __init__(self, spectrumInfo, collectTime, counts):
        """
        spectrumInfo: String.
        collectTime: Float.
        counts: A NumPy 1D array, integers.
        """

        self.spectrumInfo = spectrumInfo
        self.collectTime  = collectTime
        self.counts       = counts
        self.cpsSpectrum  = None
        self.cpsInPeak    = None
        self.roi          = None # One based index.

        self.peaks        = [] # Should be a 1D list, containing the channel number of the peaks.
        self.peakHM       = [] # Should be a 2D list, containing the lower and upper bounds of the half max of the peaks.
        self.peakHeight   = [] # Should be a 1D list, containing the heights of the peaks.

    def set_roi(self, roi):
        """
        roi: A two element list. Elements should be integers, however, a cast to int is enforced
        """

        self.roi = [ (int)( roi[0] ), (int)( roi[1] ) ]

    def counts_per_second(self):
        """
        Calculate the counts per second for the spectrum and roi, if specified.
        """

        self.cps         = self.counts / ( 1.0 * self.collectTime )
        self.cpsSpectrum = np.sum( self.counts ) / (1.0 * self.collectTime )
        self.cpsInPeak   = np.sum( self.counts[ self.roi[0]-1:self.roi[1]-1 ] ) / (1.0 * self.collectTime )
    
    def find_peak_by_fit(self, data, idx, groupNum):
        """
        data: NumPy 1D array.
        idx: Non-negative integer. The index around which to find the peak.
        groupNumber: Positive integer. The number of points need to sample.

        return: mu, halfMaxList, h
            mu: Floating point number, the channel of the peak. None if failed
            halfMaxList: A two-element 1D list, the lower and upper bound of the half max width. None.
            h: Floating point number, the peak height. None if failed
        """

        mu          = None
        halfMaxList = None
        h           = None

        # Gather data.
        halfGN = (int)( math.floor( groupNum / 2.0 ) )

        if ( idx < halfGN or idx > len(data) - 1 - halfGN ):
            # Simply refuse to calculate since there won't be enough data.
            return mu, halfMaxList, h
        
        x = np.linspace( idx - halfGN, idx + halfGN, groupNum, dtype = np.int )
        y = data[x]

        # Normalize y.
        sumY = np.sum(y)
        y = y / sumY

        # Curve fitting.
        try:
            # Initial guess.
            p0 = [ idx, 1.0 / (G_G_TEMP * np.max(y)) ]
            popt, pcov = curve_fit(scaled_Gaussian, x, y, p0 = p0)
        except RuntimeError as rEx:
            print("Failed to fit a Gaussian")
            return mu, halfMaxList, h
        except scipy.optimize.OptimizeWarning as owEx:
            print("Cannot estimate the covariance of the parameters of the Gaussian.")
        
        mu    = popt[0]
        sigma = popt[1]
        
        # Find the half max of the Gaussian.
        halfMaxList = [ -G_HALF_TEMP * sigma + mu, G_HALF_TEMP * sigma + mu ]

        # Peak height.
        h = scaled_Gaussian(mu, *popt) * sumY

        return mu, halfMaxList, h

    def find_peaks(self, gWindowWidth, gStd, slopeThreshold, ampThreshold, peakGroup):
        """
        gWindowWidth: Positive ingeter, the windows width of the Gaussian filter.
        gStd: Positive real number, the standard deviation of the Gaussian filter.
        slopeThreshold: The threshold of the gradients to be considered as a potential peak.
        ampThreshold: The threshold of the height of the cps to be considered as a potential peak.
        peakGroup: A positive integer. Usually larger than 2. The points to take into consideration when finding a potential peak by curve fitting.
        """

        # Calculate the derivatvie.
        diff = np.gradient( self.cps )

        # Gaussian filtering.
        gaussian = signal.gaussian(gWindowWidth, gStd)
        gaussian = gaussian / np.sum( gaussian ) 

        # Filter by convolution.
        convDiff = signal.convolve( diff, gaussian, mode = "same" )

        # Find the zero-crossing index in convDiff.
        zeroCrossingIdxTuple = np.where( np.diff( np.sign(convDiff) ) )
        zeroCrossingIdx = zeroCrossingIdxTuple[0]

        if ( 0 == zeroCrossingIdx.size ):
            # No peaks found.
            return

        dzCrossingIdx = []

        # Loop over the potential peaks.
        for idx in zeroCrossingIdx:
            # Check if it is a downwards zero-crossing.
            if ( convDiff[idx] >= 0 and convDiff[idx+1] <= 0 ):
                dzCrossingIdx.append(idx)

        if ( 0 == len(dzCrossingIdx) ):
            # No peaks found.
            return

        # Find peak.
        for i in range(len(dzCrossingIdx)):
            idx = dzCrossingIdx[i]

            # Check the slope.
            if ( self.cps[idx] < ampThreshold ):
                continue

            if ( convDiff[idx] < slopeThreshold ):
                continue

            # May have a potential peak here.
            mu, halfMaxList, height = self.find_peak_by_fit(self.cps, idx, peakGroup)

            if (mu is None):
                # Failed to find a peak.
                continue
            else:
                self.peaks.append(mu)
                self.peakHM.append(halfMaxList)
                self.peakHeight.append(height)

    def get_num_peaks(self):
        return len(self.peaks)

    def get_peak_channel(self, peakChannelLow, peakChannelHigh):
        """
        Find the peak channels that fall in the range defined by peakChannelLow and peakChannelHigh.

        Usually peakChannelLow and peakChannelHigh should be integers. However, this funcion
        accepts real numbers. Validity check will be made to ensoure that
        peakChannelHigh - peakChannelLow >= 2. If this condition is not met, an exception
        will be raised.

        peakChannelLow: Positive number. One based channel number.
        peakChannelHigh: Positive number. One based channel number. Should be larger than peakChannelLow.

        return: A list contains all the valid channels. The channels will be cast to their nearest integers.
            Note that channel number is one based. If no valid channel is found, a empty list will be returned.
        """

        # Validity check.
        if ( peakChannelLow <= 0 or peakChannelHigh <= 0 ):
            raise Exception("peakChannelLow (%f) and peakChannelHigh (%f) should be positive numbers." % \
                (peakChannelLow, peakChannelHigh) )
        
        if ( peakChannelHigh - peakChannelLow < 2 ):
            raise Exception("peakChannelHigh - peakChannelLow < 2")
        
        # Find valid channels.
        validChannels = []
        for p in self.peaks:
            if ( p >= peakChannelLow and p <= peakChannelHigh ):
                validChannels.append( int( p + 0.5 ) + 1 )
        
        return validChannels

    def show_peak_info(self):
        print("%s has %d peaks found." % (self.spectrumInfo, len(self.peaks)))

        for i in range( len(self.peaks) ):
            print("Peak %d, c = %.4e, FWHM = %.4e, height = %.4e" % \
                ( i + 1,\
                self.peaks[i],\
                self.peakHM[i][1] - self.peakHM[i][0],\
                self.peakHeight[i]) )

def convert_counts_string(countsStr, d = "|"):
    """
    Convert the string representation of the "counts" column into a 
    NumPy 1D array with integer data type.
    """
    
    # Remove the last delimeter if there is any.
    s = countsStr.strip()

    if ( s[-1] == d ):
        s = s[:-1]

    # Split the string.
    countList = s.split(d)

    # Create the NumPy array.
    counts = np.array( countList ).astype( np.int )

    return counts

def read_and_create_QcSpectrumEntryDict(dataDir, qcSpectrumDict):
    """
    Read the qc_spectrum.csv file and create a dictionary contains the
    entries in that file. There should be only two entries, one for the start, 
    the other for the end.

    qcSpectrumDict: The dictionary created by the input JSON file.
    """

    # Read the csv file.
    fn = dataDir + "/" + qcSpectrumDict["filename"]
    df = pd.read_csv( fn )

    qcSpectrumEntryDict = {}

    # Loop.
    for index, row in df.iterrows():
        spectrumInfo = row[qcSpectrumDict["spectrum_info"]]

        if ( qcSpectrumDict["start"] == spectrumInfo or qcSpectrumDict["end"] == spectrumInfo ):
            qcSpectrumEntryDict[ spectrumInfo ] = \
                QcSpectrumEntry( spectrumInfo, \
                    row[ qcSpectrumDict["collect_time"] ], \
                    convert_counts_string( row[ qcSpectrumDict["counts"] ] ) )
        else:
            raise Exception("Unexpected entry with %s = %s." % ( qcSpectrumDict["spectrum_info"], spectrumInfo ))

    return qcSpectrumEntryDict

def read_and_create_df_QcHistiory(dataDir, qcHistoryDict):
    """
    Read the qc_history.csv file and extract the last two rows. These two rows
    must be the "start" and "end" QC checking entries.

    The csv file is read by pandas, and the rows will be returned as pandas data frame.
    """

    # Read the csv file and select the last two rows only.
    fn = dataDir + "/" + qcHistoryDict["filename"]
    df = pd.read_csv( fn, index_col = False, low_memory = False ).tail(2)

    # Check the content.
    try:
        forLoc = df[ qcHistoryDict["qc_type"] ] == qcHistoryDict["start"]
        s = df.loc[ forLoc ]
    except KeyError as exp:
        print("Count not find %s = %s." % (qcHistoryDict["qc_type"], qcHistoryDict["start"]))
        raise
    
    try:
        e = df.loc[ df[ qcHistoryDict["qc_type"] ] == qcHistoryDict["end"] ]
    except KeyError as exp:
        print("Count not find %s = %s." % (qcHistoryDict["qc_type"], qcHistoryDict["end"]))
        raise

    return df

if __name__ == "__main__":
    # ==================================================
    # =              Input arguments.                  =
    # ==================================================

    parser = argparse.ArgumentParser(description="Run Test 55. Re-implement geometry flagging (thickness and hole).")

    parser.add_argument("--input", help = "The filename of the input JSON file.", default = "./" + DEFAULT_INPUT)
    # parser.add_argument("--voiddiam_m",\
    #     help = "Overwrite voiddiam_m in the input JSON file.",\
    #     default = -1.0, type = float)
    # parser.add_argument("--void_qty",\
    #     help = "Overwrite void_qty in the input JSON file.",\
    #     default = -1, type = int)
    # parser.add_argument("--thickdiam_m",\
    #     help = "Overwrite thickdiam_m in the input JSON file.",\
    #     default = -1.0, type = float)
    # parser.add_argument("--thicksample_qty",\
    #     help = "Overwrite thicksample_qty in the input JSON file.",\
    #     default = -1, type = int)
    # parser.add_argument("--thicktry_qty",\
    #     help = "Overwrite thicktry_qty in the input JSON file.",\
    #     default = -1, type = int)
    # parser.add_argument("--write", help = "Write multiple arrays to file system.", action = "store_true", default = False)

    args = parser.parse_args()

    inputFp = open(args.input)
    params = json.load(inputFp)

    qc_spectrum_csv = params["qc_spectrum_csv"]
    qc_history_csv  = params["qc_history_csv"]
    detector        = params["detector"]
    findingPeak     = params["findingPeak"]

    # ==================================================
    # =            Load qc_spectrum.csv.               =
    # ==================================================

    qcSpectrumEngtryDict = read_and_create_QcSpectrumEntryDict( params["workingDir"] + "/" + params["dataDir"], qc_spectrum_csv )

    # ==================================================
    # =            Load qc_history.csv.               =
    # ==================================================

    qcHistroyDF = read_and_create_df_QcHistiory( params["workingDir"] + "/" + params["dataDir"], qc_history_csv )

    # ==================================================
    # =               CPS & CPS in peak.               =
    # ==================================================

    # Set ROIs.
    for key, qcSE in qcSpectrumEngtryDict.items():
        qcSE.set_roi( detector["qc_roi"] )
        qcSE.counts_per_second()

        print( "%s spectrum CPS = %.4e, CPS in peak = %.4e" % (key, qcSE.cpsSpectrum, qcSE.cpsInPeak) )
    
    # ==================================================
    # =                  Find peaks.                   =
    # ==================================================

    for key, qcSE in qcSpectrumEngtryDict.items():
        qcSE.find_peaks(\
            findingPeak["smoothWidth"],\
            findingPeak["smoothWidth"] / findingPeak["smoothWidthStdFactor"],\
            findingPeak["slopeThreshold"],\
            findingPeak["ampThreshold"],\
            findingPeak["peakGroup"] )

        qcSE.show_peak_info()
        qcSE.get_peak_channel(qc_history_csv["peak_channel_low"], qc_history_csv["peak_channel_high"])
    
