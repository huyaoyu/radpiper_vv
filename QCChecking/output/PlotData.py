
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import time

from visdom import Visdom

PLOT_RANGE = (0, 60)

def initialize_visdom(visStartUpSec = 5):
    vis = Visdom(server = 'http://localhost', port = 8097)

    while not vis.check_connection() and visStartUpSec > 0.0:
        time.sleep(0.1)
        visStartUpSec -= 0.1
    assert vis.check_connection(), 'No connection could be formed quickly'

    print("VisdomLinePlotter initialized.")

    return vis

if __name__ == "__main__":
    vis = initialize_visdom(5)

    # Load data.
    start = np.loadtxt("start.dat", dtype = np.float)
    nStart = len(start)

    x = np.linspace(0, nStart-1, nStart)

    visLine = vis.line(\
                    X = x[ PLOT_RANGE[0]:PLOT_RANGE[1] ],\
                    Y = start[ PLOT_RANGE[0]:PLOT_RANGE[1] ] / 180.302,\
                    name = "start",\
                    opts = dict(\
                        showlegend = True,\
                        title = "start",\
                        xlabel = "index",\
                        ylabel = "CPS"
                    )\
                )

    diffStart = np.loadtxt("start_diff.dat", dtype = np.float)
    vis.line(\
            X = x[ PLOT_RANGE[0]:PLOT_RANGE[1] ],\
            Y = diffStart[ PLOT_RANGE[0]:PLOT_RANGE[1] ],\
            name = "diff_start",\
            win = visLine,\
            update = "append",\
            opts = dict(\
                showlegend = True,\
                title = "start",\
                xlabel = "index",\
                ylabel = "CPS / Diff"
            )\
        )

    convDiffStart = np.loadtxt("start_convDiff.dat", dtype = np.float)
    vis.line(\
            X = x[ PLOT_RANGE[0]:PLOT_RANGE[1] ],\
            Y = convDiffStart[ PLOT_RANGE[0]:PLOT_RANGE[1] ],\
            name = "convDiff_start",\
            win = visLine,\
            update = "append",\
            opts = dict(\
                showlegend = True,\
                title = "start",\
                xlabel = "index",\
                ylabel = "CPS / Diff"
            )\
        )
    
