from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    r = np.loadtxt("r.txt", dtype = np.float)
    mr = np.loadtxt("mr.txt", dtype = np.float)
    a = np.loadtxt("a.txt", dtype = np.float)
    x = np.cos(a) * r
    y = np.sin(a) * r

    ax = plt.gca()
    ax.cla()

    maskX = np.isfinite(x)
    maskY = np.isfinite(y)

    ax.set_xlim( ( np.min(x[maskX])*1.1, np.max(x[maskX])*1.1 ) )
    ax.set_ylim( ( np.min(y[maskY])*1.1, np.max(y[maskY])*1.1 ) )

    for cx, cy, cr in zip(x, y, mr):
        if ( np.logical_not( np.isfinite(cx) ) ):
            continue
        
        if ( np.isfinite(cr) ):
            color = "b"
        else:
            color = "r"
            ax.plot([cx, -cx], [cy, -cy], "r")

        c = plt.Circle((cx, cy), 0.002, color=color, fill=False)

        ax.add_artist(c)

    plt.show()

