from shape_reader import ShapeReader
import numpy as np
import PolygonECT as ect

def compute_ECT(infile,outfile):
    s1=ShapeReader.shape_from_file(infile)
    ECT=ect.return_ECT(s1)
    np.save(outfile, ECT)

def compute_ECT_distance_p(ECT1, ECT2):
    #receive digitalized ECT, output dECT_p12, dECT12 = dECT_p11 + dECT_p22 -2*dECT_p12
    dECT_partial=ect.ECT_distance_partial(ECT1, ECT2)
    return dECT_partial
