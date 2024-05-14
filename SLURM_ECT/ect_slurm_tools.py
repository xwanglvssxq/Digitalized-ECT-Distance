from shape_reader import ShapeReader
import numpy as np
import PolygonECT as ect

def compute_ECT(infile,outfile):
    s1=ShapeReader.shape_from_file(infile)
    s1.V = s1.V-np.mean(s1.V,0)
    scales = [sum(tmp**2)**(0.5) for tmp in s1.V]
    s1.V = s1.V/max(scales)
    s1.prepare()
    s1.compute_links()
    s1.compute_polygons()
    s1.compute_gains2()
    s1.clean_gains2()
    ECT=ect.return_ECT(s1)
    ECT = np.array(ECT, dtype=object)
    np.save(outfile, ECT)

def compute_ECT_distance_s(ECT):
    #receive digitalized ECT, output dECT_p12, dECT12 = dECT_p11 + dECT_p22 -2*dECT_p12
    dECT_s=ect.ECT_distance_s(ECT)
    return dECT_s

def compute_ECT_distance_d(ECT1, ECT2):
    #receive digitalized ECT, output dECT_p12, dECT12 = dECT_p11 + dECT_p22 -2*dECT_p12
    dECT_d=ect.ECT_distance_d(ECT1, ECT2)
    return dECT_d