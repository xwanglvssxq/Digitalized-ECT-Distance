from shape import Shape
from numpy import array, ndarray, concatenate, empty, full
from os.path import isfile, splitext
import numpy as np
# Currently supports only triangulated meshes. So no extra edges
# Also: The algorithm doesn't allow just yet the vertices that belong to just 1 triangle
class ShapeReader(object):
    @staticmethod
    def shape_from_file(file_path):
        namelist=file_path.split('/')
        name=namelist[len(namelist)-1].split('.')[0]
        vertices,faces=ShapeReader.off_parser(file_path)
        return Shape(vertices=vertices,triangles=faces,name=name)
        
    @staticmethod
    def off_parser(file_path):
        file=open(file_path,"r")
        # Checking we have valid headers
        A=file.readline().split()
        if A[0] != 'OFF':
            msg = 'The input file does not seem to be valid off file, first line should read "OFF".'
            raise TypeError(msg)
        #Reading in the number of vertices, faces and edges, and pre-formatting their arrays
        (V,F,E)=map(int,file.readline().strip().split(' '))
        vertices=empty([V,3], dtype=np.float32)
        faces=empty([F,3])
        # Read in the vertices
        for i in range(0,V):
            vertices[i]=list(map(float,file.readline().strip().split(' ')))
        #Read in the faces
        for i in range(0,F):
            line=list(map(int,file.readline().strip().split(' ')))
        # Notify the user that there are non-triangular faces.
        # Non-triangular faces wouldn't be supported by the vtk setup that we have anyway.
        # Better way would be to triangulate the polygons, that can be added if deemed useful
        # Also, we could use warnings
            if len(line)!=4:
                print("Warning: The .off contains non-triangular faces, holes might have been created.")
            if (line[0]!=3 and len(line)==4):
                print("Warning: The .off file contains a face that is defined to be non-triangular. It is a valid triangle, reading it as a triangle.")
            faces[i]=line[1:4]
        vertices.astype(np.float32)
        faces.astype(int)
        return(vertices, faces)
