# File to load .obj files

import numpy as np
import os

def load_obj(file_path):
    """
    Load .obj file and return vertices and faces
    """
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                faces.append([int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]])

    return np.array(vertices), faces
