import pandas as pd
import os
import pathlib
import numpy as np

elements = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80
}

def distance(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)

def get_charge(symbol):
    return elements[symbol]

def coulomb_matrix(molecule):
    # currently the largest molecule, needs to be generalized
    dimension=32
    
    num_atoms = len(molecule)
    coords = np.array([[float(atom[1]), float(atom[2]), float(atom[3])] for atom in molecule])
    charges = np.array([get_charge(atom[0]) for atom in molecule])
    
    coulomb_mat = np.zeros((dimension, dimension))
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                coulomb_mat[i, j] = 0.5 * charges[i] ** 2.4
            else:
                coulomb_mat[i, j] = charges[i] * charges[j] / distance(coords[i], coords[j])
    
    return coulomb_mat

# go through input files in geometry folder and parse geometry block into dictonary mols
# get relative path assuming geometries are in separate folder in the same directory
# if relative path doesn't work use absolute path instead
# for filename in os.scandir("/home/hannes/Documents/SBU/AMS561/ML/geometries"):
# parse name of molecule (filename) for key
#
# the format is
# list of lists containing [letter, x, y, z]
# geometry
# units angstrom
#  H 0 0 0
#  ...
# end
def geometry_parser():
    path = os.path.abspath(os.getcwd())
    geometries_path = os.path.join(path, 'geometries')
    mols = {}
    for filename in os.scandir(geometries_path):
        name = str(filename).split()[1]
        name = name.replace('.mdns', '')
        name = name.replace("'", "")
        name = name.replace(">", "")
        if filename.is_file():
            f = open(filename, "r")
            mol = []
            while True:
                line = f.readline()
                if "geometry" in line:
                    line = f.readline()
                    while True:
                        line = f.readline()
                        if line == "end":
                            break
                        mol.append(line.split())
                    mols[name] = mol 
                if not line:
                    break
    return mols

def make_input(): 
    # import data for corresponding method
    #b3lyp = pd.read_csv('b3lyp_eprec6.csv')
    #hf = pd.read_csv('hf_eprec6.csv')
    lda = pd.read_csv('lda_eprec6.csv')
    pbe0 = pd.read_csv('pbe0_eprec6.csv')
    
    # preparing input for pytorch; need vector (X) from CM and corresponding ediff (Y)
    mols_dict = {}
    mols = geometry_parser()
    
    # create CM
    for item in mols.items():
        mols_dict[item[0]] = coulomb_matrix(item[1]).flatten() 
        #mols_vec.append(coulomb_matrix(item[1]).flatten())
        
    Y = pbe0['Energy'].astype(float)-lda['Energy'].astype(float)
    
    # need to align molecules with ediff from csv files
    mols_vec = []
    energy_diff = []
    for i in pbe0['Unnamed: 0']:
        mols_vec.append(mols_dict[i])
    
    X = np.array(mols_vec)
    return X,Y

