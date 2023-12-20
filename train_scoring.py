import os
import numpy as np
from typing import List, Tuple

from src.molecules import Ligand, Atom, read_receptor_pdbqt

def read_vina_docked(filename) -> Tuple[List[Ligand], np.ndarray]:

    ligands, scores = [], []

    with open(filename) as file:
        for line in file:
            if line.startswith('MODEL'):
                ligand = Ligand()
            if line.startswith('ATOM'):
                atomtype = line.split()
                coord    = np.array([float(x) for x in line.split()])
                idx      = line.split()
                ligand.add_atoms(Atom(idx, atomtype, coord))
            if line.startswith(''):
                score    = float(line.split())
            if line.startswith('ENDMDL'):
                ligands.append(ligand)
                scores.append(score)
    
    return ligands, np.array(scores)

