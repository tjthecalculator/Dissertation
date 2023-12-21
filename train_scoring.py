import os
import numpy as np
from typing import List, Tuple

from src.molecules import Ligand, Atom, read_receptor_pdbqt
from src.voxels import calculate_voxel

def read_vina_docked(filename) -> Tuple[List[Ligand], np.ndarray]:

    ligands, scores = [], []

    with open(filename) as file:
        for line in file:
            if line.startswith('MODEL'):
                ligand = Ligand()
            if line.startswith('ATOM'):
                atomtype = line.split()[2]
                coord    = np.array([float(x) for x in line.split()[6:9]])
                idx      = line.split()
                ligand.add_atoms(Atom(idx, atomtype, coord))
            if line.startswith('REMARK VINA RESULT:'):
                score    = float(line.strip('REMARK VINA RESULT:').split()[0])
            if line.startswith('ENDMDL'):
                ligands.append(ligand)
                scores.append(score)
    
    return ligands, np.array(scores)

def pipeline_generator(folder: str, box_size: Tuple[int, int, int]):

    for proteins in os.listdir(folder):
        receptor = read_receptor_pdbqt(os.path.join(folder, proteins, 'receptor.pdbqt'))
        for ligandfile in os.listdir(os.path.join(folder, proteins, 'DockingResults')):
            ligands, scores = read_vina_docked(os.path.join(folder, proteins, 'DockingResults', ligandfile))
            for ligand, score in zip(ligands, scores):
                voxel = calculate_voxel(ligand, receptor, box_size)
                yield np.expand_dims(voxel, axis=-1), np.array([score], dtype=np.float32)

if __name__ == '__main__':
    pass