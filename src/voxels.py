import numpy as np 
from molecules import Ligand, Receptor
from typing import Tuple

def calculate_voxel(ligand: Ligand, receptor: Receptor, box_size: Tuple[int, int, int], resolution: float = 0.375) -> np.ndarray:
    
    voxel_size = tuple((np.array(box_size)/resolution).astype(int))
    means      = np.concatenate([ligand.get_coordinate(), receptor.get_coordinate()]).mean(axis=0)
    voxel      = np.zeros(voxel_size, dtype=np.int32)

    for atom in ligand.get_atoms() + receptor.get_atoms():
        idx = tuple((atom.get_coordinate() - means + np.array(voxel_size)//2).astype(int))
        if 0 <= idx[0] < voxel_size[0] and 0 <= idx[1] < voxel_size[1] and 0 <= idx[2] < voxel_size[2]:
            if voxel[idx] == 0:
                voxel[idx] += atom.get_atomtype()

    return voxel