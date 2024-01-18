import numpy as np 
from typing import Tuple

class Atom:

    def __init__(self, atom_idx: int, atom_type: str, coordinate: np.ndarray) -> None:
        self.atom_idx   = atom_idx
        self.atom_type  = atom_type
        self.coordinate = coordinate

    def get_atomtype(self) -> str:
        return self.atom_type
    
    def get_coordinate(self) -> np.ndarray:
        return self.coordinate
    
class Molecule:

    def __init__(self, atoms: list = None) -> None:
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        self.coordinates = np.array([atom.get_coordinate() for atom in self.atoms])

    def get_atoms(self) -> list:
        return self.atoms
    
    def get_coordinate(self) -> np.ndarray:
        return self.coordinates
    
    def add_atoms(self, atom: Atom):
        self.atoms.append(atom)

class Ligand(Molecule):

    def __init__(self, atoms: list = None) -> None:
        super(Ligand, self).__init__(atoms)
        self.branchs = []

    def add_branch(self, branch: Tuple[int, int]) -> None:
        self.branchs.append(branch)

    def translation(self, axis: str, distance: float = 0.1) -> None:
        match axis:
            case 'X'| 'x':
                translation_matrix = np.array([distance, 0, 0])
            case 'Y'| 'y':
                translation_matrix = np.array([0, distance, 0])
            case 'Z'| 'z':
                translation_matrix = np.array([0, 0, distance])
            case _:
                raise ValueError('Wrong axis provided')
        self.coordinates += translation_matrix

    def rotation(self, axis: int, angle: int = 1) -> None:
        angle = np.radians(angle)
        match axis:
            case 'X'| 'x':
                rotation_matrix = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
            case 'Y'| 'y':
                rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
            case 'Z'| 'z':
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            case _:
                raise ValueError('Wrong axis provided')
        self.coordinates *= rotation_matrix

    def cal_dihedral_angle(self, list_atom_idx: list):
        atom_a, atom_b, atom_c, atom_d = [self.coordinates[i] for i in list_atom_idx]
        
        bond_a  = atom_a - atom_b
        bond_b  = atom_b - atom_c
        bond_c  = atom_c - atom_d
        
        plane_a = np.cross(bond_a, bond_b)
        plane_b = np.cross(bond_b, bond_c)




    def bond_rotation(self, bond_idx: int, angle: int = 1) -> None:
        angle                = np.radians(angle)
        rotation_matrix      = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        bond_rotation_matrix = None
        self.coordinates     *= bond_rotation_matrix

class Receptor(Molecule):

    def __init__(self, atoms: list = None) -> None:
        super(Receptor, self).__init__(atoms)

