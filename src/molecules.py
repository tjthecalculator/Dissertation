import numpy as np 
from typing import List, Tuple

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

    def __init__(self, atoms: List[Atom] = None) -> None:
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = atoms
        self.coordinates = np.array([atom.get_coordinate() for atom in self.atoms])

    def get_atoms(self) -> List[Atom]:
        return self.atoms
    
    def get_coordinate(self) -> np.ndarray:
        return self.coordinates
    
    def add_atoms(self, atom: Atom):
        self.atoms.append(atom)

class Ligand(Molecule):

    def __init__(self, atoms: List[Atom] = None) -> None:
        super(Ligand, self).__init__(atoms)
        self.branchs = []

    def add_branch(self, branch: Tuple[int, int]) -> None:
        self.branchs.append(branch)

    def translation(self, axis: int, distance: float = 0.1) -> None:
        self.translation_matrix = None
        self.coordinates       += self.translation_matrix

    def rotation(self, axis: int, angle: int = 1) -> None:
        self.rotation_matrix = None
        self.coordinates    *= self.rotation_matrix

    def bond_rotation(self, bond_idx: int, angle: int = 1) -> None:
        self.bond_rotation_matrix = None
        self.coordinates         *= self.bond_rotation_matrix

class Receptor(Molecule):

    def __init__(self, atoms: List[Atom] = None) -> None:
        super(Receptor, self).__init__(atoms)

def read_ligand_pdbqt(filename) -> Ligand:

    ligand = Ligand()

    with open(filename) as file:
        for line in file:
            if line.startswith('ATOM'):
                atomtype = line.split()
                idx      = line.split()
                coord    = np.array([float(x) for x in line.split()[:]])
                ligand.add_atoms(Atom(idx, atomtype, coord))
            if line.startswith('BRANCH'):
                branch   = tuple([int(x) for x in line.split()])
                ligand.add_branch(branch)

    return ligand

def read_receptor_pdbqt(filename) -> Receptor:

    receptor = Receptor()

    with open(filename) as file:
        for line in file:
            if line.startswith('ATOM'):
                atomtype = line.split()
                idx      = line.split()
                coord    = np.array([float(x) for x in line.split()[:]])
                receptor.add_atoms(Atom(idx, atomtype, coord))

    return receptor
