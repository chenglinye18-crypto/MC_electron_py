import numpy as np


class Mesh:
    """
    Mesh container backed by NumPy arrays.
    Stores node coordinates and per-cell material IDs.
    """

    def __init__(self, coords: dict, regions: list, label_map: dict = None) -> None:
        self.x_nodes = np.asarray(coords["x"], dtype=float)
        self.y_nodes = np.asarray(coords["y"], dtype=float)
        self.z_nodes = np.asarray(coords["z"], dtype=float)

        if self.x_nodes.size < 2 or self.y_nodes.size < 2 or self.z_nodes.size < 2:
            raise ValueError("Mesh nodes are incomplete; need at least 2 points per axis.")

        self.nx = self.x_nodes.size - 1
        self.ny = self.y_nodes.size - 1
        self.nz = self.z_nodes.size - 1

        self.dx = np.diff(self.x_nodes)
        self.dy = np.diff(self.y_nodes)
        self.dz = np.diff(self.z_nodes)

        if label_map is None:
            label_map = {"VACUUM": 0, "OXIDE": 1, "SILICON": 2, "IGZO": 3}
        self.label_map = label_map

        self.material_id = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
        self._assign_materials(regions)

        # Optional per-cell data (initialized in initialization.cell_data_setup)
        self.volume = None
        self.donor = None
        self.acceptor = None
        self.doping = None
        self.da_total = None
        self.electron_charge = None
        self.node_volume = None
        self.node_doping_charge = None
        self.node_doping = None
        self.node_vadd = None
        self.node_charge_fac = None

    def _assign_materials(self, regions: list) -> None:
        """
        Fill material_id based on region bounds.
        Bounds are inclusive indices: [x1, x2, y1, y2, z1, z2].
        """
        for reg in regions:
            bounds = reg["bounds"]
            if len(bounds) != 6:
                continue
            x1, x2, y1, y2, z1, z2 = bounds
            label = reg["label"]
            mat_id = self.label_map[label.upper()]

            xs = slice(max(x1, 0), min(x2, self.nx - 1) + 1)
            ys = slice(max(y1, 0), min(y2, self.ny - 1) + 1)
            zs = slice(max(z1, 0), min(z2, self.nz - 1) + 1)
            self.material_id[xs, ys, zs] = mat_id

    def find_cell(self, x: float, y: float, z: float) -> tuple:
        """
        Map physical coordinates to cell indices (i, j, k).
        Returns (-1, -1, -1) if out of bounds.
        """
        i = int(np.searchsorted(self.x_nodes, x) - 1)
        j = int(np.searchsorted(self.y_nodes, y) - 1)
        k = int(np.searchsorted(self.z_nodes, z) - 1)

        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            return i, j, k
        return -1, -1, -1
