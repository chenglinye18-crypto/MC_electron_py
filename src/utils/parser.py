import os


class InputParser:
    def __init__(self) -> None:
        self.master_config = {}
        self.found_semiconductors = set()
        self._semiconductor_labels = {"IGZO", "SILICON", "ZNO", "GA2O3"}

    def parse_master(self, file_path: str) -> dict:
        """
        Parse master input file (input.txt) using key=value pairs.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: master input file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.split("//", 1)[0].split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    continue

                parsed = value
                try:
                    if "." in value or "e" in value.lower():
                        parsed = float(value)
                    else:
                        parsed = int(value)
                except ValueError:
                    parsed = value

                self.master_config[key] = parsed

        return self.master_config

    def parse_ldg(self, file_path: str) -> dict:
        """
        Parse device definition file (ldg.txt).
        Only extracts core structural info and material labels for now.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: device file not found: {file_path}")

        device = {
            "regions": [],
            "donors": [],
            "acceptors": [],
            "motion_planes": [],
            "motion_cubes": [],
            "scatter_areas": [],
            "parnumber": [],
            "surface_scatter_range": [],
            "quantum_regions": [],
            "contacts": [],
        }

        lines = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.split("//", 1)[0].split("#", 1)[0].strip()
                if line:
                    lines.append(line)

        i = 0
        while i < len(lines):
            tokens = lines[i].split()
            cmd = tokens[0].lower()

            if cmd == "end":
                break

            if cmd == "region" and len(tokens) >= 8:
                bounds = [int(t) for t in tokens[1:7]]
                label = tokens[7].upper()
                device["regions"].append({"bounds": bounds, "label": label})
                if label in self._semiconductor_labels:
                    self.found_semiconductors.add(label)

            elif cmd == "donor" and len(tokens) >= 8:
                bounds = [int(t) for t in tokens[1:7]]
                value = float(tokens[7])
                device["donors"].append({"bounds": bounds, "value": value})

            elif cmd == "acceptor" and len(tokens) >= 8:
                bounds = [int(t) for t in tokens[1:7]]
                value = float(tokens[7])
                device["acceptors"].append({"bounds": bounds, "value": value})

            elif cmd == "motionplane" and len(tokens) >= 9:
                bounds = [int(t) for t in tokens[1:7]]
                face = tokens[7].upper()
                rule = tokens[8].upper()
                device["motion_planes"].append(
                    {"bounds": bounds, "face": face, "rule": rule}
                )

            elif cmd == "motioncube" and len(tokens) >= 13:
                bounds = [int(t) for t in tokens[1:7]]
                rules = [t.upper() for t in tokens[7:13]]
                device["motion_cubes"].append({"bounds": bounds, "rules": rules})

            elif cmd == "scatterarea" and len(tokens) >= 8:
                bounds = [int(t) for t in tokens[1:7]]
                stype = int(tokens[7])
                device["scatter_areas"].append({"bounds": bounds, "type": stype})

            elif cmd == "parnumber" and len(tokens) >= 9:
                bounds = [int(t) for t in tokens[1:7]]
                e_num = int(tokens[7])
                h_num = int(tokens[8])
                device["parnumber"].append(
                    {"bounds": bounds, "electron": e_num, "hole": h_num}
                )

            elif cmd == "surface_scatter_range" and len(tokens) >= 7:
                bounds = [int(t) for t in tokens[1:7]]
                device["surface_scatter_range"].append({"bounds": bounds})

            elif cmd == "quantumregion" and len(tokens) >= 7:
                bounds = [int(t) for t in tokens[1:7]]
                device["quantum_regions"].append({"bounds": bounds})

            elif cmd == "contact":
                # Read: N PhiMS
                if i + 1 >= len(lines):
                    break
                header = lines[i + 1].split()
                if len(header) < 2:
                    i += 1
                    continue
                num_planes = int(header[0])
                phi_ms = float(header[1])
                planes = []
                for k in range(num_planes):
                    if i + 2 + k >= len(lines):
                        break
                    plane_tokens = lines[i + 2 + k].split()
                    if len(plane_tokens) < 6:
                        continue
                    planes.append([int(t) for t in plane_tokens[:6]])
                vapp_line = i + 2 + num_planes
                vapp = float(lines[vapp_line].split()[0]) if vapp_line < len(lines) else 0.0
                device["contacts"].append(
                    {"num_planes": num_planes, "phi_ms": phi_ms, "planes": planes, "vapp": vapp}
                )
                i = vapp_line

            i += 1

        return device

    def parse_lgrid(self, file_path: str, scale: float = 1e-6) -> dict:
        """
        Parse lgrid.txt and return node coordinates.
        Format:
          Nx+1, x0..xN
          Ny+1, y0..yN
          Nz+1, z0..zN
        The input values are typically in microns, so default scale is 1e-6.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: grid file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = [float(x) for x in f.read().split()]

        it = iter(data)
        try:
            nxp1 = int(next(it))
            x_nodes = [next(it) * scale for _ in range(nxp1)]
            nyp1 = int(next(it))
            y_nodes = [next(it) * scale for _ in range(nyp1)]
            nzp1 = int(next(it))
            z_nodes = [next(it) * scale for _ in range(nzp1)]
        except StopIteration:
            raise ValueError("lgrid.txt is incomplete or malformed.")

        return {"x": x_nodes, "y": y_nodes, "z": z_nodes}
