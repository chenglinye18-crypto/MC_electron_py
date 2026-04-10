import os


class InputParser:
    def __init__(self) -> None:
        self.master_config = {}
        self.found_semiconductors = set()
        self._semiconductor_labels = {"IGZO", "SILICON", "ZNO", "GA2O3"}

    def parse_master(self, file_path: str) -> dict:
        """
        Parse master input file (input.txt) using key=value pairs and
        optional material blocks, e.g.

          IGZO {
            scattering_flags = acoustic, lo_abs, lo_ems, to_abs, to_ems
            acoustic_model = deformation_potential_acoustic
          }
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: master input file not found: {file_path}")

        self.master_config = {}
        material_blocks = {}

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [raw.split("//", 1)[0].split("#", 1)[0].strip() for raw in f]

        i = 0
        while i < len(lines):
            line = lines[i]
            if not line:
                i += 1
                continue

            if "{" in line and "=" not in line:
                header = line.split("{", 1)[0].strip().upper()
                if header:
                    block = {}
                    i += 1
                    while i < len(lines):
                        inner = lines[i]
                        if not inner:
                            i += 1
                            continue
                        if inner.startswith("}"):
                            break
                        if "=" in inner:
                            key, value = inner.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            if key:
                                block[key] = self._parse_master_value(value)
                        i += 1
                    material_blocks[header] = block
                i += 1
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key:
                    self.master_config[key] = self._parse_master_value(value)

            i += 1

        if material_blocks:
            self.master_config["material_blocks"] = material_blocks

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
            "defects": {},
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
                value = float(tokens[7])*1e6  # Convert from cm^-3 to m^-3
                device["donors"].append({"bounds": bounds, "value": value})

            elif cmd == "acceptor" and len(tokens) >= 8:
                bounds = [int(t) for t in tokens[1:7]]
                value = float(tokens[7])*1e6  # Convert from cm^-3 to m^-3
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
                # New style only:
                #   contact [name]
                #     N PhiMS=...
                #     <N plane lines>
                #     attachcontact ...
                #     Vapp
                contact_name = "contact"
                if len(tokens) >= 2:
                    raw_name = tokens[1].strip()
                    if raw_name.startswith("[") and raw_name.endswith("]"):
                        raw_name = raw_name[1:-1]
                    if raw_name:
                        contact_name = raw_name

                if i + 1 >= len(lines):
                    break

                header_tokens = lines[i + 1].split()
                if not header_tokens:
                    i += 1
                    continue

                try:
                    num_planes = int(float(header_tokens[0]))
                except ValueError:
                    i += 1
                    continue

                phi_ms = 0.0
                for tk in header_tokens[1:]:
                    val = self._extract_value_token(tk, key="phims")
                    if val is not None:
                        phi_ms = val
                        break

                planes = []
                cursor = i + 2
                for _ in range(num_planes):
                    if cursor >= len(lines):
                        break
                    plane_tokens = lines[cursor].split()
                    if len(plane_tokens) >= 6:
                        planes.append([float(t) for t in plane_tokens[:6]])
                    cursor += 1

                attach_contacts = []
                while cursor < len(lines):
                    next_tokens = lines[cursor].split()
                    if not next_tokens:
                        cursor += 1
                        continue
                    if next_tokens[0].lower() != "attachcontact":
                        break
                    if len(next_tokens) >= 7:
                        attach_contacts.append([float(t) for t in next_tokens[1:7]])
                    cursor += 1

                vapp = 0.0
                if cursor < len(lines):
                    first_token = lines[cursor].split()[0]
                    parsed_vapp = self._extract_value_token(first_token, key="vapp")
                    if parsed_vapp is not None:
                        vapp = parsed_vapp

                device["contacts"].append(
                    {
                        "name": contact_name,
                        "num_planes": num_planes,
                        "phi_ms": phi_ms,
                        "planes": planes,
                        "attach_contacts": attach_contacts,
                        "vapp": vapp,
                    }
                )
                i = cursor

            elif cmd == "defects":
                material = "GENERIC"
                if len(tokens) >= 2:
                    material = tokens[1].strip().strip('"').strip("'").upper()

                defect_params = {}
                cursor = i + 1
                top_level_cmds = {
                    "default_par_number",
                    "region",
                    "donor",
                    "acceptor",
                    "motionplane",
                    "motioncube",
                    "scatterarea",
                    "ep_parm",
                    "parnumber",
                    "surface_scatter_range",
                    "quantumregion",
                    "contact",
                    "defects",
                    "end",
                }
                while cursor < len(lines):
                    tks = lines[cursor].split()
                    if not tks:
                        cursor += 1
                        continue
                    key = tks[0].lower()
                    if key in top_level_cmds:
                        break
                    if len(tks) >= 2:
                        try:
                            defect_params[key] = float(tks[1])
                        except ValueError:
                            pass
                    cursor += 1

                device["defects"][material] = defect_params
                i = cursor - 1

            i += 1

        return device

    @staticmethod
    def _extract_value_token(token: str, key: str | None = None) -> float | None:
        """
        Parse numeric token in either plain form ('4') or key-value form ('PhiMS=0').
        If key is provided, only returns value for matching key; otherwise tries plain float.
        """
        token_s = token.strip()
        if not token_s:
            return None

        if "=" in token_s:
            lhs, rhs = token_s.split("=", 1)
            if key is not None and lhs.strip().lower() != key.lower():
                return None
            try:
                return float(rhs.strip())
            except ValueError:
                return None

        try:
            return float(token_s)
        except ValueError:
            return None

    @staticmethod
    def _parse_master_value(value: str):
        value_s = value.strip()
        if not value_s:
            return ""

        if "," in value_s:
            return [InputParser._parse_master_value(item) for item in value_s.split(",") if item.strip()]

        if (value_s.startswith('"') and value_s.endswith('"')) or (
            value_s.startswith("'") and value_s.endswith("'")
        ):
            return value_s[1:-1]

        value_lower = value_s.lower()
        if value_lower in {"true", "false"}:
            return value_lower == "true"

        try:
            if "." in value_s or "e" in value_lower:
                return float(value_s)
            return int(value_s)
        except ValueError:
            return value_s

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

    def parse_monitor_file(self, file_path: str) -> list[dict]:
        """
        Parse current-monitor surface definitions.

        File format per non-comment line:
          X1 X2 Y1 Y2 Z1 Z2 FACE [NAME]

        - Coordinates are in nm.
        - FACE is one of +X, -X, +Y, -Y, +Z, -Z.
        - NAME is optional; if omitted an automatic M### name is assigned.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Error: monitor file not found: {file_path}")

        monitors: list[dict] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.split("//", 1)[0].split("#", 1)[0].strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 7:
                    raise ValueError(
                        f"Malformed monitor line {line_no} in {file_path}: "
                        "expected at least 7 tokens."
                    )
                try:
                    bounds = [float(tok) for tok in tokens[:6]]
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed monitor bounds on line {line_no} in {file_path}."
                    ) from exc
                face = tokens[6].upper()
                if face not in {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}:
                    raise ValueError(
                        f"Invalid monitor face '{tokens[6]}' on line {line_no} in {file_path}."
                    )
                name = tokens[7] if len(tokens) >= 8 else f"M{len(monitors) + 1:03d}"
                monitors.append(
                    {
                        "name": str(name),
                        "bounds": bounds,
                        "face": face,
                    }
                )

        return monitors
