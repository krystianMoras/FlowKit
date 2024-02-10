from .. import gates
from ..dimension import Dimension

def path_to_vertices_list(path):
    subpath = path.split("Z")[0]
    return [list(map(float, point.split(","))) for point in subpath[1:].split("L")]

def vertices_list_to_path(vertices):
    path = "M"
    for point in vertices:
        path += f"{point[0]},{point[1]}L"
    return path[:-1] + "Z"

def parse_dict_to_gate(gate_dict):
    gate_type = gate_dict["type_"]
    additional_attributes = {}
    for key, value in gate_dict.items():
        if key not in ["id_", "type_", "channels", "x0", "x1", "y0", "y1", "path"]:
            additional_attributes[key] = value
    if gate_type == "rect":
        channels = gate_dict["channels"]

        gate_dict["x0"], gate_dict["x1"] = sorted([
            gate_dict["x0"],
            gate_dict["x1"],
        ])
        gate_dict["y0"], gate_dict["y1"] = sorted([
            gate_dict["y0"],
            gate_dict["y1"],
        ])

        dim_x = Dimension(
            channels["x"],
            range_min=gate_dict["x0"],
            range_max=gate_dict["x1"],
        )
        dim_y = Dimension(
            channels["y"],
            range_min=gate_dict["y0"],
            range_max=gate_dict["y1"],
        )

        dimensions = [dim_x, dim_y]
        gate = gates.RectangleGate(
            gate_name=gate_dict["id_"], dimensions=dimensions, additional_attributes=additional_attributes
        )
    elif gate_type == "poly":
        channels = gate_dict["channels"]
        dim_x = Dimension(channels["x"])
        dim_y = Dimension(channels["y"])
        dimensions = [dim_x, dim_y]
        vertices = path_to_vertices_list(gate_dict["path"])
        gate = gates.PolygonGate(
            gate_name=gate_dict["id_"], dimensions=dimensions, vertices=vertices, additional_attributes=additional_attributes
        )
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")

    return gate

def gate_to_dict(gate):
    if isinstance(gate, gates.RectangleGate):
        gate_dict = {
            "id_": gate.gate_name,
            "type_": "rect",
            "channels": {
                "x": gate.dimensions[0].id,
                "y": gate.dimensions[1].id,
            },
            "x0": gate.dimensions[0].min,
            "x1": gate.dimensions[0].max,
            "y0": gate.dimensions[1].min,
            "y1": gate.dimensions[1].max,
        }
    elif isinstance(gate, gates.PolygonGate):
        gate_dict = {
            "id_": gate.gate_name,
            "type_": "poly",
            "channels": {
                "x": gate.dimensions[0].id,
                "y": gate.dimensions[1].id,
            },
            "path": vertices_list_to_path(gate.vertices),
        }
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    
    for key, value in gate.additional_attributes.items():
        gate_dict[key] = value
    return gate_dict

