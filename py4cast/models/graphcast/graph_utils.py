# Copyright 2024 Eviden.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.spatial.transform import Rotation

import trimesh

from typing import Optional, Tuple, Union


def get_edge_length(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Return the edge length.

    Args:
        mesh (trimesh.Trimesh):
            A Trimesh object

    Returns:
        float:
            Edge length.
    """
    src = mesh.edges[:, 0]
    dst = mesh.edges[:, 1]
    return np.linalg.norm(mesh.vertices[src] - mesh.vertices[dst], axis=-1)


def get_max_edge_length(mesh: trimesh.Trimesh) -> float:
    """
    Retrieve the longest edge of the finest mesh.
    It will be used to get the connectivity between the Grid and the Mesh.

    Args:
        mesh (trimesh.Trimesh):
            A Trimesh object representing the Mesh graph.

    Returns:
        float:
            The maximum edge length of the finest mesh.
    """
    edge_length = get_edge_length(mesh)
    return np.max(edge_length)


def get_g2m_connectivity(
    mesh: trimesh.Trimesh,
    meshgrid: np.ndarray,
    fraction: float = 0.6,
    n_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the connectivity for the Grid2Mesh graph.

    Args:
        mesh (trimesh.Trimesh):
            A Trimesh object for the mesh grid.
        meshgrid (np.ndarray):
           Meshgrid of the nodes for the graph.
        fraction (float, optional):
            Fraction of the longest edge of the finest mesh to use to find neighbors.
            Defaults to 0.6.
        n_workers (int, optional):
            Number of workers. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            sender (graph) and receiver (mesh) indices.
    """
    fraction = np.dtype("float32").type(fraction)
    max_length = get_max_edge_length(mesh)
    radius_query = max_length * fraction

    xx, yy = meshgrid
    coordinates = np.stack([xx.flatten(), yy.flatten()], axis=1)

    neighbors = mesh.kdtree.query_ball_point(
        x=coordinates, r=radius_query, workers=n_workers
    )

    grid_edge_index = []
    mesh_edge_index = []

    for grid_index, mesh_neighbors in enumerate(neighbors):
        grid_edge_index.append(np.repeat(grid_index, len(mesh_neighbors)))
        mesh_edge_index.append(mesh_neighbors)

    grid_edge_index = np.concatenate(grid_edge_index).astype(int)
    mesh_edge_index = np.concatenate(mesh_edge_index).astype(int)

    return grid_edge_index, mesh_edge_index  # senders, receivers


def get_m2g_connectivity(
    mesh: trimesh.Trimesh, meshgrid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the connectivity for the Mesh2Grid graph.

    Args:
        mesh (trimesh.Trimesh):
            A Trimesh object.
        meshgrid (np.ndarray):
            Meshgrid of the nodes.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            sender (mesh) and receiver (graph) indices.
    """
    from scipy.spatial import cKDTree

    # Convert the 2D meshgrid to a set of points in 2D
    xx, yy = meshgrid
    coordinates = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Convert the mesh vertices to 2D (ignoring z if it's 0 or 2D)
    vertices_2d = mesh.vertices[:, :2]

    # Use KDTree for fast proximity lookup in 2D
    tree = cKDTree(vertices_2d)

    # Find the closest vertex in the mesh for each point in the grid
    _, closest_vertex_indices = tree.query(coordinates)

    # Create edge connections between mesh and grid points
    mesh_edge_index = closest_vertex_indices.astype(int)
    grid_edge_index = np.arange(coordinates.shape[0]).astype(int)

    return mesh_edge_index, grid_edge_index  # senders, receivers


def create_node_edge_features(
    *,
    graph_type: str,
    src: np.ndarray,
    dst: np.ndarray,
    node_lat: Optional[np.ndarray] = None,
    node_lon: Optional[np.ndarray] = None,
    src_lat: Optional[np.ndarray] = None,
    src_lon: Optional[np.ndarray] = None,
    dst_lat: Optional[np.ndarray] = None,
    dst_lon: Optional[np.ndarray] = None,
    add_node_positions: bool = False,
    add_node_latitude: bool = True,
    add_node_longitude: bool = True,
    add_relative_positions: bool = True,
    relative_latitude_local_coordinates: bool = True,
    relative_longitude_local_coordinates: bool = True,
    edge_normalization_factor: Optional[float] = None,
    sine_cosine_encoding: Optional[bool] = False,
    encoding_num_freqs: Optional[int] = 10,
    encoding_multiplicative_factor: Optional[float] = 1.2,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Create edges features for the MultiMesh, Grid2Mesh and Mesh2Grid graphs.

    Args:
        graph_type (str):
            Which graph are we working on.
        src (np.ndarray):
            Indices of the sender nodes.
        dst (np.ndarray):
            Indices of the receiver nodes.
        node_lat (Optional[np.ndarray]):
            Latitude of the nodes.
            For the MultiMesh graph.
        node_lon (Optional[np.ndarray]):
            Longitudes of the nodes.
            For the MultiMesh graph.
        src_lat (Optional[np.ndarray]):
            Latitudes of the sender nodes.
        src_lon (Optional[np.ndarray]):
            Longitudes of the sender nodes.
        dst_lat (Optional[np.ndarray]):
            Latitudes of the receiver nodes.
        dst_lon (Optional[np.ndarray]:
            Longitudes of the receiver nodes.
        add_node_positions (bool, optional):
            Add unit norm absolute positions. Default to False.
        add_node_latitude (bool, optional):
            Add a feature for latitude (cos(90 - lat)). Default to True.
        add_node_longitude (bool, optional):
            Add features to longitude (cos(lon), sin(lon)). Default to True.
        add_relative_positions (bool, optional):
            Whether to compute the relative positions of the edge. Default to True.
        relative_latitude_local_coordinates (bool, optional):
            If True, relative positions are computed in a coordinate system in which the receiver has latitude 0°.
            Default to True.
        relative_longitude_local_coordinates (Optional[bool]):
            If True, relative positions are computed in a coordinate system in which the receiver has longitude 0°.
            Default to True.
        edge_normalization_factor (Optional[float]):
            Manually control the edge_normalization. If None, normalize by the longest edge length. Default to None.
        sine_cosine_encoding (Optional[bool]):
            If True, apply cosine and sine transformations to the node/edge features, similar to NERF. Default to False.
        encoding_num_freqs (Optional[int]):
            Frequency parameter. Default to 10.
        encoding_multiplicative_factor (Optional[float]):
            Used to compute the frequencies. Default to 1.2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The node and edge features.
    """
    assert graph_type in [
        "Grid2Mesh",
        "g2m",
        "MultiMesh",
        "Mesh2Grid",
        "m2g",
    ], "graph_type should be Grid2Mesh, g2m, MultiMesh, Mesh2Grid or m2g."

    common_kwargs = {
        "src": src,
        "dst": dst,
        "add_node_positions": add_node_positions,
        "add_node_latitude": add_node_latitude,
        "add_node_longitude": add_node_longitude,
        "relative_latitude_local_coordinates": relative_latitude_local_coordinates,
        "relative_longitude_local_coordinates": relative_longitude_local_coordinates,
    }

    if graph_type in ["Grid2Mesh", "g2m", "Mesh2Grid", "m2g"]:
        src_node_features, dst_node_features, edge_features = get_g2m_m2g_features(
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
            add_relative_positions=add_relative_positions,
            edge_normalization_factor=edge_normalization_factor,
            **common_kwargs,
        )
        return src_node_features, dst_node_features, edge_features
    else:
        # src_node_pos = node_lat[src]
        # dst_node_pos = node_lon[dst]
        node_features, edge_features = get_multimesh_features(
            node_lat=node_lat,
            node_lon=node_lon,
            sine_cosine_encoding=sine_cosine_encoding,
            **common_kwargs,
        )
        return node_features, edge_features


def get_g2m_m2g_features(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    add_node_positions: bool = False,
    add_node_latitude: bool = True,
    add_node_longitude: bool = True,
    add_relative_positions: bool = True,
    relative_latitude_local_coordinates: bool = True,
    relative_longitude_local_coordinates: bool = True,
    edge_normalization_factor: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create edges features for the MultiMesh, Grid2Mesh and Mesh2Grid graphs.

    Args:
        src (np.ndarray):
            Indices of the sender nodes.
        dst (np.ndarray):
            Indices of the receiver nodes.
        src_lat (np.ndarray):
            Latitudes of the sender nodes.
        src_lon (np.ndarray):
            Longitudes of the sender nodes.
        dst_lat (np.ndarray):
            Latitudes of the receiver nodes.
        dst_lon (np.ndarray):
            Longitudes of the receiver nodes.
        add_node_positions (bool, optional):
            Add unit norm absolute positions. Default to False.
        add_node_latitude (bool, optional):
            Add a feature for latitude (cos(90 - lat)). Default to True.
        add_node_longitude (bool, optioanl):
            Add features to longitude (cos(lon), sin(lon)). Default to True.
        add_relative_positions (bool, optional):
            Whether to compute the relative positions of the edge. Default to True.
        relative_latitude_local_coordinates (bool, optional):
            If True, relative positions are computed in a coordinate system in which the receiver has latitude 0°.
            Default to True.
        relative_longitude_local_coordinates (bool, optional):
            If True, relative positions are computed in a coordinate system in which the receiver has longitude 0°.
            Default to True.
        edge_normalization_factor (Optional[float]):
            Manually control the edge_normalization. If None, normalize by the longest edge length. Default to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The node and edge features of the Grid2Mesh and Mesh2Grid graphs.
    """
    num_src = src_lat.shape[0]
    num_dst = dst_lat.shape[0]
    num_edges = src.shape[0]
    dtype = src_lat.dtype

    assert dst_lat.dtype == dtype

    # Node features
    ###############
    src_node_features = []
    dst_node_features = []

    if add_node_positions:
        src_node_features.extend((src_lat, src_lon))
        dst_node_features.extend((dst_lat, dst_lon))

    if add_node_latitude:
        src_node_features.append(src_lat)
        dst_node_features.append(dst_lat)

    if add_node_longitude:
        src_node_features.append(src_lon)
        dst_node_features.append(dst_lon)

    if not src_node_features:
        src_node_features = np.zeros([num_src, 0], dtype=dtype)
        dst_node_features = np.zeros([num_dst, 0], dtype=dtype)
    else:
        src_node_features = np.stack(src_node_features, axis=-1)
        dst_node_features = np.stack(dst_node_features, axis=-1)

    # Edge features
    ###############
    edge_features = []

    if add_relative_positions:
        relative_positions = g2m_m2g_project_into_dst_local_coordinates(
            src=src,
            dst=dst,
            src_lat=src_lat,
            src_lon=src_lon,
            dst_lat=dst_lat,
            dst_lon=dst_lon,
            relative_latitude_local_coordinates=relative_latitude_local_coordinates,
            relative_longitude_local_coordinates=relative_longitude_local_coordinates,
        )

        # Get the edge length
        relative_edge_length = np.linalg.norm(
            relative_positions, axis=-1, keepdims=True
        )

    if edge_normalization_factor is None:
        # Normalize to the maximum edge length.
        # Doing so, the edge lengths will fall in the [-1., 1.] interval.
        edge_normalization_factor = relative_edge_length.max()

    edge_features.append(relative_edge_length / edge_normalization_factor)
    edge_features.append(relative_positions / edge_normalization_factor)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    return src_node_features, dst_node_features, edge_features


def g2m_m2g_project_into_dst_local_coordinates(
    src: np.ndarray,
    dst: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    relative_latitude_local_coordinates: bool = True,
    relative_longitude_local_coordinates: bool = True,
) -> np.ndarray:
    """
    Generate a local relative coordinate system defined by the receiver, adapted to a 2D lat/lon grid.
    Applied to the bipartite graphs Grid2Mesh and Mesh2Grid.

    Args:
        src (np.ndarray): Indices of the sender nodes.
        dst (np.ndarray): Indices of the receiver nodes.
        src_lat (np.ndarray): Latitudes of the sender nodes.
        src_lon (np.ndarray): Longitudes of the sender nodes.
        dst_lat (np.ndarray): Latitudes of the receiver nodes.
        dst_lon (np.ndarray): Longitudes of the receiver nodes.
        relative_latitude_local_coordinates (bool):
            If True, relative positions are computed in a system where the receiver has latitude 0°.
        relative_longitude_local_coordinates (bool):
            If True, relative positions are computed in a system where the receiver has longitude 0°.

    Returns:
        np.ndarray: 2D relative positions in the new coordinates (num_edges, 2).
    """

    # Compute differences in latitudes and longitudes
    delta_lat = src_lat[src] - dst_lat[dst]
    delta_lon = src_lon[src] - dst_lon[dst]

    if not (
        relative_latitude_local_coordinates or relative_longitude_local_coordinates
    ):
        # Return the raw latitude and longitude differences
        return np.stack([delta_lat, delta_lon], axis=-1)

    # Rotate the system based on the receiver nodes' coordinates if requested
    if relative_latitude_local_coordinates:
        # Shift the latitude reference so that the destination (receiver) is at 0°
        delta_lat = src_lat[src] - dst_lat[dst]  # already computed; it's lat difference

    if relative_longitude_local_coordinates:
        # Longitude requires handling for crossing the 180° meridian
        delta_lon = (
            np.mod(delta_lon + 180, 360) - 180
        )  # Ensure longitude difference is between [-180, 180]

    # Combine the adjusted lat/lon differences
    relative_positions = np.stack([delta_lat, delta_lon], axis=-1)

    return relative_positions


def get_rotation_matrices(
    reference_phi: np.ndarray,
    reference_theta: np.ndarray,
    rotate_latitude: bool,
    rotate_longitude: bool,
) -> np.ndarray:
    """
    Given the positions in a spherical coordinate system,
    return the rotation matrix to apply to get the local coordinate system.

    Args:
        reference_phi (np.ndarray):
            Polar angles of the reference.
        reference_theta (np.ndarray):
            Azimuthal angles of the reference.
        rotate_latitude (bool):
            Rotate to latitude 0° ?
        rotate_longitude (bool):
            Rotate to longitude 0° ?

    Returns:
        np.ndarray: The Nx3x3 rotation matrix.
    """
    if rotate_latitude and rotate_longitude:
        # First, rotate along the z-axis to get to longitude 0°.
        azimuth_angle = -reference_phi

        # Then, rotate along the y-axis to get to latitude 0°.
        # Complementary angle to π/2 (0°: North pole, π: South pole)
        polar_angle = -reference_theta + (np.pi / 2.0)

        return Rotation.from_euler(
            "zy", np.stack([azimuth_angle, polar_angle], axis=-1)
        ).as_matrix()

    elif rotate_longitude:
        return Rotation("z", -reference_phi).as_matrix()
    elif rotate_latitude:
        # We need to rotate along the z-axis and y-axis first and the undo the first rotation.
        azimuth_angle = -reference_phi
        polar_angle = -reference_theta + (np.pi / 2.0)

        return Rotation.from_euler(
            "zyz", np.stack([azimuth_angle, polar_angle, -azimuth_angle], axis=-1)
        ).as_matrix()


def apply_rotation_with_matrices(
    rotation_matrices: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    return np.einsum("bji,bi->bj", rotation_matrices, positions)


def get_multimesh_features(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    node_lat: np.ndarray,
    node_lon: np.ndarray,
    add_node_positions: bool = False,
    add_node_latitude: bool = True,
    add_node_longitude: bool = True,
    add_relative_positions: bool = True,
    relative_latitude_local_coordinates: bool = True,
    relative_longitude_local_coordinates: bool = True,
    sine_cosine_encoding: bool = False,
    encoding_num_freqs: int = 10,
    encoding_multiplicative_factor: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create edges features for the MultiMesh, Grid2Mesh and Mesh2Grid graphs.

    Args:
        src (np.ndarray):
            Indices of the sender nodes.
        dst (np.ndarray):
            Indices of the receiver nodes.
        node_lat (np.ndarray):
            Latitude of the nodes.
        node_lon (np.ndarray):
            Longitudes of the nodes.
        add_node_positions (bool, optional):
            Add unit norm absolute positions. Default to False.
        add_node_latitude (bool, optional):
            Add a feature for latitude (cos(90 - lat)). Default to True.
        add_node_longitude (bool, optioanl):
            Add features to longitude (cos(lon), sin(lon)). Default to True.
        add_relative_positions (bool, optional):
            Whether to compute the relative positions of the edge. Default to True.
        relative_latitude_local_coordinates (bool, optional):
            If True, relative positions are computed in a coordinate system in which the receiver has latitude 0°.
            Default to True
        relative_longitude_local_coordinates (bool, optional):
            If True, relative positions are computed in a coordinate system in which the receiver has longitude 0°.
            Default to True.
        sine_cosine_encoding (bool, optional):
            If True, apply cosine and sine transformations to the node/edge features, similar to NERF. Default to False.
        encoding_num_freqs (int, optional):
            Frequency parameter. Default to 10.
        encoding_multiplicative_factor (float, optional):
            Used to compute the frequencies. Default to 1.2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The node and edge features of the MultiMesh graph.
    """
    num_nodes = node_lat.shape[0]
    num_edges = src.shape[0]

    dtype = node_lat.dtype

    # Node features
    ###############
    node_features = []

    if add_node_positions:
        node_features.extend((node_lat, node_lon))

    if add_node_latitude:
        node_features.append(node_lat)

    if add_node_longitude:
        node_features.append(node_lon)

    if not node_features:
        node_features = np.zeros([num_nodes, 0], dtype=dtype)
    else:
        node_features = np.stack(node_features, axis=-1)

    # Edge features
    ###############
    edge_features = []

    # if add_relative_positions:
    #     relative_positions = multimesh_project_into_dst_local_coordinates(
    #         src=src,
    #         dst=dst,
    #         node_phi=node_phi,
    #         node_theta=node_theta,
    #         relative_latitude_local_coordinates=relative_latitude_local_coordinates,
    #         relative_longitude_local_coordinates=relative_longitude_local_coordinates,
    #     )

    #     # Get the edge length
    #     relative_edge_length = np.linalg.norm(
    #         relative_positions, axis=-1, keepdims=True
    #     )

    #     max_edge_length = relative_edge_length.max()
    #     edge_features.append(relative_edge_length / max_edge_length)
    #     edge_features.append(relative_positions / max_edge_length)

    if not edge_features:
        edge_features = np.zeros([num_edges, 0], dtype=dtype)
    else:
        edge_features = np.concatenate(edge_features, axis=-1)

    if sine_cosine_encoding:

        def sine_cosine_transform(arr: np.ndarray) -> np.ndarray:
            freqs = encoding_multiplicative_factor ** np.arange(encoding_num_freqs)
            phases = freqs * arr[..., None]
            arr_sin = np.sin(phases)
            arr_cos = np.cos(phases)
            arr_cat = np.concatenate([arr_sin, arr_cos], axis=-1)
            return arr_cat.reshape(arr.shape[0], -1)

        node_features = sine_cosine_transform(node_features)
        edge_features = sine_cosine_transform(edge_features)

    return node_features, edge_features


# def multimesh_project_into_dst_local_coordinates(
#     src: np.ndarray,
#     dst: np.ndarray,
#     node_phi: np.ndarray,
#     node_theta: np.ndarray,
#     relative_latitude_local_coordinates: bool,
#     relative_longitude_local_coordinates: bool,
# ) -> np.ndarray:
#     """
#     Generate a local relative coordinate system defined by the receiver.
#     Applied to the bipartite graphs Grid2Mesh and Mesh2Grid.

#     Args:
#         src (np.ndarray):
#             Indices of the receiver nodes.
#         dst (np.ndarray):
#             Indices of the receiver nodes.
#         node_phi (np.ndarray):
#             Polar angles of the nodes.
#         node_theta (np.ndarray):
#             Azimuthal angles of the nodes.
#         relative_latitude_local_coordinates (bool):
#             If True, relative positions are computed in a coordinate system in which the receiver has latitude 0°.
#             Default to True.
#         relative_longitude_local_coordinates (bool):
#             If True, relative positions are computed in a coordinate system in which the receiver has longitude 0°.
#             Default to True.

#     Returns:
#         np.ndarray: 3D relative positions in the new coordinates (num_edges, 3).
#     """
#     node_xyz = np.stack(spherical_to_cartesian(node_phi, node_theta), axis=-1)

#     # No rotation
#     if not (
#         relative_latitude_local_coordinates or relative_longitude_local_coordinates
#     ):
#         return node_xyz[src] - node_xyz[dst]

#     # Get the rotation matrices
#     rotation_matrices = get_rotation_matrices(
#         reference_phi=node_phi,
#         reference_theta=node_theta,
#         rotate_latitude=relative_latitude_local_coordinates,
#         rotate_longitude=relative_longitude_local_coordinates,
#     )

#     # Rotate the edge according to the rotation matrix of its receiver
#     edge_rotation_matrices = rotation_matrices[dst]

#     dst_pos_in_rotated_space = apply_rotation_with_matrices(
#         edge_rotation_matrices, node_xyz[dst]
#     )
#     src_pos_in_rotated_space = apply_rotation_with_matrices(
#         edge_rotation_matrices, node_xyz[src]
#     )

#     try:
#         np.allclose(
#             dst_pos_in_rotated_space[:, 0], np.ones_like(dst_pos_in_rotated_space[:, 0])
#         )
#         np.allclose(
#             dst_pos_in_rotated_space[:, 1],
#             np.zeros_like(dst_pos_in_rotated_space[:, 1]),
#         )
#         np.allclose(
#             dst_pos_in_rotated_space[:, 2],
#             np.zeros_like(dst_pos_in_rotated_space[:, 2]),
#         )
#     except ValueError:
#         raise ValueError("Error in the projection to local coordinate system.")

#     return src_pos_in_rotated_space - dst_pos_in_rotated_space
