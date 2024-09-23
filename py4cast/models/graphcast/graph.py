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
import torch
from torch import Tensor
import torch_geometric as pyg
import trimesh

from py4cast.models.graphcast.graph_utils import (
    create_node_edge_features,
    get_g2m_connectivity,
    get_m2g_connectivity,
    cartesian_to_spherical,
    spherical_to_latlon,
)


class Graph:
    """
    This class is used to construct all graphs (Grid2Mesh, MultiMesh and Mesh2grid) and
    to add the node and edge features.

    Args:
        meshgrid (Tensor):
            A tensor concatening x and y coordinates. Dimension [2, x, y]
        n_subdivisions (int):
            How many times to subdivide the original icosphere. Defaults to 6.
    """

    def __init__(
        self,
        meshgrid: Tensor,
        n_subdivisions: int = 6,
    ) -> None:
        self.grid_nodes_lon, self.grid_nodes_lat = (
            meshgrid[0].numpy(),
            meshgrid[1].numpy(),
        )
        self.grid_latitude = np.unique(self.grid_nodes_lat)
        self.grid_longitude = np.unique(self.grid_nodes_lon)

        self.n_subdivisions = n_subdivisions

        self.icospheres = []

        self.grid2mesh_graph = None
        self.mesh_graph = None
        self.mesh2grid_graph = None

        self.grid_nodes_lat = self.grid_nodes_lat.reshape([-1])
        self.grid_nodes_lon = self.grid_nodes_lon.reshape([-1])

        self._create_icospheres()
        self.finest_mesh = self._finest_mesh()

        mesh_phi, mesh_theta = cartesian_to_spherical(
            self.finest_mesh.vertices[:, 0].astype(np.float32),
            self.finest_mesh.vertices[:, 1].astype(np.float32),
            self.finest_mesh.vertices[:, 2].astype(np.float32),
        )
        self.mesh_nodes_latitude, self.mesh_nodes_longitude = spherical_to_latlon(
            mesh_phi, mesh_theta
        )

        self._node_edge_features_kwargs = {
            "add_node_positions": False,
            "add_node_latitude": True,
            "add_node_longitude": True,
            "add_relative_positions": True,
            "relative_latitude_local_coordinates": True,
            "relative_longitude_local_coordinates": True,
        }

    def _create_icospheres(self) -> None:
        """
        Create the multimesh structure.
        The multimesh is built by refining an icosphere with unit sphere.
        By default, the poles are vertices of the multimesh. We rotate around the y axis
        by half the supplementary to the angle between faces divided by 2.
        <https://en.wikipedia.org/wiki/Regular_icosahedron>
        """
        # Get the rotation matrix
        angle_between_face = np.arccos(-np.sqrt(5.0) / 3.0)
        rotation_angle = (np.pi - angle_between_face) / 2.0
        rotation_matrix = trimesh.transformations.euler_matrix(
            0.0, -rotation_angle, 0.0, "sxyz"
        )

        for i in range(self.n_subdivisions + 1):
            icosphere_ = trimesh.creation.icosphere(subdivisions=i, radius=1.0)
            icosphere_.apply_transform(rotation_matrix)
            self.icospheres.append(icosphere_)

    def create_MultiMesh(self, save_mesh: bool = False) -> pyg.data.Data:
        """
        Creates the MultiMesh graph based on the icospheres.

        Args:
            save_mesh (bool, optional):
                Whether to save the icosphers or not. Defaults to False.

        Returns:
            pyg.data.Data: The Mesh graph in Pytorch Geometric format
        """
        all_faces = [
            trimesh.exchange.export.export_dict(icosphere)["faces"]
            for icosphere in self.icospheres
        ]
        all_faces = np.concatenate(all_faces)
        self.finest_mesh.faces = all_faces

        # Get node and edge features
        node_features, edge_features = create_node_edge_features(
            graph_type="MultiMesh",
            src=self.finest_mesh.edges[:, 0],
            dst=self.finest_mesh.edges[:, 1],
            node_lat=self.mesh_nodes_latitude,
            node_lon=self.mesh_nodes_longitude,
            **self._node_edge_features_kwargs,
        )

        # Create the PyG graph (pyg.data.Data)
        self.mesh_graph = self._create_pyg_graph(
            self.finest_mesh.edges[:, 0], self.finest_mesh.edges[:, 1]
        )
        # Add node and edge features
        self.mesh_graph.x = torch.from_numpy(node_features)
        self.mesh_graph.edge_attr = torch.from_numpy(edge_features).to(torch.float32)

        # Export the icospheres if necessary
        if save_mesh:
            for i, mesh in self.icospheres:
                self._export_to_meshio(
                    mesh, filename=f"./icosphere_split_order_{i}.vtk"
                )

            self._export_to_meshio(self.finest_mesh, filename="./multimesh.vtk")

    def create_Grid2Mesh(
        self,
        fraction: float = 0.6,
        n_workers: int = 1,
    ) -> pyg.data.Data:
        """
        Creates the Grid2Mesh graph.

        Args:
            fraction (int, optional):
                Fraction of the length of the longest edge in the finest mesh.
                It will be used to map grid nodes to mesh nodes. Using the value
                0.6000308 allows to get the same number of edges as in the paper.
                Defaults to 0.6.
            n_workers (int, optional):
                Number of workers for parallel processing. Defaults to 1.

        Raises:
            ValueError: The finest mesh with all merged edges must exist
                        before building the three graphs.

        Returns:
            pyg.data.Data: The Grid2Mesh graph in Pytorch Geometric format
        """
        # if self.finest_mesh is None:
        #     raise ValueError("Expected MultiMesh to be created before Graph2Mesh.")

        # Get the graph connectivity for Grid2Mesh
        src, dst = get_g2m_connectivity(
            self.icospheres[-1],
            self.grid_latitude,
            self.grid_longitude,
            fraction=fraction,  # 0.6000308 to get the same number as in the paper
            n_workers=n_workers,
        )

        # Get node and edge features
        src_node_features, dst_node_features, edge_features = create_node_edge_features(
            graph_type="Grid2Mesh",
            src=src,
            dst=dst,
            src_lat=self.grid_nodes_lat,
            src_lon=self.grid_nodes_lon,
            dst_lat=self.mesh_nodes_latitude,
            dst_lon=self.mesh_nodes_longitude,
            **self._node_edge_features_kwargs,
        )

        # Create the PyG graph (pyg.data.Data)
        self.grid2mesh_graph = self._create_pyg_graph(src, dst)
        # Add node and edge features
        self.grid2mesh_graph.x_s = torch.from_numpy(src_node_features)
        self.grid2mesh_graph.x_r = torch.from_numpy(dst_node_features)
        self.grid2mesh_graph.edge_attr = torch.from_numpy(edge_features).to(
            torch.float32
        )

    def create_Mesh2Grid(
        self, edge_normalization_factor: float = None
    ) -> pyg.data.Data:
        """
        Create the Mesh2Grid graph

        Args:
            edge_normalization_factor: Manually control the edge_normalization.
                                       If None, normalize by the longest edge length. Default to None.

        Raises:
            ValueError:
                Raise an error if the MultiMesh is not constructed yet.

        Returns:
            pyg.data.Data:
                The Mesh2Grid graph in PyG format.
        """
        # if self.finest_mesh is None:
        #     raise ValueError("Expected MultiMesh to be created before Mesh2Grid.")

        # Get the graph connectivity for Mesh2Grid
        src, dst = get_m2g_connectivity(
            self.icospheres[-1], self.grid_latitude, self.grid_longitude
        )

        # Get node and edge features
        src_node_features, dst_node_features, edge_features = create_node_edge_features(
            graph_type="Mesh2Grid",
            src=src,
            dst=dst,
            src_lat=self.mesh_nodes_latitude,
            src_lon=self.mesh_nodes_longitude,
            dst_lat=self.grid_nodes_lat,
            dst_lon=self.grid_nodes_lon,
            edge_normalization_factor=edge_normalization_factor,
            **self._node_edge_features_kwargs,
        )

        # Create the PyG graph (pyg.data.Data)
        self.mesh2grid_graph = self._create_pyg_graph(src, dst)
        # Add node and edge features
        self.mesh2grid_graph.x_s = None  # torch.from_numpy(src_node_features)
        self.mesh2grid_graph.x_t = None  # torch.from_numpy(dst_node_features)
        self.mesh2grid_graph.edge_attr = torch.from_numpy(edge_features).to(
            torch.float32
        )

    def _finest_mesh(self) -> trimesh.Trimesh:
        return self.icospheres[-1].copy()

    def _create_pyg_graph(
        self, src_index: np.ndarray, dst_index: np.ndarray
    ) -> pyg.data.Data:
        """
        Creates a PyG Data object ie a graph.

        Args:
            src_index (Tensor):
                Indices of the sender nodes.
            dst_index (Tensor):
                Indices of the receiver nodes.

        Returns:
            pyg.data.Data:
                A graph as defined by PyG.
        """
        return pyg.data.Data(
            x=None,
            edge_index=torch.stack(
                [
                    torch.from_numpy(np.copy(src_index)),
                    torch.from_numpy(np.copy(dst_index)),
                ],
                axis=0,
            ),  # .long(),
            edge_attrs=None,
            y=None,
        )

    def _export_to_meshio(self, mesh: trimesh.Trimesh, filename: str = None) -> None:
        """
        Optional method to save the MultiMesh in VTK format.

        Args:
            mesh (trimesh.Trimesh): _description_
            filename (str, optional): _description_. Defaults to None.
        """
        import meshio

        mesh_dict = trimesh.exchange.export.export_dict(mesh)
        cells = [("triangle", mesh_dict["face"])]
        mesh = meshio.Mesh(  # noqa: F821 (undefined name ``meshio``)
            mesh_dict["vertices"], cells
        )
        mesh.write(filename)
