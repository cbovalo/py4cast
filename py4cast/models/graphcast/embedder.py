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

from torch import Tensor
import torch.nn as nn
from py4cast.models.graphcast.gnn_models import MLP

from typing import Tuple


class Embedder(nn.Module):
    """
    Embedding class for the encoder and decoder inputs.
    All features are embedded using a single layer MLP.

    Args:
        in_channel (int):
            Input size
        out_channel (int):
            Output size
        hidden_channels (int):
            Number of neurons per hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by default LayerNorm
    """

    def __init__(
        self,
        in_channel: int = 128,
        out_channel: int = 128,
        hidden_channels: int = 128,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_bias: bool = True,
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        self.mlp = MLP(
            in_channel=in_channel,
            out_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_bias=use_bias,
            use_norm=use_norm,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GraphCastEncoderEmbedder(nn.Module):
    """
    This class aims at representing the input features into a latent space.
    These new features will be fed into the GraphCast Processor.

    Args:
        in_grid_node_channel (int):
            Number of grid node features, by default:
            (5 surface variables + 6 atmospheric variables x 37 levels) x 2 steps +
             5 forcings x 3 steps + 5 constants = 474
        in_mesh_node_channel (int):
            Number of nesh node feature, by default 3
                cos(latitude), cos(longitude) and sin(longitude)
        in_mesh_edge_channel (int):
            Number of mesh edge features, by default 4
                length of the edge and the vector difference of the 3d positions of the sender and receiver nodes
        in_grid2mesh_edge_channel (int):
            Number of grid2mesh edge features, by default 4
                Same as mesh edge features
        out_channel (int):
            Size of the output channel, by default 512
        hidden_channels (int):
            Size of each hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by defaut LayerNorm
    """

    def __init__(
        self,
        in_grid_node_channel: int = 474,
        in_mesh_node_channel: int = 3,
        in_mesh_edge_channel: int = 4,
        in_grid2mesh_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        kwargs = {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "activation": activation,
            "use_norm": use_norm,
            "norm": norm,
        }

        self.grid_node_embedder = Embedder(
            in_channel=in_grid_node_channel,
            out_channel=out_channel,
            use_bias=True,
            **kwargs,
        )

        self.mesh_node_embedder = Embedder(
            in_channel=in_mesh_node_channel, out_channel=out_channel, **kwargs
        )

        self.grid2mesh_edge_embedder = Embedder(
            in_channel=in_grid2mesh_edge_channel, out_channel=out_channel, **kwargs
        )

    def forward(
        self,
        grid_node_feat: Tensor,
        mesh_node_feat: Tensor,
        # mesh_edge_feat: Tensor,
        grid2mesh_edge_feat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Grid node feature embedding
        grid_node_feat = self.grid_node_embedder(grid_node_feat)

        # Mesh node feature embedding
        mesh_node_feat = self.mesh_node_embedder(mesh_node_feat)

        # Mesh edge feature embedding
        # mesh_edge_feat = self.mesh_edge_embedder(mesh_edge_feat)

        # print("Grid2Mesh edge feature embedding")
        grid2mesh_edge_feat = self.grid2mesh_edge_embedder(grid2mesh_edge_feat)

        return (
            grid_node_feat,
            mesh_node_feat,
            # mesh_edge_feat,
            grid2mesh_edge_feat,
        )


class GraphCastProcessorEmbedder(nn.Module):
    """
    Embed the mesh edges into a latent space.

    Args:
        in_mesh_edge_channel (int):
            Number of mesh edge features, by default 4
                length of the edge and the vector difference of the 3d positions of the sender and receiver nodes
        out_channel (int):
            Size of the output channel, by default 512
        hidden_channels (int):
            Size of each hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by defaut LayerNorm
    """

    def __init__(
        self,
        in_mesh_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        kwargs = {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "activation": activation,
            "use_norm": use_norm,
            "norm": norm,
        }

        self.mesh_edge_embedder = Embedder(
            in_channel=in_mesh_edge_channel, out_channel=out_channel, **kwargs
        )

    def forward(self, mesh_edge_feat: Tensor) -> Tensor:
        # Mesh edge feature embedding
        return self.mesh_edge_embedder(mesh_edge_feat)


class GraphCastDecoderEmbedder(nn.Module):
    """
    This class aims at representing the features of the Mesh2Grid graph into a latent space.
    These new features will be fed into the GraphCast Decoder.

    Args:
        in_mesh2grid_edge_channel (int):
            Number of mesh2grid edge features, by default 4
                Same as mesh edge features
        out_channel (int):
            Size of the output channel, by default 512
        hidden_channels (int):
            Size of each hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by defaut LayerNorm
    """

    def __init__(
        self,
        in_mesh2grid_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        kwargs = {
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "activation": activation,
            "use_norm": use_norm,
            "norm": norm,
        }

        self.mesh2grid_edge_embedder = Embedder(
            in_channel=in_mesh2grid_edge_channel, out_channel=out_channel, **kwargs
        )

    def forward(
        self,
        mesh2grid_edge_feat: Tensor,
    ) -> Tensor:
        # Mesh2Grid edge feature embedding
        return self.mesh2grid_edge_embedder(mesh2grid_edge_feat)
