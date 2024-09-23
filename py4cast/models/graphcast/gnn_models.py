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
import torch.nn as nn
from torch_geometric.utils import scatter

from typing import Tuple


class MLP(nn.Module):
    """
    MultiLayer Perceptron.
    Used for embeddings and mesh/node updates.

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
            Normalization layer applied at the end of the fully connected network, by default LayerNorm
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_bias: bool = True,
        use_norm: bool = True,
        norm: str = "LayerNorm",
        final_activation: bool = True,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.use_norm = use_norm
        self.norm = norm
        self.final_activation = final_activation

        self.activation = getattr(nn, activation)
        if norm == "LayerNorm":
            self.norm = AutoCastLayerNorm
        # self.norm = getattr(nn, norm)

        layers = nn.ModuleList()
        layers.append(nn.Linear(self.in_channel, self.hidden_channels, bias=self.use_bias))
        layers.append(self.activation())

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=self.use_bias))
            layers.append(self.activation())

        layers.append(nn.Linear(self.hidden_channels, self.out_channel, bias=self.use_bias))

        if self.final_activation:
            layers.append(self.activation())

        if self.use_norm:
            layers.append(self.norm(self.out_channel))

        self.gnn_mlp = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        def custom_init(module):
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                stddev = 1.0 / np.sqrt(fan_in)
                nn.init.trunc_normal_(module.weight, std=stddev)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(custom_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn_mlp(x)


class AutoCastLayerNorm(nn.LayerNorm):
    """
    This class comes from ECMWF.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type_as(x)


class MeshEdgeModel(nn.Module):
    """
    Model that is used to update the mesh edge attributes based on its source and target node features and
    its current edge features.

    Args:
        node_src_channel (int):
            Input dimension of the source node features, by default 512
        node_dst_channel (int):
            Input dimension of the destination node features, by default 512
        in_edge_channel (int):
            Input dimension of the edge features, by default 512
        out_edge_channel (int):
            Output dimension of the edge features, by default 512
        edge_hidden_channel (int):
            Number of neurons per MLP layer, by default 512
        edge_num_layer (int):
            Number of layer composing the MLP, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by default LayerNorm
    """

    def __init__(
        self,
        node_src_channel: int = 512,
        node_dst_channel: int = 512,
        in_edge_channel: int = 512,
        out_edge_channel: int = 512,
        edge_hidden_channel: int = 512,
        edge_num_layer: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        self.mesh_edge_mlp = MLP(
            in_channel=node_src_channel + node_dst_channel + in_edge_channel,
            out_channel=out_edge_channel,
            hidden_channels=edge_hidden_channel,
            num_layers=edge_num_layer,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

    def forward(
        self, src: Tensor, dst: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        x = torch.cat([src[edge_index[0]], dst[edge_index[1]], edge_attr], dim=1)

        return self.mesh_edge_mlp(x)


class MeshNodeModel(nn.Module):
    """
    Model that is used to update the mesh node features based on
    its current node features, its graph connectivity and its edge attributes.

    Args:
        in_node_channel (int):
            Input dimension of the node features, by default 512
        in_edge_channel (int):
            Input dimension of the edge features, by default 512
        out_node_channel (int):
            Output dimension of the node features, by default 512
        node_hidden_channel (int):
            Number of neurons per MLP layer, by default 512
        node_num_layer (int):
            Number of layer composing the MLP, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by default LayerNorm
        reduce (str):
            Aggregation operator, by default sum
    """

    def __init__(
        self,
        in_node_channel: int = 512,
        in_edge_channel: int = 512,
        out_node_channel: int = 512,
        node_hidden_channel: int = 512,
        node_num_layer: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
        aggregation: str = "sum",
    ):
        super().__init__()

        self.mesh_node_mlp = MLP(
            in_channel=in_node_channel + in_edge_channel,
            out_channel=out_node_channel,
            hidden_channels=node_hidden_channel,
            num_layers=node_num_layer,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )
        self.aggregation = aggregation

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = scatter(
            edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce=self.aggregation
        )
        out = torch.cat([x, out], dim=1)

        return self.mesh_node_mlp(out)


class GridNodeModel(nn.Module):
    """
    Model that is used to update the grid node features. No aggregation is done as
    grid nodes are only senders in the Grid2Mesh subgraph.

    Args:
        in_node_channel (int):
            Input dimension of the node features, by default 512
        out_node_channel (int):
            Output dimension of the node features, by default 512
        node_hidden_channel (int):
            Number of neurons per MLP layer, by default 512
        node_num_layer (int):
            Number of layer composing the MLP, by default 1
        activation (str):
            Activation function, by default SiLU
        use_norm (bool):
            Whether to use normalization, by default True
        norm (str):
            Normalization layer, by default LayerNorm
    """

    def __init__(
        self,
        in_node_channel: int = 512,
        out_node_channel: int = 512,
        node_hidden_channel: int = 512,
        node_num_layer: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
    ):
        super().__init__()

        self.grid_node_mlp = MLP(
            in_channel=in_node_channel,
            out_channel=out_node_channel,
            hidden_channels=node_hidden_channel,
            num_layers=node_num_layer,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.grid_node_mlp(x)


class InteractionNetwork(nn.Module):
    """
    An InteractionNetwork is a GraphNetwork without the global features.
    This method is a wrapper to the PyG implementation MetaLayer that is inspired
    by the `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    Args:
        node_src_channel (int):
            Input dimension of the source node features, by default 512
        node_dst_channel (int):
            Input dimension of the destination node features, by default 512
        in_edge_channel (int):
            Input dimension of the mesh edge features, by default 512
        in_node_channel (int):
            Input dimension of the mesh node features, by default 512
        out_edge_channel (int):
            Output dimension of the mesh edge features, by default 512
        out_node_channel (int):
            Output dimension of the mesh node features, by default 512
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
        aggregation (str):
            Aggregation function, by default sum
    """

    def __init__(
        self,
        node_src_channel: int = 512,
        node_dst_channel: int = 512,
        in_edge_channel: int = 512,
        in_node_channel: int = 512,
        out_edge_channel: int = 512,
        out_node_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
        activation_checkpointing: bool = False,
        aggregation: str = "sum",
    ):
        super().__init__()

        edge_args = (
            node_src_channel,
            node_dst_channel,
            in_edge_channel,
            out_edge_channel,
            hidden_channels,
            num_layers,
            activation,
            use_norm,
            norm,
        )
        self.edge_model = MeshEdgeModel(*edge_args)

        node_args = (
            in_node_channel,
            out_edge_channel,
            out_node_channel,
            hidden_channels,
            num_layers,
            activation,
            use_norm,
            norm,
            aggregation,
        )
        self.node_model = MeshNodeModel(*node_args)

    def forward(
        self, s_x: Tensor, r_x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # We first update the edge attributes based on the source and destination node features.
        edge_attr_res = self.edge_model(s_x, r_x, edge_index, edge_attr)
        # We then update the mesh node features based on the edge attributes.
        r_x_res = self.node_model(r_x, edge_index, edge_attr_res)

        # Finally we add a residual connection to the mesh node and egde features.
        edge_attr = edge_attr + edge_attr_res
        r_x = r_x + r_x_res

        return r_x, edge_attr
