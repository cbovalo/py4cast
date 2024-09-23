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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses_json import dataclass_json
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn

from py4cast.models.graphcast.graph import Graph
from py4cast.models.graphcast.embedder import GraphCastEncoderEmbedder, GraphCastProcessorEmbedder, GraphCastDecoderEmbedder
from py4cast.models.graphcast.gnn_models import GridNodeModel, InteractionNetwork, MLP

from typing import Tuple
from py4cast.models.base import ModelABC
from pathlib import Path
from py4cast.datasets.base import Statics


@dataclass_json
@dataclass(slots=True)
class GraphCastSettings:
    """
    Settings for graphcast model
    """

    tmp_dir: Path | str = "/tmp"  # nosec B108

    # Graph configuration
    n_subdivisions: int = 6
    fraction: float = .6
    mesh2grid_edge_normalization_factor: float = None

    # Architecture configuration
    input_mesh_node_channel: int = 3
    input_mesh_edge_channel: int = 4
    input_grid2mesh_edge_channel: int = 4
    output_channel: int = 512
    hidden_channels: int = 512
    num_layers: int = 1
    num_block: int = 16
    activation: str = "SiLU"
    use_norm: bool = True
    norm: str = "LayerNorm"
    aggregation: str = "sum"


class GraphCast(ModelABC, nn.Module):
    """
    Implemention of GraphCast from the Atos/Eviden code.
    """
    settings_kls = GraphCastSettings
    onnx_supported = False  # TODO: confirm it is not onnx supported
    input_dims: str = ("batch", "ngrid", "features")
    output_dims: str = ("batch", "ngrid", "features")

    @classmethod
    def rank_zero_setup(cls, settings: GraphCastSettings, statics: Statics):
        """
        This is a static method to allow multi-GPU
        trainig frameworks to call this method once
        on rank zero before instantiating the model.
        """
        # this doesn't take long and it prevents discrepencies
        graph = Graph(statics.meshgrid)
        # Create the graphs
        graph.create_Grid2Mesh(fraction=settings.fraction, n_workers=10)  # TODO: retrieve n_workers from tl_settings ?
        graph.create_MultiMesh()
        graph.create_Mesh2Grid(edge_normalization_factor=settings.mesh2grid_edge_normalization_factor)

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: GraphCastSettings,
        input_shape: tuple,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.grid_size = (len(self.graph.grid_latitude), len(self.graph.grid_longitude))
        self.num_grid_nodes = self.grid_size[0] * self.grid_size[1]

        # Instantiate the model
        # Encoder
        self.encoder = GraphCastEncoder(
            in_grid_node_channel=num_input_features,
            in_mesh_node_channel=settings.input_mesh_node_channel + settings.input_grid_node_channel,
            in_grid2mesh_edge_channel=settings.input_grid2mesh_edge_channel,
            out_channel=settings.output_channel,
            hidden_channels=settings.hidden_channels,
            num_layers=settings.num_layers,
            activation=settings.activation,
            use_norm=settings.use_norm,
            norm=settings.norm,
        )

        # Processor
        self.processor = GraphCastProcessor(
            in_node_channel=settings.hidden_channels,
            in_edge_channel=settings.input_mesh_edge_channel,
            out_channel=settings.output_channel,
            hidden_channels=settings.hidden_channels,
            num_layers=settings.num_layers,
            num_block=settings.num_block,
            activation=settings.activation,
            use_norm=settings.use_norm,
            norm=settings.norm,
            aggregation=settings.aggregation,
        )

        # Decoder
        self.decoder = GraphCastDecoder(
            in_grid_node_channel=settings.hidden_channels,
            in_mesh_node_channel=settings.hidden_channels,
            in_mesh2grid_edge_channel=settings.input_mesh_edge_channel,
            out_channel=settings.output_channel,
            hidden_channels=settings.hidden_channels,
            num_layers=settings.num_layers,
            activation=settings.activation,
            use_norm=settings.use_norm,
            norm=settings.norm,
            aggregation=settings.aggregation,
        )

        # Output Layer
        self.final_layer = MLP(
            in_channel=settings.hidden_channels,
            out_channel=num_output_features,
            hidden_channels=settings.hidden_channels,
            num_layers=settings.num_layers,
            activation=settings.activation,
            use_norm=False,
        )

    def encoder_forward(
            self,
            grid_node_feat: torch.Tensor,
            static_grid_node_feat: torch.Tensor,
            mesh_node_feat: torch.Tensor,
            edge_index: torch.Tensor,
            # mesh_edge_feat: torch.Tensor,
            grid2mesh_edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert grid_node_feat.shape[0] == static_grid_node_feat.shape[0]
        grid_node_feat = torch.cat([grid_node_feat, static_grid_node_feat], dim=-1)

        # We do as in GraphCast code, we add some zeros to the mesh node features to
        # 'make sure capacity of the embedded is identical for the grid nodes and the mesh nodes'.
        # https://github.com/google-deepmind/graphcast/blob/8debd7289bb2c498485f79dbd98d8b4933bfc6a7/graphcast/graphcast.py#L629
        dummy_mesh_node_feat = torch.zeros(
            (mesh_node_feat.shape[0],) + grid_node_feat.shape[1:],
            dtype=grid_node_feat.dtype,
            device=grid_node_feat.device
        )
        mesh_node_feat = torch.cat([dummy_mesh_node_feat, mesh_node_feat], dim=-1)

        (
            grid_node_feat,
            mesh_node_feat,
            grid2mesh_edge_feat,
        ) = self.encoder(
            grid_node_feat,
            mesh_node_feat,
            edge_index,
            grid2mesh_edge_feat,
        )
        return grid_node_feat, mesh_node_feat, grid2mesh_edge_feat

    def processor_forward(
        self,
        mesh_node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        mesh_edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processor(mesh_node_feat, edge_index, mesh_edge_feat)

    def decoder_forward(
        self,
        grid_node_feat: torch.Tensor,
        mesh_node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        grid_node_feat = self.decoder(
            grid_node_feat,
            mesh_node_feat,
            edge_index,
            edge_attr,
        )
        return grid_node_feat

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Prepare the inputs
        grid_node_feat = inputs.squeeze(dim=0).view(
            self.num_grid_nodes, -1
        )

        # Encoder (Embedding and Grid2Mesh GNN)
        (
            grid_node_feat,
            mesh_node_feat,
            grid2mesh_edge_feat,
        ) = checkpoint(
            self.encoder_forward,
            grid_node_feat,
            self.graph.grid2mesh_graph.x_s,
            self.graph.grid2mesh_graph.x_r,
            self.graph.grid2mesh_graph.edge_index,
            self.graph.grid2mesh_graph.edge_attr,
            use_reentrant=False,
            debug=False
        )

        # Processor (Mesh GNN)
        mesh_node_feat, mesh_edge_feat = checkpoint(
            self.processor_forward,
            mesh_node_feat,
            self.graph.mesh_graph.edge_index,
            self.graph.mesh_graph.edge_attr,
            use_reentrant=False,
            debug=False
        )

        # Decoder (Mesh2Grid GNN)
        grid_node_feat = checkpoint(
            self.decoder_forward,
            grid_node_feat,
            mesh_node_feat,
            self.graph.mesh2grid_graph.edge_index,
            self.graph.mesh2grid_graph.edge_attr,
            use_reentrant=False,
            debug=False
        )

        # Final output (MLP)
        output = checkpoint(
            self.final_layer, grid_node_feat, use_reentrant=False, debug=False
        )
        output = output.contiguous().view(*self.grid_size, -1)

        return output.unsqueeze(dim=0)


class GraphCastEncoder(nn.Module):
    """
    This is the implementation of GraphCast Encoder.
    The goal of this class is to transform the input data into a latent representation.
    for the processor running on the multi-mesh.
    Input features (grid nodes, mesh nodes, mesh edges and grid2mesh edges) are fed into a 1-hidden layer MLP.

    Args:
        in_grid_node_channel (int):
            Number of grid node features, by default:
                (5 surface variables + 6 atmospheric variables x 37 levels) x 2 steps +
                 5 forcings X 3 steps + 5 constants = 474
        in_mesh_node_channel (int):
            Number of nesh node feature, by default 3
                cos(latitude), cos(longitude) and sin(longitude)
        in_mesh_edge_channel (int):
            Number of mesh edge features, by default 4
                length of the edge and the vector difference of the 3d positions of the sender and receiver nodes
        in_grid2mesh_edge_channel (int):
            Number of grid2mesh edge features, by default 4
                Same as mesh egde features
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
        aggregation (str):
            Aggregation function, by default sum
    """

    def __init__(
        self,
        in_grid_node_channel: int = 474,
        in_mesh_node_channel: int = 3,
        # in_mesh_edge_channel: int = 4,
        in_grid2mesh_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
        aggregation: str = "sum",
    ):
        super().__init__()

        self.encoder_embedder = GraphCastEncoderEmbedder(
            in_grid_node_channel=in_grid_node_channel,
            in_mesh_node_channel=in_mesh_node_channel,
            # in_mesh_edge_channel=in_mesh_edge_channel,
            in_grid2mesh_edge_channel=in_grid2mesh_edge_channel,
            out_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

        self.grid2mesh_mesh_gnn = InteractionNetwork(
            node_src_channel=out_channel,
            node_dst_channel=out_channel,
            in_edge_channel=out_channel,
            in_node_channel=out_channel,
            out_edge_channel=out_channel,
            out_node_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
            aggregation=aggregation,
        )

        self.grid2mesh_grid_gnn = GridNodeModel(
            in_node_channel=out_channel,
            out_node_channel=out_channel,
            node_hidden_channel=hidden_channels,
            node_num_layer=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        grid2mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # First, we embed the different features.
        grid_nfeat, mesh_nfeat, grid2mesh_efeat = self.encoder_embedder(
            grid_nfeat, mesh_nfeat, grid2mesh_efeat
        )
        # Then we apply the Interaction Network to the Grid2Mesh graph
        # and the MLP to the Grid node features. Residual connections are
        # done directly within the following models.
        mesh_nfeat, grid2mesh_efeat = self.grid2mesh_mesh_gnn(
            grid_nfeat, mesh_nfeat, edge_index, grid2mesh_efeat
        )
        grid_nfeat = self.grid2mesh_grid_gnn(grid_nfeat)

        return grid_nfeat, mesh_nfeat, grid2mesh_efeat


class GraphCastProcessor(nn.Module):
    """
    Processor block operating only on the Mesh graph.
    This module is a sequence of N Interaction Networks.

    Args:
        in_node_channel (int):
            Input dimension of the source node features, by default 512
        in_edge_channel (int):
            Input dimension of the mesh edge features, by default 4
        out_channel (int):
            Output dimension, by default 512
        hidden_channels (int):
            Size of each hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        num_block (int):
            Number of Interaction Network blocks, by default 16
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
        in_node_channel: int = 512,
        in_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        num_block: int = 16,
        activation: str = "SiLU",
        use_norm: bool = True,
        norm: str = "LayerNorm",
        aggregation: str = "sum",
    ):
        super().__init__()

        self.processor_embedder = GraphCastProcessorEmbedder(
            in_mesh_edge_channel=in_edge_channel,
            out_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

        interaction_network_args = (
            in_node_channel,
            in_node_channel,
            in_node_channel,
            in_node_channel,
            out_channel,
            out_channel,
            hidden_channels,
            num_layers,
            activation,
            use_norm,
            norm,
            aggregation,
        )

        self.layers = nn.ModuleList()
        for _ in range(num_block):
            self.layers.append(InteractionNetwork(*interaction_network_args))

    def forward(
        self,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mesh_efeat = self.processor_embedder(mesh_efeat)

        for i in range(len(self.layers)):
            # layer = self.layers[i]
            mesh_nfeat, mesh_efeat = self.layers[i](
                mesh_nfeat, mesh_nfeat, edge_index, mesh_efeat
            )

        return mesh_nfeat, mesh_efeat



class GraphCastDecoder(nn.Module):
    """
    This is the implementation of GraphCast Decoder.
    The goal of the module is to bring back information from the latent graph (multi-mesh) to
    the original grid and to extract the outputs.

    Args:
        in_grid_node_channel (int):
            Number of grid node features, by default:
                (5 surface variables + 6 atmospheric variables x 37 levels) x 2 steps +
                 5 forcings X 3 steps + 5 constants = 474
        in_mesh_node_channel (int):
            Number of nesh node feature, by default 3
                cos(latitude), cos(longitude) and sin(longitude)
        in_mesh_edge_channel (int):
            Number of mesh edge features, by default 4
                length of the edge and the vector difference of the 3d positions of the sender and receiver nodes
        in_grid2mesh_edge_channel (int):
            Number of grid2mesh edge features, by default 4
                Same as mesh egde features
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
        aggregation (str):
            Aggregation function, by default sum
    """
    def __init__(
        self,
        in_grid_node_channel: int = 512,
        in_mesh_node_channel: int = 512,
        in_mesh2grid_edge_channel: int = 4,
        out_channel: int = 512,
        hidden_channels: int = 512,
        num_layers: int = 1,
        activation: str = 'SiLU',
        use_norm: bool = True,
        norm: str = 'LayerNorm',
        aggregation: str = 'sum',
    ):
        super().__init__()

        self.decoder_embedder = GraphCastDecoderEmbedder(
            in_mesh2grid_edge_channel=in_mesh2grid_edge_channel,
            out_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
        )

        self.mesh2grid_gnn = InteractionNetwork(
            node_src_channel=out_channel,
            node_dst_channel=out_channel,
            in_edge_channel=out_channel,
            in_node_channel=out_channel,
            out_edge_channel=out_channel,
            out_node_channel=out_channel,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            activation=activation,
            use_norm=use_norm,
            norm=norm,
            aggregation=aggregation
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        mesh2grid_efeat: Tensor,
    ) -> Tensor:
        # First the edge attributes of Mesh2Grid are encoded into a latent representation.
        mesh2grid_efeat = self.decoder_embedder(mesh2grid_efeat)

        # Then we apply the Interaction Network to the Mesh2Grid graph.
        # Residual connections are done directly within the following models.
        # The final MLP to output the predictions is done outside this module.
        grid_nfeat, _ = self.mesh2grid_gnn(mesh_nfeat, grid_nfeat, edge_index, mesh2grid_efeat)

        return grid_nfeat
