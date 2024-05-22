"""
Base classes defining our software components
and their interfaces
"""
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import einops
import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_tensor_fn
from tqdm import tqdm

from py4cast.plots import DomainInfo
from py4cast.utils import RegisterFieldsMixin, torch_save


@dataclass(slots=True)
class TensorWrapper:
    """
    Wrapper around a torch tensor.
    We do this to allow lightning's introspection to see our batch size
    and move our tensors to the right device.
    """

    tensor: torch.Tensor


class NamedTensor(TensorWrapper):
    """
    NamedTensor is a wrapper around a torch tensor
    adding several attributes :

    * a 'names' attribute with the names of the
    tensor's dimensions (like https://pytorch.org/docs/stable/named_tensor.html).

    Torch's named tensors are still experimental and subject to change.

    * a 'feature_names' attribute containing the names of the features
    along the last dimension of the tensor.

    NamedTensor can be concatenated along the last dimension
    using the | operator.
    nt3 = nt1 | nt2
    """

    SPATIAL_DIM_NAMES = ("lat", "lon", "ngrid")

    def __init__(
        self, tensor: torch.Tensor, names: List[str], feature_names: List[str]
    ):
        if len(tensor.shape) != len(names):
            raise ValueError(
                f"Number of names ({len(names)}) must match number of dimensions ({len(tensor.shape)})"
            )
        if tensor.shape[-1] != len(feature_names):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}:{feature_names}) must match"
                f"number of features ({tensor.shape[-1]})"
            )

        super().__init__(tensor)

        self.names = names
        # build lookup table for fast indexing
        self.feature_names_to_idx = {
            feature_name: idx for idx, feature_name in enumerate(feature_names)
        }
        self.feature_names = feature_names

    @property
    def ndims(self):
        """
        Number of dimensions of the tensor.
        """
        return len(self.names)

    @property
    def num_spatial_dims(self):
        """
        Number of spatial dimensions of the tensor.
        """
        return len([x for x in self.names if x in self.SPATIAL_DIM_NAMES])

    def __str__(self):
        head = "--- NamedTensor ---\n"
        head += f"Names: {self.names}\nTensor Shape: {self.tensor.shape})\nFeatures:\n"
        table = [
            [feature, self[feature].min(), self[feature].max()]
            for feature in self.feature_names
        ]
        headers = ["Feature name", "Min", "Max"]
        table_string = str(tabulate(table, headers=headers, tablefmt="simple_outline"))
        return head + table_string

    def __or__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        """
        Concatenate two NamedTensors along the last dimension.
        """
        if other is None:
            return self

        if not isinstance(other, NamedTensor):
            raise ValueError("Can only concatenate NamedTensor with NamedTensor")

        # check features names are distinct between the two tensors
        if set(self.feature_names) & set(other.feature_names):
            raise ValueError(
                f"Feature names must be distinct between the two tensors for"
                f"unambiguous concat, self:{self.feature_names} other:{other.feature_names}"
            )

        if self.names != other.names:
            raise ValueError(
                f"NamedTensors must have the same dimension names to concatenate, self:{self.names} other:{other.names}"
            )
        try:
            return NamedTensor(
                torch.cat([self.tensor, other.tensor], dim=-1),
                self.names.copy(),
                self.feature_names + other.feature_names,
            )
        except Exception as e:
            raise ValueError(f"Error while concatenating {self} and {other}") from e

    def __ror__(self, other: Union["NamedTensor", None]) -> "NamedTensor":
        return self.__or__(other)

    @staticmethod
    def concat(nts: List["NamedTensor"]) -> "NamedTensor":
        """
        Safely concat a list of NamedTensors along the last dimension
        in one shot.
        """
        if len(nts) == 0:
            raise ValueError("Cannot concatenate an empty list of NamedTensors")
        if len(nts) == 1:
            return nts[0].clone()
        else:
            # Check features names are distinct between the n named tensors
            feature_names = set()
            for nt in nts:
                if feature_names & set(nt.feature_names):
                    raise ValueError(
                        f"Feature names must be distinct between the named tensors to concat"
                        f"Found duplicates: {feature_names & set(nt.feature_names)}"
                    )
                feature_names |= set(nt.feature_names)

            # Check that all named tensors have the same names
            if not all(nt.names == nts[0].names for nt in nts):
                raise ValueError(
                    "NamedTensors must have the same dimension names to concatenate"
                )

            # Concat in one shot
            return NamedTensor(
                torch.cat([nt.tensor for nt in nts], dim=-1),
                nts[0].names.copy(),
                list(chain.from_iterable(nt.feature_names for nt in nts)),
            )

    def clone(self):
        return NamedTensor(
            tensor=deepcopy(self.tensor).to(self.tensor.device),
            names=self.names.copy(),
            feature_names=self.feature_names.copy(),
        )

    def __getitem__(self, feature_name: str) -> torch.Tensor:
        """
        Get one feature of the tensor by name.
        """
        try:
            return self.tensor[..., self.feature_names_to_idx[feature_name]].unsqueeze(
                -1
            )
        except KeyError:
            raise ValueError(
                f"Feature {feature_name} not found in {self.feature_names}"
            )

    def type_(self, new_type):
        """
        Modify the type of the underlying torch tensor
        by calling torch's .type method

        in_place operation for this class, the internal
        tensor is replaced by the new one.
        """
        self.tensor = self.tensor.type(new_type)

    def flatten_(self, flatten_dim_name: str, start_dim: int, end_dim: int):
        """
        Flatten the underlying tensor from start_dim to end_dim.
        Deletes flattened dimension names and insert
        the new one.
        """
        self.tensor = torch.flatten(self.tensor, start_dim, end_dim)

        # Remove the flattened dimensions from the names
        # and insert the replacing one
        self.names = (
            self.names[:start_dim] + [flatten_dim_name] + self.names[end_dim + 1 :]
        )

    def dim_size(self, dim_name: str) -> int:
        """
        Return the size of a dimension given its name.
        """
        try:
            return self.tensor.size(self.names.index(dim_name))
        except ValueError as ve:
            raise ValueError(f"Dimension {dim_name} not found in {self.names}") from ve

    def unsqueeze_and_expand_from_(self, other: "NamedTensor"):
        """
        Unsqueeze and expand the tensor to have the same number of spatial dimensions
        as another NamedTensor.
        Injects new dimensions where the missing names are.
        """
        missing_names = set(other.names) - set(self.names)
        missing_names &= set(self.SPATIAL_DIM_NAMES)

        if missing_names:
            index_to_unsqueeze = [
                (name, other.names.index(name)) for name in missing_names
            ]
            for name, idx in sorted(index_to_unsqueeze, key=lambda x: x[1]):
                self.tensor = torch.unsqueeze(self.tensor, idx)
                self.names.insert(idx, name)

            expander = []
            for _, name in enumerate(self.names):
                expander.append(other.dim_size(name) if name in missing_names else -1)

            self.tensor = self.tensor.expand(*expander)

    @staticmethod
    def new_like(tensor: torch.Tensor, other: "NamedTensor") -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor
        and a tensor of the same shape as the input tensor.
        """
        return NamedTensor(tensor, other.names.copy(), other.feature_names.copy())

    @staticmethod
    def expand_to_batch_like(
        tensor: torch.Tensor, other: "NamedTensor"
    ) -> "NamedTensor":
        """
        Create a new NamedTensor with the same names and feature names as another NamedTensor
        with an extra first dimension called 'batch' using the supplied tensor.
        Supplied new 'batched' tensor must have one more dimension than other.
        """
        names = ["batch"] + other.names
        if tensor.dim() != len(names):
            raise ValueError(
                f"Tensor dim {tensor.dim()} must match number of names {len(names)} with extra batch dim"
            )
        return NamedTensor(tensor, ["batch"] + other.names, other.feature_names.copy())


@dataclass(slots=True)
class Item:
    """
    Dataclass holding one Item
    """

    inputs: List[NamedTensor]
    outputs: List[NamedTensor]

    forcing: Union[
        List[NamedTensor], None
    ] = None  # Pour les champs de forcage (SST, Rayonnement, ... )

    # J'hesite entre inputs/outputs et pronostics/diagnostics
    # Les concepts sont totalement different (les variables ne contiendrait pas la meme chose,
    # le get_item serait different et le unroll_prediction aussi)
    # A discuter. J'aurais tendance a m'orienter vers pronostics/diagnostics qui me parait plus adapter
    # (et est en phase avec la denomination du CEP)
    # Ca ferait peut etre moins de trucs etrange dans les denominations

    def __str__(self) -> str:
        """
        Utility method to explore a batch/item shapes and names.
        """
        table = []
        for attr in (f.name for f in fields(self)):
            if getattr(self, attr) is not None:
                for item in getattr(self, attr):
                    names = item.names[0] if len(item.names) == 1 else item.names
                    table.append(
                        [
                            attr,
                            names,
                            list(item.tensor.shape),
                            item.feature_names,
                            item.tensor.min(),
                            item.tensor.max(),
                        ]
                    )
        headers = [
            "Type",
            "Dimension Names",
            "Torch Shape",
            "feature names",
            "Min",
            "Max",
        ]
        return str(tabulate(table, headers=headers, tablefmt="simple_outline"))


@dataclass
class ItemBatch(Item):
    """
    Dataclass holding a batch of items.
    """

    @cached_property
    def batch_size(self):
        return self.inputs[0].dim_size("batch")

    @cached_property
    def num_input_steps(self):
        return self.inputs[0].dim_size("in_step")

    @cached_property
    def num_pred_steps(self):
        return self.outputs[0].dim_size("out_step")

    def cat(
        self, attr: Literal["inputs", "outputs", "forcing"], ndims: int
    ) -> Union[NamedTensor, None]:
        """
        Cat all the NamedTensor of the same attr having a specific number of dims.

        It assumes that all the variable are on the same grid
        and only the last dimension is different

        Args:
            attr (str): inputs/outputs/forcing
            dim (int): Dimension of the state variable to consider

        """
        tensors = [x for x in getattr(self, attr) if x.num_spatial_dims == ndims]

        if len(tensors) == 0:
            return None
        else:
            return NamedTensor.concat(tensors)

    def warn_if_1d(self) -> None:
        """
        Issue a warning if 1D state variables are present in the batch.
        """
        extra_names_in = [x.names for x in self.inputs if x.num_spatial_dims == 0]
        extra_names_out = [x.names for x in self.outputs if x.num_spatial_dims == 0]

        if len(extra_names_in) > 0:
            warnings.warn(f"1D inputs are ignored {extra_names_in}")
        if len(extra_names_out) > 0:
            warnings.warn(f"1D outputs are ignored {extra_names_out}")

    def cat_forcing(self) -> Union[NamedTensor, None]:
        """
        Concat (Torch cat) all forcing in 2D fields along features dimension.
        1D forcing values are replicated at all grid points.
        """
        f1 = self.cat("forcing", 0)

        if f1 is not None:
            f1.unsqueeze_and_expand_from_(self.inputs[0])

        f2 = self.cat("forcing", 2)
        return [f1 | f2] if any((f1, f2)) else None

    def cat_2D(self) -> "ItemBatch":
        """
        Concat (Torch cat) all inputs/outputs/forcing in 2D fields along features dimension.
        1d (ndims=1) state variables for input and outputs are ignored (only taken into account for forcing).
        2d and 3d are concatenated.
        """
        cat_inputs = [self.cat("inputs", 2)]
        cat_outputs = [self.cat("outputs", 2)]

        self.warn_if_1d()

        cat_forcing = self.cat_forcing()

        return ItemBatch(inputs=cat_inputs, outputs=cat_outputs, forcing=cat_forcing)


def collate_fn(items: List[Item]) -> ItemBatch:
    """
    Collate a list of item. Add one dimension to each state variable.
    Necessary to form batch.
    """
    # Here we postpone that for each batch the same dimension should be present.
    batch_of_items = {}

    # Iterate over inputs, outputs and forcing fields
    for field_name in (f.name for f in fields(Item)):
        batch_of_items[field_name] = []
        item_zero_field_value = getattr(items[0], field_name)
        if item_zero_field_value is not None:
            for i in range(len(item_zero_field_value)):
                new_tensor = collate_tensor_fn(
                    [getattr(item, field_name)[i].tensor for item in items]
                ).type(torch.float32)
                new_state = NamedTensor.expand_to_batch_like(
                    new_tensor, item_zero_field_value[i]
                )
                batch_of_items[field_name].append(new_state)
        else:
            batch_of_items[field_name] = None
    return ItemBatch(**batch_of_items)


@dataclass
class Statics(RegisterFieldsMixin):
    """
    Static fields of the dataset.
    Tensor can be registered as buffer in a lightning module
    using the register_buffers method.
    """

    # border_mask: torch.Tensor
    grid_static_features: NamedTensor
    grid_shape: Tuple[int, int]
    border_mask: torch.Tensor = field(init=False)
    interior_mask: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.border_mask = self.grid_static_features["border_mask"]
        self.interior_mask = 1.0 - self.border_mask

    @cached_property
    def meshgrid(self) -> torch.Tensor:
        """
        Return a tensor concatening X,Y
        """
        return einops.rearrange(
            torch.cat(
                [
                    self.grid_static_features["x"],
                    self.grid_static_features["y"],
                ],
                dim=-1,
            ),
            ("x y n -> n x y"),
        )


@dataclass
class Stats:
    fname: Path

    def __post_init__(self):
        self.stats = torch.load(self.fname, "cpu")

    def items(self):
        return self.stats.items()

    def __getitem__(self, shortname: str):
        return self.stats[shortname]

    def to_list(
        self, aggregate: Literal["mean", "std", "min", "max"], shortnames: List[str]
    ) -> list:
        """
        Get a tensor with the stats inside.
        The order is the one of the shortnames.

        Args:
            aggregate : Statistics wanted
            names (List[str]): Field for which we want stats

        Returns:
            _type_: _description_
        """
        if len(shortnames) > 0:
            return torch.stack(
                [self[name][aggregate] for name in shortnames], dim=0
            ).type(torch.float32)
        else:
            return []


@dataclass(slots=True)
class DatasetInfo:
    """
    This dataclass holds all the informations
    about the dataset that other classes
    and functions need to interact with it.
    """

    name: str  # Name of the dataset
    domain_info: DomainInfo  # Information used for plotting
    units: Dict[str, str]  # d[shortname] = unit (str)
    weather_dim: int
    forcing_dim: int
    step_duration: float  # Duration (in hour) of one step in the dataset. 0.25 means 15 minutes.
    statics: Statics  # A lot of static variable
    stats: Stats
    diff_stats: Stats
    state_weights: Dict[str, float]
    shortnames: Callable = None

    def summary(self):
        """
        Print a table summarizing variables present in the dataset (and their role)
        """
        print(f"\n Summarizing {self.name} \n")
        print(f"Step_duration {self.step_duration}")
        print(f"Static fields {self.statics.grid_static_features.feature_names}]")

        for p in ["forcing", "input_output", "diagnostic"]:
            names = self.shortnames(p)
            mean = self.stats.to_list("mean", names)
            std = self.stats.to_list("std", names)
            mini = self.stats.to_list("min", names)
            maxi = self.stats.to_list("max", names)
            units = [self.units[name] for name in names]
            if p != "forcing":
                diff_mean = self.diff_stats.to_list("mean", names)
                diff_std = self.diff_stats.to_list("std", names)
                weight = [self.state_weights[name] for name in names]

                data = list(
                    zip(
                        names, units, mean, std, mini, maxi, diff_mean, diff_std, weight
                    )
                )
                table = tabulate(
                    data,
                    headers=[
                        "Name",
                        "Unit",
                        "Mean",
                        "Std",
                        "Minimum",
                        "Maximum",
                        "DiffMean",
                        "DiffStd",
                        "Weight in Loss",
                    ],
                    tablefmt="simple_outline",
                )
            else:
                data = list(zip(names, units, mean, std, mini, maxi))
                table = tabulate(
                    data,
                    headers=["Name", "Unit", "Mean", "Std", "Minimun", "Maximum"],
                    tablefmt="simple_outline",
                )
            if data:
                print(p.upper())  # Print the kind of variable
                print(table)  # Print the table


@dataclass(slots=True)
class TorchDataloaderSettings:
    """
    Settings for the torch dataloader
    """

    batch_size: int = 1
    num_workers: int = 10
    pin_memory: bool = False
    prefetch_factor: Union[int, None] = None
    persistent_workers: bool = False


class DatasetABC(ABC):
    """
    Abstract Base class defining the mandatory
    properties and methods a dataset MUST
    implement.
    """

    @abstractmethod
    def torch_dataloader(self, tl_settings: TorchDataloaderSettings) -> DataLoader:
        """
        Builds a torch dataloader from self.
        """

    @abstractproperty
    def dataset_info(self) -> DatasetInfo:
        """
        Return the object DatasetInfo
        """

    @abstractproperty
    def meshgrid(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """

    @cached_property
    def grid_shape(self) -> tuple:
        x, _ = self.meshgrid
        return x.shape

    @abstractproperty
    def geopotential_info(self) -> np.array:
        """
        array of shape (num_lat, num_lon)
        with geopotential value for each datapoint
        """

    @abstractproperty
    def limited_area(self) -> bool:
        """
        Returns True if the dataset is
        compatible with Limited area models
        """

    @abstractproperty
    def split(self) -> Literal["train", "valid", "test"]:
        pass

    @abstractproperty
    def cache_dir(self) -> Path:
        """
        Cache directory of the dataset.
        Used at least to get statistics.
        """
        pass

    def compute_parameters_stats(self):
        """
        Compute mean and standard deviation for this dataset.
        """
        means = []
        squares = []
        mins = []
        maxs = []
        print(
            "We are going to compute statistics on the dataset. This can take a while."
        )
        if self.settings.standardize:
            raise ValueError("Your dataset should not be standardized.")
        # When computing stat may force every body to be input/ouput
        for batch in tqdm(self.torch_dataloader()):
            # Here we assume that data are in 2 or 3 D
            inputs = torch.cat([x.tensor for x in batch.inputs], dim=-1)
            outputs = torch.cat([x.tensor for x in batch.outputs], dim=-1)
            # Check that no variable is a forcing variable
            f_names = [x.names for x in batch.forcing if x.ndims == 2]
            if len(f_names) > 0:
                raise ValueError(
                    "During statistics computation no forcing are accepted. All variables should be in in/out mode."
                )

            in_out = torch.cat([inputs, outputs], dim=1)
            in_out = in_out.flatten(
                1, 3
            )  # Flatten everybody to be (Batch, X, Features)

            means.append(in_out.mean(dim=1))
            squares.append((in_out**2).mean(dim=1))
            maxi, _ = torch.max(in_out, 1)
            mini, _ = torch.min(in_out, 1)
            mins.append(mini)  # (N_batch, d_features,)
            maxs.append(maxi)  # (N_batch, d_features,)

        mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
        second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
        std = torch.sqrt(second_moment - mean**2)  # (d_features)
        maxi, _ = torch.max(torch.cat(maxs, dim=0), dim=0)
        mini, _ = torch.min(torch.cat(mins, dim=0), dim=0)

        # Store in dictionnary
        store_d = {}
        # Get variables names
        names = []
        [names.extend(x.feature_names) for x in batch.inputs]
        # Storing variable statistics
        for i, name in enumerate(names):
            store_d[name] = {
                "mean": mean[i],
                "std": std[i],
                "min": mini[i],
                "max": maxi[i],
            }
        torch_save(store_d, self.cache_dir / "parameters_stats.pt")

    def compute_time_step_stats(self):
        diff_means = []
        diff_squares = []
        if not self.settings.standardize:
            print(
                f"Your dataset {self} is not normalized. If you do not normalized your output it could be intended."
            )
            print("Otherwise consider standardizing your inputs.")
        for batch in tqdm(self.torch_dataloader()):
            # Here we postpone that data are in 2 or 3 D
            inputs = torch.cat([x.tensor for x in batch.inputs], dim=-1)
            outputs = torch.cat([x.tensor for x in batch.outputs], dim=-1)

            # Check that no variable is a forcing variable
            f_names = [x.feature_names for x in batch.forcing if x.ndims == 2]
            if any(f_names):
                raise ValueError(
                    "During statistics computation no forcing are accepted. All variables should be in in/out mode."
                )

            in_out = torch.cat([inputs, outputs], dim=1)
            diff = (
                in_out[:, 1:] - in_out[:, :-1]
            )  # Substract information on time dimension
            diff = diff.flatten(1, 3)  # Flatten everybody to be (Batch, X, Features)

            diff_means.append(diff.mean(dim=1))
            diff_squares.append((diff**2).mean(dim=1))

        # This should not work
        # We should be aware that dataset shoul
        diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)
        diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)
        store_d = {}
        # Get variables names
        names = []
        [names.extend(x.feature_names) for x in batch.inputs]
        # Storing variable statistics
        for i, name in enumerate(names):
            store_d[name] = {
                "mean": diff_mean[i],
                "std": diff_std[i],
            }
        torch_save(store_d, self.cache_dir / "diff_stats.pt")

    def dataset_extra_statics(self) -> List[NamedTensor]:
        """
        Datasets can override this method to add
        more static data.
        """
        return []

    @cached_property
    def grid_static_features(self):
        """
        Grid static features
        """
        # -- Static grid node features --
        xy = self.meshgrid  # (2, N_x, N_y)
        grid_xy = torch.tensor(xy)
        # Need to rearange
        pos_max = torch.max(torch.abs(grid_xy))
        grid_xy = (
            einops.rearrange(grid_xy, ("n x y -> x y n")) / pos_max
        )  # Rearange and divide  by maximum coordinate

        # (Nx, Ny, 1)
        geopotential = torch.tensor(self.geopotential_info).unsqueeze(
            2
        )  # (N_x, N_y, 1)
        gp_min = torch.min(geopotential)
        gp_max = torch.max(geopotential)
        # Rescale geopotential to [0,1]
        geopotential = (geopotential - gp_min) / (gp_max - gp_min)  # (N_x,N_y, 1)
        grid_border_mask = torch.tensor(self.border_mask).unsqueeze(2)  # (N_x, N_y,1)

        feature_names = []
        for x in self.dataset_extra_statics:
            feature_names += x.feature_names
        state_var = NamedTensor(
            tensor=torch.cat(
                [grid_xy, geopotential, grid_border_mask]
                + [x.tensor for x in self.dataset_extra_statics],
                dim=-1,
            ),
            feature_names=["x", "y", "geopotential", "border_mask"]
            + feature_names,  # Noms des champs 2D
            names=["lat", "lon", "features"],
        )
        state_var.type_(torch.float32)
        return state_var

    @cached_property
    def statics(self) -> Statics:
        return Statics(
            **{
                "grid_static_features": self.grid_static_features,
                "grid_shape": self.grid_shape,
            }
        )

    @cached_property
    def stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "parameters_stats.pt")

    @cached_property
    def diff_stats(self) -> Stats:
        return Stats(fname=self.cache_dir / "diff_stats.pt")

    @abstractproperty
    def diagnostic_dim(self) -> int:
        """
        Return the number of diagnostic variables (output only)
        """
        pass

    @abstractclassmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_tests: int,
    ) -> Tuple["DatasetABC", "DatasetABC", "DatasetABC"]:
        """
        Load a dataset from a json file + the number of expected timesteps
        taken as inputs (num_input_steps) and to predict (num_pred_steps)
        Return the train, valid and test datasets, in that order
        """
        pass