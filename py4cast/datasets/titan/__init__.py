import datetime as dt
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple, Union

import gif
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import typer
import xarray as xr
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from py4cast.datasets.base import (
    DatasetABC,
    DatasetInfo,
    Grid,
    GridConfig,
    Item,
    NamedTensor,
    Param,
    ParamConfig,
    Period,
    Settings,
    Stats,
    TorchDataloaderSettings,
    collate_fn,
    get_param_list,
)
from py4cast.datasets.titan.settings import (
    DEFAULT_CONFIG,
    FORMATSTR,
    METADATA,
    SCRATCH_PATH,
)
from py4cast.forcingutils import generate_toa_radiation_forcing, get_year_hour_forcing
from py4cast.plots import DomainInfo
from py4cast.utils import merge_dicts

app = typer.Typer()


def get_weight_per_lvl(
    level: int,
    level_type: Literal["isobaricInhPa", "heightAboveGround", "surface", "meanSea"],
):
    if level_type == "isobaricInhPa":
        return 1 + (level) / (1000)
    else:
        return 2


#############################################################
#                            GRID                           #
#############################################################


def load_grid_info(name: str) -> GridConfig:
    path = SCRATCH_PATH / f"conf_{name}.grib"
    conf_ds = xr.open_dataset(path)
    grid_info = METADATA["GRIDS"][name]
    full_size = grid_info["size"]
    landsea_mask = None
    grid_conf = GridConfig(
        full_size,
        conf_ds.latitude.values,
        conf_ds.longitude.values,
        conf_ds.h.values,
        landsea_mask,
    )
    return grid_conf


#############################################################
#                              PARAMS                       #
#############################################################


def load_param_info(name: str) -> ParamConfig:
    info = METADATA["WEATHER_PARAMS"][name]
    grib_name = info["grib"]
    grib_param = info["param"]
    unit = info["unit"]
    level_type = info["type_level"]
    long_name = info["long_name"]
    grid = info["grid"]
    if grid not in ["PAAROME_1S100", "PAAROME_1S40", "PA_01D"]:
        raise NotImplementedError(
            "Parameter native grid must be in ['PAAROME_1S100', 'PAAROME_1S40', 'PA_01D']"
        )
    return ParamConfig(unit, level_type, long_name, grid, grib_name, grib_param)


def get_grid_coords(param:Param) -> List[int]:
    return METADATA["GRIDS"][param.grid.name]["extent"]


def get_filepath(
    ds_name: str,
    param: Param,
    date: dt.datetime,
    file_format: Literal["npy", "grib"],
) -> Path:
    """
    Returns the path of the file containing the parameter data.
    - in grib format, data is grouped by level type.
    - in npy format, data is saved as npy, rescaled to the wanted grid, and each
    2D array is saved as one file to optimize IO during training."""
    if file_format == "grib":
        folder = SCRATCH_PATH / "grib" / date.strftime(FORMATSTR)
        return folder / param.grib_name
    else:
        npy_path = get_dataset_path(ds_name, param.grid) / "data"
        filename = f"{param.name}_{param.level}_{param.level_type}.npy"
        return npy_path / date.strftime(FORMATSTR) / filename


def process_sample_dataset(ds_name: str, date: dt.datetime, params: List[Param]):
    """Saves each 2D parameter data of the given date as one NPY file."""
    for param in params:
        dest_file = get_filepath(ds_name, param, date, "npy")
        dest_file.parent.mkdir(exist_ok=True)
        if not dest_file.exists():
            try:
                arr = load_data_from_disk(ds_name, param, date, "grib")
                np.save(dest_file, arr)
            except Exception as e:
                print(e)
                print(
                    f"WARNING: Could not load grib data {param.name} {param.level} {date}. Skipping sample."
                )
                break


def fit_to_grid(
    param: Param,
    arr: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    get_grid_coords: Callable[[Param], List[str]],
) -> np.ndarray:
    # already on good grid, nothing to do:
    if param.grid.name == param.native_grid:
        return arr

    # crop arpege data to arome domain:
    if param.native_grid == "PA_01D" and param.grid.name in [
        "PAAROME_1S100",
        "PAAROME_1S40",
    ]:
        grid_coords = get_grid_coords(param)
        # Mask ARPEGE data to AROME bounding box
        mask_lon = (lons >= grid_coords[2]) & (lons <= grid_coords[3])
        mask_lat = (lats >= grid_coords[1]) & (lats <= grid_coords[0])
        arr = arr[mask_lat, :][:, mask_lon]

    anti_aliasing = param.grid.name == "PAAROME_1S40"  # True if downsampling
    # upscale or downscale to grid resolution:
    return resize(arr, param.grid.full_size, anti_aliasing=anti_aliasing)


@lru_cache(maxsize=50)
def read_grib(path_grib: Path) -> xr.Dataset:
    return xr.load_dataset(path_grib, engine="cfgrib", backend_kwargs={"indexpath": ""})


def load_data_grib(param: Param, path: Path) -> np.ndarray:
    ds = read_grib(path)
    assert param.grib_param is not None
    level_type = ds[param.grib_param].attrs["GRIB_typeOfLevel"]
    lats = ds.latitude.values
    lons = ds.longitude.values
    if level_type != "isobaricInhPa":  # Only one level
        arr = ds[param.grib_param].values
    else:
        arr = ds[param.grib_param].sel(isobaricInhPa=param.level).values
    return arr, lons, lats


def load_data_from_disk(
    ds_name: str,
    param: Param,
    date: dt.datetime,
    file_format: Literal["npy", "grib"] = "grib",
):
    """
    Function to load invidiual parameter and lead time from a file stored in disk
    """
    data_path = get_filepath(ds_name, param, date, file_format)
    if file_format == "grib":
        arr, lons, lats = load_data_grib(param, data_path)
        arr = fit_to_grid(arr, lons, lats)
    else:
        arr = np.load(data_path)

    subdomain = param.grid.subdomain
    arr = arr[subdomain[0] : subdomain[1], subdomain[2] : subdomain[3]]
    if file_format == "grib":
        arr = arr[::-1]
    return arr  # invert latitude


def exists(
    ds_name: str,
    param: Param,
    date: dt.datetime,
    file_format: Literal["npy", "grib"] = "grib",
) -> bool:
    filepath = get_filepath(ds_name, param, date, file_format)
    return filepath.exists()


def get_param_tensor(
    param: Param,
    stats: Stats,
    dates: List[dt.datetime],
    settings: Settings,
    no_standardize: bool = False,
) -> torch.tensor:
    """
    Fetch data on disk fo the given parameter and all involved dates
    Unless specified, normalize the samples with parameter-specific constants
    returns a tensor
    """
    arrays = [
        load_data_from_disk(settings.dataset_name, param, date, settings.file_format)
        for date in dates
    ]
    arr = np.stack(arrays)
    # Extend dimension to match 3D (level dimension)
    if len(arr.shape) != 4:
        arr = np.expand_dims(arr, axis=1)
    arr = np.transpose(arr, axes=[0, 2, 3, 1])  # shape = (steps, lvl, x, y)
    if settings.standardize and not no_standardize:
        name = param.parameter_short_name
        means = np.asarray(stats[name]["mean"])
        std = np.asarray(stats[name]["std"])
        arr = (arr - means) / std
    return torch.from_numpy(arr)


def generate_forcings(
    date: dt.datetime, output_terms: Tuple[float], grid: Grid
) -> List[NamedTensor]:
    """
    Generate all the forcing in this function.
    Return a list of NamedTensor.
    """
    lforcings = []
    time_forcing = NamedTensor(  # doy : day_of_year
        feature_names=["cos_hour", "sin_hour", "cos_doy", "sin_doy"],
        tensor=get_year_hour_forcing(date, output_terms).type(torch.float32),
        names=["timestep", "features"],
    )
    solar_forcing = NamedTensor(
        feature_names=["toa_radiation"],
        tensor=generate_toa_radiation_forcing(
            grid.lat, grid.lon, date, output_terms
        ).type(torch.float32),
        names=["timestep", "lat", "lon", "features"],
    )
    lforcings.append(time_forcing)
    lforcings.append(solar_forcing)

    return lforcings


#############################################################
#                            SAMPLE                         #
#############################################################


@dataclass(slots=True)
class Sample:
    """Describes a sample"""

    date_t0: dt.datetime
    settings: Settings
    params: List[Param]
    stats: Stats
    grid: Grid
    pred_timesteps: Tuple[float] = field(init=False)  # gaps in hours btw step and t0
    all_dates: Tuple[dt.datetime] = field(init=False)  # date of each step

    def __post_init__(self):
        """Setups time variables to be able to define a sample.
        For example for n_inputs = 2, n_preds = 3, step_duration = 3h:
        all_steps = [-1, 0, 1, 2, 3]
        all_timesteps = [-3h, 0h, 3h, 6h, 9h]
        pred_timesteps = [3h, 6h, 9h]
        all_dates = [24/10/22 21:00,  24/10/23 00:00, 24/10/23 03:00, 24/10/23 06:00, 24/10/23 09:00]
        """
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        all_steps = list(range(-n_inputs + 1, n_preds + 1))
        all_timesteps = [self.settings.step_duration * step for step in all_steps]
        self.pred_timesteps = all_timesteps[-n_preds:]
        self.all_dates = [self.date_t0 + dt.timedelta(hours=ts) for ts in all_timesteps]

    def __repr__(self):
        return f"Date T0 {self.date_t0}, leadtimes {self.pred_timesteps}"

    def is_valid(self) -> bool:
        """Check that all the files necessary for this sample exist.

        Args:
            param_list (List): List of parameters
        Returns:
            Boolean: Whether the sample is available or not
        """
        for date in self.all_dates:
            for param in self.params:
                if not exists(
                    self.settings.dataset_name, param, date, self.settings.file_format
                ):
                    print("invalid sample")
                    return False
        return True

    def load(self, no_standardize: bool = False) -> Item:
        """
        Return inputs, outputs, forcings as tensors concatenated into a Item.
        """
        linputs, loutputs = [], []

        # Reading parameters from files
        for param in self.params:
            state_kwargs = {
                "feature_names": [param.parameter_short_name],
                "names": ["timestep", "lat", "lon", "features"],
            }
            if param.kind == "input":
                # forcing is taken for every predicted step
                dates = self.all_dates[-self.settings.num_pred_steps :]
                tensor = get_param_tensor(
                    param, self.stats, dates, self.settings, no_standardize
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))

            elif param.kind == "output":
                dates = self.all_dates[-self.settings.num_pred_steps :]
                tensor = get_param_tensor(
                    param, self.stats, dates, self.settings, no_standardize
                )
                tmp_state = NamedTensor(tensor=tensor, **deepcopy(state_kwargs))
                loutputs.append(tmp_state)

            else:  # input_output
                tensor = get_param_tensor(
                    param, self.stats, self.all_dates, self.settings, no_standardize
                )
                state_kwargs["names"][0] = "timestep"
                tmp_state = NamedTensor(
                    tensor=tensor[-self.settings.num_pred_steps :],
                    **deepcopy(state_kwargs),
                )

                loutputs.append(tmp_state)
                tmp_state = NamedTensor(
                    tensor=tensor[: self.settings.num_input_steps],
                    **deepcopy(state_kwargs),
                )
                linputs.append(tmp_state)

        lforcings = generate_forcings(
            date=self.date_t0, output_terms=self.pred_timesteps, grid=self.grid
        )

        for forcing in lforcings:
            forcing.unsqueeze_and_expand_from_(linputs[0])

        return Item(
            inputs=NamedTensor.concat(linputs),
            outputs=NamedTensor.concat(loutputs),
            forcing=NamedTensor.concat(lforcings),
        )

    def plot(self, item: Item, step: int, save_path: Path = None) -> None:
        # Retrieve the named tensor
        ntensor = item.inputs if step <= 0 else item.outputs

        # Retrieve the timestep data index
        if step <= 0:  # input step
            index_tensor = step + self.settings.num_input_steps - 1
        else:  # output step
            index_tensor = step - 1

        # Sort parameters by level, to plot each level on one line
        levels = sorted(list(set([p.level for p in self.params])))
        dict_params = {level: [] for level in levels}
        for param in self.params:
            if param.parameter_short_name in ntensor.feature_names:
                dict_params[param.level].append(param)

        # Groups levels 0m, 2m and 10m on one "surf" level
        dict_params["surf"] = []
        for lvl in [0, 2, 10]:
            if lvl in levels:
                dict_params["surf"] += dict_params.pop(lvl)

        # Plot settings
        kwargs = {"projection": self.grid.projection}
        nrows = len(dict_params.keys())
        ncols = max([len(param_list) for param_list in dict_params.values()])
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 15), subplot_kw=kwargs)

        for i, level in enumerate(dict_params.keys()):
            for j, param in enumerate(dict_params[level]):
                pname = param.parameter_short_name
                tensor = ntensor[pname][index_tensor, :, :, 0]
                arr = tensor.numpy()[::-1]  # invert latitude
                vmin, vmax = self.stats[pname]["min"], self.stats[pname]["max"]
                img = axs[i, j].imshow(
                    arr, vmin=vmin, vmax=vmax, extent=self.grid.grid_limits
                )
                axs[i, j].set_title(pname)
                axs[i, j].coastlines(resolution="50m")
                cbar = fig.colorbar(img, ax=axs[i, j], fraction=0.04, pad=0.04)
                cbar.set_label(param.unit)

        hours_delta = dt.timedelta(hours=self.settings.step_duration * step)
        plt.suptitle(f"Run: {self.date_t0} - Valid time: {self.date_t0 + hours_delta}")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
            plt.close()

    @gif.frame
    def plot_frame(self, item: Item, step: int) -> None:
        self.plot(item, step)

    def plot_gif(self, save_path: Path):
        # We don't want to standardize data for plots
        item = self.load(no_standardize=True)
        frames = []
        n_inputs, n_preds = self.settings.num_input_steps, self.settings.num_pred_steps
        steps = list(range(-n_inputs + 1, n_preds + 1))
        for step in tqdm.tqdm(steps, desc="Making gif"):
            frame = self.plot_frame(item, step)
            frames.append(frame)
        gif.save(frames, str(save_path), duration=250)


#############################################################
#                            DATASET                        #
#############################################################


def get_dataset_path(name: str, grid: Grid):
    str_subdomain = "-".join([str(i) for i in grid.subdomain])
    subdataset_name = f"{name}_{grid.name}_{str_subdomain}"
    return SCRATCH_PATH / "subdatasets" / subdataset_name


class TitanDataset(DatasetABC, Dataset):
    # Si on doit travailler avec plusieurs grilles, on fera un super dataset qui contient
    # plusieurs datasets chacun sur une seule grille
    def __init__(
        self,
        name: str,
        grid: Grid,
        period: Period,
        params: List[Param],
        settings: Settings,
    ):
        self.name = name
        self.grid = grid
        if grid.name not in ["PAAROME_1S100", "PAAROME_1S40"]:
            raise NotImplementedError(
                "Grid must be in ['PAAROME_1S100', 'PAAROME_1S40']"
            )
        self.period = period
        self.params = params
        self.settings = settings
        self.shuffle = self.period.name == "train"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_input, n_pred = self.settings.num_input_steps, self.settings.num_pred_steps
        filename = f"valid_samples_{self.period.name}_{n_input}_{n_pred}.txt"
        self.valid_samples_file = self.cache_dir / filename

    @cached_property
    def dataset_info(self) -> DatasetInfo:
        """Returns a DatasetInfo object describing the dataset.

        Returns:
            DatasetInfo: _description_
        """
        shortnames = {
            "input": self.shortnames("input"),
            "input_output": self.shortnames("input_output"),
            "output": self.shortnames("output"),
        }
        return DatasetInfo(
            name=str(self),
            domain_info=self.domain_info,
            shortnames=shortnames,
            units=self.units,
            weather_dim=self.weather_dim,
            forcing_dim=self.forcing_dim,
            step_duration=self.settings.step_duration,
            statics=self.statics,
            stats=self.stats,
            diff_stats=self.diff_stats,
            state_weights=self.state_weights,
        )

    def write_list_valid_samples(self):
        print(f"Writing list of valid samples for {self.period.name} set...")
        with open(self.valid_samples_file, "w") as f:
            for date in tqdm.tqdm(
                self.period.date_list, f"{self.period.name} samples validation"
            ):
                sample = Sample(date, self.settings, self.params, self.stats, self.grid)
                if sample.is_valid():
                    f.write(f"{date.strftime('%Y-%m-%d_%Hh%M')}\n")

    @cached_property
    def sample_list(self):
        """Creates the list of samples."""
        print("Start creating samples...")
        stats = self.stats if self.settings.standardize else None
        if self.valid_samples_file.exists():
            print(f"Retrieving valid samples from file {self.valid_samples_file}")
            with open(self.valid_samples_file, "r") as f:
                dates_str = [line[:-1] for line in f.readlines()]
                dateformat = "%Y-%m-%d_%Hh%M"
                dates = [dt.datetime.strptime(ds, dateformat) for ds in dates_str]
                dates = list(set(dates).intersection(set(self.period.date_list)))
                samples = [
                    Sample(date, self.settings, self.params, stats, self.grid)
                    for date in dates
                ]
        else:
            print(
                f"Valid samples file {self.valid_samples_file} does not exist. Computing samples list..."
            )
            samples = []
            for date in tqdm.tqdm(self.period.date_list):
                sample = Sample(date, self.settings, self.params, stats, self.grid)
                if sample.is_valid():
                    samples.append(sample)
        print(f"--> All {len(samples)} {self.period.name} samples are now defined")
        return samples

    @cached_property
    def dataset_extra_statics(self):
        """Add the LandSea Mask to the statics."""
        return [
            NamedTensor(
                feature_names=["LandSeaMask"],
                tensor=torch.from_numpy(self.grid.landsea_mask)
                .type(torch.float32)
                .unsqueeze(2),
                names=["lat", "lon", "features"],
            )
        ]

    def __len__(self):
        return len(self.sample_list)

    @cached_property
    def forcing_dim(self) -> int:
        """Return the number of forcings."""
        res = 4  # For date (hour and year)
        res += 1  # For solar forcing
        for param in self.params:
            if param.kind == "input":
                res += 1
        return res

    @cached_property
    def weather_dim(self) -> int:
        """Return the dimension of pronostic variable."""
        res = 0
        for param in self.params:
            if param.kind == "input_output":
                res += 1
        return res

    def __getitem__(self, index: int) -> Item:
        sample = self.sample_list[index]
        item = sample.load()
        return item

    @classmethod
    def from_dict(
        cls,
        name: str,
        conf: dict,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:

        conf["grid"]["load_grid_info_func"] = load_grid_info
        grid = Grid(**conf["grid"])

        param_list = get_param_list(conf, grid, load_param_info, get_weight_per_lvl)

        train_settings = Settings(
            name, num_input_steps, num_pred_steps_train, **conf["settings"]
        )
        train_period = Period(**conf["periods"]["train"], name="train")
        train_ds = TitanDataset(name, grid, train_period, param_list, train_settings)

        valid_settings = Settings(
            name, num_input_steps, num_pred_steps_val_test, **conf["settings"]
        )
        valid_period = Period(**conf["periods"]["valid"], name="valid")
        valid_ds = TitanDataset(name, grid, valid_period, param_list, valid_settings)

        test_period = Period(**conf["periods"]["test"], name="test")
        test_ds = TitanDataset(name, grid, test_period, param_list, valid_settings)
        return train_ds, valid_ds, test_ds

    @classmethod
    def from_json(
        cls,
        fname: Path,
        num_input_steps: int,
        num_pred_steps_train: int,
        num_pred_steps_val_test: int,
        config_override: Union[Dict, None] = None,
    ) -> Tuple["TitanDataset", "TitanDataset", "TitanDataset"]:
        with open(fname, "r") as fp:
            conf = json.load(fp)
            if config_override is not None:
                conf = merge_dicts(conf, config_override)
        return cls.from_dict(
            fname.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )

    def __str__(self) -> str:
        return f"titan_{self.grid.name}"

    def torch_dataloader(
        self, tl_settings: TorchDataloaderSettings = TorchDataloaderSettings()
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=tl_settings.batch_size,
            num_workers=tl_settings.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=tl_settings.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=tl_settings.pin_memory,
        )

    @property
    def meshgrid(self) -> np.array:
        """array of shape (2, num_lat, num_lon) of (X, Y) values"""
        return self.grid.meshgrid

    @property
    def geopotential_info(self) -> np.array:
        """array of shape (num_lat, num_lon) with geopotential value for each datapoint"""
        return self.grid.geopotential

    @property
    def border_mask(self) -> np.array:
        return self.grid.border_mask

    def shortnames(
        self,
        kind: List[Literal["input", "output", "input_output"]] = [
            "input",
            "output",
            "input_output",
        ],
    ) -> List[str]:
        """
        List of readable names for the parameters in the dataset.
        Does not include grid information (such as geopotentiel and LandSeaMask).
        Make the difference between inputs, outputs.
        """
        return [p.parameter_short_name for p in self.params if p.kind == kind]

    @cached_property
    def units(self) -> Dict[str, str]:
        """
        Return a dictionnary with name and units
        """
        return {p.parameter_short_name: p.unit for p in self.params}

    @cached_property
    def state_weights(self):
        """Weights used in the loss function."""
        kinds = ["output", "input_output"]
        return {
            p.parameter_short_name: p.state_weight
            for p in self.params
            if p.kind in kinds
        }

    @property
    def cache_dir(self) -> Path:
        return get_dataset_path(self.name, self.grid)

    @cached_property
    def domain_info(self) -> DomainInfo:
        """Information on the domain considered. Usefull information for plotting."""
        return DomainInfo(
            grid_limits=self.grid.grid_limits, projection=self.grid.projection
        )


@app.command()
def prepare(
    path_config: Path = DEFAULT_CONFIG,
    num_input_steps: int = 1,
    num_pred_steps_train: int = 1,
    num_pred_steps_val_test: int = 1,
    convert_grib2npy: bool = False,
    compute_stats: bool = True,
    write_valid_samples_list: bool = True,
):
    """Prepares Titan dataset for training.
    This command will:
        - create all needed folders
        - convert gribs to npy and rescale data to the wanted grid
        - establish a list of valid samples for each set
        - computes statistics on all weather parameters."""
    print("--> Preparing Titan Dataset...")

    print("Load dataset configuration...")
    with open(path_config, "r") as fp:
        conf = json.load(fp)

    print("Creating folders...")
    train_ds, valid_ds, test_ds = TitanDataset.from_dict(
        path_config.stem,
        conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
    train_ds.cache_dir.mkdir(exist_ok=True)
    data_dir = train_ds.cache_dir / "data"
    data_dir.mkdir(exist_ok=True)
    print(f"Dataset will be saved in {train_ds.cache_dir}")

    if convert_grib2npy:
        print("Converting gribs to npy...")
        param_list = get_param_list(
            conf, train_ds.grid, load_param_info, get_weight_per_lvl
        )
        sum_dates = (
            list(train_ds.period.date_list)
            + list(valid_ds.period.date_list)
            + list(test_ds.period.date_list)
        )
        dates = sorted(list(set(sum_dates)))
        for date in tqdm.tqdm(dates):
            process_sample_dataset(date, param_list)
        print("Done!")

    conf["settings"]["standardize"] = False
    train_ds, valid_ds, test_ds = TitanDataset.from_dict(
        path_config.stem,
        conf,
        num_input_steps,
        num_pred_steps_train,
        num_pred_steps_val_test,
    )
    if compute_stats:
        print("Computing stats on each parameter...")
        train_ds.compute_parameters_stats()
    if write_valid_samples_list:
        train_ds.write_list_valid_samples()
        valid_ds.write_list_valid_samples()
        test_ds.write_list_valid_samples()

    if compute_stats:
        print("Computing time stats on each parameters, between 2 timesteps...")
        conf["settings"]["standardize"] = True
        train_ds, valid_ds, test_ds = TitanDataset.from_dict(
            path_config.stem,
            conf,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
        )
        train_ds.compute_time_step_stats()


@app.command()
def describe(path_config: Path = DEFAULT_CONFIG):
    """Describes Titan."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    train_ds.dataset_info.summary()
    print("Len dataset : ", len(train_ds))
    print("First Item description :")
    print(train_ds[0])


@app.command()
def plot(path_config: Path = DEFAULT_CONFIG):
    """Plots a png and a gif for one sample."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    print("Plot gif of one sample...")
    sample = train_ds.sample_list[0]
    sample.plot_gif("test.gif")
    print("Plot png for one step of sample...")
    item = sample.load(no_standardize=True)
    sample.plot(item, 0, "test.png")


@app.command()
def speedtest(path_config: Path = DEFAULT_CONFIG, n_iter: int = 5):
    """Makes a loading speed test."""
    train_ds, _, _ = TitanDataset.from_json(path_config, 2, 1, 5)
    data_iter = iter(train_ds.torch_dataloader())
    print("Dataset file_format: ", train_ds.settings.file_format)
    print("Speed test:")
    start_time = time.time()
    for _ in tqdm.trange(n_iter, desc="Loading samples"):
        next(data_iter)
    delta = time.time() - start_time
    print("Elapsed time : ", delta)
    speed = n_iter / delta
    print(f"Loading speed: {round(speed, 3)} batch(s)/sec")


if __name__ == "__main__":
    app()
