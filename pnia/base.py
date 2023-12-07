"""
Base classes defining our software components
and their interfaces
"""

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractDataset(ABCMeta):

    @abstractproperty
    def grid_info(self) -> np.array:
        """
        array of shape (2, num_lat, num_lon)
        of (lat, lon) values
        """

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
    def border_mask(self) -> np.array:
        pass
    