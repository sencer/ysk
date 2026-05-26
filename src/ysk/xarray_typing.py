from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import xarray as xr

if TYPE_CHECKING:
  from pathlib import Path


class _AssignCoords(Protocol):
  def assign_coords(
    self,
    coords: object = None,
    **coords_kwargs: object,
  ) -> xr.Dataset: ...


class _DataArrayAstype(Protocol):
  def astype(self, dtype: object, **kwargs: object) -> xr.DataArray: ...


class _DataArrayIsnull(Protocol):
  def isnull(self) -> xr.DataArray: ...


class _DataArrayItem(Protocol):
  def item(self) -> object: ...


class _DataArrayCompute(Protocol):
  def compute(self) -> xr.DataArray: ...


class _DataArraySel(Protocol):
  def sel(self, **indexers: object) -> xr.DataArray: ...


class _DatasetSwapDims(Protocol):
  def swap_dims(
    self, dims_dict: object = None, **dims_kwargs: object
  ) -> xr.Dataset: ...


class _DatasetSetXindex(Protocol):
  def set_xindex(
    self,
    coord_names: object,
    index_cls: object = None,
    **options: object,
  ) -> xr.Dataset: ...


class _OpenZarr(Protocol):
  def __call__(
    self,
    store: object,
    *,
    group: str | None = None,
    consolidated: bool | None = None,
    chunks: object = None,
    create_default_indexes: bool = True,
  ) -> xr.Dataset: ...


class _ToNetcdf(Protocol):
  def to_netcdf(self, path: object) -> object: ...


class _ToZarr(Protocol):
  def to_zarr(
    self,
    store: object,
    *,
    mode: str | None = None,
    group: str | None = None,
    encoding: object = None,
    append_dim: str | None = None,
  ) -> object: ...


def assign_coords(
  dataset: xr.Dataset,
  coords: object = None,
  **coords_kwargs: object,
) -> xr.Dataset:
  """Assign coordinates to a dataset with typed wrapper support.

  Args:
    dataset: Dataset to update.
    coords: Positional coordinate mapping accepted by xarray.
    **coords_kwargs: Keyword coordinates accepted by xarray.

  Returns:
    Dataset with assigned coordinates.
  """

  return cast("_AssignCoords", dataset).assign_coords(coords, **coords_kwargs)


def astype_float32(array: xr.DataArray) -> xr.DataArray:
  """Cast a data array to float32.

  Args:
    array: Data array to cast.

  Returns:
    The cast data array.
  """

  return cast("_DataArrayAstype", array).astype("float32")


def data_array_isnull(array: xr.DataArray) -> xr.DataArray:
  """Return the null mask for a data array.

  Args:
    array: Data array to test.

  Returns:
    Boolean data array indicating null values.
  """

  return cast("_DataArrayIsnull", array).isnull()


def data_array_item(array: xr.DataArray) -> object:
  """Compute a scalar data array and return its Python value.

  Args:
    array: Scalar data array.

  Returns:
    The scalar Python value.
  """

  computed = cast("_DataArrayCompute", array).compute()
  return cast("_DataArrayItem", computed).item()


def open_zarr(path: Path | str, *, group: str | None = None) -> xr.Dataset:
  """Open a Zarr dataset with the package's default xarray options.

  Args:
    path: Zarr store path.
    group: Optional Zarr group name.

  Returns:
    The opened dataset.
  """

  opener = cast("_OpenZarr", xr.open_zarr)
  return opener(
    path,
    group=group,
    consolidated=False,
    chunks={},
    create_default_indexes=False,
  )


def sel_data_array(array: xr.DataArray, **indexers: object) -> xr.DataArray:
  """Select data array values with typed wrapper support.

  Args:
    array: Data array to select from.
    **indexers: xarray indexers.

  Returns:
    The selected data array.
  """

  return cast("_DataArraySel", array).sel(**indexers)


def swap_dims(dataset: xr.Dataset, dims: object) -> xr.Dataset:
  """Swap dataset dimensions with typed wrapper support.

  Args:
    dataset: Dataset to update.
    dims: Dimension mapping accepted by xarray.

  Returns:
    Dataset with swapped dimensions.
  """

  return cast("_DatasetSwapDims", dataset).swap_dims(dims)


def set_xindex(dataset: xr.Dataset, coord_names: object) -> xr.Dataset:
  """Set an xarray index with typed wrapper support.

  Args:
    dataset: Dataset to update.
    coord_names: Coordinate name or names accepted by xarray.

  Returns:
    Dataset with the requested xarray index.
  """

  return cast("_DatasetSetXindex", dataset).set_xindex(coord_names)


def to_netcdf(dataset: xr.Dataset, path: Path | str) -> None:
  """Write a dataset to NetCDF.

  Args:
    dataset: Dataset to write.
    path: Output NetCDF path.
  """

  cast("_ToNetcdf", dataset).to_netcdf(path)


def to_zarr(
  dataset: xr.Dataset,
  path: Path | str,
  *,
  mode: str | None = None,
  group: str | None = None,
  encoding: object = None,
  append_dim: str | None = None,
) -> None:
  """Write a dataset to Zarr.

  Args:
    dataset: Dataset to write.
    path: Output Zarr store path.
    mode: Optional Zarr write mode.
    group: Optional Zarr group name.
    encoding: Optional xarray encoding mapping.
    append_dim: Optional dimension to append along.
  """

  cast("_ToZarr", dataset).to_zarr(
    path,
    mode=mode,
    group=group,
    encoding=encoding,
    append_dim=append_dim,
  )
