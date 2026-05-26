from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag, StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
import warnings

import numpy as np
import pandas as pd
from turkiye import get_division
from zarr.errors import GroupNotFoundError

from ysk.schema import slugify
from ysk.xarray_typing import open_zarr

if TYPE_CHECKING:
  import xarray as xr

ResultLevel = Literal["satir", "sandik", "mahalle", "belde", "ilce", "il"]


class Secim(StrEnum):
  """Supported election dates for result selection."""

  YEREL_2009 = "2009-03-29"
  REFERANDUM_2010 = "2010-09-12"
  GENEL_2011 = "2011-06-12"
  YEREL_2014 = "2014-03-30"
  CUMHURBASKANLIGI_2014 = "2014-08-10"
  GENEL_HAZIRAN_2015 = "2015-06-07"
  GENEL_KASIM_2015 = "2015-11-01"
  REFERANDUM_2017 = "2017-04-16"
  GENEL_2018 = "2018-06-24"
  YEREL_2019 = "2019-03-31"
  ISTANBUL_YENILEME_2019 = "2019-06-23"
  GENEL_2023 = "2023-05-14"
  CUMHURBASKANLIGI_IKINCI_TUR_2023 = "2023-05-28"
  YEREL_2024 = "2024-03-31"
  YEREL_YENILEME_2024 = "2024-06-02"


SecimTarihi = Secim


class Makam(StrEnum):
  """Supported election offices or ballot types."""

  ALL = "ALL"
  BELEDIYE_BASKANI = "BB"
  BUYUKSEHIR_BELEDIYE_BASKANI = "BBB"
  BELEDIYE_MECLISI = "BM"
  IL_GENEL_MECLISI = "IGM"
  CUMHURBASKANI = "CB"
  MILLETVEKILI = "MV"
  REFERANDUM = "REF"


ElectionSelection = Secim | str
OfficeSelection = Makam | str
ScalarOrList = str | StrEnum | Sequence[str | StrEnum]


class Kolon(IntFlag):
  """Column groups returned by :func:`filter_columns`."""

  TOPLAM = auto()
  PARTI = auto()
  BAGIMSIZ = auto()


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_ZARR = PACKAGE_ROOT / "data" / "secim.zarr"
BASE_COLUMNS = [
  "secim",
  "makam",
  "il_id",
  "il",
  "ilce_id",
  "ilce",
  "mahalle",
  "sandik_no",
  "kayitli_secmen",
  "oy_kullanan",
  "gecerli_oy",
  "gecersiz_oy",
  "gecerli_itirazsiz",
  "gecerli_itirazli",
]
TOTAL_COLUMNS = [
  "kayitli_secmen",
  "oy_kullanan",
  "gecerli_oy",
  "gecersiz_oy",
  "gecerli_itirazsiz",
  "gecerli_itirazli",
]
PUBLIC_RENAMES = {
  "kayitli_secmen": "secmen",
  "gecerli_oy": "gecerli",
  "gecersiz_oy": "gecersiz",
}
PUBLIC_TOTAL_COLUMNS = [
  "secmen",
  "oy_kullanan",
  "gecerli",
  "gecersiz",
  "gecerli_itirazsiz",
  "gecerli_itirazli",
  "kadin",
  "erkek",
]
TOPLAM_KOLONLARI = frozenset(PUBLIC_TOTAL_COLUMNS)
DEFAULT_PARTILER = frozenset({
  "ak_parti",
  "btp",
  "buyuk_birlik",
  "chp",
  "dem_parti",
  "deva_partisi",
  "dp",
  "emep",
  "huda_par",
  "iyi_parti",
  "mhp",
  "milli_yol",
  "saadet",
  "tip",
  "tkp",
  "vatan_partisi",
  "yeniden_refah",
  "zafer_partisi",
})
DEFAULT_BAGIMSIZLAR = frozenset({"bagimsiz_toplam_oy"})
PARTILER = DEFAULT_PARTILER
BAGIMSIZLAR = DEFAULT_BAGIMSIZLAR
ILCE_ADI_DUZELTMELERI = {
  "ONDOKUZMAYIS": "19 MAYIS",
}
GROUP_KEYS: dict[str, list[str]] = {
  "mahalle": [
    "secim",
    "makam",
    "il_id",
    "il",
    "ilce_id",
    "ilce",
    "belde_adi",
    "mahalle",
  ],
  "belde": ["secim", "makam", "il_id", "il", "ilce_id", "ilce", "belde_adi"],
  "ilce": ["secim", "makam", "il_id", "il", "ilce_id", "ilce"],
  "il": ["secim", "makam", "il_id", "il"],
}


def load_dataset(path: Path | str | None = None) -> xr.Dataset:
  """Load the packaged or requested election result Zarr dataset.

  Args:
    path: Optional path to a Zarr dataset. When omitted, the packaged dataset
      is used if available, otherwise ``data/secim.zarr`` in the source tree.

  Returns:
    The opened election result dataset.
  """

  return open_zarr(_zarr_path(path))


def load_demographics(path: Path | str | None = None) -> xr.Dataset | None:
  """Load the demographics group when it is present in the Zarr dataset.

  Args:
    path: Optional path to a Zarr dataset. Uses the default dataset path when
      omitted.

  Returns:
    The demographics dataset, or ``None`` when the group is missing.
  """

  try:
    return open_zarr(_zarr_path(path), group="demografi")
  except (FileNotFoundError, GroupNotFoundError):
    return None


def election_results(  # noqa: PLR0913
  election: ElectionSelection | Sequence[ElectionSelection],
  office: OfficeSelection | Sequence[OfficeSelection],
  *,
  path: Path | str | None = None,
  dataset: xr.Dataset | None = None,
  level: ResultLevel = "sandik",
  province: str | None = None,
  district: str | None = None,
  choices: list[str] | None = None,
  columns: Sequence[str] | None = None,
  align_historical_divisions: bool = False,
  demographics: xr.Dataset | None = None,
) -> pd.DataFrame:
  """Return election results as a tidy DataFrame at the requested level.

  Multiple elections or offices can be selected at once. Set
  ``align_historical_divisions=True`` to project older district-level results
  onto the administrative divisions for the latest selected election year.

  Args:
    election: Election date, enum value, or sequence of election selections.
    office: Office, ballot type, enum value, or sequence of office selections.
    path: Optional path to a Zarr dataset.
    dataset: Pre-opened dataset to use instead of loading from ``path``.
    level: Aggregation level to return.
    province: Optional province filter.
    district: Optional district filter.
    choices: Optional choice names to include.
    columns: Optional output column names to include. Vote-choice columns are
      loaded selectively when possible.
    align_historical_divisions: Whether to align historical district results to
      the latest selected election year.
    demographics: Optional demographic dataset used for province or district
      aggregates.

  Returns:
    A result DataFrame indexed by the requested geography level.

  Raises:
    ValueError: If ``level`` is unknown.
  """

  dataset = load_dataset(path) if dataset is None else dataset
  dates = _as_list(election)
  offices = _office_list(office, dataset=dataset, dates=dates)
  choices = _normalize_choices(choices)
  columns = _normalize_columns(columns)
  if choices is None:
    choices = _choices_from_columns(columns)
  if demographics is None and level in {"ilce", "il"}:
    demographics = load_demographics(path)
  if len(dates) > 1 or len(offices) > 1:
    return _multi_election_results(
      dataset,
      dates=dates,
      offices=offices,
      level=level,
      province=province,
      district=district,
      choices=choices,
      columns=columns,
      align_historical_divisions=align_historical_divisions,
      demographics=demographics,
    )
  selected = dataset.sel(secim=dates[0]).sel(makam=offices[0])
  if province is not None:
    selected = _select_province(selected, province)
  if district is not None:
    selected = _select_district(selected, district)

  if level in {"satir", "sandik"}:
    if level == "sandik":
      selected = selected.isel(yer_dim=(selected.yer_turu != "ilce").to_numpy())
    frame = _wide_rows(
      selected,
      choices=choices,
      metadata_columns=_metadata_columns_for_level(level, columns=columns),
    )
    return _finalize_frame(
      _indexed(frame, level),
      dates=dates,
      offices=offices,
      columns=columns,
    )

  if level not in GROUP_KEYS:
    msg = f"unknown level {level!r}"
    raise ValueError(msg)
  sandik = selected.sel(yer_turu="sandik")
  frame = _wide_rows(
    sandik,
    choices=choices,
    metadata_columns=_metadata_columns_for_level(level, columns=columns),
  )
  aggregated = _aggregate(frame, GROUP_KEYS[level])
  aggregated = _attach_demographics(
    aggregated,
    selected=selected,
    demographics=demographics,
    dates=dates,
    level=level,
  )
  return _finalize_frame(
    aggregated,
    dates=dates,
    offices=offices,
    columns=columns,
  )


def _as_list(value: ScalarOrList) -> list[str]:
  if isinstance(value, str):
    return [str(value)]
  return [str(item) for item in value]


def _office_list(
  value: OfficeSelection | Sequence[OfficeSelection],
  *,
  dataset: xr.Dataset,
  dates: Sequence[str],
) -> list[str]:
  offices = _as_list(value)
  if not any(office.upper() == Makam.ALL for office in offices):
    return offices
  explicit = [office for office in offices if office.upper() != Makam.ALL]
  available = _available_offices(dataset, dates=dates)
  return list(dict.fromkeys([*explicit, *available]))


def _available_offices(dataset: xr.Dataset, *, dates: Sequence[str]) -> list[str]:
  offices: list[str] = []
  for date in dates:
    try:
      selected = dataset.sel(secim=date)
    except KeyError:
      continue
    if "makam" not in selected:
      continue
    offices.extend(str(office) for office in selected.makam.to_numpy())
  return sorted(dict.fromkeys(offices))


def _zarr_path(path: Path | str | None) -> Path:
  if path is not None:
    return Path(path)
  if DEFAULT_ZARR.exists():
    return DEFAULT_ZARR
  source_tree_zarr = PACKAGE_ROOT.parents[1] / "data" / "secim.zarr"
  if source_tree_zarr.exists():
    return source_tree_zarr
  msg = (
    "default dataset not found; reinstall a wheel that includes data/secim.zarr "
    "or pass path=... to load_dataset()/election_results()"
  )
  raise FileNotFoundError(msg)


def _normalize_choices(choices: list[str] | None) -> list[str] | None:
  if choices is None:
    return None
  return [slugify(choice) for choice in choices]


def _normalize_columns(columns: Sequence[str] | None) -> list[str] | None:
  if columns is None:
    return None
  return list(dict.fromkeys(slugify(column) for column in columns))


def _choices_from_columns(columns: Sequence[str] | None) -> list[str] | None:
  if columns is None:
    return None
  return [
    column
    for column in columns
    if _internal_column_name(column) not in set(BASE_COLUMNS) | {"belde_adi"}
    and column not in PUBLIC_TOTAL_COLUMNS
  ]


def _internal_column_name(column: str) -> str:
  public_to_internal = {public: internal for internal, public in PUBLIC_RENAMES.items()}
  return public_to_internal.get(column, column)


def _select_province(dataset: xr.Dataset, province: str) -> xr.Dataset:
  selected = dataset.sel(il=province)
  if selected.sizes.get("yer_dim", 0) > 0:
    return selected
  msg = f"unknown province {province!r}"
  raise ValueError(msg)


def _select_district(dataset: xr.Dataset, district: str) -> xr.Dataset:
  wanted = _normalize_district_name(district)
  names = pd.Series(dataset.ilce.to_numpy(), dtype="string").map(
    _normalize_district_name
  )
  mask = names.eq(wanted) | names.str.endswith(f" - {wanted}", na=False)
  return dataset.isel(yer_dim=mask.to_numpy())


def _multi_election_results(  # noqa: PLR0913
  dataset: xr.Dataset,
  *,
  dates: list[str],
  offices: list[str],
  level: ResultLevel,
  province: str | None,
  district: str | None,
  choices: list[str] | None,
  columns: list[str] | None,
  align_historical_divisions: bool,
  demographics: xr.Dataset | None,
) -> pd.DataFrame:
  frames: list[pd.DataFrame] = []
  keys: list[tuple[str, str]] = []
  alignment_year = _historical_alignment_year(dates)
  for date in dates:
    for office in offices:
      if not _has_election_office(dataset, date=date, office=office):
        warnings.warn(
          f"{date} secim doesn't have {office} makam; skipping.",
          stacklevel=2,
        )
        continue
      frame = election_results(
        date,
        office,
        dataset=dataset,
        level=level,
        province=province,
        district=district,
        choices=choices,
        columns=columns,
        demographics=demographics,
      )
      if align_historical_divisions:
        frame = _align_administrative_divisions(
          frame,
          level=level,
          target_year=alignment_year,
        )
      drop_columns = [
        column
        for column in ("secim", "makam")
        if column in frame.columns and not isinstance(frame.columns, pd.MultiIndex)
      ]
      if drop_columns:
        frame = frame.drop(columns=drop_columns)
      frames.append(frame)
      keys.append((date, office))
  if not frames:
    msg = "no available election/office combinations found"
    raise ValueError(msg)
  return (
    _squeeze_column_levels(
      pd.concat(
        frames,
        axis=1,
        keys=keys,
        names=["secim", "makam", "alan"],
      ),
      dates=dates,
      offices=offices,
    )
    .sort_index(axis=0)
    .sort_index(axis=1)
  )


def _has_election_office(dataset: xr.Dataset, *, date: str, office: str) -> bool:
  try:
    dataset.sel(secim=date).sel(makam=office)
  except KeyError:
    return False
  return True


def _historical_alignment_year(dates: Sequence[str]) -> int:
  return min(int(date[:4]) for date in dates)


def _align_administrative_divisions(
  frame: pd.DataFrame,
  *,
  level: ResultLevel,
  target_year: int,
) -> pd.DataFrame:
  if level != "ilce" or not isinstance(frame.index, pd.MultiIndex):
    return frame
  if not {"il", "ilce"}.issubset(frame.index.names):
    return frame
  reset = frame.reset_index()
  changed = False
  for index, row in reset[["il", "ilce"]].iterrows():
    division = get_division(il=row["il"], ilce=row["ilce"], year=target_year)
    if division.geometry_level != "ilce" or len(division.geometry_key) < 2:
      continue
    aligned_il, aligned_ilce = division.geometry_key[:2]
    if row["il"] != aligned_il or row["ilce"] != aligned_ilce:
      reset.loc[index, "il"] = aligned_il
      reset.loc[index, "ilce"] = aligned_ilce
      changed = True
  if not changed:
    return frame
  group_keys = list(frame.index.names)
  aggregations = {
    column: "sum" if pd.api.types.is_numeric_dtype(reset[column]) else "first"
    for column in reset.columns
    if column not in group_keys
  }
  return reset.groupby(group_keys, dropna=False).agg(aggregations).sort_index()


def _wide_rows(
  dataset: xr.Dataset,
  *,
  choices: list[str] | None,
  metadata_columns: Sequence[str],
) -> pd.DataFrame:
  oy = dataset.oy
  choice_names = dataset.tercih.to_numpy()
  if choices is not None:
    choice_mask = pd.Series(choice_names, dtype="string").isin(choices).to_numpy()
    choice_positions = np.flatnonzero(choice_mask)
    if len(choice_positions) != len(choices):
      available = set(pd.Series(choice_names, dtype="string").dropna().astype(str))
      missing = sorted(set(choices) - available)
      msg = f"unknown choices: {', '.join(missing)}"
      raise ValueError(msg)
    oy = oy.isel(tercih_dim=choice_positions)
    choice_names = choice_names[choice_positions]

  votes = oy.to_pandas()
  votes.columns = choice_names
  votes = votes.dropna(axis=1, how="all").fillna(0)

  meta = _metadata_frame(dataset, index=votes.index, columns=metadata_columns)
  frame = pd.concat([meta, votes], axis=1)
  return _clean_locations(frame)


def _metadata_columns_for_level(
  level: ResultLevel,
  *,
  columns: Sequence[str] | None,
) -> list[str]:
  if columns is None:
    return BASE_COLUMNS
  wanted = {_internal_column_name(column) for column in columns}
  if level in {"satir", "sandik"}:
    wanted.update(["il", "ilce", "belde_adi", "mahalle", "sandik_no"])
  elif level in GROUP_KEYS:
    wanted.update(GROUP_KEYS[level])
  wanted.update(column for column in TOTAL_COLUMNS if column in wanted)
  return [column for column in [*BASE_COLUMNS, "belde_adi"] if column in wanted]


def _metadata_frame(
  dataset: xr.Dataset,
  *,
  index: pd.Index,
  columns: Sequence[str],
) -> pd.DataFrame:
  data: dict[str, object] = {}
  for column in columns:
    if column in dataset:
      data[column] = dataset[column].to_numpy()
  return pd.DataFrame(data, index=index)


def _clean_locations(frame: pd.DataFrame) -> pd.DataFrame:
  cleaned = frame.copy()
  if "ilce" in cleaned:
    ilce = pd.Series(cleaned["ilce"], dtype="string", index=cleaned.index)
    ilce = ilce.map(_normalize_district_name)
    split = ilce.str.split(" - ", n=1, expand=True)
    if split.shape[1] == 2:
      belde = split[1].fillna("")
      cleaned["ilce"] = split[0].fillna(ilce).to_numpy(dtype=object)
      cleaned["belde_adi"] = belde.replace("", pd.NA).to_numpy(dtype=object)
    else:
      cleaned["belde_adi"] = pd.NA
  if "mahalle" in cleaned and "belde_adi" in cleaned:
    mahalle = pd.Series(cleaned["mahalle"], dtype="string", index=cleaned.index)
    belde = pd.Series(cleaned["belde_adi"], dtype="string", index=cleaned.index)
    prefix = belde.fillna("") + "-"
    has_prefix = belde.notna() & pd.Series(
      [
        value.startswith(start)
        for value, start in zip(
          mahalle.fillna("").astype(str),
          prefix.astype(str),
          strict=True,
        )
      ],
      index=cleaned.index,
    )
    cleaned.loc[has_prefix, "mahalle"] = [
      value[len(start) :]
      for value, start in zip(
        mahalle[has_prefix].astype(str),
        prefix[has_prefix].astype(str),
        strict=True,
      )
    ]
  if "mahalle" in cleaned:
    cleaned["mahalle"] = (
      pd
      .Series(cleaned["mahalle"], dtype="string", index=cleaned.index)
      .map(_normalize_neighborhood_name)
      .to_numpy(dtype=object)
    )
  return cleaned


def _normalize_district_name(value: object) -> str:
  if value is None:
    return ""
  text = str(value).strip()
  for source, target in ILCE_ADI_DUZELTMELERI.items():
    if text == source:
      return target
    if text.startswith(f"{source} - "):
      return text.replace(source, target, 1)
  return text


def _normalize_neighborhood_name(value: object) -> str:
  if value is None:
    return ""
  text = str(value).strip()
  text = text.replace(".", ". ")
  return " ".join(text.split())


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
  if keys == GROUP_KEYS["ilce"] and "belde_adi" in frame.columns:
    frame = frame[pd.Series(frame["belde_adi"], dtype="string").isna()]
  available_keys = [key for key in keys if key in frame.columns]
  value_columns = [
    column
    for column in frame.columns
    if column not in available_keys
    and column not in set(BASE_COLUMNS) - set(TOTAL_COLUMNS)
  ]
  aggregations = {
    column: "sum" for column in value_columns if column not in available_keys
  }
  return (
    frame[available_keys + value_columns]
    .groupby(
      available_keys,
      dropna=False,
    )
    .agg(aggregations)
    .sort_index()
  )


def _attach_demographics(
  frame: pd.DataFrame,
  *,
  selected: xr.Dataset,
  demographics: xr.Dataset | None,
  dates: Sequence[str],
  level: ResultLevel,
) -> pd.DataFrame:
  if level not in {"ilce", "il"}:
    return frame
  demo = _demographics_frame(demographics, selected, dates=dates)
  if demo.empty:
    return frame
  reset = frame.reset_index()
  if level == "ilce":
    keys = [
      key for key in ("secim", "il_id", "ilce_id") if key in reset and key in demo
    ]
    wanted = demo[[*keys, "kadin", "erkek"]].drop_duplicates(keys)
  else:
    keys = [key for key in ("secim", "il_id") if key in reset and key in demo]
    wanted = (
      demo
      .groupby(keys, dropna=False)[["kadin", "erkek"]]
      .sum(min_count=1)
      .reset_index()
    )
  if not keys:
    return frame
  merged = reset.merge(wanted, on=keys, how="left")
  return merged.set_index(list(frame.index.names))


def _demographics_frame(
  demographics: xr.Dataset | None,
  selected: xr.Dataset,
  *,
  dates: Sequence[str],
) -> pd.DataFrame:
  if demographics is not None:
    return _demographics_frame_from_group(demographics, dates=dates)
  if "secmen" not in selected:
    return pd.DataFrame()
  return _demographics_frame_from_embedded(selected)


def _demographics_frame_from_group(
  demographics: xr.Dataset,
  *,
  dates: Sequence[str],
) -> pd.DataFrame:
  wanted_vars = [name for name in ("kadin", "erkek", "toplam") if name in demographics]
  if not wanted_vars:
    return pd.DataFrame()
  selected = demographics
  mask: xr.DataArray | None = None
  if "kapsam" in selected:
    mask = selected["kapsam"] == "cinsiyet_ilce"
  if "secim" in selected:
    election_mask = selected["secim"].isin(list(dates))
    mask = election_mask if mask is None else mask & election_mask
  if mask is not None:
    selected = selected.isel(demografi=mask.to_numpy())
  columns = [
    column
    for column in ("secim", "il_id", "il", "ilce_id", "ilce")
    if column in selected
  ]
  return (
    selected[columns + wanted_vars]
    .to_dataframe()
    .reset_index(drop=True)
    .drop_duplicates()
  )


def _demographics_frame_from_embedded(selected: xr.Dataset) -> pd.DataFrame:
  if "yer_turu" in selected:
    selected = selected.isel(yer_dim=(selected.yer_turu == "ilce").to_numpy())
  columns = [
    column
    for column in ("secim", "il_id", "il", "ilce_id", "ilce")
    if column in selected
  ]
  frame = selected[columns].to_dataframe().reset_index()
  secmen = selected.secmen.to_pandas()
  secmen.columns = selected.grup.to_numpy()
  frame = pd.concat(
    [frame, secmen[["kadin", "erkek", "toplam"]].reset_index(drop=True)],
    axis=1,
  )
  return frame.drop(columns=["yer_dim"], errors="ignore").drop_duplicates(
    ["secim", "il_id", "ilce_id"],
  )


def _indexed(frame: pd.DataFrame, level: str) -> pd.DataFrame:
  if level == "satir":
    return frame
  keys = ["il", "ilce", "belde_adi", "mahalle", "sandik_no"]
  index_keys = [
    key
    for key in keys
    if key in frame.columns and not pd.Series(frame[key], dtype="string").isna().all()
  ]
  result = frame.set_index(index_keys).sort_index()
  if "belde_adi" in result.index.names:
    result.index = result.index.set_names(
      ["belde" if name == "belde_adi" else name for name in result.index.names],
    )
  return result


def _finalize_frame(
  frame: pd.DataFrame,
  *,
  dates: list[str],
  offices: list[str],
  columns: Sequence[str] | None = None,
) -> pd.DataFrame:
  result = frame.rename(columns=PUBLIC_RENAMES)
  drop_columns = [
    column
    for column in ("secim", "makam", "belde_adi", "il_id", "ilce_id")
    if column in result
  ]
  if len(dates) > 1 and "secim" in drop_columns:
    drop_columns.remove("secim")
  if len(offices) > 1 and "makam" in drop_columns:
    drop_columns.remove("makam")
  if drop_columns:
    result = result.drop(columns=drop_columns)
  result = result[_sorted_columns(result)]
  if columns is not None:
    result = result.loc[:, [column for column in result.columns if column in columns]]
  result = _clean_index(result, dates=dates, offices=offices)
  return result.sort_index(axis=0)


def _sorted_columns(frame: pd.DataFrame) -> list[str]:
  totals = [column for column in PUBLIC_TOTAL_COLUMNS if column in frame.columns]
  rest = sorted(column for column in frame.columns if column not in totals)
  return totals + rest


def filter_columns(
  data: pd.DataFrame,
  *,
  types: Kolon = Kolon.TOPLAM | Kolon.PARTI | Kolon.BAGIMSIZ,
) -> pd.DataFrame:
  """Return the total, party, and/or independent columns in ``data``.

  Args:
    data: Result DataFrame to filter.
    types: Column group flags to keep.

  Returns:
    A DataFrame containing only the selected column groups.
  """

  wanted = _selected_column_names(types)
  columns = [column for column in data.columns if _column_name(column) in wanted]
  return data.loc[:, columns]


def _selected_column_names(types: Kolon) -> frozenset[str]:
  selected: set[str] = set()
  if types & Kolon.TOPLAM:
    selected.update(TOPLAM_KOLONLARI)
  if types & Kolon.PARTI:
    selected.update(PARTILER)
  if types & Kolon.BAGIMSIZ:
    selected.update(BAGIMSIZLAR)
  return frozenset(selected)


def _column_name(column: object) -> str:
  if isinstance(column, tuple):
    tuple_column = cast("tuple[object, ...]", column)
    return str(tuple_column[-1])
  return str(column)


def _squeeze_column_levels(
  frame: pd.DataFrame,
  *,
  dates: list[str],
  offices: list[str],
) -> pd.DataFrame:
  if not isinstance(frame.columns, pd.MultiIndex):
    return frame
  result = frame
  if len(dates) == 1:
    result = result.droplevel("secim", axis=1)
  if len(offices) == 1:
    result = result.droplevel("makam", axis=1)
  return result


def _clean_index(
  frame: pd.DataFrame,
  *,
  dates: list[str],
  offices: list[str],
) -> pd.DataFrame:
  result = frame
  if isinstance(result.index, pd.MultiIndex):
    names = ["belde" if name == "belde_adi" else name for name in result.index.names]
    result.index = result.index.set_names(names)
    drop_levels: list[str] = []
    if len(dates) == 1 and "secim" in result.index.names:
      drop_levels.append("secim")
    if len(offices) == 1 and "makam" in result.index.names:
      drop_levels.append("makam")
    drop_levels.extend(
      level for level in ("il_id", "ilce_id") if level in result.index.names
    )
    for level in drop_levels:
      result.index = result.index.droplevel(level)
  elif result.index.name == "belde_adi":
    result.index = result.index.rename("belde")
  return result
