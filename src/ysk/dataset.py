from __future__ import annotations

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

from ysk.schema import ID_COORDS, LABEL_COORDS, MEASURE_COLUMNS
from ysk.tables import TABLE_NAMES, NormalizedTables
from ysk.xarray_typing import (
  assign_coords,
  astype_float32,
  open_zarr,
  sel_data_array,
  swap_dims,
  to_netcdf,
  to_zarr,
)

if TYPE_CHECKING:
  from collections.abc import Callable, Iterable
  from typing import Protocol

  class _ParquetSchema(Protocol):
    @property
    def names(self) -> Iterable[str]: ...


INDEX_COLUMNS = ["election", "election_id", "election_kind", "scope"]
TURKISH_SCHEMA_RENAMES = {
  "observation": "gozlem",
  "choice": "tercih",
  "election": "secim",
  "election_id": "secim_id",
  "election_kind": "secim_turu",
  "scope": "kapsam",
  "province": "il",
  "province_id": "il_id",
  "district": "ilce",
  "district_id": "ilce_id",
  "neighborhood": "mahalle",
  "neighborhood_id": "mahalle_id",
  "country": "ulke",
  "country_id": "ulke_id",
  "foreign_office": "temsilcilik",
  "foreign_office_id": "temsilcilik_id",
  "customs": "gumruk",
  "customs_id": "gumruk_id",
  "ballot_box_no": "sandik_no",
  "ballot_box_id": "sandik_id",
  "ballot_box_code": "sandik_kodu",
  "ballot_result_id": "sandik_sonuc_id",
  "last_updated_at": "son_guncelleme",
  "votes": "oy",
  "registered_voters": "kayitli_secmen",
  "voters": "oy_kullanan",
  "valid_votes": "gecerli_oy",
  "invalid_votes": "gecersiz_oy",
  "valid_uncontested": "gecerli_itirazsiz",
  "valid_contested": "gecerli_itirazli",
  "choice_column": "tercih_kolonu",
  "choice_type": "tercih_turu",
  "choice_label_raw": "tercih_adi",
}
TURKISH_VALUE_RENAMES = {
  "scope": {
    "customs": "gumruk",
    "domestic_ballot_box": "yurtici_sandik",
    "domestic_district": "yurtici_ilce",
    "foreign_country": "yurtdisi_ulke",
  },
}
TURKISH_DEMOGRAPHIC_RENAMES = {
  "election": "secim",
  "election_id": "secim_id",
  "election_kind": "secim_turu",
  "endpoint": "kaynak",
  "scope": "kapsam",
  "province": "il",
  "province_id": "il_id",
  "district": "ilce",
  "district_id": "ilce_id",
  "yas_GRUBU": "yas_grubu",
  "kadin": "kadin",
  "erkek": "erkek",
  "toplam": "toplam",
  "kadin_ORAN": "kadin_oran",
  "erkek_ORAN": "erkek_oran",
  "secmen_SAYISI": "secmen_sayisi",
  "sandik_SAYISI": "sandik_sayisi",
  "seyyar_SANDIK_SAYISI": "seyyar_sandik_sayisi",
  "toplam_SANDIK_SAYISI": "toplam_sandik_sayisi",
  "cezaevi_SANDIK_SAYISI": "cezaevi_sandik_sayisi",
  "ilce_KODU": "ilce_kodu",
}
TURKISH_DEMOGRAPHIC_VALUE_RENAMES = {
  "scope": {
    "domestic_voters_district": "yurtici_secmen_ilce",
    "domestic_voters_province": "yurtici_secmen_il",
    "domestic_voters_total": "yurtici_secmen_toplam",
    "foreign_age_gender": "yurtdisi_yas_cinsiyet",
    "foreign_voters": "yurtdisi_secmen",
    "gender_district": "cinsiyet_ilce",
    "gender_province": "cinsiyet_il",
    "gender_total": "cinsiyet_toplam",
    "participation_age_gender": "katilim_yas_cinsiyet",
  },
}
MAKAM_RENAMES = {
  "MAYOR": "BB",
  "BB": "BB",
  "METROPOLITAN_MAYOR": "BBB",
  "METROPOLITAN-MAYOR": "BBB",
  "MUNICIPAL_COUNCIL": "BM",
  "MUNICIPAL-COUNCIL": "BM",
  "PROVINCIAL_COUNCIL": "IGM",
  "PROVINCIAL-COUNCIL": "IGM",
  "CB": "CB",
  "MV": "MV",
  "REF": "REF",
}
MAKAM_ADLARI = {
  "BB": "belediye_baskani",
  "BBB": "buyuksehir_belediye_baskani",
  "BM": "belediye_meclisi",
  "IGM": "il_genel_meclisi",
  "CB": "cumhurbaskani",
  "MV": "milletvekili",
  "REF": "referandum",
}
YER_TURU_RENAMES = {
  "gumruk": "gumruk",
  "yurtici_ilce": "ilce",
  "yurtici_sandik": "sandik",
  "yurtdisi_ulke": "ulke",
}
KAPSAM_RENAMES = {
  "gumruk": "gumruk",
  "yurtici_ilce": "yurtici",
  "yurtici_sandik": "yurtici",
  "yurtdisi_ulke": "yurtdisi",
}
LEGACY_META_COLUMNS = {
  "election",
  "scope",
  "il",
  "ilce",
  "muhtarlik",
  "sandik",
  "sandik_dokum_cetveli",
  "denetim_tutanagi",
  "toplam",
  "secmen",
  "gecerli_itirazsiz",
  "gecerli_itirazli",
  "gecerli",
  "gecersiz",
}
LEGACY_RENAMES = {
  "il": "province",
  "ilce": "district",
  "muhtarlik": "neighborhood",
  "sandik": "ballot_box_no",
  "toplam": "voters",
  "secmen": "registered_voters",
  "gecerli_itirazsiz": "valid_uncontested",
  "gecerli_itirazli": "valid_contested",
  "gecerli": "valid_votes",
  "gecersiz": "invalid_votes",
}


def parquet_columns(path: Path) -> set[str]:
  """Return the column names stored in a Parquet file.

  Args:
    path: Parquet file path.

  Returns:
    The set of column names in the file schema.
  """

  read_schema = cast("Callable[[Path], _ParquetSchema]", pq.read_schema)
  schema = read_schema(path)
  return set(schema.names)


def build_dataset(frame: pd.DataFrame) -> xr.Dataset:
  """Build an xarray result dataset from a normalized long result frame.

  Args:
    frame: Normalized long result frame with optional ``choice`` and ``votes``
      columns.

  Returns:
    An xarray dataset with observation coordinates and vote variables.
  """

  if frame.empty:
    return xr.Dataset()

  data = frame.reset_index(drop=True).copy()
  value_cols = {"choice", "votes"}
  obs_cols = [col for col in data.columns if col not in value_cols]
  obs_key = pd.MultiIndex.from_frame(data[obs_cols], names=obs_cols)
  data["observation"] = pd.factorize(obs_key, sort=False)[0].astype(np.int64)
  observations = data.drop_duplicates("observation").sort_values("observation")

  coords: dict[str, tuple[str, np.ndarray]] = {
    "observation": ("observation", observations["observation"].to_numpy()),
  }
  coord_cols = INDEX_COLUMNS + sorted((ID_COORDS | LABEL_COORDS) & set(data.columns))
  for col in coord_cols:
    if col in observations.columns:
      coords[col] = ("observation", observations[col].to_numpy())

  dataset = xr.Dataset(coords=coords)

  measures = sorted(MEASURE_COLUMNS & set(observations.columns))
  for measure in measures:
    dataset[measure] = (
      "observation",
      pd.to_numeric(observations[measure], errors="coerce"),
    )

  if "choice" in data.columns and "votes" in data.columns:
    choices = pd.Index(sorted(data["choice"].dropna().unique()), name="choice")
    pivot = (
      data
      .pivot_table(
        index="observation",
        columns="choice",
        values="votes",
        aggfunc="sum",
      )
      .reindex(index=observations["observation"], columns=choices)
      .astype("float64")
    )
    dataset = assign_coords(dataset, choice=choices.to_numpy())
    dataset["votes"] = (
      ("observation", "choice"),
      pivot.to_numpy(),
    )

  return dataset


def to_turkish_schema(dataset: xr.Dataset) -> xr.Dataset:
  """Rename dataset dimensions, variables, and known values to Turkish names.

  Args:
    dataset: Dataset using the normalized English schema.

  Returns:
    A dataset renamed to the public Turkish schema.
  """

  for name, values in TURKISH_VALUE_RENAMES.items():
    if name in dataset.variables:
      replaced = dataset[name].to_series().replace(values)
      dataset = dataset.assign(
        {
          name: (
            dataset[name].dims,
            replaced.to_numpy(),
          ),
        },
      )
  renames = {
    source: target
    for source, target in TURKISH_SCHEMA_RENAMES.items()
    if source in dataset.dims or source in dataset.variables
  }
  return dataset.rename(renames)


def build_demographic_dataset(tables_dir: Path | str) -> xr.Dataset:
  """Build the demographic xarray group from partitioned normalized tables.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.

  Returns:
    A demographic xarray dataset, or an empty dataset when no demographics are
    present.
  """

  frames = [
    pd.read_parquet(path)
    for path in sorted(Path(tables_dir).glob("election=*/demographics.parquet"))
  ]
  frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
  if frame.empty:
    return xr.Dataset()

  for name, values in TURKISH_DEMOGRAPHIC_VALUE_RENAMES.items():
    if name in frame.columns:
      frame[name] = frame[name].replace(values)
  frame = frame.rename(columns=TURKISH_DEMOGRAPHIC_RENAMES)
  frame.insert(0, "demografi", np.arange(len(frame), dtype=np.int64))

  coord_cols = [
    col
    for col in (
      "demografi",
      "secim",
      "secim_id",
      "secim_turu",
      "kapsam",
      "kaynak",
      "il",
      "il_id",
      "ilce",
      "ilce_id",
      "ilce_kodu",
      "yas_grubu",
    )
    if col in frame.columns
  ]
  coords: dict[str, tuple[str, np.ndarray]] = {
    col: ("demografi", frame[col].to_numpy()) for col in coord_cols
  }
  dataset = xr.Dataset(coords=coords)

  for col in (
    "kadin",
    "erkek",
    "toplam",
    "kadin_oran",
    "erkek_oran",
    "secmen_sayisi",
    "sandik_sayisi",
    "seyyar_sandik_sayisi",
    "toplam_sandik_sayisi",
    "cezaevi_sandik_sayisi",
  ):
    if col in frame.columns:
      dataset[col] = (
        "demografi",
        pd.to_numeric(frame[col], errors="coerce").to_numpy(),
      )
  return dataset


def write_demographic_group(tables_dir: Path | str, path: Path | str) -> None:
  """Write the demographic dataset group into a Zarr store.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.
    path: Zarr store path to update.
  """

  dataset = build_demographic_dataset(tables_dir)
  if not dataset.sizes and not dataset.data_vars:
    return
  to_zarr(
    dataset,
    Path(path),
    mode="a",
    group="demografi",
  )


def _district_votes_for_partition(partition: Path) -> pd.DataFrame:
  observation_path = partition / "observations.parquet"
  votes_path = partition / "votes.parquet"
  observations = pd.read_parquet(
    observation_path,
    columns=[
      "observation_id",
      "election",
      "election_id",
      "election_kind",
      "scope",
      "province_id",
      "province",
      "district_id",
      "district",
    ],
  )
  observations = observations[
    observations["scope"].eq("domestic_district") & observations["district_id"].notna()
  ]
  votes = pd.read_parquet(
    votes_path,
    columns=["observation_id", "choice", "votes"],
    filters=[("scope", "==", "domestic_district")],
  )
  merged = votes.merge(observations, on="observation_id", how="inner")
  return (
    merged
    .groupby(
      [
        "election",
        "election_id",
        "election_kind",
        "province_id",
        "province",
        "district_id",
        "district",
        "choice",
      ],
      dropna=False,
    )["votes"]
    .sum()
    .reset_index()
  )


def _district_demographics_for_partition(partition: Path) -> pd.DataFrame:
  demographic_path = partition / "demographics.parquet"
  frame = pd.read_parquet(demographic_path)
  gender = frame[frame["scope"].eq("gender_district")].copy()
  if gender.empty:
    return pd.DataFrame()
  gender = gender.rename(columns=TURKISH_DEMOGRAPHIC_RENAMES)
  return gender[
    [
      "secim",
      "ilce_id",
      "kadin",
      "erkek",
      "toplam",
      "kadin_oran",
      "erkek_oran",
    ]
  ].drop_duplicates()


def build_district_analysis_dataset(tables_dir: Path | str) -> xr.Dataset:
  """Build an election-by-district analysis dataset from partitioned tables.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.

  Returns:
    An xarray dataset indexed by election, district, and choice.
  """

  partitions = partition_dirs(tables_dir)
  vote_frames = [_district_votes_for_partition(partition) for partition in partitions]
  vote_frame = pd.concat(vote_frames, ignore_index=True)
  vote_frame = vote_frame.rename(columns=TURKISH_SCHEMA_RENAMES)
  demo_frames = [
    _district_demographics_for_partition(partition) for partition in partitions
  ]
  demo_frame = (
    pd.concat([frame for frame in demo_frames if not frame.empty], ignore_index=True)
    if demo_frames
    else pd.DataFrame()
  )

  secimler = pd.Index(
    [partition.name.removeprefix("election=") for partition in partitions],
    name="secim",
  )
  tercihler = pd.Index(sorted(vote_frame["tercih"].dropna().unique()), name="tercih")
  ilceler = pd.Index(
    sorted(vote_frame["ilce_id"].dropna().astype("int64").unique()),
    name="ilce_id",
  )
  district_index = pd.MultiIndex.from_product(
    [secimler, ilceler],
    names=["secim", "ilce_id"],
  )

  vote_pivot = (
    vote_frame
    .assign(ilce_id=vote_frame["ilce_id"].astype("int64"))
    .pivot_table(
      index=["secim", "ilce_id"],
      columns="tercih",
      values="oy",
      aggfunc="sum",
    )
    .reindex(index=district_index, columns=tercihler)
    .astype("float32")
  )

  metadata = (
    vote_frame
    .assign(ilce_id=vote_frame["ilce_id"].astype("int64"))
    .drop_duplicates(["secim", "ilce_id"])
    .set_index(["secim", "ilce_id"])
    .reindex(district_index)
  )
  secim_metadata = (
    vote_frame.drop_duplicates("secim").set_index("secim").reindex(secimler)
  )

  dataset = xr.Dataset(
    coords={
      "secim": ("secim", secimler.to_numpy()),
      "ilce_id": ("ilce_id", ilceler.to_numpy()),
      "tercih": ("tercih", tercihler.to_numpy()),
      "secim_id": ("secim", secim_metadata["secim_id"].to_numpy()),
      "secim_turu": ("secim", secim_metadata["secim_turu"].to_numpy()),
      "il": (
        ("secim", "ilce_id"),
        metadata["il"].to_numpy().reshape(len(secimler), len(ilceler)),
      ),
      "sehir": (
        ("secim", "ilce_id"),
        metadata["il"].to_numpy().reshape(len(secimler), len(ilceler)),
      ),
      "il_id": (
        ("secim", "ilce_id"),
        metadata["il_id"].to_numpy().reshape(len(secimler), len(ilceler)),
      ),
      "ilce": (
        ("secim", "ilce_id"),
        metadata["ilce"].to_numpy().reshape(len(secimler), len(ilceler)),
      ),
    },
    data_vars={
      "oy": (
        ("secim", "ilce_id", "tercih"),
        vote_pivot.to_numpy().reshape(len(secimler), len(ilceler), len(tercihler)),
      ),
    },
  )

  if not demo_frame.empty:
    demo_indexed = (
      demo_frame
      .assign(ilce_id=demo_frame["ilce_id"].astype("int64"))
      .drop_duplicates(["secim", "ilce_id"])
      .set_index(["secim", "ilce_id"])
      .reindex(district_index)
    )
    for name in ("kadin", "erkek", "toplam", "kadin_oran", "erkek_oran"):
      dataset[name] = (
        ("secim", "ilce_id"),
        pd
        .to_numeric(demo_indexed[name], errors="coerce")
        .to_numpy()
        .reshape(len(secimler), len(ilceler)),
      )
  return dataset


def write_district_analysis_dataset(
  tables_dir: Path | str,
  path: Path | str,
) -> None:
  """Write the district analysis dataset to a Zarr store.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.
    path: Output Zarr store path.
  """

  dataset = build_district_analysis_dataset(tables_dir)
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  to_zarr(
    dataset,
    output,
    mode="w",
    encoding={
      "oy": {
        "chunks": (
          1,
          min(dataset.sizes["ilce_id"], 1000),
          dataset.sizes["tercih"],
        ),
      },
    },
  )


def _global_place_index(partitions: Iterable[Path]) -> pd.DataFrame:
  frames: list[pd.DataFrame] = []
  for partition in partitions:
    frame = pd.read_parquet(
      partition / "observations.parquet",
      columns=[
        "scope",
        "province_id",
        "province",
        "district_id",
        "district",
        "ballot_box_no",
      ],
    )
    frame = frame[
      frame["scope"].eq("domestic_ballot_box")
      & frame["province_id"].notna()
      & frame["district_id"].notna()
      & frame["ballot_box_no"].notna()
    ]
    frames.append(
      frame[
        ["province_id", "province", "district_id", "district", "ballot_box_no"]
      ].drop_duplicates(),
    )
  places = pd.concat(frames, ignore_index=True).drop_duplicates()
  places = places.sort_values(
    ["province_id", "district_id", "ballot_box_no"],
    kind="stable",
  ).reset_index(drop=True)
  places.insert(0, "yer", np.arange(len(places), dtype=np.int64))
  return places


def _election_place_dataset(  # noqa: PLR0914
  partition: Path,
  *,
  places: pd.DataFrame,
  choices: pd.Index,
) -> xr.Dataset:
  base_columns = {
    "observation_id",
    "election_id",
    "election_kind",
    "scope",
    "province_id",
    "district_id",
    "ballot_box_no",
  }
  measure_sources = {
    "registered_voters": "kayitli_secmen",
    "voters": "oy_kullanan",
    "valid_votes": "gecerli_oy",
    "invalid_votes": "gecersiz_oy",
    "valid_uncontested": "gecerli_itirazsiz",
    "valid_contested": "gecerli_itirazli",
  }
  available_columns = parquet_columns(partition / "observations.parquet")
  observations = pd.read_parquet(
    partition / "observations.parquet",
    columns=sorted(base_columns | (set(measure_sources) & available_columns)),
    filters=[("scope", "==", "domestic_ballot_box")],
  )
  votes_path = partition / "votes.parquet"
  vote_columns = parquet_columns(votes_path)
  vote_filters = (
    [("scope", "==", "domestic_ballot_box")] if "scope" in vote_columns else None
  )
  votes = pd.read_parquet(
    votes_path,
    columns=["observation_id", "choice", "votes"],
    filters=vote_filters,
  )
  ballot_observations = observations[
    observations["scope"].eq("domestic_ballot_box")
    & observations["province_id"].notna()
    & observations["district_id"].notna()
    & observations["ballot_box_no"].notna()
  ].copy()
  ballot_observations = ballot_observations.merge(
    places[
      [
        "yer",
        "province_id",
        "district_id",
        "ballot_box_no",
      ]
    ],
    on=["province_id", "district_id", "ballot_box_no"],
    how="left",
  )

  vote_frame = votes.merge(
    ballot_observations[["observation_id", "yer"]],
    on="observation_id",
    how="inner",
  )
  vote_pivot = (
    vote_frame
    .pivot_table(
      index="yer",
      columns="choice",
      values="votes",
      aggfunc="sum",
    )
    .reindex(index=places["yer"], columns=choices)
    .astype("float32")
  )

  measure_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
  for source, target in measure_sources.items():
    if source in ballot_observations.columns:
      values = (
        ballot_observations
        .groupby("yer", dropna=False)[source]
        .sum()
        .reindex(places["yer"])
        .to_numpy()
        .reshape(1, len(places))
      )
      measure_vars[target] = (("secim", "yer"), values)

  demographics = pd.read_parquet(partition / "demographics.parquet")
  gender = demographics[demographics["scope"].eq("gender_district")].copy()
  if not gender.empty:
    gender = gender.rename(columns=TURKISH_DEMOGRAPHIC_RENAMES)
    district_gender = gender.drop_duplicates("ilce_id").set_index("ilce_id")
    place_districts = places["district_id"].astype("float64")
    for name in ("kadin", "erkek", "toplam", "kadin_oran", "erkek_oran"):
      if name in district_gender.columns:
        values = (
          district_gender
          .reindex(place_districts)[name]
          .to_numpy()
          .reshape(1, len(places))
        )
        measure_vars[name] = (("secim", "yer"), values)

  election_slug = partition.name.removeprefix("election=")
  election_meta = observations.iloc[0]
  return xr.Dataset(
    coords={
      "secim": ("secim", np.array([election_slug], dtype=object)),
      "secim_id": ("secim", np.array([election_meta["election_id"]])),
      "secim_turu": ("secim", np.array([election_meta["election_kind"]], dtype=object)),
      "yer": ("yer", places["yer"].to_numpy()),
      "tercih": ("tercih", choices.to_numpy()),
      "il_id": ("yer", places["province_id"].to_numpy()),
      "il": ("yer", places["province"].to_numpy()),
      "sehir": ("yer", places["province"].to_numpy()),
      "ilce_id": ("yer", places["district_id"].to_numpy()),
      "ilce": ("yer", places["district"].to_numpy()),
      "sandik_no": ("yer", places["ballot_box_no"].to_numpy()),
    },
    data_vars={
      "oy": (
        ("secim", "yer", "tercih"),
        vote_pivot.to_numpy().reshape(1, len(places), len(choices)),
      ),
    }
    | measure_vars,
  )


def build_place_analysis_dataset(tables_dir: Path | str) -> xr.Dataset:
  """Build an election-by-ballot-box analysis dataset from partitioned tables.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.

  Returns:
    An xarray dataset indexed by election, ballot-box place, and choice.
  """

  partitions = partition_dirs(tables_dir)
  places = _global_place_index(partitions)
  choices = global_choices_from_partitions(partitions)
  datasets = [
    _election_place_dataset(partition, places=places, choices=choices)
    for partition in partitions
  ]
  return xr.concat(datasets, dim="secim")


def _place_zarr_encoding(dataset: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
  yer_chunk = min(dataset.sizes["yer"], 10_000)
  tercih_count = dataset.sizes["tercih"]
  encoding: dict[str, dict[str, tuple[int, ...]]] = {
    "oy": {"chunks": (1, yer_chunk, tercih_count)},
  }
  for name, variable in dataset.data_vars.items():
    if not isinstance(name, str):
      continue
    if name != "oy" and variable.dims == ("secim", "yer"):
      encoding[name] = {"chunks": (1, yer_chunk)}
  return encoding


def write_place_analysis_dataset(tables_dir: Path | str, path: Path | str) -> None:
  """Write the ballot-box analysis dataset to a Zarr store.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.
    path: Output Zarr store path.
  """

  partitions = partition_dirs(tables_dir)
  places = _global_place_index(partitions)
  choices = global_choices_from_partitions(partitions)
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  for index, partition in enumerate(partitions):
    dataset = _election_place_dataset(partition, places=places, choices=choices)
    if index == 0:
      to_zarr(
        dataset,
        output,
        mode="w",
        encoding=_place_zarr_encoding(dataset),
      )
    else:
      to_zarr(
        dataset,
        output,
        mode="a",
        append_dim="secim",
      )


def dataset_with_broadcast_demographics(
  sonuc: xr.Dataset,
  demografi: xr.Dataset,
) -> xr.Dataset:
  """Attach district demographic variables to each result observation.

  Args:
    sonuc: Election result dataset.
    demografi: Demographic dataset.

  Returns:
    ``sonuc`` with district demographic variables broadcast to observations.
  """

  gender = demografi.where(demografi["kapsam"] == "cinsiyet_ilce", drop=True)
  gender_frame = (
    gender[["kadin", "erkek", "toplam", "kadin_oran", "erkek_oran"]]
    .to_dataframe()
    .reset_index()
    .drop_duplicates(["secim", "ilce_id"])
    .set_index(["secim", "ilce_id"])
  )
  observation_index = pd.MultiIndex.from_arrays(
    [
      sonuc["secim"].to_numpy(),
      sonuc["ilce_id"].to_numpy(),
    ],
    names=["secim", "ilce_id"],
  )
  dataset = sonuc
  for name in ("kadin", "erkek", "toplam", "kadin_oran", "erkek_oran"):
    values = (
      pd
      .to_numeric(gender_frame.reindex(observation_index)[name], errors="coerce")
      .astype("float32")
      .to_numpy()
    )
    dataset[name] = ("gozlem", values)
  return dataset


def write_selectable_analysis_dataset(
  source: Path | str,
  output: Path | str,
) -> None:
  """Write a selectable analysis dataset with broadcast demographics.

  Args:
    source: Source Zarr store containing result and demographic groups.
    output: Output Zarr store path.
  """

  sonuc = open_zarr(source)
  demografi = open_zarr(source, group="demografi")
  dataset = dataset_with_broadcast_demographics(sonuc, demografi)
  dataset = swap_dims(dataset, {"gozlem": "secim"})
  output_path = Path(output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  yer_chunk = min(dataset.sizes["secim"], 50_000)
  encoding: dict[str, dict[str, tuple[int, ...]]] = {
    "oy": {"chunks": (min(dataset.sizes["secim"], 5_000), dataset.sizes["tercih"])},
  }
  for name, variable in dataset.data_vars.items():
    if not isinstance(name, str):
      continue
    if name != "oy" and variable.dims == ("secim",):
      encoding[name] = {"chunks": (yer_chunk,)}
  to_zarr(
    dataset,
    output_path,
    mode="w",
    encoding=encoding,
  )


def _secim_tarihi(values: np.ndarray) -> np.ndarray:
  return pd.Series(values, dtype="string").str.slice(0, 10).to_numpy(dtype=object)


def _float32_array(array: xr.DataArray) -> xr.DataArray:
  return astype_float32(array)


def _makam_values(dataset: xr.Dataset) -> np.ndarray:
  if "secim_turu" in dataset.variables:
    raw = pd.Series(dataset["secim_turu"].to_numpy(), dtype="string")
  else:
    labels = pd.Series(dataset.indexes["secim"].astype(str), dtype="string")
    raw = labels.str.slice(11)
  return raw.replace(MAKAM_RENAMES).to_numpy(dtype=object)


def _choice_ids(dataset: xr.Dataset, choices: np.ndarray) -> np.ndarray:
  choice = pd.Series(choices, dtype="string")
  if "tercih_turu" in dataset.variables:
    choice_type = pd.Series(dataset["tercih_turu"].to_numpy(), dtype="string")
  else:
    choice_type = pd.Series(["tercih"] * len(choice), dtype="string")
  return (choice_type.fillna("tercih") + ":" + choice.fillna("")).to_numpy(dtype=object)


def redesign_secim_dataset(dataset: xr.Dataset) -> xr.Dataset:  # noqa: C901, PLR0912, PLR0915
  """Return the analysis schema with sparse observed-place rows.

  The row dimension is intentionally named ``yer_dim`` even though each row is
  an observed election/place result. This avoids allocating a mostly-empty
  dense ``secim x makam x yer`` cube while keeping ``.sel(il=...)`` and
  chained ``.sel(secim=...).sel(makam=...)`` access natural.

  Args:
    dataset: Source result dataset.

  Returns:
    A redesigned dataset in the final analysis schema.

  Raises:
    ValueError: If the source dataset has no supported row dimension.
  """
  source = dataset
  if "yer_dim" in source.sizes:
    row_dim = "yer_dim"
  elif "gozlem" in source.sizes:
    row_dim = "gozlem"
  elif "secim" in source.sizes:
    row_dim = "secim"
  else:
    msg = "dataset must contain a gozlem or secim row dimension"
    raise ValueError(msg)

  if "tercih_dim" in source.sizes:
    choice_dim = "tercih_dim"
  elif "tercih" in source.sizes:
    choice_dim = "tercih"
  else:
    choice_dim = "choice"
  if row_dim != "yer_dim" or choice_dim != "tercih_dim":
    source = source.rename({row_dim: "yer_dim", choice_dim: "tercih_dim"})

  old_row_labels = (
    dataset.indexes[row_dim].astype(str)
    if row_dim in dataset.indexes
    else pd.Index(np.arange(dataset.sizes[row_dim]).astype(str))
  )
  election_labels = (
    source["secim"].to_numpy().astype(object)
    if "secim" in source.variables
    else old_row_labels.to_numpy(dtype=object)
  )
  old_choices = source["tercih_dim"].to_numpy()
  source = assign_coords(
    source,
    {
      "yer_dim": np.arange(source.sizes["yer_dim"], dtype=np.int64),
      "tercih_dim": np.arange(source.sizes["tercih_dim"], dtype=np.int64),
      "gozlem_id": ("yer_dim", old_row_labels.to_numpy(dtype=object)),
      "secim_kaynak": ("yer_dim", election_labels),
      "secim": ("yer_dim", _secim_tarihi(election_labels)),
      "makam": ("yer_dim", _makam_values(source)),
      "tercih": ("tercih_dim", old_choices.astype(object)),
      "tercih_id": ("tercih_dim", _choice_ids(source, old_choices)),
    },
  )

  if "makam" in source.coords:
    makam = pd.Series(source["makam"].to_numpy(), dtype="string")
    source = assign_coords(
      source,
      makam_adi=("yer_dim", makam.replace(MAKAM_ADLARI).to_numpy(dtype=object)),
    )

  if "kapsam" in source.variables:
    kapsam = pd.Series(source["kapsam"].to_numpy(), dtype="string")
    source = assign_coords(
      source,
      {
        "kapsam_detay": ("yer_dim", kapsam.to_numpy(dtype=object)),
        "kapsam": ("yer_dim", kapsam.replace(KAPSAM_RENAMES).to_numpy(dtype=object)),
        "yer_turu": (
          "yer_dim",
          kapsam.replace(YER_TURU_RENAMES).to_numpy(dtype=object),
        ),
      },
    )

  if "tercih_adi" in source.variables:
    source = source.rename({"tercih_adi": "tercih_orijinal"})

  demographic_vars = [name for name in ("toplam", "kadin", "erkek") if name in source]
  if demographic_vars:
    groups = np.array(["toplam", "kadin", "erkek"], dtype=object)
    values: list[xr.DataArray] = []
    for group in groups:
      if group in source:
        values.append(_float32_array(cast("xr.DataArray", source[group])))
      else:
        values.append(
          xr.full_like(source[demographic_vars[0]], np.nan, dtype="float32")
        )
    source["secmen"] = xr.concat(
      values,
      dim=xr.IndexVariable("grup", groups),
    ).transpose("yer_dim", "grup")

  ratio_vars = [name for name in ("kadin_oran", "erkek_oran") if name in source]
  if ratio_vars:
    groups = np.array(["toplam", "kadin", "erkek"], dtype=object)
    ratios: list[xr.DataArray] = [
      xr.full_like(source[ratio_vars[0]], np.nan, dtype="float32"),
    ]
    for _group, var_name in (("kadin", "kadin_oran"), ("erkek", "erkek_oran")):
      if var_name in source:
        ratios.append(_float32_array(source[var_name]))
      else:
        ratios.append(xr.full_like(source[ratio_vars[0]], np.nan, dtype="float32"))
    source["secmen_oran"] = xr.concat(
      ratios,
      dim=xr.IndexVariable("grup", groups),
    ).transpose("yer_dim", "grup")

  for name in ("toplam", "kadin", "erkek", "kadin_oran", "erkek_oran"):
    if name in source:
      source = source.drop_vars(name)

  source = assign_coords(
    source,
    yer_surumu_no=("yer_dim", np.arange(source.sizes["yer_dim"], dtype=np.int64)),
    yer_kok_no=("yer_dim", np.arange(source.sizes["yer_dim"], dtype=np.int64)),
    yer_iliskisi_dim=np.array([], dtype=np.int64),
  )
  empty = np.array([], dtype="U1")
  source["kaynak_yer_surumu_no"] = ("yer_iliskisi_dim", np.array([], dtype=np.int64))
  source["hedef_yer_surumu_no"] = ("yer_iliskisi_dim", np.array([], dtype=np.int64))
  source["yer_iliskisi_turu"] = ("yer_iliskisi_dim", empty)
  source["yer_iliskisi_oran"] = ("yer_iliskisi_dim", np.array([], dtype="float32"))

  return source


def _final_secim_encoding(dataset: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
  row_count = max(1, dataset.sizes.get("yer_dim", 1))
  choice_count = max(1, dataset.sizes.get("tercih_dim", 1))
  group_count = max(1, dataset.sizes.get("grup", 1))
  row_chunk = min(row_count, 50_000)
  vote_row_chunk = min(row_count, 5_000)
  encoding: dict[str, dict[str, tuple[int, ...]]] = {}
  for name, variable in dataset.variables.items():
    if not isinstance(name, str):
      continue
    if variable.dims == ("yer_dim", "tercih_dim"):
      encoding[name] = {"chunks": (vote_row_chunk, choice_count)}
    elif variable.dims == ("yer_dim", "grup"):
      encoding[name] = {"chunks": (row_chunk, group_count)}
    elif variable.dims == ("yer_dim",):
      encoding[name] = {"chunks": (row_chunk,)}
    elif variable.dims == ("tercih_dim",):
      encoding[name] = {"chunks": (choice_count,)}
  return encoding


def write_final_secim_dataset(
  tables_dir: Path | str,
  path: Path | str,
  *,
  vote_dtype: str = "float32",
) -> None:
  """Build and atomically replace the final packaged election Zarr dataset.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.
    path: Final output Zarr store path.
    vote_dtype: Numeric dtype used when building intermediate vote arrays.

  Raises:
    FileExistsError: If temporary output paths already exist or the backup name
      is already taken.
  """

  output = Path(path)
  tmp = output.with_name(f"{output.name}.tmp")
  row_tmp = output.with_name(f"{output.name}.rows.tmp")
  if tmp.exists() or row_tmp.exists():
    msg = f"temporary output already exists: {tmp} or {row_tmp}"
    raise FileExistsError(msg)

  write_partitioned_dataset(tables_dir, row_tmp, vote_dtype=vote_dtype)
  row_dataset = open_zarr(row_tmp)
  final = redesign_secim_dataset(row_dataset)
  to_zarr(
    final,
    tmp,
    mode="w",
    encoding=_final_secim_encoding(final),
  )
  write_demographic_group(tables_dir, tmp)
  if output.exists():
    backup_dir = output.parent / "bak"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    backup = backup_dir / f"{output.name}.previous.{timestamp}"
    if backup.exists():
      msg = f"backup output already exists: {backup}"
      raise FileExistsError(msg)
    output.rename(backup)
  tmp.rename(output)
  shutil.rmtree(row_tmp)


def district_vote_demographic_frame(
  sonuc: xr.Dataset,
  demografi: xr.Dataset,
  *,
  secim: str,
  tercih: str | Iterable[str] | None = None,
) -> pd.DataFrame:
  """Return district vote totals joined with district demographics.

  Args:
    sonuc: Election result dataset.
    demografi: Demographic dataset.
    secim: Election slug to select.
    tercih: Optional choice or choices to include.

  Returns:
    A DataFrame of district vote totals and demographic columns.
  """

  mask = (sonuc["secim"] == secim) & (sonuc["kapsam"] == "yurtici_sandik")
  selected = sonuc.isel(gozlem=mask.values)
  vote_array = selected["oy"]
  if tercih is not None:
    vote_array = sel_data_array(vote_array, tercih=tercih)
  votes = vote_array.to_dataframe(name="oy").reset_index()
  vote_index = ["secim", "il", "ilce", "il_id", "ilce_id", "tercih"]
  vote_totals = (
    votes
    .groupby(vote_index, dropna=False)["oy"]
    .sum()
    .reset_index()
    .pivot_table(
      index=["secim", "il", "ilce", "il_id", "ilce_id"],
      columns="tercih",
      values="oy",
      fill_value=0,
    )
    .add_prefix("oy_")
    .reset_index()
  )

  demo_mask = (demografi["secim"] == secim) & (demografi["kapsam"] == "cinsiyet_ilce")
  demo = (
    demografi
    .isel(demografi=demo_mask.values)[
      ["kadin", "erkek", "toplam", "kadin_oran", "erkek_oran"]
    ]
    .to_dataframe()
    .reset_index()
  )
  demo = demo[
    [
      "secim",
      "ilce_id",
      "kadin",
      "erkek",
      "toplam",
      "kadin_oran",
      "erkek_oran",
    ]
  ].drop_duplicates()
  return vote_totals.merge(
    demo,
    on=["secim", "ilce_id"],
    how="left",
  )


def build_dataset_from_tables(
  tables: NormalizedTables,
  *,
  choices: pd.Index | None = None,
  vote_dtype: str = "float32",
) -> xr.Dataset:
  """Build an xarray result dataset from normalized table objects.

  Args:
    tables: Normalized table bundle.
    choices: Optional global choice index to enforce on the vote array.
    vote_dtype: Numeric dtype used for vote values.

  Returns:
    An xarray dataset with observation and choice dimensions.

  Raises:
    ValueError: If observations are present without ``observation_id``.
  """

  observations = tables.observations.reset_index(drop=True).copy()
  votes = tables.votes.reset_index(drop=True).copy()
  if observations.empty:
    return xr.Dataset()

  if "observation_id" not in observations.columns:
    msg = "observations table must contain observation_id"
    raise ValueError(msg)

  observation_ids = pd.Index(observations["observation_id"], name="observation")
  coords: dict[str, tuple[str, np.ndarray]] = {
    "observation": ("observation", observation_ids.to_numpy()),
  }
  coord_cols = INDEX_COLUMNS + sorted((ID_COORDS | LABEL_COORDS) & set(observations))
  for col in coord_cols:
    if col in observations.columns:
      coords[col] = ("observation", observations[col].to_numpy())

  dataset = xr.Dataset(coords=coords)
  for measure in sorted(MEASURE_COLUMNS & set(observations.columns)):
    dataset[measure] = (
      "observation",
      pd.to_numeric(observations[measure], errors="coerce").to_numpy(),
    )

  if not votes.empty:
    choices = (
      pd.Index(sorted(votes["choice"].dropna().unique()), name="choice")
      if choices is None
      else choices
    )
    pivot = (
      votes
      .pivot_table(
        index="observation_id",
        columns="choice",
        values="votes",
        aggfunc="sum",
      )
      .reindex(index=observation_ids, columns=choices)
      .astype(np.dtype(vote_dtype))
    )
    dataset = assign_coords(dataset, choice=choices.to_numpy())
    dataset["votes"] = (("observation", "choice"), pivot.to_numpy())

    if not tables.choices.empty:
      choice_meta = tables.choices.drop_duplicates("choice").set_index("choice")
      for col in ("choice_column", "choice_type", "choice_label_raw"):
        if col in choice_meta.columns:
          dataset[col] = (
            "choice",
            choice_meta.reindex(choices)[col].astype("object").to_numpy(),
          )

  return dataset


def is_partitioned_tables_dir(path: Path | str) -> bool:
  """Return whether ``path`` contains election-partitioned normalized tables.

  Args:
    path: Directory to inspect.

  Returns:
    ``True`` when at least one partition contains observations.
  """

  return any(Path(path).glob("election=*/observations.parquet"))


def partition_dirs(path: Path | str) -> list[Path]:
  """Return sorted ``election=*`` partition directories under ``path``.

  Args:
    path: Root table directory.

  Returns:
    Sorted partition directories.
  """

  return sorted(
    partition for partition in Path(path).glob("election=*") if partition.is_dir()
  )


def read_partition_tables(partition: Path | str) -> NormalizedTables:
  """Read normalized Parquet tables from a single partition directory.

  Args:
    partition: Partition directory to read.

  Returns:
    Loaded normalized table bundle.
  """

  path = Path(partition)
  frames = {
    name: pd.read_parquet(path / f"{name}.parquet")
    if (path / f"{name}.parquet").exists()
    else pd.DataFrame()
    for name in TABLE_NAMES
  }
  return NormalizedTables(**frames)


def global_choices_from_partitions(partitions: Iterable[Path]) -> pd.Index:
  """Return the sorted union of choice identifiers across partitions.

  Args:
    partitions: Partition directories to scan.

  Returns:
    A sorted choice index.
  """

  values: list[np.ndarray] = []
  for partition in partitions:
    choices_path = partition / "choices.parquet"
    votes_path = partition / "votes.parquet"
    if choices_path.exists():
      frame = pd.read_parquet(choices_path, columns=["choice"])
    elif votes_path.exists():
      frame = pd.read_parquet(votes_path, columns=["choice"])
    else:
      continue
    if "choice" in frame.columns:
      values.append(frame["choice"].dropna().unique())
  if not values:
    return pd.Index([], name="choice")
  return pd.Index(sorted(pd.unique(np.concatenate(values))), name="choice")


def _zarr_encoding(dataset: xr.Dataset) -> dict[str, dict[str, tuple[int, ...]]]:
  obs_dim = "gozlem" if "gozlem" in dataset.sizes else "observation"
  choice_dim = "tercih" if "tercih" in dataset.sizes else "choice"
  votes_name = "oy" if "oy" in dataset.variables else "votes"
  obs_count = max(1, dataset.sizes.get(obs_dim, 1))
  choice_count = max(1, dataset.sizes.get(choice_dim, 1))
  obs_chunk = min(obs_count, 50_000)
  vote_obs_chunk = min(obs_count, 5_000)
  encoding: dict[str, dict[str, tuple[int, ...]]] = {}
  for name, variable in dataset.variables.items():
    if not isinstance(name, str):
      continue
    if name == votes_name:
      encoding[name] = {"chunks": (vote_obs_chunk, choice_count)}
    elif variable.dims == (obs_dim,):
      encoding[name] = {"chunks": (obs_chunk,)}
    elif variable.dims == (choice_dim,):
      encoding[name] = {"chunks": (choice_count,)}
  return encoding


def write_partitioned_dataset(
  tables_dir: Path | str,
  path: Path | str,
  *,
  vote_dtype: str = "float32",
) -> None:
  """Write all normalized table partitions into one appendable Zarr dataset.

  Args:
    tables_dir: Directory containing ``election=*`` table partitions.
    path: Output Zarr store path.
    vote_dtype: Numeric dtype used for vote values.

  Raises:
    ValueError: If no partitioned tables are found.
  """

  partitions = partition_dirs(tables_dir)
  if not partitions:
    msg = f"no partitioned tables found in {tables_dir}"
    raise ValueError(msg)

  choices = global_choices_from_partitions(partitions)
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)

  for index, partition in enumerate(partitions):
    tables = read_partition_tables(partition)
    dataset = build_dataset_from_tables(
      tables,
      choices=choices,
      vote_dtype=vote_dtype,
    )
    dataset = to_turkish_schema(dataset)
    append_dim = "gozlem" if "gozlem" in dataset.sizes else "observation"
    if index == 0:
      to_zarr(
        dataset,
        output,
        mode="w",
        encoding=_zarr_encoding(dataset),
      )
      continue

    to_zarr(
      dataset,
      output,
      mode="a",
      append_dim=append_dim,
    )


def read_legacy_csvs(path: Path | str) -> pd.DataFrame:
  """Read legacy wide CSV exports and return a normalized long vote frame.

  Args:
    path: Directory containing legacy CSV exports.

  Returns:
    A normalized long vote DataFrame.
  """

  frames: list[pd.DataFrame] = []
  for csv_path in sorted(Path(path).glob("*.csv")):
    stem = csv_path.stem
    parts = stem.split("-")
    if len(parts) < 3:
      continue
    election = "-".join(parts[:-1])
    scope = parts[-1]
    frame = pd.read_csv(csv_path).rename(columns=LEGACY_RENAMES)
    frame.insert(0, "scope", scope)
    frame.insert(0, "election", election)
    id_vars = [
      col
      for col in frame.columns
      if col in (LEGACY_META_COLUMNS | set(LEGACY_RENAMES.values()))
    ]
    value_vars = [col for col in frame.columns if col not in id_vars]
    long = frame.melt(
      id_vars=id_vars,
      value_vars=value_vars,
      var_name="choice",
      value_name="votes",
    )
    frames.append(long)
  return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_dataset(dataset: xr.Dataset, path: Path | str) -> None:
  """Write a dataset to Zarr or NetCDF based on the output suffix.

  Args:
    dataset: Dataset to write.
    path: Output path. A ``.zarr`` suffix writes Zarr; other suffixes write
      NetCDF.
  """

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.suffix == ".zarr":
    to_zarr(dataset, output, mode="w")
  else:
    to_netcdf(dataset, output)
