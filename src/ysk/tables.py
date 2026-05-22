from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
  from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class DemographicSource:
  """Source metadata attached to normalized demographic rows."""

  election: str
  election_id: int
  election_kind: str
  endpoint: str
  scope: str
  params: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class NormalizedTables:
  """Normalized election tables split by observation, vote, and metadata role."""

  observations: pd.DataFrame
  votes: pd.DataFrame
  choices: pd.DataFrame
  geography: pd.DataFrame
  demographics: pd.DataFrame

  @classmethod
  def empty(cls) -> NormalizedTables:
    """Return an empty table bundle with the expected table names.

    Returns:
      A ``NormalizedTables`` instance with empty DataFrames.
    """

    return cls(
      observations=pd.DataFrame(),
      votes=pd.DataFrame(),
      choices=pd.DataFrame(),
      geography=pd.DataFrame(),
      demographics=pd.DataFrame(),
    )


TABLE_NAMES = ("observations", "votes", "choices", "geography", "demographics")


def concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
  """Concatenate non-empty frames, returning an empty frame when none exist.

  Args:
    frames: Frames to concatenate.

  Returns:
    A concatenated DataFrame, or an empty DataFrame.
  """

  materialized = [frame for frame in frames if not frame.empty]
  return pd.concat(materialized, ignore_index=True) if materialized else pd.DataFrame()


def merge_tables(tables: Iterable[NormalizedTables]) -> NormalizedTables:
  """Merge multiple normalized table bundles into one bundle.

  Args:
    tables: Normalized table bundles to merge.

  Returns:
    A single merged table bundle.
  """

  materialized = list(tables)
  if not materialized:
    return NormalizedTables.empty()
  return NormalizedTables(
    observations=concat_frames(table.observations for table in materialized),
    votes=concat_frames(table.votes for table in materialized),
    choices=concat_frames(table.choices for table in materialized).drop_duplicates(),
    geography=concat_frames(
      table.geography for table in materialized
    ).drop_duplicates(),
    demographics=concat_frames(
      table.demographics for table in materialized
    ).drop_duplicates(),
  )


def write_tables(tables: NormalizedTables, output_dir: Path | str) -> None:
  """Write a normalized table bundle as Parquet files.

  Args:
    tables: Normalized table bundle to write.
    output_dir: Directory that will receive one Parquet file per table.
  """

  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  for name in TABLE_NAMES:
    frame = getattr(tables, name)
    frame.to_parquet(output / f"{name}.parquet", index=False)


def write_partitioned_tables(
  tables: NormalizedTables,
  output_dir: Path | str,
  partition: str,
) -> None:
  """Write a table bundle under an ``election=<partition>`` directory.

  Args:
    tables: Normalized table bundle to write.
    output_dir: Root output directory.
    partition: Partition name used after ``election=``.
  """

  output = Path(output_dir) / f"election={partition}"
  write_tables(tables, output)


def partition_exists(output_dir: Path | str, partition: str) -> bool:
  """Return whether all normalized Parquet tables exist for a partition.

  Args:
    output_dir: Root output directory.
    partition: Partition name used after ``election=``.

  Returns:
    ``True`` when every normalized table file exists.
  """

  output = Path(output_dir) / f"election={partition}"
  return all((output / f"{name}.parquet").exists() for name in TABLE_NAMES)


def read_tables(input_dir: Path | str) -> NormalizedTables:
  """Read normalized tables from a flat or election-partitioned directory.

  Args:
    input_dir: Directory containing table Parquet files or ``election=*``
      partitions.

  Returns:
    The loaded normalized table bundle.
  """

  path = Path(input_dir)
  frames: dict[str, pd.DataFrame] = {}
  for name in TABLE_NAMES:
    table_path = path / f"{name}.parquet"
    if table_path.exists():
      frames[name] = pd.read_parquet(table_path)
      continue
    partition_paths = sorted(path.glob(f"election=*/{name}.parquet"))
    frames[name] = (
      pd.concat(
        (pd.read_parquet(partition_path) for partition_path in partition_paths),
        ignore_index=True,
      )
      if partition_paths
      else pd.DataFrame()
    )
  return NormalizedTables(**frames)


def clean_district(frame: pd.DataFrame) -> pd.DataFrame:
  """Normalize YSK district names in-place and return ``frame``.

  Args:
    frame: DataFrame that may contain a ``district`` column.

  Returns:
    The same DataFrame with normalized district values.
  """

  if "district" in frame.columns:
    frame["district"] = frame["district"].mask(
      frame["district"].str.endswith(" MERKEZ", na=False), "MERKEZ"
    )
  return frame


def geography_from_observations(observations: pd.DataFrame) -> pd.DataFrame:
  """Extract distinct geography metadata rows from observations.

  Args:
    observations: Normalized observation table.

  Returns:
    Distinct geography rows.
  """

  if observations.empty:
    return pd.DataFrame()
  geography_cols = [
    col
    for col in (
      "election",
      "scope",
      "province_id",
      "province",
      "district_id",
      "district",
      "neighborhood_id",
      "neighborhood",
      "country_id",
      "country",
      "foreign_office_id",
      "foreign_office",
      "customs_id",
      "customs",
      "secim_cevresi_id",
      "belde_id",
      "birim_id",
      "cezaevi_id",
    )
    if col in observations.columns
  ]
  return observations[geography_cols].drop_duplicates().reset_index(drop=True)


def normalize_demographic_rows(
  rows: list[dict[str, object]] | dict[str, Any],
  source: DemographicSource,
) -> pd.DataFrame:
  """Normalize raw demographic endpoint rows with source metadata.

  Args:
    rows: Raw demographic rows or a single raw row.
    source: Source metadata to attach to each row.

  Returns:
    A normalized demographic DataFrame.
  """

  if not rows:
    return pd.DataFrame()
  materialized = [rows] if isinstance(rows, dict) else rows
  frame = pd.DataFrame(materialized)
  frame = frame.rename(
    columns={
      "il_ID": "province_id",
      "il_ADI": "province",
      "ilce_ID": "district_id",
      "ilce_ADI": "district",
      "muhtarlik_ID": "neighborhood_id",
      "muhtarlik_ADI": "neighborhood",
    },
  )
  frame = clean_district(frame)
  frame.insert(0, "params", repr(sorted(source.params.items())))
  frame.insert(0, "scope", source.scope)
  frame.insert(0, "endpoint", source.endpoint)
  frame.insert(0, "election_kind", source.election_kind)
  frame.insert(0, "election_id", source.election_id)
  frame.insert(0, "election", source.election)
  return frame
