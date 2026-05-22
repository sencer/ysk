from __future__ import annotations

import re
from typing import TYPE_CHECKING
import unicodedata

import pandas as pd

from ysk.tables import NormalizedTables, clean_district, geography_from_observations

if TYPE_CHECKING:
  from collections.abc import Iterable, Sequence

  from ysk.catalog import SecimKaydi

RESULT_ENDPOINT = "getSecimSandikSonucList"
CHOICE_ENDPOINT = "getSandikSecimSonucBaslikList"
CINSIYET_ENDPOINT = "getKadinErkekCinsiyetOraniIlGroupIlce"

CORE_RENAMES = {
  "il_ID": "province_id",
  "il_ADI": "province",
  "ilce_ID": "district_id",
  "ilce_ADI": "district",
  "muhtarlik_ID": "neighborhood_id",
  "muhtarlik_ADI": "neighborhood",
  "sandik_ID": "ballot_box_id",
  "sandik_NO": "ballot_box_no",
  "sandik_RUMUZ": "ballot_box_code",
  "sandik_SONUC_ID": "ballot_result_id",
  "secim_CEVRESI_ID": "secim_cevresi_id",
  "belde_ID": "belde_id",
  "birim_ID": "birim_id",
  "cezaevi_ID": "cezaevi_id",
  "ulke_ID": "country_id",
  "ulke_ADI": "country",
  "dis_TEMSILCILIK_ID": "foreign_office_id",
  "dis_TEMSILCILIK_ADI": "foreign_office",
  "gumruk_ID": "customs_id",
  "gumruk_ADI": "customs",
  "secmen_SAYISI": "registered_voters",
  "oy_KULLANAN_SECMEN_SAYISI": "voters",
  "itirazsiz_GECERLI_OY_SAYISI": "valid_uncontested",
  "itirazli_GECERLI_OY_SAYISI": "valid_contested",
  "gecerli_OY_TOPLAMI": "valid_votes",
  "gecersiz_OY_TOPLAMI": "invalid_votes",
  "son_ISLEM_TARIHI": "last_updated_at",
}

MEASURE_COLUMNS = {
  "registered_voters",
  "voters",
  "valid_uncontested",
  "valid_contested",
  "valid_votes",
  "invalid_votes",
}

ID_COORDS = {
  "province_id",
  "district_id",
  "neighborhood_id",
  "ballot_box_id",
  "ballot_result_id",
  "country_id",
  "foreign_office_id",
  "customs_id",
  "secim_cevresi_id",
  "belde_id",
  "birim_id",
  "cezaevi_id",
}

LABEL_COORDS = {
  "province",
  "district",
  "neighborhood",
  "ballot_box_no",
  "ballot_box_code",
  "country",
  "foreign_office",
  "customs",
  "last_updated_at",
}


CHOICE_TYPE_PREFIXES = {
  "aday": "candidate",
  "parti": "party",
  "bagimsiz": "independent",
  "ittifak": "alliance",
}


def slugify(value: object) -> str:
  """Convert a label into a stable ASCII identifier.

  Args:
    value: Value to normalize.

  Returns:
    A lowercase ASCII identifier, or ``"unknown"`` for empty input.
  """

  text = str(value).strip().lower()
  text = unicodedata.normalize("NFKD", text)
  text = "".join(ch for ch in text if not unicodedata.combining(ch))
  text = text.translate(
    str.maketrans({"\u0131": "i", "\u011f": "g", "\u00fc": "u", "\u015f": "s"}),
  )
  text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
  return text or "unknown"


def choice_columns(columns: Iterable[str]) -> list[str]:
  """Return raw YSK vote columns from an iterable of column names.

  Args:
    columns: Candidate column names.

  Returns:
    Column names matching YSK candidate, party, independent, or alliance vote
    fields.
  """

  patterns = (
    r"^aday\d+_ALDIGI_OY$",
    r"^parti\d+_ALDIGI_OY$",
    r"^bagimsiz\d+_ALDIGI_OY$",
    r"^ittifak\d+_ALDIGI_OY$",
  )
  return [col for col in columns if any(re.match(pattern, col) for pattern in patterns)]


def choice_lookup(rows: Sequence[dict[str, object]]) -> dict[str, str]:
  """Map raw YSK vote column names to normalized choice identifiers.

  Args:
    rows: Raw choice metadata rows from YSK.

  Returns:
    A mapping from raw YSK column name to normalized choice identifier.
  """

  lookup: dict[str, str] = {}
  for row in rows:
    column = row.get("column_NAME")
    name = row.get("ad") or row.get("adi") or row.get("kisa_AD") or column
    if column is not None:
      lookup[str(column)] = slugify(name)
  return lookup


def choice_type(column: str) -> str:
  """Return the normalized choice type for a raw YSK vote column.

  Args:
    column: Raw YSK vote column name.

  Returns:
    The normalized choice type, or ``"unknown"``.
  """

  for prefix, value in CHOICE_TYPE_PREFIXES.items():
    if column.startswith(prefix):
      return value
  return "unknown"


def normalize_choice_table(
  election: SecimKaydi,
  rows: Sequence[dict[str, object]],
) -> pd.DataFrame:
  """Normalize raw YSK choice metadata rows for one election.

  Args:
    election: Election catalog entry.
    rows: Raw choice metadata rows from YSK.

  Returns:
    A normalized choice metadata DataFrame.
  """

  records: list[dict[str, object]] = []
  for row in rows:
    column = row.get("column_NAME")
    if column is None:
      continue
    label = row.get("ad") or row.get("adi") or row.get("kisa_AD") or column
    column_text = str(column)
    records.append(
      {
        "election": election.slug,
        "election_id": election.ysk_id,
        "election_kind": election.kind.value,
        "choice_column": column_text,
        "choice": slugify(label),
        "choice_type": choice_type(column_text),
        "choice_label_raw": label,
      },
    )
  return pd.DataFrame.from_records(records)


def normalize_result_frame(
  election: SecimKaydi,
  rows: Sequence[dict[str, object]],
  *,
  scope: str,
  choice_names: dict[str, str] | None = None,
) -> pd.DataFrame:
  """Normalize raw YSK result rows into a long vote DataFrame.

  Args:
    election: Election catalog entry.
    rows: Raw result rows from YSK.
    scope: Result scope label to attach to the rows.
    choice_names: Optional mapping from raw vote columns to normalized choices.

  Returns:
    A long result DataFrame with one row per observation and choice.
  """

  if not rows:
    return pd.DataFrame()

  frame = pd.DataFrame(rows).rename(columns=CORE_RENAMES)
  frame = clean_district(frame)
  choices = choice_columns(frame.columns)
  if choice_names:
    choices = [col for col in choices if col in choice_names or col in frame.columns]

  meta_cols = [
    col
    for col in sorted((ID_COORDS | LABEL_COORDS | MEASURE_COLUMNS) & set(frame.columns))
    if col in frame.columns
  ]
  normalized = frame[meta_cols].copy()
  normalized.insert(0, "scope", scope)
  normalized.insert(0, "election_kind", election.kind.value)
  normalized.insert(0, "election_id", election.ysk_id)
  normalized.insert(0, "election", election.slug)

  if not choices:
    normalized["choice"] = pd.NA
    normalized["votes"] = pd.NA
    return normalized

  melted = normalized.join(frame[choices]).melt(
    id_vars=list(normalized.columns),
    value_vars=choices,
    var_name="choice_column",
    value_name="votes",
  )
  melted["choice"] = melted["choice_column"].map(
    lambda col: choice_names.get(col, slugify(col)) if choice_names else slugify(col),
  )
  return melted.drop(columns=["choice_column"])


def normalize_result_tables(
  election: SecimKaydi,
  rows: Sequence[dict[str, object]],
  *,
  scope: str,
  request_key: str,
  choices: pd.DataFrame,
) -> NormalizedTables:
  """Normalize raw YSK result rows into the package table bundle.

  Args:
    election: Election catalog entry.
    rows: Raw result rows from YSK.
    scope: Result scope label to attach to the rows.
    request_key: Stable key identifying the request that produced the rows.
    choices: Normalized choice metadata for the election.

  Returns:
    Normalized observation, vote, choice, geography, and demographic tables.
  """

  if not rows:
    return NormalizedTables.empty()

  frame = pd.DataFrame(rows).rename(columns=CORE_RENAMES)
  frame = clean_district(frame)
  vote_columns = choice_columns(frame.columns)
  if {"choice_column", "choice"} <= set(choices.columns):
    choice_map = dict(zip(choices["choice_column"], choices["choice"], strict=False))
  else:
    choice_map = {}

  meta_cols = [
    col
    for col in sorted((ID_COORDS | LABEL_COORDS | MEASURE_COLUMNS) & set(frame.columns))
    if col in frame.columns
  ]
  observations = frame[meta_cols].copy()
  observations.insert(0, "source_row", range(len(observations)))
  observations.insert(0, "request_key", request_key)
  observations.insert(
    0,
    "observation_id",
    [
      f"{election.slug}:{scope}:{request_key}:{source_row}"
      for source_row in range(len(observations))
    ],
  )
  observations.insert(0, "scope", scope)
  observations.insert(0, "election_kind", election.kind.value)
  observations.insert(0, "election_id", election.ysk_id)
  observations.insert(0, "election", election.slug)

  if vote_columns:
    votes = (
      observations[["observation_id", "election", "scope"]]
      .join(frame[vote_columns])
      .melt(
        id_vars=["observation_id", "election", "scope"],
        value_vars=vote_columns,
        var_name="choice_column",
        value_name="votes",
      )
    )
    votes["choice"] = votes["choice_column"].map(
      lambda column: choice_map.get(column, slugify(column)),
    )
  else:
    votes = pd.DataFrame(
      columns=[
        "observation_id",
        "election",
        "scope",
        "choice_column",
        "choice",
        "votes",
      ],
    )

  return NormalizedTables(
    observations=observations,
    votes=votes,
    choices=choices,
    geography=geography_from_observations(observations),
    demographics=pd.DataFrame(),
  )
