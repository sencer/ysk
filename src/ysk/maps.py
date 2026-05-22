from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from numbers import Real
from typing import Literal

import pandas as pd

from ysk.analysis import TOPLAM_KOLONLARI

HoverCallback = Callable[[pd.Series], str]
TooltipLevel = Literal["auto", "il", "ilce"]

PARTI_RENKLERI = {
  "ak_parti": "#F58220",
  "btp": "#D71920",
  "buyuk_birlik": "#C8102E",
  "chp": "#E30613",
  "dem_parti": "#7B3294",
  "deva_partisi": "#1E9B8F",
  "dp": "#C62828",
  "dsp": "#4AA3DF",
  "emep": "#B71C1C",
  "huda_par": "#006A4E",
  "iyi_parti": "#00A6D6",
  "mhp": "#1F3A93",
  "milli_yol": "#5D4037",
  "saadet": "#2EAD4B",
  "tip": "#B00020",
  "tkp": "#C1121F",
  "vatan_partisi": "#B00020",
  "yeniden_refah": "#006B3F",
  "zafer_partisi": "#26A69A",
}
PARTI_ADLARI = {
  "ak_parti": "AK Parti",
  "bagimsiz_toplam_oy": "Bağımsız",  # noqa: RUF001
  "btp": "BTP",
  "buyuk_birlik": "Büyük Birlik Partisi",
  "chp": "CHP",
  "dem_parti": "DEM Parti",
  "deva_partisi": "DEVA Partisi",
  "dp": "Demokrat Parti",
  "dsp": "DSP",
  "emep": "EMEP",
  "huda_par": "HÜDA PAR",
  "iyi_parti": "İYİ Parti",
  "mhp": "MHP",
  "milli_yol": "Milli Yol Partisi",
  "saadet": "Saadet Partisi",
  "tip": "TİP",
  "tkp": "TKP",
  "vatan_partisi": "Vatan Partisi",
  "yeniden_refah": "Yeniden Refah Partisi",
  "zafer_partisi": "Zafer Partisi",
}
KOLON_ADLARI = {
  "secmen": "Seçmen",
  "oy_kullanan": "Oy kullanan",
  "gecerli": "Geçerli oy",
  "gecersiz": "Geçersiz oy",
  "gecerli_itirazsiz": "Geçerli itirazsız",  # noqa: RUF001
  "gecerli_itirazli": "Geçerli itirazlı",  # noqa: RUF001
  "kadin": "Kadın",  # noqa: RUF001
  "erkek": "Erkek",
}
LOCATION_COLUMNS = {
  "OBJECTID",
  "adm1",
  "adm1_en",
  "adm1_tr",
  "adm2_en",
  "adm2_tr",
  "pcode",
  "Shape_Area",
  "Shape_Leng",
  "secim",
  "makam",
  "il",
  "ilce",
  "ilce_secim",
  "belde",
  "belde_adi",
  "mahalle",
  "sandik_no",
  "geometry",
}


def result_tooltip(
  *,
  topk: int | None = None,
  threshold: float | None = None,
  choices: Sequence[str] | None = None,
  total_column: str = "gecerli",
  level: TooltipLevel = "auto",
) -> HoverCallback:
  """Build a turkiye hover callback for election result rows.

  The returned function formats the selected place name, common totals, and
  either the top ``topk`` choices or choices meeting ``threshold``.

  Args:
    topk: Number of leading choices to show.
    threshold: Minimum vote share to show, expressed from 0 to 1.
    choices: Optional choice columns to consider. When omitted, numeric vote
      columns are detected from each row.
    total_column: Column used as the denominator for vote shares.
    level: Place-name detail to show in the tooltip.

  Returns:
    A callback that converts a result row to HTML tooltip text.

  Raises:
    ValueError: If exactly one truncation mode is not provided, ``topk`` is not
      positive, or ``threshold`` is outside 0 to 1.
  """

  if (topk is None) == (threshold is None):
    msg = "exactly one of topk=... or threshold=... must be provided"
    raise ValueError(msg)
  if topk is not None and topk <= 0:
    msg = "topk must be positive"
    raise ValueError(msg)
  if threshold is not None and not 0 <= threshold <= 1:
    msg = "threshold must be between 0 and 1"
    raise ValueError(msg)

  def hover(row: pd.Series) -> str:
    total = _numeric_value(row.get(total_column))
    results = _hover_vote_results(row, choices=choices, total=total)
    if topk is not None:
      results = results[:topk]
    else:
      threshold_value = threshold
      if threshold_value is None:
        msg = "threshold must be provided when topk is not provided"
        raise ValueError(msg)
      results = [
        result
        for result in results
        if total > 0 and result[1] / total >= threshold_value
      ]

    lines = [f"<b>{_display_place_from_row(row, level=level)}</b>"]
    lines.extend(
      f"{_party_label(column)}: {_format_value(row[column])}"
      for column in ("secmen", total_column)
      if column in row
    )
    turnout = _format_turnout(row)
    if turnout is not None:
      lines.append(f"Katılım: {turnout}")  # noqa: RUF001
    lines.extend(
      f"{_party_label(choice)}: {_format_vote_share(votes, total)}"
      for choice, votes in results
    )
    return "<br>".join(lines)

  return hover


def _hover_vote_results(
  row: pd.Series,
  *,
  choices: Sequence[str] | None,
  total: float,
) -> list[tuple[str, float]]:
  selected_choices = choices if choices is not None else _vote_columns(row)
  results = [
    (choice, value)
    for choice in selected_choices
    if (value := _numeric_value(row.get(choice))) > 0
  ]
  if total <= 0:
    return sorted(results, key=lambda result: (-result[1], result[0].casefold()))
  return sorted(
    results,
    key=lambda result: (-result[1] / total, -result[1], result[0].casefold()),
  )


def _vote_columns(row: pd.Series) -> list[str]:
  return [
    column
    for column in row.index
    if isinstance(column, str)
    and column not in LOCATION_COLUMNS
    and column not in TOPLAM_KOLONLARI
    and not column.endswith(("_oran", "_orani"))
    and _numeric_value(row[column]) > 0
  ]


def _numeric_value(value: object) -> float:
  if _is_missing_value(value):
    return 0.0
  if isinstance(value, Real):
    return float(value)
  if not isinstance(value, str):
    return 0.0
  try:
    return float(value)
  except ValueError:
    return 0.0


def _format_vote_share(votes: float, total: float) -> str:
  if total <= 0:
    return f"{votes:,.0f}"
  return f"{votes:,.0f} ({votes / total:.1%})"


def _format_turnout(row: pd.Series) -> str | None:
  registered = _numeric_value(row.get("secmen"))
  voters = _numeric_value(row.get("oy_kullanan"))
  if registered <= 0 or voters <= 0:
    return None
  return f"{voters / registered:.1%}"


def _display_place_from_row(row: pd.Series, *, level: TooltipLevel) -> str:
  if level == "il":
    return _province_from_row(row)
  if level == "ilce":
    return _district_from_row(row)
  if _has_value(row, "adm2_tr") and _has_value(row, "adm1_tr"):
    return f"{_turkish_title(row['adm2_tr'])}, {_turkish_title(row['adm1_tr'])}"
  if _has_value(row, "ilce") and _has_value(row, "il"):
    return f"{_turkish_title(row['ilce'])}, {_turkish_title(row['il'])}"
  province = _province_from_row(row)
  if province:
    return province
  return ""


def _province_from_row(row: pd.Series) -> str:
  if _has_value(row, "adm1_tr"):
    return _turkish_title(row["adm1_tr"])
  if _has_value(row, "il"):
    return _turkish_title(row["il"])
  return ""


def _district_from_row(row: pd.Series) -> str:
  province = _province_from_row(row)
  if _has_value(row, "adm2_tr"):
    district = _turkish_title(row["adm2_tr"])
  elif _has_value(row, "ilce"):
    district = _turkish_title(row["ilce"])
  else:
    return province
  return f"{district}, {province}" if province else district


def _has_value(row: pd.Series, column: str) -> bool:
  return (
    column in row
    and not _is_missing_value(row[column])
    and bool(str(row[column]).strip())
  )


def _turkish_title(value: object) -> str:
  text = str(value).strip()
  lowered = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()  # noqa: RUF001
  return " ".join(_capitalize_turkish_word(word) for word in lowered.split())


def _capitalize_turkish_word(word: str) -> str:
  if not word:
    return word
  first = "İ" if word[0] == "i" else word[0].upper()
  return first + word[1:]


def _party_label(column: str) -> str:
  if column in PARTI_ADLARI:
    return PARTI_ADLARI[column]
  if column in KOLON_ADLARI:
    return KOLON_ADLARI[column]
  return _turkish_title(column.replace("_", " "))


def _format_value(value: object) -> str:
  if _is_missing_value(value):
    return "Veri yok"
  if isinstance(value, int | float):
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.3g}"
  return str(value)


def _is_missing_value(value: object) -> bool:
  return (
    value is None
    or value is pd.NA
    or value is pd.NaT
    or (isinstance(value, float) and math.isnan(value))
  )


def party_color(party: str) -> str:
  """Return the configured hex color for a normalized party column name.

  Args:
    party: Normalized party column name.

  Returns:
    The configured hex color.

  Raises:
    KeyError: If ``party`` is not in ``PARTI_RENKLERI``.
  """

  return PARTI_RENKLERI[party]
