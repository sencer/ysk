from typing import cast

import pandas as pd
import pytest

import ysk
from ysk import (
  BAGIMSIZLAR,
  PARTI_RENKLERI,
  PARTILER,
  TOPLAM_KOLONLARI,
  Kolon,
  Makam,
  Secim,
  SecimTarihi,
  election_results,
  party_color,
  result_tooltip,
)


def _guclukonak_row() -> pd.Series:
  frame = election_results(
    Secim.YEREL_2024,
    Makam.BELEDIYE_BASKANI,
    province="SIRNAK",
    level="ilce",
  ).reset_index()
  return frame[frame["ilce"] == "GUCLUKONAK"].iloc[0]


def test_secim_tarihi_and_makam_are_string_compatible_enums() -> None:
  assert Secim.YEREL_2024 == "2024-03-31"
  assert SecimTarihi.YEREL_2024 == Secim.YEREL_2024
  assert str(Secim.YEREL_2024) == "2024-03-31"
  assert Makam.BELEDIYE_MECLISI == "BM"
  assert str(Makam.BELEDIYE_MECLISI) == "BM"


def test_turkish_constants_include_column_groups() -> None:
  assert {
    "secmen",
    "oy_kullanan",
    "gecerli",
    "gecersiz",
    "gecerli_itirazsiz",
    "gecerli_itirazli",
    "kadin",
    "erkek",
  } == TOPLAM_KOLONLARI
  assert {"ak_parti", "chp", "mhp", "dem_parti"}.issubset(PARTILER)
  assert {"bagimsiz_toplam_oy"} == BAGIMSIZLAR


def test_party_color_uses_fixed_party_color_constant() -> None:
  assert PARTI_RENKLERI["ak_parti"] == "#F58220"
  assert PARTI_RENKLERI["chp"] == "#E30613"
  assert PARTI_RENKLERI["mhp"] == "#1F3A93"
  assert PARTI_RENKLERI["dem_parti"] == "#7B3294"
  assert PARTI_RENKLERI["saadet"] == "#2EAD4B"
  assert PARTI_RENKLERI["yeniden_refah"] == "#006B3F"
  assert PARTI_RENKLERI.keys() >= PARTILER
  assert party_color("ak_parti") == "#F58220"
  with pytest.raises(KeyError):
    party_color("bilinmeyen_parti")


def test_removed_turkish_methods_are_not_reexported_from_ysk() -> None:
  assert "ham_veri" not in ysk.__all__
  assert "secim" not in ysk.__all__
  assert "division_relationships" not in ysk.__all__
  assert "yer_iliskileri" not in ysk.__all__
  assert "sonuc_ipucu" not in ysk.__all__
  assert "parti_rengi" not in ysk.__all__
  assert "parti_renkleri" not in ysk.__all__
  assert "party_colors" not in ysk.__all__
  assert "grafik" not in ysk.__all__
  assert "interaktif_grafik" not in ysk.__all__
  assert "turkiye_harita_yolu" not in ysk.__all__
  assert "turkiye_haritasi" not in ysk.__all__
  assert not hasattr(ysk, "ham_veri")
  assert not hasattr(ysk, "secim")
  assert not hasattr(ysk, "division_relationships")
  assert not hasattr(ysk, "yer_iliskileri")
  assert not hasattr(ysk, "sonuc_ipucu")
  assert not hasattr(ysk, "parti_rengi")
  assert not hasattr(ysk, "parti_renkleri")
  assert not hasattr(ysk, "party_colors")
  assert not hasattr(ysk, "grafik")
  assert not hasattr(ysk, "interaktif_grafik")
  assert not hasattr(ysk, "turkiye_harita_yolu")
  assert not hasattr(ysk, "turkiye_haritasi")


def test_ilce_level_mayor_results_exclude_belde_rows() -> None:
  guclukonak = election_results(
    Secim.YEREL_2024,
    Makam.BELEDIYE_BASKANI,
    province="SIRNAK",
    level="ilce",
  ).reset_index()
  row = guclukonak.loc[guclukonak["ilce"].eq("GUCLUKONAK")].iloc[0]

  assert row["buyuk_birlik"] == 825
  assert row["ak_parti"] == 750


def test_sandik_level_does_not_include_broadcast_demographics() -> None:
  data = election_results(
    Secim.YEREL_2024,
    Makam.BUYUKSEHIR_BELEDIYE_BASKANI,
    province="ISTANBUL",
    district="ADALAR",
  )

  assert "kadin" not in data.columns
  assert "erkek" not in data.columns


def test_province_level_demographics_sum_district_demographics() -> None:
  districts = election_results(
    Secim.YEREL_2024,
    Makam.BUYUKSEHIR_BELEDIYE_BASKANI,
    province="ISTANBUL",
    level="ilce",
  )
  provinces = election_results(
    Secim.YEREL_2024,
    Makam.BUYUKSEHIR_BELEDIYE_BASKANI,
    province="ISTANBUL",
    level="il",
  )

  assert provinces.loc["ISTANBUL", "kadin"] == districts["kadin"].sum()
  assert provinces.loc["ISTANBUL", "erkek"] == districts["erkek"].sum()


def test_filter_columns_returns_filtered_dataframe() -> None:
  data = election_results(
    Secim.YEREL_2024,
    Makam.BELEDIYE_BASKANI,
    province="SIRNAK",
    level="ilce",
  )

  totals = ysk.filter_columns(data, types=Kolon.TOPLAM)
  parties = ysk.filter_columns(data, types=Kolon.PARTI)
  independent = ysk.filter_columns(data, types=Kolon.BAGIMSIZ)
  combined = ysk.filter_columns(data, types=Kolon.PARTI | Kolon.BAGIMSIZ)

  assert list(totals.columns) == [
    column for column in data.columns if column in TOPLAM_KOLONLARI
  ]
  assert "ak_parti" in parties.columns
  assert "buyuk_birlik" in parties.columns
  assert "bagimsiz_toplam_oy" not in parties.columns
  assert list(independent.columns) == ["bagimsiz_toplam_oy"]
  assert {"ak_parti", "bagimsiz_toplam_oy"}.issubset(combined.columns)


def test_choices_accept_display_names() -> None:
  data = election_results(
    Secim.YEREL_2024,
    Makam.BUYUKSEHIR_BELEDIYE_BASKANI,
    province="ISTANBUL",
    level="ilce",
    choices=["CHP", "AK PARTI"],
  )

  assert {"chp", "ak_parti"}.issubset(data.columns)


def test_multi_election_can_align_historical_district_axes() -> None:
  current = election_results(
    Secim.YEREL_2024,
    Makam.BELEDIYE_BASKANI,
    province="HAKKARI",
    level="ilce",
  )
  unaligned = election_results(
    [Secim.YEREL_2014, Secim.YEREL_2024],
    Makam.BELEDIYE_BASKANI,
    province="HAKKARI",
    level="ilce",
  )
  aligned = election_results(
    [Secim.YEREL_2014, Secim.YEREL_2024],
    Makam.BELEDIYE_BASKANI,
    province="HAKKARI",
    level="ilce",
    align_historical_divisions=True,
  )

  assert ("HAKKARI", "DERECIK") in current.index
  assert ("HAKKARI", "DERECIK") in unaligned.index
  assert ("HAKKARI", "DERECIK") not in aligned.index

  expected_secmen = cast(
    "int",
    current.loc[("HAKKARI", "SEMDINLI"), "secmen"],
  ) + cast(
    "int",
    current.loc[("HAKKARI", "DERECIK"), "secmen"],
  )
  assert (
    aligned.loc[("HAKKARI", "SEMDINLI"), (str(Secim.YEREL_2024), "secmen")]
    == expected_secmen
  )


def test_result_tooltip_formats_top_choices() -> None:
  hover = result_tooltip(topk=2)
  row = _guclukonak_row()
  text = hover(row)

  assert text.startswith("<b>Guclukonak, ")
  assert "Seçmen:" in text
  assert "Katılım:" in text  # noqa: RUF001
  assert "Oy kullanan:" not in text
  assert "Geçerli oy:" in text
  assert "Büyük Birlik Partisi: 825" in text
  assert "AK Parti: 750" in text
  assert "chp" not in text
  assert row["buyuk_birlik"] == 825


def test_result_tooltip_filters_by_threshold() -> None:
  row = _guclukonak_row()

  text = result_tooltip(threshold=0.2)(row)

  assert "Büyük Birlik Partisi: 825" in text
  assert "AK Parti: 750" in text
  assert "dem_parti" not in text


def test_result_tooltip_prefers_turkish_boundary_names() -> None:
  row = pd.Series({
    "adm1_tr": "ŞIRNAK",
    "adm2_tr": "GÜÇLÜKONAK",
    "secmen": 100,
    "gecerli": 80,
    "iyi_parti": 7,
    "huda_par": 6,
    "bagimsiz_toplam_oy": 5,
  })

  text = result_tooltip(topk=3)(row)

  assert text.startswith("<b>Güçlükonak, Şırnak</b>")  # noqa: RUF001
  assert "İYİ Parti: 7" in text
  assert "HÜDA PAR: 6" in text
  assert "Bağımsız: 5" in text  # noqa: RUF001


def test_result_tooltip_uses_configured_province_level() -> None:
  row = pd.Series({
    "adm1_tr": "ŞIRNAK",
    "adm2_tr": "GÜÇLÜKONAK",
    "ilce": "GUCLU KONAK",
    "secmen": 100,
    "gecerli": 80,
    "chp": 7,
  })

  text = result_tooltip(topk=1, level="il")(row)

  assert text.startswith("<b>Şırnak</b>")  # noqa: RUF001
  assert "Guclu" not in text


def test_result_tooltip_uses_configured_district_level() -> None:
  row = pd.Series({
    "adm1_tr": "ŞIRNAK",
    "adm2_tr": "GÜÇLÜKONAK",
    "secmen": 100,
    "gecerli": 80,
    "chp": 7,
  })

  text = result_tooltip(topk=1, level="ilce")(row)

  assert text.startswith("<b>Güçlükonak, Şırnak</b>")  # noqa: RUF001


def test_result_tooltip_requires_one_truncation_mode() -> None:
  with pytest.raises(ValueError, match="exactly one"):
    result_tooltip()
  with pytest.raises(ValueError, match="exactly one"):
    result_tooltip(topk=3, threshold=0.1)
