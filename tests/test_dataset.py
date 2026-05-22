from pathlib import Path

import pandas as pd
import xarray as xr

from ysk.dataset import (
  build_dataset,
  build_dataset_from_tables,
  build_demographic_dataset,
  build_district_analysis_dataset,
  build_place_analysis_dataset,
  dataset_with_broadcast_demographics,
  district_vote_demographic_frame,
  to_turkish_schema,
  write_partitioned_dataset,
)
from ysk.schema import slugify
from ysk.tables import NormalizedTables, write_partitioned_tables
from ysk.xarray_typing import (
  data_array_isnull,
  data_array_item,
  open_zarr,
  sel_data_array,
  swap_dims,
)


def _item(array: xr.DataArray, **indexers: object) -> object:
  selected = sel_data_array(array, **indexers) if indexers else array
  return data_array_item(selected)


def _isnull_item(array: xr.DataArray, **indexers: object) -> object:
  selected = sel_data_array(array, **indexers) if indexers else array
  return data_array_item(data_array_isnull(selected))


def test_slugify_turkish_text() -> None:
  assert slugify("RECEP TAYYIP ERDOGAN") == "recep_tayyip_erdogan"
  assert slugify("KILICDAROGLU") == "kilicdaroglu"


def test_build_dataset_groups_choices_by_observation() -> None:
  frame = pd.DataFrame(
    [
      {
        "election": "2023-05-14-CB",
        "election_id": 20230,
        "election_kind": "CB",
        "scope": "domestic_ballot_box",
        "province": "adana",
        "district": "aladag",
        "ballot_box_no": 1001,
        "registered_voters": 200,
        "choice": "rte",
        "votes": 80,
      },
      {
        "election": "2023-05-14-CB",
        "election_id": 20230,
        "election_kind": "CB",
        "scope": "domestic_ballot_box",
        "province": "adana",
        "district": "aladag",
        "ballot_box_no": 1001,
        "registered_voters": 200,
        "choice": "kilicdaroglu",
        "votes": 100,
      },
    ],
  )

  dataset = build_dataset(frame)

  assert dataset.sizes["observation"] == 1
  assert dataset.sizes["choice"] == 2
  assert _item(dataset["votes"], choice="rte") == 80
  assert _item(dataset["votes"], choice="kilicdaroglu") == 100


def test_build_dataset_from_normalized_tables() -> None:
  tables = NormalizedTables(
    observations=pd.DataFrame(
      [
        {
          "election": "2023-05-14-CB",
          "election_id": 20230,
          "election_kind": "CB",
          "scope": "domestic_ballot_box",
          "observation_id": "obs-1",
          "province": "adana",
          "district": "aladag",
          "ballot_box_no": 1001,
          "registered_voters": 200,
        },
      ],
    ),
    votes=pd.DataFrame(
      [
        {
          "observation_id": "obs-1",
          "election": "2023-05-14-CB",
          "scope": "domestic_ballot_box",
          "choice_column": "aday1_ALDIGI_OY",
          "choice": "rte",
          "votes": 80,
        },
      ],
    ),
    choices=pd.DataFrame(
      [
        {
          "election": "2023-05-14-CB",
          "choice_column": "aday1_ALDIGI_OY",
          "choice": "rte",
          "choice_type": "candidate",
          "choice_label_raw": "RECEP TAYYIP ERDOGAN",
        },
      ],
    ),
    geography=pd.DataFrame(),
    demographics=pd.DataFrame(),
  )

  dataset = build_dataset_from_tables(tables)

  assert dataset.sizes["observation"] == 1
  assert dataset.sizes["choice"] == 1
  assert dataset["registered_voters"].item() == 200
  assert _item(dataset["choice_type"], choice="rte") == "candidate"


def test_write_partitioned_dataset_appends_observations(tmp_path: Path) -> None:
  base = tmp_path / "tables"
  write_partitioned_tables(
    NormalizedTables(
      observations=pd.DataFrame(
        [
          {
            "election": "e1",
            "election_id": 1,
            "election_kind": "CB",
            "scope": "domestic_district",
            "observation_id": "e1-obs",
            "registered_voters": 100,
          },
        ],
      ),
      votes=pd.DataFrame(
        [
          {
            "observation_id": "e1-obs",
            "choice": "a",
            "votes": 60,
          },
        ],
      ),
      choices=pd.DataFrame(
        [
          {
            "choice": "a",
            "choice_type": "candidate",
            "choice_label_raw": "A",
          },
        ],
      ),
      geography=pd.DataFrame(),
      demographics=pd.DataFrame(),
    ),
    base,
    "e1",
  )
  write_partitioned_tables(
    NormalizedTables(
      observations=pd.DataFrame(
        [
          {
            "election": "e2",
            "election_id": 2,
            "election_kind": "MV",
            "scope": "domestic_district",
            "observation_id": "e2-obs",
            "registered_voters": 120,
          },
        ],
      ),
      votes=pd.DataFrame(
        [
          {
            "observation_id": "e2-obs",
            "choice": "b",
            "votes": 70,
          },
        ],
      ),
      choices=pd.DataFrame(
        [
          {
            "choice": "b",
            "choice_type": "party",
            "choice_label_raw": "B",
          },
        ],
      ),
      geography=pd.DataFrame(),
      demographics=pd.DataFrame(),
    ),
    base,
    "e2",
  )

  write_partitioned_dataset(base, tmp_path / "ysk.zarr")
  dataset = open_zarr(tmp_path / "ysk.zarr")

  assert dataset.sizes["gozlem"] == 2
  assert dataset.sizes["tercih"] == 2
  assert _item(dataset["oy"], gozlem="e1-obs", tercih="a") == 60
  assert _isnull_item(dataset["oy"], gozlem="e1-obs", tercih="b")
  assert _item(dataset["oy"], gozlem="e2-obs", tercih="b") == 70


def test_to_turkish_schema_renames_axes_and_variables() -> None:
  frame = pd.DataFrame(
    [
      {
        "election": "2023-05-14-CB",
        "election_id": 20230,
        "election_kind": "CB",
        "scope": "domestic_ballot_box",
        "province": "ANKARA",
        "district": "CANKAYA",
        "ballot_box_no": 1001,
        "choice": "rte",
        "votes": 80,
      },
      {
        "election": "2023-05-14-CB",
        "election_id": 20230,
        "election_kind": "CB",
        "scope": "domestic_ballot_box",
        "province": "ANKARA",
        "district": "CANKAYA",
        "ballot_box_no": 1001,
        "choice": "kilicdaroglu",
        "votes": 100,
      },
      {
        "election": "2023-05-14-CB",
        "election_id": 20230,
        "election_kind": "CB",
        "scope": "domestic_ballot_box",
        "province": "ISTANBUL",
        "district": "KADIKOY",
        "ballot_box_no": 2001,
        "choice": "kilicdaroglu",
        "votes": 120,
      },
    ],
  )
  dataset = build_dataset(frame)
  renamed = to_turkish_schema(dataset)

  assert "gozlem" in renamed.sizes
  assert "tercih" in renamed.sizes
  assert "secim" in renamed.coords
  assert "il" in renamed.coords
  assert "oy" in renamed
  assert data_array_item(sel_data_array(renamed["oy"], tercih="rte").sum()) == 80


def test_build_demographic_dataset_uses_turkish_schema(tmp_path: Path) -> None:
  partition = tmp_path / "election=e1"
  partition.mkdir(parents=True)
  pd.DataFrame(
    [
      {
        "election": "e1",
        "election_id": 1,
        "election_kind": "CB",
        "endpoint": "getKadinErkekCinsiyetOraniIlGroupIlce",
        "scope": "gender_district",
        "province": "ANKARA",
        "province_id": 6,
        "district": "CANKAYA",
        "district_id": 123,
        "yas_GRUBU": "18-24",
        "kadin": 10,
        "erkek": 11,
        "toplam": 21,
        "kadin_ORAN": 47.6,
        "erkek_ORAN": 52.4,
      },
    ],
  ).to_parquet(partition / "demographics.parquet", index=False)

  dataset = build_demographic_dataset(tmp_path)

  assert dataset.sizes["demografi"] == 1
  assert dataset["secim"].item() == "e1"
  assert dataset["kapsam"].item() == "cinsiyet_ilce"
  assert dataset["il"].item() == "ANKARA"
  assert dataset["yas_grubu"].item() == "18-24"
  assert dataset["kadin"].item() == 10


def test_district_vote_demographic_frame_joins_by_district() -> None:
  sonuc = xr.Dataset(
    coords={
      "gozlem": ("gozlem", ["g1", "g2"]),
      "tercih": ("tercih", ["parti_a", "parti_b"]),
      "secim": ("gozlem", ["e1", "e1"]),
      "kapsam": ("gozlem", ["yurtici_sandik", "yurtici_sandik"]),
      "il": ("gozlem", ["ANKARA", "ANKARA"]),
      "ilce": ("gozlem", ["CANKAYA", "CANKAYA"]),
      "il_id": ("gozlem", [6, 6]),
      "ilce_id": ("gozlem", [123, 123]),
    },
    data_vars={
      "oy": (("gozlem", "tercih"), [[10.0, 20.0], [5.0, 7.0]]),
    },
  )
  demografi = xr.Dataset(
    coords={
      "demografi": ("demografi", [0]),
      "secim": ("demografi", ["e1"]),
      "kapsam": ("demografi", ["cinsiyet_ilce"]),
      "il": ("demografi", ["ANKARA"]),
      "ilce": ("demografi", ["CANKAYA"]),
      "il_id": ("demografi", [6]),
      "ilce_id": ("demografi", [123]),
    },
    data_vars={
      "kadin": ("demografi", [100.0]),
      "erkek": ("demografi", [90.0]),
      "toplam": ("demografi", [190.0]),
      "kadin_oran": ("demografi", [52.6]),
      "erkek_oran": ("demografi", [47.4]),
    },
  )

  frame = district_vote_demographic_frame(sonuc, demografi, secim="e1")

  assert len(frame) == 1
  assert frame["oy_parti_a"].item() == 15
  assert frame["oy_parti_b"].item() == 27
  assert frame["kadin"].item() == 100


def test_build_district_analysis_dataset_shapes_for_sel(tmp_path: Path) -> None:
  base = tmp_path / "tables"
  write_partitioned_tables(
    NormalizedTables(
      observations=pd.DataFrame(
        [
          {
            "observation_id": "obs-1",
            "election": "e1",
            "election_id": 1,
            "election_kind": "MV",
            "scope": "domestic_district",
            "province_id": 6,
            "province": "ANKARA",
            "district_id": 123,
            "district": "CANKAYA",
          },
        ],
      ),
      votes=pd.DataFrame(
        [
          {
            "observation_id": "obs-1",
            "scope": "domestic_district",
            "choice": "parti_a",
            "votes": 10,
          },
          {
            "observation_id": "obs-1",
            "scope": "domestic_district",
            "choice": "parti_b",
            "votes": 20,
          },
        ],
      ),
      choices=pd.DataFrame(),
      geography=pd.DataFrame(),
      demographics=pd.DataFrame(
        [
          {
            "election": "e1",
            "election_id": 1,
            "election_kind": "MV",
            "endpoint": "getKadinErkekCinsiyetOraniIlGroupIlce",
            "scope": "gender_district",
            "district_id": 123,
            "kadin": 100,
            "erkek": 90,
            "toplam": 190,
            "kadin_ORAN": 52.6,
            "erkek_ORAN": 47.4,
          },
        ],
      ),
    ),
    base,
    "e1",
  )

  dataset = build_district_analysis_dataset(base)
  selected = dataset.sel(secim="e1")

  assert dataset.sizes["secim"] == 1
  assert dataset.sizes["ilce_id"] == 1
  assert dataset.sizes["tercih"] == 2
  assert _item(selected["oy"], ilce_id=123, tercih="parti_a") == 10
  assert _item(selected["kadin"], ilce_id=123) == 100


def test_build_place_analysis_dataset_keeps_sandik_rows(tmp_path: Path) -> None:
  base = tmp_path / "tables"
  write_partitioned_tables(
    NormalizedTables(
      observations=pd.DataFrame(
        [
          {
            "observation_id": "obs-1",
            "election": "e1",
            "election_id": 1,
            "election_kind": "MV",
            "scope": "domestic_ballot_box",
            "province_id": 6,
            "province": "ANKARA",
            "district_id": 123,
            "district": "CANKAYA",
            "ballot_box_no": 1001,
            "registered_voters": 200,
          },
        ],
      ),
      votes=pd.DataFrame(
        [
          {"observation_id": "obs-1", "choice": "parti_a", "votes": 10},
          {"observation_id": "obs-1", "choice": "parti_b", "votes": 20},
        ],
      ),
      choices=pd.DataFrame(
        [
          {"choice": "parti_a"},
          {"choice": "parti_b"},
        ],
      ),
      geography=pd.DataFrame(),
      demographics=pd.DataFrame(
        [
          {
            "election": "e1",
            "scope": "gender_district",
            "district_id": 123,
            "kadin": 100,
            "erkek": 90,
            "toplam": 190,
            "kadin_ORAN": 52.6,
            "erkek_ORAN": 47.4,
          },
        ],
      ),
    ),
    base,
    "e1",
  )

  dataset = build_place_analysis_dataset(base).sel(secim="e1")

  assert dataset.sizes["yer"] == 1
  assert dataset.sizes["tercih"] == 2
  assert _item(dataset["oy"], tercih="parti_a") == 10
  assert dataset["kayitli_secmen"].item() == 200
  assert dataset["kadin"].item() == 100
  assert dataset["sandik_no"].item() == 1001


def test_dataset_with_broadcast_demographics_supports_secim_sel() -> None:
  sonuc = xr.Dataset(
    coords={
      "gozlem": ("gozlem", ["g1", "g2"]),
      "tercih": ("tercih", ["parti_a"]),
      "secim": ("gozlem", ["e1", "e2"]),
      "ilce_id": ("gozlem", [123, 123]),
    },
    data_vars={"oy": (("gozlem", "tercih"), [[10.0], [20.0]])},
  )
  demografi = xr.Dataset(
    coords={
      "demografi": ("demografi", [0, 1]),
      "secim": ("demografi", ["e1", "e2"]),
      "kapsam": ("demografi", ["cinsiyet_ilce", "cinsiyet_ilce"]),
      "ilce_id": ("demografi", [123, 123]),
    },
    data_vars={
      "kadin": ("demografi", [100.0, 120.0]),
      "erkek": ("demografi", [90.0, 110.0]),
      "toplam": ("demografi", [190.0, 230.0]),
      "kadin_oran": ("demografi", [52.6, 52.2]),
      "erkek_oran": ("demografi", [47.4, 47.8]),
    },
  )

  dataset = swap_dims(
    dataset_with_broadcast_demographics(sonuc, demografi),
    {"gozlem": "secim"},
  )

  selected = dataset.sel(secim="e2")
  assert _item(selected["oy"], tercih="parti_a") == 20
  assert selected["kadin"].item() == 120
