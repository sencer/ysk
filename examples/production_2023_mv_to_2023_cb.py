from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

import ysk

EPS = 1e-9
PENALTY_LAMBDA = 0.001

SOURCE_PARTIES = [
  "ak_parti",
  "bbp",
  "chp",
  "cumhur_ittifaki",
  "iyi_parti",
  "memleket",
  "mhp",
  "tip",
  "yeniden_refah",
  "yesil_sol_parti",
  "zafer_partisi",
]
TARGET_CHOICES = [
  "kemal_kilicdaroglu",
  "muharrem_ince",
  "recep_tayyip_erdogan",
  "sinan_ogan",
]


@dataclass(frozen=True)
class ElectionPair:
  mv: pd.DataFrame
  cb: pd.DataFrame
  source_parties: list[str]
  target_choices: list[str]


def rowsum_to_one(matrix: np.ndarray) -> np.ndarray:
  matrix = np.maximum(matrix, EPS)
  return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), EPS)


def as_logits(transition: np.ndarray) -> np.ndarray:
  probabilities = np.clip(transition, EPS, 1.0)
  log_probabilities = np.log(probabilities)
  return log_probabilities[:, :-1] - log_probabilities[:, [-1]]


def as_transition(
  flat_logits: np.ndarray,
  *,
  n_source: int,
  n_target: int,
) -> np.ndarray:
  logits = flat_logits.reshape(n_source, n_target - 1)
  logits = np.column_stack([logits, np.zeros(n_source)])
  return softmax(logits, axis=1)


def fit_national_transition(mv_votes: np.ndarray, cb_votes: np.ndarray) -> np.ndarray:
  transition = np.full(
    (mv_votes.shape[1], cb_votes.shape[1]),
    1.0 / cb_votes.shape[1],
  )
  mv_votes_t = mv_votes.T

  for _ in range(180):
    prediction = np.maximum(mv_votes @ transition, EPS)
    transition *= (mv_votes_t @ (cb_votes / prediction)) / np.maximum(
      mv_votes.sum(axis=0)[:, None],
      EPS,
    )
    transition = rowsum_to_one(transition)

  return transition


def district_prediction(
  mv_votes: np.ndarray,
  cb_total: np.ndarray,
  transition: np.ndarray,
) -> np.ndarray:
  raw = mv_votes @ transition
  return raw / np.maximum(raw.sum(axis=1, keepdims=True), EPS) * cb_total[:, None]


def province_loss(
  flat_logits: np.ndarray,
  mv_votes: np.ndarray,
  cb_votes: np.ndarray,
  cb_total: np.ndarray,
  national_transition: np.ndarray,
) -> float:
  transition = as_transition(
    flat_logits,
    n_source=national_transition.shape[0],
    n_target=national_transition.shape[1],
  )
  probabilities = district_prediction(mv_votes, cb_total, transition) / np.maximum(
    cb_total[:, None],
    EPS,
  )
  negative_log_likelihood = -float(
    np.sum(cb_votes * np.log(np.clip(probabilities, 1e-12, 1.0)))
  )
  shrinkage = (
    PENALTY_LAMBDA
    * float(cb_total.sum())
    * float(np.square(transition - national_transition).sum())
  )
  return negative_log_likelihood + shrinkage


def add_other_choice(
  table: pd.DataFrame,
  *,
  selected_columns: list[str],
  total_column: str = "gecerli",
) -> tuple[pd.DataFrame, list[str]]:
  table = table.copy()
  table["diger"] = np.maximum(
    table[total_column] - table[selected_columns].sum(axis=1),
    0,
  )
  return table, [*selected_columns, "diger"]


def prepare_pair(mv: pd.DataFrame, cb: pd.DataFrame) -> ElectionPair:
  shared_places = mv.index.intersection(cb.index)
  mv = mv.loc[shared_places]
  cb = cb.loc[shared_places]
  keep = (mv["gecerli"] > 0) & (cb["gecerli"] > 0)
  mv = mv.loc[keep]
  cb = cb.loc[keep]

  mv, source_parties = add_other_choice(mv, selected_columns=SOURCE_PARTIES)
  cb, target_choices = add_other_choice(cb, selected_columns=TARGET_CHOICES)
  return ElectionPair(mv, cb, source_parties, target_choices)


def fit_province_transitions(pair: ElectionPair) -> dict[str, np.ndarray]:
  mv_votes = pair.mv[pair.source_parties].to_numpy(dtype=float)
  cb_votes = pair.cb[pair.target_choices].to_numpy(dtype=float)
  national_transition = fit_national_transition(mv_votes, cb_votes)
  start = as_logits(national_transition).ravel()

  province_transitions: dict[str, np.ndarray] = {}
  province_values = pair.mv.index.get_level_values("il")
  for province in province_values.unique():
    in_province = province_values == province
    province_mv = mv_votes[in_province]
    province_cb = cb_votes[in_province]
    province_cb_total = province_cb.sum(axis=1)

    if len(province_mv) < 3 or province_mv.sum() <= 0:
      province_transitions[str(province)] = national_transition
      continue

    result = minimize(
      province_loss,
      start,
      args=(province_mv, province_cb, province_cb_total, national_transition),
      method="L-BFGS-B",
      options={"maxiter": 500, "ftol": 1e-8, "maxls": 50},
    )
    province_transitions[str(province)] = as_transition(
      result.x,
      n_source=len(pair.source_parties),
      n_target=len(pair.target_choices),
    )

  return province_transitions


def aggregate_flows(
  pair: ElectionPair,
  province_transitions: dict[str, np.ndarray],
) -> pd.DataFrame:
  totals = np.zeros((len(pair.source_parties), len(pair.target_choices)))
  province_values = pair.mv.index.get_level_values("il")

  for province in province_values.unique():
    in_province = province_values == province
    mv_votes = pair.mv.loc[in_province, pair.source_parties].to_numpy(dtype=float)
    cb_votes = pair.cb.loc[in_province, pair.target_choices].to_numpy(dtype=float)
    transition = province_transitions[str(province)]

    local_flows = np.einsum("ds,st->dst", mv_votes, transition)
    scale = cb_votes.sum(axis=1) / np.maximum(local_flows.sum(axis=(1, 2)), EPS)
    totals += (local_flows * scale[:, None, None]).sum(axis=0)

  rows = []
  for source_index, source_party in enumerate(pair.source_parties):
    source_total = totals[source_index].sum()
    for target_index, target_choice in enumerate(pair.target_choices):
      estimated_votes = float(totals[source_index, target_index])
      rows.append({
        "source_party": source_party,
        "target_choice": target_choice,
        "estimated_votes": estimated_votes,
        "source_share": estimated_votes / max(source_total, EPS),
      })

  return pd.DataFrame(rows)


results = ysk.election_results(
  ysk.Secim.GENEL_2023,
  [ysk.Makam.MILLETVEKILI, ysk.Makam.CUMHURBASKANI],
  level="mahalle",
)
mv = results.xs("MV", axis=1, level="makam")
cb = results.xs("CB", axis=1, level="makam")

pair = prepare_pair(mv, cb)
province_transitions = fit_province_transitions(pair)
flows = aggregate_flows(pair, province_transitions)

sys.stdout.write(
  flows.sort_values("estimated_votes", ascending=False).head(20).to_string(index=False)
  + "\n"
)
