from __future__ import annotations

import argparse
from pathlib import Path

from turkiye import plot_interactive

from ysk import Makam, Secim, election_results


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "output",
    nargs="?",
    type=Path,
    default=Path("tmp/ilce_chp_orani.jpg"),
  )
  args = parser.parse_args()

  oy = election_results(Secim.YEREL_2024, Makam.BELEDIYE_BASKANI, level="ilce")
  oy["chp_orani"] = oy["chp"] / oy["gecerli"]
  fig = plot_interactive(
    oy,
    color="chp_orani",
    tooltip_fn=lambda row: (
      f"<b>{row['ilce']}, {row['il']}</b><br>"
      f"CHP orani: {row['chp_orani']:.1%}<br>"
      f"Secmen: {row['secmen']:,.0f}<br>"
      f"CHP: {row['chp']:,.0f}"
    ),
    width=1400,
    height=800,
  )
  fig.update_layout(title=f"{Secim.YEREL_2024} {Makam.BELEDIYE_BASKANI}")

  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.write_image(args.output, format=args.output.suffix.removeprefix(".") or "jpg")


if __name__ == "__main__":
  main()
