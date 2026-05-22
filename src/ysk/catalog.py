from __future__ import annotations

from dataclasses import dataclass
import enum


class SecimTuru(enum.StrEnum):
  """Election type identifiers used by the YSK catalog."""

  MAYOR = "MAYOR"
  MUNICIPAL_COUNCIL = "MUNICIPAL_COUNCIL"
  PROVINCIAL_COUNCIL = "PROVINCIAL_COUNCIL"
  METROPOLITAN_MAYOR = "METROPOLITAN_MAYOR"
  REFERENDUM = "REF"
  PARLIAMENT = "MV"
  PRESIDENT = "CB"


@dataclass(frozen=True, slots=True)
class SecimKaydi:
  """Metadata for a YSK election endpoint."""

  slug: str
  ysk_id: int
  kind: SecimTuru
  label: str
  secim_turu: int


SECIMLER: tuple[SecimKaydi, ...] = (
  SecimKaydi(
    "2009-03-29-MAYOR", 4290, SecimTuru.MAYOR, "29 Mart 2009 Belediye Baskanligi", 2
  ),
  SecimKaydi(
    "2009-03-29-MUNICIPAL-COUNCIL",
    4290,
    SecimTuru.MUNICIPAL_COUNCIL,
    "29 Mart 2009 Belediye Meclisi Uyeligi",
    3,
  ),
  SecimKaydi(
    "2009-03-29-PROVINCIAL-COUNCIL",
    4290,
    SecimTuru.PROVINCIAL_COUNCIL,
    "29 Mart 2009 Il Genel Meclisi Uyeligi",
    4,
  ),
  SecimKaydi(
    "2009-03-29-METROPOLITAN-MAYOR",
    4290,
    SecimTuru.METROPOLITAN_MAYOR,
    "29 Mart 2009 Buyuksehir Belediye Baskanligi",
    6,
  ),
  SecimKaydi("2010-09-12-REF", 6522, SecimTuru.REFERENDUM, "2010 Halkoylamasi", 7),
  SecimKaydi(
    "2011-06-12-MV",
    7468,
    SecimTuru.PARLIAMENT,
    "24. Donem Milletvekili Genel Secimi",
    8,
  ),
  SecimKaydi(
    "2014-03-30-MAYOR", 11979, SecimTuru.MAYOR, "30 Mart 2014 Belediye Baskanligi", 2
  ),
  SecimKaydi(
    "2014-03-30-MUNICIPAL-COUNCIL",
    11979,
    SecimTuru.MUNICIPAL_COUNCIL,
    "30 Mart 2014 Belediye Meclisi Uyeligi",
    3,
  ),
  SecimKaydi(
    "2014-03-30-PROVINCIAL-COUNCIL",
    11979,
    SecimTuru.PROVINCIAL_COUNCIL,
    "30 Mart 2014 Il Genel Meclisi Uyeligi",
    4,
  ),
  SecimKaydi(
    "2014-03-30-METROPOLITAN-MAYOR",
    11979,
    SecimTuru.METROPOLITAN_MAYOR,
    "30 Mart 2014 Buyuksehir Belediye Baskanligi",
    6,
  ),
  SecimKaydi(
    "2014-08-10-CB", 13340, SecimTuru.PRESIDENT, "Onikinci Cumhurbaskani Secimi", 9
  ),
  SecimKaydi(
    "2015-06-07-MV",
    13884,
    SecimTuru.PARLIAMENT,
    "25. Donem Milletvekili Genel Secimi",
    8,
  ),
  SecimKaydi(
    "2015-11-01-MV",
    14868,
    SecimTuru.PARLIAMENT,
    "26. Donem Milletvekili Genel Secimi",
    8,
  ),
  SecimKaydi("2017-04-16-REF", 15575, SecimTuru.REFERENDUM, "2017 Halkoylamasi", 7),
  SecimKaydi(
    "2018-06-24-CB",
    16300,
    SecimTuru.PRESIDENT,
    "Cumhurbaskani ve 27. Donem Milletvekili Genel Secimi",
    9,
  ),
  SecimKaydi(
    "2018-06-24-MV",
    16300,
    SecimTuru.PARLIAMENT,
    "Cumhurbaskani ve 27. Donem Milletvekili Genel Secimi",
    8,
  ),
  SecimKaydi(
    "2019-03-31-MAYOR", 16400, SecimTuru.MAYOR, "31 Mart 2019 Belediye Baskanligi", 2
  ),
  SecimKaydi(
    "2019-03-31-MUNICIPAL-COUNCIL",
    16400,
    SecimTuru.MUNICIPAL_COUNCIL,
    "31 Mart 2019 Belediye Meclisi Uyeligi",
    3,
  ),
  SecimKaydi(
    "2019-03-31-PROVINCIAL-COUNCIL",
    16400,
    SecimTuru.PROVINCIAL_COUNCIL,
    "31 Mart 2019 Il Genel Meclisi Uyeligi",
    4,
  ),
  SecimKaydi(
    "2019-03-31-METROPOLITAN-MAYOR",
    16400,
    SecimTuru.METROPOLITAN_MAYOR,
    "31 Mart 2019 Buyuksehir Belediye Baskanligi",
    6,
  ),
  SecimKaydi(
    "2019-06-23-METROPOLITAN-MAYOR",
    16643,
    SecimTuru.METROPOLITAN_MAYOR,
    "Istanbul Buyuksehir Belediye Baskanligi Yenileme Secimi",
    6,
  ),
  SecimKaydi(
    "2023-05-14-CB",
    20230,
    SecimTuru.PRESIDENT,
    "Cumhurbaskani ve 28. Donem Milletvekili Genel Secimi",
    9,
  ),
  SecimKaydi(
    "2023-05-14-MV",
    20230,
    SecimTuru.PARLIAMENT,
    "Cumhurbaskani ve 28. Donem Milletvekili Genel Secimi",
    8,
  ),
  SecimKaydi(
    "2023-05-28-CB",
    20240,
    SecimTuru.PRESIDENT,
    "Cumhurbaskani Seciminin Ikinci Oylamasi",
    9,
  ),
  SecimKaydi(
    "2024-03-31-MAYOR", 20260, SecimTuru.MAYOR, "31 Mart 2024 Belediye Baskanligi", 2
  ),
  SecimKaydi(
    "2024-03-31-MUNICIPAL-COUNCIL",
    20260,
    SecimTuru.MUNICIPAL_COUNCIL,
    "31 Mart 2024 Belediye Meclisi Uyeligi",
    3,
  ),
  SecimKaydi(
    "2024-03-31-PROVINCIAL-COUNCIL",
    20260,
    SecimTuru.PROVINCIAL_COUNCIL,
    "31 Mart 2024 Il Genel Meclisi Uyeligi",
    4,
  ),
  SecimKaydi(
    "2024-03-31-METROPOLITAN-MAYOR",
    20260,
    SecimTuru.METROPOLITAN_MAYOR,
    "31 Mart 2024 Buyuksehir Belediye Baskanligi",
    6,
  ),
  SecimKaydi(
    "2024-06-02-MAYOR",
    20273,
    SecimTuru.MAYOR,
    "2 Haziran 2024 Yenileme Secimi Belediye Baskanligi",
    2,
  ),
  SecimKaydi(
    "2024-06-02-MUNICIPAL-COUNCIL",
    20273,
    SecimTuru.MUNICIPAL_COUNCIL,
    "2 Haziran 2024 Yenileme Secimi Belediye Meclisi Uyeligi",
    3,
  ),
)


def by_slug(slug: str) -> SecimKaydi:
  """Return the catalog entry for ``slug``.

  Args:
    slug: Election slug to look up.

  Returns:
    The matching election catalog entry.

  Raises:
    ValueError: If ``slug`` is not known.
  """

  for secim in SECIMLER:
    if secim.slug == slug:
      return secim
  known = ", ".join(secim.slug for secim in SECIMLER)
  msg = f"Unknown election {slug!r}. Known elections: {known}"
  raise ValueError(msg)
