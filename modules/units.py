"""Axis units for chart labels.

Axis text is assembled here rather than in the UI so the rule that produces
"Actual (µg/m³)" from a label and a unit can be tested without a Streamlit
runtime, and so every chart in the app renders units the same way.

A unit is suggested from the axis label or the column behind it, and is only
ever a starting point: the picker stays editable, and the suggestion seeds it
rather than overriding a choice already made. Nothing in the pipeline verifies
what units a file is in, so a suggestion is a convenience, not a claim.
"""

from __future__ import annotations

import re
from typing import Optional

NO_UNIT = "None"
CUSTOM_UNIT = "Custom…"

#: Offered in this order: mass concentrations first, since calibration targets
#: are reported that way, then mixing ratios, then the meteorological channels
#: that ride along in co-location datasets.
AXIS_UNITS = (
    NO_UNIT,
    "µg/m³",
    "mg/m³",
    "ng/m³",
    "ppm",
    "ppb",
    "ppt",
    "°C",
    "°F",
    "K",
    "°",
    "%",
    "hPa",
    "kPa",
    "mbar",
    "m/s",
    "km/h",
    "mm",
    "W/m²",
    "particles/cm³",
    "AQI",
    "count",
    "ratio",
    CUSTOM_UNIT,
)


def resolve_unit(choice: Optional[str], custom_text: str = "") -> str:
    """Turn a dropdown choice plus its free-text box into a unit string.

    Returns "" when no unit was chosen, including the case where the user
    picked "Custom" and then left the box empty.
    """
    if not choice or choice == NO_UNIT:
        return ""
    if choice == CUSTOM_UNIT:
        return custom_text.strip()
    return choice.strip()


def format_axis_label(label: str, unit: str = "") -> str:
    """Combine an axis label and a unit into the text drawn on the axis.

    A label that already carries the unit in parentheses is left alone, so
    typing "Actual (µg/m³)" and also picking the unit does not produce
    "Actual (µg/m³) (µg/m³)". A label that *is* the unit is left alone too,
    which is what stops a histogram's "Count" axis reading "Count (count)".
    """
    label = (label or "").strip()
    unit = (unit or "").strip()
    if not unit:
        return label
    if not label:
        return unit
    if label.endswith(f"({unit})"):
        return label
    if label.lower() == unit.lower():
        return label
    return f"{label} ({unit})"


# Matched against whole tokens, because these are short enough to appear inside
# unrelated words -- "co" sits in "concentration", "ws" in "raws".
_TOKEN_UNITS = {
    "co": "ppm",
    "co2": "ppm",
    "o3": "ppb",
    "no": "ppb",
    "no2": "ppb",
    "nox": "ppb",
    "so2": "ppb",
    "nh3": "ppb",
    "rh": "%",
    "ws": "m/s",
    "wd": "°",
    "aqi": "AQI",
    "t": "°C",
}

# Matched anywhere in the label. Ordered: the first hit wins, so anything that
# is a prefix of another entry comes after it.
_SUBSTRING_UNITS = (
    ("pm2.5", "µg/m³"),
    ("pm25", "µg/m³"),
    ("pm10", "µg/m³"),
    ("pm1", "µg/m³"),
    ("particulate", "µg/m³"),
    ("concentration", "µg/m³"),
    ("temperature", "°C"),
    ("temp", "°C"),
    ("humidity", "%"),
    ("dewpoint", "°C"),
    ("pressure", "hPa"),
    ("windspeed", "m/s"),
    ("wind", "m/s"),
    ("rainfall", "mm"),
    ("rain", "mm"),
    ("precipitation", "mm"),
    ("irradiance", "W/m²"),
    ("radiation", "W/m²"),
    ("solar", "W/m²"),
    ("voc", "ppb"),
    ("ozone", "ppb"),
)


def suggest_unit(text: Optional[str]) -> str:
    """Suggest a unit for an axis from its label or source column name.

    Returns "" when nothing matches, which leaves the picker on "None" rather
    than inventing a unit. Prefixes the pipeline adds ("reference_pm25",
    "sensor_pm25_raw") do not get in the way, since matching is on tokens and
    substrings of the lowercased name.
    """
    if not text:
        return ""
    lowered = text.strip().lower()

    for needle, unit in _SUBSTRING_UNITS:
        if needle in lowered:
            return unit

    for token in re.split(r"[^a-z0-9.]+", lowered):
        if token in _TOKEN_UNITS:
            return _TOKEN_UNITS[token]
    return ""


def unit_index(unit: str) -> int:
    """Position of ``unit`` in :data:`AXIS_UNITS`, or 0 (no unit) if absent.

    Used to seed a picker with a suggestion without hard-coding offsets.
    """
    try:
        return AXIS_UNITS.index(unit)
    except ValueError:
        return 0
