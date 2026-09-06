"""Axis units appended to chart labels.

The rule lives outside the UI so it can be checked without a Streamlit runtime,
and so every chart in the app formats units the same way.
"""

from __future__ import annotations

import pytest

from modules.units import (
    AXIS_UNITS,
    CUSTOM_UNIT,
    NO_UNIT,
    format_axis_label,
    resolve_unit,
    suggest_unit,
    unit_index,
)

MICROGRAMS = "µg/m³"
CELSIUS = "°C"


class TestUnitCatalogue:
    def test_the_units_this_app_is_for_are_offered(self):
        for unit in (MICROGRAMS, "mg/m³", "ppb", "ppm", CELSIUS, "%", "hPa"):
            assert unit in AXIS_UNITS

    def test_no_unit_is_first_so_nothing_is_asserted_by_default(self):
        assert AXIS_UNITS[0] == NO_UNIT

    def test_custom_is_offered_last(self):
        assert AXIS_UNITS[-1] == CUSTOM_UNIT

    def test_entries_are_unique(self):
        assert len(set(AXIS_UNITS)) == len(AXIS_UNITS)

    def test_the_micro_sign_is_the_micro_sign_not_a_greek_mu(self):
        """Both render alike; only one is the SI character."""
        assert MICROGRAMS in AXIS_UNITS
        assert "μg/m³" not in AXIS_UNITS


class TestResolveUnit:
    def test_a_plain_choice_is_returned(self):
        assert resolve_unit(MICROGRAMS) == MICROGRAMS

    @pytest.mark.parametrize("choice", [None, "", NO_UNIT])
    def test_no_unit_resolves_to_nothing(self, choice):
        assert resolve_unit(choice) == ""

    def test_custom_uses_the_typed_text(self):
        assert resolve_unit(CUSTOM_UNIT, "grains/gallon") == "grains/gallon"

    def test_custom_left_blank_resolves_to_nothing(self):
        assert resolve_unit(CUSTOM_UNIT, "   ") == ""

    def test_surrounding_whitespace_is_dropped(self):
        assert resolve_unit(CUSTOM_UNIT, "  ppb  ") == "ppb"


class TestFormatAxisLabel:
    def test_the_unit_goes_in_parentheses(self):
        assert format_axis_label("Actual", MICROGRAMS) == f"Actual ({MICROGRAMS})"

    def test_no_unit_leaves_the_label_alone(self):
        assert format_axis_label("Actual", "") == "Actual"

    def test_a_unit_with_no_label_stands_alone(self):
        assert format_axis_label("", CELSIUS) == CELSIUS

    def test_an_already_suffixed_label_is_not_doubled(self):
        """Typing the unit and also picking it should not repeat it."""
        already = f"Actual ({MICROGRAMS})"
        assert format_axis_label(already, MICROGRAMS) == already

    def test_a_label_that_is_the_unit_is_not_doubled(self):
        assert format_axis_label("Count", "count") == "Count"
        assert format_axis_label("count", "count") == "count"

    def test_a_different_unit_is_still_appended(self):
        assert format_axis_label(f"Actual ({CELSIUS})", MICROGRAMS) == (
            f"Actual ({CELSIUS}) ({MICROGRAMS})"
        )

    def test_whitespace_is_tidied(self):
        assert format_axis_label("  Actual  ", f"  {MICROGRAMS} ") == f"Actual ({MICROGRAMS})"

    def test_empty_label_and_empty_unit_give_empty_text(self):
        assert format_axis_label("", "") == ""

    def test_none_is_tolerated_for_either_side(self):
        assert format_axis_label(None, None) == ""
        assert format_axis_label(None, CELSIUS) == CELSIUS

    @pytest.mark.parametrize("unit", [u for u in AXIS_UNITS if u not in (NO_UNIT, CUSTOM_UNIT)])
    def test_every_offered_unit_formats(self, unit):
        assert format_axis_label("Value", unit) == f"Value ({unit})"


class TestPickerToAxisText:
    """The path the expander takes: dropdown choice -> text drawn on the axis."""

    def test_choosing_micrograms_labels_the_axis(self):
        unit = resolve_unit(MICROGRAMS)
        assert format_axis_label("Predicted", unit) == f"Predicted ({MICROGRAMS})"

    def test_choosing_none_leaves_the_axis_as_typed(self):
        assert format_axis_label("Predicted", resolve_unit(NO_UNIT)) == "Predicted"

    def test_a_custom_unit_reaches_the_axis(self):
        unit = resolve_unit(CUSTOM_UNIT, "mol/m²")
        assert format_axis_label("Column density", unit) == "Column density (mol/m²)"


class TestSuggestUnit:
    """Suggestions seed the picker; they never overrule what the user chose."""

    @pytest.mark.parametrize("column,expected", [
        ("pm25", MICROGRAMS),
        ("pm2.5", MICROGRAMS),
        ("pm10", MICROGRAMS),
        ("pm1", MICROGRAMS),
        ("Concentration", MICROGRAMS),
        ("temperature", CELSIUS),
        ("humidity", "%"),
        ("pressure", "hPa"),
        ("no2", "ppb"),
        ("o3", "ppb"),
        ("co", "ppm"),
        ("voc", "ppb"),
        ("wind_speed", "m/s"),
        ("rainfall", "mm"),
        ("solar_radiation", "W/m²"),
    ])
    def test_common_channels_are_recognised(self, column, expected):
        assert suggest_unit(column) == expected

    @pytest.mark.parametrize("column", [
        "reference_pm25", "sensor_pm25_raw", "reference_reference_temperature",
        "sensor_lcs_pm25", "SENSOR_PM25",
    ])
    def test_pipeline_prefixes_do_not_hide_the_channel(self, column):
        """Alignment prefixes columns, sometimes twice over."""
        assert suggest_unit(column) in (MICROGRAMS, CELSIUS)

    @pytest.mark.parametrize("text", ["", None, "timestamp", "index", "Actual (Reference)"])
    def test_nothing_recognisable_suggests_nothing(self, text):
        assert suggest_unit(text) == ""

    @pytest.mark.parametrize("label", ["Count", "Frequency"])
    def test_tally_axes_are_left_alone(self, label):
        """Suggesting "count" for a "Count" axis produced "Count (count)"."""
        assert suggest_unit(label) == ""

    def test_a_short_token_is_not_matched_inside_a_word(self):
        """"co" sits inside "concentration"; the wrong hit would be ppm."""
        assert suggest_unit("concentration") == MICROGRAMS

    def test_a_short_token_still_matches_on_its_own(self):
        assert suggest_unit("sensor_co") == "ppm"

    def test_pm25_wins_over_a_bare_pm1_prefix(self):
        assert suggest_unit("pm10") == MICROGRAMS


class TestUnitIndex:
    def test_a_suggestion_maps_to_its_position(self):
        assert AXIS_UNITS[unit_index(MICROGRAMS)] == MICROGRAMS

    def test_no_suggestion_lands_on_none(self):
        assert unit_index("") == 0
        assert AXIS_UNITS[unit_index("")] == NO_UNIT

    def test_an_unlisted_unit_falls_back_to_none(self):
        assert unit_index("furlongs") == 0

    def test_every_suggestion_the_matcher_can_make_is_offered(self):
        """A suggestion outside AXIS_UNITS would silently seed the picker to None."""
        from modules.units import _SUBSTRING_UNITS, _TOKEN_UNITS

        suggested = {u for _, u in _SUBSTRING_UNITS} | set(_TOKEN_UNITS.values())
        missing = sorted(u for u in suggested if u not in AXIS_UNITS)
        assert not missing, f"suggested but not offered: {missing}"


class TestSuggestionThroughToTheAxis:
    def test_a_pm_column_labels_the_axis_in_micrograms(self):
        unit = resolve_unit(AXIS_UNITS[unit_index(suggest_unit("reference_pm25"))])
        assert format_axis_label("Actual", unit) == f"Actual ({MICROGRAMS})"

    def test_an_unrecognised_column_leaves_the_axis_bare(self):
        unit = resolve_unit(AXIS_UNITS[unit_index(suggest_unit("mystery"))])
        assert format_axis_label("Actual", unit) == "Actual"
