"""Legend placement, and keeping the PNG equal to what is on screen.

Regression cover for a mismatch a user hit on the Time-Series Overview: the
chart was narrowed by clicking legend entries, but the downloaded PNG still
carried every series. Legend clicks are browser state; the PNG is rendered
server-side from the Python figure, which never hears about them. The fix is to
leave legend clicks off by default, so the only thing that filters traces is the
variable picker -- which the figure, and therefore the PNG, already follows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.eda import create_time_series_figure
from modules.plots import LEGEND_POSITIONS, apply_legend_layout, create_multi_model_timeseries


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=48, freq="h")
    return pd.DataFrame({
        "timestamp": index,
        "sensor_pm25": range(48),
        "sensor_humidity": range(48, 96),
        "sensor_temperature": range(96, 144),
        "reference_pm25": range(144, 192),
    })


def _figure(position="top", interactive=False):
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], name="a"))
    return apply_legend_layout(fig, position, interactive)


class TestLegendPlacement:
    @pytest.mark.parametrize("position", LEGEND_POSITIONS)
    def test_every_advertised_position_is_accepted(self, position):
        assert _figure(position) is not None

    def test_hidden_removes_the_legend(self):
        assert _figure("hidden").layout.showlegend is False

    @pytest.mark.parametrize("position", ["top", "right", "bottom"])
    def test_other_positions_keep_it(self, position):
        assert _figure(position).layout.showlegend is True

    def test_right_places_the_legend_outside_the_plot(self):
        legend = _figure("right").layout.legend
        assert legend.orientation == "v"
        assert legend.x > 1.0

    def test_right_reserves_margin_so_it_is_not_clipped(self):
        assert _figure("right").layout.margin.r > _figure("top").layout.margin.r

    def test_top_reserves_headroom_so_it_clears_the_title(self):
        """A wrapped horizontal legend colliding with the title was the complaint."""
        assert _figure("top").layout.margin.t > _figure("hidden").layout.margin.t

    def test_bottom_reserves_footroom(self):
        assert _figure("bottom").layout.margin.b > _figure("top").layout.margin.b

    def test_an_unknown_position_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown legend position"):
            _figure("upside-down")
        with pytest.raises(ValueError, match="top, right, bottom, hidden"):
            _figure("upside-down")


class TestLegendClicksAreOffByDefault:
    def test_clicks_do_nothing_unless_asked_for(self):
        legend = _figure("top").layout.legend
        assert legend.itemclick is False
        assert legend.itemdoubleclick is False

    def test_opting_in_restores_plotly_behaviour(self):
        legend = _figure("top", interactive=True).layout.legend
        assert legend.itemclick == "toggle"
        assert legend.itemdoubleclick == "toggleothers"

    def test_hidden_legend_needs_no_click_policy(self):
        assert _figure("hidden").layout.showlegend is False


class TestDownloadMatchesTheSelection:
    """The PNG is rendered from this figure, so its traces are the download."""

    def test_only_the_requested_columns_are_drawn(self, frame):
        fig = create_time_series_figure(
            frame, "timestamp", ["sensor_pm25", "sensor_humidity"]
        )
        assert [t.name for t in fig.data] == ["sensor_pm25", "sensor_humidity"]

    def test_unselected_columns_are_absent_not_merely_hidden(self, frame):
        fig = create_time_series_figure(frame, "timestamp", ["sensor_pm25"])
        assert "reference_pm25" not in [t.name for t in fig.data]
        assert all(t.visible is None or t.visible is True for t in fig.data)

    def test_the_exported_png_payload_carries_the_same_traces(self, frame):
        """Pins the property the download helper depends on."""
        import json

        fig = create_time_series_figure(frame, "timestamp", ["sensor_pm25"])
        payload = json.loads(fig.to_json())
        assert [t["name"] for t in payload["data"]] == ["sensor_pm25"]

    def test_requested_order_is_preserved(self, frame):
        fig = create_time_series_figure(
            frame, "timestamp", ["sensor_temperature", "sensor_pm25"]
        )
        assert [t.name for t in fig.data] == ["sensor_temperature", "sensor_pm25"]

    def test_a_missing_column_is_skipped_rather_than_raising(self, frame):
        fig = create_time_series_figure(frame, "timestamp", ["sensor_pm25", "absent"])
        assert [t.name for t in fig.data] == ["sensor_pm25"]

    def test_default_still_picks_numeric_columns(self, frame):
        assert len(create_time_series_figure(frame, "timestamp").data) > 0


class TestFiguresAcceptLegendSettings:
    @pytest.mark.parametrize("position,orientation", [("top", "h"), ("right", "v")])
    def test_time_series_overview_honours_position(self, frame, position, orientation):
        fig = create_time_series_figure(
            frame, "timestamp", ["sensor_pm25"], legend_position=position
        )
        assert fig.layout.legend.orientation == orientation

    def test_time_series_overview_defaults_to_non_interactive(self, frame):
        fig = create_time_series_figure(frame, "timestamp", ["sensor_pm25"])
        assert fig.layout.legend.itemclick is False

    def test_multi_model_overlay_honours_position(self, merged, config_factory):
        from pipeline.run_pipeline import run_modeling_stage

        out = run_modeling_stage(merged, config_factory(["ridge"]))
        fig = create_multi_model_timeseries(
            out["training_results"], legend_position="right"
        )
        assert fig.layout.legend.orientation == "v"
        assert fig.layout.legend.itemclick is False

    def test_multi_model_overlay_can_hide_the_legend(self, merged, config_factory):
        from pipeline.run_pipeline import run_modeling_stage

        out = run_modeling_stage(merged, config_factory(["ridge"]))
        fig = create_multi_model_timeseries(out["training_results"], legend_position="hidden")
        assert fig.layout.showlegend is False

    def test_an_empty_result_set_still_returns_a_figure(self):
        assert create_multi_model_timeseries({}) is not None
