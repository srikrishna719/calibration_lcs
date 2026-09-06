"""Download buttons must not do their work until they are clicked.

Regression cover: PNG payloads were rendered eagerly for every chart on every
script rerun. Kaleido takes seconds per figure, so a page with six charts spent
roughly fifteen seconds re-rendering images nobody had asked for each time a
checkbox moved.
"""

from __future__ import annotations

from unittest import mock

import pandas as pd
import plotly.graph_objects as go
import pytest

from modules import download_helpers
from modules.download_helpers import (
    _deferred_csv,
    _deferred_png,
    render_chart_download,
    render_df_download,
)


@pytest.fixture
def figure() -> go.Figure:
    return go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


class TestDeferral:
    def test_building_the_png_callable_touches_the_figure_not_at_all(self, figure):
        with mock.patch.object(go.Figure, "to_json", autospec=True) as to_json:
            callback = _deferred_png(figure)
            assert to_json.call_count == 0
        assert callable(callback)

    def test_building_the_csv_callable_does_not_serialize(self, frame):
        with mock.patch.object(pd.DataFrame, "to_csv", autospec=True) as to_csv:
            callback = _deferred_csv(frame)
            assert to_csv.call_count == 0
        assert callable(callback)

    def test_calling_it_renders_a_real_png(self, figure):
        data = _deferred_png(figure)()
        assert data[:4] == b"\x89PNG"
        assert len(data) > 1000

    def test_calling_it_serializes_the_csv(self, frame):
        data = _deferred_csv(frame)()
        assert data.decode("utf-8").splitlines()[0] == "a,b"

    def test_repeat_downloads_reuse_the_render(self, figure):
        callback = _deferred_png(figure)
        assert callback() == callback()

    def test_render_failure_explains_itself(self, figure):
        with mock.patch.object(
            download_helpers, "_png_from_figure_json", side_effect=OSError("no runtime")
        ):
            with pytest.raises(RuntimeError, match="PNG export failed"):
                _deferred_png(figure)()


class TestButtonsPassCallables:
    """The whole point: st.download_button must receive a callable, not bytes."""

    @staticmethod
    def _columns(count):
        containers = []
        for _ in range(count):
            container = mock.MagicMock()
            container.__enter__ = mock.Mock(return_value=container)
            container.__exit__ = mock.Mock(return_value=False)
            containers.append(container)
        return containers

    def test_table_download_defers(self, frame):
        with mock.patch.object(download_helpers.st, "columns", side_effect=lambda spec, **kw: self._columns(len(spec))), \
             mock.patch.object(download_helpers.st, "download_button") as button, \
             mock.patch.object(pd.DataFrame, "to_csv", autospec=True) as to_csv:
            render_df_download(frame, key="t", filename="table.csv")

        assert to_csv.call_count == 0, "serialized during render instead of on click"
        assert callable(button.call_args.kwargs["data"])

    def test_chart_downloads_defer(self, figure, frame):
        with mock.patch.object(download_helpers.st, "columns", side_effect=lambda spec, **kw: self._columns(len(spec))), \
             mock.patch.object(download_helpers.st, "download_button") as button, \
             mock.patch.object(go.Figure, "to_json", autospec=True) as to_json, \
             mock.patch.object(pd.DataFrame, "to_csv", autospec=True) as to_csv:
            render_chart_download(figure, frame, key="c", filename_prefix="chart")

        assert to_json.call_count == 0, "figure serialized during render instead of on click"
        assert to_csv.call_count == 0
        assert button.call_count == 2
        for call in button.call_args_list:
            assert callable(call.kwargs["data"])

    def test_chart_without_source_data_still_offers_the_png(self, figure):
        with mock.patch.object(download_helpers.st, "columns", side_effect=lambda spec, **kw: self._columns(len(spec))), \
             mock.patch.object(download_helpers.st, "download_button") as button:
            render_chart_download(figure, None, key="c", filename_prefix="chart")

        assert button.call_count == 1
        assert button.call_args.kwargs["file_name"] == "chart.png"


@pytest.mark.slow
class TestCostOfARerun:
    def test_rendering_many_charts_costs_nothing_until_clicked(self):
        """A Results page carries six charts; a rerun must not render any."""
        figures = [go.Figure(go.Scatter(x=list(range(50)), y=list(range(50)))) for _ in range(6)]
        with mock.patch.object(go.Figure, "to_image", autospec=True) as to_image:
            callbacks = [_deferred_png(f) for f in figures]
            assert to_image.call_count == 0
        assert len(callbacks) == 6
