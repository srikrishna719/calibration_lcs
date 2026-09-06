"""Validation and defaulting for pipeline configuration.

Configs arrive from two places: the packaged ``default.yaml`` and a file the
user uploads in the app. The uploaded one used to be passed straight through,
so a config missing a section produced a bare ``KeyError`` -- shown in the UI
as ``x 'data'``, which says nothing about what to fix.

Anything absent is filled from the packaged defaults, so a config carrying only
the overrides a user cares about works. Only values that are present and
unusable raise, and they raise naming the key, what was wrong, and what is
accepted.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.yaml"

# Sections read with config["..."] somewhere in the pipeline, so they must exist
# by the time anything runs.
REQUIRED_SECTIONS = (
    "app", "data", "preprocessing", "alignment",
    "feature_engineering", "normalization", "training", "evaluation",
)

# Keys with no sensible default: the pipeline cannot guess which column is which.
REQUIRED_DATA_KEYS = ("timestamp_column", "target_column")

_VALIDATION_METHODS = {
    "timeseriessplit", "timeseries", "tscv",
    "kfold", "crossvalidation", "kfoldcrossvalidation",
    "holdout", "holdoutvalidation", "trainsettestsplit", "traintestsplit",
}


class ConfigError(ValueError):
    """A configuration value is present but unusable."""


def load_default_config() -> Dict[str, Any]:
    """Read the packaged default configuration."""
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required to read the default configuration.")
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_config_text(text: str, suffix: str) -> Any:
    """Parse YAML or JSON config text, reporting syntax errors readably."""
    normalized = str(suffix or "").lower()
    try:
        if normalized in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover
                raise ImportError("PyYAML is required to read YAML configuration files.")
            return yaml.safe_load(text)
        if normalized == ".json":
            return json.loads(text)
    except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001 - yaml raises its own types
        if isinstance(exc, (ImportError, ConfigError)):
            raise
        raise ConfigError(f"Config file could not be parsed: {exc}") from exc
    raise ConfigError(f"Config file must be YAML, YML, or JSON, not '{suffix}'.")


def _merge_defaults(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively fill ``overrides`` with anything missing from ``base``."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_defaults(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


# Older configs carried these alongside the training equivalents, leaving two
# places to look and no rule for which won. They are folded into training.* on
# load, so nothing downstream has to know about them.
LEGACY_KEYS = {
    ("validation", "method"): "validation_method",
    ("modelling", "objective"): "modelling_objective",
}


def _migrate_legacy_sections(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the retired top-level blocks into ``training`` and drop them."""
    training = config.setdefault("training", {})
    for (section, key), training_key in LEGACY_KEYS.items():
        block = config.get(section)
        if not isinstance(block, Mapping):
            continue
        value = block.get(key)
        # An explicit training value wins; the legacy block only fills a gap.
        if value is not None and training.get(training_key) is None:
            training[training_key] = value
    for section, _ in LEGACY_KEYS:
        config.pop(section, None)
    return config


def unknown_sections(config: Mapping[str, Any]) -> List[str]:
    """Top-level keys the pipeline does not read -- usually a typo."""
    known = set(load_default_config().keys()) | {"validation", "modelling"}
    return sorted(str(key) for key in config if key not in known)


def _require_number(section: str, key: str, value: Any, *, minimum=None, maximum=None,
                    integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{section}.{key} must be a number, got {value!r}."
        )
    if integer and float(value) != int(value):
        raise ConfigError(f"{section}.{key} must be a whole number, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{section}.{key} must be at least {minimum}, got {value!r}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{section}.{key} must be at most {maximum}, got {value!r}.")


def validate_config(config: Any, source: str = "configuration") -> Dict[str, Any]:
    """Return a complete, usable config, or raise explaining what is wrong.

    Missing sections and keys are filled from the packaged defaults. Values that
    are present but unusable raise :class:`ConfigError` naming the key.
    """
    if config is None:
        raise ConfigError(f"The {source} is empty.")
    if not isinstance(config, Mapping):
        raise ConfigError(
            f"The {source} must be a mapping of sections, got {type(config).__name__}. "
            "Check the file is a YAML/JSON object rather than a list or a bare value."
        )

    for section in config:
        if section in REQUIRED_SECTIONS and not isinstance(config[section], Mapping):
            raise ConfigError(
                f"Section '{section}' must be a mapping of settings, "
                f"got {type(config[section]).__name__}."
            )

    # Migrate before merging: defaults would otherwise fill training.* first and
    # the legacy block would never get a chance to supply the value.
    migrated = _migrate_legacy_sections(deepcopy(dict(config)))
    merged = _merge_defaults(load_default_config(), migrated)
    merged.pop("validation", None)
    merged.pop("modelling", None)

    data = merged["data"]
    for key in REQUIRED_DATA_KEYS:
        if not str(data.get(key) or "").strip():
            raise ConfigError(
                f"data.{key} must name a column in your CSVs; it is empty. "
                "Set it in the Column setup on the Upload step, or in the config file."
            )

    training = merged["training"]
    _require_number("training", "test_size", training.get("test_size"), minimum=0, maximum=1)
    if not 0 < float(training["test_size"]) < 1:
        raise ConfigError(
            f"training.test_size must be between 0 and 1 exclusive, got {training['test_size']!r}."
        )
    _require_number(
        "training", "cross_validation_folds", training.get("cross_validation_folds"),
        minimum=2, integer=True,
    )

    selected = training.get("selected_models")
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ConfigError(
            "training.selected_models must be a non-empty list of model names, "
            f"got {selected!r}."
        )

    method = str(training.get("validation_method", "")).strip().lower()
    normalized_method = method.replace("_", "").replace("-", "").replace(" ", "")
    if normalized_method and normalized_method not in _VALIDATION_METHODS:
        raise ConfigError(
            f"training.validation_method must be TimeSeriesSplit, K-Fold or Holdout, "
            f"got {training.get('validation_method')!r}."
        )

    _require_number("app", "random_state", merged["app"].get("random_state", 42), integer=True)

    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    parsed = parse_config_text(text, path.suffix)
    return validate_config(parsed, source=f"config file '{path.name}'")


def load_config_text(text: str, suffix: str, source: str = "uploaded config") -> Tuple[Dict[str, Any], List[str]]:
    """Validate uploaded config text, returning it with any unknown-section warnings."""
    parsed = parse_config_text(text, suffix)
    validated = validate_config(parsed, source=source)
    return validated, unknown_sections(parsed if isinstance(parsed, Mapping) else {})
