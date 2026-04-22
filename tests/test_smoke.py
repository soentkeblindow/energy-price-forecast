"""Smoke tests to confirm the project skeleton is importable."""

import energy_price_forecast


def test_package_imports() -> None:
    """The main package should be importable without errors."""
    assert energy_price_forecast is not None


def test_subpackages_importable() -> None:
    """All planned subpackages should be importable."""
    from energy_price_forecast import (
        dashboard,
        data,
        evaluation,
        features,
        models,
    )

    assert data is not None
    assert dashboard is not None
    assert evaluation is not None
    assert features is not None
    assert models is not None
