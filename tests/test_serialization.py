from typing import Any

import pytest

from evonet.core import Nnet
from evonet.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    from_dict,
    to_dict,
)


def test_to_dict_includes_format_metadata() -> None:
    net = Nnet()

    data = to_dict(net)

    assert data["format"] == FORMAT_NAME
    assert data["format_version"] == FORMAT_VERSION


def test_from_dict_accepts_current_format() -> None:
    data: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "layers": [],
    }

    net = from_dict(data)

    assert isinstance(net, Nnet)
    assert net.layers == []


def test_from_dict_accepts_legacy_format_without_metadata() -> None:
    data: dict[str, Any] = {
        "layers": [],
    }

    net = from_dict(data)

    assert isinstance(net, Nnet)
    assert net.layers == []


def test_from_dict_rejects_unknown_format() -> None:
    data: dict[str, Any] = {
        "format": "other",
        "format_version": FORMAT_VERSION,
        "layers": [],
    }

    with pytest.raises(ValueError, match=r"Unsupported serialization format"):
        from_dict(data)


def test_from_dict_rejects_unsupported_format_version() -> None:
    data: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION + 1,
        "layers": [],
    }

    with pytest.raises(ValueError, match=r"Unsupported EvoNet format version"):
        from_dict(data)


@pytest.mark.parametrize(
    "data",
    [
        {
            "format": FORMAT_NAME,
            "layers": [],
        },
        {
            "format_version": FORMAT_VERSION,
            "layers": [],
        },
    ],
)
def test_from_dict_rejects_incomplete_format_metadata(
    data: dict[str, Any],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"must define both 'format' and 'format_version'",
    ):
        from_dict(data)
