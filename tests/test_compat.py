"""Tests for the engrava compatibility helpers."""

from __future__ import annotations

import importlib.metadata
import warnings

import pytest

from engrava_mcp._compat import (
    ENGRAVA_SUPPORTED_RANGE,
    EngravaVersionWarning,
    incompatible_engrava_message,
    warn_if_engrava_out_of_range,
)


class TestIncompatibleEngravaMessage:
    """The hard-incompatibility message is actionable and complete."""

    def test_names_engrava_the_range_and_the_original_error(self) -> None:
        original = ImportError("cannot import name 'SqliteEngravaCore' from 'engrava'")
        message = incompatible_engrava_message(original)

        assert "engrava" in message
        assert ENGRAVA_SUPPORTED_RANGE in message
        assert ">=0.6,<0.7" in message
        assert str(original) in message

    def test_recovery_instruction_is_present(self) -> None:
        message = incompatible_engrava_message(ImportError("boom"))

        assert "Reinstall" in message
        assert "align" in message.lower()


class TestWarnIfEngravaOutOfRange:
    """The startup version check is a soft warning, never a hard failure."""

    @pytest.mark.parametrize("version", ["0.4.0", "0.5.0", "0.7.0", "1.0.0"])
    def test_out_of_range_version_warns(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: version)

        with pytest.warns(EngravaVersionWarning, match=version):
            warn_if_engrava_out_of_range()

    @pytest.mark.parametrize("version", ["0.6.0", "0.6.2", "0.6.99"])
    def test_in_range_version_does_not_warn(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: version)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_engrava_out_of_range()

    def test_out_of_range_warning_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.4.0")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warn_if_engrava_out_of_range()  # must return, never raise

    def test_missing_metadata_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(_name: str) -> str:
            msg = "engrava"
            raise importlib.metadata.PackageNotFoundError(msg)

        monkeypatch.setattr(importlib.metadata, "version", _raise)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_engrava_out_of_range()

    @pytest.mark.parametrize("version", ["dev", "0", "garbage", "x.y.z", "0.x", "0."])
    def test_unparseable_version_is_silent(
        self, monkeypatch: pytest.MonkeyPatch, version: str
    ) -> None:
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: version)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_engrava_out_of_range()

    def test_pre_release_minor_suffix_is_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A "0.7.0rc1" pre-release is still out of range (minor 7 >= 7).
        monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.7rc1")

        with pytest.warns(EngravaVersionWarning):
            warn_if_engrava_out_of_range()
