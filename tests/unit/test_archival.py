"""storage.archival's pure/local-only helpers -- no DB, no S3."""

import hashlib
from datetime import date
from pathlib import Path

import pytest

from marketpulse.storage.archival import _md5_hex, months_before_cutoff, verify_upload

pytestmark = pytest.mark.unit


# --- months_before_cutoff -----------------------------------------------------


def test_partition_within_retention_is_not_archivable() -> None:
    # Retention 3 months, as_of March -> December/January/February are hot.
    assert not months_before_cutoff(2026, 1, as_of=date(2026, 3, 15), retention_months=3)


def test_partition_older_than_retention_is_archivable() -> None:
    assert months_before_cutoff(2025, 11, as_of=date(2026, 3, 15), retention_months=3)


def test_partition_exactly_at_cutoff_boundary_is_not_archivable() -> None:
    # retention_months=3, as_of March: cutoff month is December (inclusive/hot).
    assert not months_before_cutoff(2025, 12, as_of=date(2026, 3, 1), retention_months=3)


def test_cutoff_handles_year_boundary() -> None:
    assert months_before_cutoff(2025, 1, as_of=date(2026, 3, 1), retention_months=3)
    assert not months_before_cutoff(2025, 12, as_of=date(2026, 1, 15), retention_months=1)


# --- checksum / verify --------------------------------------------------------


def test_verify_upload_passes_on_matching_etag(tmp_path: Path) -> None:
    path = tmp_path / "export.parquet"
    path.write_bytes(b"some parquet bytes")
    etag = hashlib.md5(b"some parquet bytes").hexdigest()  # noqa: S324

    assert verify_upload(path, etag)


def test_verify_upload_fails_closed_on_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "export.parquet"
    path.write_bytes(b"some parquet bytes")

    assert not verify_upload(path, "0" * 32)


def test_md5_hex_matches_hashlib_for_multi_chunk_file(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    payload = b"x" * (1024 * 1024 + 17)  # spans the 1 MiB chunk boundary
    path.write_bytes(payload)

    assert _md5_hex(path) == hashlib.md5(payload).hexdigest()  # noqa: S324
