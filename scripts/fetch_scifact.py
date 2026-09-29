"""Download the SciFact release and verify its checksum.

Usage: uv run python scripts/fetch_scifact.py
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "scifact"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    tarball = DEST / "data.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(URL, tarball)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: {digest} != {SHA256}")
    with tarfile.open(tarball) as tf:
        tf.extractall(DEST, filter="data")
    print(f"SciFact OK at {DEST / 'data'}")


if __name__ == "__main__":
    main()
