"""
pytest configuration – workspace-level fixes
=============================================
Works around a Windows issue where pytest fails to create its temp directory
when the Windows username contains spaces (e.g. "Abdul Salam").  Pytest tries
to create a directory like ``C:\\...\\Temp\\pytest-of-Abdul Salam`` but that
path triggers an Access Denied error on some Windows setups.

The fix: register a ``pytest_configure`` hook that redirects the base temp
directory to a repo-local ``.tmp/pytest`` folder **before** pytest tries to
create the system temp path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Safe base directory (inside the repo, no spaces)
_SAFE_TMP = Path(__file__).parent / ".tmp" / "pytest"


def pytest_configure(config: pytest.Config) -> None:
    """
    Redirect ``tmp_path`` base to a safe repo-local dir when the default
    pytest-managed temp root would contain a space.

    Spaces in the path cause ``os.mkdir`` to fail with WinError 5 on some
    Windows systems (username with spaces → ``pytest-of-First Last``).
    """
    # Only redirect when not already overridden by the user
    if config.option.basetemp is not None:
        return

    # Detect whether the default tmp root would contain a space
    username = os.environ.get("USERNAME", os.environ.get("USER", ""))
    if " " not in username:
        return  # No problem on this machine

    # Create the safe directory and tell pytest to use it
    _SAFE_TMP.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(_SAFE_TMP)
