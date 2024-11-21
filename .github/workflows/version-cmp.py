#!/usr/bin/env python
# Can’t be an isolated script since we want to access zarrs’ metadata

import importlib.metadata as im
import os
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version


def set_outputs(version: Version | str) -> None:
    is_prerelease = version.is_prerelease if isinstance(version, Version) else False
    is_prerelease_json = "true" if is_prerelease else "false"
    print(f"{version=!s} {is_prerelease=}")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as f:
        print(f"version={version}", file=f)
        print(f"is_prerelease={is_prerelease_json}", file=f)


version_tag_str = os.environ["GITHUB_REF_NAME"]
assert version_tag_str.startswith("v"), "should be enforced in `if:` condition"
try:
    version_tag = Version(version_tag_str[1:])
except InvalidVersion:
    set_outputs("")
    sys.exit(0)

if version_tag_str[1:] != str(version_tag):
    sys.exit(f"Tag version not normalized: {version_tag_str} should be v{version_tag}")

if version_tag != (version_meta := Version(im.version("zarrs"))):
    sys.exit(f"Version mismatch: {version_tag} (tag) != {version_meta} (metadata)")

set_outputs(version_meta)
