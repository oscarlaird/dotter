from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).parent.resolve()
LEAN_BINARY = ROOT / ".lake" / "build" / "bin" / "lean_hello_bin"
PACKAGE_BINARY = ROOT / "src" / "lean_hello" / "bin" / "lean_hello_bin"


class BuildLeanBinary(build_py):
    """Build the Lean executable and copy it into the Python package."""

    def run(self) -> None:
        subprocess.run(["lake", "build", "lean_hello_bin"], cwd=ROOT, check=True)

        PACKAGE_BINARY.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LEAN_BINARY, PACKAGE_BINARY)
        os.chmod(PACKAGE_BINARY, 0o755)

        super().run()


setup(cmdclass={"build_py": BuildLeanBinary})
