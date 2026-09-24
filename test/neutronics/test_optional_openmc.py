import subprocess
import sys


def test_import_without_openmc():
    """libra_toolbox should import when openmc is not installed (see #24)"""
    # A None entry in sys.modules makes `import openmc` raise ModuleNotFoundError,
    # which simulates an environment where openmc is not installed.
    code = (
        "import sys; sys.modules['openmc'] = None; "
        "import libra_toolbox; import libra_toolbox.neutronics"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
