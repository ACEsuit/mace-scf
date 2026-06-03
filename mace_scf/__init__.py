from packaging.version import Version
from .__version__ import __version__

import mace
assert Version(mace.__version__) == Version("0.3.14"), "Currently, this repo is tied to mace==v0.3.14. please checkout this tag in mace."

import graph_longrange
assert Version(graph_longrange.__version__) >= Version("0.3.0"), "please update your graph_longrange version to >= 0.3.0"