"""An importable target for the extension-detection tests to point at.

Store resolution must report an advertised extension **without importing it**.
Proving that needs a target that is genuinely importable — an unimportable one
cannot distinguish "never imported" from "tried and failed".

Executing this module's body bumps a counter that lives on :mod:`sys`, not on
this module.  That placement is the point: a counter stored here would vanish
with the module, so an implementation that imported the target and then dropped
it from :data:`sys.modules` would look innocent.  The counter survives that, and
it counts *body execution*, so it is indifferent to how the execution was
reached — ``EntryPoint.load()``, ``importlib.import_module``, ``__import__``,
``runpy``, or reading the source and executing it.

Nothing else in the suite imports this module, and its filename does not match
pytest's collection pattern, so any increment is attributable to the code under
test.
"""

from __future__ import annotations

import sys

#: Attribute name used on :mod:`sys` to hold the execution counter.
COUNTER_ATTRIBUTE = "_engrava_mcp_extension_probe_executions"

setattr(sys, COUNTER_ATTRIBUTE, getattr(sys, COUNTER_ATTRIBUTE, 0) + 1)

#: Marker an entry point can name.  Never read by anything: the tests care only
#: about whether executing this module happened at all.
MANIFEST = "extension-probe-manifest"
