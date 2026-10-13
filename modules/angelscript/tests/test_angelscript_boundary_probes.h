#pragma once

#include "tests/test_macros.h"

void as_boundary_probes();

#ifndef ANGELSCRIPT_BOUNDARY_PROBES_IMPL
TEST_CASE("[AngelScript] boundary probes") {
	as_boundary_probes();
}
#endif
