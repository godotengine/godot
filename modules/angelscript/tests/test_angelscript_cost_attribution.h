#pragma once

#include "tests/test_macros.h"

void as_cost_attribution();

#ifndef ANGELSCRIPT_COST_ATTRIBUTION_IMPL
TEST_CASE("[AngelScript] cost attribution") {
	as_cost_attribution();
}
#endif
