#pragma once
#include "tests/test_macros.h"
void as_property_registration_regression();
#ifndef ANGELSCRIPT_PROPERTY_REGISTRATION_IMPL
TEST_CASE("[AngelScript] native property registration preserves validation") {
	as_property_registration_regression();
}
#endif
