/* register_types.cpp */

#include "register_types.h"

#include "core/engine.h"
#include "test.h"

void initialize_test_module() {
	ClassDB::register_class<test>();
}

void uninitialize_test_module() {
	// Nothing to do here in this example.
}
