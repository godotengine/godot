/* register_types.cpp */

#include "register_types.h"

#include "core/object/class_db.h"
#include "test.h"

void initialize_testmod_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<TestNode>();
}

void uninitialize_testmod_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	// Nothing to do here in this example.
}
