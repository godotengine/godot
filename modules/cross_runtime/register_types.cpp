// register_types.cpp
#include "register_types.h"

#include "core/object/class_db.h"

void register_command_processing();

void initialize_cross_runtime_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		register_command_processing();
	}
}

void uninitialize_cross_runtime_module(ModuleInitializationLevel p_level) {}