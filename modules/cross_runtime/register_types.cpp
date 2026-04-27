#include "register_types.h"
#include "dot_renderer.h" //imports the class DotRenderer

#include "core/object/class_db.h" //imports the registration system

//called during startup
void initialize_cross_runtime_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<DotRenderer>(); //tells godot to register this class too
}

//called during shutdown
void uninitialize_cross_runtime_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
