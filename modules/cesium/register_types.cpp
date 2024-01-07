#include "register_types.h"

#include "core/object/class_db.h"
#include "Ellipsoid.h"
#include "Globe.h"

using namespace Cesium;
void initialize_cesium_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	ClassDB::register_class<Ellipsoid>();
	ClassDB::register_class<Globe>();
}

void uninitialize_cesium_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
