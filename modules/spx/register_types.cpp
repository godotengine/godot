#include "register_types.h"
#include "spx_draw_tiles.h"
#include "spx.h"

void initialize_spx_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<SpxDrawTiles>();
}

void uninitialize_spx_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
