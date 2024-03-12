#include "register_types.h"
#include "core/object/class_db.h"
#include "unknoterNode.h"

void initialize_unknoter_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
			return;
	}
	ClassDB::register_class<UnknoterNode>();
}

void uninitialize_unknoter_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
			return;
	}
}
