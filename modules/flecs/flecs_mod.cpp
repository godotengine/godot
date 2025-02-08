#include "flecs_mod.h"

void FlecsMod::_bind_methods() {
	BIND_ENUM_CONSTANT(ModuleSyncDirection::NONE);
	BIND_ENUM_CONSTANT(ModuleSyncDirection::FLECS_TO_GODOT);
	BIND_ENUM_CONSTANT(ModuleSyncDirection::GODOT_TO_FLECS);
}