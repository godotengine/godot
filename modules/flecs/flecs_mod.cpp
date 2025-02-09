#include "flecs_mod.h"

void FlecsMod::_bind_methods() {
	BIND_ENUM_CONSTANT(NONE);
	BIND_ENUM_CONSTANT(FLECS_TO_GODOT);
	BIND_ENUM_CONSTANT(GODOT_TO_FLECS);
}