#include "flecs_sync_mod.h"

void FlecsSyncMod::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_sync_direction"), &FlecsSyncMod::get_sync_direction);
	ClassDB::bind_method(D_METHOD("set_sync_direction"), &FlecsSyncMod::set_sync_direction);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sync_direction", PROPERTY_HINT_ENUM, "NONE,FLECS_TO_GODOT,GODOT_TO_FLECS"), "set_sync_direction", "get_sync_direction");
}
