#include "flecs_transform_module.h"

void FlecsTransformMod::_bind_methods() {
}

void FlecsTransformMod::initialize(flecs::entity &prefab, flecs::world &world) {

	// Import module if not already imported
	world.import <modules::FlecsTransformModule>();

	prefab.auto_override<components::FlecsLocation>();
	prefab.auto_override<components::FlecsRotation>();
	prefab.auto_override<components::FlecsScale>();

}
