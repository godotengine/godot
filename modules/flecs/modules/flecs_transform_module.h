#ifndef FLECS_TRANSFORM_MODULE_H
#define FLECS_TRANSFORM_MODULE_H

#include "../components/flecs_transform_component.h"
#include "../flecs_mod.h"
#include "../flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"

namespace modules {

struct FlecsTransformModule {
	FlecsTransformModule(flecs::world &world) {
		
		world.module<FlecsTransformModule>();

		world.component<components::FlecsLocation>();
		world.component<components::FlecsRotation>();
		world.component<components::FlecsScale>();
	}
};

} //namespace modules

class FlecsTransformMod : public FlecsMod {
	GDCLASS(FlecsTransformMod, FlecsMod);

protected:
	static void _bind_methods();

public:
	virtual void initialize(flecs::entity &prefab, flecs::world &world) override;
};

#endif