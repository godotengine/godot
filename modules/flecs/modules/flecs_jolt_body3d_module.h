#ifndef FLECS_JOLT_BODY3D_MODULE_H
#define FLECS_JOLT_BODY3D_MODULE_H

#include "../components/flecs_jolt_body3d_component.h"
#include "../flecs_sync_mod.h"
#include "../flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"


namespace modules {

struct FlecsJoltBody3DModule {
	FlecsJoltBody3DModule(flecs::world &world) {


		world.module<FlecsJoltBody3DModule>();
		world.component<components::FlecsJoltBody3D>();
		_register_systems(world);

	}

	void _register_systems(flecs::world &world);
};

} //namespace modules

class FlecsJoltBody3DMod : public FlecsSyncMod {
	GDCLASS(FlecsJoltBody3DMod, FlecsSyncMod);

protected:
	static void _bind_methods();

public:
	virtual void initialize(flecs::entity &prefab, flecs::world &world) override;
	virtual void initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) override;

};

#endif // FLECS_JOLT_BODY3D_MODULE_H