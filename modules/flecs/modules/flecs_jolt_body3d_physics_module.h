#ifndef FLECS_JOLT_BODY3D_PHYSICS_MODULE_H
#define FLECS_JOLT_BODY3D_PHYSICS_MODULE_H

#include "../components/flecs_physics_component.h"
#include "../flecs_mod.h"
#include "../flecs_world.h"
#include "core/object/class_db.h"
#include "core/object/object.h"


namespace modules {

struct FlecsJoltBody3DPhysicsModule {
	FlecsJoltBody3DPhysicsModule(flecs::world &world) {
		world.module<FlecsJoltBody3DPhysicsModule>();
		world.component<components::FlecsForce>();
		world.component<components::FlecsImpulse>();
		_register_systems(world);
	}

	void _register_systems(flecs::world &world);
};

} //namespace modules

class FlecsJoltBody3DPhysicsMod : public FlecsMod {
	GDCLASS(FlecsJoltBody3DPhysicsMod, FlecsMod);


protected:
	static void _bind_methods();

public:
	virtual TypedArray<FlecsMod> get_required_modules() const override;
	virtual void initialize(flecs::entity &prefab, flecs::world &world) override;
	virtual void initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) override;


	ModuleSyncDirection get_sync_direction() const;
	void set_sync_direction(ModuleSyncDirection p_sync_direction);

protected:

	ModuleSyncDirection sync_direction = ModuleSyncDirection::FLECS_TO_GODOT;

};

#endif // FLECS_JOLT_BODY3D_PHYSICS_MODULE_H