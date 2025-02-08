#include "flecs_jolt_body3d_physics_module.h"
#include "flecs_jolt_body3d_module.h"

void modules::FlecsJoltBody3DPhysicsModule::_register_systems(flecs::world &world) {

	world.system<components::FlecsJoltBody3D, components::FlecsForce, components::FlecsImpulse>("jolt_body3d_apply_force")
		.kind(flecs::OnUpdate)
		.each([](flecs::entity e, components::FlecsJoltBody3D &jolt_body, components::FlecsForce &force, components::FlecsImpulse &impulse) {
			if (force.value.length() > 0.0001) {
				if (jolt_body.value) {
					jolt_body.value->apply_central_force(force.value);
				}
				force.value = Vector3();
			}
		});


	world.system<components::FlecsJoltBody3D, components::FlecsForce, components::FlecsImpulse>("jolt_body3d_apply_force_at_position")
		.kind(flecs::OnUpdate)
		.each([](flecs::entity e, components::FlecsJoltBody3D &jolt_body, components::FlecsForce &force, components::FlecsImpulse &impulse) {
			if (force.position_force.length() > 0.0001) {
				if (jolt_body.value) {
					jolt_body.value->apply_force(force.position_force, force.position);
				}
				force.position_force = Vector3();
			}
		});


	world.system<components::FlecsJoltBody3D, components::FlecsImpulse>("jolt_body3d_apply_impulse")
		.kind(flecs::OnUpdate)

		.each([](flecs::entity e, components::FlecsJoltBody3D &jolt_body, components::FlecsImpulse &impulse) {
			if (impulse.value.length() > 0.0001) {
				if (jolt_body.value) {
					jolt_body.value->apply_central_impulse(impulse.value);
				}
				impulse.value = Vector3();
			}
		});


	world.system<components::FlecsJoltBody3D, components::FlecsImpulse>("jolt_body3d_apply_impulse_at_position")
		.kind(flecs::OnUpdate)
		.each([](flecs::entity e, components::FlecsJoltBody3D &jolt_body, components::FlecsImpulse &impulse) {
			if (impulse.position_impulse.length() > 0.0001) {
				if (jolt_body.value) {
					jolt_body.value->apply_impulse(impulse.position_impulse, impulse.position);
				}
				impulse.position_impulse = Vector3();
			}

		});
}

void FlecsJoltBody3DPhysicsMod::_bind_methods() {
}


TypedArray<FlecsMod> FlecsJoltBody3DPhysicsMod::get_required_modules() const {
	TypedArray<FlecsMod> deps;
	deps.append(memnew(FlecsJoltBody3DMod)); 
	return deps;
}

void FlecsJoltBody3DPhysicsMod::initialize(flecs::entity &prefab, flecs::world &world) {
	world.import <modules::FlecsJoltBody3DPhysicsModule>();

	prefab.auto_override<components::FlecsForce>();
	prefab.auto_override<components::FlecsImpulse>();
}

void FlecsJoltBody3DPhysicsMod::initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {
}
