#include "flecs_jolt_body3d_mod.h"
#include "../flecs_entity_node.h"
#include "modules/jolt_physics/jolt_physics_server_3d.h"
#include "modules/jolt_physics/objects/jolt_body_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/physics/physics_body_3d.h"
#include <flecs_transform_component.h>

void modules::FlecsJoltBody3DModule::_register_systems(flecs::world &world) {
	world.system<const components::FlecsJoltBody3D, components::FlecsLocation, components::FlecsRotation, components::FlecsScale>("jolt_body_3d_to_flecs_sync")
			.with<components::JoltBody3DToFlecsSyncTag>()
			.kind(flecs::PostUpdate)
			.multi_threaded()
			.each([](flecs::entity e, const components::FlecsJoltBody3D &jolt_body, components::FlecsLocation &location, components::FlecsRotation &rotation, components::FlecsScale &scale) {
				if (jolt_body.value) {
					Transform3D transform = jolt_body.value->get_transform_scaled();

					location.value = transform.origin;
					rotation.value = transform.basis.get_rotation_quaternion();
					scale.value = transform.basis.get_scale();
				}
			});

	world.system<components::FlecsJoltBody3D, const components::FlecsLocation, const components::FlecsRotation, const components::FlecsScale>("flecs_to_jolt_body_3d_sync")
			.with<components::FlecsToJoltBody3DSyncTag>()
			.multi_threaded()
			.kind(flecs::PostUpdate)
			.each([](flecs::entity e, components::FlecsJoltBody3D &jolt_body, const components::FlecsLocation &location, const components::FlecsRotation &rotation, const components::FlecsScale &scale) {
				if (jolt_body.value) {
					Transform3D transform{};

					transform.origin = location.value;
					transform.basis.set_quaternion(rotation.value);
					transform.scale_basis(scale.value);

					jolt_body.value->set_transform(transform);
				}
			});
}

void FlecsJoltBody3DMod::_bind_methods() {
}

void FlecsJoltBody3DMod::initialize(flecs::entity &prefab, flecs::world &world) {
	world.import <modules::FlecsJoltBody3DModule>();

	prefab.auto_override<components::FlecsJoltBody3D>();

	switch (sync_direction) {
		case ModuleSyncDirection::FLECS_TO_GODOT:
			prefab.add<components::FlecsToJoltBody3DSyncTag>();
			break;
		case ModuleSyncDirection::GODOT_TO_FLECS:
			prefab.add<components::JoltBody3DToFlecsSyncTag>();
			break;
	}
}

void FlecsJoltBody3DMod::initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {
	Node3D *node = Object::cast_to<Node3D>(entity->get_parent());
	if (node) {
		if (PhysicsBody3D *body = Object::cast_to<PhysicsBody3D>(node)) {
			JoltBody3D *jolt_body = JoltPhysicsServer3D::get_singleton()->get_body(body->get_rid());
			entity->get_entity()->get_entity().set<components::FlecsJoltBody3D>({ jolt_body });
		}
	}
}
