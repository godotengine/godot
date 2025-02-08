#include "flecs_root_node_module.h"
#include "../components/flecs_root_node_component.h"
#include "../components/flecs_transform_component.h"
#include "../flecs_entity_node.h"
#include "modules/jolt_physics/jolt_physics_server_3d.h"
#include "modules/jolt_physics/objects/jolt_body_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/physics/physics_body_3d.h"

void modules::FlecsRootNodeModule::_register_systems(flecs::world &world) {
	world.system<const components::FlecsRootNode, components::FlecsLocation, components::FlecsRotation, components::FlecsScale>("godot_to_flecs_root_node_sync")
			.with<components::GodotToFlecsRootNodeSyncTag>()
			.without<components::HasPhysicsBody3DTag>()
			.kind(flecs::PostUpdate)
			.each([](flecs::entity e, const components::FlecsRootNode &root_node, components::FlecsLocation &location, components::FlecsRotation &rotation, components::FlecsScale &scale) {
				location.value = root_node.value->get_position();
				rotation.value = root_node.value->get_quaternion();
				scale.value = root_node.value->get_scale();
			});

	world.system<const components::FlecsRootNode, components::FlecsLocation, components::FlecsRotation, components::FlecsScale>("godot_physics_body_3d_to_flecs_root_node_sync")
			.with<components::GodotToFlecsRootNodeSyncTag>()
			.with<components::HasPhysicsBody3DTag>()
			.kind(flecs::PostUpdate)
			.multi_threaded()
			.each([](flecs::entity e, const components::FlecsRootNode &root_node, components::FlecsLocation &location, components::FlecsRotation &rotation, components::FlecsScale &scale) {
				if (root_node.jolt_body) {
					Transform3D transform = root_node.jolt_body->get_transform_scaled();
					location.value = transform.origin;
					rotation.value = transform.basis.get_rotation_quaternion();
					scale.value = transform.basis.get_scale();
				}
			});

	world.system<components::FlecsRootNode, const components::FlecsLocation, const components::FlecsRotation, const components::FlecsScale>("flecs_to_godot_physics_body_3d_sync")
			.with<components::FlecsRootNodeToGodotSyncTag>()
			.with<components::HasPhysicsBody3DTag>()
			.multi_threaded()
			.kind(flecs::PostUpdate)
			.each([](flecs::entity e, components::FlecsRootNode &root_node, const components::FlecsLocation &location, const components::FlecsRotation &rotation, const components::FlecsScale &scale) {
				if (root_node.jolt_body) {
					Transform3D transform{};

					transform.origin = location.value;
					transform.basis.set_quaternion(rotation.value);
					transform.scale_basis(scale.value);

					root_node.jolt_body->set_transform(transform);
				}
			});

	world.system<components::FlecsRootNode, const components::FlecsLocation, const components::FlecsRotation, const components::FlecsScale>("flecs_to_godot_root_node_sync")
			.with<components::FlecsRootNodeToGodotSyncTag>()
			.kind(flecs::PostUpdate)
			.each([](flecs::entity e, components::FlecsRootNode &root_node, const components::FlecsLocation &location, const components::FlecsRotation &rotation, const components::FlecsScale &scale) {
				root_node.value->set_position(location.value);
				root_node.value->set_quaternion(rotation.value);
				root_node.value->set_scale(scale.value);
			});
}

void FlecsRootNodeMod::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_sync_direction"), &FlecsRootNodeMod::get_sync_direction);
	ClassDB::bind_method(D_METHOD("set_sync_direction", "sync_direction"), &FlecsRootNodeMod::set_sync_direction);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "sync_direction", PROPERTY_HINT_ENUM, "NONE,FLECS_TO_GODOT,GODOT_TO_FLECS"), "set_sync_direction", "get_sync_direction");
}

void FlecsRootNodeMod::initialize(flecs::entity &prefab, flecs::world &world) {
	world.import <modules::FlecsRootNodeModule>();

	prefab.auto_override<components::FlecsRootNode>();

	switch (sync_direction) {
		case FlecsMod::ModuleSyncDirection::FLECS_TO_GODOT:
			prefab.add<components::FlecsRootNodeToGodotSyncTag>();
			break;
		case FlecsMod::ModuleSyncDirection::GODOT_TO_FLECS:
			prefab.add<components::GodotToFlecsRootNodeSyncTag>();
			break;
	}
}

void FlecsRootNodeMod::initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {
	Node3D *node = Object::cast_to<Node3D>(entity->get_parent());
	if (node) {
		if (PhysicsBody3D *body = Object::cast_to<PhysicsBody3D>(node)) {
			JoltBody3D *jolt_body = JoltPhysicsServer3D::get_singleton()->get_body(body->get_rid());
			entity->get_entity()->get_entity().set<components::FlecsRootNode>({ node, body->get_rid(), jolt_body });
			entity->get_entity()->get_entity().add<components::HasPhysicsBody3DTag>();
		} else {
			entity->get_entity()->get_entity().set<components::FlecsRootNode>({ node });
		}
	}
}

FlecsMod::ModuleSyncDirection FlecsRootNodeMod::get_sync_direction() const {
	return sync_direction;
}

void FlecsRootNodeMod::set_sync_direction(FlecsMod::ModuleSyncDirection p_sync_direction) {
	sync_direction = p_sync_direction;
}
