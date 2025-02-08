#include "flecs_root_node_module.h"
#include "../components/flecs_root_node_component.h"
#include "../components/flecs_transform_component.h"
#include "../flecs_entity_node.h"
#include "scene/3d/node_3d.h"


void modules::FlecsRootNodeModule::_register_systems(flecs::world &world) {
	world.system<const components::FlecsRootNode, components::FlecsLocation, components::FlecsRotation, components::FlecsScale>("godot_to_flecs_root_node_sync")
			.with<components::GodotToFlecsRootNodeSyncTag>()
			.kind(flecs::PostUpdate)
			.each([](flecs::entity e, const components::FlecsRootNode &root_node, components::FlecsLocation &location, components::FlecsRotation &rotation, components::FlecsScale &scale) {
				location.value = root_node.value->get_position();
				rotation.value = root_node.value->get_quaternion();
				scale.value = root_node.value->get_scale();
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
		entity->get_entity()->get_entity().set<components::FlecsRootNode>({ node });
	}
}
