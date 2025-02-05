#include "flecs_root_node_module.h"
#include "../flecs_entity_node.h"
#include "../components/flecs_root_node_component.h"
#include "scene/3d/node_3d.h"

void FlecsRootNodeMod::_bind_methods() {
}

void FlecsRootNodeMod::initialize(flecs::entity &prefab, flecs::world &world) {
	world.import <modules::FlecsRootNodeModule>();

	prefab.auto_override<components::FlecsRootNode>();
}

void FlecsRootNodeMod::initialize_entity_data(FlecsEntityNode *entity, flecs::world &world) {

	entity->get_entity()->get_entity().set<components::FlecsRootNode>({Object::cast_to<Node3D>(entity->get_parent())});
}
