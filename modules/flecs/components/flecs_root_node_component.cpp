#include "flecs_root_node_component.h"
#include "scene/3d/node_3d.h"

void FlecsRootNodeComponent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_root_node"), &FlecsRootNodeComponent::get_root_node);
	ClassDB::bind_method(D_METHOD("set_root_node"), &FlecsRootNodeComponent::set_root_node);
}


Node3D *FlecsRootNodeComponent::get_root_node() const {
	return entity.get<components::FlecsRootNode>()->value;
}

void FlecsRootNodeComponent::set_root_node(Node3D *p_root_node) {
	entity.set<components::FlecsRootNode>({p_root_node});
}

void FlecsRootNodeComponent::add_component(flecs::entity p_entity) {
	p_entity.add<components::FlecsRootNode>();
}

void FlecsRootNodeComponent::remove_component(flecs::entity p_entity) {
	p_entity.remove<components::FlecsRootNode>();
}

bool FlecsRootNodeComponent::has_component(flecs::entity p_entity) const {
	return p_entity.has<components::FlecsRootNode>();
}
