#include "flecs_packed_scene_component.h"
#include "scene/resources/packed_scene.h"

void FlecsPackedSceneComponent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_scene"), &FlecsPackedSceneComponent::get_scene);
	ClassDB::bind_method(D_METHOD("set_scene"), &FlecsPackedSceneComponent::set_scene);
}

Ref<PackedScene> FlecsPackedSceneComponent::get_scene() const {
	return entity.get<components::FlecsPackedScene>()->value;
}

void FlecsPackedSceneComponent::set_scene(Ref<PackedScene> p_scene) {
	entity.get_mut<components::FlecsPackedScene>()->value = p_scene;
}

void FlecsPackedSceneComponent::add_component(flecs::entity p_entity) {
	p_entity.add<components::FlecsPackedScene>();
}

void FlecsPackedSceneComponent::remove_component(flecs::entity p_entity) {
	p_entity.remove<components::FlecsPackedScene>();
}

bool FlecsPackedSceneComponent::has_component(flecs::entity p_entity) const {
	return p_entity.has<components::FlecsPackedScene>();
}
