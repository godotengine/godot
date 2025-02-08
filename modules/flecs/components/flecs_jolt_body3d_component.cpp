#include "flecs_jolt_body3d_component.h"

void FlecsJoltBody3DComponent::_bind_methods() {
}

void FlecsJoltBody3DComponent::set_jolt_body(JoltBody3D *p_jolt_body) {
	entity.set<components::FlecsJoltBody3D>({p_jolt_body});
}

JoltBody3D * FlecsJoltBody3DComponent::get_jolt_body() const {
	return entity.get<components::FlecsJoltBody3D>()->value;
}


void FlecsJoltBody3DComponent::add_component(flecs::entity p_entity) {
	p_entity.add<components::FlecsJoltBody3D>();
}

void FlecsJoltBody3DComponent::remove_component(flecs::entity p_entity) {
	p_entity.remove<components::FlecsJoltBody3D>();
}

bool FlecsJoltBody3DComponent::has_component(flecs::entity p_entity) const {
	return p_entity.has<components::FlecsJoltBody3D>();
}

