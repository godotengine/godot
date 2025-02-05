#include "flecs_transform_component.h"

void FlecsTransformComponent::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_location"), &FlecsTransformComponent::get_location);
	ClassDB::bind_method(D_METHOD("get_rotation"), &FlecsTransformComponent::get_rotation);
	ClassDB::bind_method(D_METHOD("get_scale"), &FlecsTransformComponent::get_scale);
	ClassDB::bind_method(D_METHOD("set_location", "location"), &FlecsTransformComponent::set_location);
	ClassDB::bind_method(D_METHOD("set_rotation", "rotation"), &FlecsTransformComponent::set_rotation);
	ClassDB::bind_method(D_METHOD("set_scale", "scale"), &FlecsTransformComponent::set_scale);

}

Vector3 FlecsTransformComponent::get_location() const {
	return entity.get<components::FlecsLocation>()->value;
}

Quaternion FlecsTransformComponent::get_rotation() const {
	return entity.get<components::FlecsRotation>()->value;
}

Vector3 FlecsTransformComponent::get_scale() const {
	return entity.get<components::FlecsScale>()->value;
}

void FlecsTransformComponent::set_location(Vector3 location) {
	entity.set<components::FlecsLocation>({location});
}

void FlecsTransformComponent::set_rotation(Quaternion rotation) {
	entity.set<components::FlecsRotation>({rotation});
}


void FlecsTransformComponent::set_scale(Vector3 scale) {
	entity.set<components::FlecsScale>({scale});
}

void FlecsTransformComponent::add_component(flecs::entity p_entity) {
	p_entity.add<components::FlecsLocation>();
	p_entity.add<components::FlecsRotation>();
	p_entity.add<components::FlecsScale>();
}



void FlecsTransformComponent::remove_component(flecs::entity p_entity) {
	p_entity.remove<components::FlecsLocation>();
	p_entity.remove<components::FlecsRotation>();
	p_entity.remove<components::FlecsScale>();
}



bool FlecsTransformComponent::has_component(flecs::entity p_entity) const {
	return p_entity.has<components::FlecsLocation>() && p_entity.has<components::FlecsRotation>() && p_entity.has<components::FlecsScale>();
}
