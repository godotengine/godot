#include "flecs_physics_component.h"

void FlecsPhysicsComponent::_bind_methods() {

	ClassDB::bind_method(D_METHOD("apply_force"), &FlecsPhysicsComponent::apply_force);
	ClassDB::bind_method(D_METHOD("apply_impulse"), &FlecsPhysicsComponent::apply_impulse);
	ClassDB::bind_method(D_METHOD("apply_force_at_position"), &FlecsPhysicsComponent::apply_force_at_position);
	ClassDB::bind_method(D_METHOD("apply_impulse_at_position"), &FlecsPhysicsComponent::apply_impulse_at_position);
	ClassDB::bind_method(D_METHOD("get_force"), &FlecsPhysicsComponent::get_force);
	ClassDB::bind_method(D_METHOD("get_impulse"), &FlecsPhysicsComponent::get_impulse);
	
}

void FlecsPhysicsComponent::add_component(flecs::entity p_entity) {
	p_entity.add<components::FlecsForce>();
	p_entity.add<components::FlecsImpulse>();
}


void FlecsPhysicsComponent::remove_component(flecs::entity p_entity) {
	p_entity.remove<components::FlecsForce>();
	p_entity.remove<components::FlecsImpulse>();
}


bool FlecsPhysicsComponent::has_component(flecs::entity p_entity) const {
	return p_entity.has<components::FlecsForce>() && p_entity.has<components::FlecsImpulse>();
}


void FlecsPhysicsComponent::apply_force(const Vector3 &p_force) {
	components::FlecsForce* force = entity.get_mut<components::FlecsForce>();
	force->value += p_force;
}

void FlecsPhysicsComponent::apply_impulse(const Vector3 &p_impulse) {
	components::FlecsImpulse* impulse = entity.get_mut<components::FlecsImpulse>();
	impulse->value += p_impulse;
}

void FlecsPhysicsComponent::apply_force_at_position(const Vector3 &p_force, const Vector3 &p_position) {
	components::FlecsForce* force = entity.get_mut<components::FlecsForce>();
	force->value += p_force;
	force->position = p_position;
}

void FlecsPhysicsComponent::apply_impulse_at_position(const Vector3 &p_impulse, const Vector3 &p_position) {
	components::FlecsImpulse* impulse = entity.get_mut<components::FlecsImpulse>();
	impulse->value += p_impulse;
	impulse->position = p_position;
}

Vector3 FlecsPhysicsComponent::get_force() const {
	return entity.get<components::FlecsForce>()->value;
}

Vector3 FlecsPhysicsComponent::get_impulse() const {
	return entity.get<components::FlecsImpulse>()->value;
}
