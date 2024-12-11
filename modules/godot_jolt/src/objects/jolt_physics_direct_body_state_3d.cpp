#include "jolt_physics_direct_body_state_3d.hpp"

#include "objects/jolt_body_impl_3d.hpp"
#include "spaces/jolt_physics_direct_space_state_3d.hpp"
#include "spaces/jolt_space_3d.hpp"

JoltPhysicsDirectBodyState3D::JoltPhysicsDirectBodyState3D(JoltBodyImpl3D* p_body)
	: body(p_body) { }

Vector3 JoltPhysicsDirectBodyState3D::get_total_gravity() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_gravity();
}

real_t JoltPhysicsDirectBodyState3D::get_total_angular_damp() const {
	QUIET_FAIL_NULL_D_ED(body);
	return (real_t)body->get_total_angular_damp();
}

real_t JoltPhysicsDirectBodyState3D::get_total_linear_damp() const {
	QUIET_FAIL_NULL_D_ED(body);
	return (real_t)body->get_total_linear_damp();
}

Vector3 JoltPhysicsDirectBodyState3D::get_center_of_mass() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_center_of_mass();
}

Vector3 JoltPhysicsDirectBodyState3D::get_center_of_mass_local() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_center_of_mass_local();
}

Basis JoltPhysicsDirectBodyState3D::get_principal_inertia_axes() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_principal_inertia_axes();
}

real_t JoltPhysicsDirectBodyState3D::get_inverse_mass() const {
	QUIET_FAIL_NULL_D_ED(body);
	return 1.0 / body->get_mass();
}

Vector3 JoltPhysicsDirectBodyState3D::get_inverse_inertia() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_inverse_inertia();
}

Basis JoltPhysicsDirectBodyState3D::get_inverse_inertia_tensor() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_inverse_inertia_tensor();
}

Vector3 JoltPhysicsDirectBodyState3D::get_linear_velocity() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_linear_velocity();
}

void JoltPhysicsDirectBodyState3D::set_linear_velocity(const Vector3& p_velocity) {
	QUIET_FAIL_NULL_ED(body);
	return body->set_linear_velocity(p_velocity);
}

Vector3 JoltPhysicsDirectBodyState3D::get_angular_velocity() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_angular_velocity();
}

void JoltPhysicsDirectBodyState3D::set_angular_velocity(const Vector3& p_velocity) {
	QUIET_FAIL_NULL_ED(body);
	return body->set_angular_velocity(p_velocity);
}

void JoltPhysicsDirectBodyState3D::set_transform(const Transform3D& p_transform) {
	QUIET_FAIL_NULL_ED(body);
	return body->set_transform(p_transform);
}

Transform3D JoltPhysicsDirectBodyState3D::get_transform() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_transform_scaled();
}

Vector3 JoltPhysicsDirectBodyState3D::get_velocity_at_local_position(
	const Vector3& p_local_position
) const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_velocity_at_position(body->get_position() + p_local_position);
}

void JoltPhysicsDirectBodyState3D::apply_central_impulse(const Vector3& p_impulse) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_central_impulse(p_impulse);
}

void JoltPhysicsDirectBodyState3D::apply_impulse(
	const Vector3& p_impulse,
	const Vector3& p_position
) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_impulse(p_impulse, p_position);
}

void JoltPhysicsDirectBodyState3D::apply_torque_impulse(const Vector3& p_impulse) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_torque_impulse(p_impulse);
}

void JoltPhysicsDirectBodyState3D::apply_central_force(const Vector3& p_force) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_central_force(p_force);
}

void JoltPhysicsDirectBodyState3D::apply_force(const Vector3& p_force, const Vector3& p_position) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_force(p_force, p_position);
}

void JoltPhysicsDirectBodyState3D::apply_torque(const Vector3& p_torque) {
	QUIET_FAIL_NULL_ED(body);
	return body->apply_torque(p_torque);
}

void JoltPhysicsDirectBodyState3D::add_constant_central_force(const Vector3& p_force) {
	QUIET_FAIL_NULL_ED(body);
	return body->add_constant_central_force(p_force);
}

void JoltPhysicsDirectBodyState3D::add_constant_force(
	const Vector3& p_force,
	const Vector3& p_position
) {
	QUIET_FAIL_NULL_ED(body);
	return body->add_constant_force(p_force, p_position);
}

void JoltPhysicsDirectBodyState3D::add_constant_torque(const Vector3& p_torque) {
	QUIET_FAIL_NULL_ED(body);
	return body->add_constant_torque(p_torque);
}

Vector3 JoltPhysicsDirectBodyState3D::get_constant_force() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_constant_force();
}

void JoltPhysicsDirectBodyState3D::set_constant_force(const Vector3& p_force) {
	QUIET_FAIL_NULL_ED(body);
	return body->set_constant_force(p_force);
}

Vector3 JoltPhysicsDirectBodyState3D::get_constant_torque() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_constant_torque();
}

void JoltPhysicsDirectBodyState3D::set_constant_torque(const Vector3& p_torque) {
	QUIET_FAIL_NULL_ED(body);
	return body->set_constant_torque(p_torque);
}

bool JoltPhysicsDirectBodyState3D::is_sleeping() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->is_sleeping();
}

void JoltPhysicsDirectBodyState3D::set_sleep_state(bool p_enabled) {
	QUIET_FAIL_NULL_ED(body);
	body->set_is_sleeping(p_enabled);
}

int32_t JoltPhysicsDirectBodyState3D::get_contact_count() const {
	QUIET_FAIL_NULL_D_ED(body);
	return body->get_contact_count();
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_local_position(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).position;
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_local_normal(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).normal;
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_impulse(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).impulse;
}

int32_t JoltPhysicsDirectBodyState3D::get_contact_local_shape(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).shape_index;
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_local_velocity_at_position(int32_t p_contact_idx
) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).velocity;
}

RID JoltPhysicsDirectBodyState3D::get_contact_collider(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).collider_rid;
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_collider_position(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).collider_position;
}

ObjectID JoltPhysicsDirectBodyState3D::get_contact_collider_id(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).collider_id;
}

Object* JoltPhysicsDirectBodyState3D::get_contact_collider_object(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return ObjectDB::get_instance(body->get_contact(p_contact_idx).collider_id);
}

int32_t JoltPhysicsDirectBodyState3D::get_contact_collider_shape(int32_t p_contact_idx) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).collider_shape_index;
}

Vector3 JoltPhysicsDirectBodyState3D::get_contact_collider_velocity_at_position(
	int32_t p_contact_idx
) const {
	QUIET_FAIL_NULL_D_ED(body);
	ERR_FAIL_INDEX_D(p_contact_idx, body->get_contact_count());
	return body->get_contact(p_contact_idx).collider_velocity;
}

real_t JoltPhysicsDirectBodyState3D::get_step() const {
	QUIET_FAIL_NULL_D_ED(body);
	return (real_t)body->get_space()->get_last_step();
}

void JoltPhysicsDirectBodyState3D::integrate_forces() {
	const auto step = (float)get_step();

	Vector3 linear_velocity = get_linear_velocity();
	Vector3 angular_velocity = get_angular_velocity();

	linear_velocity *= MAX(1.0f - (float)get_total_linear_damp() * step, 0.0f);
	angular_velocity *= MAX(1.0f - (float)get_total_angular_damp() * step, 0.0f);

	linear_velocity += get_total_gravity() * step;

	set_linear_velocity(linear_velocity);
	set_angular_velocity(angular_velocity);
}

PhysicsDirectSpaceState3D* JoltPhysicsDirectBodyState3D::get_space_state() {
	return body->get_space()->get_direct_state();
}
