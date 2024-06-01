#include "jolt_object_impl_3d.hpp"

#include "objects/jolt_group_filter.hpp"
#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_layer_mapper.hpp"
#include "spaces/jolt_space_3d.hpp"

JoltObjectImpl3D::JoltObjectImpl3D(ObjectType p_object_type)
	: object_type(p_object_type) { }

JoltObjectImpl3D::~JoltObjectImpl3D() = default;

Object* JoltObjectImpl3D::get_instance() const {
	return ObjectDB::get_instance(instance_id);
}

Object* JoltObjectImpl3D::get_instance_unsafe() const {
	// HACK(mihe): This is being deliberately and incorrectly cast to a godot-cpp `Object` when in
	// reality it's a Godot `Object`. This is meant to be used in places where an `Object` is
	// returned through a parameter, such as in `PhysicsServer3DExtensionRayResult`, because
	// godot-cpp is unable to do the necessary unwrapping of the instance bindings in such cases.
	//
	// Dereferencing this pointer from the extension will lead to bad things.
	return reinterpret_cast<Object*>(get_instance());
}

Object* JoltObjectImpl3D::get_instance_wrapped() const {
	return ObjectDB::get_instance(instance_id);
}

void JoltObjectImpl3D::set_space(JoltSpace3D* p_space) {
	if (space == p_space) {
		return;
	}

	_space_changing();

	if (space != nullptr) {
		_remove_from_space();
	}

	space = p_space;

	if (space != nullptr) {
		_add_to_space();
	}

	_space_changed();
}

void JoltObjectImpl3D::set_collision_layer(uint32_t p_layer) {
	if (p_layer == collision_layer) {
		return;
	}

	collision_layer = p_layer;

	_collision_layer_changed();
}

void JoltObjectImpl3D::set_collision_mask(uint32_t p_mask) {
	if (p_mask == collision_mask) {
		return;
	}

	collision_mask = p_mask;

	_collision_mask_changed();
}

void JoltObjectImpl3D::_remove_from_space() {
	QUIET_FAIL_COND(jolt_id.IsInvalid());

	space->remove_body(jolt_id);

	jolt_id = {};
}

void JoltObjectImpl3D::_reset_space() {
	ERR_FAIL_NULL(space);

	_space_changing();
	_remove_from_space();
	_add_to_space();
	_space_changed();
}

bool JoltObjectImpl3D::can_collide_with(const JoltObjectImpl3D& p_other) const {
	return (collision_mask & p_other.get_collision_layer()) != 0;
}

bool JoltObjectImpl3D::can_interact_with(const JoltObjectImpl3D& p_other) const {
	if (const JoltBodyImpl3D* other_body = p_other.as_body()) {
		return can_interact_with(*other_body);
	} else if (const JoltAreaImpl3D* other_area = p_other.as_area()) {
		return can_interact_with(*other_area);
	} else if (const JoltSoftBodyImpl3D* other_soft_body = p_other.as_soft_body()) {
		return can_interact_with(*other_soft_body);
	} else {
		ERR_FAIL_D_MSG(vformat("Unhandled object type: '%d'", p_other.get_type()));
	}
}

void JoltObjectImpl3D::pre_step(
	[[maybe_unused]] float p_step,
	[[maybe_unused]] JPH::Body& p_jolt_body
) { }

void JoltObjectImpl3D::post_step(
	[[maybe_unused]] float p_step,
	[[maybe_unused]] JPH::Body& p_jolt_body
) { }

String JoltObjectImpl3D::to_string() const {
	Object* instance = ObjectDB::get_instance(instance_id);
	return instance != nullptr ? instance->to_string() : "<unknown>";
}

void JoltObjectImpl3D::_update_object_layer() {
	if (space == nullptr) {
		return;
	}

	space->get_body_iface().SetObjectLayer(jolt_id, _get_object_layer());
}

void JoltObjectImpl3D::_collision_layer_changed() {
	_update_object_layer();
}

void JoltObjectImpl3D::_collision_mask_changed() {
	_update_object_layer();
}
