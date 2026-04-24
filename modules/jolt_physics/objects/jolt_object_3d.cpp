/**************************************************************************/
/*  jolt_object_3d.cpp                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "jolt_object_3d.h"

#include "../jolt_physics_server_3d.h"
#include "../spaces/jolt_space_3d.h"
#include "jolt_group_filter.h"

void JoltObject3D::_remove_from_space() {
	if (!in_space()) {
		return;
	}

	space->remove_object(*this);
}

void JoltObject3D::_reset_space() {
	ERR_FAIL_NULL(space);

	space_changing_to = space;
	_space_changing();
	_remove_from_space();
	_add_to_space();
	_space_changed();
	space_changing_to = nullptr;
}

void JoltObject3D::_update_object_layer() {
	if (!in_space()) {
		return;
	}

	space->get_body_iface().SetObjectLayer(jolt_body->GetID(), _get_object_layer());
}

void JoltObject3D::_collision_layer_changed() {
	_update_object_layer();
}

void JoltObject3D::_collision_mask_changed() {
	_update_object_layer();
}

JoltObject3D::JoltObject3D(ObjectType p_object_type) :
		needs_destruction_element(this),
		object_type(p_object_type) {
}

JoltObject3D::~JoltObject3D() {
	JoltSpace3D *body_space = cached_body_space != nullptr ? cached_body_space : space;
	if (jolt_body != nullptr && body_space != nullptr) {
		destroy_jolt_body(body_space, false);
	}
}

Object *JoltObject3D::get_instance() const {
	return ObjectDB::get_instance(instance_id);
}

void JoltObject3D::set_space(JoltSpace3D *p_space) {
	if (space == p_space) {
		return;
	}

	space_changing_to = p_space;
	_space_changing();

	if (space != nullptr) {
		_remove_from_space();
	}

	space = p_space;

	if (space != nullptr) {
		_add_to_space();
	}

	_space_changed();
	space_changing_to = nullptr;
}

void JoltObject3D::set_collision_layer(uint32_t p_layer) {
	if (p_layer == collision_layer) {
		return;
	}

	collision_layer = p_layer;

	_collision_layer_changed();
}

void JoltObject3D::set_collision_mask(uint32_t p_mask) {
	if (p_mask == collision_mask) {
		return;
	}

	collision_mask = p_mask;

	_collision_mask_changed();
}

bool JoltObject3D::can_collide_with(const JoltObject3D &p_other) const {
	return (collision_mask & p_other.get_collision_layer()) != 0;
}

bool JoltObject3D::can_interact_with(const JoltObject3D &p_other) const {
	if (const JoltBody3D *other_body = p_other.as_body()) {
		return can_interact_with(*other_body);
	} else if (const JoltArea3D *other_area = p_other.as_area()) {
		return can_interact_with(*other_area);
	} else if (const JoltSoftBody3D *other_soft_body = p_other.as_soft_body()) {
		return can_interact_with(*other_soft_body);
	} else {
		ERR_FAIL_V_MSG(false, vformat("Unhandled object type: '%d'. This should not happen. Please report this.", p_other.get_type()));
	}
}

void JoltObject3D::enqueue_needs_destruction(JoltSpace3D *p_space) {
	if (p_space != nullptr) {
		p_space->enqueue_needs_destruction(&needs_destruction_element);
	}
}

void JoltObject3D::dequeue_needs_destruction(JoltSpace3D *p_space) {
	if (p_space != nullptr) {
		p_space->dequeue_needs_destruction(&needs_destruction_element);
	}
}

void JoltObject3D::destroy_jolt_body(JoltSpace3D *p_space, bool p_notify) {
	if (jolt_body == nullptr || p_space == nullptr) {
		return;
	}

	dequeue_needs_destruction(p_space);

	if (p_notify) {
		_jolt_body_destroying();
	}

	JPH::BodyInterface &body_iface = p_space->get_body_iface();
	const JPH::BodyID jolt_id = jolt_body->GetID();
	if (body_iface.IsAdded(jolt_id)) {
		body_iface.RemoveBody(jolt_id);
	}
	body_iface.DestroyBody(jolt_id);

	jolt_body = nullptr;
	cached_body_space = nullptr;
}

String JoltObject3D::to_string() const {
	static const String fallback_name = "<unknown>";

	if (JoltPhysicsServer3D::get_singleton()->is_on_separate_thread()) {
		return fallback_name; // Calling `Object::to_string` is not thread-safe.
	}

	Object *instance = get_instance();
	return instance != nullptr ? instance->to_string() : fallback_name;
}
