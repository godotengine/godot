/**************************************************************************/
/*  interpolated_camera.cpp                                               */
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

#include "interpolated_camera.h"

#include "core/engine.h"

void InterpolatedCamera::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_process_mode();
		} break;
		case NOTIFICATION_INTERNAL_PROCESS:
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!enabled) {
				break;
			}
			if (has_node(target)) {
				Spatial *node = Object::cast_to<Spatial>(get_node(target));
				if (!node) {
					break;
				}

				float delta = speed * (process_mode == INTERPOLATED_CAMERA_PROCESS_PHYSICS ? get_physics_process_delta_time() : get_process_delta_time());
				Transform target_xform = node->get_global_transform();
				Transform local_transform = get_global_transform();
				local_transform = local_transform.interpolate_with(target_xform, delta);
				set_global_transform(local_transform);
				Camera *cam = Object::cast_to<Camera>(node);
				if (cam) {
					if (cam->get_projection() == get_projection()) {
						float new_near = Math::lerp(get_znear(), cam->get_znear(), delta);
						float new_far = Math::lerp(get_zfar(), cam->get_zfar(), delta);

						if (cam->get_projection() == PROJECTION_ORTHOGONAL) {
							float size = Math::lerp(get_size(), cam->get_size(), delta);
							set_orthogonal(size, new_near, new_far);
						} else {
							float fov = Math::lerp(get_fov(), cam->get_fov(), delta);
							set_perspective(fov, new_near, new_far);
						}
					}
				}
			}

		} break;
	}
}

void InterpolatedCamera::set_process_mode(InterpolatedCameraProcessMode p_mode) {
	if (process_mode == p_mode) {
		return;
	}
	process_mode = p_mode;
	_update_process_mode();
}

InterpolatedCamera::InterpolatedCameraProcessMode InterpolatedCamera::get_process_mode() const {
	return process_mode;
}

void InterpolatedCamera::_set_target(const Object *p_target) {
	ERR_FAIL_NULL(p_target);
	set_target(Object::cast_to<Spatial>(p_target));
}

void InterpolatedCamera::set_target(const Spatial *p_target) {
	ERR_FAIL_NULL(p_target);
	target = get_path_to(p_target);
}

void InterpolatedCamera::set_target_path(const NodePath &p_path) {
	target = p_path;
}

NodePath InterpolatedCamera::get_target_path() const {
	return target;
}

void InterpolatedCamera::set_interpolation_enabled(bool p_enable) {
	if (enabled == p_enable) {
		return;
	}
	enabled = p_enable;
	_update_process_mode();
}

void InterpolatedCamera::_update_process_mode() {
	if (Engine::get_singleton()->is_editor_hint() || !enabled) {
		set_process_internal(false);
		set_physics_process_internal(false);
	} else if (process_mode == INTERPOLATED_CAMERA_PROCESS_IDLE) {
		set_process_internal(true);
		set_physics_process_internal(false);
	} else {
		set_process_internal(false);
		set_physics_process_internal(true);
	}
}

bool InterpolatedCamera::is_interpolation_enabled() const {
	return enabled;
}

void InterpolatedCamera::set_speed(real_t p_speed) {
	speed = p_speed;
}

real_t InterpolatedCamera::get_speed() const {
	return speed;
}

void InterpolatedCamera::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target_path", "target_path"), &InterpolatedCamera::set_target_path);
	ClassDB::bind_method(D_METHOD("get_target_path"), &InterpolatedCamera::get_target_path);
	ClassDB::bind_method(D_METHOD("set_target", "target"), &InterpolatedCamera::_set_target);

	ClassDB::bind_method(D_METHOD("set_speed", "speed"), &InterpolatedCamera::set_speed);
	ClassDB::bind_method(D_METHOD("get_speed"), &InterpolatedCamera::get_speed);

	ClassDB::bind_method(D_METHOD("set_interpolation_enabled", "target_path"), &InterpolatedCamera::set_interpolation_enabled);
	ClassDB::bind_method(D_METHOD("is_interpolation_enabled"), &InterpolatedCamera::is_interpolation_enabled);

	ClassDB::bind_method(D_METHOD("set_process_mode", "mode"), &InterpolatedCamera::set_process_mode);
	ClassDB::bind_method(D_METHOD("get_process_mode"), &InterpolatedCamera::get_process_mode);

	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "target"), "set_target_path", "get_target_path");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "speed"), "set_speed", "get_speed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_interpolation_enabled", "is_interpolation_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_mode", "get_process_mode");

	BIND_ENUM_CONSTANT(INTERPOLATED_CAMERA_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(INTERPOLATED_CAMERA_PROCESS_IDLE);
}

InterpolatedCamera::InterpolatedCamera() {
	enabled = false;
	speed = 1;
	process_mode = INTERPOLATED_CAMERA_PROCESS_IDLE;
}
