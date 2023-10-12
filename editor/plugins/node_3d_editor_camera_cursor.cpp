/**************************************************************************/
/*  node_3d_editor_camera_cursor.cpp                                      */
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

#include "node_3d_editor_camera_cursor.h"

#include "editor/editor_settings.h"

Node3DEditorCameraCursor::Values::Values() {
	// These rotations place the camera in +X +Y +Z, aka south east, facing north west.
	position.zero();
	eye_position.zero();
	x_rot = 0.5;
	y_rot = -0.5;
	distance = 4.0;
	fov_scale = 1.0;
}

Node3DEditorCameraCursor::Values Node3DEditorCameraCursor::get_current_values() const {
	return current_values;
}

Node3DEditorCameraCursor::Values Node3DEditorCameraCursor::get_target_values() const {
	return target_values;
}

void Node3DEditorCameraCursor::move(const Vector3& p_delta) {
	target_values.position += p_delta;
	target_values.eye_position += p_delta;
}

void Node3DEditorCameraCursor::move_to(const Vector3& p_position) {
	target_values.position = p_position;
	recalculate_eye_position(target_values);
}

void Node3DEditorCameraCursor::rotate(real_t p_x, real_t p_y, bool p_around_eye) {
	rotate_to(target_values.x_rot + p_x, target_values.y_rot + p_y, p_around_eye);
}

void Node3DEditorCameraCursor::rotate_to(real_t p_x, real_t p_y, bool p_around_eye) {
	target_values.x_rot = p_x;
	target_values.y_rot = p_y;
	if (p_around_eye) {
		recalculate_position(target_values);
	}
	else {
		recalculate_eye_position(target_values);
	}
}

void Node3DEditorCameraCursor::set_fov_scale(real_t p_fov_scale) {
	target_values.fov_scale = p_fov_scale;
}

void Node3DEditorCameraCursor::set_freelook_mode(bool p_enabled) {
	freelook_mode = p_enabled;
	stop_interpolation(false);
}

void Node3DEditorCameraCursor::move_freelook(const Vector3& p_direction, real_t p_speed, real_t p_delta) {
	if (!freelook_mode) {
		return;
	}
	Transform3D camera_transform = values_to_camera_transform(current_values);

	const FreelookNavigationScheme navigation_scheme = (FreelookNavigationScheme)EDITOR_GET("editors/3d/freelook/freelook_navigation_scheme").operator int();

	Vector3 forward;
	if (navigation_scheme == FREELOOK_FULLY_AXIS_LOCKED) {
		// Forward/backward keys will always go straight forward/backward, never moving on the Y axis.
		forward = Vector3(0, 0, -1).rotated(Vector3(0, 1, 0), camera_transform.basis.get_euler().y);
	}
	else {
		// Forward/backward keys will be relative to the camera pitch.
		forward = camera_transform.basis.xform(Vector3(0, 0, -1));
	}

	const Vector3 right = camera_transform.basis.xform(Vector3(1, 0, 0));

	Vector3 up;
	if (navigation_scheme == FREELOOK_PARTIALLY_AXIS_LOCKED || navigation_scheme == FREELOOK_FULLY_AXIS_LOCKED) {
		// Up/down keys will always go up/down regardless of camera pitch.
		up = Vector3(0, 1, 0);
	}
	else {
		// Up/down keys will be relative to the camera pitch.
		up = camera_transform.basis.xform(Vector3(0, 1, 0));
	}
	Vector3 direction = (right * p_direction.x) + (up * p_direction.y) + (forward * p_direction.z);
	const Vector3 motion = direction * p_speed * p_delta;
	move(motion);
}

void Node3DEditorCameraCursor::move_distance(real_t p_delta) {
	target_values.distance = MAX(0.0, target_values.distance + p_delta);
	recalculate_eye_position(target_values);
}

void Node3DEditorCameraCursor::move_distance_to(real_t p_distance) {
	target_values.distance = MAX(0.0, p_distance);
	recalculate_eye_position(target_values);
}

void Node3DEditorCameraCursor::stop_interpolation(bool p_go_to_target) {
	if (p_go_to_target) {
		current_values = target_values;
	}
	else {
		target_values = current_values;
	}
}

bool Node3DEditorCameraCursor::update_interpolation(float p_interp_delta) {
	Values old_values = current_values;
	current_values = target_values;

	if (freelook_mode) {
		// Higher inertia should increase "lag" (lerp with factor between 0 and 1)
		// Inertia of zero should produce instant movement (lerp with factor of 1) in this case it returns a really high value and gets clamped to 1.
		const real_t inertia = EDITOR_GET("editors/3d/freelook/freelook_inertia");
		real_t factor = (1.0 / inertia) * p_interp_delta;

		// We interpolate a different point here, because in freelook mode the focus point (cursor.pos) orbits around eye_pos
		current_values.eye_position = old_values.eye_position.lerp(target_values.eye_position, CLAMP(factor, 0, 1));

		const real_t orbit_inertia = EDITOR_GET("editors/3d/navigation_feel/orbit_inertia");
		current_values.x_rot = Math::lerp(old_values.x_rot, target_values.x_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));
		current_values.y_rot = Math::lerp(old_values.y_rot, target_values.y_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));

		if (Math::abs(current_values.x_rot - target_values.x_rot) < 0.1) {
			current_values.x_rot = target_values.x_rot;
		}

		if (Math::abs(current_values.y_rot - target_values.y_rot) < 0.1) {
			current_values.y_rot = target_values.y_rot;
		}

		recalculate_position(current_values);
	}
	else {
		const real_t orbit_inertia = EDITOR_GET("editors/3d/navigation_feel/orbit_inertia");
		const real_t translation_inertia = EDITOR_GET("editors/3d/navigation_feel/translation_inertia");
		const real_t zoom_inertia = EDITOR_GET("editors/3d/navigation_feel/zoom_inertia");

		current_values.x_rot = Math::lerp(old_values.x_rot, target_values.x_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));
		current_values.y_rot = Math::lerp(old_values.y_rot, target_values.y_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));

		if (Math::abs(current_values.x_rot - target_values.x_rot) < 0.1) {
			current_values.x_rot = target_values.x_rot;
		}

		if (Math::abs(current_values.y_rot - target_values.y_rot) < 0.1) {
			current_values.y_rot = target_values.y_rot;
		}

		current_values.position = old_values.position.lerp(target_values.position, MIN(1.f, p_interp_delta * (1 / translation_inertia)));
		current_values.distance = Math::lerp(old_values.distance, target_values.distance, MIN((real_t)1.0, p_interp_delta * (1 / zoom_inertia)));
		recalculate_eye_position(current_values);
	}

	real_t tolerance = 0.001;
	bool something_changed = false;
	if (!Math::is_equal_approx(old_values.x_rot, current_values.x_rot, tolerance) || !Math::is_equal_approx(old_values.y_rot, current_values.y_rot, tolerance)) {
		something_changed = true;
	}
	else if (!old_values.position.is_equal_approx(current_values.position)) {
		something_changed = true;
	}
	else if (!Math::is_equal_approx(old_values.distance, current_values.distance, tolerance)) {
		something_changed = true;
	}
	else if (!Math::is_equal_approx(old_values.fov_scale, current_values.fov_scale, tolerance)) {
		something_changed = true;
	}
	return something_changed;
}

void Node3DEditorCameraCursor::set_orthogonal(float p_z_near, float p_z_far) {
	orthogonal = true;
	z_near = p_z_near;
	z_far = p_z_far;
	recalculate_eye_position(current_values);
	recalculate_eye_position(target_values);
	stop_interpolation(true);
}

void Node3DEditorCameraCursor::set_perspective(float p_fov, float p_z_near, float p_z_far) {
	orthogonal = false;
	perspective_fov = p_fov;
	z_near = p_z_near;
	z_far = p_z_far;
	recalculate_eye_position(current_values);
	recalculate_eye_position(target_values);
	stop_interpolation(true);
}

Transform3D Node3DEditorCameraCursor::get_current_camera_transform() const {
	return values_to_camera_transform(current_values);
}

Transform3D Node3DEditorCameraCursor::get_target_camera_transform() const {
	return values_to_camera_transform(target_values);
}

void Node3DEditorCameraCursor::set_camera_transform(const Transform3D& p_transform) {
	target_values = Values();
	Transform3D transform = p_transform;
	Transform3D eye_transform = p_transform;
	transform.translate_local(0, 0, -target_values.distance);
	target_values.position = transform.origin;
	Vector3 euler = transform.basis.get_euler();
	target_values.x_rot = -euler.x;
	target_values.y_rot = -euler.y;
	recalculate_eye_position(target_values);
}

Transform3D Node3DEditorCameraCursor::values_to_camera_transform(const Values& p_values) const {
	Transform3D camera_transform;
	camera_transform.translate_local(p_values.position);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -p_values.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -p_values.y_rot);
	if (orthogonal) {
		camera_transform.translate_local(0, 0, (z_far - z_near) / 2.0);
	}
	else {
		camera_transform.translate_local(0, 0, p_values.distance);
	}
	return camera_transform;
}

void Node3DEditorCameraCursor::recalculate_eye_position(Values& p_values) {
	Vector3 forward = values_to_camera_transform(p_values).basis.xform(Vector3(0, 0, -1));
	p_values.eye_position = p_values.position - p_values.distance * forward;
}

void Node3DEditorCameraCursor::recalculate_position(Values& p_values) {
	Vector3 forward = values_to_camera_transform(p_values).basis.xform(Vector3(0, 0, -1));
	p_values.position = p_values.eye_position + forward * p_values.distance;
}

Node3DEditorCameraCursor::Node3DEditorCameraCursor() {
	freelook_mode = false;
	orthogonal = false;
	z_near = 0.0;
	z_far = 0.0;
	perspective_fov = 0.0;
	recalculate_eye_position(current_values);
	recalculate_eye_position(target_values);
}

//////
