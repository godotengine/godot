/**************************************************************************/
/*  vr_window.cpp                                                         */
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

#include "vr_window.h"

#include "core/input/input_event.h"
#include "scene/resources/material.h"
#include "scene/resources/primitive_meshes.h"

/* VRCollisionWindow */

bool VRCollisionWindow::raycast(const Vector3 &p_global_origin, const Vector3 &p_global_dir, Vector3 &r_position) {
	if (!is_enabled()) {
		return false;
	}

	Size2 half_size = 0.5 * size;
	Transform3D transform = get_global_transform();

	// transform our inputs into local space, makes a lot of this work soo much easier
	Vector3 local_origin = transform.xform_inv(p_global_origin);
	Vector3 local_dir = transform.basis.xform_inv(p_global_dir);

	// We only support intersections with the "front" of our window so we can do some quick checks first
	if (local_origin.z < -curve_depth) {
		// Origin of our ray lies "behind" our window
		return false;
	}

	if (local_dir.z > 0.0) {
		// We're pointing away from our window
		return false;
	} else {
		// adjust our direction vector length so z = -1.0;
		local_dir /= -local_dir.z;
	}

	// project on our curved surface first

	// check at where we're intersecting at z = 0
	Vector3 intersect_test = local_origin + (local_dir * local_origin.z);
	if ((intersect_test.x < -half_size.x) || (intersect_test.x > half_size.x)) {
		// if we're "outside" our "bowl" we can't possibly intersect with the inner side
		return false;
	}

	if (curve_depth > 0.0) {
		float max_delta = 0.001;

		// We know we're "inside" our "bowl" so we intersect somewhere, let's find it
		float z = -curve_depth * cos(0.5 * Math_PI * intersect_test.x / half_size.x);
		if (abs(intersect_test.z - z) > max_delta) {
			// remember our start
			Vector3 start_test = intersect_test;

			// check our end
			intersect_test = start_test + (local_dir * curve_depth);
			z = -curve_depth * cos(0.5 * Math_PI * intersect_test.x / half_size.x);

			if (abs(intersect_test.z - z) > max_delta) {
				Vector3 end_test = intersect_test;
				int max_attempts = 10;

				// check at the half way point
				intersect_test = (start_test + end_test) * 0.5;
				z = -curve_depth * cos(0.5 * Math_PI * intersect_test.x / half_size.x);

				// find our closest point by raymarching
				for (int i = 0; i < max_attempts && abs(intersect_test.z - z) > max_delta; i++) {
					if (intersect_test.z > z) {
						start_test = intersect_test;
					} else {
						end_test = intersect_test;
					}

					intersect_test = (start_test + end_test) * 0.5;
					z = -curve_depth * cos(0.5 * Math_PI * intersect_test.x / half_size.x);
				}
			}
		}

		// Update to our actual depth found
		intersect_test.z = z;
	}

	// Check our height
	if ((intersect_test.y < -half_size.y) || (intersect_test.y > half_size.y)) {
		// to high/low, we're out of bounds
		return false;
	}

	// return our position
	r_position = transform.xform(intersect_test);

	// if we got here, we've got a positive hit test.
	return true;
}

bool VRCollisionWindow::within_sphere(const Vector3 &p_global_origin, float p_radius, Vector3 &r_position) {
	if (!is_enabled()) {
		return false;
	}

	Size2 half_size = 0.5 * size;
	Transform3D transform = get_global_transform();

	// transform our inputs into local space, makes a lot of this work soo much easier
	Vector3 local_origin = transform.xform_inv(p_global_origin);

	if ((local_origin.x < -half_size.x) || (local_origin.x > half_size.x)) {
		return false;
	}
	if ((local_origin.y < -half_size.y) || (local_origin.y > half_size.y)) {
		return false;
	}

	float z = -curve_depth * cos(Math_PI * local_origin.x / half_size.x);
	if (abs(local_origin.z - z) > p_radius) {
		return false;
	}

	// set our hit point
	local_origin.z = z;
	r_position = local_origin;

	return true;
}

/* VRWindow */

void VRWindow::input(const Ref<InputEvent> &p_event) {
	if (!window_is_visible) {
		// window not visible, don't process input...
		return;
	}

	// Only forward key based events...
	if (p_event->is_class("InputEventKey")) {
		const Ref<InputEventKey> event = p_event;
		print_line("input key " + keycode_get_string(event->get_keycode()) + (event->is_pressed() ? " pressed" : " released"));

		subviewport->push_input(p_event, false);
	} else if (p_event->is_class("InputEventAction")) {
		subviewport->push_input(p_event, false);
	}
}

Vector2 VRWindow::_calc_mouse_position(const Vector3 &p_position) {
	// convert to local space
	Vector3 local_position = mesh_instance->get_global_transform().xform_inv(p_position);

	// Due to the way we curve our monitor by purely updating the z-position, we can do a linear mapping here
	Vector2 position = (Vector2(local_position.x, local_position.y) + (mesh_size * 0.5)) / viewport_scale;
	position.x = CLAMP(position.x, 0.0, viewport_size.x);
	position.y = viewport_size.y - CLAMP(position.y, 0.0, viewport_size.y);

	return position;
}

void VRWindow::_on_interact_enter(const Vector3 &p_position) {
	// don't seem to have a good event for this

	last_position = _calc_mouse_position(p_position);
	// Reset the buttons state
	buttons_state = BitField<MouseButtonMask>();
}

void VRWindow::_on_interact_moved(const Vector3 &p_position, float p_pressure) {
	// Note, we potentially can have the left and right hand sent mixed signals here
	// For now we accept that.

	Vector2 pos = _calc_mouse_position(p_position);

	Ref<InputEventMouseMotion> mouse_event;
	mouse_event.instantiate();

	mouse_event->set_position(pos);
	mouse_event->set_global_position(pos);
	mouse_event->set_relative(pos - last_position);
	mouse_event->set_button_mask(buttons_state);
	mouse_event->set_pressure(p_pressure);

	subviewport->push_input(mouse_event, true);

	last_position = pos;
}

void VRWindow::_on_interact_leave(const Vector3 &p_position) {
	// don't seem to have a good event for this
	last_position = _calc_mouse_position(p_position);
	// Reset the buttons state
	buttons_state = BitField<MouseButtonMask>();
}

void VRWindow::_on_interact_pressed(const Vector3 &p_position, MouseButton p_button) {
	// Note, we potentially can have the left and right hand sent mixed signals here
	// For now we accept that.

	buttons_state.set_flag(mouse_button_to_mask(p_button));

	Vector2 pos = _calc_mouse_position(p_position);

	Ref<InputEventMouseButton> mouse_event;
	mouse_event.instantiate();

	mouse_event->set_button_index(p_button);
	mouse_event->set_pressed(true);
	mouse_event->set_position(pos);
	mouse_event->set_global_position(pos);
	mouse_event->set_button_mask(buttons_state);

	subviewport->push_input(mouse_event, true);
}

void VRWindow::_on_interact_scrolled(const Vector3 &p_position, const Vector2 p_scroll_delta) {
	Vector2 mouse_position = _calc_mouse_position(p_position);

	if (p_scroll_delta.y > 0) {
		_on_scroll_input(mouse_position, MouseButton::WHEEL_UP, p_scroll_delta.y);
	} else if (p_scroll_delta.y < 0) {
		_on_scroll_input(mouse_position, MouseButton::WHEEL_DOWN, -p_scroll_delta.y);
	}

	if (p_scroll_delta.x > 0) {
		_on_scroll_input(mouse_position, MouseButton::WHEEL_RIGHT, p_scroll_delta.x);
	} else if (p_scroll_delta.x < 0) {
		_on_scroll_input(mouse_position, MouseButton::WHEEL_LEFT, -p_scroll_delta.x);
	}
}

void VRWindow::_on_scroll_input(const Vector2 &p_position, MouseButton p_wheel_button, float p_delta) {
	buttons_state.set_flag(mouse_button_to_mask(p_wheel_button));
	Ref<InputEventMouseButton> scroll_press_event;
	scroll_press_event.instantiate();
	scroll_press_event->set_position(p_position);
	scroll_press_event->set_global_position(p_position);
	scroll_press_event->set_pressed(true);
	scroll_press_event->set_button_index(p_wheel_button);
	scroll_press_event->set_button_mask(buttons_state);
	scroll_press_event->set_factor(p_delta);
	subviewport->push_input(scroll_press_event, true);

	buttons_state.clear_flag(mouse_button_to_mask(p_wheel_button));
	Ref<InputEventMouseButton> scroll_release_event = scroll_press_event->duplicate();
	scroll_release_event->set_pressed(false);
	scroll_release_event->set_button_mask(buttons_state);
	subviewport->push_input(scroll_release_event, true);
}

void VRWindow::_on_interact_released(const Vector3 &p_position, MouseButton p_button) {
	// Note, we potentially can have the left and right hand sent mixed signals here
	// For now we accept that.

	buttons_state.clear_flag(mouse_button_to_mask(p_button));

	Vector2 pos = _calc_mouse_position(p_position);

	Ref<InputEventMouseButton> mouse_event;
	mouse_event.instantiate();

	mouse_event->set_button_index(p_button);
	mouse_event->set_pressed(false);
	mouse_event->set_position(pos);
	mouse_event->set_global_position(pos);
	mouse_event->set_button_mask(buttons_state);

	subviewport->push_input(mouse_event, true);
}

void VRWindow::set_window_is_visible(bool p_visible) {
	window_is_visible = p_visible;
	set_process_input(p_visible);

	// If not visible we don't hide the whole node, we just hide our display and disable rendering to the viewport.
	if (mesh_instance) {
		mesh_instance->set_visible(p_visible);
	}

	if (subviewport) {
		subviewport->set_update_mode(p_visible ? SubViewport::UPDATE_ALWAYS : SubViewport::UPDATE_DISABLED);
	}

	if (collision) {
		collision->set_enabled(p_visible);
	}
}

void VRWindow::set_transparent_background(bool p_is_transparent) {
	transparent_background = p_is_transparent;

	if (subviewport != nullptr && material.is_valid()) {
		subviewport->set_transparent_background(transparent_background);
		material->set_shader(transparent_background ? transparent_shader : opaque_shader);
	}
}

void VRWindow::set_curve_depth(float p_curve_depth) {
	curve_depth = p_curve_depth;

	if (material.is_valid()) {
		material->set_shader_parameter("curve_depth", curve_depth);
	}

	if (collision) {
		collision->set_curve_depth(curve_depth);
	}
}

VRWindow::VRWindow(Size2i p_viewport_size, real_t p_viewport_scale) {
	viewport_size = p_viewport_size;
	viewport_scale = p_viewport_scale;
	mesh_size = viewport_size * viewport_scale;

	subviewport = memnew(SubViewport);

	subviewport->set_size(viewport_size);
	subviewport->set_clear_mode(SubViewport::CLEAR_MODE_ALWAYS);
	subviewport->set_update_mode(window_is_visible ? SubViewport::UPDATE_ALWAYS : SubViewport::UPDATE_DISABLED);
	subviewport->set_transparent_background(transparent_background);
	subviewport->set_disable_3d(true);
	subviewport->set_embedding_subwindows(true); // We don't support opening new windows in VR (yet) so make sure we embed popups etc.

	add_child(subviewport);

	// Create our shader for our viewport material
	opaque_shader.instantiate();
	opaque_shader->set_code(R"(
shader_type spatial;
render_mode unshaded, shadows_disabled, cull_disabled;

uniform sampler2D display_color : source_color;
uniform float curve_depth = 0.0;

void vertex() {
	VERTEX.z -= curve_depth * sin(UV.x * PI);
}

void fragment() {
	vec4 color = texture(display_color, UV);
	ALBEDO = color.rgb;
}
)");

	transparent_shader.instantiate();
	transparent_shader->set_code(R"(
shader_type spatial;
render_mode unshaded, shadows_disabled, cull_disabled;

uniform sampler2D display_color : source_color;
uniform float curve_depth = 0.0;

void vertex() {
	VERTEX.z -= curve_depth * sin(UV.x * PI);
}

void fragment() {
	vec4 color = texture(display_color, UV);
	ALBEDO = color.rgb;
	ALPHA = color.a;
}
)");

	// Create a material so we can show our viewport.
	material.instantiate();
	material->set_shader(transparent_background ? transparent_shader : opaque_shader);
	material->set_shader_parameter("display_color", subviewport->get_texture());
	material->set_shader_parameter("curve_depth", curve_depth);

	// Create a mesh to display our viewport in.
	Ref<QuadMesh> mesh;
	mesh.instantiate();
	mesh->set_size(mesh_size);
	mesh->set_subdivide_width(15); // subdivide horizontally so we can curve our display
	mesh->set_material(material);

	// Create a mesh instance to make the mesh visible
	mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_mesh(mesh);
	mesh_instance->set_visible(window_is_visible);
	add_child(mesh_instance);

	// Create our hit test
	collision = memnew(VRCollisionWindow);
	collision->set_size(mesh_size);
	collision->set_curve_depth(curve_depth);
	mesh_instance->add_child(collision);

	collision->connect("interact_enter", callable_mp(this, &VRWindow::_on_interact_enter));
	collision->connect("interact_moved", callable_mp(this, &VRWindow::_on_interact_moved));
	collision->connect("interact_leave", callable_mp(this, &VRWindow::_on_interact_leave));
	collision->connect("interact_pressed", callable_mp(this, &VRWindow::_on_interact_pressed));
	collision->connect("interact_released", callable_mp(this, &VRWindow::_on_interact_released));
	collision->connect("interact_scrolled", callable_mp(this, &VRWindow::_on_interact_scrolled));

	set_process_input(true);
}

VRWindow::~VRWindow() {
}
