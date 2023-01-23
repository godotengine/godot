/**************************************************************************/
/*  scene_preview.cpp                                                     */
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

#include "scene_preview.h"

#include "core/config/project_settings.h"
#include "core/object/object.h"
#include "editor/editor_scale.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/node_2d.h"
#include "scene/gui/texture_button.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/world_2d.h"

void Scene3DPreview::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
		rot_x -= mm->get_relative().y * 0.01;
		rot_y -= mm->get_relative().x * 0.01;
		if (rot_x < -Math_PI / 2) {
			rot_x = -Math_PI / 2;
		} else if (rot_x > Math_PI / 2) {
			rot_x = Math_PI / 2;
		}
		_update_rotation();
	}
}

void Scene3DPreview::_update_theme_item_cache() {
	SubViewportContainer::_update_theme_item_cache();

	theme_cache.light_1_on = get_theme_icon(SNAME("MaterialPreviewLight1"), SNAME("EditorIcons"));
	theme_cache.light_1_off = get_theme_icon(SNAME("MaterialPreviewLight1Off"), SNAME("EditorIcons"));
	theme_cache.light_2_on = get_theme_icon(SNAME("MaterialPreviewLight2"), SNAME("EditorIcons"));
	theme_cache.light_2_off = get_theme_icon(SNAME("MaterialPreviewLight2Off"), SNAME("EditorIcons"));
}

void Scene3DPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			light_1_switch->set_texture_normal(theme_cache.light_1_on);
			light_1_switch->set_texture_pressed(theme_cache.light_1_off);
			light_2_switch->set_texture_normal(theme_cache.light_2_on);
			light_2_switch->set_texture_pressed(theme_cache.light_2_off);
		} break;
		case NOTIFICATION_RESIZED: {
			set_custom_minimum_size(Vector2(1, MAX(150, get_size().width / 16.0 * 9.0)));
		} break;
	}
}

void Scene3DPreview::_update_rotation() {
	Transform3D t;
	t.basis.rotate(Vector3(0, 1, 0), -rot_y);
	t.basis.rotate(Vector3(1, 0, 0), -rot_x);
	rotation->set_transform(t);
}

void Scene3DPreview::edit(Node3D *p_node) {
	if (current != nullptr) {
		current->queue_free();
	}

	current = p_node;
	rotation->add_child(current);

	rot_x = Math::deg_to_rad(-15.0);
	rot_y = Math::deg_to_rad(30.0);
	_update_rotation();

	AABB aabb = _calculate_aabb(current);
	Vector3 ofs = aabb.get_center();
	float m = aabb.get_longest_axis_size();
	if (m != 0) {
		m = 1.0 / m;
		m *= 0.5;
		Transform3D xform;
		xform.basis.scale(Vector3(m, m, m));
		xform.origin = -xform.basis.xform(ofs); //-ofs*m;
		//xform.origin.z -= aabb.get_longest_axis_size() * 2;
		current->set_transform(xform);
	}
}

AABB Scene3DPreview::_calculate_aabb(Node3D *p_node) {
	AABB aabb;
	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(p_node);
	if (mesh_instance != nullptr) {
		aabb = mesh_instance->get_aabb();
	} else {
		aabb = AABB();
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node3D *child = Object::cast_to<Node3D>(p_node->get_child(i));
		AABB child_aabb = _calculate_aabb(child);
		aabb.merge_with(child->get_transform().xform(child_aabb));
	}
	return aabb;
}

void Scene3DPreview::_button_pressed(Node *p_button) {
	if (p_button == light_1_switch) {
		light1->set_visible(!light_1_switch->is_pressed());
	}

	if (p_button == light_2_switch) {
		light2->set_visible(!light_2_switch->is_pressed());
	}
}

Scene3DPreview::Scene3DPreview() {
	viewport = memnew(SubViewport);
	Ref<World3D> world_3d;
	world_3d.instantiate();
	viewport->set_world_3d(world_3d); //use own world
	add_child(viewport);
	viewport->set_disable_input(true);
	viewport->set_msaa_3d(Viewport::MSAA_4X);
	set_stretch(true);
	camera = memnew(Camera3D);
	camera->set_transform(Transform3D(Basis(), Vector3(0, 0, 1.1)));
	camera->set_perspective(45, 0.1, 10);
	viewport->add_child(camera);

	if (GLOBAL_GET("rendering/lights_and_shadows/use_physical_light_units")) {
		camera_attributes.instantiate();
		camera->set_attributes(camera_attributes);
	}

	light1 = memnew(DirectionalLight3D);
	light1->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
	viewport->add_child(light1);

	light2 = memnew(DirectionalLight3D);
	light2->set_transform(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
	light2->set_color(Color(0.7, 0.7, 0.7));
	viewport->add_child(light2);

	rotation = memnew(Node3D);
	viewport->add_child(rotation);

	set_custom_minimum_size(Size2(1, 150) * EDSCALE);

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

	hb->add_spacer();

	VBoxContainer *vb_light = memnew(VBoxContainer);
	hb->add_child(vb_light);

	light_1_switch = memnew(TextureButton);
	light_1_switch->set_toggle_mode(true);
	vb_light->add_child(light_1_switch);
	light_1_switch->connect("pressed", callable_mp(this, &Scene3DPreview::_button_pressed).bind(light_1_switch));

	light_2_switch = memnew(TextureButton);
	light_2_switch->set_toggle_mode(true);
	vb_light->add_child(light_2_switch);
	light_2_switch->connect("pressed", callable_mp(this, &Scene3DPreview::_button_pressed).bind(light_2_switch));

	rot_x = 0;
	rot_y = 0;
}
///////////////////////

void Scene2DPreview::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
		current->set_position(current->get_position() + mm->get_relative());
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::WHEEL_UP) {
			current->set_scale(current->get_scale() / 0.9);
		} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN) {
			current->set_scale(current->get_scale() * 0.9);
		}
	}
}

void Scene2DPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			set_custom_minimum_size(Vector2(1, MAX(150, get_size().width / 16.0 * 9.0)));
		} break;
	}
}

void Scene2DPreview::edit(Node2D *p_node) {
	if (current != nullptr) {
		current->queue_free();
	}

	current = p_node;
	viewport->add_child(p_node);
	//maybe todo: calculate bounding box of scene and move/scale to fit all
}

Scene2DPreview::Scene2DPreview() {
	viewport = memnew(SubViewport);
	Ref<World2D> world_2d;
	world_2d.instantiate();
	viewport->set_world_2d(world_2d);
	add_child(viewport);
	viewport->set_disable_input(true);
	set_stretch(true);
	set_custom_minimum_size(Size2(1, 150) * EDSCALE);
}

///////////////////////

void SceneControlPreview::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_RESIZED: {
			set_custom_minimum_size(Vector2(1, MAX(150, get_size().width / 16.0 * 9.0)));
		} break;
	}
}

void SceneControlPreview::edit(Control *p_node) {
	if (current != nullptr) {
		current->queue_free();
	}

	current = p_node;
	viewport->add_child(p_node);
}

SceneControlPreview::SceneControlPreview() {
	viewport = memnew(SubViewport);
	Ref<World2D> world_2d;
	world_2d.instantiate();
	viewport->set_world_2d(world_2d);
	add_child(viewport);
	viewport->set_disable_input(true);
	set_stretch(true);
	set_custom_minimum_size(Size2(1, 150) * EDSCALE);
}
