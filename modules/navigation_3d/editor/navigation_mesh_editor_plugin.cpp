/**************************************************************************/
/*  navigation_mesh_editor_plugin.cpp                                     */
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

#include "navigation_mesh_editor_plugin.h"

#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/main/viewport.h"
#include "scene/resources/mesh.h"
#include "scene/resources/navigation_mesh.h"

void NavigationMeshEditor::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && (mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
		rot_x -= mm->get_relative().y * 0.01;
		rot_y -= mm->get_relative().x * 0.01;

		rot_x = CLAMP(rot_x, -Math::PI / 2, Math::PI / 2);
		_update_rotation();
	}
}

void NavigationMeshEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			if (metadata_label) {
				Ref<Font> metadata_label_font = get_theme_font(SNAME("expression"), EditorStringName(EditorFonts));
				metadata_label->add_theme_font_override(SceneStringName(font), metadata_label_font);
			}

			camera->get_environment()->set_bg_color(get_theme_color(SNAME("dark_color_2"), EditorStringName(Editor)));
		} break;
	}
}

void NavigationMeshEditor::_update_theme_item_cache() {
	SubViewportContainer::_update_theme_item_cache();
}

void NavigationMeshEditor::_update_rotation() {
	Transform3D t;
	t.basis.rotate(Vector3(0, 1, 0), -rot_y);
	t.basis.rotate(Vector3(1, 0, 0), -rot_x);
	rotation->set_transform(t);
}

void NavigationMeshEditor::edit(Ref<NavigationMesh> p_navigation_mesh) {
	if (navigation_mesh.is_valid()) {
		navigation_mesh->disconnect_changed(callable_mp(this, &NavigationMeshEditor::_navigation_mesh_changed));
	}

	navigation_mesh = p_navigation_mesh;

	if (navigation_mesh.is_valid()) {
		navigation_mesh->connect_changed(callable_mp(this, &NavigationMeshEditor::_navigation_mesh_changed));
	}

	_navigation_mesh_changed();
}

void NavigationMeshEditor::_navigation_mesh_changed() {
	if (navigation_mesh.is_valid() && navigation_mesh->get_polygon_count() > 0) {
		metadata_label->set_text(
				vformat(TTR("%d Vertices") + "\n" + TTR("%d Polygons"),
						navigation_mesh->get_vertices().size(),
						navigation_mesh->get_polygon_count()));

		mesh = navigation_mesh->get_debug_mesh();
		mesh_instance->set_mesh(mesh);
	} else {
		metadata_label->set_text("");
		mesh = Ref<Mesh>();
		mesh_instance->set_mesh(mesh);
	}

	rot_x = Math::deg_to_rad(-15.0);
	rot_y = Math::deg_to_rad(30.0);
	_update_rotation();

	if (mesh.is_valid()) {
		AABB aabb = mesh->get_aabb();
		Vector3 ofs = aabb.get_center();
		float m = aabb.get_longest_axis_size();
		if (m != 0) {
			m = 1.0 / m;
			m *= 0.5;
			Transform3D xform;
			xform.basis.scale(Vector3(m, m, m));
			xform.origin = -xform.basis.xform(ofs);
			mesh_instance->set_transform(xform);
		}
	}
}

NavigationMeshEditor::NavigationMeshEditor() {
	viewport = memnew(SubViewport);
	viewport->set_debug_draw(Viewport::DebugDraw::DEBUG_DRAW_UNSHADED);
	Ref<World3D> world_3d;
	world_3d.instantiate();
	viewport->set_world_3d(world_3d);
	add_child(viewport);
	viewport->set_disable_input(true);
	viewport->set_msaa_3d(Viewport::MSAA_4X);
	set_stretch(true);
	camera = memnew(Camera3D);
	camera->set_transform(Transform3D(Basis(), Vector3(0, 0, 1.1)));
	camera->set_perspective(45, 0.1, 10);
	viewport->add_child(camera);

	Ref<Environment> env;
	env.instantiate();
	env->set_background(Environment::BG_COLOR);
	env->set_bg_color(Color(0.01, 0.01, 0.01, 1.0));
	camera->set_environment(env);

	rotation = memnew(Node3D);
	viewport->add_child(rotation);
	mesh_instance = memnew(MeshInstance3D);
	mesh_instance->set_cast_shadows_setting(GeometryInstance3D::ShadowCastingSetting::SHADOW_CASTING_SETTING_OFF);
	rotation->add_child(mesh_instance);

	set_custom_minimum_size(Size2(1, 150) * EDSCALE);

	metadata_label = memnew(Label);
	metadata_label->set_focus_mode(FOCUS_ACCESSIBILITY);
	metadata_label->add_theme_color_override(SceneStringName(font_color), Color(1, 1, 1));
	metadata_label->add_theme_color_override("font_shadow_color", Color(0, 0, 0));

	metadata_label->add_theme_font_size_override(SceneStringName(font_size), 14 * EDSCALE);
	metadata_label->add_theme_color_override("font_outline_color", Color(0, 0, 0));
	metadata_label->add_theme_constant_override("outline_size", 8 * EDSCALE);

	metadata_label->set_h_size_flags(Control::SIZE_SHRINK_END);
	metadata_label->set_v_size_flags(Control::SIZE_SHRINK_END);

	MarginContainer *margin_container = memnew(MarginContainer);
	margin_container->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);
	add_child(margin_container);
	margin_container->add_child(metadata_label);

	rot_x = 0;
	rot_y = 0;

	EditorNode::get_singleton()->register_hdr_viewport(viewport);
}

///////////////////////

bool EditorInspectorPluginNavigationMesh::can_handle(Object *p_object) {
	return Object::cast_to<NavigationMesh>(p_object) != nullptr;
}

void EditorInspectorPluginNavigationMesh::parse_begin(Object *p_object) {
	NavigationMesh *navmesh = Object::cast_to<NavigationMesh>(p_object);
	if (!navmesh) {
		return;
	}
	Ref<NavigationMesh> m(navmesh);

	NavigationMeshEditor *editor = memnew(NavigationMeshEditor);
	editor->edit(m);
	add_custom_control(editor);
}

NavigationMeshEditorPlugin::NavigationMeshEditorPlugin() {
	Ref<EditorInspectorPluginNavigationMesh> plugin;
	plugin.instantiate();
	add_inspector_plugin(plugin);
}
