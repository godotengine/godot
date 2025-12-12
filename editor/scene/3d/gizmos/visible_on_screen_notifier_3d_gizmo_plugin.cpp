/**************************************************************************/
/*  visible_on_screen_notifier_3d_gizmo_plugin.cpp                        */
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

#include "visible_on_screen_notifier_3d_gizmo_plugin.h"

#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/3d/gizmos/gizmo_3d_helper.h"
#include "editor/scene/3d/node_3d_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/3d/visible_on_screen_notifier_3d.h"

VisibleOnScreenNotifier3DGizmoPlugin::VisibleOnScreenNotifier3DGizmoPlugin() {
	helper.instantiate();
	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/visibility_notifier");
	create_material("visibility_notifier_material", gizmo_color);
	gizmo_color.a = 0.1;
	create_material("visibility_notifier_solid_material", gizmo_color);
	create_handle_material("handles");
}

VisibleOnScreenNotifier3DGizmoPlugin::~VisibleOnScreenNotifier3DGizmoPlugin() {
}

bool VisibleOnScreenNotifier3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<VisibleOnScreenNotifier3D>(p_spatial) != nullptr;
}

String VisibleOnScreenNotifier3DGizmoPlugin::get_gizmo_name() const {
	return "VisibleOnScreenNotifier3D";
}

int VisibleOnScreenNotifier3DGizmoPlugin::get_priority() const {
	return -1;
}

String VisibleOnScreenNotifier3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	return helper->box_get_handle_name(p_id);
}

Variant VisibleOnScreenNotifier3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());
	return notifier->get_aabb();
}

void VisibleOnScreenNotifier3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());
	helper->initialize_handle_action(notifier->get_aabb().get_size(), p_gizmo->get_node_3d()->get_global_transform());
}

void VisibleOnScreenNotifier3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	AABB aabb = notifier->get_aabb();
	int axis = p_id / 2;
	int sign = p_id % 2 * -2 + 1;

	float neg_end = aabb.position[axis];
	float pos_end = aabb.position[axis] + aabb.size[axis];

	Vector3 new_size = Vector3(aabb.size);
	Vector3 new_position;
	helper->box_set_handle(sg, p_id, new_size, new_position);

	// Adjust position
	if (Input::get_singleton()->is_key_pressed(Key::ALT)) {
		new_position = aabb.position + (aabb.get_size() - new_size) * 0.5;
	} else {
		new_position = aabb.position;
		if (sign > 0) {
			pos_end = neg_end + new_size[axis];
		} else {
			neg_end = pos_end - new_size[axis];
			new_position[axis] = neg_end;
		}
	}

	AABB new_aabb;
	new_aabb.position = new_position;
	new_aabb.size = new_size;
	notifier->set_aabb(new_aabb);
}

void VisibleOnScreenNotifier3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	if (p_cancel) {
		notifier->set_aabb(p_restore);
		return;
	}

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Change Notifier AABB"));
	ur->add_do_method(notifier, "set_aabb", notifier->get_aabb());
	ur->add_undo_method(notifier, "set_aabb", p_restore);
	ur->commit_action();
}

void VisibleOnScreenNotifier3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	VisibleOnScreenNotifier3D *notifier = Object::cast_to<VisibleOnScreenNotifier3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Vector<Vector3> lines;
	AABB aabb = notifier->get_aabb();

	for (int i = 0; i < 12; i++) {
		Vector3 a, b;
		aabb.get_edge(i, a, b);
		lines.push_back(a);
		lines.push_back(b);
	}

	Vector<Vector3> handles = helper->box_get_handles(aabb.get_size());
	// Offset handles to AABB center
	for (int i = 0; i < handles.size(); i++) {
		handles.write[i] = handles[i] + aabb.get_center();
	}

	Ref<Material> material = get_material("visibility_notifier_material", p_gizmo);

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);

	if (p_gizmo->is_selected()) {
		Ref<Material> solid_material = get_material("visibility_notifier_solid_material", p_gizmo);
		p_gizmo->add_solid_box(solid_material, aabb.get_size(), aabb.get_center());
	}

	p_gizmo->add_handles(handles, get_material("handles"));
}
