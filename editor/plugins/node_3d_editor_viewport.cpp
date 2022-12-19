/*************************************************************************/
/*  node_3d_editor_viewport.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "node_3d_editor_viewport.h"

#include "animation_player_editor_plugin.h"
#include "core/config/project_settings.h"
#include "core/input/input_event.h"
#include "core/input/input_map.h"
#include "core/os/keyboard.h"
#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene_tree_dock.h"
#include "node_3d_editor.h"
#include "node_3d_editor_selected_item.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/decal.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/visual_instance_3d.h"
#include "scene/resources/packed_scene.h"
#include "viewport_navigation_control.h"
#include "viewport_rotation_control.h"

static Key _get_key_modifier_setting(const String &p_property) {
	switch (EDITOR_GET(p_property).operator int()) {
		case 0:
			return Key::NONE;
		case 1:
			return Key::SHIFT;
		case 2:
			return Key::ALT;
		case 3:
			return Key::META;
		case 4:
			return Key::CTRL;
	}
	return Key::NONE;
}

static Key _get_key_modifier(Ref<InputEventWithModifiers> e) {
	if (e->is_shift_pressed()) {
		return Key::SHIFT;
	}
	if (e->is_alt_pressed()) {
		return Key::ALT;
	}
	if (e->is_ctrl_pressed()) {
		return Key::CTRL;
	}
	if (e->is_meta_pressed()) {
		return Key::META;
	}
	return Key::NONE;
}

void Node3DEditorViewport::_view_settings_confirmed(real_t p_interp_delta) {
	// Set FOV override multiplier back to the default, so that the FOV
	// setting specified in the View menu is correctly applied.
	cursor.fov_scale = 1.0;

	_update_camera(p_interp_delta);
}

void Node3DEditorViewport::_update_navigation_controls_visibility() {
	bool show_viewport_rotation_gizmo = EDITOR_GET("editors/3d/navigation/show_viewport_rotation_gizmo") && (!previewing_cinema && !previewing_camera);
	rotation_control->set_visible(show_viewport_rotation_gizmo);

	bool show_viewport_navigation_gizmo = EDITOR_GET("editors/3d/navigation/show_viewport_navigation_gizmo") && (!previewing_cinema && !previewing_camera);
	position_control->set_visible(show_viewport_navigation_gizmo);
	look_control->set_visible(show_viewport_navigation_gizmo);
}

void Node3DEditorViewport::_update_camera(real_t p_interp_delta) {
	bool is_orthogonal = camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL;

	Cursor old_camera_cursor = camera_cursor;
	camera_cursor = cursor;

	if (p_interp_delta > 0) {
		//-------
		// Perform smoothing

		if (is_freelook_active()) {
			// Higher inertia should increase "lag" (lerp with factor between 0 and 1)
			// Inertia of zero should produce instant movement (lerp with factor of 1) in this case it returns a really high value and gets clamped to 1.
			const real_t inertia = EDITOR_GET("editors/3d/freelook/freelook_inertia");
			real_t factor = (1.0 / inertia) * p_interp_delta;

			// We interpolate a different point here, because in freelook mode the focus point (cursor.pos) orbits around eye_pos
			camera_cursor.eye_pos = old_camera_cursor.eye_pos.lerp(cursor.eye_pos, CLAMP(factor, 0, 1));

			const real_t orbit_inertia = EDITOR_GET("editors/3d/navigation_feel/orbit_inertia");
			camera_cursor.x_rot = Math::lerp(old_camera_cursor.x_rot, cursor.x_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));
			camera_cursor.y_rot = Math::lerp(old_camera_cursor.y_rot, cursor.y_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));

			if (Math::abs(camera_cursor.x_rot - cursor.x_rot) < 0.1) {
				camera_cursor.x_rot = cursor.x_rot;
			}

			if (Math::abs(camera_cursor.y_rot - cursor.y_rot) < 0.1) {
				camera_cursor.y_rot = cursor.y_rot;
			}

			Vector3 forward = to_camera_transform(camera_cursor).basis.xform(Vector3(0, 0, -1));
			camera_cursor.pos = camera_cursor.eye_pos + forward * camera_cursor.distance;

		} else {
			const real_t orbit_inertia = EDITOR_GET("editors/3d/navigation_feel/orbit_inertia");
			const real_t translation_inertia = EDITOR_GET("editors/3d/navigation_feel/translation_inertia");
			const real_t zoom_inertia = EDITOR_GET("editors/3d/navigation_feel/zoom_inertia");

			camera_cursor.x_rot = Math::lerp(old_camera_cursor.x_rot, cursor.x_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));
			camera_cursor.y_rot = Math::lerp(old_camera_cursor.y_rot, cursor.y_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));

			if (Math::abs(camera_cursor.x_rot - cursor.x_rot) < 0.1) {
				camera_cursor.x_rot = cursor.x_rot;
			}

			if (Math::abs(camera_cursor.y_rot - cursor.y_rot) < 0.1) {
				camera_cursor.y_rot = cursor.y_rot;
			}

			camera_cursor.pos = old_camera_cursor.pos.lerp(cursor.pos, MIN(1.f, p_interp_delta * (1 / translation_inertia)));
			camera_cursor.distance = Math::lerp(old_camera_cursor.distance, cursor.distance, MIN((real_t)1.0, p_interp_delta * (1 / zoom_inertia)));
		}
	}

	//-------
	// Apply camera transform

	real_t tolerance = 0.001;
	bool equal = true;
	if (!Math::is_equal_approx(old_camera_cursor.x_rot, camera_cursor.x_rot, tolerance) || !Math::is_equal_approx(old_camera_cursor.y_rot, camera_cursor.y_rot, tolerance)) {
		equal = false;
	} else if (!old_camera_cursor.pos.is_equal_approx(camera_cursor.pos)) {
		equal = false;
	} else if (!Math::is_equal_approx(old_camera_cursor.distance, camera_cursor.distance, tolerance)) {
		equal = false;
	} else if (!Math::is_equal_approx(old_camera_cursor.fov_scale, camera_cursor.fov_scale, tolerance)) {
		equal = false;
	}

	if (!equal || p_interp_delta == 0 || is_orthogonal != orthogonal) {
		camera->set_global_transform(to_camera_transform(camera_cursor));

		if (orthogonal) {
			float half_fov = Math::deg_to_rad(get_fov()) / 2.0;
			float height = 2.0 * cursor.distance * Math::tan(half_fov);
			camera->set_orthogonal(height, get_znear(), get_zfar());
		} else {
			camera->set_perspective(get_fov(), get_znear(), get_zfar());
		}

		update_transform_gizmo_view();
		rotation_control->queue_redraw();
		position_control->queue_redraw();
		look_control->queue_redraw();
		spatial_editor->update_grid();
	}
}

Transform3D Node3DEditorViewport::to_camera_transform(const Cursor &p_cursor) const {
	Transform3D camera_transform;
	camera_transform.translate_local(p_cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -p_cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -p_cursor.y_rot);

	if (orthogonal) {
		camera_transform.translate_local(0, 0, (get_zfar() - get_znear()) / 2.0);
	} else {
		camera_transform.translate_local(0, 0, p_cursor.distance);
	}

	return camera_transform;
}

int Node3DEditorViewport::get_selected_count() const {
	const HashMap<Node *, Object *> &selection = editor_selection->get_selection();

	int count = 0;

	for (const KeyValue<Node *, Object *> &E : selection) {
		Node3D *sp = Object::cast_to<Node3D>(E.key);
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		count++;
	}

	return count;
}

void Node3DEditorViewport::cancel_transform() {
	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
		Node3D *sp = Object::cast_to<Node3D>(E->get());
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		sp->set_global_transform(se->original);
	}

	finish_transform();
	set_message(TTR("Transform Aborted."), 3);
}

void Node3DEditorViewport::_update_shrink() {
	bool shrink = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_HALF_RESOLUTION));
	subviewport_container->set_stretch_shrink(shrink ? 2 : 1);
	subviewport_container->set_texture_filter(shrink ? TEXTURE_FILTER_NEAREST : TEXTURE_FILTER_PARENT_NODE);
}

float Node3DEditorViewport::get_znear() const {
	return CLAMP(spatial_editor->get_znear(), Node3DEditor::MIN_Z, Node3DEditor::MAX_Z);
}

float Node3DEditorViewport::get_zfar() const {
	return CLAMP(spatial_editor->get_zfar(), Node3DEditor::MIN_Z, Node3DEditor::MAX_Z);
}

float Node3DEditorViewport::get_fov() const {
	return CLAMP(spatial_editor->get_fov() * cursor.fov_scale, Node3DEditor::MIN_FOV, Node3DEditor::MAX_FOV);
}

Transform3D Node3DEditorViewport::_get_camera_transform() const {
	return camera->get_global_transform();
}

Vector3 Node3DEditorViewport::_get_camera_position() const {
	return _get_camera_transform().origin;
}

Point2 Node3DEditorViewport::_point_to_screen(const Vector3 &p_point) {
	return camera->unproject_position(p_point) * subviewport_container->get_stretch_shrink();
}

Vector3 Node3DEditorViewport::_get_ray_pos(const Vector2 &p_pos) const {
	return camera->project_ray_origin(p_pos / subviewport_container->get_stretch_shrink());
}

Vector3 Node3DEditorViewport::_get_camera_normal() const {
	return -_get_camera_transform().basis.get_column(2);
}

Vector3 Node3DEditorViewport::_get_ray(const Vector2 &p_pos) const {
	return camera->project_ray_normal(p_pos / subviewport_container->get_stretch_shrink());
}

void Node3DEditorViewport::_clear_selected() {
	_edit.gizmo = Ref<EditorNode3DGizmo>();
	_edit.gizmo_handle = -1;
	_edit.gizmo_handle_secondary = false;
	_edit.gizmo_initial_value = Variant();

	Node3D *selected = spatial_editor->get_single_selected_node();
	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;

	if (se && se->gizmo.is_valid()) {
		se->subgizmos.clear();
		se->gizmo->redraw();
		se->gizmo.unref();
		spatial_editor->update_transform_gizmo();
	} else {
		editor_selection->clear();
		Node3DEditor::get_singleton()->edit(nullptr);
	}
}

void Node3DEditorViewport::_select_clicked(bool p_allow_locked) {
	Node *node = Object::cast_to<Node3D>(ObjectDB::get_instance(clicked));
	Node3D *selected = Object::cast_to<Node3D>(node);
	clicked = ObjectID();

	if (!selected) {
		return;
	}

	if (!p_allow_locked) {
		// Replace the node by the group if grouped
		while (node && node != EditorNode::get_singleton()->get_edited_scene()->get_parent()) {
			Node3D *selected_tmp = Object::cast_to<Node3D>(node);
			if (selected_tmp && node->has_meta("_edit_group_")) {
				selected = selected_tmp;
			}
			node = node->get_parent();
		}
	}

	if (p_allow_locked || !_is_node_locked(selected)) {
		if (clicked_wants_append) {
			if (editor_selection->is_selected(selected)) {
				editor_selection->remove_node(selected);
			} else {
				editor_selection->add_node(selected);
			}
		} else {
			if (!editor_selection->is_selected(selected)) {
				editor_selection->clear();
				editor_selection->add_node(selected);
				EditorNode::get_singleton()->edit_node(selected);
			}
		}

		if (editor_selection->get_selected_node_list().size() == 1) {
			EditorNode::get_singleton()->edit_node(editor_selection->get_selected_node_list()[0]);
		}
	}
}

ObjectID Node3DEditorViewport::_select_ray(const Point2 &p_pos) const {
	Vector3 ray = _get_ray(p_pos);
	Vector3 pos = _get_ray_pos(p_pos);
	Vector2 shrinked_pos = p_pos / subviewport_container->get_stretch_shrink();

	if (viewport->get_debug_draw() == Viewport::DEBUG_DRAW_SDFGI_PROBES) {
		RS::get_singleton()->sdfgi_set_debug_probe_select(pos, ray);
	}

	Vector<ObjectID> instances = RenderingServer::get_singleton()->instances_cull_ray(pos, pos + ray * camera->get_far(), get_tree()->get_root()->get_world_3d()->get_scenario());
	HashSet<Ref<EditorNode3DGizmo>> found_gizmos;

	Node *edited_scene = get_tree()->get_edited_scene_root();
	ObjectID closest;
	Node *item = nullptr;
	float closest_dist = 1e20;

	for (int i = 0; i < instances.size(); i++) {
		Node3D *spat = Object::cast_to<Node3D>(ObjectDB::get_instance(instances[i]));

		if (!spat) {
			continue;
		}

		Vector<Ref<Node3DGizmo>> gizmos = spat->get_gizmos();

		for (int j = 0; j < gizmos.size(); j++) {
			Ref<EditorNode3DGizmo> seg = gizmos[j];

			if ((!seg.is_valid()) || found_gizmos.has(seg)) {
				continue;
			}

			found_gizmos.insert(seg);
			Vector3 point;
			Vector3 normal;

			bool inters = seg->intersect_ray(camera, shrinked_pos, point, normal);

			if (!inters) {
				continue;
			}

			const real_t dist = pos.distance_to(point);

			if (dist < 0) {
				continue;
			}

			if (dist < closest_dist) {
				item = Object::cast_to<Node>(spat);
				if (item != edited_scene) {
					item = edited_scene->get_deepest_editable_node(item);
				}

				closest = item->get_instance_id();
				closest_dist = dist;
			}
		}
	}

	if (!item) {
		return ObjectID();
	}

	return closest;
}

void Node3DEditorViewport::_find_items_at_pos(const Point2 &p_pos, Vector<_RayResult> &r_results, bool p_include_locked_nodes) {
	Vector3 ray = _get_ray(p_pos);
	Vector3 pos = _get_ray_pos(p_pos);

	Vector<ObjectID> instances = RenderingServer::get_singleton()->instances_cull_ray(pos, pos + ray * camera->get_far(), get_tree()->get_root()->get_world_3d()->get_scenario());
	HashSet<Node3D *> found_nodes;

	for (int i = 0; i < instances.size(); i++) {
		Node3D *spat = Object::cast_to<Node3D>(ObjectDB::get_instance(instances[i]));

		if (!spat) {
			continue;
		}

		if (found_nodes.has(spat)) {
			continue;
		}

		if (!p_include_locked_nodes && _is_node_locked(spat)) {
			continue;
		}

		Vector<Ref<Node3DGizmo>> gizmos = spat->get_gizmos();
		for (int j = 0; j < gizmos.size(); j++) {
			Ref<EditorNode3DGizmo> seg = gizmos[j];

			if (!seg.is_valid()) {
				continue;
			}

			Vector3 point;
			Vector3 normal;

			bool inters = seg->intersect_ray(camera, p_pos, point, normal);

			if (!inters) {
				continue;
			}

			const real_t dist = pos.distance_to(point);

			if (dist < 0) {
				continue;
			}

			found_nodes.insert(spat);

			_RayResult res;
			res.item = spat;
			res.depth = dist;
			r_results.push_back(res);
			break;
		}
	}

	r_results.sort();
}

Vector3 Node3DEditorViewport::_get_screen_to_space(const Vector3 &p_vector3) {
	Projection cm;
	if (orthogonal) {
		cm.set_orthogonal(camera->get_size(), get_size().aspect(), get_znear() + p_vector3.z, get_zfar());
	} else {
		cm.set_perspective(get_fov(), get_size().aspect(), get_znear() + p_vector3.z, get_zfar());
	}
	Vector2 screen_he = cm.get_viewport_half_extents();

	Transform3D camera_transform;
	camera_transform.translate_local(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
	camera_transform.translate_local(0, 0, cursor.distance);

	return camera_transform.xform(Vector3(((p_vector3.x / get_size().width) * 2.0 - 1.0) * screen_he.x, ((1.0 - (p_vector3.y / get_size().height)) * 2.0 - 1.0) * screen_he.y, -(get_znear() + p_vector3.z)));
}

void Node3DEditorViewport::_select_region() {
	if (cursor.region_begin == cursor.region_end) {
		if (!clicked_wants_append) {
			_clear_selected();
		}
		return; //nothing really
	}

	const real_t z_offset = MAX(0.0, 5.0 - get_znear());

	Vector3 box[4] = {
		Vector3(
				MIN(cursor.region_begin.x, cursor.region_end.x),
				MIN(cursor.region_begin.y, cursor.region_end.y),
				z_offset),
		Vector3(
				MAX(cursor.region_begin.x, cursor.region_end.x),
				MIN(cursor.region_begin.y, cursor.region_end.y),
				z_offset),
		Vector3(
				MAX(cursor.region_begin.x, cursor.region_end.x),
				MAX(cursor.region_begin.y, cursor.region_end.y),
				z_offset),
		Vector3(
				MIN(cursor.region_begin.x, cursor.region_end.x),
				MAX(cursor.region_begin.y, cursor.region_end.y),
				z_offset)
	};

	Vector<Plane> frustum;

	Vector3 cam_pos = _get_camera_position();

	for (int i = 0; i < 4; i++) {
		Vector3 a = _get_screen_to_space(box[i]);
		Vector3 b = _get_screen_to_space(box[(i + 1) % 4]);
		if (orthogonal) {
			frustum.push_back(Plane((a - b).normalized(), a));
		} else {
			frustum.push_back(Plane(a, b, cam_pos));
		}
	}

	Plane near(-_get_camera_normal(), cam_pos);
	near.d -= get_znear();
	frustum.push_back(near);

	Plane far = -near;
	far.d += get_zfar();
	frustum.push_back(far);

	if (spatial_editor->get_single_selected_node()) {
		Node3D *single_selected = spatial_editor->get_single_selected_node();
		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(single_selected);

		if (se) {
			Ref<EditorNode3DGizmo> old_gizmo;
			if (!clicked_wants_append) {
				se->subgizmos.clear();
				old_gizmo = se->gizmo;
				se->gizmo.unref();
			}

			bool found_subgizmos = false;
			Vector<Ref<Node3DGizmo>> gizmos = single_selected->get_gizmos();
			for (int j = 0; j < gizmos.size(); j++) {
				Ref<EditorNode3DGizmo> seg = gizmos[j];
				if (!seg.is_valid()) {
					continue;
				}

				if (se->gizmo.is_valid() && se->gizmo != seg) {
					continue;
				}

				Vector<int> subgizmos = seg->subgizmos_intersect_frustum(camera, frustum);
				if (!subgizmos.is_empty()) {
					se->gizmo = seg;
					for (int i = 0; i < subgizmos.size(); i++) {
						int subgizmo_id = subgizmos[i];
						if (!se->subgizmos.has(subgizmo_id)) {
							se->subgizmos.insert(subgizmo_id, se->gizmo->get_subgizmo_transform(subgizmo_id));
						}
					}
					found_subgizmos = true;
					break;
				}
			}

			if (!clicked_wants_append || found_subgizmos) {
				if (se->gizmo.is_valid()) {
					se->gizmo->redraw();
				}

				if (old_gizmo != se->gizmo && old_gizmo.is_valid()) {
					old_gizmo->redraw();
				}

				spatial_editor->update_transform_gizmo();
			}

			if (found_subgizmos) {
				return;
			}
		}
	}

	if (!clicked_wants_append) {
		_clear_selected();
	}

	Vector<ObjectID> instances = RenderingServer::get_singleton()->instances_cull_convex(frustum, get_tree()->get_root()->get_world_3d()->get_scenario());
	HashSet<Node3D *> found_nodes;
	Vector<Node *> selected;

	Node *edited_scene = get_tree()->get_edited_scene_root();

	for (int i = 0; i < instances.size(); i++) {
		Node3D *sp = Object::cast_to<Node3D>(ObjectDB::get_instance(instances[i]));
		if (!sp || _is_node_locked(sp)) {
			continue;
		}

		if (found_nodes.has(sp)) {
			continue;
		}

		found_nodes.insert(sp);

		Node *item = Object::cast_to<Node>(sp);
		if (item != edited_scene) {
			item = edited_scene->get_deepest_editable_node(item);
		}

		// Replace the node by the group if grouped
		if (item->is_class("Node3D")) {
			Node3D *sel = Object::cast_to<Node3D>(item);
			while (item && item != EditorNode::get_singleton()->get_edited_scene()->get_parent()) {
				Node3D *selected_tmp = Object::cast_to<Node3D>(item);
				if (selected_tmp && item->has_meta("_edit_group_")) {
					sel = selected_tmp;
				}
				item = item->get_parent();
			}
			item = sel;
		}

		if (_is_node_locked(item)) {
			continue;
		}

		Vector<Ref<Node3DGizmo>> gizmos = sp->get_gizmos();
		for (int j = 0; j < gizmos.size(); j++) {
			Ref<EditorNode3DGizmo> seg = gizmos[j];
			if (!seg.is_valid()) {
				continue;
			}

			if (seg->intersect_frustum(camera, frustum)) {
				selected.push_back(item);
			}
		}
	}

	for (int i = 0; i < selected.size(); i++) {
		if (!editor_selection->is_selected(selected[i])) {
			editor_selection->add_node(selected[i]);
		}
	}

	if (editor_selection->get_selected_node_list().size() == 1) {
		EditorNode::get_singleton()->edit_node(editor_selection->get_selected_node_list()[0]);
	}
}

void Node3DEditorViewport::_update_name() {
	String name;

	switch (view_type) {
		case VIEW_TYPE_USER: {
			if (orthogonal) {
				name = TTR("Orthogonal");
			} else {
				name = TTR("Perspective");
			}
		} break;
		case VIEW_TYPE_TOP: {
			if (orthogonal) {
				name = TTR("Top Orthogonal");
			} else {
				name = TTR("Top Perspective");
			}
		} break;
		case VIEW_TYPE_BOTTOM: {
			if (orthogonal) {
				name = TTR("Bottom Orthogonal");
			} else {
				name = TTR("Bottom Perspective");
			}
		} break;
		case VIEW_TYPE_LEFT: {
			if (orthogonal) {
				name = TTR("Left Orthogonal");
			} else {
				name = TTR("Left Perspective");
			}
		} break;
		case VIEW_TYPE_RIGHT: {
			if (orthogonal) {
				name = TTR("Right Orthogonal");
			} else {
				name = TTR("Right Perspective");
			}
		} break;
		case VIEW_TYPE_FRONT: {
			if (orthogonal) {
				name = TTR("Front Orthogonal");
			} else {
				name = TTR("Front Perspective");
			}
		} break;
		case VIEW_TYPE_REAR: {
			if (orthogonal) {
				name = TTR("Rear Orthogonal");
			} else {
				name = TTR("Rear Perspective");
			}
		} break;
	}

	if (auto_orthogonal) {
		// TRANSLATORS: This will be appended to the view name when Auto Orthogonal is enabled.
		name += TTR(" [auto]");
	}

	view_menu->set_text(name);
	view_menu->reset_size();
}

void Node3DEditorViewport::_compute_edit(const Point2 &p_point) {
	_edit.original_local = spatial_editor->are_local_coords_enabled();
	_edit.click_ray = _get_ray(p_point);
	_edit.click_ray_pos = _get_ray_pos(p_point);
	_edit.plane = TRANSFORM_VIEW;
	spatial_editor->update_transform_gizmo();
	_edit.center = spatial_editor->get_gizmo_transform().origin;

	Node3D *selected = spatial_editor->get_single_selected_node();
	Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;

	if (se && se->gizmo.is_valid()) {
		for (const KeyValue<int, Transform3D> &E : se->subgizmos) {
			int subgizmo_id = E.key;
			se->subgizmos[subgizmo_id] = se->gizmo->get_subgizmo_transform(subgizmo_id);
		}
		se->original_local = selected->get_transform();
		se->original = selected->get_global_transform();
	} else {
		List<Node *> &selection = editor_selection->get_selected_node_list();

		for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
			Node3D *sp = Object::cast_to<Node3D>(E->get());
			if (!sp) {
				continue;
			}

			Node3DEditorSelectedItem *sel_item = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);

			if (!sel_item) {
				continue;
			}

			sel_item->original_local = sel_item->sp->get_local_gizmo_transform();
			sel_item->original = sel_item->sp->get_global_gizmo_transform();
		}
	}
}

bool Node3DEditorViewport::_transform_gizmo_select(const Vector2 &p_screenpos, bool p_highlight_only) {
	if (!spatial_editor->is_gizmo_visible()) {
		return false;
	}
	if (get_selected_count() == 0) {
		if (p_highlight_only) {
			spatial_editor->select_gizmo_highlight_axis(-1);
		}
		return false;
	}

	Vector3 ray_pos = _get_ray_pos(p_screenpos);
	Vector3 ray = _get_ray(p_screenpos);

	Transform3D gt = spatial_editor->get_gizmo_transform();

	if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE) {
		int col_axis = -1;
		real_t col_d = 1e20;

		for (int i = 0; i < 3; i++) {
			const Vector3 grabber_pos = gt.origin + gt.basis.get_column(i).normalized() * gizmo_scale * (Node3DEditor::GIZMO_ARROW_OFFSET + (Node3DEditor::GIZMO_ARROW_SIZE * 0.5));
			const real_t grabber_radius = gizmo_scale * Node3DEditor::GIZMO_ARROW_SIZE;

			Vector3 r;

			if (Geometry3D::segment_intersects_sphere(ray_pos, ray_pos + ray * Node3DEditor::MAX_Z, grabber_pos, grabber_radius, &r)) {
				const real_t d = r.distance_to(ray_pos);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		bool is_plane_translate = false;
		// plane select
		if (col_axis == -1) {
			col_d = 1e20;

			for (int i = 0; i < 3; i++) {
				Vector3 ivec2 = gt.basis.get_column((i + 1) % 3).normalized();
				Vector3 ivec3 = gt.basis.get_column((i + 2) % 3).normalized();

				// Allow some tolerance to make the plane easier to click,
				// even if the click is actually slightly outside the plane.
				const Vector3 grabber_pos = gt.origin + (ivec2 + ivec3) * gizmo_scale * (Node3DEditor::GIZMO_PLANE_SIZE + Node3DEditor::GIZMO_PLANE_DST * 0.6667);

				Vector3 r;
				Plane plane(gt.basis.get_column(i).normalized(), gt.origin);

				if (plane.intersects_ray(ray_pos, ray, &r)) {
					const real_t dist = r.distance_to(grabber_pos);
					// Allow some tolerance to make the plane easier to click,
					// even if the click is actually slightly outside the plane.
					if (dist < (gizmo_scale * Node3DEditor::GIZMO_PLANE_SIZE * 1.5)) {
						const real_t d = ray_pos.distance_to(r);
						if (d < col_d) {
							col_d = d;
							col_axis = i;

							is_plane_translate = true;
						}
					}
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				spatial_editor->select_gizmo_highlight_axis(col_axis + (is_plane_translate ? 6 : 0));

			} else {
				//handle plane translate
				_edit.mode = TRANSFORM_TRANSLATE;
				_compute_edit(p_screenpos);
				_edit.plane = TransformPlane(TRANSFORM_X_AXIS + col_axis + (is_plane_translate ? 3 : 0));
			}
			return true;
		}
	}

	if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE) {
		int col_axis = -1;

		Vector3 hit_position;
		Vector3 hit_normal;
		real_t ray_length = gt.origin.distance_to(ray_pos) + (Node3DEditor::GIZMO_CIRCLE_SIZE * gizmo_scale) * 4.0f;
		if (Geometry3D::segment_intersects_sphere(ray_pos, ray_pos + ray * ray_length, gt.origin, gizmo_scale * (Node3DEditor::GIZMO_CIRCLE_SIZE), &hit_position, &hit_normal)) {
			if (hit_normal.dot(_get_camera_normal()) < 0.05) {
				hit_position = gt.xform_inv(hit_position).abs();
				int min_axis = hit_position.min_axis_index();
				if (hit_position[min_axis] < gizmo_scale * Node3DEditor::GIZMO_RING_HALF_WIDTH) {
					col_axis = min_axis;
				}
			}
		}

		if (col_axis == -1) {
			float col_d = 1e20;

			for (int i = 0; i < 3; i++) {
				Plane plane(gt.basis.get_column(i).normalized(), gt.origin);
				Vector3 r;
				if (!plane.intersects_ray(ray_pos, ray, &r)) {
					continue;
				}

				const real_t dist = r.distance_to(gt.origin);
				const Vector3 r_dir = (r - gt.origin).normalized();

				if (_get_camera_normal().dot(r_dir) <= 0.005) {
					if (dist > gizmo_scale * (Node3DEditor::GIZMO_CIRCLE_SIZE - Node3DEditor::GIZMO_RING_HALF_WIDTH) && dist < gizmo_scale * (Node3DEditor::GIZMO_CIRCLE_SIZE + Node3DEditor::GIZMO_RING_HALF_WIDTH)) {
						const real_t d = ray_pos.distance_to(r);
						if (d < col_d) {
							col_d = d;
							col_axis = i;
						}
					}
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				spatial_editor->select_gizmo_highlight_axis(col_axis + 3);
			} else {
				//handle rotate
				_edit.mode = TRANSFORM_ROTATE;
				_compute_edit(p_screenpos);
				_edit.plane = TransformPlane(TRANSFORM_X_AXIS + col_axis);
			}
			return true;
		}
	}

	if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE) {
		int col_axis = -1;
		float col_d = 1e20;

		for (int i = 0; i < 3; i++) {
			const Vector3 grabber_pos = gt.origin + gt.basis.get_column(i).normalized() * gizmo_scale * Node3DEditor::GIZMO_SCALE_OFFSET;
			const real_t grabber_radius = gizmo_scale * Node3DEditor::GIZMO_ARROW_SIZE;

			Vector3 r;

			if (Geometry3D::segment_intersects_sphere(ray_pos, ray_pos + ray * Node3DEditor::MAX_Z, grabber_pos, grabber_radius, &r)) {
				const real_t d = r.distance_to(ray_pos);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		bool is_plane_scale = false;
		// plane select
		if (col_axis == -1) {
			col_d = 1e20;

			for (int i = 0; i < 3; i++) {
				const Vector3 ivec2 = gt.basis.get_column((i + 1) % 3).normalized();
				const Vector3 ivec3 = gt.basis.get_column((i + 2) % 3).normalized();

				// Allow some tolerance to make the plane easier to click,
				// even if the click is actually slightly outside the plane.
				const Vector3 grabber_pos = gt.origin + (ivec2 + ivec3) * gizmo_scale * (Node3DEditor::GIZMO_PLANE_SIZE + Node3DEditor::GIZMO_PLANE_DST * 0.6667);

				Vector3 r;
				Plane plane(gt.basis.get_column(i).normalized(), gt.origin);

				if (plane.intersects_ray(ray_pos, ray, &r)) {
					const real_t dist = r.distance_to(grabber_pos);
					// Allow some tolerance to make the plane easier to click,
					// even if the click is actually slightly outside the plane.
					if (dist < (gizmo_scale * Node3DEditor::GIZMO_PLANE_SIZE * 1.5)) {
						const real_t d = ray_pos.distance_to(r);
						if (d < col_d) {
							col_d = d;
							col_axis = i;

							is_plane_scale = true;
						}
					}
				}
			}
		}

		if (col_axis != -1) {
			if (p_highlight_only) {
				spatial_editor->select_gizmo_highlight_axis(col_axis + (is_plane_scale ? 12 : 9));

			} else {
				//handle scale
				_edit.mode = TRANSFORM_SCALE;
				_compute_edit(p_screenpos);
				_edit.plane = TransformPlane(TRANSFORM_X_AXIS + col_axis + (is_plane_scale ? 3 : 0));
			}
			return true;
		}
	}

	if (p_highlight_only) {
		spatial_editor->select_gizmo_highlight_axis(-1);
	}

	return false;
}

void Node3DEditorViewport::_transform_gizmo_apply(Node3D *p_node, const Transform3D &p_transform, bool p_local) {
	if (p_transform.basis.determinant() == 0) {
		return;
	}

	if (p_local) {
		p_node->set_transform(p_transform);
	} else {
		p_node->set_global_transform(p_transform);
	}
}

Transform3D Node3DEditorViewport::_compute_transform(TransformMode p_mode, const Transform3D &p_original, const Transform3D &p_original_local, Vector3 p_motion, double p_extra, bool p_local, bool p_orthogonal) {
	switch (p_mode) {
		case TRANSFORM_SCALE: {
			if (_edit.snap || spatial_editor->is_snap_enabled()) {
				p_motion.snap(Vector3(p_extra, p_extra, p_extra));
			}
			Transform3D s;
			if (p_local) {
				s.basis = p_original_local.basis.scaled_local(p_motion + Vector3(1, 1, 1));
				s.origin = p_original_local.origin;
			} else {
				s.basis.scale(p_motion + Vector3(1, 1, 1));
				Transform3D base = Transform3D(Basis(), _edit.center);
				s = base * (s * (base.inverse() * p_original));

				// Recalculate orthogonalized scale without moving origin.
				if (p_orthogonal) {
					s.basis = p_original_local.basis.scaled_orthogonal(p_motion + Vector3(1, 1, 1));
					// The scaled_orthogonal() does not require orthogonal Basis,
					// but it may make a bit skew by precision problems.
					s.basis.orthogonalize();
				}
			}

			return s;
		}
		case TRANSFORM_TRANSLATE: {
			if (_edit.snap || spatial_editor->is_snap_enabled()) {
				p_motion.snap(Vector3(p_extra, p_extra, p_extra));
			}

			if (p_local) {
				p_motion = p_original.basis.xform(p_motion);
			}

			// Apply translation
			Transform3D t = p_original;
			t.origin += p_motion;

			return t;
		}
		case TRANSFORM_ROTATE: {
			Transform3D r;

			if (p_local) {
				Vector3 axis = p_original_local.basis.xform(p_motion);
				r.basis = Basis(axis.normalized(), p_extra) * p_original_local.basis;
				r.origin = p_original_local.origin;
			} else {
				Basis local = p_original.basis * p_original_local.basis.inverse();
				Vector3 axis = local.xform_inv(p_motion);
				r.basis = local * Basis(axis.normalized(), p_extra) * p_original_local.basis;
				r.origin = Basis(p_motion, p_extra).xform(p_original.origin - _edit.center) + _edit.center;
			}

			return r;
		}
		default: {
			ERR_FAIL_V_MSG(Transform3D(), "Invalid mode in '_compute_transform'");
		}
	}
}

void Node3DEditorViewport::_surface_mouse_enter() {
	if (!surface->has_focus() && (!get_viewport()->gui_get_focus_owner() || !get_viewport()->gui_get_focus_owner()->is_text_field())) {
		surface->grab_focus();
	}
}

void Node3DEditorViewport::_surface_mouse_exit() {
	_remove_preview_node();
	_reset_preview_material();
	_remove_preview_material();
}

void Node3DEditorViewport::_surface_focus_enter() {
	view_menu->set_disable_shortcuts(false);
}

void Node3DEditorViewport::_surface_focus_exit() {
	view_menu->set_disable_shortcuts(true);
}

bool Node3DEditorViewport ::_is_node_locked(const Node *p_node) {
	return p_node->get_meta("_edit_lock_", false);
}

void Node3DEditorViewport::_list_select(Ref<InputEventMouseButton> b) {
	_find_items_at_pos(b->get_position(), selection_results, spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT);

	Node *scene = EditorNode::get_singleton()->get_edited_scene();

	for (int i = 0; i < selection_results.size(); i++) {
		Node3D *item = selection_results[i].item;
		if (item != scene && item->get_owner() != scene && item != scene->get_deepest_editable_node(item)) {
			//invalid result
			selection_results.remove_at(i);
			i--;
		}
	}

	clicked_wants_append = b->is_shift_pressed();

	if (selection_results.size() == 1) {
		clicked = selection_results[0].item->get_instance_id();
		selection_results.clear();

		if (clicked.is_valid()) {
			_select_clicked(spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT);
		}
	} else if (!selection_results.is_empty()) {
		NodePath root_path = get_tree()->get_edited_scene_root()->get_path();
		StringName root_name = root_path.get_name(root_path.get_name_count() - 1);

		for (int i = 0; i < selection_results.size(); i++) {
			Node3D *spat = selection_results[i].item;

			Ref<Texture2D> icon = EditorNode::get_singleton()->get_object_icon(spat, "Node");

			String node_path = "/" + root_name + "/" + root_path.rel_path_to(spat->get_path());

			int locked = 0;
			if (_is_node_locked(spat)) {
				locked = 1;
			} else {
				Node *ed_scene = EditorNode::get_singleton()->get_edited_scene();
				Node *node = spat;

				while (node && node != ed_scene->get_parent()) {
					Node3D *selected_tmp = Object::cast_to<Node3D>(node);
					if (selected_tmp && node->has_meta("_edit_group_")) {
						locked = 2;
					}
					node = node->get_parent();
				}
			}

			String suffix;
			if (locked == 1) {
				suffix = " (" + TTR("Locked") + ")";
			} else if (locked == 2) {
				suffix = " (" + TTR("Grouped") + ")";
			}
			selection_menu->add_item((String)spat->get_name() + suffix);
			selection_menu->set_item_icon(i, icon);
			selection_menu->set_item_metadata(i, node_path);
			selection_menu->set_item_tooltip(i, String(spat->get_name()) + "\nType: " + spat->get_class() + "\nPath: " + node_path);
		}

		selection_menu->set_position(get_screen_position() + b->get_position());
		selection_menu->reset_size();
		selection_menu->popup();
	}
}

void Node3DEditorViewport::_sinput(const Ref<InputEvent> &p_event) {
	if (previewing) {
		return; //do NONE
	}

	EditorPlugin::AfterGUIInput after = EditorPlugin::AFTER_GUI_INPUT_PASS;
	{
		EditorNode *en = EditorNode::get_singleton();
		EditorPluginList *force_input_forwarding_list = en->get_editor_plugins_force_input_forwarding();
		if (!force_input_forwarding_list->is_empty()) {
			EditorPlugin::AfterGUIInput discard = force_input_forwarding_list->forward_3d_gui_input(camera, p_event, true);
			if (discard == EditorPlugin::AFTER_GUI_INPUT_STOP) {
				return;
			}
			if (discard == EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
				after = EditorPlugin::AFTER_GUI_INPUT_CUSTOM;
			}
		}
	}
	{
		EditorNode *en = EditorNode::get_singleton();
		EditorPluginList *over_plugin_list = en->get_editor_plugins_over();
		if (!over_plugin_list->is_empty()) {
			EditorPlugin::AfterGUIInput discard = over_plugin_list->forward_3d_gui_input(camera, p_event, false);
			if (discard == EditorPlugin::AFTER_GUI_INPUT_STOP) {
				return;
			}
			if (discard == EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
				after = EditorPlugin::AFTER_GUI_INPUT_CUSTOM;
			}
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {
		emit_signal(SNAME("clicked"), this);

		const real_t zoom_factor = 1 + (Node3DEditor::ZOOM_FREELOOK_MULTIPLIER - 1) * b->get_factor();
		switch (b->get_button_index()) {
			case MouseButton::WHEEL_UP: {
				if (is_freelook_active()) {
					scale_freelook_speed(zoom_factor);
				} else {
					scale_cursor_distance(1.0 / zoom_factor);
				}
			} break;
			case MouseButton::WHEEL_DOWN: {
				if (is_freelook_active()) {
					scale_freelook_speed(1.0 / zoom_factor);
				} else {
					scale_cursor_distance(zoom_factor);
				}
			} break;
			case MouseButton::RIGHT: {
				NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();

				if (b->is_pressed() && _edit.gizmo.is_valid()) {
					//restore
					_edit.gizmo->commit_handle(_edit.gizmo_handle, _edit.gizmo_handle_secondary, _edit.gizmo_initial_value, true);
					_edit.gizmo = Ref<EditorNode3DGizmo>();
				}

				if (_edit.mode == TRANSFORM_NONE && b->is_pressed()) {
					if (b->is_alt_pressed()) {
						if (nav_scheme == NAVIGATION_MAYA) {
							break;
						}

						_list_select(b);
						return;
					}
				}

				if (_edit.mode != TRANSFORM_NONE && b->is_pressed()) {
					cancel_transform();
				}

				if (b->is_pressed()) {
					const Key mod = _get_key_modifier(b);
					if (!orthogonal) {
						if (mod == _get_key_modifier_setting("editors/3d/freelook/freelook_activation_modifier")) {
							set_freelook_active(true);
						}
					}
				} else {
					set_freelook_active(false);
				}

				if (freelook_active && !surface->has_focus()) {
					// Focus usually doesn't trigger on right-click, but in case of freelook it should,
					// otherwise using keyboard navigation would misbehave
					surface->grab_focus();
				}

			} break;
			case MouseButton::MIDDLE: {
				if (b->is_pressed() && _edit.mode != TRANSFORM_NONE) {
					switch (_edit.plane) {
						case TRANSFORM_VIEW: {
							_edit.plane = TRANSFORM_X_AXIS;
							set_message(TTR("X-Axis Transform."), 2);
							view_type = VIEW_TYPE_USER;
							_update_name();
						} break;
						case TRANSFORM_X_AXIS: {
							_edit.plane = TRANSFORM_Y_AXIS;
							set_message(TTR("Y-Axis Transform."), 2);

						} break;
						case TRANSFORM_Y_AXIS: {
							_edit.plane = TRANSFORM_Z_AXIS;
							set_message(TTR("Z-Axis Transform."), 2);

						} break;
						case TRANSFORM_Z_AXIS: {
							_edit.plane = TRANSFORM_VIEW;
							set_message(TTR("View Plane Transform."), 2);

						} break;
						case TRANSFORM_YZ:
						case TRANSFORM_XZ:
						case TRANSFORM_XY: {
						} break;
					}
				}
			} break;
			case MouseButton::LEFT: {
				if (b->is_pressed()) {
					clicked_wants_append = b->is_shift_pressed();

					if (_edit.mode != TRANSFORM_NONE && _edit.instant) {
						commit_transform();
						break; // just commit the edit, stop processing the event so we don't deselect the object
					}
					NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();
					if ((nav_scheme == NAVIGATION_MAYA || nav_scheme == NAVIGATION_MODO) && b->is_alt_pressed()) {
						break;
					}

					if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_LIST_SELECT) {
						_list_select(b);
						break;
					}

					_edit.mouse_pos = b->get_position();
					_edit.original_mouse_pos = b->get_position();
					_edit.snap = spatial_editor->is_snap_enabled();
					_edit.mode = TRANSFORM_NONE;
					_edit.original = spatial_editor->get_gizmo_transform(); // To prevent to break when flipping with scale.

					bool can_select_gizmos = spatial_editor->get_single_selected_node();

					{
						int idx = view_menu->get_popup()->get_item_index(VIEW_GIZMOS);
						can_select_gizmos = can_select_gizmos && view_menu->get_popup()->is_item_checked(idx);
					}

					// Gizmo handles
					if (can_select_gizmos) {
						Vector<Ref<Node3DGizmo>> gizmos = spatial_editor->get_single_selected_node()->get_gizmos();

						bool intersected_handle = false;
						for (int i = 0; i < gizmos.size(); i++) {
							Ref<EditorNode3DGizmo> seg = gizmos[i];

							if ((!seg.is_valid())) {
								continue;
							}

							int gizmo_handle = -1;
							bool gizmo_secondary = false;
							seg->handles_intersect_ray(camera, _edit.mouse_pos, b->is_shift_pressed(), gizmo_handle, gizmo_secondary);
							if (gizmo_handle != -1) {
								_edit.gizmo = seg;
								_edit.gizmo_handle = gizmo_handle;
								_edit.gizmo_handle_secondary = gizmo_secondary;
								_edit.gizmo_initial_value = seg->get_handle_value(gizmo_handle, gizmo_secondary);
								intersected_handle = true;
								break;
							}
						}

						if (intersected_handle) {
							break;
						}
					}

					// Transform gizmo
					if (_transform_gizmo_select(_edit.mouse_pos)) {
						break;
					}

					// Subgizmos
					if (can_select_gizmos) {
						Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(spatial_editor->get_single_selected_node());
						Vector<Ref<Node3DGizmo>> gizmos = spatial_editor->get_single_selected_node()->get_gizmos();

						bool intersected_subgizmo = false;
						for (int i = 0; i < gizmos.size(); i++) {
							Ref<EditorNode3DGizmo> seg = gizmos[i];

							if ((!seg.is_valid())) {
								continue;
							}

							int subgizmo_id = seg->subgizmos_intersect_ray(camera, _edit.mouse_pos);
							if (subgizmo_id != -1) {
								ERR_CONTINUE(!se);
								if (b->is_shift_pressed()) {
									if (se->subgizmos.has(subgizmo_id)) {
										se->subgizmos.erase(subgizmo_id);
									} else {
										se->subgizmos.insert(subgizmo_id, seg->get_subgizmo_transform(subgizmo_id));
									}
								} else {
									se->subgizmos.clear();
									se->subgizmos.insert(subgizmo_id, seg->get_subgizmo_transform(subgizmo_id));
								}

								if (se->subgizmos.is_empty()) {
									se->gizmo = Ref<EditorNode3DGizmo>();
								} else {
									se->gizmo = seg;
								}

								seg->redraw();
								spatial_editor->update_transform_gizmo();
								intersected_subgizmo = true;
								break;
							}
						}

						if (intersected_subgizmo) {
							break;
						}
					}

					clicked = ObjectID();

					if ((spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT && b->is_command_or_control_pressed()) || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE) {
						begin_transform(TRANSFORM_ROTATE, false);
						break;
					}

					if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE) {
						begin_transform(TRANSFORM_TRANSLATE, false);
						break;
					}

					if (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE) {
						begin_transform(TRANSFORM_SCALE, false);
						break;
					}

					if (after != EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
						//clicking is always deferred to either move or release
						clicked = _select_ray(b->get_position());
						selection_in_progress = true;

						if (clicked.is_null()) {
							//default to regionselect
							cursor.region_select = true;
							cursor.region_begin = b->get_position();
							cursor.region_end = b->get_position();
						}
					}

					surface->queue_redraw();
				} else {
					if (_edit.gizmo.is_valid()) {
						_edit.gizmo->commit_handle(_edit.gizmo_handle, _edit.gizmo_handle_secondary, _edit.gizmo_initial_value, false);
						_edit.gizmo = Ref<EditorNode3DGizmo>();
						break;
					}

					if (after != EditorPlugin::AFTER_GUI_INPUT_CUSTOM) {
						selection_in_progress = false;

						if (clicked.is_valid()) {
							_select_clicked(false);
						}

						if (cursor.region_select) {
							_select_region();
							cursor.region_select = false;
							surface->queue_redraw();
						}
					}

					if (_edit.mode != TRANSFORM_NONE) {
						Node3D *selected = spatial_editor->get_single_selected_node();
						Node3DEditorSelectedItem *se = selected ? editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(selected) : nullptr;

						if (se && se->gizmo.is_valid()) {
							Vector<int> ids;
							Vector<Transform3D> restore;

							for (const KeyValue<int, Transform3D> &GE : se->subgizmos) {
								ids.push_back(GE.key);
								restore.push_back(GE.value);
							}

							se->gizmo->commit_subgizmos(ids, restore, false);
						} else {
							commit_transform();
						}
						_edit.mode = TRANSFORM_NONE;
						set_message("");
						spatial_editor->update_transform_gizmo();
					}
					surface->queue_redraw();
				}

			} break;
			default:
				break;
		}
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {
		_edit.mouse_pos = m->get_position();

		if (spatial_editor->get_single_selected_node()) {
			Vector<Ref<Node3DGizmo>> gizmos = spatial_editor->get_single_selected_node()->get_gizmos();

			Ref<EditorNode3DGizmo> found_gizmo;
			int found_handle = -1;
			bool found_handle_secondary = false;

			for (int i = 0; i < gizmos.size(); i++) {
				Ref<EditorNode3DGizmo> seg = gizmos[i];
				if (!seg.is_valid()) {
					continue;
				}

				seg->handles_intersect_ray(camera, _edit.mouse_pos, false, found_handle, found_handle_secondary);

				if (found_handle != -1) {
					found_gizmo = seg;
					break;
				}
			}

			if (found_gizmo.is_valid()) {
				spatial_editor->select_gizmo_highlight_axis(-1);
			}

			bool current_hover_handle_secondary = false;
			int curreny_hover_handle = spatial_editor->get_current_hover_gizmo_handle(current_hover_handle_secondary);
			if (found_gizmo != spatial_editor->get_current_hover_gizmo() || found_handle != curreny_hover_handle || found_handle_secondary != current_hover_handle_secondary) {
				spatial_editor->set_current_hover_gizmo(found_gizmo);
				spatial_editor->set_current_hover_gizmo_handle(found_handle, found_handle_secondary);
				spatial_editor->get_single_selected_node()->update_gizmos();
			}
		}

		if (spatial_editor->get_current_hover_gizmo().is_null() && (m->get_button_mask() & MouseButton::MASK_LEFT) == MouseButton::NONE && !_edit.gizmo.is_valid()) {
			_transform_gizmo_select(_edit.mouse_pos, true);
		}

		NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();
		NavigationMode nav_mode = NAVIGATION_NONE;

		if (_edit.gizmo.is_valid()) {
			_edit.gizmo->set_handle(_edit.gizmo_handle, _edit.gizmo_handle_secondary, camera, m->get_position());
			Variant v = _edit.gizmo->get_handle_value(_edit.gizmo_handle, _edit.gizmo_handle_secondary);
			String n = _edit.gizmo->get_handle_name(_edit.gizmo_handle, _edit.gizmo_handle_secondary);
			set_message(n + ": " + String(v));

		} else if ((m->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE || _edit.instant) {
			if (nav_scheme == NAVIGATION_MAYA && m->is_alt_pressed()) {
				nav_mode = NAVIGATION_ORBIT;
			} else if (nav_scheme == NAVIGATION_MODO && m->is_alt_pressed() && m->is_shift_pressed()) {
				nav_mode = NAVIGATION_PAN;
			} else if (nav_scheme == NAVIGATION_MODO && m->is_alt_pressed() && m->is_ctrl_pressed()) {
				nav_mode = NAVIGATION_ZOOM;
			} else if (nav_scheme == NAVIGATION_MODO && m->is_alt_pressed()) {
				nav_mode = NAVIGATION_ORBIT;
			} else {
				const bool movement_threshold_passed = _edit.original_mouse_pos.distance_to(_edit.mouse_pos) > 8 * EDSCALE;

				// enable region-select if nothing has been selected yet or multi-select (shift key) is active
				if (selection_in_progress && movement_threshold_passed) {
					if (get_selected_count() == 0 || clicked_wants_append) {
						cursor.region_select = true;
						cursor.region_begin = _edit.original_mouse_pos;
						clicked = ObjectID();
					}
				}

				if (cursor.region_select) {
					cursor.region_end = m->get_position();
					surface->queue_redraw();
					return;
				}

				if (clicked.is_valid() && movement_threshold_passed) {
					_compute_edit(_edit.original_mouse_pos);
					clicked = ObjectID();
					_edit.mode = TRANSFORM_TRANSLATE;
				}

				if (_edit.mode == TRANSFORM_NONE) {
					return;
				}

				update_transform(m->get_position(), _get_key_modifier(m) == Key::SHIFT);
			}
		} else if ((m->get_button_mask() & MouseButton::MASK_RIGHT) != MouseButton::NONE || freelook_active) {
			if (nav_scheme == NAVIGATION_MAYA && m->is_alt_pressed()) {
				nav_mode = NAVIGATION_ZOOM;
			} else if (freelook_active) {
				nav_mode = NAVIGATION_LOOK;
			} else if (orthogonal) {
				nav_mode = NAVIGATION_PAN;
			}

		} else if ((m->get_button_mask() & MouseButton::MASK_MIDDLE) != MouseButton::NONE) {
			const Key mod = _get_key_modifier(m);
			if (nav_scheme == NAVIGATION_GODOT) {
				if (mod == _get_key_modifier_setting("editors/3d/navigation/pan_modifier")) {
					nav_mode = NAVIGATION_PAN;
				} else if (mod == _get_key_modifier_setting("editors/3d/navigation/zoom_modifier")) {
					nav_mode = NAVIGATION_ZOOM;
				} else if (mod == Key::ALT || mod == _get_key_modifier_setting("editors/3d/navigation/orbit_modifier")) {
					// Always allow Alt as a modifier to better support graphic tablets.
					nav_mode = NAVIGATION_ORBIT;
				}
			} else if (nav_scheme == NAVIGATION_MAYA) {
				if (mod == _get_key_modifier_setting("editors/3d/navigation/pan_modifier")) {
					nav_mode = NAVIGATION_PAN;
				}
			}
		} else if (EDITOR_GET("editors/3d/navigation/emulate_3_button_mouse")) {
			// Handle trackpad (no external mouse) use case
			const Key mod = _get_key_modifier(m);

			if (mod != Key::NONE) {
				if (mod == _get_key_modifier_setting("editors/3d/navigation/pan_modifier")) {
					nav_mode = NAVIGATION_PAN;
				} else if (mod == _get_key_modifier_setting("editors/3d/navigation/zoom_modifier")) {
					nav_mode = NAVIGATION_ZOOM;
				} else if (mod == Key::ALT || mod == _get_key_modifier_setting("editors/3d/navigation/orbit_modifier")) {
					// Always allow Alt as a modifier to better support graphic tablets.
					nav_mode = NAVIGATION_ORBIT;
				}
			}
		}

		switch (nav_mode) {
			case NAVIGATION_PAN: {
				_nav_pan(m, _get_warped_mouse_motion(m));

			} break;

			case NAVIGATION_ZOOM: {
				_nav_zoom(m, m->get_relative());

			} break;

			case NAVIGATION_ORBIT: {
				_nav_orbit(m, _get_warped_mouse_motion(m));

			} break;

			case NAVIGATION_LOOK: {
				_nav_look(m, _get_warped_mouse_motion(m));

			} break;

			default: {
			}
		}
	}

	Ref<InputEventMagnifyGesture> magnify_gesture = p_event;
	if (magnify_gesture.is_valid()) {
		if (is_freelook_active()) {
			scale_freelook_speed(magnify_gesture->get_factor());
		} else {
			scale_cursor_distance(1.0 / magnify_gesture->get_factor());
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();
		NavigationMode nav_mode = NAVIGATION_NONE;

		if (nav_scheme == NAVIGATION_GODOT) {
			const Key mod = _get_key_modifier(pan_gesture);

			if (mod == _get_key_modifier_setting("editors/3d/navigation/pan_modifier")) {
				nav_mode = NAVIGATION_PAN;
			} else if (mod == _get_key_modifier_setting("editors/3d/navigation/zoom_modifier")) {
				nav_mode = NAVIGATION_ZOOM;
			} else if (mod == Key::ALT || mod == _get_key_modifier_setting("editors/3d/navigation/orbit_modifier")) {
				// Always allow Alt as a modifier to better support graphic tablets.
				nav_mode = NAVIGATION_ORBIT;
			}

		} else if (nav_scheme == NAVIGATION_MAYA) {
			if (pan_gesture->is_alt_pressed()) {
				nav_mode = NAVIGATION_PAN;
			}
		}

		switch (nav_mode) {
			case NAVIGATION_PAN: {
				_nav_pan(pan_gesture, pan_gesture->get_delta());

			} break;

			case NAVIGATION_ZOOM: {
				_nav_zoom(pan_gesture, pan_gesture->get_delta());

			} break;

			case NAVIGATION_ORBIT: {
				_nav_orbit(pan_gesture, pan_gesture->get_delta());

			} break;

			case NAVIGATION_LOOK: {
				_nav_look(pan_gesture, pan_gesture->get_delta());

			} break;

			default: {
			}
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed()) {
			return;
		}

		if (EDITOR_GET("editors/3d/navigation/emulate_numpad")) {
			const Key code = k->get_physical_keycode();
			if (code >= Key::KEY_0 && code <= Key::KEY_9) {
				k->set_keycode(code - Key::KEY_0 + Key::KP_0);
			}
		}

		if (_edit.mode == TRANSFORM_NONE) {
			if (k->get_keycode() == Key::ESCAPE && !cursor.region_select) {
				_clear_selected();
				return;
			}
		} else {
			// We're actively transforming, handle keys specially
			TransformPlane new_plane = TRANSFORM_VIEW;
			String new_message;
			if (ED_IS_SHORTCUT("spatial_editor/lock_transform_x", p_event)) {
				new_plane = TRANSFORM_X_AXIS;
				new_message = TTR("X-Axis Transform.");
			} else if (ED_IS_SHORTCUT("spatial_editor/lock_transform_y", p_event)) {
				new_plane = TRANSFORM_Y_AXIS;
				new_message = TTR("Y-Axis Transform.");
			} else if (ED_IS_SHORTCUT("spatial_editor/lock_transform_z", p_event)) {
				new_plane = TRANSFORM_Z_AXIS;
				new_message = TTR("Z-Axis Transform.");
			} else if (_edit.mode != TRANSFORM_ROTATE) { // rotating on a plane doesn't make sense
				if (ED_IS_SHORTCUT("spatial_editor/lock_transform_yz", p_event)) {
					new_plane = TRANSFORM_YZ;
					new_message = TTR("YZ-Plane Transform.");
				} else if (ED_IS_SHORTCUT("spatial_editor/lock_transform_xz", p_event)) {
					new_plane = TRANSFORM_XZ;
					new_message = TTR("XZ-Plane Transform.");
				} else if (ED_IS_SHORTCUT("spatial_editor/lock_transform_xy", p_event)) {
					new_plane = TRANSFORM_XY;
					new_message = TTR("XY-Plane Transform.");
				}
			}

			if (new_plane != TRANSFORM_VIEW) {
				if (new_plane != _edit.plane) {
					// lock me once and get a global constraint
					_edit.plane = new_plane;
					spatial_editor->set_local_coords_enabled(false);
				} else if (!spatial_editor->are_local_coords_enabled()) {
					// lock me twice and get a local constraint
					spatial_editor->set_local_coords_enabled(true);
				} else {
					// lock me thrice and we're back where we started
					_edit.plane = TRANSFORM_VIEW;
					spatial_editor->set_local_coords_enabled(false);
				}
				update_transform(_edit.mouse_pos, Input::get_singleton()->is_key_pressed(Key::SHIFT));
				set_message(new_message, 2);
				accept_event();
				return;
			}
		}
		if (ED_IS_SHORTCUT("spatial_editor/snap", p_event)) {
			if (_edit.mode != TRANSFORM_NONE) {
				_edit.snap = !_edit.snap;
			}
		}
		if (ED_IS_SHORTCUT("spatial_editor/bottom_view", p_event)) {
			_menu_option(VIEW_BOTTOM);
		}
		if (ED_IS_SHORTCUT("spatial_editor/top_view", p_event)) {
			_menu_option(VIEW_TOP);
		}
		if (ED_IS_SHORTCUT("spatial_editor/rear_view", p_event)) {
			_menu_option(VIEW_REAR);
		}
		if (ED_IS_SHORTCUT("spatial_editor/front_view", p_event)) {
			_menu_option(VIEW_FRONT);
		}
		if (ED_IS_SHORTCUT("spatial_editor/left_view", p_event)) {
			_menu_option(VIEW_LEFT);
		}
		if (ED_IS_SHORTCUT("spatial_editor/right_view", p_event)) {
			_menu_option(VIEW_RIGHT);
		}
		if (ED_IS_SHORTCUT("spatial_editor/orbit_view_down", p_event)) {
			// Clamp rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
			cursor.x_rot = CLAMP(cursor.x_rot - Math_PI / 12.0, -1.57, 1.57);
			view_type = VIEW_TYPE_USER;
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/orbit_view_up", p_event)) {
			// Clamp rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
			cursor.x_rot = CLAMP(cursor.x_rot + Math_PI / 12.0, -1.57, 1.57);
			view_type = VIEW_TYPE_USER;
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/orbit_view_right", p_event)) {
			cursor.y_rot -= Math_PI / 12.0;
			view_type = VIEW_TYPE_USER;
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/orbit_view_left", p_event)) {
			cursor.y_rot += Math_PI / 12.0;
			view_type = VIEW_TYPE_USER;
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/orbit_view_180", p_event)) {
			cursor.y_rot += Math_PI;
			view_type = VIEW_TYPE_USER;
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/focus_origin", p_event)) {
			_menu_option(VIEW_CENTER_TO_ORIGIN);
		}
		if (ED_IS_SHORTCUT("spatial_editor/focus_selection", p_event)) {
			_menu_option(VIEW_CENTER_TO_SELECTION);
		}
		// Orthgonal mode doesn't work in freelook.
		if (!freelook_active && ED_IS_SHORTCUT("spatial_editor/switch_perspective_orthogonal", p_event)) {
			_menu_option(orthogonal ? VIEW_PERSPECTIVE : VIEW_ORTHOGONAL);
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/align_transform_with_view", p_event)) {
			_menu_option(VIEW_ALIGN_TRANSFORM_WITH_VIEW);
		}
		if (ED_IS_SHORTCUT("spatial_editor/align_rotation_with_view", p_event)) {
			_menu_option(VIEW_ALIGN_ROTATION_WITH_VIEW);
		}
		if (ED_IS_SHORTCUT("spatial_editor/insert_anim_key", p_event)) {
			if (!get_selected_count() || _edit.mode != TRANSFORM_NONE) {
				return;
			}

			if (!AnimationPlayerEditor::get_singleton()->get_track_editor()->has_keying()) {
				set_message(TTR("Keying is disabled (no key inserted)."));
				return;
			}

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				spatial_editor->emit_signal(SNAME("transform_key_request"), sp, "", sp->get_transform());
			}

			set_message(TTR("Animation Key Inserted."));
		}
		if (ED_IS_SHORTCUT("spatial_editor/cancel_transform", p_event) && _edit.mode != TRANSFORM_NONE) {
			cancel_transform();
		}
		if (!is_freelook_active()) {
			if (ED_IS_SHORTCUT("spatial_editor/instant_translate", p_event)) {
				begin_transform(TRANSFORM_TRANSLATE, true);
			}
			if (ED_IS_SHORTCUT("spatial_editor/instant_rotate", p_event)) {
				begin_transform(TRANSFORM_ROTATE, true);
			}
			if (ED_IS_SHORTCUT("spatial_editor/instant_scale", p_event)) {
				begin_transform(TRANSFORM_SCALE, true);
			}
		}

		// Freelook doesn't work in orthogonal mode.
		if (!orthogonal && ED_IS_SHORTCUT("spatial_editor/freelook_toggle", p_event)) {
			set_freelook_active(!is_freelook_active());

		} else if (k->get_keycode() == Key::ESCAPE) {
			set_freelook_active(false);
		}

		if (k->get_keycode() == Key::SPACE) {
			if (!k->is_pressed()) {
				emit_signal(SNAME("toggle_maximize_view"), this);
			}
		}

		if (ED_IS_SHORTCUT("spatial_editor/decrease_fov", p_event)) {
			scale_fov(-0.05);
		}

		if (ED_IS_SHORTCUT("spatial_editor/increase_fov", p_event)) {
			scale_fov(0.05);
		}

		if (ED_IS_SHORTCUT("spatial_editor/reset_fov", p_event)) {
			reset_fov();
		}
	}

	// freelook uses most of the useful shortcuts, like save, so its ok
	// to consider freelook active as end of the line for future events.
	if (freelook_active) {
		accept_event();
	}
}

void Node3DEditorViewport::_nav_pan(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative) {
	const NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();

	real_t pan_speed = 1 / 150.0;
	if (p_event.is_valid() && nav_scheme == NAVIGATION_MAYA && p_event->is_shift_pressed()) {
		pan_speed *= 10;
	}

	Transform3D camera_transform;

	camera_transform.translate_local(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
	const bool invert_x_axis = EDITOR_GET("editors/3d/navigation/invert_x_axis");
	const bool invert_y_axis = EDITOR_GET("editors/3d/navigation/invert_y_axis");
	Vector3 translation(
			(invert_x_axis ? -1 : 1) * -p_relative.x * pan_speed,
			(invert_y_axis ? -1 : 1) * p_relative.y * pan_speed,
			0);
	translation *= cursor.distance / Node3DEditor::DISTANCE_DEFAULT;
	camera_transform.translate_local(translation);
	cursor.pos = camera_transform.origin;
}

void Node3DEditorViewport::_nav_zoom(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative) {
	const NavigationScheme nav_scheme = (NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();

	real_t zoom_speed = 1 / 80.0;
	if (p_event.is_valid() && nav_scheme == NAVIGATION_MAYA && p_event->is_shift_pressed()) {
		zoom_speed *= 10;
	}

	NavigationZoomStyle zoom_style = (NavigationZoomStyle)EDITOR_GET("editors/3d/navigation/zoom_style").operator int();
	if (zoom_style == NAVIGATION_ZOOM_HORIZONTAL) {
		if (p_relative.x > 0) {
			scale_cursor_distance(1 - p_relative.x * zoom_speed);
		} else if (p_relative.x < 0) {
			scale_cursor_distance(1.0 / (1 + p_relative.x * zoom_speed));
		}
	} else {
		if (p_relative.y > 0) {
			scale_cursor_distance(1 + p_relative.y * zoom_speed);
		} else if (p_relative.y < 0) {
			scale_cursor_distance(1.0 / (1 - p_relative.y * zoom_speed));
		}
	}
}

void Node3DEditorViewport::_nav_orbit(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative) {
	if (lock_rotation) {
		_nav_pan(p_event, p_relative);
		return;
	}

	if (orthogonal && auto_orthogonal) {
		_menu_option(VIEW_PERSPECTIVE);
	}

	const real_t degrees_per_pixel = EDITOR_GET("editors/3d/navigation_feel/orbit_sensitivity");
	const real_t radians_per_pixel = Math::deg_to_rad(degrees_per_pixel);
	const bool invert_y_axis = EDITOR_GET("editors/3d/navigation/invert_y_axis");
	const bool invert_x_axis = EDITOR_GET("editors/3d/navigation/invert_x_axis");

	if (invert_y_axis) {
		cursor.x_rot -= p_relative.y * radians_per_pixel;
	} else {
		cursor.x_rot += p_relative.y * radians_per_pixel;
	}
	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.x_rot = CLAMP(cursor.x_rot, -1.57, 1.57);

	if (invert_x_axis) {
		cursor.y_rot -= p_relative.x * radians_per_pixel;
	} else {
		cursor.y_rot += p_relative.x * radians_per_pixel;
	}
	view_type = VIEW_TYPE_USER;
	_update_name();
}

void Node3DEditorViewport::_nav_look(Ref<InputEventWithModifiers> p_event, const Vector2 &p_relative) {
	if (orthogonal) {
		_nav_pan(p_event, p_relative);
		return;
	}

	if (orthogonal && auto_orthogonal) {
		_menu_option(VIEW_PERSPECTIVE);
	}

	// Scale mouse sensitivity with camera FOV scale when zoomed in to make it easier to point at things.
	const real_t degrees_per_pixel = real_t(EDITOR_GET("editors/3d/freelook/freelook_sensitivity")) * MIN(1.0, cursor.fov_scale);
	const real_t radians_per_pixel = Math::deg_to_rad(degrees_per_pixel);
	const bool invert_y_axis = EDITOR_GET("editors/3d/navigation/invert_y_axis");

	// Note: do NOT assume the camera has the "current" transform, because it is interpolated and may have "lag".
	const Transform3D prev_camera_transform = to_camera_transform(cursor);

	if (invert_y_axis) {
		cursor.x_rot -= p_relative.y * radians_per_pixel;
	} else {
		cursor.x_rot += p_relative.y * radians_per_pixel;
	}
	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.x_rot = CLAMP(cursor.x_rot, -1.57, 1.57);

	cursor.y_rot += p_relative.x * radians_per_pixel;

	// Look is like the opposite of Orbit: the focus point rotates around the camera
	Transform3D camera_transform = to_camera_transform(cursor);
	Vector3 pos = camera_transform.xform(Vector3(0, 0, 0));
	Vector3 prev_pos = prev_camera_transform.xform(Vector3(0, 0, 0));
	Vector3 diff = prev_pos - pos;
	cursor.pos += diff;

	view_type = VIEW_TYPE_USER;
	_update_name();
}

void Node3DEditorViewport::set_freelook_active(bool active_now) {
	if (!freelook_active && active_now) {
		// Sync camera cursor to cursor to "cut" interpolation jumps due to changing referential
		cursor = camera_cursor;

		// Make sure eye_pos is synced, because freelook referential is eye pos rather than orbit pos
		Vector3 forward = to_camera_transform(cursor).basis.xform(Vector3(0, 0, -1));
		cursor.eye_pos = cursor.pos - cursor.distance * forward;
		// Also sync the camera cursor, otherwise switching to freelook will be trippy if inertia is active
		camera_cursor.eye_pos = cursor.eye_pos;

		if (EDITOR_GET("editors/3d/freelook/freelook_speed_zoom_link")) {
			// Re-adjust freelook speed from the current zoom level
			real_t base_speed = EDITOR_GET("editors/3d/freelook/freelook_base_speed");
			freelook_speed = base_speed * cursor.distance;
		}

		previous_mouse_position = get_local_mouse_position();

		// Hide mouse like in an FPS (warping doesn't work)
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_CAPTURED);

	} else if (freelook_active && !active_now) {
		// Sync camera cursor to cursor to "cut" interpolation jumps due to changing referential
		cursor = camera_cursor;

		// Restore mouse
		Input::get_singleton()->set_mouse_mode(Input::MOUSE_MODE_VISIBLE);

		// Restore the previous mouse position when leaving freelook mode.
		// This is done because leaving `Input.MOUSE_MODE_CAPTURED` will center the cursor
		// due to OS limitations.
		warp_mouse(previous_mouse_position);
	}

	freelook_active = active_now;
}

void Node3DEditorViewport::scale_fov(real_t p_fov_offset) {
	cursor.fov_scale = CLAMP(cursor.fov_scale + p_fov_offset, 0.1, 2.5);
	surface->queue_redraw();
}

void Node3DEditorViewport::reset_fov() {
	cursor.fov_scale = 1.0;
	surface->queue_redraw();
}

void Node3DEditorViewport::scale_cursor_distance(real_t scale) {
	real_t min_distance = MAX(camera->get_near() * 4, Node3DEditor::ZOOM_FREELOOK_MIN);
	real_t max_distance = MIN(camera->get_far() / 4, Node3DEditor::ZOOM_FREELOOK_MAX);
	if (unlikely(min_distance > max_distance)) {
		cursor.distance = (min_distance + max_distance) / 2;
	} else {
		cursor.distance = CLAMP(cursor.distance * scale, min_distance, max_distance);
	}

	if (cursor.distance == max_distance || cursor.distance == min_distance) {
		zoom_failed_attempts_count++;
	} else {
		zoom_failed_attempts_count = 0;
	}

	zoom_indicator_delay = Node3DEditor::ZOOM_FREELOOK_INDICATOR_DELAY_S;
	surface->queue_redraw();
}

void Node3DEditorViewport::scale_freelook_speed(real_t scale) {
	real_t min_speed = MAX(camera->get_near() * 4, Node3DEditor::ZOOM_FREELOOK_MIN);
	real_t max_speed = MIN(camera->get_far() / 4, Node3DEditor::ZOOM_FREELOOK_MAX);
	if (unlikely(min_speed > max_speed)) {
		freelook_speed = (min_speed + max_speed) / 2;
	} else {
		freelook_speed = CLAMP(freelook_speed * scale, min_speed, max_speed);
	}

	zoom_indicator_delay = Node3DEditor::ZOOM_FREELOOK_INDICATOR_DELAY_S;
	surface->queue_redraw();
}

Point2i Node3DEditorViewport::_get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_ev_mouse_motion) const {
	Point2i relative;
	if (bool(EDITOR_GET("editors/3d/navigation/warped_mouse_panning"))) {
		relative = Input::get_singleton()->warp_mouse_motion(p_ev_mouse_motion, surface->get_global_rect());
	} else {
		relative = p_ev_mouse_motion->get_relative();
	}
	return relative;
}

void Node3DEditorViewport::_update_freelook(real_t delta) {
	if (!is_freelook_active()) {
		return;
	}

	const FreelookNavigationScheme navigation_scheme = (FreelookNavigationScheme)EDITOR_GET("editors/3d/freelook/freelook_navigation_scheme").operator int();

	Vector3 forward;
	if (navigation_scheme == FREELOOK_FULLY_AXIS_LOCKED) {
		// Forward/backward keys will always go straight forward/backward, never moving on the Y axis.
		forward = Vector3(0, 0, -1).rotated(Vector3(0, 1, 0), camera->get_rotation().y);
	} else {
		// Forward/backward keys will be relative to the camera pitch.
		forward = camera->get_transform().basis.xform(Vector3(0, 0, -1));
	}

	const Vector3 right = camera->get_transform().basis.xform(Vector3(1, 0, 0));

	Vector3 up;
	if (navigation_scheme == FREELOOK_PARTIALLY_AXIS_LOCKED || navigation_scheme == FREELOOK_FULLY_AXIS_LOCKED) {
		// Up/down keys will always go up/down regardless of camera pitch.
		up = Vector3(0, 1, 0);
	} else {
		// Up/down keys will be relative to the camera pitch.
		up = camera->get_transform().basis.xform(Vector3(0, 1, 0));
	}

	Vector3 direction;

	// Use actions from the inputmap, as this is the only way to reliably detect input in this method.
	// See #54469 for more discussion and explanation.
	Input *inp = Input::get_singleton();
	if (inp->is_action_pressed("spatial_editor/freelook_left")) {
		direction -= right;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_right")) {
		direction += right;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_forward")) {
		direction += forward;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_backwards")) {
		direction -= forward;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_up")) {
		direction += up;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_down")) {
		direction -= up;
	}

	real_t speed = freelook_speed;

	if (inp->is_action_pressed("spatial_editor/freelook_speed_modifier")) {
		speed *= 3.0;
	}
	if (inp->is_action_pressed("spatial_editor/freelook_slow_modifier")) {
		speed *= 0.333333;
	}

	const Vector3 motion = direction * speed * delta;
	cursor.pos += motion;
	cursor.eye_pos += motion;
}

void Node3DEditorViewport::set_message(String p_message, float p_time) {
	message = p_message;
	message_time = p_time;
}

void Node3DEditorViewport::_project_settings_changed() {
	//update shadow atlas if changed
	int shadowmap_size = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_size");
	bool shadowmap_16_bits = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_16_bits");
	int atlas_q0 = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_quadrant_0_subdiv");
	int atlas_q1 = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_quadrant_1_subdiv");
	int atlas_q2 = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_quadrant_2_subdiv");
	int atlas_q3 = GLOBAL_GET("rendering/lights_and_shadows/positional_shadow/atlas_quadrant_3_subdiv");

	viewport->set_positional_shadow_atlas_size(shadowmap_size);
	viewport->set_positional_shadow_atlas_16_bits(shadowmap_16_bits);
	viewport->set_positional_shadow_atlas_quadrant_subdiv(0, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q0));
	viewport->set_positional_shadow_atlas_quadrant_subdiv(1, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q1));
	viewport->set_positional_shadow_atlas_quadrant_subdiv(2, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q2));
	viewport->set_positional_shadow_atlas_quadrant_subdiv(3, Viewport::PositionalShadowAtlasQuadrantSubdiv(atlas_q3));

	_update_shrink();

	// Update MSAA, screen-space AA and debanding if changed

	const int msaa_mode = GLOBAL_GET("rendering/anti_aliasing/quality/msaa_3d");
	viewport->set_msaa_3d(Viewport::MSAA(msaa_mode));
	const int ssaa_mode = GLOBAL_GET("rendering/anti_aliasing/quality/screen_space_aa");
	viewport->set_screen_space_aa(Viewport::ScreenSpaceAA(ssaa_mode));
	const bool use_taa = GLOBAL_GET("rendering/anti_aliasing/quality/use_taa");
	viewport->set_use_taa(use_taa);

	const bool transparent_background = GLOBAL_GET("rendering/viewport/transparent_background");
	viewport->set_transparent_background(transparent_background);

	const bool use_debanding = GLOBAL_GET("rendering/anti_aliasing/quality/use_debanding");
	viewport->set_use_debanding(use_debanding);

	const bool use_occlusion_culling = GLOBAL_GET("rendering/occlusion_culling/use_occlusion_culling");
	viewport->set_use_occlusion_culling(use_occlusion_culling);

	const float mesh_lod_threshold = GLOBAL_GET("rendering/mesh_lod/lod_change/threshold_pixels");
	viewport->set_mesh_lod_threshold(mesh_lod_threshold);

	const Viewport::Scaling3DMode scaling_3d_mode = Viewport::Scaling3DMode(int(GLOBAL_GET("rendering/scaling_3d/mode")));
	viewport->set_scaling_3d_mode(scaling_3d_mode);

	const float scaling_3d_scale = GLOBAL_GET("rendering/scaling_3d/scale");
	viewport->set_scaling_3d_scale(scaling_3d_scale);

	const float fsr_sharpness = GLOBAL_GET("rendering/scaling_3d/fsr_sharpness");
	viewport->set_fsr_sharpness(fsr_sharpness);

	const float texture_mipmap_bias = GLOBAL_GET("rendering/textures/default_filters/texture_mipmap_bias");
	viewport->set_texture_mipmap_bias(texture_mipmap_bias);
}

void Node3DEditorViewport::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			EditorNode::get_singleton()->connect("project_settings_changed", callable_mp(this, &Node3DEditorViewport::_project_settings_changed));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			bool vp_visible = is_visible_in_tree();

			set_process(vp_visible);
			set_physics_process(vp_visible);

			if (vp_visible) {
				orthogonal = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_ORTHOGONAL));
				_update_name();
				_update_camera(0);
			} else {
				set_freelook_active(false);
			}
			call_deferred(SNAME("update_transform_gizmo_view"));
		} break;

		case NOTIFICATION_RESIZED: {
			call_deferred(SNAME("update_transform_gizmo_view"));
		} break;

		case NOTIFICATION_PROCESS: {
			real_t delta = get_process_delta_time();

			if (zoom_indicator_delay > 0) {
				zoom_indicator_delay -= delta;
				if (zoom_indicator_delay <= 0) {
					surface->queue_redraw();
					zoom_limit_label->hide();
				}
			}

			_update_navigation_controls_visibility();
			_update_freelook(delta);

			Node *scene_root = SceneTreeDock::get_singleton()->get_editor_data()->get_edited_scene_root();
			if (previewing_cinema && scene_root != nullptr) {
				Camera3D *cam = scene_root->get_viewport()->get_camera_3d();
				if (cam != nullptr && cam != previewing) {
					//then switch the viewport's camera to the scene's viewport camera
					if (previewing != nullptr) {
						previewing->disconnect("tree_exited", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
					}
					previewing = cam;
					previewing->connect("tree_exited", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
					RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), cam->get_camera());
					surface->queue_redraw();
				}
			}

			_update_camera(delta);

			const HashMap<Node *, Object *> &selection = editor_selection->get_selection();

			bool changed = false;
			bool exist = false;

			for (const KeyValue<Node *, Object *> &E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E.key);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				Transform3D t = sp->get_global_gizmo_transform();
				VisualInstance3D *vi = Object::cast_to<VisualInstance3D>(sp);
				AABB new_aabb = vi ? vi->get_aabb() : _calculate_spatial_bounds(sp);

				exist = true;
				if (se->last_xform == t && se->aabb == new_aabb && !se->last_xform_dirty) {
					continue;
				}
				changed = true;
				se->last_xform_dirty = false;
				se->last_xform = t;

				se->aabb = new_aabb;

				Transform3D t_offset = t;

				// apply AABB scaling before item's global transform
				{
					const Vector3 offset(0.005, 0.005, 0.005);
					Basis aabb_s;
					aabb_s.scale(se->aabb.size + offset);
					t.translate_local(se->aabb.position - offset / 2);
					t.basis = t.basis * aabb_s;
				}
				{
					const Vector3 offset(0.01, 0.01, 0.01);
					Basis aabb_s;
					aabb_s.scale(se->aabb.size + offset);
					t_offset.translate_local(se->aabb.position - offset / 2);
					t_offset.basis = t_offset.basis * aabb_s;
				}

				RenderingServer::get_singleton()->instance_set_transform(se->sbox_instance, t);
				RenderingServer::get_singleton()->instance_set_transform(se->sbox_instance_offset, t_offset);
				RenderingServer::get_singleton()->instance_set_transform(se->sbox_instance_xray, t);
				RenderingServer::get_singleton()->instance_set_transform(se->sbox_instance_xray_offset, t_offset);
			}

			if (changed || (spatial_editor->is_gizmo_visible() && !exist)) {
				spatial_editor->update_transform_gizmo();
			}

			if (message_time > 0) {
				if (message != last_message) {
					surface->queue_redraw();
					last_message = message;
				}

				message_time -= get_physics_process_delta_time();
				if (message_time < 0) {
					surface->queue_redraw();
				}
			}

			bool show_info = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_INFORMATION));
			if (show_info != info_label->is_visible()) {
				info_label->set_visible(show_info);
			}

			Camera3D *current_camera;

			if (previewing) {
				current_camera = previewing;
			} else {
				current_camera = camera;
			}

			if (show_info) {
				const String viewport_size = vformat(String::utf8("%d × %d"), viewport->get_size().x, viewport->get_size().y);
				String text;
				text += vformat(TTR("X: %s\n"), rtos(current_camera->get_position().x).pad_decimals(1));
				text += vformat(TTR("Y: %s\n"), rtos(current_camera->get_position().y).pad_decimals(1));
				text += vformat(TTR("Z: %s\n"), rtos(current_camera->get_position().z).pad_decimals(1));
				text += "\n";
				text += vformat(
						TTR("Size: %s (%.1fMP)\n"),
						viewport_size,
						viewport->get_size().x * viewport->get_size().y * 0.000001);

				text += "\n";
				text += vformat(TTR("Objects: %d\n"), viewport->get_render_info(Viewport::RENDER_INFO_TYPE_VISIBLE, Viewport::RENDER_INFO_OBJECTS_IN_FRAME));
				text += vformat(TTR("Primitive Indices: %d\n"), viewport->get_render_info(Viewport::RENDER_INFO_TYPE_VISIBLE, Viewport::RENDER_INFO_PRIMITIVES_IN_FRAME));
				text += vformat(TTR("Draw Calls: %d"), viewport->get_render_info(Viewport::RENDER_INFO_TYPE_VISIBLE, Viewport::RENDER_INFO_DRAW_CALLS_IN_FRAME));

				info_label->set_text(text);
			}

			// FPS Counter.
			bool show_fps = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_FRAME_TIME));

			if (show_fps != fps_label->is_visible()) {
				cpu_time_label->set_visible(show_fps);
				gpu_time_label->set_visible(show_fps);
				fps_label->set_visible(show_fps);
				RS::get_singleton()->viewport_set_measure_render_time(viewport->get_viewport_rid(), show_fps);
				for (int i = 0; i < FRAME_TIME_HISTORY; i++) {
					cpu_time_history[i] = 0;
					gpu_time_history[i] = 0;
				}
				cpu_time_history_index = 0;
				gpu_time_history_index = 0;
			}
			if (show_fps) {
				cpu_time_history[cpu_time_history_index] = RS::get_singleton()->viewport_get_measured_render_time_cpu(viewport->get_viewport_rid());
				cpu_time_history_index = (cpu_time_history_index + 1) % FRAME_TIME_HISTORY;
				double cpu_time = 0.0;
				for (int i = 0; i < FRAME_TIME_HISTORY; i++) {
					cpu_time += cpu_time_history[i];
				}
				cpu_time /= FRAME_TIME_HISTORY;
				// Prevent unrealistically low values.
				cpu_time = MAX(0.01, cpu_time);

				gpu_time_history[gpu_time_history_index] = RS::get_singleton()->viewport_get_measured_render_time_gpu(viewport->get_viewport_rid());
				gpu_time_history_index = (gpu_time_history_index + 1) % FRAME_TIME_HISTORY;
				double gpu_time = 0.0;
				for (int i = 0; i < FRAME_TIME_HISTORY; i++) {
					gpu_time += gpu_time_history[i];
				}
				gpu_time /= FRAME_TIME_HISTORY;
				// Prevent division by zero for the FPS counter (and unrealistically low values).
				// This limits the reported FPS to 100000.
				gpu_time = MAX(0.01, gpu_time);

				// Color labels depending on performance level ("good" = green, "OK" = yellow, "bad" = red).
				// Middle point is at 15 ms.
				cpu_time_label->set_text(vformat(TTR("CPU Time: %s ms"), rtos(cpu_time).pad_decimals(2)));
				cpu_time_label->add_theme_color_override(
						"font_color",
						frame_time_gradient->get_color_at_offset(
								Math::remap(cpu_time, 0, 30, 0, 1)));

				gpu_time_label->set_text(vformat(TTR("GPU Time: %s ms"), rtos(gpu_time).pad_decimals(2)));
				// Middle point is at 15 ms.
				gpu_time_label->add_theme_color_override(
						"font_color",
						frame_time_gradient->get_color_at_offset(
								Math::remap(gpu_time, 0, 30, 0, 1)));

				const double fps = 1000.0 / gpu_time;
				fps_label->set_text(vformat(TTR("FPS: %d"), fps));
				// Middle point is at 60 FPS.
				fps_label->add_theme_color_override(
						"font_color",
						frame_time_gradient->get_color_at_offset(
								Math::remap(fps, 110, 10, 0, 1)));
			}

			bool show_cinema = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_CINEMATIC_PREVIEW));
			cinema_label->set_visible(show_cinema);
			if (show_cinema) {
				float cinema_half_width = cinema_label->get_size().width / 2.0f;
				cinema_label->set_anchor_and_offset(SIDE_LEFT, 0.5f, -cinema_half_width);
			}

			if (lock_rotation) {
				float locked_half_width = locked_label->get_size().width / 2.0f;
				locked_label->set_anchor_and_offset(SIDE_LEFT, 0.5f, -locked_half_width);
			}
		} break;

		case NOTIFICATION_PHYSICS_PROCESS: {
			if (!update_preview_node) {
				return;
			}
			if (preview_node->is_inside_tree()) {
				preview_node_pos = spatial_editor->snap_point(_get_instance_position(preview_node_viewport_pos));
				Transform3D preview_gl_transform = Transform3D(Basis(), preview_node_pos);
				preview_node->set_global_transform(preview_gl_transform);
				if (!preview_node->is_visible()) {
					preview_node->show();
				}
			}
			update_preview_node = false;
		} break;

		case NOTIFICATION_ENTER_TREE: {
			surface->connect("draw", callable_mp(this, &Node3DEditorViewport::_draw));
			surface->connect("gui_input", callable_mp(this, &Node3DEditorViewport::_sinput));
			surface->connect("mouse_entered", callable_mp(this, &Node3DEditorViewport::_surface_mouse_enter));
			surface->connect("mouse_exited", callable_mp(this, &Node3DEditorViewport::_surface_mouse_exit));
			surface->connect("focus_entered", callable_mp(this, &Node3DEditorViewport::_surface_focus_enter));
			surface->connect("focus_exited", callable_mp(this, &Node3DEditorViewport::_surface_focus_exit));

			_init_gizmo_instance(index);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_finish_gizmo_instances();
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			view_menu->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));
			preview_camera->set_icon(get_theme_icon(SNAME("Camera3D"), SNAME("EditorIcons")));
			Control *gui_base = EditorNode::get_singleton()->get_gui_base();

			view_menu->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			view_menu->add_theme_style_override("hover", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			view_menu->add_theme_style_override("pressed", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			view_menu->add_theme_style_override("focus", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			view_menu->add_theme_style_override("disabled", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));

			preview_camera->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			preview_camera->add_theme_style_override("hover", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			preview_camera->add_theme_style_override("pressed", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			preview_camera->add_theme_style_override("focus", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			preview_camera->add_theme_style_override("disabled", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));

			frame_time_gradient->set_color(0, get_theme_color(SNAME("success_color"), SNAME("Editor")));
			frame_time_gradient->set_color(1, get_theme_color(SNAME("warning_color"), SNAME("Editor")));
			frame_time_gradient->set_color(2, get_theme_color(SNAME("error_color"), SNAME("Editor")));

			info_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			cpu_time_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			gpu_time_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			fps_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			cinema_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
			locked_label->add_theme_style_override("normal", gui_base->get_theme_stylebox(SNAME("Information3dViewport"), SNAME("EditorStyles")));
		} break;

		case NOTIFICATION_DRAG_END: {
			// Clear preview material when dropped outside applicable object.
			if (spatial_editor->get_preview_material().is_valid() && !is_drag_successful()) {
				_remove_preview_material();
			}
		} break;
	}
}

static void draw_indicator_bar(Control &p_surface, real_t p_fill, const Ref<Texture2D> p_icon, const Ref<Font> p_font, int p_font_size, const String &p_text, const Color &p_color) {
	// Adjust bar size from control height
	const Vector2 surface_size = p_surface.get_size();
	const real_t h = surface_size.y / 2.0;
	const real_t y = (surface_size.y - h) / 2.0;

	const Rect2 r(10 * EDSCALE, y, 6 * EDSCALE, h);
	const real_t sy = r.size.y * p_fill;

	// Note: because this bar appears over the viewport, it has to stay readable for any background color
	// Draw both neutral dark and bright colors to account this
	p_surface.draw_rect(r, p_color * Color(1, 1, 1, 0.2));
	p_surface.draw_rect(Rect2(r.position.x, r.position.y + r.size.y - sy, r.size.x, sy), p_color * Color(1, 1, 1, 0.6));
	p_surface.draw_rect(r.grow(1), Color(0, 0, 0, 0.7), false, Math::round(EDSCALE));

	const Vector2 icon_size = p_icon->get_size();
	const Vector2 icon_pos = Vector2(r.position.x - (icon_size.x - r.size.x) / 2, r.position.y + r.size.y + 2 * EDSCALE);
	p_surface.draw_texture(p_icon, icon_pos, p_color);

	// Draw text below the bar (for speed/zoom information).
	p_surface.draw_string_outline(p_font, Vector2(icon_pos.x, icon_pos.y + icon_size.y + 16 * EDSCALE), p_text, HORIZONTAL_ALIGNMENT_LEFT, -1.f, p_font_size, Math::round(2 * EDSCALE), Color(0, 0, 0));
	p_surface.draw_string(p_font, Vector2(icon_pos.x, icon_pos.y + icon_size.y + 16 * EDSCALE), p_text, HORIZONTAL_ALIGNMENT_LEFT, -1.f, p_font_size, p_color);
}

void Node3DEditorViewport::_draw() {
	EditorPluginList *over_plugin_list = EditorNode::get_singleton()->get_editor_plugins_over();
	if (!over_plugin_list->is_empty()) {
		over_plugin_list->forward_3d_draw_over_viewport(surface);
	}

	EditorPluginList *force_over_plugin_list = EditorNode::get_singleton()->get_editor_plugins_force_over();
	if (!force_over_plugin_list->is_empty()) {
		force_over_plugin_list->forward_3d_force_draw_over_viewport(surface);
	}

	if (surface->has_focus()) {
		Size2 size = surface->get_size();
		Rect2 r = Rect2(Point2(), size);
		get_theme_stylebox(SNAME("FocusViewport"), SNAME("EditorStyles"))->draw(surface->get_canvas_item(), r);
	}

	if (cursor.region_select) {
		const Rect2 selection_rect = Rect2(cursor.region_begin, cursor.region_end - cursor.region_begin);

		surface->draw_rect(
				selection_rect,
				get_theme_color(SNAME("box_selection_fill_color"), SNAME("Editor")));

		surface->draw_rect(
				selection_rect,
				get_theme_color(SNAME("box_selection_stroke_color"), SNAME("Editor")),
				false,
				Math::round(EDSCALE));
	}

	RID ci = surface->get_canvas_item();

	if (message_time > 0) {
		Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
		int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
		Point2 msgpos = Point2(5, get_size().y - 20);
		font->draw_string(ci, msgpos + Point2(1, 1), message, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0, 0, 0, 0.8));
		font->draw_string(ci, msgpos + Point2(-1, -1), message, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0, 0, 0, 0.8));
		font->draw_string(ci, msgpos, message, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(1, 1, 1, 1));
	}

	if (_edit.mode == TRANSFORM_ROTATE && _edit.show_rotation_line) {
		Point2 center = _point_to_screen(_edit.center);

		Color handle_color;
		switch (_edit.plane) {
			case TRANSFORM_X_AXIS:
				handle_color = get_theme_color(SNAME("axis_x_color"), SNAME("Editor"));
				break;
			case TRANSFORM_Y_AXIS:
				handle_color = get_theme_color(SNAME("axis_y_color"), SNAME("Editor"));
				break;
			case TRANSFORM_Z_AXIS:
				handle_color = get_theme_color(SNAME("axis_z_color"), SNAME("Editor"));
				break;
			default:
				handle_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
				break;
		}
		handle_color = handle_color.from_hsv(handle_color.get_h(), 0.25, 1.0, 1);

		RenderingServer::get_singleton()->canvas_item_add_line(
				ci,
				_edit.mouse_pos,
				center,
				handle_color,
				Math::round(2 * EDSCALE));
	}
	if (previewing) {
		Size2 ss = Size2(GLOBAL_GET("display/window/size/viewport_width"), GLOBAL_GET("display/window/size/viewport_height"));
		float aspect = ss.aspect();
		Size2 s = get_size();

		Rect2 draw_rect;

		switch (previewing->get_keep_aspect_mode()) {
			case Camera3D::KEEP_WIDTH: {
				draw_rect.size = Size2(s.width, s.width / aspect);
				draw_rect.position.x = 0;
				draw_rect.position.y = (s.height - draw_rect.size.y) * 0.5;

			} break;
			case Camera3D::KEEP_HEIGHT: {
				draw_rect.size = Size2(s.height * aspect, s.height);
				draw_rect.position.y = 0;
				draw_rect.position.x = (s.width - draw_rect.size.x) * 0.5;

			} break;
		}

		draw_rect = Rect2(Vector2(), s).intersection(draw_rect);

		surface->draw_rect(draw_rect, Color(0.6, 0.6, 0.1, 0.5), false, Math::round(2 * EDSCALE));

	} else {
		if (zoom_indicator_delay > 0.0) {
			if (is_freelook_active()) {
				// Show speed

				real_t min_speed = MAX(camera->get_near() * 4, Node3DEditor::ZOOM_FREELOOK_MIN);
				real_t max_speed = MIN(camera->get_far() / 4, Node3DEditor::ZOOM_FREELOOK_MAX);
				real_t scale_length = (max_speed - min_speed);

				if (!Math::is_zero_approx(scale_length)) {
					real_t logscale_t = 1.0 - Math::log1p(freelook_speed - min_speed) / Math::log1p(scale_length);

					// Display the freelook speed to help the user get a better sense of scale.
					const int precision = freelook_speed < 1.0 ? 2 : 1;
					draw_indicator_bar(
							*surface,
							1.0 - logscale_t,
							get_theme_icon(SNAME("ViewportSpeed"), SNAME("EditorIcons")),
							get_theme_font(SNAME("font"), SNAME("Label")),
							get_theme_font_size(SNAME("font_size"), SNAME("Label")),
							vformat("%s u/s", String::num(freelook_speed).pad_decimals(precision)),
							Color(1.0, 0.95, 0.7));
				}

			} else {
				// Show zoom
				zoom_limit_label->set_visible(zoom_failed_attempts_count > 15);

				real_t min_distance = MAX(camera->get_near() * 4, Node3DEditor::ZOOM_FREELOOK_MIN);
				real_t max_distance = MIN(camera->get_far() / 4, Node3DEditor::ZOOM_FREELOOK_MAX);
				real_t scale_length = (max_distance - min_distance);

				if (!Math::is_zero_approx(scale_length)) {
					real_t logscale_t = 1.0 - Math::log1p(cursor.distance - min_distance) / Math::log1p(scale_length);

					// Display the zoom center distance to help the user get a better sense of scale.
					const int precision = cursor.distance < 1.0 ? 2 : 1;
					draw_indicator_bar(
							*surface,
							logscale_t,
							get_theme_icon(SNAME("ViewportZoom"), SNAME("EditorIcons")),
							get_theme_font(SNAME("font"), SNAME("Label")),
							get_theme_font_size(SNAME("font_size"), SNAME("Label")),
							vformat("%s u", String::num(cursor.distance).pad_decimals(precision)),
							Color(0.7, 0.95, 1.0));
				}
			}
		}
	}
}

void Node3DEditorViewport::_menu_option(int p_option) {
	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	switch (p_option) {
		case VIEW_TOP: {
			cursor.y_rot = 0;
			cursor.x_rot = Math_PI / 2.0;
			set_message(TTR("Top View."), 2);
			view_type = VIEW_TYPE_TOP;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_BOTTOM: {
			cursor.y_rot = 0;
			cursor.x_rot = -Math_PI / 2.0;
			set_message(TTR("Bottom View."), 2);
			view_type = VIEW_TYPE_BOTTOM;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_LEFT: {
			cursor.x_rot = 0;
			cursor.y_rot = Math_PI / 2.0;
			set_message(TTR("Left View."), 2);
			view_type = VIEW_TYPE_LEFT;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_RIGHT: {
			cursor.x_rot = 0;
			cursor.y_rot = -Math_PI / 2.0;
			set_message(TTR("Right View."), 2);
			view_type = VIEW_TYPE_RIGHT;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_FRONT: {
			cursor.x_rot = 0;
			cursor.y_rot = Math_PI;
			set_message(TTR("Front View."), 2);
			view_type = VIEW_TYPE_FRONT;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_REAR: {
			cursor.x_rot = 0;
			cursor.y_rot = 0;
			set_message(TTR("Rear View."), 2);
			view_type = VIEW_TYPE_REAR;
			_set_auto_orthogonal();
			_update_name();

		} break;
		case VIEW_CENTER_TO_ORIGIN: {
			cursor.pos = Vector3(0, 0, 0);

		} break;
		case VIEW_CENTER_TO_SELECTION: {
			focus_selection();

		} break;
		case VIEW_ALIGN_TRANSFORM_WITH_VIEW: {
			if (!get_selected_count()) {
				break;
			}

			Transform3D camera_transform = camera->get_global_transform();

			List<Node *> &selection = editor_selection->get_selected_node_list();

			undo_redo->create_action(TTR("Align Transform with View"));

			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				Transform3D xform;
				if (orthogonal) {
					xform = sp->get_global_transform();
					xform.basis = Basis::from_euler(camera_transform.basis.get_euler());
				} else {
					xform = camera_transform;
					xform.scale_basis(sp->get_scale());
				}

				if (Object::cast_to<Decal>(E)) {
					// Adjust rotation to match Decal's default orientation.
					// This makes the decal "look" in the same direction as the camera,
					// rather than pointing down relative to the camera orientation.
					xform.basis.rotate_local(Vector3(1, 0, 0), Math_TAU * 0.25);
				}

				undo_redo->add_do_method(sp, "set_global_transform", xform);
				undo_redo->add_undo_method(sp, "set_global_transform", sp->get_global_gizmo_transform());
			}
			undo_redo->commit_action();

		} break;
		case VIEW_ALIGN_ROTATION_WITH_VIEW: {
			if (!get_selected_count()) {
				break;
			}

			Transform3D camera_transform = camera->get_global_transform();

			List<Node *> &selection = editor_selection->get_selected_node_list();

			undo_redo->create_action(TTR("Align Rotation with View"));
			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				Basis basis = camera_transform.basis;

				if (Object::cast_to<Decal>(E)) {
					// Adjust rotation to match Decal's default orientation.
					// This makes the decal "look" in the same direction as the camera,
					// rather than pointing down relative to the camera orientation.
					basis.rotate_local(Vector3(1, 0, 0), Math_TAU * 0.25);
				}

				undo_redo->add_do_method(sp, "set_rotation", basis.get_euler_normalized());
				undo_redo->add_undo_method(sp, "set_rotation", sp->get_rotation());
			}
			undo_redo->commit_action();

		} break;
		case VIEW_ENVIRONMENT: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_ENVIRONMENT);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			if (current) {
				camera->set_environment(Ref<Resource>());
			} else {
				camera->set_environment(Node3DEditor::get_singleton()->get_viewport_environment());
			}

			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_PERSPECTIVE: {
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ORTHOGONAL), false);
			orthogonal = false;
			auto_orthogonal = false;
			call_deferred(SNAME("update_transform_gizmo_view"));
			_update_name();

		} break;
		case VIEW_ORTHOGONAL: {
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ORTHOGONAL), true);
			orthogonal = true;
			auto_orthogonal = false;
			call_deferred(SNAME("update_transform_gizmo_view"));
			_update_name();

		} break;
		case VIEW_AUTO_ORTHOGONAL: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_AUTO_ORTHOGONAL);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			view_menu->get_popup()->set_item_checked(idx, current);
			if (auto_orthogonal) {
				auto_orthogonal = false;
				_update_name();
			}
		} break;
		case VIEW_LOCK_ROTATION: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_LOCK_ROTATION);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			lock_rotation = !current;
			view_menu->get_popup()->set_item_checked(idx, !current);
			if (lock_rotation) {
				locked_label->show();
			} else {
				locked_label->hide();
			}

		} break;
		case VIEW_AUDIO_LISTENER: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			viewport->set_as_audio_listener_3d(current);
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_AUDIO_DOPPLER: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_DOPPLER);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			camera->set_doppler_tracking(current ? Camera3D::DOPPLER_TRACKING_IDLE_STEP : Camera3D::DOPPLER_TRACKING_DISABLED);
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_CINEMATIC_PREVIEW: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_CINEMATIC_PREVIEW);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			view_menu->get_popup()->set_item_checked(idx, current);
			previewing_cinema = true;
			_toggle_cinema_preview(current);

			if (current) {
				preview_camera->hide();
			} else {
				if (previewing != nullptr) {
					preview_camera->show();
				}
			}
		} break;
		case VIEW_GIZMOS: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_GIZMOS);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			uint32_t layers = ((1 << 20) - 1) | (1 << (GIZMO_BASE_LAYER + index)) | (1 << GIZMO_GRID_LAYER) | (1 << MISC_TOOL_LAYER);
			if (current) {
				layers |= (1 << GIZMO_EDIT_LAYER);
			}
			camera->set_cull_mask(layers);
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_HALF_RESOLUTION: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_HALF_RESOLUTION);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			view_menu->get_popup()->set_item_checked(idx, !current);
			_update_shrink();
		} break;
		case VIEW_INFORMATION: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_INFORMATION);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			view_menu->get_popup()->set_item_checked(idx, !current);

		} break;
		case VIEW_FRAME_TIME: {
			int idx = view_menu->get_popup()->get_item_index(VIEW_FRAME_TIME);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			view_menu->get_popup()->set_item_checked(idx, !current);

		} break;
		case VIEW_DISPLAY_NORMAL:
		case VIEW_DISPLAY_WIREFRAME:
		case VIEW_DISPLAY_OVERDRAW:
		case VIEW_DISPLAY_SHADELESS:
		case VIEW_DISPLAY_LIGHTING:
		case VIEW_DISPLAY_NORMAL_BUFFER:
		case VIEW_DISPLAY_DEBUG_SHADOW_ATLAS:
		case VIEW_DISPLAY_DEBUG_DIRECTIONAL_SHADOW_ATLAS:
		case VIEW_DISPLAY_DEBUG_VOXEL_GI_ALBEDO:
		case VIEW_DISPLAY_DEBUG_VOXEL_GI_LIGHTING:
		case VIEW_DISPLAY_DEBUG_VOXEL_GI_EMISSION:
		case VIEW_DISPLAY_DEBUG_SCENE_LUMINANCE:
		case VIEW_DISPLAY_DEBUG_SSAO:
		case VIEW_DISPLAY_DEBUG_SSIL:
		case VIEW_DISPLAY_DEBUG_PSSM_SPLITS:
		case VIEW_DISPLAY_DEBUG_DECAL_ATLAS:
		case VIEW_DISPLAY_DEBUG_SDFGI:
		case VIEW_DISPLAY_DEBUG_SDFGI_PROBES:
		case VIEW_DISPLAY_DEBUG_GI_BUFFER:
		case VIEW_DISPLAY_DEBUG_DISABLE_LOD:
		case VIEW_DISPLAY_DEBUG_CLUSTER_OMNI_LIGHTS:
		case VIEW_DISPLAY_DEBUG_CLUSTER_SPOT_LIGHTS:
		case VIEW_DISPLAY_DEBUG_CLUSTER_DECALS:
		case VIEW_DISPLAY_DEBUG_CLUSTER_REFLECTION_PROBES:
		case VIEW_DISPLAY_DEBUG_OCCLUDERS:
		case VIEW_DISPLAY_MOTION_VECTORS: {
			static const int display_options[] = {
				VIEW_DISPLAY_NORMAL,
				VIEW_DISPLAY_WIREFRAME,
				VIEW_DISPLAY_OVERDRAW,
				VIEW_DISPLAY_SHADELESS,
				VIEW_DISPLAY_LIGHTING,
				VIEW_DISPLAY_NORMAL_BUFFER,
				VIEW_DISPLAY_WIREFRAME,
				VIEW_DISPLAY_DEBUG_SHADOW_ATLAS,
				VIEW_DISPLAY_DEBUG_DIRECTIONAL_SHADOW_ATLAS,
				VIEW_DISPLAY_DEBUG_VOXEL_GI_ALBEDO,
				VIEW_DISPLAY_DEBUG_VOXEL_GI_LIGHTING,
				VIEW_DISPLAY_DEBUG_VOXEL_GI_EMISSION,
				VIEW_DISPLAY_DEBUG_SCENE_LUMINANCE,
				VIEW_DISPLAY_DEBUG_SSAO,
				VIEW_DISPLAY_DEBUG_SSIL,
				VIEW_DISPLAY_DEBUG_GI_BUFFER,
				VIEW_DISPLAY_DEBUG_DISABLE_LOD,
				VIEW_DISPLAY_DEBUG_PSSM_SPLITS,
				VIEW_DISPLAY_DEBUG_DECAL_ATLAS,
				VIEW_DISPLAY_DEBUG_SDFGI,
				VIEW_DISPLAY_DEBUG_SDFGI_PROBES,
				VIEW_DISPLAY_DEBUG_CLUSTER_OMNI_LIGHTS,
				VIEW_DISPLAY_DEBUG_CLUSTER_SPOT_LIGHTS,
				VIEW_DISPLAY_DEBUG_CLUSTER_DECALS,
				VIEW_DISPLAY_DEBUG_CLUSTER_REFLECTION_PROBES,
				VIEW_DISPLAY_DEBUG_OCCLUDERS,
				VIEW_DISPLAY_MOTION_VECTORS,
				VIEW_MAX
			};
			static const Viewport::DebugDraw debug_draw_modes[] = {
				Viewport::DEBUG_DRAW_DISABLED,
				Viewport::DEBUG_DRAW_WIREFRAME,
				Viewport::DEBUG_DRAW_OVERDRAW,
				Viewport::DEBUG_DRAW_UNSHADED,
				Viewport::DEBUG_DRAW_LIGHTING,
				Viewport::DEBUG_DRAW_NORMAL_BUFFER,
				Viewport::DEBUG_DRAW_WIREFRAME,
				Viewport::DEBUG_DRAW_SHADOW_ATLAS,
				Viewport::DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS,
				Viewport::DEBUG_DRAW_VOXEL_GI_ALBEDO,
				Viewport::DEBUG_DRAW_VOXEL_GI_LIGHTING,
				Viewport::DEBUG_DRAW_VOXEL_GI_EMISSION,
				Viewport::DEBUG_DRAW_SCENE_LUMINANCE,
				Viewport::DEBUG_DRAW_SSAO,
				Viewport::DEBUG_DRAW_SSIL,
				Viewport::DEBUG_DRAW_GI_BUFFER,
				Viewport::DEBUG_DRAW_DISABLE_LOD,
				Viewport::DEBUG_DRAW_PSSM_SPLITS,
				Viewport::DEBUG_DRAW_DECAL_ATLAS,
				Viewport::DEBUG_DRAW_SDFGI,
				Viewport::DEBUG_DRAW_SDFGI_PROBES,
				Viewport::DEBUG_DRAW_CLUSTER_OMNI_LIGHTS,
				Viewport::DEBUG_DRAW_CLUSTER_SPOT_LIGHTS,
				Viewport::DEBUG_DRAW_CLUSTER_DECALS,
				Viewport::DEBUG_DRAW_CLUSTER_REFLECTION_PROBES,
				Viewport::DEBUG_DRAW_OCCLUDERS,
				Viewport::DEBUG_DRAW_MOTION_VECTORS,
			};

			int idx = 0;

			while (display_options[idx] != VIEW_MAX) {
				int id = display_options[idx];
				int item_idx = view_menu->get_popup()->get_item_index(id);
				if (item_idx != -1) {
					view_menu->get_popup()->set_item_checked(item_idx, id == p_option);
				}
				item_idx = display_submenu->get_item_index(id);
				if (item_idx != -1) {
					display_submenu->set_item_checked(item_idx, id == p_option);
				}

				if (id == p_option) {
					viewport->set_debug_draw(debug_draw_modes[idx]);
				}
				idx++;
			}
		} break;
	}
}

void Node3DEditorViewport::_set_auto_orthogonal() {
	if (!orthogonal && view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUTO_ORTHOGONAL))) {
		_menu_option(VIEW_ORTHOGONAL);
		auto_orthogonal = true;
	}
}

void Node3DEditorViewport::_preview_exited_scene() {
	preview_camera->disconnect("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview));
	preview_camera->set_pressed(false);
	_toggle_camera_preview(false);
	preview_camera->connect("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview));
	view_menu->show();
}

void Node3DEditorViewport::_init_gizmo_instance(int p_idx) {
	uint32_t layer = 1 << (GIZMO_BASE_LAYER + p_idx);

	for (int i = 0; i < 3; i++) {
		move_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(move_gizmo_instance[i], spatial_editor->get_move_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(move_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(move_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(move_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(move_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		move_plane_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(move_plane_gizmo_instance[i], spatial_editor->get_move_plane_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(move_plane_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_plane_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(move_plane_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(move_plane_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(move_plane_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		rotate_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(rotate_gizmo_instance[i], spatial_editor->get_rotate_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(rotate_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(rotate_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(rotate_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		scale_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(scale_gizmo_instance[i], spatial_editor->get_scale_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(scale_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(scale_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(scale_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(scale_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(scale_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		scale_plane_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(scale_plane_gizmo_instance[i], spatial_editor->get_scale_plane_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(scale_plane_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(scale_plane_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(scale_plane_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(scale_plane_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(scale_plane_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);

		axis_gizmo_instance[i] = RS::get_singleton()->instance_create();
		RS::get_singleton()->instance_set_base(axis_gizmo_instance[i], spatial_editor->get_axis_gizmo(i)->get_rid());
		RS::get_singleton()->instance_set_scenario(axis_gizmo_instance[i], get_tree()->get_root()->get_world_3d()->get_scenario());
		RS::get_singleton()->instance_set_visible(axis_gizmo_instance[i], true);
		RS::get_singleton()->instance_geometry_set_cast_shadows_setting(axis_gizmo_instance[i], RS::SHADOW_CASTING_SETTING_OFF);
		RS::get_singleton()->instance_set_layer_mask(axis_gizmo_instance[i], layer);
		RS::get_singleton()->instance_geometry_set_flag(axis_gizmo_instance[i], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
		RS::get_singleton()->instance_geometry_set_flag(axis_gizmo_instance[i], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
	}

	// Rotation white outline
	rotate_gizmo_instance[3] = RS::get_singleton()->instance_create();
	RS::get_singleton()->instance_set_base(rotate_gizmo_instance[3], spatial_editor->get_rotate_gizmo(3)->get_rid());
	RS::get_singleton()->instance_set_scenario(rotate_gizmo_instance[3], get_tree()->get_root()->get_world_3d()->get_scenario());
	RS::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
	RS::get_singleton()->instance_geometry_set_cast_shadows_setting(rotate_gizmo_instance[3], RS::SHADOW_CASTING_SETTING_OFF);
	RS::get_singleton()->instance_set_layer_mask(rotate_gizmo_instance[3], layer);
	RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[3], RS::INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING, true);
	RS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[3], RS::INSTANCE_FLAG_USE_BAKED_LIGHT, false);
}

void Node3DEditorViewport::_finish_gizmo_instances() {
	for (int i = 0; i < 3; i++) {
		RS::get_singleton()->free(move_gizmo_instance[i]);
		RS::get_singleton()->free(move_plane_gizmo_instance[i]);
		RS::get_singleton()->free(rotate_gizmo_instance[i]);
		RS::get_singleton()->free(scale_gizmo_instance[i]);
		RS::get_singleton()->free(scale_plane_gizmo_instance[i]);
		RS::get_singleton()->free(axis_gizmo_instance[i]);
	}
	// Rotation white outline
	RS::get_singleton()->free(rotate_gizmo_instance[3]);
}

void Node3DEditorViewport::_toggle_camera_preview(bool p_activate) {
	ERR_FAIL_COND(p_activate && !preview);
	ERR_FAIL_COND(!p_activate && !previewing);

	previewing_camera = p_activate;
	_update_navigation_controls_visibility();

	if (!p_activate) {
		previewing->disconnect("tree_exiting", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
		previewing = nullptr;
		RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), camera->get_camera()); //restore
		if (!preview) {
			preview_camera->hide();
		}
		surface->queue_redraw();

	} else {
		previewing = preview;
		previewing->connect("tree_exiting", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
		RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), preview->get_camera()); //replace
		surface->queue_redraw();
	}
}

void Node3DEditorViewport::_toggle_cinema_preview(bool p_activate) {
	previewing_cinema = p_activate;
	_update_navigation_controls_visibility();

	if (!previewing_cinema) {
		if (previewing != nullptr) {
			previewing->disconnect("tree_exited", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
		}

		previewing = nullptr;
		RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), camera->get_camera()); //restore
		preview_camera->set_pressed(false);
		if (!preview) {
			preview_camera->hide();
		} else {
			preview_camera->show();
		}
		view_menu->show();
		surface->queue_redraw();
	}
}

void Node3DEditorViewport::_selection_result_pressed(int p_result) {
	if (selection_results.size() <= p_result) {
		return;
	}

	clicked = selection_results[p_result].item->get_instance_id();

	if (clicked.is_valid()) {
		_select_clicked(spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT);
	}
}

void Node3DEditorViewport::_selection_menu_hide() {
	selection_results.clear();
	selection_menu->clear();
	selection_menu->reset_size();
}

void Node3DEditorViewport::set_can_preview(Camera3D *p_preview) {
	preview = p_preview;

	if (!preview_camera->is_pressed() && !previewing_cinema) {
		preview_camera->set_visible(p_preview);
	}
}

void Node3DEditorViewport::update_transform_gizmo_view() {
	if (!is_visible_in_tree()) {
		return;
	}

	Transform3D xform = spatial_editor->get_gizmo_transform();

	Transform3D camera_xform = camera->get_transform();

	if (xform.origin.is_equal_approx(camera_xform.origin)) {
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(axis_gizmo_instance[i], false);
		}
		// Rotation white outline
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
		return;
	}

	const Vector3 camz = -camera_xform.get_basis().get_column(2).normalized();
	const Vector3 camy = -camera_xform.get_basis().get_column(1).normalized();
	const Plane p = Plane(camz, camera_xform.origin);
	const real_t gizmo_d = MAX(Math::abs(p.distance_to(xform.origin)), CMP_EPSILON);
	const real_t d0 = camera->unproject_position(camera_xform.origin + camz * gizmo_d).y;
	const real_t d1 = camera->unproject_position(camera_xform.origin + camz * gizmo_d + camy).y;
	const real_t dd = MAX(Math::abs(d0 - d1), CMP_EPSILON);

	const real_t gizmo_size = EDITOR_GET("editors/3d/manipulator_gizmo_size");
	// At low viewport heights, multiply the gizmo scale based on the viewport height.
	// This prevents the gizmo from growing very large and going outside the viewport.
	const int viewport_base_height = 400 * MAX(1, EDSCALE);
	gizmo_scale =
			(gizmo_size / Math::abs(dd)) * MAX(1, EDSCALE) *
			MIN(viewport_base_height, subviewport_container->get_size().height) / viewport_base_height /
			subviewport_container->get_stretch_shrink();
	Vector3 scale = Vector3(1, 1, 1) * gizmo_scale;

	// if the determinant is zero, we should disable the gizmo from being rendered
	// this prevents supplying bad values to the renderer and then having to filter it out again
	if (xform.basis.determinant() == 0) {
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], false);
			RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], false);
		}
		// Rotation white outline
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], false);
		return;
	}

	for (int i = 0; i < 3; i++) {
		Transform3D axis_angle;
		if (xform.basis.get_column(i).normalized().dot(xform.basis.get_column((i + 1) % 3).normalized()) < 1.0) {
			axis_angle = axis_angle.looking_at(xform.basis.get_column(i).normalized(), xform.basis.get_column((i + 1) % 3).normalized());
		}
		axis_angle.basis.scale(scale);
		axis_angle.origin = xform.origin;
		RenderingServer::get_singleton()->instance_set_transform(move_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE));
		RenderingServer::get_singleton()->instance_set_transform(move_plane_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_MOVE));
		RenderingServer::get_singleton()->instance_set_transform(rotate_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE));
		RenderingServer::get_singleton()->instance_set_transform(scale_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(scale_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE));
		RenderingServer::get_singleton()->instance_set_transform(scale_plane_gizmo_instance[i], axis_angle);
		RenderingServer::get_singleton()->instance_set_visible(scale_plane_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SCALE));
		RenderingServer::get_singleton()->instance_set_transform(axis_gizmo_instance[i], xform);
	}

	bool show_axes = spatial_editor->is_gizmo_visible() && _edit.mode != TRANSFORM_NONE;
	RenderingServer *rs = RenderingServer::get_singleton();
	rs->instance_set_visible(axis_gizmo_instance[0], show_axes && (_edit.plane == TRANSFORM_X_AXIS || _edit.plane == TRANSFORM_XY || _edit.plane == TRANSFORM_XZ));
	rs->instance_set_visible(axis_gizmo_instance[1], show_axes && (_edit.plane == TRANSFORM_Y_AXIS || _edit.plane == TRANSFORM_XY || _edit.plane == TRANSFORM_YZ));
	rs->instance_set_visible(axis_gizmo_instance[2], show_axes && (_edit.plane == TRANSFORM_Z_AXIS || _edit.plane == TRANSFORM_XZ || _edit.plane == TRANSFORM_YZ));

	// Rotation white outline
	xform.orthonormalize();
	xform.basis.scale(scale);
	RenderingServer::get_singleton()->instance_set_transform(rotate_gizmo_instance[3], xform);
	RenderingServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[3], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == Node3DEditor::TOOL_MODE_ROTATE));
}

void Node3DEditorViewport::set_state(const Dictionary &p_state) {
	if (p_state.has("position")) {
		cursor.pos = p_state["position"];
	}
	if (p_state.has("x_rotation")) {
		cursor.x_rot = p_state["x_rotation"];
	}
	if (p_state.has("y_rotation")) {
		cursor.y_rot = p_state["y_rotation"];
	}
	if (p_state.has("distance")) {
		cursor.distance = p_state["distance"];
	}

	if (p_state.has("use_orthogonal")) {
		bool orth = p_state["use_orthogonal"];

		if (orth) {
			_menu_option(VIEW_ORTHOGONAL);
		} else {
			_menu_option(VIEW_PERSPECTIVE);
		}
	}
	if (p_state.has("view_type")) {
		view_type = ViewType(p_state["view_type"].operator int());
		_update_name();
	}
	if (p_state.has("auto_orthogonal")) {
		auto_orthogonal = p_state["auto_orthogonal"];
		_update_name();
	}
	if (p_state.has("auto_orthogonal_enabled")) {
		bool enabled = p_state["auto_orthogonal_enabled"];
		view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUTO_ORTHOGONAL), enabled);
	}
	if (p_state.has("display_mode")) {
		int display = p_state["display_mode"];

		int idx = view_menu->get_popup()->get_item_index(display);
		if (!view_menu->get_popup()->is_item_checked(idx)) {
			_menu_option(display);
		}
	}
	if (p_state.has("lock_rotation")) {
		lock_rotation = p_state["lock_rotation"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_LOCK_ROTATION);
		view_menu->get_popup()->set_item_checked(idx, lock_rotation);
	}
	if (p_state.has("use_environment")) {
		bool env = p_state["use_environment"];

		if (env != camera->get_environment().is_valid()) {
			_menu_option(VIEW_ENVIRONMENT);
		}
	}
	if (p_state.has("listener")) {
		bool listener = p_state["listener"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER);
		viewport->set_as_audio_listener_3d(listener);
		view_menu->get_popup()->set_item_checked(idx, listener);
	}
	if (p_state.has("doppler")) {
		bool doppler = p_state["doppler"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_DOPPLER);
		camera->set_doppler_tracking(doppler ? Camera3D::DOPPLER_TRACKING_IDLE_STEP : Camera3D::DOPPLER_TRACKING_DISABLED);
		view_menu->get_popup()->set_item_checked(idx, doppler);
	}
	if (p_state.has("gizmos")) {
		bool gizmos = p_state["gizmos"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_GIZMOS);
		if (view_menu->get_popup()->is_item_checked(idx) != gizmos) {
			_menu_option(VIEW_GIZMOS);
		}
	}
	if (p_state.has("information")) {
		bool information = p_state["information"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_INFORMATION);
		if (view_menu->get_popup()->is_item_checked(idx) != information) {
			_menu_option(VIEW_INFORMATION);
		}
	}
	if (p_state.has("frame_time")) {
		bool fps = p_state["frame_time"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_FRAME_TIME);
		if (view_menu->get_popup()->is_item_checked(idx) != fps) {
			_menu_option(VIEW_FRAME_TIME);
		}
	}
	if (p_state.has("half_res")) {
		bool half_res = p_state["half_res"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_HALF_RESOLUTION);
		view_menu->get_popup()->set_item_checked(idx, half_res);
	}
	if (p_state.has("cinematic_preview")) {
		previewing_cinema = p_state["cinematic_preview"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_CINEMATIC_PREVIEW);
		view_menu->get_popup()->set_item_checked(idx, previewing_cinema);
	}

	if (preview_camera->is_connected("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview))) {
		preview_camera->disconnect("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview));
	}
	if (p_state.has("previewing")) {
		Node *pv = EditorNode::get_singleton()->get_edited_scene()->get_node(p_state["previewing"]);
		if (Object::cast_to<Camera3D>(pv)) {
			previewing = Object::cast_to<Camera3D>(pv);
			previewing->connect("tree_exiting", callable_mp(this, &Node3DEditorViewport::_preview_exited_scene));
			RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), previewing->get_camera()); //replace
			surface->queue_redraw();
			preview_camera->set_pressed(true);
			preview_camera->show();
		}
	}
	preview_camera->connect("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview));
}

Dictionary Node3DEditorViewport::get_state() const {
	Dictionary d;
	d["position"] = cursor.pos;
	d["x_rotation"] = cursor.x_rot;
	d["y_rotation"] = cursor.y_rot;
	d["distance"] = cursor.distance;
	d["use_environment"] = camera->get_environment().is_valid();
	d["use_orthogonal"] = camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL;
	d["view_type"] = view_type;
	d["auto_orthogonal"] = auto_orthogonal;
	d["auto_orthogonal_enabled"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUTO_ORTHOGONAL));
	if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL))) {
		d["display_mode"] = VIEW_DISPLAY_NORMAL;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME))) {
		d["display_mode"] = VIEW_DISPLAY_WIREFRAME;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW))) {
		d["display_mode"] = VIEW_DISPLAY_OVERDRAW;
	} else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS))) {
		d["display_mode"] = VIEW_DISPLAY_SHADELESS;
	}
	d["listener"] = viewport->is_audio_listener_3d();
	d["doppler"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUDIO_DOPPLER));
	d["gizmos"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_GIZMOS));
	d["information"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_INFORMATION));
	d["frame_time"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_FRAME_TIME));
	d["half_res"] = subviewport_container->get_stretch_shrink() > 1;
	d["cinematic_preview"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_CINEMATIC_PREVIEW));
	if (previewing) {
		d["previewing"] = EditorNode::get_singleton()->get_edited_scene()->get_path_to(previewing);
	}
	if (lock_rotation) {
		d["lock_rotation"] = lock_rotation;
	}

	return d;
}

void Node3DEditorViewport::_bind_methods() {
	ClassDB::bind_method(D_METHOD("update_transform_gizmo_view"), &Node3DEditorViewport::update_transform_gizmo_view); // Used by call_deferred.
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw"), &Node3DEditorViewport::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw"), &Node3DEditorViewport::drop_data_fw);

	ADD_SIGNAL(MethodInfo("toggle_maximize_view", PropertyInfo(Variant::OBJECT, "viewport")));
	ADD_SIGNAL(MethodInfo("clicked", PropertyInfo(Variant::OBJECT, "viewport")));
}

void Node3DEditorViewport::reset() {
	orthogonal = false;
	auto_orthogonal = false;
	lock_rotation = false;
	message_time = 0;
	message = "";
	last_message = "";
	view_type = VIEW_TYPE_USER;

	cursor = Cursor();
	_update_name();
}

void Node3DEditorViewport::focus_selection() {
	Vector3 center;
	int count = 0;

	const List<Node *> &selection = editor_selection->get_selected_node_list();

	for (Node *E : selection) {
		Node3D *sp = Object::cast_to<Node3D>(E);
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		if (se->gizmo.is_valid()) {
			for (const KeyValue<int, Transform3D> &GE : se->subgizmos) {
				center += se->gizmo->get_subgizmo_transform(GE.key).origin;
				count++;
			}
		}

		center += sp->get_global_gizmo_transform().origin;
		count++;
	}

	if (count != 0) {
		center /= count;
	}

	cursor.pos = center;
}

void Node3DEditorViewport::assign_pending_data_pointers(Node3D *p_preview_node, AABB *p_preview_bounds, AcceptDialog *p_accept) {
	preview_node = p_preview_node;
	preview_bounds = p_preview_bounds;
	accept = p_accept;
}

Vector3 Node3DEditorViewport::_get_instance_position(const Point2 &p_pos) const {
	const float MAX_DISTANCE = 50.0;
	const float FALLBACK_DISTANCE = 5.0;

	Vector3 world_ray = _get_ray(p_pos);
	Vector3 world_pos = _get_ray_pos(p_pos);

	PhysicsDirectSpaceState3D *ss = get_tree()->get_root()->get_world_3d()->get_direct_space_state();

	PhysicsDirectSpaceState3D::RayParameters ray_params;
	ray_params.from = world_pos;
	ray_params.to = world_pos + world_ray * camera->get_far();

	PhysicsDirectSpaceState3D::RayResult result;
	if (ss->intersect_ray(ray_params, result)) {
		return result.position;
	}

	const bool is_orthogonal = camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL;

	// The XZ plane.
	Vector3 intersection;
	Plane plane(Vector3(0, 1, 0));
	if (plane.intersects_ray(world_pos, world_ray, &intersection)) {
		if (is_orthogonal || world_pos.distance_to(intersection) <= MAX_DISTANCE) {
			return intersection;
		}
	}

	// Plane facing the camera using fallback distance.
	if (is_orthogonal) {
		plane = Plane(world_ray, cursor.pos - world_ray * (cursor.distance - FALLBACK_DISTANCE));
	} else {
		plane = Plane(world_ray, world_pos + world_ray * FALLBACK_DISTANCE);
	}
	if (plane.intersects_ray(world_pos, world_ray, &intersection)) {
		return intersection;
	}

	// Not likely, but just in case...
	return world_pos + world_ray * FALLBACK_DISTANCE;
}

AABB Node3DEditorViewport::_calculate_spatial_bounds(const Node3D *p_parent, bool p_exclude_top_level_transform) {
	AABB bounds;

	const VisualInstance3D *visual_instance = Object::cast_to<VisualInstance3D>(p_parent);
	if (visual_instance) {
		bounds = visual_instance->get_aabb();
	}

	for (int i = 0; i < p_parent->get_child_count(); i++) {
		Node3D *child = Object::cast_to<Node3D>(p_parent->get_child(i));
		if (child) {
			AABB child_bounds = _calculate_spatial_bounds(child, false);

			if (bounds.size == Vector3() && p_parent) {
				bounds = child_bounds;
			} else {
				bounds.merge_with(child_bounds);
			}
		}
	}

	if (bounds.size == Vector3() && !p_parent) {
		bounds = AABB(Vector3(-0.2, -0.2, -0.2), Vector3(0.4, 0.4, 0.4));
	}

	if (!p_exclude_top_level_transform) {
		bounds = p_parent->get_transform().xform(bounds);
	}

	return bounds;
}

Node *Node3DEditorViewport::_sanitize_preview_node(Node *p_node) const {
	Node3D *node_3d = Object::cast_to<Node3D>(p_node);
	if (node_3d == nullptr) {
		Node3D *replacement_node = memnew(Node3D);
		replacement_node->set_name(p_node->get_name());
		p_node->replace_by(replacement_node);
		memdelete(p_node);
		p_node = replacement_node;
	} else {
		VisualInstance3D *visual_instance = Object::cast_to<VisualInstance3D>(node_3d);
		if (visual_instance == nullptr) {
			Node3D *replacement_node = memnew(Node3D);
			replacement_node->set_name(node_3d->get_name());
			replacement_node->set_visible(node_3d->is_visible());
			replacement_node->set_transform(node_3d->get_transform());
			replacement_node->set_rotation_edit_mode(node_3d->get_rotation_edit_mode());
			replacement_node->set_rotation_order(node_3d->get_rotation_order());
			replacement_node->set_as_top_level(node_3d->is_set_as_top_level());
			p_node->replace_by(replacement_node);
			memdelete(p_node);
			p_node = replacement_node;
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_sanitize_preview_node(p_node->get_child(i));
	}

	return p_node;
}

void Node3DEditorViewport::_create_preview_node(const Vector<String> &files) const {
	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		Ref<Resource> res = ResourceLoader::load(path);
		ERR_CONTINUE(res.is_null());
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		Ref<Mesh> mesh = Ref<Mesh>(Object::cast_to<Mesh>(*res));
		if (mesh != nullptr || scene != nullptr) {
			if (mesh != nullptr) {
				MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
				mesh_instance->set_mesh(mesh);
				preview_node->add_child(mesh_instance);
			} else {
				if (scene.is_valid()) {
					Node *instance = scene->instantiate();
					if (instance) {
						instance = _sanitize_preview_node(instance);
						preview_node->add_child(instance);
					}
				}
			}
			EditorNode::get_singleton()->get_scene_root()->add_child(preview_node);
		}
	}
	*preview_bounds = _calculate_spatial_bounds(preview_node);
}

void Node3DEditorViewport::_remove_preview_node() {
	if (preview_node->get_parent()) {
		for (int i = preview_node->get_child_count() - 1; i >= 0; i--) {
			Node *node = preview_node->get_child(i);
			node->queue_free();
			preview_node->remove_child(node);
		}
		EditorNode::get_singleton()->get_scene_root()->remove_child(preview_node);
	}
}

bool Node3DEditorViewport::_apply_preview_material(ObjectID p_target, const Point2 &p_point) const {
	_reset_preview_material();

	if (p_target.is_null()) {
		return false;
	}

	spatial_editor->set_preview_material_target(p_target);

	Object *target_inst = ObjectDB::get_instance(p_target);

	bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CTRL);

	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(target_inst);
	if (is_ctrl && mesh_instance) {
		Ref<Mesh> mesh = mesh_instance->get_mesh();
		int surface_count = mesh->get_surface_count();

		Vector3 world_ray = _get_ray(p_point);
		Vector3 world_pos = _get_ray_pos(p_point);

		int closest_surface = -1;
		float closest_dist = 1e20;

		Transform3D gt = mesh_instance->get_global_transform();

		Transform3D ai = gt.affine_inverse();
		Vector3 xform_ray = ai.basis.xform(world_ray).normalized();
		Vector3 xform_pos = ai.xform(world_pos);

		for (int surface_idx = 0; surface_idx < surface_count; surface_idx++) {
			Ref<TriangleMesh> surface_mesh = mesh->generate_surface_triangle_mesh(surface_idx);

			Vector3 rpos, rnorm;
			if (surface_mesh->intersect_ray(xform_pos, xform_ray, rpos, rnorm)) {
				Vector3 hitpos = gt.xform(rpos);

				const real_t dist = world_pos.distance_to(hitpos);

				if (dist < 0) {
					continue;
				}

				if (dist < closest_dist) {
					closest_surface = surface_idx;
					closest_dist = dist;
				}
			}
		}

		if (closest_surface == -1) {
			return false;
		}

		if (spatial_editor->get_preview_material() != mesh_instance->get_surface_override_material(closest_surface)) {
			spatial_editor->set_preview_material_surface(closest_surface);
			spatial_editor->set_preview_reset_material(mesh_instance->get_surface_override_material(closest_surface));
			mesh_instance->set_surface_override_material(closest_surface, spatial_editor->get_preview_material());
		}

		return true;
	}

	GeometryInstance3D *geometry_instance = Object::cast_to<GeometryInstance3D>(target_inst);
	if (geometry_instance && spatial_editor->get_preview_material() != geometry_instance->get_material_override()) {
		spatial_editor->set_preview_reset_material(geometry_instance->get_material_override());
		geometry_instance->set_material_override(spatial_editor->get_preview_material());
		return true;
	}

	return false;
}

void Node3DEditorViewport::_reset_preview_material() const {
	ObjectID last_target = spatial_editor->get_preview_material_target();
	if (last_target.is_null()) {
		return;
	}
	Object *last_target_inst = ObjectDB::get_instance(last_target);

	MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(last_target_inst);
	GeometryInstance3D *geometry_instance = Object::cast_to<GeometryInstance3D>(last_target_inst);
	if (mesh_instance && spatial_editor->get_preview_material_surface() != -1) {
		mesh_instance->set_surface_override_material(spatial_editor->get_preview_material_surface(), spatial_editor->get_preview_reset_material());
		spatial_editor->set_preview_material_surface(-1);
	} else if (geometry_instance) {
		geometry_instance->set_material_override(spatial_editor->get_preview_reset_material());
	}
}

void Node3DEditorViewport::_remove_preview_material() {
	preview_material_label->hide();
	preview_material_label_desc->hide();

	spatial_editor->set_preview_material(Ref<Material>());
	spatial_editor->set_preview_reset_material(Ref<Material>());
	spatial_editor->set_preview_material_target(ObjectID());
	spatial_editor->set_preview_material_surface(-1);
}

bool Node3DEditorViewport::_cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node) {
	if (p_desired_node->get_scene_file_path() == p_target_scene_path) {
		return true;
	}

	int childCount = p_desired_node->get_child_count();
	for (int i = 0; i < childCount; i++) {
		Node *child = p_desired_node->get_child(i);
		if (_cyclical_dependency_exists(p_target_scene_path, child)) {
			return true;
		}
	}
	return false;
}

bool Node3DEditorViewport::_create_instance(Node *parent, String &path, const Point2 &p_point) {
	Ref<Resource> res = ResourceLoader::load(path);
	ERR_FAIL_COND_V(res.is_null(), false);

	Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
	Ref<Mesh> mesh = Ref<Mesh>(Object::cast_to<Mesh>(*res));

	Node *instantiated_scene = nullptr;

	if (mesh != nullptr || scene != nullptr) {
		if (mesh != nullptr) {
			MeshInstance3D *mesh_instance = memnew(MeshInstance3D);
			mesh_instance->set_mesh(mesh);

			// Adjust casing according to project setting. The file name is expected to be in snake_case, but will work for others.
			String name = path.get_file().get_basename();
			mesh_instance->set_name(Node::adjust_name_casing(name));

			instantiated_scene = mesh_instance;
		} else {
			if (!scene.is_valid()) { // invalid scene
				return false;
			} else {
				instantiated_scene = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
			}
		}
	}

	if (instantiated_scene == nullptr) {
		return false;
	}

	if (!EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path().is_empty()) { // Cyclic instantiation.
		if (_cyclical_dependency_exists(EditorNode::get_singleton()->get_edited_scene()->get_scene_file_path(), instantiated_scene)) {
			memdelete(instantiated_scene);
			return false;
		}
	}

	if (scene != nullptr) {
		instantiated_scene->set_scene_file_path(ProjectSettings::get_singleton()->localize_path(path));
	}

	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	undo_redo->add_do_method(parent, "add_child", instantiated_scene, true);
	undo_redo->add_do_method(instantiated_scene, "set_owner", EditorNode::get_singleton()->get_edited_scene());
	undo_redo->add_do_reference(instantiated_scene);
	undo_redo->add_undo_method(parent, "remove_child", instantiated_scene);

	String new_name = parent->validate_child_name(instantiated_scene);
	EditorDebuggerNode *ed = EditorDebuggerNode::get_singleton();
	undo_redo->add_do_method(ed, "live_debug_instantiate_node", EditorNode::get_singleton()->get_edited_scene()->get_path_to(parent), path, new_name);
	undo_redo->add_undo_method(ed, "live_debug_remove_node", NodePath(String(EditorNode::get_singleton()->get_edited_scene()->get_path_to(parent)) + "/" + new_name));

	Node3D *node3d = Object::cast_to<Node3D>(instantiated_scene);
	if (node3d) {
		Transform3D gl_transform;
		Node3D *parent_node3d = Object::cast_to<Node3D>(parent);
		if (parent_node3d) {
			gl_transform = parent_node3d->get_global_gizmo_transform();
		}

		gl_transform.origin = preview_node_pos;
		gl_transform.basis *= node3d->get_transform().basis;

		undo_redo->add_do_method(instantiated_scene, "set_global_transform", gl_transform);
	}

	return true;
}

void Node3DEditorViewport::_perform_drop_data() {
	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	if (spatial_editor->get_preview_material_target().is_valid()) {
		GeometryInstance3D *geometry_instance = Object::cast_to<GeometryInstance3D>(ObjectDB::get_instance(spatial_editor->get_preview_material_target()));
		MeshInstance3D *mesh_instance = Object::cast_to<MeshInstance3D>(ObjectDB::get_instance(spatial_editor->get_preview_material_target()));
		if (mesh_instance && spatial_editor->get_preview_material_surface() != -1) {
			undo_redo->create_action(vformat(TTR("Set Surface %d Override Material"), spatial_editor->get_preview_material_surface()));
			undo_redo->add_do_method(geometry_instance, "set_surface_override_material", spatial_editor->get_preview_material_surface(), spatial_editor->get_preview_material());
			undo_redo->add_undo_method(geometry_instance, "set_surface_override_material", spatial_editor->get_preview_material_surface(), spatial_editor->get_preview_reset_material());
			undo_redo->commit_action();
		} else if (geometry_instance) {
			undo_redo->create_action(TTR("Set Material Override"));
			undo_redo->add_do_method(geometry_instance, "set_material_override", spatial_editor->get_preview_material());
			undo_redo->add_undo_method(geometry_instance, "set_material_override", spatial_editor->get_preview_reset_material());
			undo_redo->commit_action();
		}

		_remove_preview_material();
		return;
	}

	_remove_preview_node();

	Vector<String> error_files;

	undo_redo->create_action(TTR("Create Node"));

	for (int i = 0; i < selected_files.size(); i++) {
		String path = selected_files[i];
		Ref<Resource> res = ResourceLoader::load(path);
		if (res.is_null()) {
			continue;
		}
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		Ref<Mesh> mesh = Ref<Mesh>(Object::cast_to<Mesh>(*res));
		if (mesh != nullptr || scene != nullptr) {
			bool success = _create_instance(target_node, path, drop_pos);
			if (!success) {
				error_files.push_back(path);
			}
		}
	}

	undo_redo->commit_action();

	if (error_files.size() > 0) {
		String files_str;
		for (int i = 0; i < error_files.size(); i++) {
			files_str += error_files[i].get_file().get_basename() + ",";
		}
		files_str = files_str.substr(0, files_str.length() - 1);
		accept->set_text(vformat(TTR("Error instantiating scene from %s"), files_str.get_data()));
		accept->popup_centered();
	}
}

bool Node3DEditorViewport::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	preview_node_viewport_pos = p_point;

	bool can_instantiate = false;

	if (!preview_node->is_inside_tree() && spatial_editor->get_preview_material().is_null()) {
		Dictionary d = p_data;
		if (d.has("type") && (String(d["type"]) == "files")) {
			Vector<String> files = d["files"];

			List<String> scene_extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &scene_extensions);
			List<String> mesh_extensions;
			ResourceLoader::get_recognized_extensions_for_type("Mesh", &mesh_extensions);
			List<String> material_extensions;
			ResourceLoader::get_recognized_extensions_for_type("Material", &material_extensions);
			List<String> texture_extensions;
			ResourceLoader::get_recognized_extensions_for_type("Texture", &texture_extensions);

			for (int i = 0; i < files.size(); i++) {
				String extension = files[i].get_extension().to_lower();

				// Check if dragged files with mesh or scene extension can be created at least once.
				if (mesh_extensions.find(extension) ||
						scene_extensions.find(extension) ||
						material_extensions.find(extension) ||
						texture_extensions.find(extension)) {
					Ref<Resource> res = ResourceLoader::load(files[i]);
					if (res.is_null()) {
						continue;
					}
					Ref<PackedScene> scn = res;
					Ref<Mesh> mesh = res;
					Ref<Material> mat = res;
					Ref<Texture2D> tex = res;
					if (scn.is_valid()) {
						Node *instantiated_scene = scn->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
						if (!instantiated_scene) {
							continue;
						}
						memdelete(instantiated_scene);
					} else if (mat.is_valid()) {
						Ref<BaseMaterial3D> base_mat = res;
						Ref<ShaderMaterial> shader_mat = res;

						if (base_mat.is_null() && !shader_mat.is_null()) {
							break;
						}

						spatial_editor->set_preview_material(mat);
						break;
					} else if (mesh.is_valid()) {
						// Let the mesh pass.
					} else if (tex.is_valid()) {
						Ref<StandardMaterial3D> new_mat = memnew(StandardMaterial3D);
						new_mat->set_texture(BaseMaterial3D::TEXTURE_ALBEDO, tex);

						spatial_editor->set_preview_material(new_mat);
						break;
					} else {
						continue;
					}
					can_instantiate = true;
					break;
				}
			}
			if (can_instantiate) {
				_create_preview_node(files);
				preview_node->hide();
			}
		}
	} else {
		if (preview_node->is_inside_tree()) {
			can_instantiate = true;
		}
	}

	if (can_instantiate) {
		update_preview_node = true;
		return true;
	}

	if (spatial_editor->get_preview_material().is_valid()) {
		preview_material_label->show();
		preview_material_label_desc->show();

		ObjectID new_preview_material_target = _select_ray(p_point);
		return _apply_preview_material(new_preview_material_target, p_point);
	}

	return false;
}

void Node3DEditorViewport::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	bool is_shift = Input::get_singleton()->is_key_pressed(Key::SHIFT);
	bool is_ctrl = Input::get_singleton()->is_key_pressed(Key::CTRL);

	selected_files.clear();
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "files") {
		selected_files = d["files"];
	}

	List<Node *> selected_nodes = EditorNode::get_singleton()->get_editor_selection()->get_selected_node_list();
	Node *root_node = EditorNode::get_singleton()->get_edited_scene();
	if (selected_nodes.size() == 1) {
		Node *selected_node = selected_nodes[0];
		target_node = root_node;
		if (is_ctrl) {
			target_node = selected_node;
		} else if (is_shift && selected_node != root_node) {
			target_node = selected_node->get_parent();
		}
	} else if (selected_nodes.size() == 0) {
		if (root_node) {
			target_node = root_node;
		} else {
			// Create a root node so we can add child nodes to it.
			SceneTreeDock::get_singleton()->add_root_node(memnew(Node3D));
			target_node = get_tree()->get_edited_scene_root();
		}
	} else {
		accept->set_text(TTR("Cannot drag and drop into multiple selected nodes."));
		accept->popup_centered();
		_remove_preview_node();
		return;
	}

	drop_pos = p_point;

	_perform_drop_data();
}

void Node3DEditorViewport::begin_transform(TransformMode p_mode, bool instant) {
	if (get_selected_count() > 0) {
		_edit.mode = p_mode;
		_compute_edit(_edit.mouse_pos);
		_edit.instant = instant;
		_edit.snap = spatial_editor->is_snap_enabled();
	}
}

void Node3DEditorViewport::commit_transform() {
	ERR_FAIL_COND(_edit.mode == TRANSFORM_NONE);
	static const char *_transform_name[4] = {
		TTRC("None"),
		TTRC("Rotate"),
		// TRANSLATORS: This refers to the movement that changes the position of an object.
		TTRC("Translate"),
		TTRC("Scale"),
	};
	Ref<EditorUndoRedoManager> &undo_redo = EditorNode::get_undo_redo();
	undo_redo->create_action(_transform_name[_edit.mode]);

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {
		Node3D *sp = Object::cast_to<Node3D>(E->get());
		if (!sp) {
			continue;
		}

		Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
		if (!se) {
			continue;
		}

		undo_redo->add_do_method(sp, "set_global_transform", sp->get_global_gizmo_transform());
		undo_redo->add_undo_method(sp, "set_global_transform", se->original);
	}
	undo_redo->commit_action();

	finish_transform();
	set_message("");
}

void Node3DEditorViewport::update_transform(Point2 p_mousepos, bool p_shift) {
	Vector3 ray_pos = _get_ray_pos(p_mousepos);
	Vector3 ray = _get_ray(p_mousepos);
	double snap = EDITOR_GET("interface/inspector/default_float_step");
	int snap_step_decimals = Math::range_step_decimals(snap);

	switch (_edit.mode) {
		case TRANSFORM_SCALE: {
			Vector3 motion_mask;
			Plane plane;
			bool plane_mv = false;

			switch (_edit.plane) {
				case TRANSFORM_VIEW:
					motion_mask = Vector3(0, 0, 0);
					plane = Plane(_get_camera_normal(), _edit.center);
					break;
				case TRANSFORM_X_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(0).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_Y_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(1).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_Z_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(2).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_YZ:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(2).normalized() + spatial_editor->get_gizmo_transform().basis.get_column(1).normalized();
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(0).normalized(), _edit.center);
					plane_mv = true;
					break;
				case TRANSFORM_XZ:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(2).normalized() + spatial_editor->get_gizmo_transform().basis.get_column(0).normalized();
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(1).normalized(), _edit.center);
					plane_mv = true;
					break;
				case TRANSFORM_XY:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(0).normalized() + spatial_editor->get_gizmo_transform().basis.get_column(1).normalized();
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(2).normalized(), _edit.center);
					plane_mv = true;
					break;
			}

			Vector3 intersection;
			if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
				break;
			}

			Vector3 click;
			if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
				break;
			}

			Vector3 motion = intersection - click;
			if (_edit.plane != TRANSFORM_VIEW) {
				if (!plane_mv) {
					motion = motion_mask.dot(motion) * motion_mask;

				} else {
					// Alternative planar scaling mode
					if (p_shift) {
						motion = motion_mask.dot(motion) * motion_mask;
					}
				}

			} else {
				const real_t center_click_dist = click.distance_to(_edit.center);
				const real_t center_inters_dist = intersection.distance_to(_edit.center);
				if (center_click_dist == 0) {
					break;
				}

				const real_t scale = center_inters_dist - center_click_dist;
				motion = Vector3(scale, scale, scale);
			}

			motion /= click.distance_to(_edit.center);

			// Disable local transformation for TRANSFORM_VIEW
			bool local_coords = (spatial_editor->are_local_coords_enabled() && _edit.plane != TRANSFORM_VIEW);

			if (_edit.snap || spatial_editor->is_snap_enabled()) {
				snap = spatial_editor->get_scale_snap() / 100;
			}
			Vector3 motion_snapped = motion;
			motion_snapped.snap(Vector3(snap, snap, snap));
			// This might not be necessary anymore after issue #288 is solved (in 4.0?).
			// TRANSLATORS: Refers to changing the scale of a node in the 3D editor.
			set_message(TTR("Scaling:") + " (" + String::num(motion_snapped.x, snap_step_decimals) + ", " +
					String::num(motion_snapped.y, snap_step_decimals) + ", " + String::num(motion_snapped.z, snap_step_decimals) + ")");
			motion = _edit.original.basis.inverse().xform(motion);

			List<Node *> &selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				if (sp->has_meta("_edit_lock_")) {
					continue;
				}

				if (se->gizmo.is_valid()) {
					for (KeyValue<int, Transform3D> &GE : se->subgizmos) {
						Transform3D xform = GE.value;
						Transform3D new_xform = _compute_transform(TRANSFORM_SCALE, se->original * xform, xform, motion, snap, local_coords, true); // Force orthogonal with subgizmo.
						if (!local_coords) {
							new_xform = se->original.affine_inverse() * new_xform;
						}
						se->gizmo->set_subgizmo_transform(GE.key, new_xform);
					}
				} else {
					Transform3D new_xform = _compute_transform(TRANSFORM_SCALE, se->original, se->original_local, motion, snap, local_coords, sp->get_rotation_edit_mode() != Node3D::ROTATION_EDIT_MODE_BASIS);
					_transform_gizmo_apply(se->sp, new_xform, local_coords);
				}
			}

			spatial_editor->update_transform_gizmo();
			surface->queue_redraw();

		} break;

		case TRANSFORM_TRANSLATE: {
			Vector3 motion_mask;
			Plane plane;
			bool plane_mv = false;

			switch (_edit.plane) {
				case TRANSFORM_VIEW:
					plane = Plane(_get_camera_normal(), _edit.center);
					break;
				case TRANSFORM_X_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(0).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_Y_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(1).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_Z_AXIS:
					motion_mask = spatial_editor->get_gizmo_transform().basis.get_column(2).normalized();
					plane = Plane(motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized(), _edit.center);
					break;
				case TRANSFORM_YZ:
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(0).normalized(), _edit.center);
					plane_mv = true;
					break;
				case TRANSFORM_XZ:
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(1).normalized(), _edit.center);
					plane_mv = true;
					break;
				case TRANSFORM_XY:
					plane = Plane(spatial_editor->get_gizmo_transform().basis.get_column(2).normalized(), _edit.center);
					plane_mv = true;
					break;
			}

			Vector3 intersection;
			if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
				break;
			}

			Vector3 click;
			if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
				break;
			}

			Vector3 motion = intersection - click;
			if (_edit.plane != TRANSFORM_VIEW) {
				if (!plane_mv) {
					motion = motion_mask.dot(motion) * motion_mask;
				}
			}

			// Disable local transformation for TRANSFORM_VIEW
			bool local_coords = (spatial_editor->are_local_coords_enabled() && _edit.plane != TRANSFORM_VIEW);

			if (_edit.snap || spatial_editor->is_snap_enabled()) {
				snap = spatial_editor->get_translate_snap();
			}
			Vector3 motion_snapped = motion;
			motion_snapped.snap(Vector3(snap, snap, snap));
			// TRANSLATORS: Refers to changing the position of a node in the 3D editor.
			set_message(TTR("Translating:") + " (" + String::num(motion_snapped.x, snap_step_decimals) + ", " +
					String::num(motion_snapped.y, snap_step_decimals) + ", " + String::num(motion_snapped.z, snap_step_decimals) + ")");
			motion = spatial_editor->get_gizmo_transform().basis.inverse().xform(motion);

			List<Node *> &selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				if (sp->has_meta("_edit_lock_")) {
					continue;
				}

				if (se->gizmo.is_valid()) {
					for (KeyValue<int, Transform3D> &GE : se->subgizmos) {
						Transform3D xform = GE.value;
						Transform3D new_xform = _compute_transform(TRANSFORM_TRANSLATE, se->original * xform, xform, motion, snap, local_coords, true); // Force orthogonal with subgizmo.
						new_xform = se->original.affine_inverse() * new_xform;
						se->gizmo->set_subgizmo_transform(GE.key, new_xform);
					}
				} else {
					Transform3D new_xform = _compute_transform(TRANSFORM_TRANSLATE, se->original, se->original_local, motion, snap, local_coords, sp->get_rotation_edit_mode() != Node3D::ROTATION_EDIT_MODE_BASIS);
					_transform_gizmo_apply(se->sp, new_xform, false);
				}
			}

			spatial_editor->update_transform_gizmo();
			surface->queue_redraw();

		} break;

		case TRANSFORM_ROTATE: {
			Plane plane = Plane(_get_camera_normal(), _edit.center);

			Vector3 local_axis;
			Vector3 global_axis;
			switch (_edit.plane) {
				case TRANSFORM_VIEW:
					// local_axis unused
					global_axis = _get_camera_normal();
					break;
				case TRANSFORM_X_AXIS:
					local_axis = Vector3(1, 0, 0);
					break;
				case TRANSFORM_Y_AXIS:
					local_axis = Vector3(0, 1, 0);
					break;
				case TRANSFORM_Z_AXIS:
					local_axis = Vector3(0, 0, 1);
					break;
				case TRANSFORM_YZ:
				case TRANSFORM_XZ:
				case TRANSFORM_XY:
					break;
			}

			if (_edit.plane != TRANSFORM_VIEW) {
				global_axis = spatial_editor->get_gizmo_transform().basis.xform(local_axis).normalized();
			}

			Vector3 intersection;
			if (!plane.intersects_ray(ray_pos, ray, &intersection)) {
				break;
			}

			Vector3 click;
			if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click)) {
				break;
			}

			static const float orthogonal_threshold = Math::cos(Math::deg_to_rad(87.0f));
			bool axis_is_orthogonal = ABS(plane.normal.dot(global_axis)) < orthogonal_threshold;

			double angle = 0.0f;
			if (axis_is_orthogonal) {
				_edit.show_rotation_line = false;
				Vector3 projection_axis = plane.normal.cross(global_axis);
				Vector3 delta = intersection - click;
				float projection = delta.dot(projection_axis);
				angle = (projection * (Math_PI / 2.0f)) / (gizmo_scale * Node3DEditor::GIZMO_CIRCLE_SIZE);
			} else {
				_edit.show_rotation_line = true;
				Vector3 click_axis = (click - _edit.center).normalized();
				Vector3 current_axis = (intersection - _edit.center).normalized();
				angle = click_axis.signed_angle_to(current_axis, global_axis);
			}

			if (_edit.snap || spatial_editor->is_snap_enabled()) {
				snap = spatial_editor->get_rotate_snap();
			}
			angle = Math::rad_to_deg(angle) + snap * 0.5; //else it won't reach +180
			angle -= Math::fmod(angle, snap);
			set_message(vformat(TTR("Rotating %s degrees."), String::num(angle, snap_step_decimals)));
			angle = Math::deg_to_rad(angle);

			bool local_coords = (spatial_editor->are_local_coords_enabled() && _edit.plane != TRANSFORM_VIEW); // Disable local transformation for TRANSFORM_VIEW

			List<Node *> &selection = editor_selection->get_selected_node_list();
			for (Node *E : selection) {
				Node3D *sp = Object::cast_to<Node3D>(E);
				if (!sp) {
					continue;
				}

				Node3DEditorSelectedItem *se = editor_selection->get_node_editor_data<Node3DEditorSelectedItem>(sp);
				if (!se) {
					continue;
				}

				if (sp->has_meta("_edit_lock_")) {
					continue;
				}

				Vector3 compute_axis = local_coords ? local_axis : global_axis;
				if (se->gizmo.is_valid()) {
					for (KeyValue<int, Transform3D> &GE : se->subgizmos) {
						Transform3D xform = GE.value;

						Transform3D new_xform = _compute_transform(TRANSFORM_ROTATE, se->original * xform, xform, compute_axis, angle, local_coords, true); // Force orthogonal with subgizmo.
						if (!local_coords) {
							new_xform = se->original.affine_inverse() * new_xform;
						}
						se->gizmo->set_subgizmo_transform(GE.key, new_xform);
					}
				} else {
					Transform3D new_xform = _compute_transform(TRANSFORM_ROTATE, se->original, se->original_local, compute_axis, angle, local_coords, sp->get_rotation_edit_mode() != Node3D::ROTATION_EDIT_MODE_BASIS);
					_transform_gizmo_apply(se->sp, new_xform, local_coords);
				}
			}

			spatial_editor->update_transform_gizmo();
			surface->queue_redraw();

		} break;
		default: {
		}
	}
}

void Node3DEditorViewport::finish_transform() {
	spatial_editor->set_local_coords_enabled(_edit.original_local);
	spatial_editor->update_transform_gizmo();
	_edit.mode = TRANSFORM_NONE;
	_edit.instant = false;
	surface->queue_redraw();
}

// Register a shortcut and also add it as an input action with the same events.
void Node3DEditorViewport::register_shortcut_action(const String &p_path, const String &p_name, Key p_keycode) {
	Ref<Shortcut> sc = ED_SHORTCUT(p_path, p_name, p_keycode);
	shortcut_changed_callback(sc, p_path);
	// Connect to the change event on the shortcut so the input binding can be updated.
	sc->connect("changed", callable_mp(this, &Node3DEditorViewport::shortcut_changed_callback).bind(sc, p_path));
}

// Update the action in the InputMap to the provided shortcut events.
void Node3DEditorViewport::shortcut_changed_callback(const Ref<Shortcut> p_shortcut, const String &p_shortcut_path) {
	InputMap *im = InputMap::get_singleton();
	if (im->has_action(p_shortcut_path)) {
		im->action_erase_events(p_shortcut_path);
	} else {
		im->add_action(p_shortcut_path);
	}

	for (int i = 0; i < p_shortcut->get_events().size(); i++) {
		im->action_add_event(p_shortcut_path, p_shortcut->get_events()[i]);
	}
}

Node3DEditorViewport::Node3DEditorViewport(Node3DEditor *p_spatial_editor, int p_index) {
	cpu_time_history_index = 0;
	gpu_time_history_index = 0;

	_edit.mode = TRANSFORM_NONE;
	_edit.plane = TRANSFORM_VIEW;
	_edit.snap = true;
	_edit.show_rotation_line = true;
	_edit.instant = false;
	_edit.gizmo_handle = -1;
	_edit.gizmo_handle_secondary = false;

	index = p_index;
	editor_selection = EditorNode::get_singleton()->get_editor_selection();

	orthogonal = false;
	auto_orthogonal = false;
	lock_rotation = false;
	message_time = 0;
	zoom_indicator_delay = 0.0;

	spatial_editor = p_spatial_editor;
	SubViewportContainer *c = memnew(SubViewportContainer);
	subviewport_container = c;
	c->set_stretch(true);
	add_child(c);
	c->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	viewport = memnew(SubViewport);
	viewport->set_disable_input(true);

	c->add_child(viewport);
	surface = memnew(Control);
	surface->set_drag_forwarding(this);
	add_child(surface);
	surface->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	surface->set_clip_contents(true);
	camera = memnew(Camera3D);
	camera->set_disable_gizmos(true);
	camera->set_cull_mask(((1 << 20) - 1) | (1 << (GIZMO_BASE_LAYER + p_index)) | (1 << GIZMO_EDIT_LAYER) | (1 << GIZMO_GRID_LAYER) | (1 << MISC_TOOL_LAYER));
	viewport->add_child(camera);
	camera->make_current();
	surface->set_focus_mode(FOCUS_ALL);

	VBoxContainer *vbox = memnew(VBoxContainer);
	surface->add_child(vbox);
	vbox->set_offset(SIDE_LEFT, 10 * EDSCALE);
	vbox->set_offset(SIDE_TOP, 10 * EDSCALE);

	view_menu = memnew(MenuButton);
	view_menu->set_flat(false);
	view_menu->set_h_size_flags(0);
	view_menu->set_shortcut_context(this);
	vbox->add_child(view_menu);

	display_submenu = memnew(PopupMenu);
	view_menu->get_popup()->set_hide_on_checkable_item_selection(false);
	view_menu->get_popup()->add_child(display_submenu);

	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/top_view"), VIEW_TOP);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/bottom_view"), VIEW_BOTTOM);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/left_view"), VIEW_LEFT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/right_view"), VIEW_RIGHT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/front_view"), VIEW_FRONT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/rear_view"), VIEW_REAR);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_radio_check_item(TTR("Perspective") + " (" + ED_GET_SHORTCUT("spatial_editor/switch_perspective_orthogonal")->get_as_text() + ")", VIEW_PERSPECTIVE);
	view_menu->get_popup()->add_radio_check_item(TTR("Orthogonal") + " (" + ED_GET_SHORTCUT("spatial_editor/switch_perspective_orthogonal")->get_as_text() + ")", VIEW_ORTHOGONAL);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), true);
	view_menu->get_popup()->add_check_item(TTR("Auto Orthogonal Enabled"), VIEW_AUTO_ORTHOGONAL);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUTO_ORTHOGONAL), true);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_lock_rotation", TTR("Lock View Rotation")), VIEW_LOCK_ROTATION);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_normal", TTR("Display Normal")), VIEW_DISPLAY_NORMAL);
	view_menu->get_popup()->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_wireframe", TTR("Display Wireframe")), VIEW_DISPLAY_WIREFRAME);
	view_menu->get_popup()->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_overdraw", TTR("Display Overdraw")), VIEW_DISPLAY_OVERDRAW);
	view_menu->get_popup()->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_lighting", TTR("Display Lighting")), VIEW_DISPLAY_LIGHTING);
	view_menu->get_popup()->add_radio_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_unshaded", TTR("Display Unshaded")), VIEW_DISPLAY_SHADELESS);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), true);
	display_submenu->set_hide_on_checkable_item_selection(false);
	display_submenu->add_radio_check_item(TTR("Directional Shadow Splits"), VIEW_DISPLAY_DEBUG_PSSM_SPLITS);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("Normal Buffer"), VIEW_DISPLAY_NORMAL_BUFFER);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("Shadow Atlas"), VIEW_DISPLAY_DEBUG_SHADOW_ATLAS);
	display_submenu->add_radio_check_item(TTR("Directional Shadow Map"), VIEW_DISPLAY_DEBUG_DIRECTIONAL_SHADOW_ATLAS);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("Decal Atlas"), VIEW_DISPLAY_DEBUG_DECAL_ATLAS);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("VoxelGI Lighting"), VIEW_DISPLAY_DEBUG_VOXEL_GI_LIGHTING);
	display_submenu->add_radio_check_item(TTR("VoxelGI Albedo"), VIEW_DISPLAY_DEBUG_VOXEL_GI_ALBEDO);
	display_submenu->add_radio_check_item(TTR("VoxelGI Emission"), VIEW_DISPLAY_DEBUG_VOXEL_GI_EMISSION);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("SDFGI Cascades"), VIEW_DISPLAY_DEBUG_SDFGI);
	display_submenu->add_radio_check_item(TTR("SDFGI Probes"), VIEW_DISPLAY_DEBUG_SDFGI_PROBES);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("Scene Luminance"), VIEW_DISPLAY_DEBUG_SCENE_LUMINANCE);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("SSAO"), VIEW_DISPLAY_DEBUG_SSAO);
	display_submenu->add_radio_check_item(TTR("SSIL"), VIEW_DISPLAY_DEBUG_SSIL);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("VoxelGI/SDFGI Buffer"), VIEW_DISPLAY_DEBUG_GI_BUFFER);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("Disable Mesh LOD"), VIEW_DISPLAY_DEBUG_DISABLE_LOD);
	display_submenu->add_separator();
	display_submenu->add_radio_check_item(TTR("OmniLight3D Cluster"), VIEW_DISPLAY_DEBUG_CLUSTER_OMNI_LIGHTS);
	display_submenu->add_radio_check_item(TTR("SpotLight3D Cluster"), VIEW_DISPLAY_DEBUG_CLUSTER_SPOT_LIGHTS);
	display_submenu->add_radio_check_item(TTR("Decal Cluster"), VIEW_DISPLAY_DEBUG_CLUSTER_DECALS);
	display_submenu->add_radio_check_item(TTR("ReflectionProbe Cluster"), VIEW_DISPLAY_DEBUG_CLUSTER_REFLECTION_PROBES);
	display_submenu->add_radio_check_item(TTR("Occlusion Culling Buffer"), VIEW_DISPLAY_DEBUG_OCCLUDERS);
	display_submenu->add_radio_check_item(TTR("Motion Vectors"), VIEW_DISPLAY_MOTION_VECTORS);

	display_submenu->set_name("display_advanced");
	view_menu->get_popup()->add_submenu_item(TTR("Display Advanced..."), "display_advanced", VIEW_DISPLAY_ADVANCED);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_environment", TTR("View Environment")), VIEW_ENVIRONMENT);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_gizmos", TTR("View Gizmos")), VIEW_GIZMOS);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_information", TTR("View Information")), VIEW_INFORMATION);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_fps", TTR("View Frame Time")), VIEW_FRAME_TIME);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ENVIRONMENT), true);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_half_resolution", TTR("Half Resolution")), VIEW_HALF_RESOLUTION);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_audio_listener", TTR("Audio Listener")), VIEW_AUDIO_LISTENER);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_audio_doppler", TTR("Enable Doppler")), VIEW_AUDIO_DOPPLER);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_GIZMOS), true);

	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_cinematic_preview", TTR("Cinematic Preview")), VIEW_CINEMATIC_PREVIEW);

	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/focus_origin"), VIEW_CENTER_TO_ORIGIN);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/focus_selection"), VIEW_CENTER_TO_SELECTION);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/align_transform_with_view"), VIEW_ALIGN_TRANSFORM_WITH_VIEW);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/align_rotation_with_view"), VIEW_ALIGN_ROTATION_WITH_VIEW);
	view_menu->get_popup()->connect("id_pressed", callable_mp(this, &Node3DEditorViewport::_menu_option));
	display_submenu->connect("id_pressed", callable_mp(this, &Node3DEditorViewport::_menu_option));
	view_menu->set_disable_shortcuts(true);

	// TODO: Re-evaluate with new OpenGL3 renderer, and implement.
	//if (OS::get_singleton()->get_current_video_driver() == OS::RENDERING_DRIVER_OPENGL3) {
	if (false) {
		// Alternate display modes only work when using the Vulkan renderer; make this explicit.
		const int normal_idx = view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL);
		const int wireframe_idx = view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME);
		const int overdraw_idx = view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW);
		const int shadeless_idx = view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS);
		const String unsupported_tooltip = TTR("Not available when using the OpenGL renderer.");

		view_menu->get_popup()->set_item_disabled(normal_idx, true);
		view_menu->get_popup()->set_item_tooltip(normal_idx, unsupported_tooltip);
		view_menu->get_popup()->set_item_disabled(wireframe_idx, true);
		view_menu->get_popup()->set_item_tooltip(wireframe_idx, unsupported_tooltip);
		view_menu->get_popup()->set_item_disabled(overdraw_idx, true);
		view_menu->get_popup()->set_item_tooltip(overdraw_idx, unsupported_tooltip);
		view_menu->get_popup()->set_item_disabled(shadeless_idx, true);
		view_menu->get_popup()->set_item_tooltip(shadeless_idx, unsupported_tooltip);
	}

	register_shortcut_action("spatial_editor/freelook_left", TTR("Freelook Left"), Key::A);
	register_shortcut_action("spatial_editor/freelook_right", TTR("Freelook Right"), Key::D);
	register_shortcut_action("spatial_editor/freelook_forward", TTR("Freelook Forward"), Key::W);
	register_shortcut_action("spatial_editor/freelook_backwards", TTR("Freelook Backwards"), Key::S);
	register_shortcut_action("spatial_editor/freelook_up", TTR("Freelook Up"), Key::E);
	register_shortcut_action("spatial_editor/freelook_down", TTR("Freelook Down"), Key::Q);
	register_shortcut_action("spatial_editor/freelook_speed_modifier", TTR("Freelook Speed Modifier"), Key::SHIFT);
	register_shortcut_action("spatial_editor/freelook_slow_modifier", TTR("Freelook Slow Modifier"), Key::ALT);

	ED_SHORTCUT("spatial_editor/lock_transform_x", TTR("Lock Transformation to X axis"), Key::X);
	ED_SHORTCUT("spatial_editor/lock_transform_y", TTR("Lock Transformation to Y axis"), Key::Y);
	ED_SHORTCUT("spatial_editor/lock_transform_z", TTR("Lock Transformation to Z axis"), Key::Z);
	ED_SHORTCUT("spatial_editor/lock_transform_yz", TTR("Lock Transformation to YZ plane"), KeyModifierMask::SHIFT | Key::X);
	ED_SHORTCUT("spatial_editor/lock_transform_xz", TTR("Lock Transformation to XZ plane"), KeyModifierMask::SHIFT | Key::Y);
	ED_SHORTCUT("spatial_editor/lock_transform_xy", TTR("Lock Transformation to XY plane"), KeyModifierMask::SHIFT | Key::Z);
	ED_SHORTCUT("spatial_editor/cancel_transform", TTR("Cancel Transformation"), Key::ESCAPE);
	ED_SHORTCUT("spatial_editor/instant_translate", TTR("Begin Translate Transformation"));
	ED_SHORTCUT("spatial_editor/instant_rotate", TTR("Begin Rotate Transformation"));
	ED_SHORTCUT("spatial_editor/instant_scale", TTR("Begin Scale Transformation"));

	preview_camera = memnew(CheckBox);
	preview_camera->set_text(TTR("Preview"));
	preview_camera->set_shortcut(ED_SHORTCUT("spatial_editor/toggle_camera_preview", TTR("Toggle Camera Preview"), KeyModifierMask::CMD_OR_CTRL | Key::P));
	vbox->add_child(preview_camera);
	preview_camera->set_h_size_flags(0);
	preview_camera->hide();
	preview_camera->connect("toggled", callable_mp(this, &Node3DEditorViewport::_toggle_camera_preview));
	previewing = nullptr;
	gizmo_scale = 1.0;

	preview_node = nullptr;

	bottom_center_vbox = memnew(VBoxContainer);
	bottom_center_vbox->set_anchors_preset(LayoutPreset::PRESET_CENTER);
	bottom_center_vbox->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -20 * EDSCALE);
	bottom_center_vbox->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, -10 * EDSCALE);
	bottom_center_vbox->set_h_grow_direction(GROW_DIRECTION_BOTH);
	bottom_center_vbox->set_v_grow_direction(GROW_DIRECTION_BEGIN);
	surface->add_child(bottom_center_vbox);

	info_label = memnew(Label);
	info_label->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -90 * EDSCALE);
	info_label->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -90 * EDSCALE);
	info_label->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, -10 * EDSCALE);
	info_label->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, -10 * EDSCALE);
	info_label->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	info_label->set_v_grow_direction(GROW_DIRECTION_BEGIN);
	surface->add_child(info_label);
	info_label->hide();

	cinema_label = memnew(Label);
	cinema_label->set_anchor_and_offset(SIDE_TOP, ANCHOR_BEGIN, 10 * EDSCALE);
	cinema_label->set_h_grow_direction(GROW_DIRECTION_END);
	cinema_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	surface->add_child(cinema_label);
	cinema_label->set_text(TTR("Cinematic Preview"));
	cinema_label->hide();
	previewing_cinema = false;

	locked_label = memnew(Label);
	locked_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	locked_label->set_h_size_flags(SIZE_SHRINK_CENTER);
	bottom_center_vbox->add_child(locked_label);
	locked_label->set_text(TTR("View Rotation Locked"));
	locked_label->hide();

	zoom_limit_label = memnew(Label);
	zoom_limit_label->set_text(TTR("To zoom further, change the camera's clipping planes (View -> Settings...)"));
	zoom_limit_label->set_name("ZoomLimitMessageLabel");
	zoom_limit_label->add_theme_color_override("font_color", Color(1, 1, 1, 1));
	zoom_limit_label->hide();
	bottom_center_vbox->add_child(zoom_limit_label);

	preview_material_label = memnew(Label);
	preview_material_label->set_anchors_and_offsets_preset(LayoutPreset::PRESET_BOTTOM_LEFT);
	preview_material_label->set_offset(Side::SIDE_TOP, -70 * EDSCALE);
	preview_material_label->set_text(TTR("Overriding material..."));
	preview_material_label->add_theme_color_override("font_color", Color(1, 1, 1, 1));
	preview_material_label->hide();
	surface->add_child(preview_material_label);

	preview_material_label_desc = memnew(Label);
	preview_material_label_desc->set_anchors_and_offsets_preset(LayoutPreset::PRESET_BOTTOM_LEFT);
	preview_material_label_desc->set_offset(Side::SIDE_TOP, -50 * EDSCALE);
	preview_material_label_desc->set_text(TTR("Drag and drop to override the material of any geometry node.\nHold Ctrl when dropping to override a specific surface."));
	preview_material_label_desc->add_theme_color_override("font_color", Color(0.8, 0.8, 0.8, 1));
	preview_material_label_desc->add_theme_constant_override("line_spacing", 0);
	preview_material_label_desc->hide();
	surface->add_child(preview_material_label_desc);

	frame_time_gradient = memnew(Gradient);
	// The color is set when the theme changes.
	frame_time_gradient->add_point(0.5, Color());

	top_right_vbox = memnew(VBoxContainer);
	top_right_vbox->set_anchors_and_offsets_preset(PRESET_TOP_RIGHT, PRESET_MODE_MINSIZE, 10.0 * EDSCALE);
	top_right_vbox->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	// Make sure frame time labels don't touch the viewport's edge.
	top_right_vbox->set_custom_minimum_size(Size2(100, 0) * EDSCALE);
	// Prevent visible spacing between frame time labels.
	top_right_vbox->add_theme_constant_override("separation", 0);

	const int navigation_control_size = 150;

	position_control = memnew(ViewportNavigationControl);
	position_control->set_navigation_mode(Node3DEditorViewport::NAVIGATION_MOVE);
	position_control->set_custom_minimum_size(Size2(navigation_control_size, navigation_control_size) * EDSCALE);
	position_control->set_h_size_flags(SIZE_SHRINK_END);
	position_control->set_anchor_and_offset(SIDE_LEFT, ANCHOR_BEGIN, 0 * EDSCALE);
	position_control->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -navigation_control_size * EDSCALE);
	position_control->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_BEGIN, navigation_control_size * EDSCALE);
	position_control->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0 * EDSCALE);
	position_control->set_viewport(this);
	surface->add_child(position_control);

	look_control = memnew(ViewportNavigationControl);
	look_control->set_navigation_mode(Node3DEditorViewport::NAVIGATION_LOOK);
	look_control->set_custom_minimum_size(Size2(navigation_control_size, navigation_control_size) * EDSCALE);
	look_control->set_h_size_flags(SIZE_SHRINK_END);
	look_control->set_anchor_and_offset(SIDE_LEFT, ANCHOR_END, -navigation_control_size * EDSCALE);
	look_control->set_anchor_and_offset(SIDE_TOP, ANCHOR_END, -navigation_control_size * EDSCALE);
	look_control->set_anchor_and_offset(SIDE_RIGHT, ANCHOR_END, 0 * EDSCALE);
	look_control->set_anchor_and_offset(SIDE_BOTTOM, ANCHOR_END, 0 * EDSCALE);
	look_control->set_viewport(this);
	surface->add_child(look_control);

	rotation_control = memnew(ViewportRotationControl);
	rotation_control->set_custom_minimum_size(Size2(80, 80) * EDSCALE);
	rotation_control->set_h_size_flags(SIZE_SHRINK_END);
	rotation_control->set_viewport(this);
	top_right_vbox->add_child(rotation_control);

	// Individual Labels are used to allow coloring each label with its own color.
	cpu_time_label = memnew(Label);
	top_right_vbox->add_child(cpu_time_label);
	cpu_time_label->hide();

	gpu_time_label = memnew(Label);
	top_right_vbox->add_child(gpu_time_label);
	gpu_time_label->hide();

	fps_label = memnew(Label);
	top_right_vbox->add_child(fps_label);
	fps_label->hide();

	surface->add_child(top_right_vbox);

	accept = nullptr;

	freelook_active = false;
	freelook_speed = EDITOR_GET("editors/3d/freelook/freelook_base_speed");

	selection_menu = memnew(PopupMenu);
	add_child(selection_menu);
	selection_menu->set_min_size(Size2(100, 0) * EDSCALE);
	selection_menu->connect("id_pressed", callable_mp(this, &Node3DEditorViewport::_selection_result_pressed));
	selection_menu->connect("popup_hide", callable_mp(this, &Node3DEditorViewport::_selection_menu_hide));

	if (p_index == 0) {
		view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER), true);
		viewport->set_as_audio_listener_3d(true);
	}

	view_type = VIEW_TYPE_USER;
	_update_name();

	EditorSettings::get_singleton()->connect("settings_changed", callable_mp(this, &Node3DEditorViewport::update_transform_gizmo_view));
}

Node3DEditorViewport::~Node3DEditorViewport() {
	memdelete(frame_time_gradient);
}
