/*************************************************************************/
/*  spatial_editor_plugin.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "spatial_editor_plugin.h"

#include "camera_matrix.h"
#include "core/os/input.h"
#include "editor/animation_editor.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/plugins/animation_player_editor_plugin.h"
#include "editor/plugins/script_editor_plugin.h"
#include "editor/script_editor_debugger.h"
#include "editor/spatial_editor_gizmos.h"
#include "os/keyboard.h"
#include "print_string.h"
#include "project_settings.h"
#include "scene/3d/camera.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/surface_tool.h"
#include "sort.h"

#define DISTANCE_DEFAULT 4

#define GIZMO_ARROW_SIZE 0.3
#define GIZMO_RING_HALF_WIDTH 0.1
//#define GIZMO_SCALE_DEFAULT 0.28
#define GIZMO_SCALE_DEFAULT 0.15
#define GIZMO_PLANE_SIZE 0.2
#define GIZMO_PLANE_DST 0.3
#define GIZMO_CIRCLE_SIZE 0.9

#define ZOOM_MIN_DISTANCE 0.001
#define ZOOM_MULTIPLIER 1.08
#define ZOOM_INDICATOR_DELAY_S 1.5

#define FREELOOK_MIN_SPEED 0.1

#define MIN_Z 0.01
#define MAX_Z 10000

#define MIN_FOV 0.01
#define MAX_FOV 179

void SpatialEditorViewport::_update_camera(float p_interp_delta) {
	if (orthogonal) {
		//camera->set_orthogonal(size.width*cursor.distance,get_znear(),get_zfar());
		camera->set_orthogonal(2 * cursor.distance, 0.1, 8192);
	} else
		camera->set_perspective(get_fov(), get_znear(), get_zfar());

	//when not being manipulated, move softly
	float free_orbit_inertia = EDITOR_DEF("editors/3d/free_orbit_inertia", 0.15);
	float free_translation_inertia = EDITOR_DEF("editors/3d/free_translation_inertia", 0.15);
	//when being manipulated, move more quickly
	float manip_orbit_inertia = EDITOR_DEF("editors/3d/manipulation_orbit_inertia", 0.075);
	float manip_translation_inertia = EDITOR_DEF("editors/3d/manipulation_translation_inertia", 0.075);

	//determine if being manipulated
	bool manipulated = (Input::get_singleton()->get_mouse_button_mask() & (2 | 4)) || Input::get_singleton()->is_key_pressed(KEY_SHIFT) || Input::get_singleton()->is_key_pressed(KEY_ALT) || Input::get_singleton()->is_key_pressed(KEY_CONTROL);

	float orbit_inertia = MAX(0.00001, manipulated ? manip_orbit_inertia : free_orbit_inertia);
	float translation_inertia = MAX(0.0001, manipulated ? manip_translation_inertia : free_translation_inertia);

	Cursor old_camera_cursor = camera_cursor;
	camera_cursor = cursor;

	camera_cursor.x_rot = Math::lerp(old_camera_cursor.x_rot, cursor.x_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));
	camera_cursor.y_rot = Math::lerp(old_camera_cursor.y_rot, cursor.y_rot, MIN(1.f, p_interp_delta * (1 / orbit_inertia)));

	camera_cursor.pos = old_camera_cursor.pos.linear_interpolate(cursor.pos, MIN(1.f, p_interp_delta * (1 / translation_inertia)));
	camera_cursor.distance = Math::lerp(old_camera_cursor.distance, cursor.distance, MIN(1.f, p_interp_delta * (1 / translation_inertia)));

	if (p_interp_delta == 0 || is_freelook_active()) {
		camera_cursor = cursor;
	}

	float tolerance = 0.0001;
	bool equal = true;
	if (Math::abs(old_camera_cursor.x_rot - camera_cursor.x_rot) > tolerance || Math::abs(old_camera_cursor.y_rot - camera_cursor.y_rot) > tolerance)
		equal = false;

	if (equal && old_camera_cursor.pos.distance_squared_to(camera_cursor.pos) > tolerance * tolerance)
		equal = false;

	if (equal && Math::abs(old_camera_cursor.distance - camera_cursor.distance) > tolerance)
		equal = false;

	if (!equal || p_interp_delta == 0 || is_freelook_active()) {

		camera->set_global_transform(to_camera_transform(camera_cursor));
		update_transform_gizmo_view();
	}
}

Transform SpatialEditorViewport::to_camera_transform(const Cursor &p_cursor) const {
	Transform camera_transform;
	camera_transform.translate(p_cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -p_cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -p_cursor.y_rot);

	if (orthogonal)
		camera_transform.translate(0, 0, 4096);
	else
		camera_transform.translate(0, 0, p_cursor.distance);

	return camera_transform;
}

String SpatialEditorGizmo::get_handle_name(int p_idx) const {

	if (get_script_instance() && get_script_instance()->has_method("get_handle_name"))
		return get_script_instance()->call("get_handle_name", p_idx);

	return "";
}

Variant SpatialEditorGizmo::get_handle_value(int p_idx) const {

	if (get_script_instance() && get_script_instance()->has_method("get_handle_value"))
		return get_script_instance()->call("get_handle_value", p_idx);

	return Variant();
}

void SpatialEditorGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	if (get_script_instance() && get_script_instance()->has_method("set_handle"))
		get_script_instance()->call("set_handle", p_idx, p_camera, p_point);
}

void SpatialEditorGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	if (get_script_instance() && get_script_instance()->has_method("commit_handle"))
		get_script_instance()->call("commit_handle", p_idx, p_restore, p_cancel);
}

bool SpatialEditorGizmo::intersect_frustum(const Camera *p_camera, const Vector<Plane> &p_frustum) {

	return false;
}

bool SpatialEditorGizmo::intersect_ray(const Camera *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal, int *r_gizmo_handle, bool p_sec_first) {

	return false;
}

SpatialEditorGizmo::SpatialEditorGizmo() {

	selected = false;
}

int SpatialEditorViewport::get_selected_count() const {

	Map<Node *, Object *> &selection = editor_selection->get_selection();

	int count = 0;

	for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

		Spatial *sp = Object::cast_to<Spatial>(E->key());
		if (!sp)
			continue;

		SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
		if (!se)
			continue;

		count++;
	}

	return count;
}

float SpatialEditorViewport::get_znear() const {

	return CLAMP(spatial_editor->get_znear(), MIN_Z, MAX_Z);
}
float SpatialEditorViewport::get_zfar() const {

	return CLAMP(spatial_editor->get_zfar(), MIN_Z, MAX_Z);
}
float SpatialEditorViewport::get_fov() const {

	return CLAMP(spatial_editor->get_fov(), MIN_FOV, MAX_FOV);
}

Transform SpatialEditorViewport::_get_camera_transform() const {

	return camera->get_global_transform();
}

Vector3 SpatialEditorViewport::_get_camera_pos() const {

	return _get_camera_transform().origin;
}

Point2 SpatialEditorViewport::_point_to_screen(const Vector3 &p_point) {

	return camera->unproject_position(p_point);
}

Vector3 SpatialEditorViewport::_get_ray_pos(const Vector2 &p_pos) const {

	return camera->project_ray_origin(p_pos);
}

Vector3 SpatialEditorViewport::_get_camera_normal() const {

	return -_get_camera_transform().basis.get_axis(2);
}

Vector3 SpatialEditorViewport::_get_ray(const Vector2 &p_pos) const {

	return camera->project_ray_normal(p_pos);
}
/*
void SpatialEditorViewport::_clear_id(Spatial *p_node) {


	editor_selection->remove_node(p_node);


}
*/
void SpatialEditorViewport::_clear_selected() {

	editor_selection->clear();
}

void SpatialEditorViewport::_select_clicked(bool p_append, bool p_single) {

	if (!clicked)
		return;

	Spatial *sp = Object::cast_to<Spatial>(ObjectDB::get_instance(clicked));
	if (!sp)
		return;

	_select(sp, clicked_wants_append, true);
}

void SpatialEditorViewport::_select(Spatial *p_node, bool p_append, bool p_single) {

	if (!p_append) {

		// should not modify the selection..

		editor_selection->clear();
		editor_selection->add_node(p_node);

	} else {

		if (editor_selection->is_selected(p_node) && p_single) {
			//erase
			editor_selection->remove_node(p_node);
		} else {

			editor_selection->add_node(p_node);
		}
	}
}

ObjectID SpatialEditorViewport::_select_ray(const Point2 &p_pos, bool p_append, bool &r_includes_current, int *r_gizmo_handle, bool p_alt_select) {

	if (r_gizmo_handle)
		*r_gizmo_handle = -1;

	Vector3 ray = _get_ray(p_pos);
	Vector3 pos = _get_ray_pos(p_pos);

	Vector<ObjectID> instances = VisualServer::get_singleton()->instances_cull_ray(pos, ray, get_tree()->get_root()->get_world()->get_scenario());
	Set<Ref<SpatialEditorGizmo> > found_gizmos;

	Node *edited_scene = get_tree()->get_edited_scene_root();
	ObjectID closest = 0;
	Spatial *item = NULL;
	float closest_dist = 1e20;
	int selected_handle = -1;

	for (int i = 0; i < instances.size(); i++) {

		Spatial *spat = Object::cast_to<Spatial>(ObjectDB::get_instance(instances[i]));

		if (!spat)
			continue;

		Ref<SpatialEditorGizmo> seg = spat->get_gizmo();

		if ((!seg.is_valid()) || found_gizmos.has(seg)) {
			continue;
		}

		found_gizmos.insert(seg);
		Vector3 point;
		Vector3 normal;

		int handle = -1;
		bool inters = seg->intersect_ray(camera, p_pos, point, normal, NULL, p_alt_select);

		if (!inters)
			continue;

		float dist = pos.distance_to(point);

		if (dist < 0)
			continue;

		if (dist < closest_dist) {
			//make sure that whathever is selected is editable
			while (spat && spat != edited_scene && spat->get_owner() != edited_scene && !edited_scene->is_editable_instance(spat->get_owner())) {

				spat = Object::cast_to<Spatial>(spat->get_owner());
			}

			if (spat) {
				item = spat;
				closest = spat->get_instance_id();
				closest_dist = dist;
				selected_handle = handle;
			} else {
				ERR_PRINT("Bug?");
			}
		}

		//	if (editor_selection->is_selected(spat))
		//		r_includes_current=true;
	}

	if (!item)
		return 0;

	if (!editor_selection->is_selected(item) || (r_gizmo_handle && selected_handle >= 0)) {

		if (r_gizmo_handle)
			*r_gizmo_handle = selected_handle;
	}

	return closest;
}

void SpatialEditorViewport::_find_items_at_pos(const Point2 &p_pos, bool &r_includes_current, Vector<_RayResult> &results, bool p_alt_select) {

	Vector3 ray = _get_ray(p_pos);
	Vector3 pos = _get_ray_pos(p_pos);

	Vector<ObjectID> instances = VisualServer::get_singleton()->instances_cull_ray(pos, ray, get_tree()->get_root()->get_world()->get_scenario());
	Set<Ref<SpatialEditorGizmo> > found_gizmos;

	r_includes_current = false;

	for (int i = 0; i < instances.size(); i++) {

		Spatial *spat = Object::cast_to<Spatial>(ObjectDB::get_instance(instances[i]));

		if (!spat)
			continue;

		Ref<SpatialEditorGizmo> seg = spat->get_gizmo();

		if (!seg.is_valid())
			continue;

		if (found_gizmos.has(seg))
			continue;

		found_gizmos.insert(seg);
		Vector3 point;
		Vector3 normal;

		int handle = -1;
		bool inters = seg->intersect_ray(camera, p_pos, point, normal, NULL, p_alt_select);

		if (!inters)
			continue;

		float dist = pos.distance_to(point);

		if (dist < 0)
			continue;

		if (editor_selection->is_selected(spat))
			r_includes_current = true;

		_RayResult res;
		res.item = spat;
		res.depth = dist;
		res.handle = handle;
		results.push_back(res);
	}

	if (results.empty())
		return;

	results.sort();
}

Vector3 SpatialEditorViewport::_get_screen_to_space(const Vector3 &p_vector3) {

	CameraMatrix cm;
	cm.set_perspective(get_fov(), get_size().aspect(), get_znear(), get_zfar());
	float screen_w, screen_h;
	cm.get_viewport_size(screen_w, screen_h);

	Transform camera_transform;
	camera_transform.translate(cursor.pos);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
	camera_transform.translate(0, 0, cursor.distance);

	return camera_transform.xform(Vector3(((p_vector3.x / get_size().width) * 2.0 - 1.0) * screen_w, ((1.0 - (p_vector3.y / get_size().height)) * 2.0 - 1.0) * screen_h, -get_znear()));
}

void SpatialEditorViewport::_select_region() {

	if (cursor.region_begin == cursor.region_end)
		return; //nothing really

	Vector3 box[4] = {
		Vector3(
				MIN(cursor.region_begin.x, cursor.region_end.x),
				MIN(cursor.region_begin.y, cursor.region_end.y),
				0),
		Vector3(
				MAX(cursor.region_begin.x, cursor.region_end.x),
				MIN(cursor.region_begin.y, cursor.region_end.y),
				0),
		Vector3(
				MAX(cursor.region_begin.x, cursor.region_end.x),
				MAX(cursor.region_begin.y, cursor.region_end.y),
				0),
		Vector3(
				MIN(cursor.region_begin.x, cursor.region_end.x),
				MAX(cursor.region_begin.y, cursor.region_end.y),
				0)
	};

	Vector<Plane> frustum;

	Vector3 cam_pos = _get_camera_pos();
	Set<Ref<SpatialEditorGizmo> > found_gizmos;

	for (int i = 0; i < 4; i++) {

		Vector3 a = _get_screen_to_space(box[i]);
		Vector3 b = _get_screen_to_space(box[(i + 1) % 4]);
		frustum.push_back(Plane(a, b, cam_pos));
	}

	Plane near(cam_pos, -_get_camera_normal());
	near.d -= get_znear();

	frustum.push_back(near);

	Plane far = -near;
	far.d += 500.0;

	frustum.push_back(far);

	Vector<ObjectID> instances = VisualServer::get_singleton()->instances_cull_convex(frustum, get_tree()->get_root()->get_world()->get_scenario());

	for (int i = 0; i < instances.size(); i++) {

		Spatial *sp = Object::cast_to<Spatial>(ObjectDB::get_instance(instances[i]));
		if (!sp)
			continue;

		Ref<SpatialEditorGizmo> seg = sp->get_gizmo();

		if (!seg.is_valid())
			continue;

		if (found_gizmos.has(seg))
			continue;

		if (seg->intersect_frustum(camera, frustum))
			_select(sp, true, false);
	}
}

void SpatialEditorViewport::_update_name() {

	String ortho = orthogonal ? TTR("Orthogonal") : TTR("Perspective");

	if (name != "")
		view_menu->set_text("[ " + name + " " + ortho + " ]");
	else
		view_menu->set_text("[ " + ortho + " ]");
}

void SpatialEditorViewport::_compute_edit(const Point2 &p_point) {

	_edit.click_ray = _get_ray(Vector2(p_point.x, p_point.y));
	_edit.click_ray_pos = _get_ray_pos(Vector2(p_point.x, p_point.y));
	_edit.plane = TRANSFORM_VIEW;
	spatial_editor->update_transform_gizmo();
	_edit.center = spatial_editor->get_gizmo_transform().origin;

	List<Node *> &selection = editor_selection->get_selected_node_list();

	//Vector3 center;
	//int nc=0;
	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Spatial *sp = Object::cast_to<Spatial>(E->get());
		if (!sp)
			continue;

		SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
		if (!se)
			continue;

		se->original = se->sp->get_global_transform();
		//center+=se->original.origin;
		//nc++;
	}

	/*
	if (nc)
		_edit.center=center/float(nc);
	*/
}

static int _get_key_modifier_setting(const String &p_property) {

	switch (EditorSettings::get_singleton()->get(p_property).operator int()) {

		case 0: return 0;
		case 1: return KEY_SHIFT;
		case 2: return KEY_ALT;
		case 3: return KEY_META;
		case 4: return KEY_CONTROL;
	}
	return 0;
}

static int _get_key_modifier(Ref<InputEventWithModifiers> e) {
	if (e->get_shift())
		return KEY_SHIFT;
	if (e->get_alt())
		return KEY_ALT;
	if (e->get_control())
		return KEY_CONTROL;
	if (e->get_metakey())
		return KEY_META;
	return 0;
}

bool SpatialEditorViewport::_gizmo_select(const Vector2 &p_screenpos, bool p_highlight_only) {

	if (!spatial_editor->is_gizmo_visible())
		return false;
	if (get_selected_count() == 0) {
		if (p_highlight_only)
			spatial_editor->select_gizmo_highlight_axis(-1);
		return false;
	}

	Vector3 ray_pos = _get_ray_pos(Vector2(p_screenpos.x, p_screenpos.y));
	Vector3 ray = _get_ray(Vector2(p_screenpos.x, p_screenpos.y));

	Transform gt = spatial_editor->get_gizmo_transform();
	float gs = gizmo_scale;

	if (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_MOVE) {

		int col_axis = -1;
		float col_d = 1e20;

		for (int i = 0; i < 3; i++) {

			Vector3 grabber_pos = gt.origin + gt.basis.get_axis(i) * gs;
			float grabber_radius = gs * GIZMO_ARROW_SIZE;

			Vector3 r;

			if (Geometry::segment_intersects_sphere(ray_pos, ray_pos + ray * 10000.0, grabber_pos, grabber_radius, &r)) {
				float d = r.distance_to(ray_pos);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		bool is_plane_translate = false;
		// second try
		if (col_axis == -1) {
			col_d = 1e20;

			for (int i = 0; i < 3; i++) {

				Vector3 ivec2 = gt.basis.get_axis((i + 1) % 3).normalized();
				Vector3 ivec3 = gt.basis.get_axis((i + 2) % 3).normalized();

				Vector3 grabber_pos = gt.origin + (ivec2 + ivec3) * gs * (GIZMO_PLANE_SIZE + GIZMO_PLANE_DST);

				Vector3 r;
				Plane plane(gt.origin, gt.basis.get_axis(i).normalized());

				if (plane.intersects_ray(ray_pos, ray, &r)) {

					float dist = r.distance_to(grabber_pos);
					if (dist < (gs * GIZMO_PLANE_SIZE)) {

						float d = ray_pos.distance_to(r);
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
				_compute_edit(Point2(p_screenpos.x, p_screenpos.y));
				_edit.plane = TransformPlane(TRANSFORM_X_AXIS + col_axis + (is_plane_translate ? 3 : 0));
			}
			return true;
		}
	}

	if (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_ROTATE) {

		int col_axis = -1;
		float col_d = 1e20;

		for (int i = 0; i < 3; i++) {

			Plane plane(gt.origin, gt.basis.get_axis(i).normalized());
			Vector3 r;
			if (!plane.intersects_ray(ray_pos, ray, &r))
				continue;

			float dist = r.distance_to(gt.origin);

			if (dist > gs * (GIZMO_CIRCLE_SIZE - GIZMO_RING_HALF_WIDTH) && dist < gs * (GIZMO_CIRCLE_SIZE + GIZMO_RING_HALF_WIDTH)) {

				float d = ray_pos.distance_to(r);
				if (d < col_d) {
					col_d = d;
					col_axis = i;
				}
			}
		}

		if (col_axis != -1) {

			if (p_highlight_only) {

				spatial_editor->select_gizmo_highlight_axis(col_axis + 3);
			} else {
				//handle rotate
				_edit.mode = TRANSFORM_ROTATE;
				_compute_edit(Point2(p_screenpos.x, p_screenpos.y));
				_edit.plane = TransformPlane(TRANSFORM_X_AXIS + col_axis);
			}
			return true;
		}
	}

	if (p_highlight_only)
		spatial_editor->select_gizmo_highlight_axis(-1);

	return false;
}

void SpatialEditorViewport::_smouseenter() {

	if (!surface->has_focus() && (!get_focus_owner() || !get_focus_owner()->is_text_field()))
		surface->grab_focus();
}

void SpatialEditorViewport::_smouseexit() {

	_remove_preview();
}

void SpatialEditorViewport::_list_select(Ref<InputEventMouseButton> b) {

	_find_items_at_pos(b->get_position(), clicked_includes_current, selection_results, b->get_shift());

	Node *scene = editor->get_edited_scene();

	for (int i = 0; i < selection_results.size(); i++) {
		Spatial *item = selection_results[i].item;
		if (item != scene && item->get_owner() != scene && !scene->is_editable_instance(item->get_owner())) {
			//invalid result
			selection_results.remove(i);
			i--;
		}
	}

	clicked_wants_append = b->get_shift();

	if (selection_results.size() == 1) {

		clicked = selection_results[0].item->get_instance_id();
		selection_results.clear();

		if (clicked) {
			_select_clicked(clicked_wants_append, true);
			clicked = 0;
		}

	} else if (!selection_results.empty()) {

		NodePath root_path = get_tree()->get_edited_scene_root()->get_path();
		StringName root_name = root_path.get_name(root_path.get_name_count() - 1);

		for (int i = 0; i < selection_results.size(); i++) {

			Spatial *spat = selection_results[i].item;

			Ref<Texture> icon;
			if (spat->has_meta("_editor_icon"))
				icon = spat->get_meta("_editor_icon");
			else
				icon = get_icon(has_icon(spat->get_class(), "EditorIcons") ? spat->get_class() : String("Object"), "EditorIcons");

			String node_path = "/" + root_name + "/" + root_path.rel_path_to(spat->get_path());

			selection_menu->add_item(spat->get_name());
			selection_menu->set_item_icon(i, icon);
			selection_menu->set_item_metadata(i, node_path);
			selection_menu->set_item_tooltip(i, String(spat->get_name()) + "\nType: " + spat->get_class() + "\nPath: " + node_path);
		}

		selection_menu->set_global_position(b->get_global_position());
		selection_menu->popup();
		selection_menu->call_deferred("grab_click_focus");
		selection_menu->set_invalidate_click_until_motion();
	}
}
void SpatialEditorViewport::_sinput(const Ref<InputEvent> &p_event) {

	if (previewing)
		return; //do NONE

	{
		EditorNode *en = editor;
		EditorPluginList *force_input_forwarding_list = en->get_editor_plugins_force_input_forwarding();
		if (!force_input_forwarding_list->empty()) {
			bool discard = force_input_forwarding_list->forward_spatial_gui_input(camera, p_event, true);
			if (discard)
				return;
		}
	}
	{
		EditorNode *en = editor;
		EditorPluginList *over_plugin_list = en->get_editor_plugins_over();
		if (!over_plugin_list->empty()) {
			bool discard = over_plugin_list->forward_spatial_gui_input(camera, p_event, false);
			if (discard)
				return;
		}
	}

	Ref<InputEventMouseButton> b = p_event;

	if (b.is_valid()) {

		switch (b->get_button_index()) {

			case BUTTON_WHEEL_UP: {
				scale_cursor_distance(is_freelook_active() ? ZOOM_MULTIPLIER : 1.0 / ZOOM_MULTIPLIER);
			} break;

			case BUTTON_WHEEL_DOWN: {
				scale_cursor_distance(is_freelook_active() ? 1.0 / ZOOM_MULTIPLIER : ZOOM_MULTIPLIER);
			} break;

			case BUTTON_RIGHT: {

				NavigationScheme nav_scheme = (NavigationScheme)EditorSettings::get_singleton()->get("editors/3d/navigation_scheme").operator int();

				if (b->is_pressed() && _edit.gizmo.is_valid()) {
					//restore
					_edit.gizmo->commit_handle(_edit.gizmo_handle, _edit.gizmo_initial_value, true);
					_edit.gizmo = Ref<SpatialEditorGizmo>();
				}

				if (_edit.mode == TRANSFORM_NONE && b->is_pressed()) {

					if (b->get_alt()) {

						if (nav_scheme == NAVIGATION_MAYA)
							break;

						_list_select(b);
						return;
					}
				}

				if (_edit.mode != TRANSFORM_NONE && b->is_pressed()) {
					//cancel motion
					_edit.mode = TRANSFORM_NONE;
					//_validate_selection();

					List<Node *> &selection = editor_selection->get_selected_node_list();

					for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

						Spatial *sp = Object::cast_to<Spatial>(E->get());
						if (!sp)
							continue;

						SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
						if (!se)
							continue;

						sp->set_global_transform(se->original);
					}
					surface->update();
					//VisualServer::get_singleton()->poly_clear(indicators);
					set_message(TTR("Transform Aborted."), 3);
				}

				if (b->is_pressed()) {
					int mod = _get_key_modifier(b);
					if (mod == _get_key_modifier_setting("editors/3d/freelook_activation_modifier")) {
						freelook_active = true;
					}
				} else {
					freelook_active = false;
				}

				if (freelook_active && !surface->has_focus()) {
					// Focus usually doesn't trigger on right-click, but in case of freelook it should,
					// otherwise using keyboard navigation would misbehave
					surface->grab_focus();
				}

			} break;
			case BUTTON_MIDDLE: {

				if (b->is_pressed() && _edit.mode != TRANSFORM_NONE) {

					switch (_edit.plane) {

						case TRANSFORM_VIEW: {

							_edit.plane = TRANSFORM_X_AXIS;
							set_message(TTR("X-Axis Transform."), 2);
							name = "";
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
					}
				}
			} break;
			case BUTTON_LEFT: {

				if (b->is_pressed()) {

					NavigationScheme nav_scheme = (NavigationScheme)EditorSettings::get_singleton()->get("editors/3d/navigation_scheme").operator int();
					if ((nav_scheme == NAVIGATION_MAYA || nav_scheme == NAVIGATION_MODO) && b->get_alt()) {
						break;
					}

					if (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_LIST_SELECT) {
						_list_select(b);
						break;
					}

					_edit.mouse_pos = b->get_position();
					_edit.snap = false;
					_edit.mode = TRANSFORM_NONE;

					//gizmo has priority over everything

					bool can_select_gizmos = true;

					{
						int idx = view_menu->get_popup()->get_item_index(VIEW_GIZMOS);
						can_select_gizmos = view_menu->get_popup()->is_item_checked(idx);
					}

					if (can_select_gizmos && spatial_editor->get_selected()) {

						Ref<SpatialEditorGizmo> seg = spatial_editor->get_selected()->get_gizmo();
						if (seg.is_valid()) {
							int handle = -1;
							Vector3 point;
							Vector3 normal;
							bool inters = seg->intersect_ray(camera, _edit.mouse_pos, point, normal, &handle, b->get_shift());
							if (inters && handle != -1) {

								_edit.gizmo = seg;
								_edit.gizmo_handle = handle;
								//_edit.gizmo_initial_pos=seg->get_handle_pos(gizmo_handle);
								_edit.gizmo_initial_value = seg->get_handle_value(handle);
								break;
							}
						}
					}

					if (_gizmo_select(_edit.mouse_pos))
						break;

					clicked = 0;
					clicked_includes_current = false;

					if ((spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT && b->get_control()) || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_ROTATE) {

						/* HANDLE ROTATION */
						if (get_selected_count() == 0)
							break; //bye
						//handle rotate
						_edit.mode = TRANSFORM_ROTATE;
						_compute_edit(b->get_position());
						break;
					}

					if (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_MOVE) {

						if (get_selected_count() == 0)
							break; //bye
						//handle translate
						_edit.mode = TRANSFORM_TRANSLATE;
						_compute_edit(b->get_position());
						break;
					}

					if (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SCALE) {

						if (get_selected_count() == 0)
							break; //bye
						//handle rotate
						_edit.mode = TRANSFORM_SCALE;
						_compute_edit(b->get_position());
						break;
					}

					// todo scale

					int gizmo_handle = -1;

					clicked = _select_ray(b->get_position(), b->get_shift(), clicked_includes_current, &gizmo_handle, b->get_shift());

					//clicking is always deferred to either move or release

					clicked_wants_append = b->get_shift();

					if (!clicked) {

						if (!clicked_wants_append)
							_clear_selected();

						//default to regionselect
						cursor.region_select = true;
						cursor.region_begin = b->get_position();
						cursor.region_end = b->get_position();
					}

					if (clicked && gizmo_handle >= 0) {

						Spatial *spa = Object::cast_to<Spatial>(ObjectDB::get_instance(clicked));
						if (spa) {

							Ref<SpatialEditorGizmo> seg = spa->get_gizmo();
							if (seg.is_valid()) {

								_edit.gizmo = seg;
								_edit.gizmo_handle = gizmo_handle;
								//_edit.gizmo_initial_pos=seg->get_handle_pos(gizmo_handle);
								_edit.gizmo_initial_value = seg->get_handle_value(gizmo_handle);
								//print_line("GIZMO: "+itos(gizmo_handle)+" FROMPOS: "+_edit.orig_gizmo_pos);
								break;
							}
						}
						//_compute_edit(Point2(b.x,b.y)); //in case a motion happens..
					}

					surface->update();
				} else {

					if (_edit.gizmo.is_valid()) {

						_edit.gizmo->commit_handle(_edit.gizmo_handle, _edit.gizmo_initial_value, false);
						_edit.gizmo = Ref<SpatialEditorGizmo>();
						break;
					}
					if (clicked) {
						_select_clicked(clicked_wants_append, true);
						//clickd processing was deferred
						clicked = 0;
					}

					if (cursor.region_select) {
						_select_region();
						cursor.region_select = false;
						surface->update();
					}

					if (_edit.mode != TRANSFORM_NONE) {

						static const char *_transform_name[4] = { "None", "Rotate", "Translate", "Scale" };
						undo_redo->create_action(_transform_name[_edit.mode]);

						List<Node *> &selection = editor_selection->get_selected_node_list();

						for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

							Spatial *sp = Object::cast_to<Spatial>(E->get());
							if (!sp)
								continue;

							SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
							if (!se)
								continue;

							undo_redo->add_do_method(sp, "set_global_transform", sp->get_global_transform());
							undo_redo->add_undo_method(sp, "set_global_transform", se->original);
						}
						undo_redo->commit_action();
						_edit.mode = TRANSFORM_NONE;
						//VisualServer::get_singleton()->poly_clear(indicators);
						set_message("");
					}

					surface->update();
				}

			} break;
		}
	}

	Ref<InputEventMouseMotion> m = p_event;

	if (m.is_valid()) {

		_edit.mouse_pos = m->get_position();

		if (spatial_editor->get_selected()) {

			Ref<SpatialEditorGizmo> seg = spatial_editor->get_selected()->get_gizmo();
			if (seg.is_valid()) {

				int selected_handle = -1;

				int handle = -1;
				Vector3 point;
				Vector3 normal;
				bool inters = seg->intersect_ray(camera, _edit.mouse_pos, point, normal, &handle, false);
				if (inters && handle != -1) {

					selected_handle = handle;
				}

				if (selected_handle != spatial_editor->get_over_gizmo_handle()) {
					spatial_editor->set_over_gizmo_handle(selected_handle);
					spatial_editor->get_selected()->update_gizmo();
					if (selected_handle != -1)
						spatial_editor->select_gizmo_highlight_axis(-1);
				}
			}
		}

		if (spatial_editor->get_over_gizmo_handle() == -1 && !(m->get_button_mask() & 1) && !_edit.gizmo.is_valid()) {

			_gizmo_select(_edit.mouse_pos, true);
		}

		NavigationScheme nav_scheme = (NavigationScheme)EditorSettings::get_singleton()->get("editors/3d/navigation_scheme").operator int();
		NavigationMode nav_mode = NAVIGATION_NONE;

		if (_edit.gizmo.is_valid()) {

			_edit.gizmo->set_handle(_edit.gizmo_handle, camera, m->get_position());
			Variant v = _edit.gizmo->get_handle_value(_edit.gizmo_handle);
			String n = _edit.gizmo->get_handle_name(_edit.gizmo_handle);
			set_message(n + ": " + String(v));

		} else if (m->get_button_mask() & BUTTON_MASK_LEFT) {

			if (nav_scheme == NAVIGATION_MAYA && m->get_alt()) {
				nav_mode = NAVIGATION_ORBIT;
			} else if (nav_scheme == NAVIGATION_MODO && m->get_alt() && m->get_shift()) {
				nav_mode = NAVIGATION_PAN;
			} else if (nav_scheme == NAVIGATION_MODO && m->get_alt() && m->get_control()) {
				nav_mode = NAVIGATION_ZOOM;
			} else if (nav_scheme == NAVIGATION_MODO && m->get_alt()) {
				nav_mode = NAVIGATION_ORBIT;
			} else {
				if (clicked) {

					if (!clicked_includes_current) {

						_select_clicked(clicked_wants_append, true);
						//clickd processing was deferred
					}

					_compute_edit(_edit.mouse_pos);
					clicked = 0;

					_edit.mode = TRANSFORM_TRANSLATE;
				}

				if (cursor.region_select && nav_mode == NAVIGATION_NONE) {

					cursor.region_end = m->get_position();
					surface->update();
					return;
				}

				if (_edit.mode == TRANSFORM_NONE && nav_mode == NAVIGATION_NONE)
					return;

				Vector3 ray_pos = _get_ray_pos(m->get_position());
				Vector3 ray = _get_ray(m->get_position());

				switch (_edit.mode) {

					case TRANSFORM_SCALE: {

						Plane plane = Plane(_edit.center, _get_camera_normal());

						Vector3 intersection;
						if (!plane.intersects_ray(ray_pos, ray, &intersection))
							break;

						Vector3 click;
						if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click))
							break;

						float center_click_dist = click.distance_to(_edit.center);
						float center_inters_dist = intersection.distance_to(_edit.center);
						if (center_click_dist == 0)
							break;

						float scale = (center_inters_dist / center_click_dist) * 100.0;

						if (_edit.snap || spatial_editor->is_snap_enabled()) {

							scale = Math::stepify(scale, spatial_editor->get_scale_snap());
						}

						set_message(vformat(TTR("Scaling to %s%%."), String::num(scale, 1)));
						scale /= 100.0;

						Transform r;
						r.basis.scale(Vector3(scale, scale, scale));

						List<Node *> &selection = editor_selection->get_selected_node_list();

						for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

							Spatial *sp = Object::cast_to<Spatial>(E->get());
							if (!sp)
								continue;

							SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
							if (!se)
								continue;

							Transform original = se->original;

							Transform base = Transform(Basis(), _edit.center);
							Transform t = base * (r * (base.inverse() * original));

							sp->set_global_transform(t);
						}

						surface->update();

					} break;

					case TRANSFORM_TRANSLATE: {

						Vector3 motion_mask;
						Plane plane;
						bool plane_mv;

						switch (_edit.plane) {
							case TRANSFORM_VIEW:
								motion_mask = Vector3(0, 0, 0);
								plane = Plane(_edit.center, _get_camera_normal());
								break;
							case TRANSFORM_X_AXIS:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(0);
								plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized());
								break;
							case TRANSFORM_Y_AXIS:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(1);
								plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized());
								break;
							case TRANSFORM_Z_AXIS:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(2);
								plane = Plane(_edit.center, motion_mask.cross(motion_mask.cross(_get_camera_normal())).normalized());
								break;
							case TRANSFORM_YZ:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(2) + spatial_editor->get_gizmo_transform().basis.get_axis(1);
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(0));
								plane_mv = true;
								break;
							case TRANSFORM_XZ:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(2) + spatial_editor->get_gizmo_transform().basis.get_axis(0);
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(1));
								plane_mv = true;
								break;
							case TRANSFORM_XY:
								motion_mask = spatial_editor->get_gizmo_transform().basis.get_axis(0) + spatial_editor->get_gizmo_transform().basis.get_axis(1);
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(2));
								plane_mv = true;
								break;
						}

						Vector3 intersection;
						if (!plane.intersects_ray(ray_pos, ray, &intersection))
							break;

						Vector3 click;
						if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click))
							break;

						//_validate_selection();
						Vector3 motion = intersection - click;
						if (motion_mask != Vector3()) {
							if (plane_mv)
								motion *= motion_mask;
							else
								motion = motion_mask.dot(motion) * motion_mask;
						}

						//set_message("Translating: "+motion);

						List<Node *> &selection = editor_selection->get_selected_node_list();

						float snap = 0;

						if (_edit.snap || spatial_editor->is_snap_enabled()) {

							snap = spatial_editor->get_translate_snap();
							bool local_coords = spatial_editor->are_local_coords_enabled();

							if (local_coords) {
								bool multiple = false;
								Spatial *node = NULL;
								for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

									Spatial *sp = Object::cast_to<Spatial>(E->get());
									if (!sp) {
										continue;
									}
									if (node) {
										multiple = true;
										break;
									} else {
										node = sp;
									}
								}

								if (multiple) {
									motion.snap(Vector3(snap, snap, snap));
								} else {
									Basis b = node->get_global_transform().basis.orthonormalized();
									Vector3 local_motion = b.inverse().xform(motion);
									local_motion.snap(Vector3(snap, snap, snap));
									motion = b.xform(local_motion);
								}

							} else {
								motion.snap(Vector3(snap, snap, snap));
							}
						}

						for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

							Spatial *sp = Object::cast_to<Spatial>(E->get());
							if (!sp) {
								continue;
							}

							SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
							if (!se) {
								continue;
							}

							Transform t = se->original;
							t.origin += motion;
							sp->set_global_transform(t);
						}
					} break;

					case TRANSFORM_ROTATE: {

						Plane plane;

						switch (_edit.plane) {
							case TRANSFORM_VIEW:
								plane = Plane(_edit.center, _get_camera_normal());
								break;
							case TRANSFORM_X_AXIS:
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(0));
								break;
							case TRANSFORM_Y_AXIS:
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(1));
								break;
							case TRANSFORM_Z_AXIS:
								plane = Plane(_edit.center, spatial_editor->get_gizmo_transform().basis.get_axis(2));
								break;
						}

						Vector3 intersection;
						if (!plane.intersects_ray(ray_pos, ray, &intersection))
							break;

						Vector3 click;
						if (!plane.intersects_ray(_edit.click_ray_pos, _edit.click_ray, &click))
							break;

						Vector3 y_axis = (click - _edit.center).normalized();
						Vector3 x_axis = plane.normal.cross(y_axis).normalized();

						float angle = Math::atan2(x_axis.dot(intersection - _edit.center), y_axis.dot(intersection - _edit.center));
						if (_edit.snap || spatial_editor->is_snap_enabled()) {

							float snap = spatial_editor->get_rotate_snap();

							if (snap) {
								angle = Math::rad2deg(angle) + snap * 0.5; //else it wont reach +180
								angle -= Math::fmod(angle, snap);
								set_message(vformat(TTR("Rotating %s degrees."), rtos(angle)));
								angle = Math::deg2rad(angle);
							} else
								set_message(vformat(TTR("Rotating %s degrees."), rtos(Math::rad2deg(angle))));

						} else {
							set_message(vformat(TTR("Rotating %s degrees."), rtos(Math::rad2deg(angle))));
						}

						Transform r;
						r.basis.rotate(plane.normal, angle);

						List<Node *> &selection = editor_selection->get_selected_node_list();

						for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

							Spatial *sp = Object::cast_to<Spatial>(E->get());
							if (!sp)
								continue;

							SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
							if (!se)
								continue;

							Transform original = se->original;

							Transform base = Transform(Basis(), _edit.center);
							Transform t = base * r * base.inverse() * original;

							sp->set_global_transform(t);
						}

						surface->update();
						/*
						VisualServer::get_singleton()->poly_clear(indicators);

						Vector<Vector3> points;
						Vector<Vector3> empty;
						Vector<Color> colors;
						points.push_back(intersection);
						points.push_back(_edit.original.origin);
						colors.push_back( Color(255,155,100) );
						colors.push_back( Color(255,155,100) );
						VisualServer::get_singleton()->poly_add_primitive(indicators,points,empty,colors,empty);
						*/
					} break;
					default: {}
				}
			}

		} else if (m->get_button_mask() & BUTTON_MASK_RIGHT) {

			if (nav_scheme == NAVIGATION_MAYA && m->get_alt()) {
				nav_mode = NAVIGATION_ZOOM;
			} else if (freelook_active) {
				nav_mode = NAVIGATION_LOOK;
			}

		} else if (m->get_button_mask() & BUTTON_MASK_MIDDLE) {

			if (nav_scheme == NAVIGATION_GODOT) {

				int mod = _get_key_modifier(m);

				if (mod == _get_key_modifier_setting("editors/3d/pan_modifier"))
					nav_mode = NAVIGATION_PAN;
				else if (mod == _get_key_modifier_setting("editors/3d/zoom_modifier"))
					nav_mode = NAVIGATION_ZOOM;
				else if (mod == _get_key_modifier_setting("editors/3d/orbit_modifier"))
					nav_mode = NAVIGATION_ORBIT;

			} else if (nav_scheme == NAVIGATION_MAYA) {
				if (m->get_alt())
					nav_mode = NAVIGATION_PAN;
			}

		} else if (EditorSettings::get_singleton()->get("editors/3d/emulate_3_button_mouse")) {
			// Handle trackpad (no external mouse) use case
			int mod = _get_key_modifier(m);

			if (mod) {
				if (mod == _get_key_modifier_setting("editors/3d/pan_modifier"))
					nav_mode = NAVIGATION_PAN;
				else if (mod == _get_key_modifier_setting("editors/3d/zoom_modifier"))
					nav_mode = NAVIGATION_ZOOM;
				else if (mod == _get_key_modifier_setting("editors/3d/orbit_modifier"))
					nav_mode = NAVIGATION_ORBIT;
			}
		}

		switch (nav_mode) {
			case NAVIGATION_PAN: {

				real_t pan_speed = 1 / 150.0;
				int pan_speed_modifier = 10;
				if (nav_scheme == NAVIGATION_MAYA && m->get_shift())
					pan_speed *= pan_speed_modifier;

				Point2i relative = _get_warped_mouse_motion(m);

				Transform camera_transform;

				camera_transform.translate(cursor.pos);
				camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor.x_rot);
				camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor.y_rot);
				Vector3 translation(-relative.x * pan_speed, relative.y * pan_speed, 0);
				translation *= cursor.distance / DISTANCE_DEFAULT;
				camera_transform.translate(translation);
				cursor.pos = camera_transform.origin;

			} break;

			case NAVIGATION_ZOOM: {
				real_t zoom_speed = 1 / 80.0;
				int zoom_speed_modifier = 10;
				if (nav_scheme == NAVIGATION_MAYA && m->get_shift())
					zoom_speed *= zoom_speed_modifier;

				NavigationZoomStyle zoom_style = (NavigationZoomStyle)EditorSettings::get_singleton()->get("editors/3d/zoom_style").operator int();
				if (zoom_style == NAVIGATION_ZOOM_HORIZONTAL) {
					if (m->get_relative().x > 0)
						scale_cursor_distance(1 - m->get_relative().x * zoom_speed);
					else if (m->get_relative().x < 0)
						scale_cursor_distance(1.0 / (1 + m->get_relative().x * zoom_speed));
				} else {
					if (m->get_relative().y > 0)
						scale_cursor_distance(1 + m->get_relative().y * zoom_speed);
					else if (m->get_relative().y < 0)
						scale_cursor_distance(1.0 / (1 - m->get_relative().y * zoom_speed));
				}

			} break;

			case NAVIGATION_ORBIT: {
				Point2i relative = _get_warped_mouse_motion(m);

				real_t degrees_per_pixel = EditorSettings::get_singleton()->get("editors/3d/orbit_sensitivity");
				real_t radians_per_pixel = Math::deg2rad(degrees_per_pixel);

				cursor.x_rot += relative.y * radians_per_pixel;
				cursor.y_rot += relative.x * radians_per_pixel;
				if (cursor.x_rot > Math_PI / 2.0)
					cursor.x_rot = Math_PI / 2.0;
				if (cursor.x_rot < -Math_PI / 2.0)
					cursor.x_rot = -Math_PI / 2.0;
				name = "";
				_update_name();
			} break;

			case NAVIGATION_LOOK: {
				// Freelook only works properly in perspective.
				// It technically works too in ortho, but it's awful for a user due to fov being near zero
				if (!orthogonal) {
					Point2i relative = _get_warped_mouse_motion(m);

					real_t degrees_per_pixel = EditorSettings::get_singleton()->get("editors/3d/orbit_sensitivity");
					real_t radians_per_pixel = Math::deg2rad(degrees_per_pixel);

					cursor.x_rot += relative.y * radians_per_pixel;
					cursor.y_rot += relative.x * radians_per_pixel;
					if (cursor.x_rot > Math_PI / 2.0)
						cursor.x_rot = Math_PI / 2.0;
					if (cursor.x_rot < -Math_PI / 2.0)
						cursor.x_rot = -Math_PI / 2.0;

					// Look is like Orbit, except the cursor translates, not the camera
					Transform camera_transform = to_camera_transform(cursor);
					Vector3 pos = camera_transform.xform(Vector3(0, 0, 0));
					Vector3 diff = camera->get_translation() - pos;
					cursor.pos += diff;
					freelook_target_position += diff;

					name = "";
					_update_name();
				}

			} break;

			default: {}
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (!k->is_pressed())
			return;

		if (ED_IS_SHORTCUT("spatial_editor/snap", p_event)) {
			if (_edit.mode != TRANSFORM_NONE) {
				_edit.snap = true;
			}
		}
		if (ED_IS_SHORTCUT("spatial_editor/bottom_view", p_event)) {
			cursor.y_rot = 0;
			cursor.x_rot = -Math_PI / 2.0;
			set_message(TTR("Bottom View."), 2);
			name = TTR("Bottom");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/top_view", p_event)) {
			cursor.y_rot = 0;
			cursor.x_rot = Math_PI / 2.0;
			set_message(TTR("Top View."), 2);
			name = TTR("Top");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/rear_view", p_event)) {
			cursor.x_rot = 0;
			cursor.y_rot = Math_PI;
			set_message(TTR("Rear View."), 2);
			name = TTR("Rear");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/front_view", p_event)) {
			cursor.x_rot = 0;
			cursor.y_rot = 0;
			set_message(TTR("Front View."), 2);
			name = TTR("Front");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/left_view", p_event)) {
			cursor.x_rot = 0;
			cursor.y_rot = Math_PI / 2.0;
			set_message(TTR("Left View."), 2);
			name = TTR("Left");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/right_view", p_event)) {
			cursor.x_rot = 0;
			cursor.y_rot = -Math_PI / 2.0;
			set_message(TTR("Right View."), 2);
			name = TTR("Right");
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/switch_perspective_orthogonal", p_event)) {
			_menu_option(orthogonal ? VIEW_PERSPECTIVE : VIEW_ORTHOGONAL);
			_update_name();
		}
		if (ED_IS_SHORTCUT("spatial_editor/insert_anim_key", p_event)) {
			if (!get_selected_count() || _edit.mode != TRANSFORM_NONE)
				return;

			if (!AnimationPlayerEditor::singleton->get_key_editor()->has_keying()) {
				set_message(TTR("Keying is disabled (no key inserted)."));
				return;
			}

			List<Node *> &selection = editor_selection->get_selected_node_list();

			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				Spatial *sp = Object::cast_to<Spatial>(E->get());
				if (!sp)
					continue;

				emit_signal("transform_key_request", sp, "", sp->get_transform());
			}

			set_message(TTR("Animation Key Inserted."));
		}

		if (k->get_scancode() == KEY_SPACE) {
			if (!k->is_pressed()) emit_signal("toggle_maximize_view", this);
		}
	}
}

void SpatialEditorViewport::scale_cursor_distance(real_t scale) {

	// Prevents zero distance which would short-circuit any scaling
	if (cursor.distance < ZOOM_MIN_DISTANCE)
		cursor.distance = ZOOM_MIN_DISTANCE;

	real_t prev_distance = cursor.distance;
	cursor.distance *= scale;

	if (cursor.distance < ZOOM_MIN_DISTANCE)
		cursor.distance = ZOOM_MIN_DISTANCE;

	if (is_freelook_active()) {
		// In freelook mode, cursor reference is reversed so it needs to be adjusted
		Vector3 forward = camera->get_transform().basis.xform(Vector3(0, 0, -1));
		cursor.pos += (cursor.distance - prev_distance) * forward;
	}

	zoom_indicator_delay = ZOOM_INDICATOR_DELAY_S;
	surface->update();
}

Point2i SpatialEditorViewport::_get_warped_mouse_motion(const Ref<InputEventMouseMotion> &p_ev_mouse_motion) const {
	Point2i relative;
	if (bool(EDITOR_DEF("editors/3d/warped_mouse_panning", false))) {
		relative = Input::get_singleton()->warp_mouse_motion(p_ev_mouse_motion, surface->get_global_rect());
	} else {
		relative = p_ev_mouse_motion->get_relative();
	}
	return relative;
}

void SpatialEditorViewport::_update_freelook(real_t delta) {

	if (!is_freelook_active()) {
		freelook_target_position = cursor.pos;
		return;
	}

	Vector3 forward = camera->get_transform().basis.xform(Vector3(0, 0, -1));
	Vector3 right = camera->get_transform().basis.xform(Vector3(1, 0, 0));
	Vector3 up = camera->get_transform().basis.xform(Vector3(0, 1, 0));

	int key_left = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_left")->get_shortcut().ptr())->get_scancode();
	int key_right = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_right")->get_shortcut().ptr())->get_scancode();
	int key_forward = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_forward")->get_shortcut().ptr())->get_scancode();
	int key_backwards = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_backwards")->get_shortcut().ptr())->get_scancode();
	int key_up = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_up")->get_shortcut().ptr())->get_scancode();
	int key_down = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_down")->get_shortcut().ptr())->get_scancode();
	int key_speed_modifier = Object::cast_to<InputEventKey>(ED_GET_SHORTCUT("spatial_editor/freelook_speed_modifier")->get_shortcut().ptr())->get_scancode();

	Vector3 direction;
	bool speed_modifier = false;

	const Input &input = *Input::get_singleton();

	if (input.is_key_pressed(key_left)) {
		direction -= right;
	}
	if (input.is_key_pressed(key_right)) {
		direction += right;
	}
	if (input.is_key_pressed(key_forward)) {
		direction += forward;
	}
	if (input.is_key_pressed(key_backwards)) {
		direction -= forward;
	}
	if (input.is_key_pressed(key_up)) {
		direction += up;
	}
	if (input.is_key_pressed(key_down)) {
		direction -= up;
	}
	if (input.is_key_pressed(key_speed_modifier)) {
		speed_modifier = true;
	}

	real_t inertia = EDITOR_DEF("editors/3d/freelook_inertia", 0.2);
	inertia = MAX(0, inertia);
	const real_t base_speed = EDITOR_DEF("editors/3d/freelook_base_speed", 0.5);
	const real_t modifier_speed_factor = EDITOR_DEF("editors/3d/freelook_modifier_speed_factor", 5);

	real_t speed = base_speed * cursor.distance;
	if (speed_modifier)
		speed *= modifier_speed_factor;

	// Higher inertia should increase "lag" (lerp with factor between 0 and 1)
	// Inertia of zero should produce instant movement (lerp with factor of 1) in this case it returns a really high value and gets clamped to 1.

	freelook_target_position += direction * speed;
	real_t factor = (1.0 / (inertia + 0.001)) * delta;
	cursor.pos = cursor.pos.linear_interpolate(freelook_target_position, CLAMP(factor, 0, 1));
}

void SpatialEditorViewport::set_message(String p_message, float p_time) {

	message = p_message;
	message_time = p_time;
}

void SpatialEditorViewport::_notification(int p_what) {

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {

		bool visible = is_visible_in_tree();

		set_process(visible);

		if (visible)
			_update_camera(0);

		call_deferred("update_transform_gizmo_view");
	}

	if (p_what == NOTIFICATION_RESIZED) {

		call_deferred("update_transform_gizmo_view");
	}

	if (p_what == NOTIFICATION_PROCESS) {

		//force editr camera
		/*
		current_camera=get_root_node()->get_current_camera();
		if (current_camera!=camera) {


		}
		*/

		real_t delta = get_tree()->get_idle_process_time();

		if (zoom_indicator_delay > 0) {
			zoom_indicator_delay -= delta;
			if (zoom_indicator_delay <= 0) {
				surface->update();
			}
		}

		_update_freelook(delta);

		_update_camera(get_process_delta_time());

		Map<Node *, Object *> &selection = editor_selection->get_selection();

		bool changed = false;
		bool exist = false;

		for (Map<Node *, Object *>::Element *E = selection.front(); E; E = E->next()) {

			Spatial *sp = Object::cast_to<Spatial>(E->key());
			if (!sp)
				continue;

			SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
			if (!se)
				continue;

			VisualInstance *vi = Object::cast_to<VisualInstance>(sp);

			if (se->aabb.has_no_surface()) {

				se->aabb = vi ? vi->get_aabb() : Rect3(Vector3(-0.2, -0.2, -0.2), Vector3(0.4, 0.4, 0.4));
			}

			Transform t = sp->get_global_transform();
			t.translate(se->aabb.position);

			// apply AABB scaling before item's global transform
			Basis aabb_s;
			aabb_s.scale(se->aabb.size);
			t.basis = t.basis * aabb_s;

			exist = true;
			if (se->last_xform == t)
				continue;
			changed = true;
			se->last_xform = t;
			VisualServer::get_singleton()->instance_set_transform(se->sbox_instance, t);
		}

		if (changed || (spatial_editor->is_gizmo_visible() && !exist)) {
			spatial_editor->update_transform_gizmo();
		}

		if (message_time > 0) {

			if (message != last_message) {
				surface->update();
				last_message = message;
			}

			message_time -= get_fixed_process_delta_time();
			if (message_time < 0)
				surface->update();
		}

		//update shadow atlas if changed

		int shadowmap_size = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/size");
		int atlas_q0 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_0_subdiv");
		int atlas_q1 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_1_subdiv");
		int atlas_q2 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_2_subdiv");
		int atlas_q3 = ProjectSettings::get_singleton()->get("rendering/quality/shadow_atlas/quadrant_3_subdiv");

		viewport->set_shadow_atlas_size(shadowmap_size);
		viewport->set_shadow_atlas_quadrant_subdiv(0, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q0));
		viewport->set_shadow_atlas_quadrant_subdiv(1, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q1));
		viewport->set_shadow_atlas_quadrant_subdiv(2, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q2));
		viewport->set_shadow_atlas_quadrant_subdiv(3, Viewport::ShadowAtlasQuadrantSubdiv(atlas_q3));

		//update msaa if changed

		int msaa_mode = ProjectSettings::get_singleton()->get("rendering/quality/filters/msaa");
		viewport->set_msaa(Viewport::MSAA(msaa_mode));

		bool hdr = ProjectSettings::get_singleton()->get("rendering/quality/depth/hdr");
		viewport->set_hdr(hdr);

		bool show_info = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(VIEW_INFORMATION));
		if (show_info != info->is_visible()) {
			if (show_info)
				info->show();
			else
				info->hide();
		}

		if (show_info) {

			String text;
			text += TTR("Objects Drawn") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_OBJECTS_IN_FRAME)) + "\n";
			text += TTR("Material Changes") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_MATERIAL_CHANGES_IN_FRAME)) + "\n";
			text += TTR("Shader Changes") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_SHADER_CHANGES_IN_FRAME)) + "\n";
			text += TTR("Surface Changes") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_SURFACE_CHANGES_IN_FRAME)) + "\n";
			text += TTR("Draw Calls") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_DRAW_CALLS_IN_FRAME)) + "\n";
			text += TTR("Vertices") + ": " + itos(viewport->get_render_info(Viewport::RENDER_INFO_VERTICES_IN_FRAME));

			if (info_label->get_text() != text || surface->get_size() != prev_size) {
				info_label->set_text(text);
				Size2 ms = info->get_minimum_size();
				info->set_position(surface->get_size() - ms - Vector2(20, 20) * EDSCALE);
			}
		}

		prev_size = surface->get_size();
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		surface->connect("draw", this, "_draw");
		surface->connect("gui_input", this, "_sinput");
		surface->connect("mouse_entered", this, "_smouseenter");
		surface->connect("mouse_exited", this, "_smouseexit");
		info->add_style_override("panel", get_stylebox("panel", "Panel"));
		preview_camera->set_icon(get_icon("Camera", "EditorIcons"));
		_init_gizmo_instance(index);
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {

		_finish_gizmo_instances();
	}

	if (p_what == NOTIFICATION_MOUSE_ENTER) {
	}

	if (p_what == NOTIFICATION_DRAW) {
	}
}

// TODO That should be part of the drawing API...
static void stroke_rect(CanvasItem *ci, Rect2 rect, Color color, real_t width = 1.0) {

	// a---b
	// |   |
	// c---d
	Vector2 a(rect.position);
	Vector2 b(rect.position.x + rect.size.x, rect.position.y);
	Vector2 c(rect.position.x, rect.position.y + rect.size.y);
	Vector2 d(rect.position + rect.size);

	ci->draw_line(a, b, color, width);
	ci->draw_line(b, d, color, width);
	ci->draw_line(d, c, color, width);
	ci->draw_line(c, a, color, width);
}

void SpatialEditorViewport::_draw() {

	if (surface->has_focus()) {
		Size2 size = surface->get_size();
		Rect2 r = Rect2(Point2(), size);
		get_stylebox("Focus", "EditorStyles")->draw(surface->get_canvas_item(), r);
	}

	RID ci = surface->get_canvas_item();

	if (cursor.region_select) {

		VisualServer::get_singleton()->canvas_item_add_rect(ci, Rect2(cursor.region_begin, cursor.region_end - cursor.region_begin), Color(0.7, 0.7, 1.0, 0.3));
	}

	if (message_time > 0) {
		Ref<Font> font = get_font("font", "Label");
		Point2 msgpos = Point2(5, get_size().y - 20);
		font->draw(ci, msgpos + Point2(1, 1), message, Color(0, 0, 0, 0.8));
		font->draw(ci, msgpos + Point2(-1, -1), message, Color(0, 0, 0, 0.8));
		font->draw(ci, msgpos, message, Color(1, 1, 1, 1));
	}

	if (_edit.mode == TRANSFORM_ROTATE) {

		Point2 center = _point_to_screen(_edit.center);
		VisualServer::get_singleton()->canvas_item_add_line(ci, _edit.mouse_pos, center, Color(0.4, 0.7, 1.0, 0.8));
	}

	if (previewing) {

		Size2 ss = Size2(ProjectSettings::get_singleton()->get("display/window/size/width"), ProjectSettings::get_singleton()->get("display/window/size/height"));
		float aspect = ss.aspect();
		Size2 s = get_size();

		Rect2 draw_rect;

		switch (previewing->get_keep_aspect_mode()) {
			case Camera::KEEP_WIDTH: {

				draw_rect.size = Size2(s.width, s.width / aspect);
				draw_rect.position.x = 0;
				draw_rect.position.y = (s.height - draw_rect.size.y) * 0.5;

			} break;
			case Camera::KEEP_HEIGHT: {

				draw_rect.size = Size2(s.height * aspect, s.height);
				draw_rect.position.y = 0;
				draw_rect.position.x = (s.width - draw_rect.size.x) * 0.5;

			} break;
		}

		draw_rect = Rect2(Vector2(), s).clip(draw_rect);

		stroke_rect(surface, draw_rect, Color(0.6, 0.6, 0.1, 0.5), 2.0);

	} else {

		if (zoom_indicator_delay > 0.0) {
			// Show indicative zoom factor

			real_t min_distance = ZOOM_MIN_DISTANCE; // TODO Why not pick znear to limit zoom?
			real_t max_distance = camera->get_zfar();
			real_t scale_length = (max_distance - min_distance);

			if (Math::abs(scale_length) > CMP_EPSILON) {
				real_t logscale_t = 1.0 - Math::log(1 + cursor.distance - min_distance) / Math::log(1 + scale_length);

				// There is no real maximum distance so that factor can become negative,
				// Let's make it look asymptotic instead (will decrease slower and slower).
				if (logscale_t < 0.25)
					logscale_t = 0.25 * Math::exp(4.0 * logscale_t - 1.0);

				Vector2 surface_size = surface->get_size();
				real_t h = surface_size.y / 2.0;
				real_t y = (surface_size.y - h) / 2.0;

				Rect2 r(10, y, 6, h);
				real_t sy = r.size.y * logscale_t;

				surface->draw_rect(r, Color(1, 1, 1, 0.2));
				surface->draw_rect(Rect2(r.position.x, r.position.y + r.size.y - sy, r.size.x, sy), Color(1, 1, 1, 0.6));
				stroke_rect(surface, r.grow(1), Color(0, 0, 0, 0.7));
			}
		}
	}
}

void SpatialEditorViewport::_menu_option(int p_option) {

	switch (p_option) {

		case VIEW_TOP: {

			cursor.x_rot = Math_PI / 2.0;
			cursor.y_rot = 0;
			name = TTR("Top");
			_update_name();
		} break;
		case VIEW_BOTTOM: {

			cursor.x_rot = -Math_PI / 2.0;
			cursor.y_rot = 0;
			name = TTR("Bottom");
			_update_name();

		} break;
		case VIEW_LEFT: {

			cursor.y_rot = Math_PI / 2.0;
			cursor.x_rot = 0;
			name = TTR("Left");
			_update_name();

		} break;
		case VIEW_RIGHT: {

			cursor.y_rot = -Math_PI / 2.0;
			cursor.x_rot = 0;
			name = TTR("Right");
			_update_name();

		} break;
		case VIEW_FRONT: {

			cursor.y_rot = 0;
			cursor.x_rot = 0;
			name = TTR("Front");
			_update_name();

		} break;
		case VIEW_REAR: {

			cursor.y_rot = Math_PI;
			cursor.x_rot = 0;
			name = TTR("Rear");
			_update_name();

		} break;
		case VIEW_CENTER_TO_ORIGIN: {

			cursor.pos = Vector3(0, 0, 0);

		} break;
		case VIEW_CENTER_TO_SELECTION: {

			focus_selection();

		} break;
		case VIEW_ALIGN_SELECTION_WITH_VIEW: {

			if (!get_selected_count())
				break;

			Transform camera_transform = camera->get_global_transform();

			List<Node *> &selection = editor_selection->get_selected_node_list();

			undo_redo->create_action(TTR("Align with view"));
			for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

				Spatial *sp = Object::cast_to<Spatial>(E->get());
				if (!sp)
					continue;

				SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
				if (!se)
					continue;

				Transform xform = camera_transform;
				xform.scale_basis(sp->get_scale());

				undo_redo->add_do_method(sp, "set_global_transform", xform);
				undo_redo->add_undo_method(sp, "set_global_transform", sp->get_global_transform());
			}
			undo_redo->commit_action();
		} break;
		case VIEW_ENVIRONMENT: {

			int idx = view_menu->get_popup()->get_item_index(VIEW_ENVIRONMENT);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			if (current) {

				camera->set_environment(RES());
			} else {

				camera->set_environment(SpatialEditor::get_singleton()->get_viewport_environment());
			}

			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_PERSPECTIVE: {

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ORTHOGONAL), false);
			orthogonal = false;
			call_deferred("update_transform_gizmo_view");
			_update_name();

		} break;
		case VIEW_ORTHOGONAL: {

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ORTHOGONAL), true);
			orthogonal = true;
			call_deferred("update_transform_gizmo_view");
			_update_name();

		} break;
		case VIEW_AUDIO_LISTENER: {

			int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			viewport->set_as_audio_listener(current);
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_AUDIO_DOPPLER: {

			int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_DOPPLER);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			camera->set_doppler_tracking(current ? Camera::DOPPLER_TRACKING_IDLE_STEP : Camera::DOPPLER_TRACKING_DISABLED);
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_GIZMOS: {

			int idx = view_menu->get_popup()->get_item_index(VIEW_GIZMOS);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			current = !current;
			if (current)
				camera->set_cull_mask(((1 << 20) - 1) | (1 << (GIZMO_BASE_LAYER + index)) | (1 << GIZMO_EDIT_LAYER) | (1 << GIZMO_GRID_LAYER));
			else
				camera->set_cull_mask(((1 << 20) - 1) | (1 << (GIZMO_BASE_LAYER + index)) | (1 << GIZMO_GRID_LAYER));
			view_menu->get_popup()->set_item_checked(idx, current);

		} break;
		case VIEW_INFORMATION: {

			int idx = view_menu->get_popup()->get_item_index(VIEW_INFORMATION);
			bool current = view_menu->get_popup()->is_item_checked(idx);
			view_menu->get_popup()->set_item_checked(idx, !current);

		} break;
		case VIEW_DISPLAY_NORMAL: {

			viewport->set_debug_draw(Viewport::DEBUG_DRAW_DISABLED);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS), false);

		} break;
		case VIEW_DISPLAY_WIREFRAME: {

			viewport->set_debug_draw(Viewport::DEBUG_DRAW_WIREFRAME);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS), false);

		} break;
		case VIEW_DISPLAY_OVERDRAW: {

			viewport->set_debug_draw(Viewport::DEBUG_DRAW_OVERDRAW);
			VisualServer::get_singleton()->scenario_set_debug(get_tree()->get_root()->get_world()->get_scenario(), VisualServer::SCENARIO_DEBUG_OVERDRAW);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS), false);

		} break;
		case VIEW_DISPLAY_SHADELESS: {

			viewport->set_debug_draw(Viewport::DEBUG_DRAW_UNSHADED);
			VisualServer::get_singleton()->scenario_set_debug(get_tree()->get_root()->get_world()->get_scenario(), VisualServer::SCENARIO_DEBUG_SHADELESS);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_WIREFRAME), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_OVERDRAW), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_SHADELESS), true);

		} break;
	}
}

void SpatialEditorViewport::_preview_exited_scene() {

	preview_camera->set_pressed(false);
	_toggle_camera_preview(false);
	view_menu->show();
}

void SpatialEditorViewport::_init_gizmo_instance(int p_idx) {

	uint32_t layer = 1 << (GIZMO_BASE_LAYER + p_idx); //|(1<<GIZMO_GRID_LAYER);

	for (int i = 0; i < 3; i++) {
		move_gizmo_instance[i] = VS::get_singleton()->instance_create();
		VS::get_singleton()->instance_set_base(move_gizmo_instance[i], spatial_editor->get_move_gizmo(i)->get_rid());
		VS::get_singleton()->instance_set_scenario(move_gizmo_instance[i], get_tree()->get_root()->get_world()->get_scenario());
		VS::get_singleton()->instance_set_visible(move_gizmo_instance[i], false);
		//VS::get_singleton()->instance_geometry_set_flag(move_gizmo_instance[i],VS::INSTANCE_FLAG_DEPH_SCALE,true);
		VS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_gizmo_instance[i], VS::SHADOW_CASTING_SETTING_OFF);
		VS::get_singleton()->instance_set_layer_mask(move_gizmo_instance[i], layer);

		move_plane_gizmo_instance[i] = VS::get_singleton()->instance_create();
		VS::get_singleton()->instance_set_base(move_plane_gizmo_instance[i], spatial_editor->get_move_plane_gizmo(i)->get_rid());
		VS::get_singleton()->instance_set_scenario(move_plane_gizmo_instance[i], get_tree()->get_root()->get_world()->get_scenario());
		VS::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], false);
		//VS::get_singleton()->instance_geometry_set_flag(move_plane_gizmo_instance[i],VS::INSTANCE_FLAG_DEPH_SCALE,true);
		VS::get_singleton()->instance_geometry_set_cast_shadows_setting(move_plane_gizmo_instance[i], VS::SHADOW_CASTING_SETTING_OFF);
		VS::get_singleton()->instance_set_layer_mask(move_plane_gizmo_instance[i], layer);

		rotate_gizmo_instance[i] = VS::get_singleton()->instance_create();
		VS::get_singleton()->instance_set_base(rotate_gizmo_instance[i], spatial_editor->get_rotate_gizmo(i)->get_rid());
		VS::get_singleton()->instance_set_scenario(rotate_gizmo_instance[i], get_tree()->get_root()->get_world()->get_scenario());
		VS::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], false);
		//VS::get_singleton()->instance_geometry_set_flag(rotate_gizmo_instance[i],VS::INSTANCE_FLAG_DEPH_SCALE,true);
		VS::get_singleton()->instance_geometry_set_cast_shadows_setting(rotate_gizmo_instance[i], VS::SHADOW_CASTING_SETTING_OFF);
		VS::get_singleton()->instance_set_layer_mask(rotate_gizmo_instance[i], layer);
	}
}

void SpatialEditorViewport::_finish_gizmo_instances() {

	for (int i = 0; i < 3; i++) {
		VS::get_singleton()->free(move_gizmo_instance[i]);
		VS::get_singleton()->free(move_plane_gizmo_instance[i]);
		VS::get_singleton()->free(rotate_gizmo_instance[i]);
	}
}
void SpatialEditorViewport::_toggle_camera_preview(bool p_activate) {

	ERR_FAIL_COND(p_activate && !preview);
	ERR_FAIL_COND(!p_activate && !previewing);

	if (!p_activate) {

		previewing->disconnect("tree_exited", this, "_preview_exited_scene");
		previewing = NULL;
		VS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), camera->get_camera()); //restore
		if (!preview)
			preview_camera->hide();
		view_menu->show();
		surface->update();

	} else {

		previewing = preview;
		previewing->connect("tree_exited", this, "_preview_exited_scene");
		VS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), preview->get_camera()); //replace
		view_menu->hide();
		surface->update();
	}
}

void SpatialEditorViewport::_selection_result_pressed(int p_result) {

	if (selection_results.size() <= p_result)
		return;

	clicked = selection_results[p_result].item->get_instance_id();

	if (clicked) {
		_select_clicked(clicked_wants_append, true);
		clicked = 0;
	}
}

void SpatialEditorViewport::_selection_menu_hide() {

	selection_results.clear();
	selection_menu->clear();
	selection_menu->set_size(Vector2(0, 0));
}

void SpatialEditorViewport::set_can_preview(Camera *p_preview) {

	preview = p_preview;

	if (!preview_camera->is_pressed()) {

		if (p_preview) {
			preview_camera->show();
		} else {
			preview_camera->hide();
		}
	}
}

void SpatialEditorViewport::update_transform_gizmo_view() {

	if (!is_visible_in_tree())
		return;

	Transform xform = spatial_editor->get_gizmo_transform();

	Transform camera_xform = camera->get_transform();
	Vector3 camz = -camera_xform.get_basis().get_axis(2).normalized();
	Vector3 camy = -camera_xform.get_basis().get_axis(1).normalized();
	Plane p(camera_xform.origin, camz);
	float gizmo_d = Math::abs(p.distance_to(xform.origin));
	float d0 = camera->unproject_position(camera_xform.origin + camz * gizmo_d).y;
	float d1 = camera->unproject_position(camera_xform.origin + camz * gizmo_d + camy).y;
	float dd = Math::abs(d0 - d1);
	if (dd == 0)
		dd = 0.0001;

	float gsize = EditorSettings::get_singleton()->get("editors/3d/manipulator_gizmo_size");
	gizmo_scale = (gsize / Math::abs(dd));
	Vector3 scale = Vector3(1, 1, 1) * gizmo_scale;

	xform.basis.scale(scale);

	//xform.basis.scale(GIZMO_SCALE_DEFAULT*Vector3(1,1,1));

	for (int i = 0; i < 3; i++) {
		VisualServer::get_singleton()->instance_set_transform(move_gizmo_instance[i], xform);
		VisualServer::get_singleton()->instance_set_visible(move_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_MOVE));
		VisualServer::get_singleton()->instance_set_transform(move_plane_gizmo_instance[i], xform);
		VisualServer::get_singleton()->instance_set_visible(move_plane_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_MOVE));
		VisualServer::get_singleton()->instance_set_transform(rotate_gizmo_instance[i], xform);
		VisualServer::get_singleton()->instance_set_visible(rotate_gizmo_instance[i], spatial_editor->is_gizmo_visible() && (spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_SELECT || spatial_editor->get_tool_mode() == SpatialEditor::TOOL_MODE_ROTATE));
	}
}

void SpatialEditorViewport::set_state(const Dictionary &p_state) {

	cursor.pos = p_state["pos"];
	cursor.x_rot = p_state["x_rot"];
	cursor.y_rot = p_state["y_rot"];
	cursor.distance = p_state["distance"];
	bool env = p_state["use_environment"];
	bool orth = p_state["use_orthogonal"];
	if (orth)
		_menu_option(VIEW_ORTHOGONAL);
	else
		_menu_option(VIEW_PERSPECTIVE);
	if (env != camera->get_environment().is_valid())
		_menu_option(VIEW_ENVIRONMENT);
	if (p_state.has("listener")) {
		bool listener = p_state["listener"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER);
		viewport->set_as_audio_listener(listener);
		view_menu->get_popup()->set_item_checked(idx, listener);
	}
	if (p_state.has("doppler")) {
		bool doppler = p_state["doppler"];

		int idx = view_menu->get_popup()->get_item_index(VIEW_AUDIO_DOPPLER);
		camera->set_doppler_tracking(doppler ? Camera::DOPPLER_TRACKING_IDLE_STEP : Camera::DOPPLER_TRACKING_DISABLED);
		view_menu->get_popup()->set_item_checked(idx, doppler);
	}

	if (p_state.has("previewing")) {
		Node *pv = EditorNode::get_singleton()->get_edited_scene()->get_node(p_state["previewing"]);
		if (Object::cast_to<Camera>(pv)) {
			previewing = Object::cast_to<Camera>(pv);
			previewing->connect("tree_exited", this, "_preview_exited_scene");
			VS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), previewing->get_camera()); //replace
			view_menu->hide();
			surface->update();
			preview_camera->set_pressed(true);
			preview_camera->show();
		}
	}
}

Dictionary SpatialEditorViewport::get_state() const {

	Dictionary d;
	d["pos"] = cursor.pos;
	d["x_rot"] = cursor.x_rot;
	d["y_rot"] = cursor.y_rot;
	d["distance"] = cursor.distance;
	d["use_environment"] = camera->get_environment().is_valid();
	d["use_orthogonal"] = camera->get_projection() == Camera::PROJECTION_ORTHOGONAL;
	d["listener"] = viewport->is_audio_listener();
	if (previewing) {
		d["previewing"] = EditorNode::get_singleton()->get_edited_scene()->get_path_to(previewing);
	}

	return d;
}

void SpatialEditorViewport::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_draw"), &SpatialEditorViewport::_draw);
	ClassDB::bind_method(D_METHOD("_smouseenter"), &SpatialEditorViewport::_smouseenter);
	ClassDB::bind_method(D_METHOD("_smouseexit"), &SpatialEditorViewport::_smouseexit);
	ClassDB::bind_method(D_METHOD("_sinput"), &SpatialEditorViewport::_sinput);
	ClassDB::bind_method(D_METHOD("_menu_option"), &SpatialEditorViewport::_menu_option);
	ClassDB::bind_method(D_METHOD("_toggle_camera_preview"), &SpatialEditorViewport::_toggle_camera_preview);
	ClassDB::bind_method(D_METHOD("_preview_exited_scene"), &SpatialEditorViewport::_preview_exited_scene);
	ClassDB::bind_method(D_METHOD("update_transform_gizmo_view"), &SpatialEditorViewport::update_transform_gizmo_view);
	ClassDB::bind_method(D_METHOD("_selection_result_pressed"), &SpatialEditorViewport::_selection_result_pressed);
	ClassDB::bind_method(D_METHOD("_selection_menu_hide"), &SpatialEditorViewport::_selection_menu_hide);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SpatialEditorViewport::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SpatialEditorViewport::drop_data_fw);

	ADD_SIGNAL(MethodInfo("toggle_maximize_view", PropertyInfo(Variant::OBJECT, "viewport")));
}

void SpatialEditorViewport::reset() {

	orthogonal = false;
	message_time = 0;
	message = "";
	last_message = "";
	name = "";

	cursor.x_rot = 0.5;
	cursor.y_rot = 0.5;
	cursor.distance = 4;
	cursor.region_select = false;
	_update_name();
}

void SpatialEditorViewport::focus_selection() {
	if (!get_selected_count())
		return;

	Vector3 center;
	int count = 0;

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Spatial *sp = Object::cast_to<Spatial>(E->get());
		if (!sp)
			continue;

		SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
		if (!se)
			continue;

		center += sp->get_global_transform().origin;
		count++;
	}

	if (count != 0) {
		center /= float(count);
	}

	cursor.pos = center;
}

void SpatialEditorViewport::assign_pending_data_pointers(Spatial *p_preview_node, Rect3 *p_preview_bounds, AcceptDialog *p_accept) {
	preview_node = p_preview_node;
	preview_bounds = p_preview_bounds;
	accept = p_accept;
}

Vector3 SpatialEditorViewport::_get_instance_position(const Point2 &p_pos) const {
	const float MAX_DISTANCE = 10;

	Vector3 world_ray = _get_ray(p_pos);
	Vector3 world_pos = _get_ray_pos(p_pos);

	Vector<ObjectID> instances = VisualServer::get_singleton()->instances_cull_ray(world_pos, world_ray, get_tree()->get_root()->get_world()->get_scenario());
	Set<Ref<SpatialEditorGizmo> > found_gizmos;

	float closest_dist = MAX_DISTANCE;

	Vector3 point = world_pos + world_ray * MAX_DISTANCE;
	Vector3 normal = Vector3(0.0, 0.0, 0.0);

	for (int i = 0; i < instances.size(); i++) {

		MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(ObjectDB::get_instance(instances[i]));

		if (!mesh_instance)
			continue;

		Ref<SpatialEditorGizmo> seg = mesh_instance->get_gizmo();

		if ((!seg.is_valid()) || found_gizmos.has(seg)) {
			continue;
		}

		found_gizmos.insert(seg);

		Vector3 hit_point;
		Vector3 hit_normal;
		bool inters = seg->intersect_ray(camera, p_pos, hit_point, hit_normal, NULL, false);

		if (!inters)
			continue;

		float dist = world_pos.distance_to(hit_point);

		if (dist < 0)
			continue;

		if (dist < closest_dist) {
			closest_dist = dist;
			point = hit_point;
			normal = hit_normal;
		}
	}
	Vector3 center = preview_bounds->get_size() * 0.5;
	return point + (center * normal);
}

Rect3 SpatialEditorViewport::_calculate_spatial_bounds(const Spatial *p_parent, const Rect3 p_bounds) {
	Rect3 bounds = p_bounds;
	for (int i = 0; i < p_parent->get_child_count(); i++) {
		Spatial *child = Object::cast_to<Spatial>(p_parent->get_child(i));
		if (child) {
			MeshInstance *mesh_instance = Object::cast_to<MeshInstance>(child);
			if (mesh_instance) {
				Rect3 mesh_instance_bounds = mesh_instance->get_aabb();
				mesh_instance_bounds.position += mesh_instance->get_global_transform().origin - p_parent->get_global_transform().origin;
				bounds.merge_with(mesh_instance_bounds);
			}
			bounds = _calculate_spatial_bounds(child, bounds);
		}
	}
	return bounds;
}

void SpatialEditorViewport::_create_preview(const Vector<String> &files) const {
	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		RES res = ResourceLoader::load(path);
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (scene != NULL) {
			if (scene.is_valid()) {
				Node *instance = scene->instance();
				if (instance) {
					preview_node->add_child(instance);
				}
			}
			editor->get_scene_root()->add_child(preview_node);
		}
	}
	*preview_bounds = _calculate_spatial_bounds(preview_node, Rect3());
}

void SpatialEditorViewport::_remove_preview() {
	if (preview_node->get_parent()) {
		for (int i = preview_node->get_child_count() - 1; i >= 0; i--) {
			Node *node = preview_node->get_child(i);
			node->queue_delete();
			preview_node->remove_child(node);
		}
		editor->get_scene_root()->remove_child(preview_node);
	}
}

bool SpatialEditorViewport::_cyclical_dependency_exists(const String &p_target_scene_path, Node *p_desired_node) {
	if (p_desired_node->get_filename() == p_target_scene_path) {
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

bool SpatialEditorViewport::_create_instance(Node *parent, String &path, const Point2 &p_point) {
	Ref<PackedScene> sdata = ResourceLoader::load(path);
	if (!sdata.is_valid()) { // invalid scene
		return false;
	}

	Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
	if (!instanced_scene) { // error on instancing
		return false;
	}

	if (editor->get_edited_scene()->get_filename() != "") { // cyclical instancing
		if (_cyclical_dependency_exists(editor->get_edited_scene()->get_filename(), instanced_scene)) {
			memdelete(instanced_scene);
			return false;
		}
	}

	instanced_scene->set_filename(ProjectSettings::get_singleton()->localize_path(path));

	editor_data->get_undo_redo().add_do_method(parent, "add_child", instanced_scene);
	editor_data->get_undo_redo().add_do_method(instanced_scene, "set_owner", editor->get_edited_scene());
	editor_data->get_undo_redo().add_do_reference(instanced_scene);
	editor_data->get_undo_redo().add_undo_method(parent, "remove_child", instanced_scene);

	String new_name = parent->validate_child_name(instanced_scene);
	ScriptEditorDebugger *sed = ScriptEditor::get_singleton()->get_debugger();
	editor_data->get_undo_redo().add_do_method(sed, "live_debug_instance_node", editor->get_edited_scene()->get_path_to(parent), path, new_name);
	editor_data->get_undo_redo().add_undo_method(sed, "live_debug_remove_node", NodePath(String(editor->get_edited_scene()->get_path_to(parent)) + "/" + new_name));

	Transform global_transform;
	Spatial *parent_spatial = Object::cast_to<Spatial>(parent);
	if (parent_spatial)
		global_transform = parent_spatial->get_global_transform();

	global_transform.origin = _get_instance_position(p_point);

	editor_data->get_undo_redo().add_do_method(instanced_scene, "set_global_transform", global_transform);

	return true;
}

void SpatialEditorViewport::_perform_drop_data() {
	_remove_preview();

	Vector<String> error_files;

	editor_data->get_undo_redo().create_action(TTR("Create Node"));

	for (int i = 0; i < selected_files.size(); i++) {
		String path = selected_files[i];
		RES res = ResourceLoader::load(path);
		if (res.is_null()) {
			continue;
		}
		Ref<PackedScene> scene = Ref<PackedScene>(Object::cast_to<PackedScene>(*res));
		if (scene != NULL) {
			bool success = _create_instance(target_node, path, drop_pos);
			if (!success) {
				error_files.push_back(path);
			}
		}
	}

	editor_data->get_undo_redo().commit_action();

	if (error_files.size() > 0) {
		String files_str;
		for (int i = 0; i < error_files.size(); i++) {
			files_str += error_files[i].get_file().get_basename() + ",";
		}
		files_str = files_str.substr(0, files_str.length() - 1);
		accept->get_ok()->set_text(TTR("Ugh"));
		accept->set_text(vformat(TTR("Error instancing scene from %s"), files_str.c_str()));
		accept->popup_centered_minsize();
	}
}

bool SpatialEditorViewport::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	bool can_instance = false;

	if (!preview_node->is_inside_tree()) {
		Dictionary d = p_data;
		if (d.has("type") && (String(d["type"]) == "files")) {
			Vector<String> files = d["files"];

			List<String> scene_extensions;
			ResourceLoader::get_recognized_extensions_for_type("PackedScene", &scene_extensions);

			for (int i = 0; i < files.size(); i++) {
				if (scene_extensions.find(files[i].get_extension())) {
					RES res = ResourceLoader::load(files[i]);
					if (res.is_null()) {
						continue;
					}

					String type = res->get_class();
					if (type == "PackedScene") {
						Ref<PackedScene> sdata = ResourceLoader::load(files[i]);
						Node *instanced_scene = sdata->instance(PackedScene::GEN_EDIT_STATE_INSTANCE);
						if (!instanced_scene) {
							continue;
						}
						memdelete(instanced_scene);
					}
					can_instance = true;
					break;
				}
			}
			if (can_instance) {
				_create_preview(files);
			}
		}
	} else {
		can_instance = true;
	}

	if (can_instance) {
		Transform global_transform = Transform(Basis(), _get_instance_position(p_point));
		preview_node->set_global_transform(global_transform);
	}

	return can_instance;
}

void SpatialEditorViewport::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	bool is_shift = Input::get_singleton()->is_key_pressed(KEY_SHIFT);

	selected_files.clear();
	Dictionary d = p_data;
	if (d.has("type") && String(d["type"]) == "files") {
		selected_files = d["files"];
	}

	List<Node *> list = editor->get_editor_selection()->get_selected_node_list();
	if (list.size() == 0) {
		Node *root_node = editor->get_edited_scene();
		if (root_node) {
			list.push_back(root_node);
		} else {
			accept->get_ok()->set_text(TTR("OK :("));
			accept->set_text(TTR("No parent to instance a child at."));
			accept->popup_centered_minsize();
			_remove_preview();
			return;
		}
	}
	if (list.size() != 1) {
		accept->get_ok()->set_text(TTR("I see.."));
		accept->set_text(TTR("This operation requires a single selected node."));
		accept->popup_centered_minsize();
		_remove_preview();
		return;
	}

	target_node = list[0];
	if (is_shift && target_node != editor->get_edited_scene()) {
		target_node = target_node->get_parent();
	}
	drop_pos = p_point;

	_perform_drop_data();
}

SpatialEditorViewport::SpatialEditorViewport(SpatialEditor *p_spatial_editor, EditorNode *p_editor, int p_index) {

	_edit.mode = TRANSFORM_NONE;
	_edit.plane = TRANSFORM_VIEW;
	_edit.edited_gizmo = 0;
	_edit.snap = 1;
	_edit.gizmo_handle = 0;

	index = p_index;
	editor = p_editor;
	editor_data = editor->get_scene_tree_dock()->get_editor_data();
	editor_selection = editor->get_editor_selection();
	undo_redo = editor->get_undo_redo();
	clicked = 0;
	clicked_includes_current = false;
	orthogonal = false;
	message_time = 0;
	zoom_indicator_delay = 0.0;

	spatial_editor = p_spatial_editor;
	ViewportContainer *c = memnew(ViewportContainer);
	c->set_stretch(true);
	add_child(c);
	c->set_area_as_parent_rect();
	viewport = memnew(Viewport);
	viewport->set_disable_input(true);

	c->add_child(viewport);
	surface = memnew(Control);
	surface->set_drag_forwarding(this);
	add_child(surface);
	surface->set_area_as_parent_rect();
	surface->set_clip_contents(true);
	camera = memnew(Camera);
	camera->set_disable_gizmo(true);
	camera->set_cull_mask(((1 << 20) - 1) | (1 << (GIZMO_BASE_LAYER + p_index)) | (1 << GIZMO_EDIT_LAYER) | (1 << GIZMO_GRID_LAYER));
	//camera->set_environment(SpatialEditor::get_singleton()->get_viewport_environment());
	viewport->add_child(camera);
	camera->make_current();
	surface->set_focus_mode(FOCUS_ALL);

	view_menu = memnew(MenuButton);
	surface->add_child(view_menu);
	view_menu->set_position(Point2(4, 4) * EDSCALE);
	view_menu->set_self_modulate(Color(1, 1, 1, 0.5));
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/top_view"), VIEW_TOP);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/bottom_view"), VIEW_BOTTOM);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/left_view"), VIEW_LEFT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/right_view"), VIEW_RIGHT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/front_view"), VIEW_FRONT);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/rear_view"), VIEW_REAR);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_item(TTR("Perspective") + " (" + ED_GET_SHORTCUT("spatial_editor/switch_perspective_orthogonal")->get_as_text() + ")", VIEW_PERSPECTIVE);
	view_menu->get_popup()->add_check_item(TTR("Orthogonal") + " (" + ED_GET_SHORTCUT("spatial_editor/switch_perspective_orthogonal")->get_as_text() + ")", VIEW_ORTHOGONAL);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_PERSPECTIVE), true);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_normal", TTR("Display Normal")), VIEW_DISPLAY_NORMAL);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_wireframe", TTR("Display Wireframe")), VIEW_DISPLAY_WIREFRAME);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_overdraw", TTR("Display Overdraw")), VIEW_DISPLAY_OVERDRAW);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_display_unshaded", TTR("Display Unshaded")), VIEW_DISPLAY_SHADELESS);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_DISPLAY_NORMAL), true);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_environment", TTR("View Environment")), VIEW_ENVIRONMENT);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_gizmos", TTR("View Gizmos")), VIEW_GIZMOS);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_information", TTR("View Information")), VIEW_INFORMATION);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_ENVIRONMENT), true);
	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_audio_listener", TTR("Audio Listener")), VIEW_AUDIO_LISTENER);
	view_menu->get_popup()->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_audio_doppler", TTR("Doppler Enable")), VIEW_AUDIO_DOPPLER);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_GIZMOS), true);

	view_menu->get_popup()->add_separator();
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/focus_origin"), VIEW_CENTER_TO_ORIGIN);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/focus_selection"), VIEW_CENTER_TO_SELECTION);
	view_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("spatial_editor/align_selection_with_view"), VIEW_ALIGN_SELECTION_WITH_VIEW);
	view_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	ED_SHORTCUT("spatial_editor/freelook_left", TTR("Freelook Left"), KEY_A);
	ED_SHORTCUT("spatial_editor/freelook_right", TTR("Freelook Right"), KEY_D);
	ED_SHORTCUT("spatial_editor/freelook_forward", TTR("Freelook Forward"), KEY_W);
	ED_SHORTCUT("spatial_editor/freelook_backwards", TTR("Freelook Backwards"), KEY_S);
	ED_SHORTCUT("spatial_editor/freelook_up", TTR("Freelook Up"), KEY_E);
	ED_SHORTCUT("spatial_editor/freelook_down", TTR("Freelook Down"), KEY_Q);
	ED_SHORTCUT("spatial_editor/freelook_speed_modifier", TTR("Freelook Speed Modifier"), KEY_SHIFT);

	preview_camera = memnew(Button);
	preview_camera->set_toggle_mode(true);
	preview_camera->set_anchor_and_margin(MARGIN_LEFT, ANCHOR_END, -90 * EDSCALE);
	preview_camera->set_anchor_and_margin(MARGIN_TOP, ANCHOR_BEGIN, 10 * EDSCALE);
	preview_camera->set_text(TTR("preview"));
	surface->add_child(preview_camera);
	preview_camera->hide();
	preview_camera->connect("toggled", this, "_toggle_camera_preview");
	previewing = NULL;
	gizmo_scale = 1.0;

	preview_node = NULL;

	info = memnew(PanelContainer);
	info->set_self_modulate(Color(1, 1, 1, 0.4));
	surface->add_child(info);
	info_label = memnew(Label);
	info->add_child(info_label);
	info->hide();

	accept = NULL;

	freelook_active = false;

	selection_menu = memnew(PopupMenu);
	add_child(selection_menu);
	selection_menu->set_custom_minimum_size(Size2(100, 0) * EDSCALE);
	selection_menu->connect("id_pressed", this, "_selection_result_pressed");
	selection_menu->connect("popup_hide", this, "_selection_menu_hide");

	if (p_index == 0) {
		view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(VIEW_AUDIO_LISTENER), true);
		viewport->set_as_audio_listener(true);
	}

	name = "";
	_update_name();

	EditorSettings::get_singleton()->connect("settings_changed", this, "update_transform_gizmo_view");
}

//////////////////////////////////////////////////////////////

void SpatialEditorViewportContainer::_gui_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {

		Vector2 size = get_size();

		int h_sep = get_constant("separation", "HSplitContainer");
		int v_sep = get_constant("separation", "VSplitContainer");

		int mid_w = size.width * ratio_h;
		int mid_h = size.height * ratio_v;

		dragging_h = mb->get_position().x > (mid_w - h_sep / 2) && mb->get_position().x < (mid_w + h_sep / 2);
		dragging_v = mb->get_position().y > (mid_h - v_sep / 2) && mb->get_position().y < (mid_h + v_sep / 2);

		drag_begin_pos = mb->get_position();
		drag_begin_ratio.x = ratio_h;
		drag_begin_ratio.y = ratio_v;

		switch (view) {
			case VIEW_USE_1_VIEWPORT: {

				dragging_h = false;
				dragging_v = false;

			} break;
			case VIEW_USE_2_VIEWPORTS: {

				dragging_h = false;

			} break;
			case VIEW_USE_2_VIEWPORTS_ALT: {

				dragging_v = false;

			} break;
			case VIEW_USE_3_VIEWPORTS: {

				if (dragging_v)
					dragging_h = false;
				else
					dragging_v = false;

			} break;
			case VIEW_USE_3_VIEWPORTS_ALT: {

				if (dragging_h)
					dragging_v = false;
				else
					dragging_h = false;
			} break;
			case VIEW_USE_4_VIEWPORTS: {

			} break;
		}
	}

	if (mb.is_valid() && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		dragging_h = false;
		dragging_v = false;
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && (dragging_h || dragging_v)) {

		if (dragging_h) {
			float new_ratio = drag_begin_ratio.x + (mm->get_position().x - drag_begin_pos.x) / get_size().width;
			new_ratio = CLAMP(new_ratio, 40 / get_size().width, (get_size().width - 40) / get_size().width);
			ratio_h = new_ratio;
			queue_sort();
			update();
		}
		if (dragging_v) {
			float new_ratio = drag_begin_ratio.y + (mm->get_position().y - drag_begin_pos.y) / get_size().height;
			new_ratio = CLAMP(new_ratio, 40 / get_size().height, (get_size().height - 40) / get_size().height);
			ratio_v = new_ratio;
			queue_sort();
			update();
		}
	}
}

void SpatialEditorViewportContainer::_notification(int p_what) {

	if (p_what == NOTIFICATION_MOUSE_ENTER || p_what == NOTIFICATION_MOUSE_EXIT) {

		mouseover = (p_what == NOTIFICATION_MOUSE_ENTER);
		update();
	}

	if (p_what == NOTIFICATION_DRAW && mouseover) {

		Ref<Texture> h_grabber = get_icon("grabber", "HSplitContainer");

		Ref<Texture> v_grabber = get_icon("grabber", "VSplitContainer");

		Vector2 size = get_size();

		int h_sep = get_constant("separation", "HSplitContainer");

		int v_sep = get_constant("separation", "VSplitContainer");

		int mid_w = size.width * ratio_h;
		int mid_h = size.height * ratio_v;

		int size_left = mid_w - h_sep / 2;
		int size_bottom = size.height - mid_h - v_sep / 2;

		switch (view) {

			case VIEW_USE_1_VIEWPORT: {

				//nothing to show

			} break;
			case VIEW_USE_2_VIEWPORTS: {

				draw_texture(v_grabber, Vector2((size.width - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));

			} break;
			case VIEW_USE_2_VIEWPORTS_ALT: {

				draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, (size.height - h_grabber->get_height()) / 2));

			} break;
			case VIEW_USE_3_VIEWPORTS: {

				draw_texture(v_grabber, Vector2((size.width - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));
				draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, mid_h + v_grabber->get_height() / 2 + (size_bottom - h_grabber->get_height()) / 2));

			} break;
			case VIEW_USE_3_VIEWPORTS_ALT: {

				draw_texture(v_grabber, Vector2((size_left - v_grabber->get_width()) / 2, mid_h - v_grabber->get_height() / 2));
				draw_texture(h_grabber, Vector2(mid_w - h_grabber->get_width() / 2, (size.height - h_grabber->get_height()) / 2));
			} break;
			case VIEW_USE_4_VIEWPORTS: {

				Vector2 half(mid_w, mid_h);
				draw_texture(v_grabber, half - v_grabber->get_size() / 2.0);
				draw_texture(h_grabber, half - h_grabber->get_size() / 2.0);

			} break;
		}
	}

	if (p_what == NOTIFICATION_SORT_CHILDREN) {

		SpatialEditorViewport *viewports[4];
		int vc = 0;
		for (int i = 0; i < get_child_count(); i++) {
			viewports[vc] = Object::cast_to<SpatialEditorViewport>(get_child(i));
			if (viewports[vc]) {
				vc++;
			}
		}

		ERR_FAIL_COND(vc != 4);

		Size2 size = get_size();

		if (size.x < 10 || size.y < 10) {
			for (int i = 0; i < 4; i++) {
				viewports[i]->hide();
			}
			return;
		}
		int h_sep = get_constant("separation", "HSplitContainer");

		int v_sep = get_constant("separation", "VSplitContainer");

		int mid_w = size.width * ratio_h;
		int mid_h = size.height * ratio_v;

		int size_left = mid_w - h_sep / 2;
		int size_right = size.width - mid_w - h_sep / 2;

		int size_top = mid_h - v_sep / 2;
		int size_bottom = size.height - mid_h - v_sep / 2;

		switch (view) {

			case VIEW_USE_1_VIEWPORT: {

				for (int i = 1; i < 4; i++) {

					viewports[i]->hide();
				}

				fit_child_in_rect(viewports[0], Rect2(Vector2(), size));

			} break;
			case VIEW_USE_2_VIEWPORTS: {

				for (int i = 1; i < 4; i++) {

					if (i == 1 || i == 3)
						viewports[i]->hide();
					else
						viewports[i]->show();
				}

				fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size.width, size_top)));
				fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size.width, size_bottom)));

			} break;
			case VIEW_USE_2_VIEWPORTS_ALT: {

				for (int i = 1; i < 4; i++) {

					if (i == 1 || i == 3)
						viewports[i]->hide();
					else
						viewports[i]->show();
				}
				fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size.height)));
				fit_child_in_rect(viewports[2], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size.height)));

			} break;
			case VIEW_USE_3_VIEWPORTS: {

				for (int i = 1; i < 4; i++) {

					if (i == 1)
						viewports[i]->hide();
					else
						viewports[i]->show();
				}

				fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size.width, size_top)));
				fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
				fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, mid_h + v_sep / 2), Vector2(size_right, size_bottom)));

			} break;
			case VIEW_USE_3_VIEWPORTS_ALT: {

				for (int i = 1; i < 4; i++) {

					if (i == 1)
						viewports[i]->hide();
					else
						viewports[i]->show();
				}

				fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size_top)));
				fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
				fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size.height)));

			} break;
			case VIEW_USE_4_VIEWPORTS: {

				for (int i = 1; i < 4; i++) {

					viewports[i]->show();
				}

				fit_child_in_rect(viewports[0], Rect2(Vector2(), Vector2(size_left, size_top)));
				fit_child_in_rect(viewports[1], Rect2(Vector2(mid_w + h_sep / 2, 0), Vector2(size_right, size_top)));
				fit_child_in_rect(viewports[2], Rect2(Vector2(0, mid_h + v_sep / 2), Vector2(size_left, size_bottom)));
				fit_child_in_rect(viewports[3], Rect2(Vector2(mid_w + h_sep / 2, mid_h + v_sep / 2), Vector2(size_right, size_bottom)));

			} break;
		}
	}
}

void SpatialEditorViewportContainer::set_view(View p_view) {

	view = p_view;
	queue_sort();
}

SpatialEditorViewportContainer::View SpatialEditorViewportContainer::get_view() {

	return view;
}

void SpatialEditorViewportContainer::_bind_methods() {

	ClassDB::bind_method("_gui_input", &SpatialEditorViewportContainer::_gui_input);
}

SpatialEditorViewportContainer::SpatialEditorViewportContainer() {

	view = VIEW_USE_1_VIEWPORT;
	mouseover = false;
	ratio_h = 0.5;
	ratio_v = 0.5;
	dragging_v = false;
	dragging_h = false;
}

///////////////////////////////////////////////////////////////////

SpatialEditor *SpatialEditor::singleton = NULL;

SpatialEditorSelectedItem::~SpatialEditorSelectedItem() {

	if (sbox_instance.is_valid())
		VisualServer::get_singleton()->free(sbox_instance);
}

void SpatialEditor::select_gizmo_highlight_axis(int p_axis) {

	for (int i = 0; i < 3; i++) {

		move_gizmo[i]->surface_set_material(0, i == p_axis ? gizmo_hl : gizmo_color[i]);
		move_plane_gizmo[i]->surface_set_material(0, (i + 6) == p_axis ? gizmo_hl : plane_gizmo_color[i]);
		rotate_gizmo[i]->surface_set_material(0, (i + 3) == p_axis ? gizmo_hl : gizmo_color[i]);
	}
}

void SpatialEditor::update_transform_gizmo() {

	List<Node *> &selection = editor_selection->get_selected_node_list();
	Rect3 center;
	bool first = true;

	Basis gizmo_basis;
	bool local_gizmo_coords = transform_menu->get_popup()->is_item_checked(transform_menu->get_popup()->get_item_index(MENU_TRANSFORM_LOCAL_COORDS));

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Spatial *sp = Object::cast_to<Spatial>(E->get());
		if (!sp)
			continue;

		SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
		if (!se)
			continue;

		Transform xf = se->sp->get_global_transform();
		if (first) {
			center.position = xf.origin;
			first = false;
			if (local_gizmo_coords) {
				gizmo_basis = xf.basis;
				gizmo_basis.orthonormalize();
			}
		} else {
			center.expand_to(xf.origin);
			gizmo_basis = Basis();
		}
		//count++;
	}

	Vector3 pcenter = center.position + center.size * 0.5;
	gizmo.visible = !first;
	gizmo.transform.origin = pcenter;
	gizmo.transform.basis = gizmo_basis;

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->update_transform_gizmo_view();
	}
}

Object *SpatialEditor::_get_editor_data(Object *p_what) {

	Spatial *sp = Object::cast_to<Spatial>(p_what);
	if (!sp)
		return NULL;

	SpatialEditorSelectedItem *si = memnew(SpatialEditorSelectedItem);

	si->sp = sp;
	si->sbox_instance = VisualServer::get_singleton()->instance_create2(selection_box->get_rid(), sp->get_world()->get_scenario());
	VS::get_singleton()->instance_geometry_set_cast_shadows_setting(si->sbox_instance, VS::SHADOW_CASTING_SETTING_OFF);

	if (Engine::get_singleton()->is_editor_hint())
		editor->call("edit_node", sp);

	return si;
}

void SpatialEditor::_generate_selection_box() {

	Rect3 aabb(Vector3(), Vector3(1, 1, 1));
	aabb.grow_by(aabb.get_longest_axis_size() / 20.0);

	Ref<SurfaceTool> st = memnew(SurfaceTool);

	st->begin(Mesh::PRIMITIVE_LINES);
	for (int i = 0; i < 12; i++) {

		Vector3 a, b;
		aabb.get_edge(i, a, b);

		/*Vector<Vector3> points;
		Vector<Color> colors;
		points.push_back(a);
		points.push_back(b);*/

		st->add_color(Color(1.0, 1.0, 0.8, 0.8));
		st->add_vertex(a);
		st->add_color(Color(1.0, 1.0, 0.8, 0.4));
		st->add_vertex(a.linear_interpolate(b, 0.2));

		st->add_color(Color(1.0, 1.0, 0.8, 0.4));
		st->add_vertex(a.linear_interpolate(b, 0.8));
		st->add_color(Color(1.0, 1.0, 0.8, 0.8));
		st->add_vertex(b);
	}

	Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
	mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	mat->set_albedo(Color(1, 1, 1));
	mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	st->set_material(mat);
	selection_box = st->commit();
}

Dictionary SpatialEditor::get_state() const {

	Dictionary d;

	d["snap_enabled"] = snap_enabled;
	d["translate_snap"] = get_translate_snap();
	d["rotate_snap"] = get_rotate_snap();
	d["scale_snap"] = get_scale_snap();

	int local_coords_index = transform_menu->get_popup()->get_item_index(MENU_TRANSFORM_LOCAL_COORDS);
	d["local_coords"] = transform_menu->get_popup()->is_item_checked(local_coords_index);

	int vc = 0;
	if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT)))
		vc = 1;
	else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS)))
		vc = 2;
	else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS)))
		vc = 3;
	else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS)))
		vc = 4;
	else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT)))
		vc = 5;
	else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT)))
		vc = 6;

	d["viewport_mode"] = vc;
	Array vpdata;
	for (int i = 0; i < 4; i++) {
		vpdata.push_back(viewports[i]->get_state());
	}

	d["viewports"] = vpdata;

	d["show_grid"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID));
	d["show_origin"] = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN));
	d["fov"] = get_fov();
	d["znear"] = get_znear();
	d["zfar"] = get_zfar();

	return d;
}
void SpatialEditor::set_state(const Dictionary &p_state) {

	Dictionary d = p_state;

	if (d.has("snap_enabled")) {
		snap_enabled = d["snap_enabled"];
		int snap_enabled_idx = transform_menu->get_popup()->get_item_index(MENU_TRANSFORM_USE_SNAP);
		transform_menu->get_popup()->set_item_checked(snap_enabled_idx, snap_enabled);
	}

	if (d.has("translate_snap"))
		snap_translate->set_text(d["translate_snap"]);

	if (d.has("rotate_snap"))
		snap_rotate->set_text(d["rotate_snap"]);

	if (d.has("scale_snap"))
		snap_scale->set_text(d["scale_snap"]);

	if (d.has("local_coords")) {
		int local_coords_idx = transform_menu->get_popup()->get_item_index(MENU_TRANSFORM_LOCAL_COORDS);
		transform_menu->get_popup()->set_item_checked(local_coords_idx, d["local_coords"]);
		update_transform_gizmo();
	}

	if (d.has("viewport_mode")) {
		int vc = d["viewport_mode"];

		if (vc == 1)
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);
		else if (vc == 2)
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS);
		else if (vc == 3)
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS);
		else if (vc == 4)
			_menu_item_pressed(MENU_VIEW_USE_4_VIEWPORTS);
		else if (vc == 5)
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS_ALT);
		else if (vc == 6)
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS_ALT);
	}

	if (d.has("viewports")) {
		Array vp = d["viewports"];
		ERR_FAIL_COND(vp.size() > 4);

		for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
			viewports[i]->set_state(vp[i]);
		}
	}

	if (d.has("zfar"))
		settings_zfar->set_value(float(d["zfar"]));
	if (d.has("znear"))
		settings_znear->set_value(float(d["znear"]));
	if (d.has("fov"))
		settings_fov->set_value(float(d["fov"]));
	if (d.has("show_grid")) {
		bool use = d["show_grid"];

		if (use != view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID))) {
			_menu_item_pressed(MENU_VIEW_GRID);
		}
	}

	if (d.has("show_origin")) {
		bool use = d["show_origin"];

		if (use != view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN))) {
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), use);
			VisualServer::get_singleton()->instance_set_visible(origin_instance, use);
		}
	}
}

void SpatialEditor::edit(Spatial *p_spatial) {

	if (p_spatial != selected) {
		if (selected) {

			Ref<SpatialEditorGizmo> seg = selected->get_gizmo();
			if (seg.is_valid()) {
				seg->set_selected(false);
				selected->update_gizmo();
			}
		}

		selected = p_spatial;
		over_gizmo_handle = -1;

		if (selected) {

			Ref<SpatialEditorGizmo> seg = selected->get_gizmo();
			if (seg.is_valid()) {
				seg->set_selected(true);
				selected->update_gizmo();
			}
		}
	}

	/*
	if (p_spatial) {
		_validate_selection();
		if (selected.has(p_spatial->get_instance_id()) && selected.size()==1)
			return;
		_select(p_spatial->get_instance_id(),false,true);

		// should become the selection
	}
	*/
}

void SpatialEditor::_xform_dialog_action() {

	Transform t;
	//translation
	Vector3 scale;
	Vector3 rotate;
	Vector3 translate;

	for (int i = 0; i < 3; i++) {
		translate[i] = xform_translate[i]->get_text().to_double();
		rotate[i] = Math::deg2rad(xform_rotate[i]->get_text().to_double());
		scale[i] = xform_scale[i]->get_text().to_double();
	}

	t.basis.scale(scale);
	t.basis.rotate(rotate);
	t.origin = translate;

	undo_redo->create_action(TTR("XForm Dialog"));

	List<Node *> &selection = editor_selection->get_selected_node_list();

	for (List<Node *>::Element *E = selection.front(); E; E = E->next()) {

		Spatial *sp = Object::cast_to<Spatial>(E->get());
		if (!sp)
			continue;

		SpatialEditorSelectedItem *se = editor_selection->get_node_editor_data<SpatialEditorSelectedItem>(sp);
		if (!se)
			continue;

		bool post = xform_type->get_selected() > 0;

		Transform tr = sp->get_global_transform();
		if (post)
			tr = tr * t;
		else {

			tr.basis = t.basis * tr.basis;
			tr.origin += t.origin;
		}

		undo_redo->add_do_method(sp, "set_global_transform", tr);
		undo_redo->add_undo_method(sp, "set_global_transform", sp->get_global_transform());
	}
	undo_redo->commit_action();
}

void SpatialEditor::_menu_item_pressed(int p_option) {

	switch (p_option) {

		case MENU_TOOL_SELECT:
		case MENU_TOOL_MOVE:
		case MENU_TOOL_ROTATE:
		case MENU_TOOL_SCALE:
		case MENU_TOOL_LIST_SELECT: {

			for (int i = 0; i < TOOL_MAX; i++)
				tool_button[i]->set_pressed(i == p_option);
			tool_mode = (ToolMode)p_option;

			//static const char *_mode[]={"Selection Mode.","Translation Mode.","Rotation Mode.","Scale Mode.","List Selection Mode."};
			//set_message(_mode[p_option],3);
			update_transform_gizmo();

		} break;
		case MENU_TRANSFORM_USE_SNAP: {

			bool is_checked = transform_menu->get_popup()->is_item_checked(transform_menu->get_popup()->get_item_index(p_option));
			snap_enabled = !is_checked;
			transform_menu->get_popup()->set_item_checked(transform_menu->get_popup()->get_item_index(p_option), snap_enabled);
		} break;
		case MENU_TRANSFORM_CONFIGURE_SNAP: {

			snap_dialog->popup_centered(Size2(200, 180));
		} break;
		case MENU_TRANSFORM_LOCAL_COORDS: {

			bool is_checked = transform_menu->get_popup()->is_item_checked(transform_menu->get_popup()->get_item_index(p_option));
			transform_menu->get_popup()->set_item_checked(transform_menu->get_popup()->get_item_index(p_option), !is_checked);
			update_transform_gizmo();

		} break;
		case MENU_TRANSFORM_DIALOG: {

			for (int i = 0; i < 3; i++) {

				xform_translate[i]->set_text("0");
				xform_rotate[i]->set_text("0");
				xform_scale[i]->set_text("1");
			}

			xform_dialog->popup_centered(Size2(320, 240) * EDSCALE);

		} break;
		case MENU_VIEW_USE_1_VIEWPORT: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_1_VIEWPORT);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_2_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_2_VIEWPORTS_ALT: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_2_VIEWPORTS_ALT);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_3_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_USE_3_VIEWPORTS_ALT: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_3_VIEWPORTS_ALT);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), true);

		} break;
		case MENU_VIEW_USE_4_VIEWPORTS: {

			viewport_base->set_view(SpatialEditorViewportContainer::VIEW_USE_4_VIEWPORTS);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), true);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), false);
			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), false);

		} break;
		case MENU_VIEW_ORIGIN: {

			bool is_checked = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(p_option));

			is_checked = !is_checked;
			VisualServer::get_singleton()->instance_set_visible(origin_instance, is_checked);

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(p_option), is_checked);
		} break;
		case MENU_VIEW_GRID: {

			bool is_checked = view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(p_option));

			grid_enabled = !is_checked;

			for (int i = 0; i < 3; ++i) {
				if (grid_enable[i]) {
					VisualServer::get_singleton()->instance_set_visible(grid_instance[i], grid_enabled);
					grid_visible[i] = grid_enabled;
				}
			}

			view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(p_option), grid_enabled);

		} break;
		case MENU_VIEW_CAMERA_SETTINGS: {

			settings_dialog->popup_centered(settings_vbc->get_combined_minimum_size() + Size2(50, 50));
		} break;
	}
}

void SpatialEditor::_init_indicators() {

	//RID mat = VisualServer::get_singleton()->fixed_material_create();
	///VisualServer::get_singleton()->fixed_material_set_flag(mat, VisualServer::FIXED_MATERIAL_FLAG_USE_ALPHA,true);
	//VisualServer::get_singleton()->fixed_material_set_flag(mat, VisualServer::FIXED_MATERIAL_FLAG_USE_COLOR_ARRAY,true);

	{

		indicator_mat.instance();
		indicator_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		//indicator_mat->set_flag(SpatialMaterial::FLAG_ONTOP,true);
		indicator_mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
		indicator_mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);

		indicator_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);

		PoolVector<Color> grid_colors[3];
		PoolVector<Vector3> grid_points[3];
		Vector<Color> origin_colors;
		Vector<Vector3> origin_points;

		Color grid_color = EditorSettings::get_singleton()->get("editors/3d/grid_color");

		for (int i = 0; i < 3; i++) {
			Vector3 axis;
			axis[i] = 1;
			Vector3 axis_n1;
			axis_n1[(i + 1) % 3] = 1;
			Vector3 axis_n2;
			axis_n2[(i + 2) % 3] = 1;

			origin_colors.push_back(Color(axis.x, axis.y, axis.z));
			origin_colors.push_back(Color(axis.x, axis.y, axis.z));
			origin_points.push_back(axis * 4096);
			origin_points.push_back(axis * -4096);
#define ORIGIN_GRID_SIZE 100

			for (int j = -ORIGIN_GRID_SIZE; j <= ORIGIN_GRID_SIZE; j++) {

				for (int k = -ORIGIN_GRID_SIZE; k <= ORIGIN_GRID_SIZE; k++) {

					Vector3 p = axis_n1 * j + axis_n2 * k;
					float trans = Math::pow(MAX(0, 1.0 - (Vector2(j, k).length() / ORIGIN_GRID_SIZE)), 2);

					Vector3 pj = axis_n1 * (j + 1) + axis_n2 * k;
					float transj = Math::pow(MAX(0, 1.0 - (Vector2(j + 1, k).length() / ORIGIN_GRID_SIZE)), 2);

					Vector3 pk = axis_n1 * j + axis_n2 * (k + 1);
					float transk = Math::pow(MAX(0, 1.0 - (Vector2(j, k + 1).length() / ORIGIN_GRID_SIZE)), 2);

					Color trans_color = grid_color;
					trans_color.a *= trans;

					Color transk_color = grid_color;
					transk_color.a *= transk;

					Color transj_color = grid_color;
					transj_color.a *= transj;

					if (j % 10 == 0 || k % 10 == 0) {
						trans_color.a *= 2;
					}
					if ((k + 1) % 10 == 0) {
						transk_color.a *= 2;
					}
					if ((j + 1) % 10 == 0) {
						transj_color.a *= 2;
					}

					grid_points[i].push_back(p);
					grid_points[i].push_back(pk);
					grid_colors[i].push_back(trans_color);
					grid_colors[i].push_back(transk_color);

					grid_points[i].push_back(p);
					grid_points[i].push_back(pj);
					grid_colors[i].push_back(trans_color);
					grid_colors[i].push_back(transj_color);
				}
			}

			grid[i] = VisualServer::get_singleton()->mesh_create();
			Array d;
			d.resize(VS::ARRAY_MAX);
			d[VisualServer::ARRAY_VERTEX] = grid_points[i];
			d[VisualServer::ARRAY_COLOR] = grid_colors[i];
			VisualServer::get_singleton()->mesh_add_surface_from_arrays(grid[i], VisualServer::PRIMITIVE_LINES, d);
			VisualServer::get_singleton()->mesh_surface_set_material(grid[i], 0, indicator_mat->get_rid());
			grid_instance[i] = VisualServer::get_singleton()->instance_create2(grid[i], get_tree()->get_root()->get_world()->get_scenario());

			grid_visible[i] = false;
			grid_enable[i] = false;
			VisualServer::get_singleton()->instance_set_visible(grid_instance[i], false);
			VisualServer::get_singleton()->instance_geometry_set_cast_shadows_setting(grid_instance[i], VS::SHADOW_CASTING_SETTING_OFF);
			VS::get_singleton()->instance_set_layer_mask(grid_instance[i], 1 << SpatialEditorViewport::GIZMO_GRID_LAYER);
		}

		origin = VisualServer::get_singleton()->mesh_create();
		Array d;
		d.resize(VS::ARRAY_MAX);
		d[VisualServer::ARRAY_VERTEX] = origin_points;
		d[VisualServer::ARRAY_COLOR] = origin_colors;

		VisualServer::get_singleton()->mesh_add_surface_from_arrays(origin, VisualServer::PRIMITIVE_LINES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(origin, 0, indicator_mat->get_rid());

		//origin = VisualServer::get_singleton()->poly_create();
		//VisualServer::get_singleton()->poly_add_primitive(origin,origin_points,Vector<Vector3>(),origin_colors,Vector<Vector3>());
		//VisualServer::get_singleton()->poly_set_material(origin,indicator_mat,true);
		origin_instance = VisualServer::get_singleton()->instance_create2(origin, get_tree()->get_root()->get_world()->get_scenario());
		VS::get_singleton()->instance_set_layer_mask(origin_instance, 1 << SpatialEditorViewport::GIZMO_GRID_LAYER);

		VisualServer::get_singleton()->instance_geometry_set_cast_shadows_setting(origin_instance, VS::SHADOW_CASTING_SETTING_OFF);

		VisualServer::get_singleton()->instance_set_visible(grid_instance[1], true);
		grid_enable[1] = true;
		grid_visible[1] = true;
		grid_enabled = true;
		last_grid_snap = 1;
	}

	{
		cursor_mesh = VisualServer::get_singleton()->mesh_create();
		PoolVector<Vector3> cursor_points;
		float cs = 0.25;
		cursor_points.push_back(Vector3(+cs, 0, 0));
		cursor_points.push_back(Vector3(-cs, 0, 0));
		cursor_points.push_back(Vector3(0, +cs, 0));
		cursor_points.push_back(Vector3(0, -cs, 0));
		cursor_points.push_back(Vector3(0, 0, +cs));
		cursor_points.push_back(Vector3(0, 0, -cs));
		cursor_material.instance();
		cursor_material->set_albedo(Color(0, 1, 1));
		cursor_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);

		Array d;
		d.resize(VS::ARRAY_MAX);
		d[VS::ARRAY_VERTEX] = cursor_points;
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(cursor_mesh, VS::PRIMITIVE_LINES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(cursor_mesh, 0, cursor_material->get_rid());

		cursor_instance = VisualServer::get_singleton()->instance_create2(cursor_mesh, get_tree()->get_root()->get_world()->get_scenario());
		VS::get_singleton()->instance_set_layer_mask(cursor_instance, 1 << SpatialEditorViewport::GIZMO_GRID_LAYER);

		VisualServer::get_singleton()->instance_geometry_set_cast_shadows_setting(cursor_instance, VS::SHADOW_CASTING_SETTING_OFF);
	}

	{

		//move gizmo

		float gizmo_alph = EditorSettings::get_singleton()->get("editors/3d/manipulator_gizmo_opacity");

		gizmo_hl = Ref<SpatialMaterial>(memnew(SpatialMaterial));
		gizmo_hl->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		gizmo_hl->set_on_top_of_alpha();
		gizmo_hl->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
		gizmo_hl->set_albedo(Color(1, 1, 1, gizmo_alph + 0.2f));
		gizmo_hl->set_cull_mode(SpatialMaterial::CULL_DISABLED);

		for (int i = 0; i < 3; i++) {

			move_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			move_plane_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));
			rotate_gizmo[i] = Ref<ArrayMesh>(memnew(ArrayMesh));

			Ref<SpatialMaterial> mat = memnew(SpatialMaterial);
			mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
			mat->set_on_top_of_alpha();
			mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
			Color col;
			col[i] = 1.0;
			col.a = gizmo_alph;
			mat->set_albedo(col);

			gizmo_color[i] = mat;

			Vector3 ivec;
			ivec[i] = 1;
			Vector3 nivec;
			nivec[(i + 1) % 3] = 1;
			nivec[(i + 2) % 3] = 1;
			Vector3 ivec2;
			ivec2[(i + 1) % 3] = 1;
			Vector3 ivec3;
			ivec3[(i + 2) % 3] = 1;

			{

				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				//translate

				const int arrow_points = 5;
				Vector3 arrow[5] = {
					nivec * 0.0 + ivec * 0.0,
					nivec * 0.01 + ivec * 0.0,
					nivec * 0.01 + ivec * 1.0,
					nivec * 0.1 + ivec * 1.0,
					nivec * 0.0 + ivec * (1 + GIZMO_ARROW_SIZE),
				};

				int arrow_sides = 6;

				for (int k = 0; k < 7; k++) {

					Basis ma(ivec, Math_PI * 2 * float(k) / arrow_sides);
					Basis mb(ivec, Math_PI * 2 * float(k + 1) / arrow_sides);

					for (int j = 0; j < arrow_points - 1; j++) {

						Vector3 points[4] = {
							ma.xform(arrow[j]),
							mb.xform(arrow[j]),
							mb.xform(arrow[j + 1]),
							ma.xform(arrow[j + 1]),
						};
						surftool->add_vertex(points[0]);
						surftool->add_vertex(points[1]);
						surftool->add_vertex(points[2]);

						surftool->add_vertex(points[0]);
						surftool->add_vertex(points[2]);
						surftool->add_vertex(points[3]);
					}
				}

				surftool->set_material(mat);
				surftool->commit(move_gizmo[i]);
			}

			// plane translation
			{
				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				Vector3 vec = ivec2 - ivec3;
				Vector3 plane[4] = {
					vec * GIZMO_PLANE_DST,
					vec * GIZMO_PLANE_DST + ivec2 * GIZMO_PLANE_SIZE,
					vec * (GIZMO_PLANE_DST + GIZMO_PLANE_SIZE),
					vec * GIZMO_PLANE_DST - ivec3 * GIZMO_PLANE_SIZE
				};

				Basis ma(ivec, Math_PI / 2);

				Vector3 points[4] = {
					ma.xform(plane[0]),
					ma.xform(plane[1]),
					ma.xform(plane[2]),
					ma.xform(plane[3]),
				};
				surftool->add_vertex(points[0]);
				surftool->add_vertex(points[1]);
				surftool->add_vertex(points[2]);

				surftool->add_vertex(points[0]);
				surftool->add_vertex(points[2]);
				surftool->add_vertex(points[3]);

				Ref<SpatialMaterial> plane_mat = memnew(SpatialMaterial);
				plane_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
				plane_mat->set_on_top_of_alpha();
				plane_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
				plane_mat->set_cull_mode(SpatialMaterial::CULL_DISABLED);
				Color col;
				col[i] = 1.0;
				col.a = gizmo_alph;
				plane_mat->set_albedo(col);
				plane_gizmo_color[i] = plane_mat; // needed, so we can draw planes from both sides
				surftool->set_material(plane_mat);
				surftool->commit(move_plane_gizmo[i]);
			}

			{

				Ref<SurfaceTool> surftool = memnew(SurfaceTool);
				surftool->begin(Mesh::PRIMITIVE_TRIANGLES);

				Vector3 circle[5] = {
					ivec * 0.02 + ivec2 * 0.02 + ivec2 * GIZMO_CIRCLE_SIZE,
					ivec * -0.02 + ivec2 * 0.02 + ivec2 * GIZMO_CIRCLE_SIZE,
					ivec * -0.02 + ivec2 * -0.02 + ivec2 * GIZMO_CIRCLE_SIZE,
					ivec * 0.02 + ivec2 * -0.02 + ivec2 * GIZMO_CIRCLE_SIZE,
					ivec * 0.02 + ivec2 * 0.02 + ivec2 * GIZMO_CIRCLE_SIZE,
				};

				for (int k = 0; k < 33; k++) {

					Basis ma(ivec, Math_PI * 2 * float(k) / 32);
					Basis mb(ivec, Math_PI * 2 * float(k + 1) / 32);

					for (int j = 0; j < 4; j++) {

						Vector3 points[4] = {
							ma.xform(circle[j]),
							mb.xform(circle[j]),
							mb.xform(circle[j + 1]),
							ma.xform(circle[j + 1]),
						};
						surftool->add_vertex(points[0]);
						surftool->add_vertex(points[1]);
						surftool->add_vertex(points[2]);

						surftool->add_vertex(points[0]);
						surftool->add_vertex(points[2]);
						surftool->add_vertex(points[3]);
					}
				}

				surftool->set_material(mat);
				surftool->commit(rotate_gizmo[i]);
			}
		}
	}

	/*for(int i=0;i<4;i++) {

		viewports[i]->init_gizmo_instance(i);
	}*/

	_generate_selection_box();

	//Object::cast_to<EditorNode>(get_scene()->get_root_node())->get_scene_root()->add_child(camera);

	//current_camera=camera;
}

void SpatialEditor::_finish_indicators() {

	VisualServer::get_singleton()->free(origin_instance);
	VisualServer::get_singleton()->free(origin);
	for (int i = 0; i < 3; i++) {
		VisualServer::get_singleton()->free(grid_instance[i]);
		VisualServer::get_singleton()->free(grid[i]);
	}
	//VisualServer::get_singleton()->free(poly);
	//VisualServer::get_singleton()->free(indicators_instance);
	//VisualServer::get_singleton()->free(indicators);

	VisualServer::get_singleton()->free(cursor_instance);
	VisualServer::get_singleton()->free(cursor_mesh);
}

bool SpatialEditor::is_any_freelook_active() const {
	for (unsigned int i = 0; i < VIEWPORTS_COUNT; ++i) {
		if (viewports[i]->is_freelook_active())
			return true;
	}
	return false;
}

void SpatialEditor::_unhandled_key_input(Ref<InputEvent> p_event) {

	if (!is_visible_in_tree() || get_viewport()->gui_has_modal_stack())
		return;

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {

		// Note: need to check is_echo because first person movement keys might still be held
		if (!is_any_freelook_active() && !p_event->is_echo()) {

			if (!k->is_pressed())
				return;

			if (ED_IS_SHORTCUT("spatial_editor/tool_select", p_event))
				_menu_item_pressed(MENU_TOOL_SELECT);

			else if (ED_IS_SHORTCUT("spatial_editor/tool_move", p_event))
				_menu_item_pressed(MENU_TOOL_MOVE);

			else if (ED_IS_SHORTCUT("spatial_editor/tool_rotate", p_event))
				_menu_item_pressed(MENU_TOOL_ROTATE);

			else if (ED_IS_SHORTCUT("spatial_editor/tool_scale", p_event))
				_menu_item_pressed(MENU_TOOL_SCALE);
		}
	}
}
void SpatialEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_READY) {

		tool_button[SpatialEditor::TOOL_MODE_SELECT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_MOVE]->set_icon(get_icon("ToolMove", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_ROTATE]->set_icon(get_icon("ToolRotate", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_SCALE]->set_icon(get_icon("ToolScale", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_LIST_SELECT]->set_icon(get_icon("ListSelect", "EditorIcons"));

		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), get_icon("Panels1", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), get_icon("Panels2", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), get_icon("Panels2Alt", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), get_icon("Panels3", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), get_icon("Panels3Alt", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), get_icon("Panels4", "EditorIcons"));

		_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);

		get_tree()->connect("node_removed", this, "_node_removed");
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		gizmos = memnew(SpatialEditorGizmos);
		_init_indicators();
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {

		_finish_indicators();
		memdelete(gizmos);
	}
	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		tool_button[SpatialEditor::TOOL_MODE_SELECT]->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_MOVE]->set_icon(get_icon("ToolMove", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_ROTATE]->set_icon(get_icon("ToolRotate", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_SCALE]->set_icon(get_icon("ToolScale", "EditorIcons"));
		tool_button[SpatialEditor::TOOL_MODE_LIST_SELECT]->set_icon(get_icon("ListSelect", "EditorIcons"));

		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT), get_icon("Panels1", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS), get_icon("Panels2", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT), get_icon("Panels2Alt", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS), get_icon("Panels3", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT), get_icon("Panels3Alt", "EditorIcons"));
		view_menu->get_popup()->set_item_icon(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS), get_icon("Panels4", "EditorIcons"));
	}
}

void SpatialEditor::add_control_to_menu_panel(Control *p_control) {

	hbc_menu->add_child(p_control);
}

void SpatialEditor::set_can_preview(Camera *p_preview) {

	for (int i = 0; i < 4; i++) {
		viewports[i]->set_can_preview(p_preview);
	}
}

VSplitContainer *SpatialEditor::get_shader_split() {

	return shader_split;
}

HSplitContainer *SpatialEditor::get_palette_split() {

	return palette_split;
}

void SpatialEditor::_request_gizmo(Object *p_obj) {

	Spatial *sp = Object::cast_to<Spatial>(p_obj);
	if (!sp)
		return;
	if (editor->get_edited_scene() && (sp == editor->get_edited_scene() || (sp->get_owner() && editor->get_edited_scene()->is_a_parent_of(sp)))) {

		Ref<SpatialEditorGizmo> seg;

		for (int i = 0; i < EditorNode::get_singleton()->get_editor_data().get_editor_plugin_count(); i++) {

			seg = EditorNode::get_singleton()->get_editor_data().get_editor_plugin(i)->create_spatial_gizmo(sp);
			if (seg.is_valid())
				break;
		}

		if (!seg.is_valid()) {
			seg = gizmos->get_gizmo(sp);
		}
		if (seg.is_valid()) {
			sp->set_gizmo(seg);
		}

		if (seg.is_valid() && sp == selected) {
			seg->set_selected(true);
			selected->update_gizmo();
		}
	}
}

void SpatialEditor::_toggle_maximize_view(Object *p_viewport) {
	if (!p_viewport) return;
	SpatialEditorViewport *current_viewport = Object::cast_to<SpatialEditorViewport>(p_viewport);
	if (!current_viewport) return;

	int index = -1;
	bool maximized = false;
	for (int i = 0; i < 4; i++) {
		if (viewports[i] == current_viewport) {
			index = i;
			if (current_viewport->get_global_rect() == viewport_base->get_global_rect())
				maximized = true;
			break;
		}
	}
	if (index == -1) return;

	if (!maximized) {

		for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
			if (i == (uint32_t)index)
				viewports[i]->set_area_as_parent_rect();
			else
				viewports[i]->hide();
		}
	} else {

		for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++)
			viewports[i]->show();

		if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_1_VIEWPORT)))
			_menu_item_pressed(MENU_VIEW_USE_1_VIEWPORT);
		else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS)))
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS);
		else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_2_VIEWPORTS_ALT)))
			_menu_item_pressed(MENU_VIEW_USE_2_VIEWPORTS_ALT);
		else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS)))
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS);
		else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_3_VIEWPORTS_ALT)))
			_menu_item_pressed(MENU_VIEW_USE_3_VIEWPORTS_ALT);
		else if (view_menu->get_popup()->is_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_USE_4_VIEWPORTS)))
			_menu_item_pressed(MENU_VIEW_USE_4_VIEWPORTS);
	}
}

void SpatialEditor::_node_removed(Node *p_node) {

	if (p_node == selected)
		selected = NULL;
}

void SpatialEditor::_bind_methods() {

	//ClassDB::bind_method("_gui_input",&SpatialEditor::_gui_input);
	ClassDB::bind_method("_unhandled_key_input", &SpatialEditor::_unhandled_key_input);
	ClassDB::bind_method("_node_removed", &SpatialEditor::_node_removed);
	ClassDB::bind_method("_menu_item_pressed", &SpatialEditor::_menu_item_pressed);
	ClassDB::bind_method("_xform_dialog_action", &SpatialEditor::_xform_dialog_action);
	ClassDB::bind_method("_get_editor_data", &SpatialEditor::_get_editor_data);
	ClassDB::bind_method("_request_gizmo", &SpatialEditor::_request_gizmo);
	ClassDB::bind_method("_toggle_maximize_view", &SpatialEditor::_toggle_maximize_view);

	ADD_SIGNAL(MethodInfo("transform_key_request"));
}

void SpatialEditor::clear() {

	settings_fov->set_value(EDITOR_DEF("editors/3d/default_fov", 55.0));
	settings_znear->set_value(EDITOR_DEF("editors/3d/default_z_near", 0.1));
	settings_zfar->set_value(EDITOR_DEF("editors/3d/default_z_far", 1500.0));

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {
		viewports[i]->reset();
	}

	VisualServer::get_singleton()->instance_set_visible(origin_instance, true);
	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_ORIGIN), true);
	for (int i = 0; i < 3; ++i) {
		if (grid_enable[i]) {
			VisualServer::get_singleton()->instance_set_visible(grid_instance[i], true);
			grid_visible[i] = true;
		}
	}

	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {

		viewports[i]->view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(SpatialEditorViewport::VIEW_AUDIO_LISTENER), i == 0);
		viewports[i]->viewport->set_as_audio_listener(i == 0);
	}

	view_menu->get_popup()->set_item_checked(view_menu->get_popup()->get_item_index(MENU_VIEW_GRID), true);
}

SpatialEditor::SpatialEditor(EditorNode *p_editor) {

	gizmo.visible = true;
	gizmo.scale = 1.0;

	viewport_environment = Ref<Environment>(memnew(Environment));
	undo_redo = p_editor->get_undo_redo();
	VBoxContainer *vbc = this;

	custom_camera = NULL;
	singleton = this;
	editor = p_editor;
	editor_selection = editor->get_editor_selection();
	editor_selection->add_editor_plugin(this);

	snap_enabled = false;
	tool_mode = TOOL_MODE_SELECT;

	//set_focus_mode(FOCUS_ALL);

	hbc_menu = memnew(HBoxContainer);
	vbc->add_child(hbc_menu);

	Vector<Variant> button_binds;
	button_binds.resize(1);

	tool_button[TOOL_MODE_SELECT] = memnew(ToolButton);
	hbc_menu->add_child(tool_button[TOOL_MODE_SELECT]);
	tool_button[TOOL_MODE_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SELECT]->set_flat(true);
	tool_button[TOOL_MODE_SELECT]->set_pressed(true);
	button_binds[0] = MENU_TOOL_SELECT;
	tool_button[TOOL_MODE_SELECT]->connect("pressed", this, "_menu_item_pressed", button_binds);
	tool_button[TOOL_MODE_SELECT]->set_tooltip(TTR("Select Mode (Q)\n") + keycode_get_string(KEY_MASK_CMD) + TTR("Drag: Rotate\nAlt+Drag: Move\nAlt+RMB: Depth list selection"));

	tool_button[TOOL_MODE_MOVE] = memnew(ToolButton);

	hbc_menu->add_child(tool_button[TOOL_MODE_MOVE]);
	tool_button[TOOL_MODE_MOVE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_MOVE]->set_flat(true);
	button_binds[0] = MENU_TOOL_MOVE;
	tool_button[TOOL_MODE_MOVE]->connect("pressed", this, "_menu_item_pressed", button_binds);
	tool_button[TOOL_MODE_MOVE]->set_tooltip(TTR("Move Mode (W)"));

	tool_button[TOOL_MODE_ROTATE] = memnew(ToolButton);
	hbc_menu->add_child(tool_button[TOOL_MODE_ROTATE]);
	tool_button[TOOL_MODE_ROTATE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_ROTATE]->set_flat(true);
	button_binds[0] = MENU_TOOL_ROTATE;
	tool_button[TOOL_MODE_ROTATE]->connect("pressed", this, "_menu_item_pressed", button_binds);
	tool_button[TOOL_MODE_ROTATE]->set_tooltip(TTR("Rotate Mode (E)"));

	tool_button[TOOL_MODE_SCALE] = memnew(ToolButton);
	hbc_menu->add_child(tool_button[TOOL_MODE_SCALE]);
	tool_button[TOOL_MODE_SCALE]->set_toggle_mode(true);
	tool_button[TOOL_MODE_SCALE]->set_flat(true);
	button_binds[0] = MENU_TOOL_SCALE;
	tool_button[TOOL_MODE_SCALE]->connect("pressed", this, "_menu_item_pressed", button_binds);
	tool_button[TOOL_MODE_SCALE]->set_tooltip(TTR("Scale Mode (R)"));

	VSeparator *vs = memnew(VSeparator);
	hbc_menu->add_child(vs);

	tool_button[TOOL_MODE_LIST_SELECT] = memnew(ToolButton);
	hbc_menu->add_child(tool_button[TOOL_MODE_LIST_SELECT]);
	tool_button[TOOL_MODE_LIST_SELECT]->set_toggle_mode(true);
	tool_button[TOOL_MODE_LIST_SELECT]->set_flat(true);
	button_binds[0] = MENU_TOOL_LIST_SELECT;
	tool_button[TOOL_MODE_LIST_SELECT]->connect("pressed", this, "_menu_item_pressed", button_binds);
	tool_button[TOOL_MODE_LIST_SELECT]->set_tooltip(TTR("Show a list of all objects at the position clicked\n(same as Alt+RMB in select mode)."));

	vs = memnew(VSeparator);
	hbc_menu->add_child(vs);

	// Drag and drop support;
	preview_node = memnew(Spatial);
	preview_bounds = Rect3();

	ED_SHORTCUT("spatial_editor/bottom_view", TTR("Bottom View"), KEY_MASK_ALT + KEY_KP_7);
	ED_SHORTCUT("spatial_editor/top_view", TTR("Top View"), KEY_KP_7);
	ED_SHORTCUT("spatial_editor/rear_view", TTR("Rear View"), KEY_MASK_ALT + KEY_KP_1);
	ED_SHORTCUT("spatial_editor/front_view", TTR("Front View"), KEY_KP_1);
	ED_SHORTCUT("spatial_editor/left_view", TTR("Left View"), KEY_MASK_ALT + KEY_KP_3);
	ED_SHORTCUT("spatial_editor/right_view", TTR("Right View"), KEY_KP_3);
	ED_SHORTCUT("spatial_editor/switch_perspective_orthogonal", TTR("Switch Perspective/Orthogonal view"), KEY_KP_5);
	ED_SHORTCUT("spatial_editor/snap", TTR("Snap"), KEY_S);
	ED_SHORTCUT("spatial_editor/insert_anim_key", TTR("Insert Animation Key"), KEY_K);
	ED_SHORTCUT("spatial_editor/focus_origin", TTR("Focus Origin"), KEY_O);
	ED_SHORTCUT("spatial_editor/focus_selection", TTR("Focus Selection"), KEY_F);
	ED_SHORTCUT("spatial_editor/align_selection_with_view", TTR("Align Selection With View"), KEY_MASK_ALT + KEY_MASK_CMD + KEY_F);

	ED_SHORTCUT("spatial_editor/tool_select", TTR("Tool Select"), KEY_Q);
	ED_SHORTCUT("spatial_editor/tool_move", TTR("Tool Move"), KEY_W);
	ED_SHORTCUT("spatial_editor/tool_rotate", TTR("Tool Rotate"), KEY_E);
	ED_SHORTCUT("spatial_editor/tool_scale", TTR("Tool Scale"), KEY_R);

	ED_SHORTCUT("spatial_editor/display_wireframe", TTR("Display Wireframe"), KEY_Z);

	PopupMenu *p;

	transform_menu = memnew(MenuButton);
	transform_menu->set_text(TTR("Transform"));
	hbc_menu->add_child(transform_menu);

	p = transform_menu->get_popup();
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/use_snap", TTR("Use Snap")), MENU_TRANSFORM_USE_SNAP);
	p->add_shortcut(ED_SHORTCUT("spatial_editor/configure_snap", TTR("Configure Snap..")), MENU_TRANSFORM_CONFIGURE_SNAP);
	p->add_separator();
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/local_coords", TTR("Local Coords")), MENU_TRANSFORM_LOCAL_COORDS);
	//p->set_item_checked(p->get_item_count()-1,true);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/transform_dialog", TTR("Transform Dialog..")), MENU_TRANSFORM_DIALOG);

	p->connect("id_pressed", this, "_menu_item_pressed");

	view_menu = memnew(MenuButton);
	view_menu->set_text(TTR("View"));
	view_menu->set_position(Point2(212, 0));
	hbc_menu->add_child(view_menu);

	p = view_menu->get_popup();

	accept = memnew(AcceptDialog);
	editor->get_gui_base()->add_child(accept);

	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/1_viewport", TTR("1 Viewport"), KEY_MASK_CMD + KEY_1), MENU_VIEW_USE_1_VIEWPORT);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports", TTR("2 Viewports"), KEY_MASK_CMD + KEY_2), MENU_VIEW_USE_2_VIEWPORTS);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/2_viewports_alt", TTR("2 Viewports (Alt)"), KEY_MASK_ALT + KEY_MASK_CMD + KEY_2), MENU_VIEW_USE_2_VIEWPORTS_ALT);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports", TTR("3 Viewports"), KEY_MASK_CMD + KEY_3), MENU_VIEW_USE_3_VIEWPORTS);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/3_viewports_alt", TTR("3 Viewports (Alt)"), KEY_MASK_ALT + KEY_MASK_CMD + KEY_3), MENU_VIEW_USE_3_VIEWPORTS_ALT);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/4_viewports", TTR("4 Viewports"), KEY_MASK_CMD + KEY_4), MENU_VIEW_USE_4_VIEWPORTS);
	p->add_separator();

	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_origin", TTR("View Origin")), MENU_VIEW_ORIGIN);
	p->add_check_shortcut(ED_SHORTCUT("spatial_editor/view_grid", TTR("View Grid")), MENU_VIEW_GRID);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("spatial_editor/settings", TTR("Settings")), MENU_VIEW_CAMERA_SETTINGS);

	p->set_item_checked(p->get_item_index(MENU_VIEW_ORIGIN), true);
	p->set_item_checked(p->get_item_index(MENU_VIEW_GRID), true);

	p->connect("id_pressed", this, "_menu_item_pressed");

	/* REST OF MENU */

	palette_split = memnew(HSplitContainer);
	palette_split->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(palette_split);

	shader_split = memnew(VSplitContainer);
	shader_split->set_h_size_flags(SIZE_EXPAND_FILL);
	palette_split->add_child(shader_split);
	viewport_base = memnew(SpatialEditorViewportContainer);
	shader_split->add_child(viewport_base);
	viewport_base->set_v_size_flags(SIZE_EXPAND_FILL);
	for (uint32_t i = 0; i < VIEWPORTS_COUNT; i++) {

		viewports[i] = memnew(SpatialEditorViewport(this, editor, i));
		viewports[i]->connect("toggle_maximize_view", this, "_toggle_maximize_view");
		viewports[i]->assign_pending_data_pointers(preview_node, &preview_bounds, accept);
		viewport_base->add_child(viewports[i]);
	}
	//vbc->add_child(viewport_base);

	/* SNAP DIALOG */

	snap_dialog = memnew(ConfirmationDialog);
	snap_dialog->set_title(TTR("Snap Settings"));
	add_child(snap_dialog);

	VBoxContainer *snap_dialog_vbc = memnew(VBoxContainer);
	snap_dialog->add_child(snap_dialog_vbc);
	//snap_dialog->set_child_rect(snap_dialog_vbc);

	snap_translate = memnew(LineEdit);
	snap_translate->set_text("1");
	snap_dialog_vbc->add_margin_child(TTR("Translate Snap:"), snap_translate);

	snap_rotate = memnew(LineEdit);
	snap_rotate->set_text("5");
	snap_dialog_vbc->add_margin_child(TTR("Rotate Snap (deg.):"), snap_rotate);

	snap_scale = memnew(LineEdit);
	snap_scale->set_text("5");
	snap_dialog_vbc->add_margin_child(TTR("Scale Snap (%):"), snap_scale);

	/* SETTINGS DIALOG */

	settings_dialog = memnew(ConfirmationDialog);
	settings_dialog->set_title(TTR("Viewport Settings"));
	add_child(settings_dialog);
	settings_vbc = memnew(VBoxContainer);
	settings_vbc->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	settings_dialog->add_child(settings_vbc);
	//settings_dialog->set_child_rect(settings_vbc);

	settings_fov = memnew(SpinBox);
	settings_fov->set_max(MAX_FOV);
	settings_fov->set_min(MIN_FOV);
	settings_fov->set_step(0.01);
	settings_fov->set_value(EDITOR_DEF("editors/3d/default_fov", 55.0));
	settings_vbc->add_margin_child(TTR("Perspective FOV (deg.):"), settings_fov);

	settings_znear = memnew(SpinBox);
	settings_znear->set_max(MAX_Z);
	settings_znear->set_min(MIN_Z);
	settings_znear->set_step(0.01);
	settings_znear->set_value(EDITOR_DEF("editors/3d/default_z_near", 0.1));
	settings_vbc->add_margin_child(TTR("View Z-Near:"), settings_znear);

	settings_zfar = memnew(SpinBox);
	settings_zfar->set_max(MAX_Z);
	settings_zfar->set_min(MIN_Z);
	settings_zfar->set_step(0.01);
	settings_zfar->set_value(EDITOR_DEF("editors/3d/default_z_far", 1500));
	settings_vbc->add_margin_child(TTR("View Z-Far:"), settings_zfar);

	//settings_dialog->get_cancel()->hide();
	/* XFORM DIALOG */

	xform_dialog = memnew(ConfirmationDialog);
	xform_dialog->set_title(TTR("Transform Change"));
	add_child(xform_dialog);

	VBoxContainer *xform_vbc = memnew(VBoxContainer);
	xform_dialog->add_child(xform_vbc);

	Label *l = memnew(Label);
	l->set_text(TTR("Translate:"));
	xform_vbc->add_child(l);

	HBoxContainer *xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {

		xform_translate[i] = memnew(LineEdit);
		xform_translate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_hbc->add_child(xform_translate[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Rotate (deg.):"));
	xform_vbc->add_child(l);

	xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {
		xform_rotate[i] = memnew(LineEdit);
		xform_rotate[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_hbc->add_child(xform_rotate[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Scale (ratio):"));
	xform_vbc->add_child(l);

	xform_hbc = memnew(HBoxContainer);
	xform_vbc->add_child(xform_hbc);

	for (int i = 0; i < 3; i++) {
		xform_scale[i] = memnew(LineEdit);
		xform_scale[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		xform_hbc->add_child(xform_scale[i]);
	}

	l = memnew(Label);
	l->set_text(TTR("Transform Type"));
	xform_vbc->add_child(l);

	xform_type = memnew(OptionButton);
	xform_type->set_h_size_flags(SIZE_EXPAND_FILL);
	xform_type->add_item(TTR("Pre"));
	xform_type->add_item(TTR("Post"));
	xform_vbc->add_child(xform_type);

	xform_dialog->connect("confirmed", this, "_xform_dialog_action");

	scenario_debug = VisualServer::SCENARIO_DEBUG_DISABLED;

	selected = NULL;

	set_process_unhandled_key_input(true);
	add_to_group("_spatial_editor_group");

	EDITOR_DEF("editors/3d/manipulator_gizmo_size", 80);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "editors/3d/manipulator_gizmo_size", PROPERTY_HINT_RANGE, "16,1024,1"));
	EDITOR_DEF("editors/3d/manipulator_gizmo_opacity", 0.2);

	over_gizmo_handle = -1;
}

SpatialEditor::~SpatialEditor() {
	memdelete(preview_node);
}

void SpatialEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		spatial_editor->show();
		spatial_editor->set_process(true);
		//VisualServer::get_singleton()->viewport_set_hide_scenario(editor->get_scene_root()->get_viewport(),false);
		spatial_editor->grab_focus();

	} else {

		spatial_editor->hide();
		spatial_editor->set_process(false);
		//VisualServer::get_singleton()->viewport_set_hide_scenario(editor->get_scene_root()->get_viewport(),true);
	}
}
void SpatialEditorPlugin::edit(Object *p_object) {

	spatial_editor->edit(Object::cast_to<Spatial>(p_object));
}

bool SpatialEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Spatial");
}

Dictionary SpatialEditorPlugin::get_state() const {
	return spatial_editor->get_state();
}

void SpatialEditorPlugin::set_state(const Dictionary &p_state) {

	spatial_editor->set_state(p_state);
}

void SpatialEditor::snap_cursor_to_plane(const Plane &p_plane) {

	//cursor.pos=p_plane.project(cursor.pos);
}

void SpatialEditorPlugin::_bind_methods() {

	ClassDB::bind_method("snap_cursor_to_plane", &SpatialEditorPlugin::snap_cursor_to_plane);
}

void SpatialEditorPlugin::snap_cursor_to_plane(const Plane &p_plane) {

	spatial_editor->snap_cursor_to_plane(p_plane);
}

SpatialEditorPlugin::SpatialEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	spatial_editor = memnew(SpatialEditor(p_node));
	spatial_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->get_viewport()->add_child(spatial_editor);

	//spatial_editor->set_area_as_parent_rect();
	spatial_editor->hide();
	spatial_editor->connect("transform_key_request", editor, "_transform_keyed");

	//spatial_editor->set_process(true);
}

SpatialEditorPlugin::~SpatialEditorPlugin() {
}
