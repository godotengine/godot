/**************************************************************************/
/*  canvas_item_editor_gizmos.cpp                                         */
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

#include "canvas_item_editor_gizmos.h"

#include "core/math/geometry_2d.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "scene/resources/mesh.h"

bool EditorCanvasItemGizmo::is_editable() const {
	ERR_FAIL_NULL_V(canvas_item, false);

	const Node *edited_root = canvas_item->get_tree()->get_edited_scene_root();
	if (canvas_item == edited_root) {
		return true;
	}
	if (canvas_item->get_owner() == edited_root) {
		return true;
	}
	if (edited_root->is_editable_instance(canvas_item->get_owner())) {
		return true;
	}

	return false;
}

void EditorCanvasItemGizmo::clear() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());

	for (Instance &instance : instances) {
		if (instance.instance.is_valid()) {
			RS::get_singleton()->free_rid(instance.instance);
			instance.instance = RID();
		}
	}

	collision_segments.clear();
	collision_rects.clear();
	collision_polygons.clear();
	instances.clear();
	handles.clear();
	handle_ids.clear();
	secondary_handles.clear();
	secondary_handle_ids.clear();
	use_boundary_handle = false;
	use_pivot_handle = false;
}

void EditorCanvasItemGizmo::redraw() {
	ERR_FAIL_NULL(gizmo_plugin);
	clear();

	if (!GDVIRTUAL_CALL(_redraw)) {
		gizmo_plugin->redraw(this);
	}

	if (CanvasItemEditor::get_singleton()->is_current_selected_gizmo(this)) {
		CanvasItemEditor::get_singleton()->update_transform_gizmo();
	}
}
bool EditorCanvasItemGizmo::_edit_use_rect() const {
	ERR_FAIL_NULL_V(gizmo_plugin, false);
	bool ret = false;
	if (GDVIRTUAL_CALL(_edit_use_rect, ret)) {
		return ret;
	}

	return gizmo_plugin->_edit_use_rect(this);
}

Rect2 EditorCanvasItemGizmo::_edit_get_rect() const {
	ERR_FAIL_NULL_V(gizmo_plugin, Rect2());
	Rect2 ret;
	if (GDVIRTUAL_CALL(_edit_get_rect, ret)) {
		return ret;
	}
	return gizmo_plugin->_edit_get_rect(this);
}

void EditorCanvasItemGizmo::_edit_set_rect(const Rect2 &p_rect) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_edit_set_rect, p_rect)) {
		return;
	}
	gizmo_plugin->_edit_set_rect(this, p_rect);
}

bool EditorCanvasItemGizmo::_has_pivot() const {
	ERR_FAIL_NULL_V(gizmo_plugin, false);
	bool ret = false;
	if (GDVIRTUAL_CALL(_has_pivot, ret)) {
		return ret;
	}
	return gizmo_plugin->_has_pivot(this);
}

Vector2 EditorCanvasItemGizmo::_get_pivot() const {
	ERR_FAIL_NULL_V(gizmo_plugin, Vector2());
	Vector2 ret;
	if (GDVIRTUAL_CALL(_get_pivot, ret)) {
		return ret;
	}

	return gizmo_plugin->_get_pivot(this);
}

void EditorCanvasItemGizmo::_set_pivot(const Vector2 &p_point) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_set_pivot, p_point)) {
		return;
	}

	gizmo_plugin->_set_pivot(this, p_point);
}

Dictionary EditorCanvasItemGizmo::_edit_get_state() const {
	ERR_FAIL_NULL_V(gizmo_plugin, Dictionary());

	// because the gdscript method has no way of calling super back into c++ code, we
	// implicitly merge what the gdscript implementation has returned with the returns from
	// the c++ implementation, letting the c++ implementation win in case of conflicts.
	Dictionary ret;
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_get_state)) {
		GDVIRTUAL_CALL(_edit_get_state, ret);
	}

	Dictionary base = gizmo_plugin->_edit_get_state(this);
	if (ret.is_empty()) {
		// skip merge logic, no point in wasting the CPU cycles for that
		return base;
	}

	// make a copy (otherwise gdscript code might be surprised if the dictionary is magically changed)
	return ret.merged(base, true);
}

void EditorCanvasItemGizmo::_edit_set_state(const Dictionary &p_state) {
	ERR_FAIL_NULL(gizmo_plugin);

	// this is a bit different, because gdscript methods cannot call super into c++ code, so
	// when the method is overridden, we still call the gizmo plugin afterward (which in turn calls the CanvasItem)
	// to ensure the canvas item also gets its state restored.
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_set_state)) {
		GDVIRTUAL_CALL(_edit_set_state, p_state);
	}

	gizmo_plugin->_edit_set_state(this, p_state);
}

String EditorCanvasItemGizmo::_get_handle_name(int p_id, bool p_secondary) const {
	ERR_FAIL_NULL_V(gizmo_plugin, "");
	String ret;
	if (GDVIRTUAL_CALL(_get_handle_name, p_id, p_secondary, ret)) {
		return ret;
	}

	return gizmo_plugin->_get_handle_name(this, p_id, p_secondary);
}

bool EditorCanvasItemGizmo::_is_handle_highlighted(int p_id, bool p_secondary) const {
	ERR_FAIL_NULL_V(gizmo_plugin, false);
	bool success = false;
	if (GDVIRTUAL_CALL(_is_handle_highlighted, p_id, p_secondary, success)) {
		return success;
	}

	return gizmo_plugin->_is_handle_highlighted(this, p_id, p_secondary);
}

Variant EditorCanvasItemGizmo::_get_handle_value(int p_id, bool p_secondary) const {
	ERR_FAIL_NULL_V(gizmo_plugin, Variant());
	Variant value;
	if (GDVIRTUAL_CALL(_get_handle_value, p_id, p_secondary, value)) {
		return value;
	}

	return gizmo_plugin->_get_handle_value(this, p_id, p_secondary);
}

void EditorCanvasItemGizmo::_begin_handle_action(int p_id, bool p_secondary) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_begin_handle_action, p_id, p_secondary)) {
		return;
	}

	gizmo_plugin->_begin_handle_action(this, p_id, p_secondary);
}

void EditorCanvasItemGizmo::_set_handle(int p_id, bool p_secondary, const Point2 &p_point) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_set_handle, p_id, p_secondary, p_point)) {
		return;
	}

	gizmo_plugin->_set_handle(this, p_id, p_secondary, p_point);
}

void EditorCanvasItemGizmo::_commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_commit_handle, p_id, p_secondary, p_restore, p_cancel)) {
		return;
	}

	gizmo_plugin->_commit_handle(this, p_id, p_secondary, p_restore, p_cancel);
}

int EditorCanvasItemGizmo::_subgizmos_intersect_point(const Point2 &p_point, real_t p_max_distance) const {
	ERR_FAIL_NULL_V(gizmo_plugin, -1);
	int id = -1;
	if (GDVIRTUAL_CALL(_subgizmos_intersect_point, p_point, p_max_distance, id)) {
		return id;
	}

	return gizmo_plugin->_subgizmos_intersect_point(this, p_point, p_max_distance);
}

Vector<int> EditorCanvasItemGizmo::_subgizmos_intersect_rect(const Rect2 &p_rect) const {
	ERR_FAIL_NULL_V(gizmo_plugin, Vector<int>());
	Vector<int> ret;
	if (GDVIRTUAL_CALL(_subgizmos_intersect_rect, p_rect, ret)) {
		return ret;
	}

	return gizmo_plugin->_subgizmos_intersect_rect(this, p_rect);
}

Transform2D EditorCanvasItemGizmo::_get_subgizmo_transform(int p_id) const {
	ERR_FAIL_NULL_V(gizmo_plugin, Transform2D());
	Transform2D ret;
	if (GDVIRTUAL_CALL(_get_subgizmo_transform, p_id, ret)) {
		return ret;
	}

	return gizmo_plugin->_get_subgizmo_transform(this, p_id);
}

void EditorCanvasItemGizmo::_set_subgizmo_transform(int p_id, const Transform2D &p_transform) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_CALL(_set_subgizmo_transform, p_id, p_transform)) {
		return;
	}

	gizmo_plugin->_set_subgizmo_transform(this, p_id, p_transform);
}

void EditorCanvasItemGizmo::_commit_subgizmos(const Vector<int> &p_ids, const Vector<Transform2D> &p_restore, bool p_cancel) {
	ERR_FAIL_NULL(gizmo_plugin);
	if (GDVIRTUAL_IS_OVERRIDDEN(_commit_subgizmos)) {
		TypedArray<Transform2D> restore;
		restore.resize(p_restore.size());
		for (int i = 0; i < p_restore.size(); i++) {
			restore[i] = p_restore[i];
		}

		if (GDVIRTUAL_CALL(_commit_subgizmos, p_ids, restore, p_cancel)) {
			return;
		}
	}

	gizmo_plugin->_commit_subgizmos(this, p_ids, p_restore, p_cancel);
}

void EditorCanvasItemGizmo::set_canvas_item(CanvasItem *p_canvas_item) {
	ERR_FAIL_NULL(p_canvas_item);
	canvas_item = p_canvas_item;
}

void EditorCanvasItemGizmo::Instance::create_instance(CanvasItem *p_base, bool p_visible) {
	ERR_FAIL_NULL(p_base);

	instance = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->canvas_item_set_parent(instance, p_base->get_canvas_item());
	RS::get_singleton()->canvas_item_set_visible(instance, p_visible);
}

void EditorCanvasItemGizmo::add_circle(const Vector2 &p_pos, float p_radius, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Instance ins;
	ins.create_instance(canvas_item, visible);
	RS::get_singleton()->canvas_item_add_circle(ins.instance, p_pos, p_radius, p_color);
	instances.push_back(ins);
}

void EditorCanvasItemGizmo::add_polygon(const Vector<Vector2> &p_polygon, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Vector<Color> colors;
	colors.reserve(p_polygon.size());
	for (int i = 0; i < p_polygon.size(); i++) {
		colors.append(p_color);
	}

	Instance ins;
	ins.create_instance(canvas_item, visible);
	RS::get_singleton()->canvas_item_add_polygon(ins.instance, p_polygon, colors);
	instances.push_back(ins);
}

void EditorCanvasItemGizmo::add_polyline(const Vector<Vector2> &p_points, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Vector<Color> colors;
	colors.reserve(p_points.size());
	for (int i = 0; i < p_points.size(); i++) {
		colors.append(p_color);
	}

	Instance ins;
	ins.create_instance(canvas_item, visible);
	RS::get_singleton()->canvas_item_add_polyline(ins.instance, p_points, colors);
	instances.push_back(ins);
}

void EditorCanvasItemGizmo::add_rect(const Rect2 &p_rect, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Instance ins;
	ins.create_instance(canvas_item, visible);
	RS::get_singleton()->canvas_item_add_rect(ins.instance, p_rect, p_color);
	instances.push_back(ins);
}

void EditorCanvasItemGizmo::add_collision_segments(const Vector<Vector2> &p_lines) {
	ERR_FAIL_COND_MSG(p_lines.size() % 2 != 0, "Collision segments must be a list of even length.");
	int from = collision_segments.size();
	collision_segments.resize(from + p_lines.size());
	for (int i = 0; i < p_lines.size(); i++) {
		collision_segments.write[from + i] = p_lines[i];
	}
}

void EditorCanvasItemGizmo::add_collision_rect(const Rect2 &p_rect) {
	collision_rects.push_back(p_rect);
}

void EditorCanvasItemGizmo::add_collision_polygon(const Vector<Vector2> &p_polygon) {
	collision_polygons.push_back(p_polygon);
}

void EditorCanvasItemGizmo::add_handles(const Vector<Vector2> &p_handles, Ref<Texture2D> p_texture, const Vector<int> &p_ids, bool p_secondary) {
	if (!_is_selected() || !is_editable()) {
		return;
	}

	ERR_FAIL_NULL(canvas_item);

	Vector<Vector2> &handle_list = p_secondary ? secondary_handles : handles;
	Vector<int> &id_list = p_secondary ? secondary_handle_ids : handle_ids;

	if (p_ids.is_empty()) {
		ERR_FAIL_COND_MSG(!id_list.is_empty(), "IDs must be provided for all handles, as handles with IDs already exist.");
	} else {
		ERR_FAIL_COND_MSG(p_handles.size() != p_ids.size(), "The number of IDs should be the same as the number of handles.");
	}

	bool is_current_hover_gizmo = CanvasItemEditor::get_singleton()->get_current_hover_gizmo() == this;
	bool current_hover_handle_secondary;
	int current_hover_handle = CanvasItemEditor::get_singleton()->get_current_hover_gizmo_handle(current_hover_handle_secondary);

	Ref<Texture2D> texture = p_texture;
	if (texture.is_null()) {
		texture = EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("EditorHandle"), SNAME("EditorIcons"));
	}
	// shouldn't happen but better be safe
	ERR_FAIL_COND(texture.is_null());
	Size2 texture_size = texture->get_size();

	Control *viewport = CanvasItemEditor::get_singleton()->get_viewport_control();

	// we draw handles in viewport space so they will change position with zoom/pan but not scale
	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * canvas_item->get_screen_transform();

	int64_t handle_count = p_handles.size();
	for (int i = 0; i < handle_count; i++) {
		Vector2 position = xform.xform(p_handles[i]);
		int id = p_ids.is_empty() ? i : p_ids[i];

		Instance ins;
		ins.create_instance(viewport, visible);
		instances.push_back(ins);

		Color modulate = Color(1, 1, 1, 1.0);
		if (_is_handle_highlighted(id, p_secondary)) {
			modulate = Color(0, 0, 1, 0.9);
		}

		if (!is_current_hover_gizmo || current_hover_handle != id || p_secondary != current_hover_handle_secondary) {
			modulate.a = 0.8;
		}

		RS::get_singleton()->canvas_item_add_texture_rect(ins.instance, Rect2(position - texture_size / 2, texture_size), texture->get_rid(), false, modulate);
	}

	// update internal handle lists
	int current_size = handle_list.size();
	handle_list.resize(current_size + handle_count);
	for (int i = 0; i < handle_count; i++) {
		handle_list.write[current_size + i] = p_handles[i];
	}

	if (!p_ids.is_empty()) {
		current_size = id_list.size();
		id_list.resize(current_size + p_ids.size());
		for (int i = 0; i < p_ids.size(); i++) {
			id_list.write[current_size + i] = p_ids[i];
		}
	}
}

bool EditorCanvasItemGizmo::intersect_rect(const Rect2 &p_rect) const {
	ERR_FAIL_NULL_V(canvas_item, false);
	ERR_FAIL_COND_V(!valid, false);

	if (!visible && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	Transform2D transform = canvas_item->get_global_transform();

	// for collision segments it is enough if at least one point
	// of a segment is inside the rectangle
	for (const Vector2 &pos : collision_segments) {
		Vector2 global_position = transform.xform(pos);
		if (p_rect.has_point(global_position)) {
			return true;
		}
	}

	// same for collision polygons
	for (const Vector<Vector2> &collision_polygon : collision_polygons) {
		for (const Vector2 &collision_point : collision_polygon) {
			Vector2 global_position = transform.xform(collision_point);
			if (p_rect.has_point(global_position)) {
				return true;
			}
		}
	}

	// for rectangles we check if they overlap
	Transform2D inverse_transform = transform.affine_inverse();
	for (const Rect2 &collision_rect : collision_rects) {
		if (collision_rect.intersects_transformed(inverse_transform, p_rect)) {
			return true;
		}
	}

	return false;
}

void EditorCanvasItemGizmo::handles_intersect_point(const Point2 &p_point, real_t p_max_distance, bool p_shift_pressed, int &r_id, bool &r_secondary) {
	r_id = -1;
	r_secondary = false;

	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);

	if (!visible) {
		return;
	}

	real_t min_d = 1e20;
	for (int i = 0; i < secondary_handles.size(); i++) {
		real_t distance = secondary_handles[i].distance_to(p_point);
		if (distance < p_max_distance && distance < min_d) {
			min_d = distance;
			if (secondary_handle_ids.is_empty()) {
				r_id = i;
			} else {
				r_id = secondary_handle_ids[i];
			}
			r_secondary = true;
		}
	}

	if (r_id != -1 && p_shift_pressed) {
		return;
	}

	min_d = 1e20;
	for (int i = 0; i < handles.size(); i++) {
		real_t distance = handles[i].distance_to(p_point);
		if (distance < p_max_distance && distance < min_d) {
			min_d = distance;
			if (handle_ids.is_empty()) {
				r_id = i;
			} else {
				r_id = handle_ids[i];
			}
			r_secondary = false;
		}
	}
}

bool EditorCanvasItemGizmo::intersect_point(const Point2 &p_point, const real_t p_max_distance) const {
	ERR_FAIL_NULL_V(canvas_item, false);
	ERR_FAIL_COND_V(!valid, false);

	if (!visible && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	for (int i = 0; i < collision_segments.size(); i += 2) {
		Vector2 a = collision_segments[i];
		Vector2 b = collision_segments[i + 1];
		Vector2 closest = Geometry2D::get_closest_point_to_segment(p_point, a, b);
		if (closest.distance_to(p_point) < p_max_distance) {
			return true;
		}
	}

	for (const Rect2 &collision_rect : collision_rects) {
		if (collision_rect.has_point(p_point)) {
			return true;
		}
	}

	for (const Vector<Vector2> &collision_polygon : collision_polygons) {
		if (Geometry2D::is_point_in_polygon(p_point, collision_polygon)) {
			return true;
		}
	}

	return false;
}

bool EditorCanvasItemGizmo::is_subgizmo_selected(int p_id) const {
	CanvasItemEditor *ed = CanvasItemEditor::get_singleton();
	ERR_FAIL_NULL_V(ed, false);
	return ed->is_current_selected_gizmo(this) && ed->is_subgizmo_selected(p_id);
}

Vector<int> EditorCanvasItemGizmo::get_subgizmo_selection() const {
	Vector<int> ret;

	CanvasItemEditor *ed = CanvasItemEditor::get_singleton();
	ERR_FAIL_NULL_V(ed, ret);

	if (ed->is_current_selected_gizmo(this)) {
		ret = ed->get_subgizmo_selection();
	}

	return ret;
}

void EditorCanvasItemGizmo::create() {
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(valid);
	valid = true;

	for (Instance &instance : instances) {
		instance.create_instance(canvas_item, visible);
	}

	transform();
}

void EditorCanvasItemGizmo::transform() {
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);
	for (const Instance &instance : instances) {
		RS::get_singleton()->canvas_item_set_transform(instance.instance, canvas_item->get_global_transform());
	}
}

void EditorCanvasItemGizmo::free() {
	ERR_FAIL_NULL(RS::get_singleton());
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);

	clear();
	valid = false;
}

void EditorCanvasItemGizmo::set_visible(bool p_visible) {
	visible = p_visible;
	for (const Instance &instance : instances) {
		RS::get_singleton()->canvas_item_set_visible(instance.instance, p_visible);
	}
}

void EditorCanvasItemGizmo::set_plugin(EditorCanvasItemGizmoPlugin *p_plugin) {
	gizmo_plugin = p_plugin;
}

void EditorCanvasItemGizmo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_circle", "position", "radius", "color"), &EditorCanvasItemGizmo::add_circle, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_polygon", "points", "color"), &EditorCanvasItemGizmo::add_polygon, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_polyline", "points", "color"), &EditorCanvasItemGizmo::add_polyline, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_rect", "rect", "color"), &EditorCanvasItemGizmo::add_rect, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_collision_segments", "segments"), &EditorCanvasItemGizmo::add_collision_segments);
	ClassDB::bind_method(D_METHOD("add_collision_rect", "rect"), &EditorCanvasItemGizmo::add_collision_rect);
	ClassDB::bind_method(D_METHOD("add_collision_polygon", "polygon"), &EditorCanvasItemGizmo::add_collision_polygon);
	ClassDB::bind_method(D_METHOD("add_handles", "handles", "color", "ids", "secondary"), &EditorCanvasItemGizmo::add_handles, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_canvas_item", "canvas_item"), &EditorCanvasItemGizmo::_set_canvas_item);
	ClassDB::bind_method(D_METHOD("get_canvas_item"), &EditorCanvasItemGizmo::get_canvas_item);
	ClassDB::bind_method(D_METHOD("get_plugin"), &EditorCanvasItemGizmo::get_plugin);
	ClassDB::bind_method(D_METHOD("clear"), &EditorCanvasItemGizmo::clear);
	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &EditorCanvasItemGizmo::set_visible);
	ClassDB::bind_method(D_METHOD("is_subgizmo_selected", "id"), &EditorCanvasItemGizmo::is_subgizmo_selected);
	ClassDB::bind_method(D_METHOD("get_subgizmo_selection"), &EditorCanvasItemGizmo::get_subgizmo_selection);
	ClassDB::bind_method(D_METHOD("_edit_set_state", "state"), &EditorCanvasItemGizmo::_edit_set_state);

	GDVIRTUAL_BIND(_redraw);
	GDVIRTUAL_BIND(_get_handle_name, "id", "secondary");
	GDVIRTUAL_BIND(_is_handle_highlighted, "id", "secondary");

	GDVIRTUAL_BIND(_edit_use_rect);
	GDVIRTUAL_BIND(_edit_get_rect);
	GDVIRTUAL_BIND(_edit_set_rect, "boundary");

	GDVIRTUAL_BIND(_has_pivot);
	GDVIRTUAL_BIND(_get_pivot);
	GDVIRTUAL_BIND(_set_pivot, "pivot");

	GDVIRTUAL_BIND(_edit_get_state);
	GDVIRTUAL_BIND(_edit_set_state, "state");

	GDVIRTUAL_BIND(_get_handle_value, "id", "secondary");
	GDVIRTUAL_BIND(_begin_handle_action, "id", "secondary");
	GDVIRTUAL_BIND(_set_handle, "id", "secondary", "point");
	GDVIRTUAL_BIND(_commit_handle, "id", "secondary", "restore", "cancel");

	GDVIRTUAL_BIND(_subgizmos_intersect_point, "point", "distance");
	GDVIRTUAL_BIND(_subgizmos_intersect_rect, "rect");
	GDVIRTUAL_BIND(_get_subgizmo_transform, "id");
	GDVIRTUAL_BIND(_set_subgizmo_transform, "id", "transform");
	GDVIRTUAL_BIND(_commit_subgizmos, "ids", "restores", "cancel");
}

EditorCanvasItemGizmo::EditorCanvasItemGizmo() {
	valid = false;
	visible = false;
	selected = false;
	canvas_item = nullptr;
	gizmo_plugin = nullptr;
	use_boundary_handle = false;
	use_pivot_handle = false;
}

EditorCanvasItemGizmo::~EditorCanvasItemGizmo() {
	if (gizmo_plugin != nullptr) {
		gizmo_plugin->unregister_gizmo(this);
	}
	clear();
}

/////

String EditorCanvasItemGizmoPlugin::get_gizmo_name() const {
	String ret;
	if (GDVIRTUAL_CALL(_get_gizmo_name, ret)) {
		return ret;
	}
	WARN_PRINT_ONCE("A CanvasItem editor gizmo has no name defined (it will appear as \"Unnamed Gizmo\" in the \"View > Gizmos\" menu). To resolve this, override the `_get_gizmo_name()` function to return a String in the script that extends EditorCanvasItemGizmoPlugin.");
	return "Unnamed Gizmo";
}

int EditorCanvasItemGizmoPlugin::get_priority() const {
	int ret = 0;
	if (GDVIRTUAL_CALL(_get_priority, ret)) {
		return ret;
	}
	return 0;
}

Transform2D EditorCanvasItemGizmoPlugin::boundary_change_to_transform(const Rect2 &p_before, const Rect2 &p_after) {
	Vector2 zero_offset;
	Size2 new_scale(1, 1);

	if (p_before.size.x != 0) {
		zero_offset.x = -p_before.position.x / p_before.size.x;
		new_scale.x = p_after.size.x / p_before.size.x;
	}

	if (p_before.size.y != 0) {
		zero_offset.y = -p_before.position.y / p_before.size.y;
		new_scale.y = p_after.size.y / p_before.size.y;
	}

	Point2 new_pos = p_after.position + p_after.size * zero_offset;
	return Transform2D().scaled(new_scale).translated(new_pos);
}

Ref<EditorCanvasItemGizmo> EditorCanvasItemGizmoPlugin::get_gizmo(CanvasItem *p_canvas_item) {
	if (get_script_instance() && get_script_instance()->has_method("_get_gizmo")) {
		return get_script_instance()->call("_get_gizmo", p_canvas_item);
	}

	Ref<EditorCanvasItemGizmo> ref = create_gizmo(p_canvas_item);
	if (ref.is_null()) {
		return ref;
	}

	ref->set_plugin(this);
	ref->set_canvas_item(p_canvas_item);
	ref->set_visible(gizmos_visible);

	current_gizmos.insert(ref.ptr());
	return ref;
}

void EditorCanvasItemGizmoPlugin::_bind_methods() {
	ClassDB::bind_static_method("EditorCanvasItemGizmoPlugin", D_METHOD("boundary_change_to_transform", "before", "after"), &EditorCanvasItemGizmoPlugin::boundary_change_to_transform);

	GDVIRTUAL_BIND(_has_gizmo, "for_canvas_item");
	GDVIRTUAL_BIND(_create_gizmo, "for_canvas_item");

	GDVIRTUAL_BIND(_get_gizmo_name);
	GDVIRTUAL_BIND(_get_priority);
	GDVIRTUAL_BIND(_can_be_hidden);
	GDVIRTUAL_BIND(_is_selectable_when_hidden);

	GDVIRTUAL_BIND(_redraw, "gizmo");

	GDVIRTUAL_BIND(_edit_use_rect, "gizmo");
	GDVIRTUAL_BIND(_edit_set_rect, "gizmo", "boundary");
	GDVIRTUAL_BIND(_edit_get_rect, "gizmo");

	GDVIRTUAL_BIND(_has_pivot, "gizmo");
	GDVIRTUAL_BIND(_set_pivot, "gizmo", "pivot");
	GDVIRTUAL_BIND(_get_pivot, "gizmo");

	GDVIRTUAL_BIND(_edit_get_state, "gizmo");
	GDVIRTUAL_BIND(_edit_set_state, "gizmo", "state");

	GDVIRTUAL_BIND(_get_handle_name, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_is_handle_highlighted, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_get_handle_value, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_begin_handle_action, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_set_handle, "gizmo", "handle_id", "secondary", "position");
	GDVIRTUAL_BIND(_commit_handle, "gizmo", "handle_id", "secondary", "restore", "cancel");

	GDVIRTUAL_BIND(_subgizmos_intersect_point, "gizmo", "point", "distance");
	GDVIRTUAL_BIND(_subgizmos_intersect_rect, "gizmo", "rect");
	GDVIRTUAL_BIND(_get_subgizmo_transform, "gizmo", "subgizmo_id");
	GDVIRTUAL_BIND(_set_subgizmo_transform, "gizmo", "subgizmo_id", "transform");
	GDVIRTUAL_BIND(_commit_subgizmos, "gizmo", "ids", "restores", "cancel");
}

bool EditorCanvasItemGizmoPlugin::has_gizmo(CanvasItem *p_canvas_item) {
	bool success = false;
	GDVIRTUAL_CALL(_has_gizmo, p_canvas_item, success);
	return success;
}

Ref<EditorCanvasItemGizmo> EditorCanvasItemGizmoPlugin::create_gizmo(CanvasItem *p_canvas_item) {
	Ref<EditorCanvasItemGizmo> ret;
	if (GDVIRTUAL_CALL(_create_gizmo, p_canvas_item, ret)) {
		return ret;
	}

	Ref<EditorCanvasItemGizmo> ref;
	if (has_gizmo(p_canvas_item)) {
		ref.instantiate();
	}

	return ref;
}

bool EditorCanvasItemGizmoPlugin::can_be_hidden() const {
	bool ret = true;
	GDVIRTUAL_CALL(_can_be_hidden, ret);
	return ret;
}

bool EditorCanvasItemGizmoPlugin::is_selectable_when_hidden() const {
	bool ret = false;
	GDVIRTUAL_CALL(_is_selectable_when_hidden, ret);
	return ret;
}

bool EditorCanvasItemGizmoPlugin::can_commit_handle_on_click() const {
	return false;
}

void EditorCanvasItemGizmoPlugin::redraw(EditorCanvasItemGizmo *p_gizmo) {
	GDVIRTUAL_CALL(_redraw, p_gizmo);
}

bool EditorCanvasItemGizmoPlugin::_edit_use_rect(const EditorCanvasItemGizmo *p_gizmo) const {
	ERR_FAIL_NULL_V(p_gizmo, false);
	bool ret = false;
	if (GDVIRTUAL_CALL(_edit_use_rect, Ref<EditorCanvasItemGizmo>(p_gizmo), ret)) {
		return ret;
	}
	CanvasItem *canvas_item = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL_V(canvas_item, false);
	return canvas_item->_edit_use_rect();
}

void EditorCanvasItemGizmoPlugin::_edit_set_rect(const EditorCanvasItemGizmo *p_gizmo, Rect2 p_boundary) {
	ERR_FAIL_NULL(p_gizmo);
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_set_rect)) {
		GDVIRTUAL_CALL(_edit_set_rect, Ref<EditorCanvasItemGizmo>(p_gizmo), p_boundary);
		return;
	}
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL(ci);
	ci->_edit_set_rect(p_boundary);
}

Rect2 EditorCanvasItemGizmoPlugin::_edit_get_rect(const EditorCanvasItemGizmo *p_gizmo) const {
	ERR_FAIL_NULL_V(p_gizmo, Rect2());
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_get_rect)) {
		Rect2 ret;
		GDVIRTUAL_CALL(_edit_get_rect, Ref<EditorCanvasItemGizmo>(p_gizmo), ret);
		return ret;
	}
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL_V(ci, Rect2());
	return ci->_edit_get_rect();
}

bool EditorCanvasItemGizmoPlugin::_has_pivot(const EditorCanvasItemGizmo *p_gizmo) const {
	ERR_FAIL_NULL_V(p_gizmo, false);
	if (GDVIRTUAL_IS_OVERRIDDEN(_has_pivot)) {
		bool ret = false;
		GDVIRTUAL_CALL(_has_pivot, Ref<EditorCanvasItemGizmo>(p_gizmo), ret);
		return ret;
	}

	CanvasItem *canvas_item = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL_V(canvas_item, false);
	return canvas_item->_edit_use_pivot();
}

void EditorCanvasItemGizmoPlugin::_set_pivot(const EditorCanvasItemGizmo *p_gizmo, const Vector2 &p_pivot) {
	ERR_FAIL_NULL(p_gizmo);
	if (GDVIRTUAL_IS_OVERRIDDEN(_set_pivot)) {
		GDVIRTUAL_CALL(_set_pivot, Ref<EditorCanvasItemGizmo>(p_gizmo), p_pivot);
		return;
	}
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL(ci);
	ci->_edit_set_pivot(p_pivot);
}

Point2 EditorCanvasItemGizmoPlugin::_get_pivot(const EditorCanvasItemGizmo *p_gizmo) const {
	ERR_FAIL_NULL_V(p_gizmo, Point2());
	if (GDVIRTUAL_IS_OVERRIDDEN(_get_pivot)) {
		Point2 ret;
		GDVIRTUAL_CALL(_get_pivot, Ref<EditorCanvasItemGizmo>(p_gizmo), ret);
		return ret;
	}
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL_V(ci, Point2());
	return ci->_edit_get_pivot();
}

Dictionary EditorCanvasItemGizmoPlugin::_edit_get_state(const EditorCanvasItemGizmo *p_gizmo) const {
	ERR_FAIL_NULL_V(p_gizmo, Dictionary());
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL_V(ci, Dictionary());

	Dictionary ret;
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_get_state)) {
		GDVIRTUAL_CALL(_edit_get_state, Ref<EditorCanvasItemGizmo>(p_gizmo), ret);
	}

	// similar to EditorCanvasItemGizmo::_edit_get_state, we merge the results
	Dictionary base = ci->_edit_get_state();

	if (ret.is_empty()) {
		return base;
	}

	return ret.merged(base, true);
}

void EditorCanvasItemGizmoPlugin::_edit_set_state(const EditorCanvasItemGizmo *p_gizmo, const Dictionary &p_state) {
	ERR_FAIL_NULL(p_gizmo);

	// first restore underlying canvas item state
	CanvasItem *ci = p_gizmo->get_canvas_item();
	ERR_FAIL_NULL(ci);
	ci->_edit_set_state(p_state);

	// then allow GDScript code to do their own restores on a known good state
	if (GDVIRTUAL_IS_OVERRIDDEN(_edit_set_state)) {
		GDVIRTUAL_CALL(_edit_set_state, Ref<EditorCanvasItemGizmo>(p_gizmo), p_state);
	}
}

bool EditorCanvasItemGizmoPlugin::_is_handle_highlighted(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	bool ret = false;
	GDVIRTUAL_CALL(_is_handle_highlighted, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

String EditorCanvasItemGizmoPlugin::_get_handle_name(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	String ret;
	GDVIRTUAL_CALL(_get_handle_name, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

Variant EditorCanvasItemGizmoPlugin::_get_handle_value(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	Variant ret;
	GDVIRTUAL_CALL(_get_handle_value, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

void EditorCanvasItemGizmoPlugin::_begin_handle_action(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) {
	GDVIRTUAL_CALL(_begin_handle_action, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary);
}

void EditorCanvasItemGizmoPlugin::_set_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Point2 &p_point) {
	GDVIRTUAL_CALL(_set_handle, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, p_point);
}

void EditorCanvasItemGizmoPlugin::_commit_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	GDVIRTUAL_CALL(_commit_handle, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, p_restore, p_cancel);
}

int EditorCanvasItemGizmoPlugin::_subgizmos_intersect_point(const EditorCanvasItemGizmo *p_gizmo, const Vector2 &p_point, real_t p_max_distance) const {
	int ret = -1;
	GDVIRTUAL_CALL(_subgizmos_intersect_point, Ref<EditorCanvasItemGizmo>(p_gizmo), p_point, p_max_distance, ret);
	return ret;
}

Vector<int> EditorCanvasItemGizmoPlugin::_subgizmos_intersect_rect(const EditorCanvasItemGizmo *p_gizmo, const Rect2 &p_rect) const {
	Vector<int> ret;
	GDVIRTUAL_CALL(_subgizmos_intersect_rect, Ref<EditorCanvasItemGizmo>(p_gizmo), p_rect, ret);
	return ret;
}

Transform2D EditorCanvasItemGizmoPlugin::_get_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id) const {
	Transform2D ret;
	GDVIRTUAL_CALL(_get_subgizmo_transform, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, ret);
	return ret;
}

void EditorCanvasItemGizmoPlugin::_set_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id, const Transform2D &p_xform) {
	GDVIRTUAL_CALL(_set_subgizmo_transform, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_xform);
}

void EditorCanvasItemGizmoPlugin::_commit_subgizmos(const EditorCanvasItemGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform2D> &p_transforms, bool p_cancel) {
	TypedArray<Transform2D> transforms;
	transforms.reserve(p_transforms.size());
	for (int i = 0; i < p_transforms.size(); i++) {
		transforms.append(p_transforms[i]);
	}

	GDVIRTUAL_CALL(_commit_subgizmos, Ref<EditorCanvasItemGizmo>(p_gizmo), p_ids, transforms, p_cancel);
}

void EditorCanvasItemGizmoPlugin::set_gizmos_visible(bool p_visible) {
	gizmos_visible = p_visible;
	for (EditorCanvasItemGizmo *gizmo : current_gizmos) {
		gizmo->set_visible(p_visible);
	}
}

bool EditorCanvasItemGizmoPlugin::is_gizmos_visible() const {
	return gizmos_visible;
}

void EditorCanvasItemGizmoPlugin::unregister_gizmo(EditorCanvasItemGizmo *p_gizmo) {
	current_gizmos.erase(p_gizmo);
}

EditorCanvasItemGizmoPlugin::EditorCanvasItemGizmoPlugin() {
	gizmos_visible = true;
}

EditorCanvasItemGizmoPlugin::~EditorCanvasItemGizmoPlugin() {
	for (EditorCanvasItemGizmo *gizmo : current_gizmos) {
		gizmo->set_plugin(nullptr);
		gizmo->get_canvas_item()->remove_gizmo(gizmo);
	}

	if (CanvasItemEditor::get_singleton()) {
		CanvasItemEditor::get_singleton()->update_all_gizmos();
	}
}
