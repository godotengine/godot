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
#include "editor/scene/canvas_item_editor_plugin.h"
#include "modules/gdscript/gdscript_tokenizer.h"
#include "scene/resources/mesh.h"

#define HANDLE_HALF_SIZE 9.5

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

	for (int i = 0; i < instances.size(); i++) {
		if (instances[i].instance.is_valid()) {
			RS::get_singleton()->free_rid(instances[i].instance);
			instances.write[i].instance = RID();
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
}

void EditorCanvasItemGizmo::redraw() {
	if (!GDVIRTUAL_CALL(_redraw)) {
		ERR_FAIL_NULL(gizmo_plugin);
		gizmo_plugin->redraw(this);
	}

	if (CanvasItemEditor::get_singleton()->is_current_selected_gizmo(this)) {
		CanvasItemEditor::get_singleton()->update_transform_gizmo();
	}
}

String EditorCanvasItemGizmo::get_handle_name(int p_id, bool p_secondary) const {
	String ret;
	if (GDVIRTUAL_CALL(_get_handle_name, p_id, p_secondary, ret)) {
		return ret;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, "");
	return gizmo_plugin->get_handle_name(this, p_id, p_secondary);
}

bool EditorCanvasItemGizmo::is_handle_highlighted(int p_id, bool p_secondary) const {
	bool success;
	if (GDVIRTUAL_CALL(_is_handle_highlighted, p_id, p_secondary, success)) {
		return success;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, false);
	return gizmo_plugin->is_handle_highlighted(this, p_id, p_secondary);
}

Variant EditorCanvasItemGizmo::get_handle_value(int p_id, bool p_secondary) const {
	Variant value;
	if (GDVIRTUAL_CALL(_get_handle_value, p_id, p_secondary, value)) {
		return value;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, Variant());
	return gizmo_plugin->get_handle_value(this, p_id, p_secondary);
}

void EditorCanvasItemGizmo::begin_handle_action(int p_id, bool p_secondary) {
	if (GDVIRTUAL_CALL(_begin_handle_action, p_id, p_secondary)) {
		return;
	}

	ERR_FAIL_NULL(gizmo_plugin);
	gizmo_plugin->begin_handle_action(this, p_id, p_secondary);
}

void EditorCanvasItemGizmo::set_handle(int p_id, bool p_secondary, const Point2 &p_point) {
	if (GDVIRTUAL_CALL(_set_handle, p_id, p_secondary, p_point)) {
		return;
	}

	ERR_FAIL_NULL(gizmo_plugin);
	gizmo_plugin->set_handle(this, p_id, p_secondary, p_point);
}

void EditorCanvasItemGizmo::commit_handle(int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	if (GDVIRTUAL_CALL(_commit_handle, p_id, p_secondary, p_restore, p_cancel)) {
		return;
	}

	ERR_FAIL_NULL(gizmo_plugin);
	gizmo_plugin->commit_handle(this, p_id, p_secondary, p_restore, p_cancel);
}

int EditorCanvasItemGizmo::subgizmos_intersect_point(const Point2 &p_point) const {
	int id;
	if (GDVIRTUAL_CALL(_subgizmos_intersect_point, p_point, id)) {
		return id;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, -1);
	return gizmo_plugin->subgizmos_intersect_point(this, p_point);
}

Vector<int> EditorCanvasItemGizmo::subgizmos_intersect_rect(const Rect2 &p_rect) const {
	Vector<int> ret;
	if (GDVIRTUAL_CALL(_subgizmos_intersect_rect, p_rect, ret)) {
		return ret;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, Vector<int>());
	return gizmo_plugin->subgizmos_intersect_rect(this, p_rect);
}

Transform2D EditorCanvasItemGizmo::get_subgizmo_transform(int p_id) const {
	Transform2D ret;
	if (GDVIRTUAL_CALL(_get_subgizmo_transform, p_id, ret)) {
		return ret;
	}

	ERR_FAIL_NULL_V(gizmo_plugin, Transform2D());
	return gizmo_plugin->get_subgizmo_transform(this, p_id);
}

void EditorCanvasItemGizmo::set_subgizmo_transform(int p_id, const Transform2D &p_transform) {
	if (GDVIRTUAL_CALL(_set_subgizmo_transform, p_id, p_transform)) {
		return;
	}

	ERR_FAIL_NULL(gizmo_plugin);
	gizmo_plugin->set_subgizmo_transform(this, p_id, p_transform);
}

void EditorCanvasItemGizmo::commit_subgizmos(const Vector<int> &p_ids, const Vector<Transform2D> &p_restore, bool p_cancel) {
	TypedArray<Transform2D> restore;
	restore.resize(p_restore.size());
	for (int i = 0; i < p_restore.size(); i++) {
		restore[i] = p_restore[i];
	}

	if (GDVIRTUAL_CALL(_commit_subgizmos, p_ids, restore, p_cancel)) {
		return;
	}

	ERR_FAIL_NULL(gizmo_plugin);
	gizmo_plugin->commit_subgizmos(this, p_ids, p_restore, p_cancel);
}

void EditorCanvasItemGizmo::set_canvas_item(CanvasItem *p_canvas_item) {
	ERR_FAIL_NULL(p_canvas_item);
	canvas_item = p_canvas_item;
}


void EditorCanvasItemGizmo::Instance::create_instance(CanvasItem *p_base, bool p_hidden) {
	ERR_FAIL_NULL(p_base);

	instance = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->canvas_item_set_parent(instance, p_base->get_canvas_item());
	int layer = p_hidden ? 0 : 1 << CanvasItemEditorViewport::GIZMO_EDIT_LAYER;
	RS::get_singleton()->canvas_item_set_visibility_layer(instance, layer);
}

void EditorCanvasItemGizmo::add_circle(const Vector2 &p_pos, float p_radius, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Instance ins;
	ins.create_instance(canvas_item, hidden);
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
	ins.create_instance(canvas_item, hidden);
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
	ins.create_instance(canvas_item, hidden);
	RS::get_singleton()->canvas_item_add_polyline(ins.instance, p_points, colors);
	instances.push_back(ins);
}


void EditorCanvasItemGizmo::add_rect(const Rect2 &p_rect, const Color &p_color) {
	ERR_FAIL_NULL(canvas_item);

	Instance ins;
	ins.create_instance(canvas_item, hidden);
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

void EditorCanvasItemGizmo::add_handles(const Vector<Vector2> &p_handles, const Color &p_color, const Vector<int> &p_ids, bool p_secondary) {
	if (!is_selected() || !is_editable()) {
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

	Vector<Vector3> vertices;
	vertices.resize(p_handles.size());
	for (int i = 0; i < p_handles.size(); i++) {
		vertices.write[i] = Vector3(p_handles[i].x, p_handles[i].y, 0);
	}

	Instance ins;
	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	Array a;
	a.resize(RS::ARRAY_MAX);
	a[RS::ARRAY_VERTEX] = vertices;
	Vector<Color> colors;
	{
		colors.resize(p_handles.size());
		Color *w = colors.ptrw();
		for (int i = 0; i < p_handles.size(); i++) {
			int id = p_ids.is_empty() ? i : p_ids[i];

			Color col(1, 1, 1, 1);
			if (is_handle_highlighted(id, p_secondary)) {
				col = Color(0, 0, 1, 0.9);
			}

			if (!is_current_hover_gizmo || current_hover_handle != id || p_secondary != current_hover_handle_secondary) {
				col.a = 0.8;
			}

			w[i] = col;
		}
	}
	a[RS::ARRAY_COLOR] = colors;
	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_POINTS, a);
	if (valid) {
		RS::get_singleton()->canvas_item_add_mesh(ins.instance, mesh->get_rid(), Transform2D(),  p_color);
		ins.create_instance(canvas_item, hidden);
	}

	instances.push_back(ins);

	int current_size = handle_list.size();
	handle_list.resize(current_size + p_handles.size());
	for (int i = 0; i < p_handles.size(); i++) {
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

bool EditorCanvasItemGizmo::intersect_rect(const Rect2 &p_rect) const{
	ERR_FAIL_NULL_V(canvas_item, false);
	ERR_FAIL_COND_V(!valid, false);

	if (hidden && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	Transform2D transform = canvas_item->get_global_transform();

	// for collision segments it is enough if at least one point
	// of a segment is inside the rectangle
	for (int i = 0; i < collision_segments.size(); i++) {
		Vector2 global_position = transform.xform(collision_segments[i]);
		if (p_rect.has_point(global_position)) {
			return true;
		}
	}

	// same for collision polygons
	for (int i = 0; i < collision_polygons.size(); i++) {
		for (int j = 0; j < collision_polygons[i].size(); j++) {
			Vector2 global_position = transform.xform(collision_polygons[i][j]);
			if (p_rect.has_point(global_position)) {
				return true;
			}
		}
	}

	// for rectangles we check if they overlap
	Transform2D inverse_transform = transform.affine_inverse();
	for (int i = 0; i < collision_rects.size(); i++) {
		if (collision_rects[i].intersects_transformed(inverse_transform, p_rect)) {
			return true;
		}
	}

	return false;
}

void EditorCanvasItemGizmo::handles_intersect_point(const Point2 &p_point, bool p_shift_pressed, int &r_id, bool &r_secondary) {
	r_id = -1;
	r_secondary = false;

	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);

	if (hidden) {
		return;
	}

	Transform2D screen_transform = canvas_item->get_global_transform_with_canvas() * canvas_item->get_viewport()->get_screen_transform();
	float min_d = 1e20;

	for (int i = 0; i < secondary_handles.size(); i++ ) {
		Vector2 screen_pos = screen_transform.xform(secondary_handles[i]);
		float distance = screen_pos.distance_to(p_point);
		if ( distance < HANDLE_HALF_SIZE && distance < min_d ) {
			min_d = distance;
			if (secondary_handle_ids.is_empty()) {
				r_id = i;
			}
			else {
				r_id = secondary_handle_ids[i];
			}
			r_secondary = true;
		}
	}

	if (r_id != -1 && p_shift_pressed) {
		return;
	}

	min_d = 1e20;

	for (int i = 0; i < handles.size(); i++ ) {
		Vector2 screen_pos = screen_transform.xform(handles[i]);
		float distance = screen_pos.distance_to(p_point);
		if ( distance < HANDLE_HALF_SIZE && distance < min_d ) {
			min_d = distance;
			if (handle_ids.is_empty()) {
				r_id = i;
			}
			else {
				r_id = handle_ids[i];
			}
			r_secondary = false;
		}
	}
}

bool EditorCanvasItemGizmo::intersect_point(const Point2 &p_point) const {
	ERR_FAIL_NULL_V(canvas_item, false);
	ERR_FAIL_COND_V(!valid, false);

	if (hidden && !gizmo_plugin->is_selectable_when_hidden()) {
		return false;
	}

	Transform2D to_local = canvas_item->get_global_transform().affine_inverse();
	Point2 local_point = to_local.xform(p_point);


	for (int i = 0; i < collision_segments.size(); i += 2) {
		Vector2 a = collision_segments[i];
		Vector2 b = collision_segments[i+1];
		Vector2 closest = Geometry2D::get_closest_point_to_segment(local_point, a, b);
		if (closest.distance_to(local_point) < 8) { // TODO: 3d uses a magic 8 here, not sure why
			return true;
		}
	}

	for (int i = 0; i < collision_rects.size(); i++) {
		if (collision_rects[i].has_point(local_point)) {
			return true;
		}
	}

	for (int i = 0; i < collision_polygons.size(); i++) {
		if (Geometry2D::is_point_in_polygon(local_point, collision_polygons[i])) {
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

	for (int i = 0; i < instances.size(); i++) {
		instances.write[i].create_instance(canvas_item, hidden);
	}

	transform();
}


void EditorCanvasItemGizmo::transform() {
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);
	for (int i = 0; i < instances.size(); i++) {
		RS::get_singleton()->canvas_item_set_transform(instances[i].instance, canvas_item->get_global_transform());
	}
}

void EditorCanvasItemGizmo::free() {
	ERR_FAIL_NULL(RS::get_singleton());
	ERR_FAIL_NULL(canvas_item);
	ERR_FAIL_COND(!valid);

	// TODO: i'm not sure why the 3D variant doesn't just call clear
	//  and repeats the freeing loop.
	clear();
	valid = false;
}

void EditorCanvasItemGizmo::set_hidden(bool p_hidden) {
	hidden = p_hidden;
	int layer = p_hidden ? 0 : 1 << CanvasItemEditorViewport::GIZMO_EDIT_LAYER;
	for (int i = 0; i < instances.size(); i++) {
		RS::get_singleton()->canvas_item_set_visibility_layer(instances[i].instance, layer);
	}
}

void EditorCanvasItemGizmo::set_plugin(EditorCanvasItemGizmoPlugin *p_plugin) {
	gizmo_plugin = p_plugin;
}

void EditorCanvasItemGizmo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_circle", "position", "radius", "color"), &EditorCanvasItemGizmo::add_polyline, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_polygon", "points", "color"), &EditorCanvasItemGizmo::add_polygon, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_polyline", "points", "color"), &EditorCanvasItemGizmo::add_polyline, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_rect", "rect", "color"), &EditorCanvasItemGizmo::add_rect, DEFVAL(Color(1, 1, 1)));
	ClassDB::bind_method(D_METHOD("add_collision_segments", "segments"), &EditorCanvasItemGizmo::add_collision_segments);
	ClassDB::bind_method(D_METHOD("add_collision_rect", "rect"), &EditorCanvasItemGizmo::add_collision_rect);
	ClassDB::bind_method(D_METHOD("add_collision_polygon", "polygon"), &EditorCanvasItemGizmo::add_collision_polygon);
	ClassDB::bind_method(D_METHOD("add_handles", "handles", "color", "ids", "secondary"), &EditorCanvasItemGizmo::add_handles, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_canvas_item", "canvas_item"), &EditorCanvasItemGizmo::_set_canvas_item);
	ClassDB::bind_method(D_METHOD("set_canvas_item"), &EditorCanvasItemGizmo::get_canvas_item);
	ClassDB::bind_method(D_METHOD("get_plugin"), &EditorCanvasItemGizmo::get_plugin);
	ClassDB::bind_method(D_METHOD("clear"), &EditorCanvasItemGizmo::clear);
	ClassDB::bind_method(D_METHOD("set_hidden", "hidden"), &EditorCanvasItemGizmo::set_hidden);
	ClassDB::bind_method(D_METHOD("is_subgizmo_selected", "id"), &EditorCanvasItemGizmo::is_subgizmo_selected);
	ClassDB::bind_method(D_METHOD("get_subgizmo_selection"), &EditorCanvasItemGizmo::get_subgizmo_selection);

	GDVIRTUAL_BIND(_redraw);
	GDVIRTUAL_BIND(_get_handle_name, "id", "secondary");
	GDVIRTUAL_BIND(_is_handle_highlighted, "id", "secondary");

	GDVIRTUAL_BIND(_get_handle_value, "id", "secondary");
	GDVIRTUAL_BIND(_begin_handle_action, "id", "secondary");
	GDVIRTUAL_BIND(_set_handle, "id", "secondary", "point");
	GDVIRTUAL_BIND(_commit_handle, "id", "secondary", "restore", "cancel");

	GDVIRTUAL_BIND(_subgizmos_intersect_point, "point");
	GDVIRTUAL_BIND(_subgizmos_intersect_rect, "rect");
	GDVIRTUAL_BIND(_get_subgizmo_transform, "id");
	GDVIRTUAL_BIND(_set_subgizmo_transform, "id", "transform");
	GDVIRTUAL_BIND(_commit_subgizmos, "ids", "restores", "cancel");
}

EditorCanvasItemGizmo::EditorCanvasItemGizmo() {
	valid = false;
	hidden = false;
	selected = false;
	canvas_item = nullptr;
	gizmo_plugin = nullptr;
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
	WARN_PRINT_ONCE("A CanvasItem editor gizmo has no name defined (it will appear as \"Unnamed Gizmo\"  in the \"View > Gizmos\" menu). To resolve this, override the `_get_gizmo_name()` function to return a String in the script that extends EditorCanvasItemGizmoPlugin.");
	return "Unnamed Gizmo";
}


int EditorCanvasItemGizmoPlugin::get_priority() const {
	int ret = 0;
	if (GDVIRTUAL_CALL(_get_priority, ret)) {
		return ret;
	}
	return 0;
}

Ref<EditorCanvasItemGizmo> EditorCanvasItemGizmoPlugin::get_gizmo(CanvasItem *p_canvas_item) {
	if (get_script_instance() && get_script_instance() -> has_method("_get_gizmo")) {
		return get_script_instance()->call("_get_gizmo", p_canvas_item);
	}

	Ref<EditorCanvasItemGizmo> ref = create_gizmo(p_canvas_item);
	if (ref.is_null()) {
		return ref;
	}

	ref->set_plugin(this);
	ref->set_canvas_item(p_canvas_item);
	ref->set_hidden(current_state == HIDDEN);

	current_gizmos.insert(ref.ptr());
	return ref;
}

void EditorCanvasItemGizmoPlugin::_bind_methods() {
	GDVIRTUAL_BIND(_has_gizmo, "for_canvas_item");
	GDVIRTUAL_BIND(_create_gizmo, "for_canvas_item");

	GDVIRTUAL_BIND(_get_gizmo_name);
	GDVIRTUAL_BIND(_get_priority);
	GDVIRTUAL_BIND(_can_be_hidden);
	GDVIRTUAL_BIND(_is_selectable_when_hidden);

	GDVIRTUAL_BIND(_redraw, "gizmo");
	GDVIRTUAL_BIND(_get_handle_name, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_is_handle_highlighted, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_get_handle_value, "gizmo", "handle_id", "secondary");

	GDVIRTUAL_BIND(_begin_handle_action, "gizmo", "handle_id", "secondary");
	GDVIRTUAL_BIND(_set_handle, "gizmo", "handle_id", "secondary", "screen_pos");
	GDVIRTUAL_BIND(_commit_handle, "gizmo", "handle_id", "secondary", "restore", "cancel");

	GDVIRTUAL_BIND(_subgizmos_intersect_point, "gizmo", "screen_pos");
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

bool EditorCanvasItemGizmoPlugin::is_handle_highlighted(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	bool ret = false;
	GDVIRTUAL_CALL(_is_handle_highlighted, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

String EditorCanvasItemGizmoPlugin::get_handle_name(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	String ret;
	GDVIRTUAL_CALL(_get_handle_name, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

Variant EditorCanvasItemGizmoPlugin::get_handle_value(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) const {
	Variant ret;
	GDVIRTUAL_CALL(_get_handle_value, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, ret);
	return ret;
}

void EditorCanvasItemGizmoPlugin::begin_handle_action(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary) {
	GDVIRTUAL_CALL(_begin_handle_action, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary);
}

void EditorCanvasItemGizmoPlugin::set_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Point2 &p_screen_pos) {
	GDVIRTUAL_CALL(_set_handle, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, p_screen_pos);
}

void EditorCanvasItemGizmoPlugin::commit_handle(const EditorCanvasItemGizmo *p_gizmo, int p_id, bool p_secondary, const Variant& p_restore, bool p_cancel) {
	GDVIRTUAL_CALL(_commit_handle, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_secondary, p_restore, p_cancel);
}

int EditorCanvasItemGizmoPlugin::subgizmos_intersect_point(const EditorCanvasItemGizmo *p_gizmo, const Vector2 &p_point) const {
	int ret = -1;
	GDVIRTUAL_CALL(_subgizmos_intersect_point, Ref<EditorCanvasItemGizmo>(p_gizmo), p_point, ret);
	return ret;
}

Vector<int> EditorCanvasItemGizmoPlugin::subgizmos_intersect_rect(const EditorCanvasItemGizmo *p_gizmo, const Rect2 &p_rect) const {
	Vector<int> ret;
	GDVIRTUAL_CALL(_subgizmos_intersect_rect, Ref<EditorCanvasItemGizmo>(p_gizmo), p_rect, ret);
	return ret;
}

Transform2D EditorCanvasItemGizmoPlugin::get_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id) const {
	Transform2D ret;
	GDVIRTUAL_CALL(_get_subgizmo_transform, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, ret);
	return ret;
}

void EditorCanvasItemGizmoPlugin::set_subgizmo_transform(const EditorCanvasItemGizmo *p_gizmo, int p_id, const Transform2D &p_xform) {
	GDVIRTUAL_CALL(_set_subgizmo_transform, Ref<EditorCanvasItemGizmo>(p_gizmo), p_id, p_xform);
}

void EditorCanvasItemGizmoPlugin::commit_subgizmos(const EditorCanvasItemGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform2D> &p_transforms, bool p_cancel) {
	TypedArray<Transform2D> transforms;
	transforms.reserve(p_transforms.size());
	for (int i = 0; i < p_transforms.size(); i++) {
		transforms[i] = p_transforms[i];
	}

	GDVIRTUAL_CALL(_commit_subgizmos, Ref<EditorCanvasItemGizmo>(p_gizmo), p_ids, transforms, p_cancel);
}

void EditorCanvasItemGizmoPlugin::set_state(int p_state) {
	current_state = p_state;
	for (EditorCanvasItemGizmo *gizmo : current_gizmos) {
		gizmo->set_hidden(p_state == HIDDEN);
	}
}

int EditorCanvasItemGizmoPlugin::get_state() const {
	return current_state;
}

void EditorCanvasItemGizmoPlugin::unregister_gizmo(EditorCanvasItemGizmo *p_gizmo) {
	current_gizmos.erase(p_gizmo);
}

EditorCanvasItemGizmoPlugin::EditorCanvasItemGizmoPlugin() {
	current_state = VISIBLE;
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



















