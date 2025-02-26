/**************************************************************************/
/*  navigation_mesh_area_3d_editor_plugin.cpp                             */
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

#include "navigation_mesh_area_3d_editor_plugin.h"

#include "core/math/geometry_2d.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/gizmos/gizmo_3d_helper.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/navigation/navigation_mesh_area_3d.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "servers/navigation_server_3d.h"

NavigationMeshArea3DGizmoPlugin::NavigationMeshArea3DGizmoPlugin() {
	current_state = VISIBLE;

	create_icon_material("navigation_mesh_area_box_3d_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoNavigationMeshAreaBox3D"), EditorStringName(EditorIcons)));
	create_icon_material("navigation_mesh_area_cylinder_3d_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoNavigationMeshAreaCylinder3D"), EditorStringName(EditorIcons)));
	create_icon_material("navigation_mesh_area_polygon_3d_icon", EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("GizmoNavigationMeshAreaPolygon3D"), EditorStringName(EditorIcons)));

	helper.instantiate();

	create_handle_material("handles");
}

bool NavigationMeshArea3DGizmoPlugin ::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<NavigationMeshArea3D>(p_spatial) != nullptr;
}

String NavigationMeshArea3DGizmoPlugin ::get_gizmo_name() const {
	return "NavigationMeshArea3D";
}

bool NavigationMeshArea3DGizmoPlugin ::can_be_hidden() const {
	return true;
}

int NavigationMeshArea3DGizmoPlugin ::get_priority() const {
	return -1;
}

String NavigationMeshArea3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());
	if (!area_node) {
		return "";
	}

	NavigationMeshAreaBox3D *area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_gizmo->get_node_3d());
	if (area_box) {
		return helper->box_get_handle_name(p_id);
	}

	NavigationMeshAreaCylinder3D *area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_gizmo->get_node_3d());
	if (area_cylinder) {
		return helper->cylinder_get_handle_name(p_id);
	}

	return "";
}

Variant NavigationMeshArea3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());
	if (!area_node) {
		return Variant();
	}

	NavigationMeshAreaBox3D *area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_gizmo->get_node_3d());
	if (area_box) {
		return area_box->get_size();
	}

	NavigationMeshAreaCylinder3D *area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_gizmo->get_node_3d());
	if (area_cylinder) {
		return p_id == 0 ? area_cylinder->get_radius() : area_cylinder->get_height();
	}

	return Variant();
}

void NavigationMeshArea3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	if (Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d())) {
		return;
	}

	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());

	//const Vector3 safe_scale = area_node->get_global_basis().get_scale().abs().maxf(0.001);
	//const Transform3D gt = Transform3D(Basis().scaled(safe_scale), Vector3());
	//Transform3D gi = gt.affine_inverse();
	//Transform3D gt = Transform3D(area_node->get_global_basis().inverse(), area_node->get_global_position());
	//Transform3D gt = area_node->get_global_transform();
	const Transform3D gbi = Transform3D(area_node->get_global_basis().inverse(), area_node->get_global_position());

	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), gbi);
}

void NavigationMeshArea3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());
	if (!area_node) {
		return;
	}

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	NavigationMeshAreaBox3D *area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_gizmo->get_node_3d());
	if (area_box) {
		Vector3 size = area_box->get_size();
		Vector3 position;
		helper->box_set_handle(sg, p_id, size, position);
		area_box->set_size(size);
		area_box->set_global_position(position);
		return;
	}

	NavigationMeshAreaCylinder3D *area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_gizmo->get_node_3d());
	if (area_cylinder) {
		real_t height = area_cylinder->get_height();
		real_t radius = area_cylinder->get_radius();
		Vector3 position;
		helper->cylinder_set_handle(sg, p_id, height, radius, position);
		area_cylinder->set_height(height);
		area_cylinder->set_radius(radius);
		area_cylinder->set_global_position(position);
		return;
	}
}

void NavigationMeshArea3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());
	if (!area_node) {
		return;
	}

	NavigationMeshAreaBox3D *area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_gizmo->get_node_3d());
	if (area_box) {
		helper->box_commit_handle(TTR("Change NavigationMeshAreaBox3D Size"), p_cancel, area_box);
		return;
	}

	NavigationMeshAreaCylinder3D *area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_gizmo->get_node_3d());
	if (area_cylinder) {
		helper->cylinder_commit_handle(p_id, TTR("Change NavigationMeshAreaCylinder3D Radius"), TTR("Change NavigationMeshAreaCylinder3D Height"), p_cancel, area_cylinder, area_cylinder, area_cylinder);
		return;
	}
}

void NavigationMeshArea3DGizmoPlugin ::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	if (!p_gizmo->is_selected() && get_state() == HIDDEN) {
		return;
	}

	NavigationMeshArea3D *area_node = Object::cast_to<NavigationMeshArea3D>(p_gizmo->get_node_3d());
	if (!area_node) {
		return;
	}

	NavigationServer3D *ns3d = NavigationServer3D::get_singleton();

	NavigationMeshAreaBox3D *area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_gizmo->get_node_3d());
	if (area_box) {
		const Vector3 safe_scale = area_box->get_global_basis().get_scale().abs().maxf(0.001);
		const Basis safe_basis = Basis().scaled(safe_scale);
		const Transform3D gbi = Transform3D(area_box->get_global_basis().inverse() * safe_basis, Vector3());

		const Ref<Material> icon = get_material("navigation_mesh_area_box_3d_icon", p_gizmo);
		p_gizmo->add_unscaled_billboard(icon, 0.05);

		Vector<Vector3> lines;
		AABB aabb;
		aabb.position = -area_box->get_size() / 2;
		aabb.size = area_box->get_size();

		for (int i = 0; i < 12; i++) {
			Vector3 a, b;
			aabb.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		lines = gbi.xform(lines);

		const Vector<Vector3> handles = helper->box_get_handles(area_box->get_size());

		const Ref<StandardMaterial3D> material = area_box->is_enabled() ? ns3d->get_debug_area_edge_material() : ns3d->get_debug_area_edge_disabled_material();

		p_gizmo->add_lines(lines, material, false);
		p_gizmo->add_collision_segments(lines);

		const Ref<Material> handles_material = get_material("handles");

		Vector<Vector3> xformed_handles;
		xformed_handles.resize(handles.size());
		Vector3 *xformed_handles_ptrw = xformed_handles.ptrw();
		const Vector3 *handles_ptr = handles.ptr();

		for (int i = 0; i < handles.size(); i++) {
			xformed_handles_ptrw[i] = gbi.xform(handles_ptr[i]);
		}
		p_gizmo->add_handles(xformed_handles, handles_material);
	}

	NavigationMeshAreaCylinder3D *area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_gizmo->get_node_3d());
	if (area_cylinder) {
		Vector3 safe_scale = area_cylinder->get_global_basis().get_scale().abs().maxf(0.001);
		if (safe_scale.x > safe_scale.z) {
			safe_scale.z = safe_scale.x;
		}
		if (safe_scale.z > safe_scale.x) {
			safe_scale.x = safe_scale.z;
		}
		const Basis safe_basis = Basis().scaled(safe_scale);
		const Transform3D gti = Transform3D(area_cylinder->get_global_basis().inverse() * safe_basis, Vector3());

		const Ref<Material> icon = get_material("navigation_mesh_area_cylinder_3d_icon", p_gizmo);
		p_gizmo->add_unscaled_billboard(icon, 0.05);

		float radius = area_cylinder->get_radius();
		float height = area_cylinder->get_height();

		Vector<Vector3> points;

		Vector3 d(0, height * 0.5, 0);
		for (int i = 0; i < 360; i++) {
			float ra = Math::deg_to_rad((float)i);
			float rb = Math::deg_to_rad((float)i + 1);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			points.push_back(Vector3(a.x, 0, a.y) + d);
			points.push_back(Vector3(b.x, 0, b.y) + d);

			points.push_back(Vector3(a.x, 0, a.y) - d);
			points.push_back(Vector3(b.x, 0, b.y) - d);

			if (i % 90 == 0) {
				points.push_back(Vector3(a.x, 0, a.y) + d);
				points.push_back(Vector3(a.x, 0, a.y) - d);
			}
		}

		const Ref<StandardMaterial3D> material = area_cylinder->is_enabled() ? ns3d->get_debug_area_edge_material() : ns3d->get_debug_area_edge_disabled_material();
		p_gizmo->add_lines(gti.xform(points), material, false);

		Vector<Vector3> collision_segments;

		for (int i = 0; i < 64; i++) {
			float ra = i * (Math_TAU / 64.0);
			float rb = (i + 1) * (Math_TAU / 64.0);
			Point2 a = Vector2(Math::sin(ra), Math::cos(ra)) * radius;
			Point2 b = Vector2(Math::sin(rb), Math::cos(rb)) * radius;

			collision_segments.push_back(Vector3(a.x, 0, a.y) + d);
			collision_segments.push_back(Vector3(b.x, 0, b.y) + d);

			collision_segments.push_back(Vector3(a.x, 0, a.y) - d);
			collision_segments.push_back(Vector3(b.x, 0, b.y) - d);

			if (i % 16 == 0) {
				collision_segments.push_back(Vector3(a.x, 0, a.y) + d);
				collision_segments.push_back(Vector3(a.x, 0, a.y) - d);
			}
		}

		p_gizmo->add_collision_segments(gti.xform(collision_segments));

		const Vector<Vector3> handles = helper->cylinder_get_handles(area_cylinder->get_height(), area_cylinder->get_radius());

		const Ref<Material> handles_material = get_material("handles");

		Vector<Vector3> xformed_handles;
		xformed_handles.resize(handles.size());
		Vector3 *xformed_handles_ptrw = xformed_handles.ptrw();
		const Vector3 *handles_ptr = handles.ptr();

		for (int i = 0; i < handles.size(); i++) {
			xformed_handles_ptrw[i] = gti.xform(handles_ptr[i]);
		}
		p_gizmo->add_handles(xformed_handles, handles_material);
	}

	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (area_polygon) {
		const Ref<Material> icon = get_material("navigation_mesh_area_polygon_3d_icon", p_gizmo);
		p_gizmo->add_unscaled_billboard(icon, 0.05);

		const Vector<Vector3> &vertices = area_polygon->get_vertices();
		if (vertices.is_empty()) {
			return;
		}

		float height = area_polygon->get_height();

		const Basis safe_basis = Basis(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y, area_polygon->get_global_basis().get_scale().abs().maxf(0.001));
		const Basis gbi = area_polygon->get_global_basis().inverse();
		const Basis safe_global_basis = gbi * safe_basis;

		const int vertex_count = vertices.size();

		Vector<Vector3> lines_mesh_vertices;
		lines_mesh_vertices.resize(vertex_count * 6);
		Vector3 *lines_mesh_vertices_ptrw = lines_mesh_vertices.ptrw();

		int vertex_index = 0;

		for (int i = 0; i < vertex_count; i++) {
			Vector3 point = vertices[i];
			Vector3 next_point = vertices[(i + 1) % vertex_count];

			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(point);
			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(next_point);

			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(Vector3(point.x, height, point.z));
			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(Vector3(next_point.x, height, next_point.z));

			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(point);
			lines_mesh_vertices_ptrw[vertex_index++] = safe_global_basis.xform(Vector3(point.x, height, point.z));
		}

		if (area_polygon->are_vertices_valid() && area_polygon->is_enabled()) {
			p_gizmo->add_lines(lines_mesh_vertices, ns3d->get_debug_area_edge_material());
		} else if (area_polygon->are_vertices_valid() && !area_polygon->is_enabled()) {
			p_gizmo->add_lines(lines_mesh_vertices, ns3d->get_debug_area_edge_disabled_material());
		} else {
			p_gizmo->add_lines(lines_mesh_vertices, ns3d->get_debug_area_edge_invalid_material());
		}
		p_gizmo->add_collision_segments(lines_mesh_vertices);
	}

	if (p_gizmo->is_selected()) {
		NavigationMeshArea3DEditorPlugin::singleton->redraw();
	}
}

int NavigationMeshArea3DGizmoPlugin ::subgizmos_intersect_ray(const EditorNode3DGizmo *p_gizmo, Camera3D *p_camera, const Vector2 &p_point) const {
	if (NavigationMeshArea3DEditorPlugin::singleton->get_mode() != 1) { // MODE_EDIT
		return -1;
	}

	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (!area_polygon) {
		return -1;
	}

	const Vector3 safe_scale = area_polygon->get_global_basis().get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y), area_polygon->get_global_position());
	const Vector<Vector3> &vertices = area_polygon->get_vertices();

	for (int idx = 0; idx < vertices.size(); ++idx) {
		Vector3 pos = gt.xform(vertices[idx]);
		if (p_camera->unproject_position(pos).distance_to(p_point) < 20) {
			return idx;
		}
	}

	return -1;
}

Vector<int> NavigationMeshArea3DGizmoPlugin ::subgizmos_intersect_frustum(const EditorNode3DGizmo *p_gizmo, const Camera3D *p_camera, const Vector<Plane> &p_frustum) const {
	Vector<int> contained_points;

	if (NavigationMeshArea3DEditorPlugin::singleton->get_mode() != 1) { // MODE_EDIT
		return contained_points;
	}

	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (!area_polygon) {
		return contained_points;
	}

	const Vector3 safe_scale = area_polygon->get_global_basis().get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y), area_polygon->get_global_position());
	const Vector<Vector3> &vertices = area_polygon->get_vertices();

	for (int idx = 0; idx < vertices.size(); ++idx) {
		Vector3 pos = gt.xform(vertices[idx]);
		bool is_contained_in_frustum = true;
		for (int i = 0; i < p_frustum.size(); ++i) {
			if (p_frustum[i].distance_to(pos) > 0) {
				is_contained_in_frustum = false;
				break;
			}
		}

		if (is_contained_in_frustum) {
			contained_points.push_back(idx);
		}
	}

	return contained_points;
}

Transform3D NavigationMeshArea3DGizmoPlugin ::get_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id) const {
	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (!area_polygon) {
		return Transform3D();
	}

	const Vector<Vector3> &vertices = area_polygon->get_vertices();
	ERR_FAIL_INDEX_V(p_id, vertices.size(), Transform3D());

	const Basis safe_basis_inverse = Basis(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y, area_polygon->get_global_basis().get_scale().abs().maxf(0.001)).inverse();
	Transform3D subgizmo_transform = Transform3D(Basis(), safe_basis_inverse.xform(vertices[p_id]));
	return subgizmo_transform;
}

void NavigationMeshArea3DGizmoPlugin ::set_subgizmo_transform(const EditorNode3DGizmo *p_gizmo, int p_id, Transform3D p_transform) {
	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (!area_polygon) {
		return;
	}

	const Basis safe_basis = Basis(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y, area_polygon->get_global_basis().get_scale().abs().maxf(0.001));
	Vector3 new_vertex_pos = p_transform.origin;

	Vector<Vector3> vertices = area_polygon->get_vertices();
	ERR_FAIL_INDEX(p_id, vertices.size());

	Vector3 vertex = safe_basis.xform(new_vertex_pos);
	vertex.y = 0.0;
	vertices.write[p_id] = vertex;

	area_polygon->set_vertices(vertices);
}

void NavigationMeshArea3DGizmoPlugin ::commit_subgizmos(const EditorNode3DGizmo *p_gizmo, const Vector<int> &p_ids, const Vector<Transform3D> &p_restore, bool p_cancel) {
	NavigationMeshAreaPolygon3D *area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_gizmo->get_node_3d());
	if (!area_polygon) {
		return;
	}

	const Basis safe_basis = Basis(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y, area_polygon->get_global_basis().get_scale().abs().maxf(0.001));

	Vector<Vector3> vertices = area_polygon->get_vertices();
	Vector<Vector3> restore_vertices = vertices;

	for (int i = 0; i < p_ids.size(); ++i) {
		const int idx = p_ids[i];
		Vector3 vertex = safe_basis.xform(p_restore[i].origin);
		vertex.y = 0.0;
		restore_vertices.write[idx] = vertex;
	}

	if (p_cancel) {
		area_polygon->set_vertices(restore_vertices);
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Set NavigationMeshAreaPolygon3D Vertices"));
	undo_redo->add_do_method(area_polygon, "set_vertices", vertices);
	undo_redo->add_undo_method(area_polygon, "set_vertices", restore_vertices);
	undo_redo->commit_action();
}

void NavigationMeshArea3DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_theme();
		} break;

		case NOTIFICATION_READY: {
			_update_theme();
			button_edit->set_pressed(true);
			get_tree()->connect("node_removed", callable_mp(this, &NavigationMeshArea3DEditorPlugin::_node_removed));
			EditorNode::get_singleton()->get_gui_base()->connect(SceneStringName(theme_changed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::_update_theme));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", callable_mp(this, &NavigationMeshArea3DEditorPlugin::_node_removed));
			EditorNode::get_singleton()->get_gui_base()->disconnect(SceneStringName(theme_changed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::_update_theme));
		} break;
	}
}

void NavigationMeshArea3DEditorPlugin::edit(Object *p_object) {
	RenderingServer *rs = RenderingServer::get_singleton();

	if (!Object::cast_to<NavigationMeshArea3D>(p_object)) {
		area_box = nullptr;
		area_cylinder = nullptr;
		area_polygon = nullptr;
		obstacle_editor->hide();
		rs->mesh_clear(point_lines_mesh_rid);
		rs->mesh_clear(point_handle_mesh_rid);
		rs->instance_set_scenario(point_lines_instance_rid, RID());
		rs->instance_set_scenario(point_handles_instance_rid, RID());
		return;
	}

	area_box = Object::cast_to<NavigationMeshAreaBox3D>(p_object);
	area_cylinder = Object::cast_to<NavigationMeshAreaCylinder3D>(p_object);
	area_polygon = Object::cast_to<NavigationMeshAreaPolygon3D>(p_object);

	if (!area_polygon) {
		obstacle_editor->hide();
		rs->mesh_clear(point_lines_mesh_rid);
		rs->mesh_clear(point_handle_mesh_rid);
		rs->instance_set_scenario(point_lines_instance_rid, RID());
		rs->instance_set_scenario(point_handles_instance_rid, RID());
	}

	if (area_box) {
		area_cylinder = nullptr;
		area_polygon = nullptr;
	} else if (area_cylinder) {
		area_box = nullptr;
		area_polygon = nullptr;
	} else if (area_polygon) {
		area_box = nullptr;
		area_cylinder = nullptr;
		obstacle_editor->show();
		if (area_polygon->get_vertices().is_empty()) {
			set_mode(MODE_CREATE);
		} else {
			set_mode(MODE_EDIT);
		}
		wip_vertices.clear();
		wip_active = false;
		edited_point = -1;
		rs->instance_set_scenario(point_lines_instance_rid, area_polygon->get_world_3d()->get_scenario());
		rs->instance_set_scenario(point_handles_instance_rid, area_polygon->get_world_3d()->get_scenario());
	}

	redraw();
}

bool NavigationMeshArea3DEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<NavigationMeshAreaPolygon3D>(p_object);
}

void NavigationMeshArea3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		obstacle_editor->show();
	} else {
		obstacle_editor->hide();
		edit(nullptr);
	}
}

void NavigationMeshArea3DEditorPlugin::action_clear_vertices() {
	if (!area_polygon) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Edit NavigationMeshAreaPolygon3D (Clear Vertices)"));
	undo_redo->add_do_method(area_polygon, "set_vertices", Vector<Vector3>());
	undo_redo->add_undo_method(area_polygon, "set_vertices", area_polygon->get_vertices());
	undo_redo->commit_action();

	area_polygon->update_gizmos();
	edit(area_polygon);
}

void NavigationMeshArea3DEditorPlugin::_update_theme() {
	button_create->set_tooltip_text(TTR("Add Vertex"));
	button_edit->set_tooltip_text(TTR("Edit Vertex"));
	button_delete->set_tooltip_text(TTR("Delete Vertex"));
	button_clear->set_tooltip_text(TTR("Clear Vertices"));
	button_create->set_button_icon(button_create->get_editor_theme_icon(SNAME("CurveCreate")));
	button_edit->set_button_icon(button_edit->get_editor_theme_icon(SNAME("CurveEdit")));
	button_delete->set_button_icon(button_delete->get_editor_theme_icon(SNAME("CurveDelete")));
	button_clear->set_button_icon(button_clear->get_editor_theme_icon(SNAME("Clear")));
}

void NavigationMeshArea3DEditorPlugin::_node_removed(Node *p_node) {
	if (area_polygon == p_node) {
		area_polygon = nullptr;

		RenderingServer *rs = RenderingServer::get_singleton();
		rs->mesh_clear(point_lines_mesh_rid);
		rs->mesh_clear(point_handle_mesh_rid);

		obstacle_editor->hide();
	}
}

void NavigationMeshArea3DEditorPlugin::set_mode(int p_option) {
	if (p_option == NavigationMeshArea3DEditorPlugin::ACTION_CLEAR) {
		button_clear->set_pressed(false);
		if (area_polygon) {
			button_clear_dialog->reset_size();
			button_clear_dialog->popup_centered();
		}
		return;
	}

	mode = p_option;

	button_create->set_pressed(p_option == NavigationMeshArea3DEditorPlugin::MODE_CREATE);
	button_edit->set_pressed(p_option == NavigationMeshArea3DEditorPlugin::MODE_EDIT);
	button_delete->set_pressed(p_option == NavigationMeshArea3DEditorPlugin::MODE_DELETE);
	button_clear->set_pressed(false);
}

void NavigationMeshArea3DEditorPlugin::_wip_cancel() {
	wip_vertices.clear();
	wip_active = false;

	edited_point = -1;

	redraw();
}

void NavigationMeshArea3DEditorPlugin::_wip_close() {
	ERR_FAIL_NULL_MSG(area_polygon, "Edited NavigationMeshAreaPolygon3D is not valid.");

	Vector<Vector2> wip_2d_vertices;
	wip_2d_vertices.resize(wip_vertices.size());
	for (int i = 0; i < wip_vertices.size(); i++) {
		const Vector3 &vert = wip_vertices[i];
		wip_2d_vertices.write[i] = Vector2(vert.x, vert.z);
	}
	Vector<int> triangulated_polygon_2d_indices = Geometry2D::triangulate_polygon(wip_2d_vertices);

	if (!triangulated_polygon_2d_indices.is_empty()) {
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();

		undo_redo->create_action(TTR("Set NavigationMeshAreaPolygon3D Vertices"));
		undo_redo->add_do_method(area_polygon, "set_vertices", wip_vertices);
		undo_redo->add_undo_method(area_polygon, "set_vertices", area_polygon->get_vertices());
		undo_redo->commit_action();

		wip_vertices.clear();
		wip_active = false;
		//mode = MODE_EDIT;
		NavigationMeshArea3DEditorPlugin::singleton->set_mode(NavigationMeshArea3DEditorPlugin::MODE_EDIT);
		button_edit->set_pressed(true);
		button_create->set_pressed(false);
		edited_point = -1;
	}
}

EditorPlugin::AfterGUIInput NavigationMeshArea3DEditorPlugin::forward_3d_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!area_polygon) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	if (!area_polygon->is_visible_in_tree()) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	Ref<InputEventMouse> mouse_event = p_event;

	if (mouse_event.is_null()) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		Vector2 mouse_position = mb->get_position();
		Vector3 ray_from = p_camera->project_ray_origin(mouse_position);
		Vector3 ray_dir = p_camera->project_ray_normal(mouse_position);

		const Vector3 safe_scale = area_polygon->get_global_basis().get_scale().abs().maxf(0.001);
		const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y), area_polygon->get_global_position());
		Transform3D gi = gt.affine_inverse();
		Plane projection_plane(Vector3(0.0, 1.0, 0.0), gt.origin);

		Vector3 spoint;

		if (!projection_plane.intersects_ray(ray_from, ray_dir, &spoint)) {
			return EditorPlugin::AFTER_GUI_INPUT_PASS;
		}

		spoint = gi.xform(spoint);

		Vector3 cpoint = Vector3(spoint.x, 0.0, spoint.z);
		Vector<Vector3> obstacle_vertices = area_polygon->get_vertices();

		real_t grab_threshold = EDITOR_GET("editors/polygon_editor/point_grab_radius");

		switch (mode) {
			case MODE_CREATE: {
				if (mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
					if (obstacle_vertices.size() >= 3) {
						int closest_idx = -1;
						Vector2 closest_edge_point;
						real_t closest_dist = 1e10;
						for (int i = 0; i < obstacle_vertices.size(); i++) {
							Vector2 points[2] = {
								p_camera->unproject_position(gt.xform(obstacle_vertices[i])),
								p_camera->unproject_position(gt.xform(obstacle_vertices[(i + 1) % obstacle_vertices.size()]))
							};

							Vector2 cp = Geometry2D::get_closest_point_to_segment(mouse_position, points);
							if (cp.distance_squared_to(points[0]) < grab_threshold || cp.distance_squared_to(points[1]) < grab_threshold) {
								continue; // Skip edge as clicked point is too close to existing vertex.
							}

							real_t d = cp.distance_to(mouse_position);
							if (d < closest_dist && d < grab_threshold) {
								closest_dist = d;
								closest_edge_point = cp;
								closest_idx = i;
							}
						}
						if (closest_idx >= 0) {
							edited_point = -1;
							Vector3 _ray_from = p_camera->project_ray_origin(closest_edge_point);
							Vector3 _ray_dir = p_camera->project_ray_normal(closest_edge_point);
							Vector3 edge_intersection_point;
							if (projection_plane.intersects_ray(_ray_from, _ray_dir, &edge_intersection_point)) {
								edge_intersection_point = gi.xform(edge_intersection_point);

								EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
								undo_redo->create_action(TTR("Edit Obstacle (Add Vertex)"));
								undo_redo->add_undo_method(area_polygon, "set_vertices", obstacle_vertices);
								obstacle_vertices.insert(closest_idx + 1, edge_intersection_point);
								undo_redo->add_do_method(area_polygon, "set_vertices", obstacle_vertices);
								undo_redo->commit_action();
								redraw();
								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}
						}
					}
					if (!wip_active) {
						wip_vertices.clear();
						wip_vertices.push_back(cpoint);
						wip_active = true;
						edited_point_pos = cpoint;
						snap_ignore = false;
						redraw();
						edited_point = 1;
						return EditorPlugin::AFTER_GUI_INPUT_STOP;
					} else {
						if (wip_vertices.size() > 1 && p_camera->unproject_position(gt.xform(wip_vertices[0])).distance_to(mouse_position) < grab_threshold) {
							_wip_close();

							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						} else {
							wip_vertices.push_back(cpoint);
							edited_point = wip_vertices.size();
							snap_ignore = false;
							redraw();
							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						}
					}
				} else if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed() && wip_active) {
					_wip_close();
				}

			} break;

			case MODE_EDIT: {
				if (mb->get_button_index() == MouseButton::LEFT) {
					if (mb->is_pressed()) {
						if (mb->is_ctrl_pressed()) {
							if (obstacle_vertices.size() < 3) {
								EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
								undo_redo->create_action(TTR("Edit Obstacle (Add Vertex)"));
								undo_redo->add_undo_method(area_polygon, "set_vertices", area_polygon->get_vertices());
								obstacle_vertices.push_back(cpoint);
								undo_redo->commit_action();
								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}

							//search edges
							int closest_idx = -1;
							Vector2 closest_pos;
							real_t closest_dist = 1e10;
							for (int i = 0; i < obstacle_vertices.size(); i++) {
								Vector2 points[2] = {
									p_camera->unproject_position(gt.xform(obstacle_vertices[i])),
									p_camera->unproject_position(gt.xform(obstacle_vertices[(i + 1) % obstacle_vertices.size()]))
								};

								Vector2 cp = Geometry2D::get_closest_point_to_segment(mouse_position, points);
								if (cp.distance_squared_to(points[0]) < CMP_EPSILON2 || cp.distance_squared_to(points[1]) < CMP_EPSILON2) {
									continue; //not valid to reuse point
								}

								real_t d = cp.distance_to(mouse_position);
								if (d < closest_dist && d < grab_threshold) {
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}

							if (closest_idx >= 0) {
								pre_move_edit = obstacle_vertices;
								obstacle_vertices.insert(closest_idx + 1, cpoint);
								edited_point = closest_idx + 1;
								edited_point_pos = cpoint;
								area_polygon->set_vertices(obstacle_vertices);
								redraw();
								snap_ignore = true;

								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}
						} else {
							int closest_idx = -1;
							Vector2 closest_pos;
							real_t closest_dist = 1e10;
							for (int i = 0; i < obstacle_vertices.size(); i++) {
								Vector2 cp = p_camera->unproject_position(gt.xform(obstacle_vertices[i]));

								real_t d = cp.distance_to(mouse_position);
								if (d < closest_dist && d < grab_threshold) {
									closest_dist = d;
									closest_pos = cp;
									closest_idx = i;
								}
							}

							if (closest_idx >= 0) {
								pre_move_edit = obstacle_vertices;
								edited_point = closest_idx;
								edited_point_pos = obstacle_vertices[closest_idx];
								redraw();
								snap_ignore = false;
								return EditorPlugin::AFTER_GUI_INPUT_STOP;
							}
						}
					} else {
						snap_ignore = false;

						if (edited_point != -1) {
							ERR_FAIL_INDEX_V(edited_point, obstacle_vertices.size(), EditorPlugin::AFTER_GUI_INPUT_PASS);
							obstacle_vertices.write[edited_point] = edited_point_pos;

							EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
							undo_redo->create_action(TTR("Edit Obstacle (Move Vertex)"));
							undo_redo->add_undo_method(area_polygon, "set_vertices", area_polygon->get_vertices());
							undo_redo->add_do_method(area_polygon, "set_vertices", obstacle_vertices);
							undo_redo->commit_action();

							edited_point = -1;
							return EditorPlugin::AFTER_GUI_INPUT_STOP;
						}
					}
				}

			} break;

			case MODE_DELETE: {
				if (mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
					int closest_idx = -1;
					real_t closest_dist = 1e10;
					for (int i = 0; i < obstacle_vertices.size(); i++) {
						Vector2 point = p_camera->unproject_position(gt.xform(obstacle_vertices[i]));
						real_t d = point.distance_to(mouse_position);
						if (d < closest_dist && d < grab_threshold) {
							closest_dist = d;
							closest_idx = i;
						}
					}

					if (closest_idx >= 0) {
						edited_point = -1;

						EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
						undo_redo->create_action(TTR("Edit Obstacle (Remove Vertex)"));
						undo_redo->add_undo_method(area_polygon, "set_vertices", obstacle_vertices);
						obstacle_vertices.remove_at(closest_idx);
						undo_redo->add_do_method(area_polygon, "set_vertices", obstacle_vertices);
						undo_redo->commit_action();
						redraw();
						return EditorPlugin::AFTER_GUI_INPUT_STOP;
					}
				}

			} break;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (edited_point != -1 && (wip_active || mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
			Vector2 mouse_position = mm->get_position();

			Vector3 ray_from = p_camera->project_ray_origin(mouse_position);
			Vector3 ray_dir = p_camera->project_ray_normal(mouse_position);

			const Vector3 safe_scale = area_polygon->get_global_basis().get_scale().abs().maxf(0.001);
			const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y), area_polygon->get_global_position());
			Transform3D gi = gt.affine_inverse();
			Plane projection_plane(Vector3(0.0, 1.0, 0.0), gt.origin);

			Vector3 intersection_point;

			if (!projection_plane.intersects_ray(ray_from, ray_dir, &intersection_point)) {
				return EditorPlugin::AFTER_GUI_INPUT_PASS;
			}

			intersection_point = gi.xform(intersection_point);

			Vector2 cpoint(intersection_point.x, intersection_point.z);

			if (snap_ignore && !Input::get_singleton()->is_key_pressed(Key::CTRL)) {
				snap_ignore = false;
			}

			if (!snap_ignore && Node3DEditor::get_singleton()->is_snap_enabled()) {
				cpoint = cpoint.snappedf(Node3DEditor::get_singleton()->get_translate_snap());
			}
			edited_point_pos = Vector3(cpoint.x, 0.0, cpoint.y);

			redraw();
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed()) {
		if (wip_active && k->get_keycode() == Key::ENTER) {
			_wip_close();
		} else if (wip_active && k->get_keycode() == Key::ESCAPE) {
			_wip_cancel();
		}
	}

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

void NavigationMeshArea3DEditorPlugin::redraw() {
	if (!area_polygon) {
		return;
	}
	RenderingServer *rs = RenderingServer::get_singleton();

	rs->mesh_clear(point_lines_mesh_rid);
	rs->mesh_clear(point_handle_mesh_rid);

	if (!area_polygon->is_visible_in_tree()) {
		return;
	}

	Vector<Vector3> edited_vertices;

	if (wip_active) {
		edited_vertices = wip_vertices;
	} else {
		edited_vertices = area_polygon->get_vertices();
	}

	if (edited_vertices.is_empty()) {
		return;
	}

	Array point_lines_mesh_array;
	point_lines_mesh_array.resize(Mesh::ARRAY_MAX);

	Vector<Vector3> point_lines_mesh_vertices;
	point_lines_mesh_vertices.resize(edited_vertices.size() * 2);
	Vector3 *point_lines_mesh_vertices_ptr = point_lines_mesh_vertices.ptrw();

	int vertex_index = 0;

	for (int i = 0; i < edited_vertices.size(); i++) {
		Vector3 point, next_point;
		if (i == edited_point) {
			point = edited_point_pos;
		} else {
			point = edited_vertices[i];
		}

		if ((wip_active && i == edited_vertices.size() - 1) || (((i + 1) % edited_vertices.size()) == edited_point)) {
			next_point = edited_point_pos;
		} else {
			next_point = edited_vertices[(i + 1) % edited_vertices.size()];
		}

		point_lines_mesh_vertices_ptr[vertex_index++] = point;
		point_lines_mesh_vertices_ptr[vertex_index++] = next_point;
	}

	point_lines_mesh_array[Mesh::ARRAY_VERTEX] = point_lines_mesh_vertices;

	rs->mesh_add_surface_from_arrays(point_lines_mesh_rid, RS::PRIMITIVE_LINES, point_lines_mesh_array);
	rs->instance_set_surface_override_material(point_lines_instance_rid, 0, line_material->get_rid());
	const Vector3 safe_scale = area_polygon->get_global_basis().get_scale().abs().maxf(0.001);
	const Transform3D gt = Transform3D(Basis().scaled(safe_scale).rotated(Vector3(0.0, 1.0, 0.0), area_polygon->get_global_rotation().y), area_polygon->get_global_position());
	rs->instance_set_transform(point_lines_instance_rid, gt);

	Array point_handle_mesh_array;
	point_handle_mesh_array.resize(Mesh::ARRAY_MAX);
	Vector<Vector3> point_handle_mesh_vertices;

	point_handle_mesh_vertices.resize(edited_vertices.size());
	Vector3 *point_handle_mesh_vertices_ptr = point_handle_mesh_vertices.ptrw();

	for (int i = 0; i < edited_vertices.size(); i++) {
		Vector3 point_handle_3d;

		if (i == edited_point) {
			point_handle_3d = edited_point_pos;
		} else {
			point_handle_3d = edited_vertices[i];
		}

		point_handle_mesh_vertices_ptr[i] = point_handle_3d;
	}

	point_handle_mesh_array[Mesh::ARRAY_VERTEX] = point_handle_mesh_vertices;

	rs->mesh_add_surface_from_arrays(point_handle_mesh_rid, RS::PRIMITIVE_POINTS, point_handle_mesh_array);
	rs->instance_set_surface_override_material(point_handles_instance_rid, 0, handle_material->get_rid());
	rs->instance_set_transform(point_handles_instance_rid, gt);
}

NavigationMeshArea3DEditorPlugin *NavigationMeshArea3DEditorPlugin::singleton = nullptr;

NavigationMeshArea3DEditorPlugin::NavigationMeshArea3DEditorPlugin() {
	singleton = this;

	line_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	line_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	line_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	line_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	line_material->set_albedo(Color(1, 0.3, 0.1, 0.8));
	line_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);

	handle_material = Ref<StandardMaterial3D>(memnew(StandardMaterial3D));
	handle_material->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	handle_material->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	handle_material->set_flag(StandardMaterial3D::FLAG_USE_POINT_SIZE, true);
	handle_material->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	handle_material->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	handle_material->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	Ref<Texture2D> handle = EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Editor3DHandle"), EditorStringName(EditorIcons));
	handle_material->set_point_size(handle->get_width());
	handle_material->set_texture(StandardMaterial3D::TEXTURE_ALBEDO, handle);
	handle_material->set_flag(StandardMaterial3D::FLAG_DISABLE_DEPTH_TEST, true);

	RenderingServer *rs = RenderingServer::get_singleton();

	point_lines_mesh_rid = rs->mesh_create();
	point_handle_mesh_rid = rs->mesh_create();

	point_lines_instance_rid = rs->instance_create();
	point_handles_instance_rid = rs->instance_create();

	rs->instance_set_base(point_lines_instance_rid, point_lines_mesh_rid);
	rs->instance_set_base(point_handles_instance_rid, point_handle_mesh_rid);

	obstacle_editor = memnew(HBoxContainer);
	obstacle_editor->hide();

	Ref<ButtonGroup> bg;
	bg.instantiate();

	button_create = memnew(Button);
	button_create->set_theme_type_variation(SceneStringName(FlatButton));
	obstacle_editor->add_child(button_create);
	button_create->set_tooltip_text(TTR("Add Vertex"));
	button_create->connect(SceneStringName(pressed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::set_mode).bind(NavigationMeshArea3DEditorPlugin::MODE_CREATE));
	button_create->set_toggle_mode(true);
	button_create->set_button_group(bg);

	button_edit = memnew(Button);
	button_edit->set_theme_type_variation(SceneStringName(FlatButton));
	obstacle_editor->add_child(button_edit);
	button_edit->connect(SceneStringName(pressed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::set_mode).bind(NavigationMeshArea3DEditorPlugin::MODE_EDIT));
	button_edit->set_toggle_mode(true);
	button_edit->set_button_group(bg);

	button_delete = memnew(Button);
	button_delete->set_theme_type_variation(SceneStringName(FlatButton));
	obstacle_editor->add_child(button_delete);
	button_delete->connect(SceneStringName(pressed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::set_mode).bind(NavigationMeshArea3DEditorPlugin::MODE_DELETE));
	button_delete->set_toggle_mode(true);
	button_delete->set_button_group(bg);

	button_clear = memnew(Button);
	button_clear->set_theme_type_variation(SceneStringName(FlatButton));
	obstacle_editor->add_child(button_clear);
	button_clear->connect(SceneStringName(pressed), callable_mp(this, &NavigationMeshArea3DEditorPlugin::set_mode).bind(NavigationMeshArea3DEditorPlugin::ACTION_CLEAR));
	button_clear->set_toggle_mode(true);

	button_clear_dialog = memnew(ConfirmationDialog);
	button_clear_dialog->set_title(TTR("Please Confirm..."));
	button_clear_dialog->set_text(TTR("Remove all vertices?"));
	button_clear_dialog->connect(SceneStringName(confirmed), callable_mp(NavigationMeshArea3DEditorPlugin::singleton, &NavigationMeshArea3DEditorPlugin::action_clear_vertices));
	obstacle_editor->add_child(button_clear_dialog);

	Node3DEditor::get_singleton()->add_control_to_menu_panel(obstacle_editor);

	Ref<NavigationMeshArea3DGizmoPlugin> gizmo_plugin = memnew(NavigationMeshArea3DGizmoPlugin());
	area_3d_gizmo_plugin = gizmo_plugin;
	Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
}

NavigationMeshArea3DEditorPlugin::~NavigationMeshArea3DEditorPlugin() {
	RenderingServer *rs = RenderingServer::get_singleton();
	ERR_FAIL_NULL(rs);

	if (point_lines_instance_rid.is_valid()) {
		rs->free(point_lines_instance_rid);
		point_lines_instance_rid = RID();
	}
	if (point_lines_mesh_rid.is_valid()) {
		rs->free(point_lines_mesh_rid);
		point_lines_mesh_rid = RID();
	}

	if (point_handles_instance_rid.is_valid()) {
		rs->free(point_handles_instance_rid);
		point_handles_instance_rid = RID();
	}
	if (point_handle_mesh_rid.is_valid()) {
		rs->free(point_handle_mesh_rid);
		point_handle_mesh_rid = RID();
	}
}
