/**************************************************************************/
/*  csg_gizmos.cpp                                                        */
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

#include "csg_gizmos.h"

#ifdef TOOLS_ENABLED

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/gizmos/gizmo_3d_helper.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"

///////////

CSGShape3DGizmoPlugin::CSGShape3DGizmoPlugin() {
	helper.instantiate();

	Color gizmo_color = EDITOR_DEF_RST("editors/3d_gizmos/gizmo_colors/csg", Color(0.0, 0.4, 1, 0.15));
	create_material("shape_union_material", gizmo_color);
	create_material("shape_union_solid_material", gizmo_color);
	gizmo_color.invert();
	create_material("shape_subtraction_material", gizmo_color);
	create_material("shape_subtraction_solid_material", gizmo_color);
	gizmo_color.r = 0.95;
	gizmo_color.g = 0.95;
	gizmo_color.b = 0.95;
	create_material("shape_intersection_material", gizmo_color);
	create_material("shape_intersection_solid_material", gizmo_color);

	create_handle_material("handles");
}

CSGShape3DGizmoPlugin::~CSGShape3DGizmoPlugin() {
}

String CSGShape3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	CSGShape3D *cs = Object::cast_to<CSGShape3D>(p_gizmo->get_node_3d());

	if (Object::cast_to<CSGSphere3D>(cs)) {
		return "Radius";
	}

	if (Object::cast_to<CSGBox3D>(cs)) {
		return helper->box_get_handle_name(p_id);
	}

	if (Object::cast_to<CSGCylinder3D>(cs)) {
		return p_id == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<CSGTorus3D>(cs)) {
		return p_id == 0 ? "InnerRadius" : "OuterRadius";
	}

	return "";
}

Variant CSGShape3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	CSGShape3D *cs = Object::cast_to<CSGShape3D>(p_gizmo->get_node_3d());

	if (Object::cast_to<CSGSphere3D>(cs)) {
		CSGSphere3D *s = Object::cast_to<CSGSphere3D>(cs);
		return s->get_radius();
	}

	if (Object::cast_to<CSGBox3D>(cs)) {
		CSGBox3D *s = Object::cast_to<CSGBox3D>(cs);
		return s->get_size();
	}

	if (Object::cast_to<CSGCylinder3D>(cs)) {
		CSGCylinder3D *s = Object::cast_to<CSGCylinder3D>(cs);
		return p_id == 0 ? s->get_radius() : s->get_height();
	}

	if (Object::cast_to<CSGTorus3D>(cs)) {
		CSGTorus3D *s = Object::cast_to<CSGTorus3D>(cs);
		return p_id == 0 ? s->get_inner_radius() : s->get_outer_radius();
	}

	return Variant();
}

void CSGShape3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void CSGShape3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	CSGShape3D *cs = Object::cast_to<CSGShape3D>(p_gizmo->get_node_3d());

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	if (Object::cast_to<CSGSphere3D>(cs)) {
		CSGSphere3D *s = Object::cast_to<CSGSphere3D>(cs);

		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		s->set_radius(d);
	}

	if (Object::cast_to<CSGBox3D>(cs)) {
		CSGBox3D *s = Object::cast_to<CSGBox3D>(cs);
		Vector3 size = s->get_size();
		Vector3 position;
		helper->box_set_handle(sg, p_id, size, position);
		s->set_size(size);
		s->set_global_position(position);
	}

	if (Object::cast_to<CSGCylinder3D>(cs)) {
		CSGCylinder3D *s = Object::cast_to<CSGCylinder3D>(cs);

		Vector3 axis;
		axis[p_id == 0 ? 0 : 1] = 1.0;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		if (p_id == 0) {
			s->set_radius(d);
		} else if (p_id == 1) {
			s->set_height(d * 2.0);
		}
	}

	if (Object::cast_to<CSGTorus3D>(cs)) {
		CSGTorus3D *s = Object::cast_to<CSGTorus3D>(cs);

		Vector3 axis;
		axis[0] = 1.0;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		if (p_id == 0) {
			s->set_inner_radius(d);
		} else if (p_id == 1) {
			s->set_outer_radius(d);
		}
	}
}

void CSGShape3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	CSGShape3D *cs = Object::cast_to<CSGShape3D>(p_gizmo->get_node_3d());

	if (Object::cast_to<CSGSphere3D>(cs)) {
		CSGSphere3D *s = Object::cast_to<CSGSphere3D>(cs);
		if (p_cancel) {
			s->set_radius(p_restore);
			return;
		}

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change Sphere Shape Radius"));
		ur->add_do_method(s, "set_radius", s->get_radius());
		ur->add_undo_method(s, "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<CSGBox3D>(cs)) {
		helper->box_commit_handle(TTR("Change Box Shape Size"), p_cancel, cs);
	}

	if (Object::cast_to<CSGCylinder3D>(cs)) {
		CSGCylinder3D *s = Object::cast_to<CSGCylinder3D>(cs);
		if (p_cancel) {
			if (p_id == 0) {
				s->set_radius(p_restore);
			} else {
				s->set_height(p_restore);
			}
			return;
		}

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		if (p_id == 0) {
			ur->create_action(TTR("Change Cylinder Radius"));
			ur->add_do_method(s, "set_radius", s->get_radius());
			ur->add_undo_method(s, "set_radius", p_restore);
		} else {
			ur->create_action(TTR("Change Cylinder Height"));
			ur->add_do_method(s, "set_height", s->get_height());
			ur->add_undo_method(s, "set_height", p_restore);
		}

		ur->commit_action();
	}

	if (Object::cast_to<CSGTorus3D>(cs)) {
		CSGTorus3D *s = Object::cast_to<CSGTorus3D>(cs);
		if (p_cancel) {
			if (p_id == 0) {
				s->set_inner_radius(p_restore);
			} else {
				s->set_outer_radius(p_restore);
			}
			return;
		}

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		if (p_id == 0) {
			ur->create_action(TTR("Change Torus Inner Radius"));
			ur->add_do_method(s, "set_inner_radius", s->get_inner_radius());
			ur->add_undo_method(s, "set_inner_radius", p_restore);
		} else {
			ur->create_action(TTR("Change Torus Outer Radius"));
			ur->add_do_method(s, "set_outer_radius", s->get_outer_radius());
			ur->add_undo_method(s, "set_outer_radius", p_restore);
		}

		ur->commit_action();
	}
}

bool CSGShape3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<CSGSphere3D>(p_spatial) || Object::cast_to<CSGBox3D>(p_spatial) || Object::cast_to<CSGCylinder3D>(p_spatial) || Object::cast_to<CSGTorus3D>(p_spatial) || Object::cast_to<CSGMesh3D>(p_spatial) || Object::cast_to<CSGPolygon3D>(p_spatial);
}

String CSGShape3DGizmoPlugin::get_gizmo_name() const {
	return "CSGShape3D";
}

int CSGShape3DGizmoPlugin::get_priority() const {
	return -1;
}

bool CSGShape3DGizmoPlugin::is_selectable_when_hidden() const {
	return true;
}

void CSGShape3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	p_gizmo->clear();

	CSGShape3D *cs = Object::cast_to<CSGShape3D>(p_gizmo->get_node_3d());

	Vector<Vector3> faces = cs->get_brush_faces();

	if (faces.size() == 0) {
		return;
	}

	Vector<Vector3> lines;
	lines.resize(faces.size() * 2);
	{
		const Vector3 *r = faces.ptr();

		for (int i = 0; i < lines.size(); i += 6) {
			int f = i / 6;
			for (int j = 0; j < 3; j++) {
				int j_n = (j + 1) % 3;
				lines.write[i + j * 2 + 0] = r[f * 3 + j];
				lines.write[i + j * 2 + 1] = r[f * 3 + j_n];
			}
		}
	}

	Ref<Material> material;
	switch (cs->get_operation()) {
		case CSGShape3D::OPERATION_UNION:
			material = get_material("shape_union_material", p_gizmo);
			break;
		case CSGShape3D::OPERATION_INTERSECTION:
			material = get_material("shape_intersection_material", p_gizmo);
			break;
		case CSGShape3D::OPERATION_SUBTRACTION:
			material = get_material("shape_subtraction_material", p_gizmo);
			break;
	}

	Ref<Material> handles_material = get_material("handles");

	p_gizmo->add_lines(lines, material);
	p_gizmo->add_collision_segments(lines);

	if (cs->is_root_shape()) {
		Array csg_meshes = cs->get_meshes();
		if (csg_meshes.size() == 2) {
			Ref<Mesh> csg_mesh = csg_meshes[1];
			if (csg_mesh.is_valid()) {
				p_gizmo->add_collision_triangles(csg_mesh->generate_triangle_mesh());
			}
		}
	}

	if (p_gizmo->is_selected()) {
		// Draw a translucent representation of the CSG node
		Ref<ArrayMesh> mesh = memnew(ArrayMesh);
		Array array;
		array.resize(Mesh::ARRAY_MAX);
		array[Mesh::ARRAY_VERTEX] = faces;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, array);

		Ref<Material> solid_material;
		switch (cs->get_operation()) {
			case CSGShape3D::OPERATION_UNION:
				solid_material = get_material("shape_union_solid_material", p_gizmo);
				break;
			case CSGShape3D::OPERATION_INTERSECTION:
				solid_material = get_material("shape_intersection_solid_material", p_gizmo);
				break;
			case CSGShape3D::OPERATION_SUBTRACTION:
				solid_material = get_material("shape_subtraction_solid_material", p_gizmo);
				break;
		}

		p_gizmo->add_mesh(mesh, solid_material);
	}

	if (Object::cast_to<CSGSphere3D>(cs)) {
		CSGSphere3D *s = Object::cast_to<CSGSphere3D>(cs);

		float r = s->get_radius();
		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CSGBox3D>(cs)) {
		CSGBox3D *s = Object::cast_to<CSGBox3D>(cs);
		Vector<Vector3> handles = helper->box_get_handles(s->get_size());
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CSGCylinder3D>(cs)) {
		CSGCylinder3D *s = Object::cast_to<CSGCylinder3D>(cs);

		Vector<Vector3> handles;
		handles.push_back(Vector3(s->get_radius(), 0, 0));
		handles.push_back(Vector3(0, s->get_height() * 0.5, 0));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CSGTorus3D>(cs)) {
		CSGTorus3D *s = Object::cast_to<CSGTorus3D>(cs);

		Vector<Vector3> handles;
		handles.push_back(Vector3(s->get_inner_radius(), 0, 0));
		handles.push_back(Vector3(s->get_outer_radius(), 0, 0));
		p_gizmo->add_handles(handles, handles_material);
	}
}

EditorPluginCSG::EditorPluginCSG() {
	Ref<CSGShape3DGizmoPlugin> gizmo_plugin = Ref<CSGShape3DGizmoPlugin>(memnew(CSGShape3DGizmoPlugin));
	Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
}

#endif // TOOLS_ENABLED
