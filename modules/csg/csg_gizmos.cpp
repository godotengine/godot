#include "csg_gizmos.h"

///////////

String CSGShapeSpatialGizmo::get_handle_name(int p_idx) const {

	if (Object::cast_to<CSGSphere>(cs)) {

		return "Radius";
	}

	if (Object::cast_to<CSGBox>(cs)) {

		static const char *hname[3] = { "Width", "Height", "Depth" };
		return hname[p_idx];
	}

	if (Object::cast_to<CSGCylinder>(cs)) {

		return p_idx == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<CSGTorus>(cs)) {

		return p_idx == 0 ? "InnerRadius" : "OuterRadius";
	}

	return "";
}
Variant CSGShapeSpatialGizmo::get_handle_value(int p_idx) const {

	if (Object::cast_to<CSGSphere>(cs)) {

		CSGSphere *s = Object::cast_to<CSGSphere>(cs);
		return s->get_radius();
	}

	if (Object::cast_to<CSGBox>(cs)) {

		CSGBox *s = Object::cast_to<CSGBox>(cs);
		switch (p_idx) {
			case 0: return s->get_width();
			case 1: return s->get_height();
			case 2: return s->get_depth();
		}
	}

	if (Object::cast_to<CSGCylinder>(cs)) {

		CSGCylinder *s = Object::cast_to<CSGCylinder>(cs);
		return p_idx == 0 ? s->get_radius() : s->get_height();
	}

	if (Object::cast_to<CSGTorus>(cs)) {

		CSGTorus *s = Object::cast_to<CSGTorus>(cs);
		return p_idx == 0 ? s->get_inner_radius() : s->get_outer_radius();
	}

	return Variant();
}
void CSGShapeSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Transform gt = cs->get_global_transform();
	gt.orthonormalize();
	Transform gi = gt.affine_inverse();

	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	Vector3 sg[2] = { gi.xform(ray_from), gi.xform(ray_from + ray_dir * 16384) };

	if (Object::cast_to<CSGSphere>(cs)) {

		CSGSphere *s = Object::cast_to<CSGSphere>(cs);

		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (d < 0.001)
			d = 0.001;

		s->set_radius(d);
	}

	if (Object::cast_to<CSGBox>(cs)) {

		CSGBox *s = Object::cast_to<CSGBox>(cs);

		Vector3 axis;
		axis[p_idx] = 1.0;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = ra[p_idx];
		if (d < 0.001)
			d = 0.001;

		switch (p_idx) {
			case 0: s->set_width(d); break;
			case 1: s->set_height(d); break;
			case 2: s->set_depth(d); break;
		}
	}

	if (Object::cast_to<CSGCylinder>(cs)) {

		CSGCylinder *s = Object::cast_to<CSGCylinder>(cs);

		Vector3 axis;
		axis[p_idx == 0 ? 0 : 1] = 1.0;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);

		if (d < 0.001)
			d = 0.001;

		if (p_idx == 0)
			s->set_radius(d);
		else if (p_idx == 1)
			s->set_height(d * 2.0);
	}

	if (Object::cast_to<CSGTorus>(cs)) {

		CSGTorus *s = Object::cast_to<CSGTorus>(cs);

		Vector3 axis;
		axis[0] = 1.0;
		Vector3 ra, rb;
		Geometry::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);

		if (d < 0.001)
			d = 0.001;

		if (p_idx == 0)
			s->set_inner_radius(d);
		else if (p_idx == 1)
			s->set_outer_radius(d);
	}
}
void CSGShapeSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	if (Object::cast_to<CSGSphere>(cs)) {
		CSGSphere *s = Object::cast_to<CSGSphere>(cs);
		if (p_cancel) {
			s->set_radius(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Sphere Shape Radius"));
		ur->add_do_method(s, "set_radius", s->get_radius());
		ur->add_undo_method(s, "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<CSGBox>(cs)) {
		CSGBox *s = Object::cast_to<CSGBox>(cs);
		if (p_cancel) {
			switch (p_idx) {
				case 0: s->set_width(p_restore); break;
				case 1: s->set_height(p_restore); break;
				case 2: s->set_depth(p_restore); break;
			}
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		ur->create_action(TTR("Change Box Shape Extents"));
		static const char *method[3] = { "set_width", "set_height", "set_depth" };
		float current;
		switch (p_idx) {
			case 0: current = s->get_width(); break;
			case 1: current = s->get_height(); break;
			case 2: current = s->get_depth(); break;
		}

		ur->add_do_method(s, method[p_idx], current);
		ur->add_undo_method(s, method[p_idx], p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<CSGCylinder>(cs)) {
		CSGCylinder *s = Object::cast_to<CSGCylinder>(cs);
		if (p_cancel) {
			if (p_idx == 0)
				s->set_radius(p_restore);
			else
				s->set_height(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		if (p_idx == 0) {
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

	if (Object::cast_to<CSGTorus>(cs)) {
		CSGTorus *s = Object::cast_to<CSGTorus>(cs);
		if (p_cancel) {
			if (p_idx == 0)
				s->set_inner_radius(p_restore);
			else
				s->set_outer_radius(p_restore);
			return;
		}

		UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();
		if (p_idx == 0) {
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
void CSGShapeSpatialGizmo::redraw() {

	clear();

	Color gizmo_color = EDITOR_GET("editors/3d_gizmos/gizmo_colors/csg");
	Ref<Material> material = create_material("shape_material", gizmo_color);

	PoolVector<Vector3> faces = cs->get_brush_faces();

	Vector<Vector3> lines;
	lines.resize(faces.size() * 2);
	{
		PoolVector<Vector3>::Read r = faces.read();

		for (int i = 0; i < lines.size(); i += 6) {
			int f = i / 6;
			for (int j = 0; j < 3; j++) {
				int j_n = (j + 1) % 3;
				lines[i + j * 2 + 0] = r[f * 3 + j];
				lines[i + j * 2 + 1] = r[f * 3 + j_n];
			}
		}
	}

	add_lines(lines, material);
	add_collision_segments(lines);

	if (Object::cast_to<CSGSphere>(cs)) {
		CSGSphere *s = Object::cast_to<CSGSphere>(cs);

		float r = s->get_radius();
		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		add_handles(handles);
	}

	if (Object::cast_to<CSGBox>(cs)) {
		CSGBox *s = Object::cast_to<CSGBox>(cs);

		Vector<Vector3> handles;
		handles.push_back(Vector3(s->get_width(), 0, 0));
		handles.push_back(Vector3(0, s->get_height(), 0));
		handles.push_back(Vector3(0, 0, s->get_depth()));
		add_handles(handles);
	}

	if (Object::cast_to<CSGCylinder>(cs)) {
		CSGCylinder *s = Object::cast_to<CSGCylinder>(cs);

		Vector<Vector3> handles;
		handles.push_back(Vector3(s->get_radius(), 0, 0));
		handles.push_back(Vector3(0, s->get_height() * 0.5, 0));
		add_handles(handles);
	}

	if (Object::cast_to<CSGTorus>(cs)) {
		CSGTorus *s = Object::cast_to<CSGTorus>(cs);

		Vector<Vector3> handles;
		handles.push_back(Vector3(s->get_inner_radius(), 0, 0));
		handles.push_back(Vector3(s->get_outer_radius(), 0, 0));
		add_handles(handles);
	}
}
CSGShapeSpatialGizmo::CSGShapeSpatialGizmo(CSGShape *p_cs) {

	cs = p_cs;
	set_spatial_node(p_cs);
}

Ref<SpatialEditorGizmo> EditorPluginCSG::create_spatial_gizmo(Spatial *p_spatial) {
	if (Object::cast_to<CSGSphere>(p_spatial) || Object::cast_to<CSGBox>(p_spatial) || Object::cast_to<CSGCylinder>(p_spatial) || Object::cast_to<CSGTorus>(p_spatial) || Object::cast_to<CSGMesh>(p_spatial) || Object::cast_to<CSGPolygon>(p_spatial)) {
		Ref<CSGShapeSpatialGizmo> csg = memnew(CSGShapeSpatialGizmo(Object::cast_to<CSGShape>(p_spatial)));
		return csg;
	}

	return Ref<SpatialEditorGizmo>();
}

EditorPluginCSG::EditorPluginCSG(EditorNode *p_editor) {

	EDITOR_DEF("editors/3d_gizmos/gizmo_colors/csg", Color(0.2, 0.5, 1, 0.1));
}
