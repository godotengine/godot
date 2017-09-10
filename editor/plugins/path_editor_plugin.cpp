/*************************************************************************/
/*  path_editor_plugin.cpp                                               */
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
#include "path_editor_plugin.h"

#include "os/keyboard.h"
#include "scene/resources/curve.h"
#include "spatial_editor_plugin.h"

String PathSpatialGizmo::get_handle_name(int p_idx) const {

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return "";

	if (p_idx < c->get_point_count()) {

		return TTR("Curve Point #") + itos(p_idx);
	}

	p_idx = p_idx - c->get_point_count() + 1;

	int idx = p_idx / 2;
	int t = p_idx % 2;
	String n = TTR("Curve Point #") + itos(idx);
	if (t == 0)
		n += " In";
	else
		n += " Out";

	return n;
}
Variant PathSpatialGizmo::get_handle_value(int p_idx) const {

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return Variant();

	if (p_idx < c->get_point_count()) {

		original = c->get_point_position(p_idx);
		return original;
	}

	p_idx = p_idx - c->get_point_count() + 1;

	int idx = p_idx / 2;
	int t = p_idx % 2;

	Vector3 ofs;
	if (t == 0)
		ofs = c->get_point_in(idx);
	else
		ofs = c->get_point_out(idx);

	original = ofs + c->get_point_position(idx);

	return ofs;
}
void PathSpatialGizmo::set_handle(int p_idx, Camera *p_camera, const Point2 &p_point) {

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return;

	Transform gt = path->get_global_transform();
	Transform gi = gt.affine_inverse();
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	if (p_idx < c->get_point_count()) {

		Plane p(gt.xform(original), p_camera->get_transform().basis.get_axis(2));

		Vector3 inters;

		if (p.intersects_ray(ray_from, ray_dir, &inters)) {

			if (SpatialEditor::get_singleton()->is_snap_enabled()) {
				float snap = SpatialEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			Vector3 local = gi.xform(inters);
			c->set_point_position(p_idx, local);
		}

		return;
	}

	p_idx = p_idx - c->get_point_count() + 1;

	int idx = p_idx / 2;
	int t = p_idx % 2;

	Vector3 base = c->get_point_position(idx);

	Plane p(gt.xform(original), p_camera->get_transform().basis.get_axis(2));

	Vector3 inters;

	if (p.intersects_ray(ray_from, ray_dir, &inters)) {

		Vector3 local = gi.xform(inters) - base;
		if (t == 0) {
			c->set_point_in(idx, local);
		} else {
			c->set_point_out(idx, local);
		}
	}
}

void PathSpatialGizmo::commit_handle(int p_idx, const Variant &p_restore, bool p_cancel) {

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return;

	UndoRedo *ur = SpatialEditor::get_singleton()->get_undo_redo();

	if (p_idx < c->get_point_count()) {

		if (p_cancel) {

			c->set_point_position(p_idx, p_restore);
			return;
		}
		ur->create_action(TTR("Set Curve Point Position"));
		ur->add_do_method(c.ptr(), "set_point_position", p_idx, c->get_point_position(p_idx));
		ur->add_undo_method(c.ptr(), "set_point_position", p_idx, p_restore);
		ur->commit_action();

		return;
	}

	p_idx = p_idx - c->get_point_count() + 1;

	int idx = p_idx / 2;
	int t = p_idx % 2;

	Vector3 ofs;

	if (p_cancel) {

		return;
	}

	if (t == 0) {

		if (p_cancel) {

			c->set_point_in(p_idx, p_restore);
			return;
		}
		ur->create_action(TTR("Set Curve In Position"));
		ur->add_do_method(c.ptr(), "set_point_in", idx, c->get_point_in(idx));
		ur->add_undo_method(c.ptr(), "set_point_in", idx, p_restore);
		ur->commit_action();

	} else {
		if (p_cancel) {

			c->set_point_out(idx, p_restore);
			return;
		}
		ur->create_action(TTR("Set Curve Out Position"));
		ur->add_do_method(c.ptr(), "set_point_out", idx, c->get_point_out(idx));
		ur->add_undo_method(c.ptr(), "set_point_out", idx, p_restore);
		ur->commit_action();
	}
}

void PathSpatialGizmo::redraw() {

	clear();

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return;

	PoolVector<Vector3> v3a = c->tessellate();
	//PoolVector<Vector3> v3a=c->get_baked_points();

	int v3s = v3a.size();
	if (v3s == 0)
		return;
	Vector<Vector3> v3p;
	PoolVector<Vector3>::Read r = v3a.read();

	// BUG: the following won't work when v3s, avoid drawing as a temporary workaround.
	for (int i = 0; i < v3s - 1; i++) {

		v3p.push_back(r[i]);
		v3p.push_back(r[i + 1]);
		//v3p.push_back(r[i]);
		//v3p.push_back(r[i]+Vector3(0,0.2,0));
	}

	if (v3p.size() > 1) {
		add_lines(v3p, PathEditorPlugin::singleton->path_material);
		add_collision_segments(v3p);
	}

	if (PathEditorPlugin::singleton->get_edited_path() == path) {
		v3p.clear();
		Vector<Vector3> handles;
		Vector<Vector3> sec_handles;

		for (int i = 0; i < c->get_point_count(); i++) {

			Vector3 p = c->get_point_position(i);
			handles.push_back(p);
			if (i > 0) {
				v3p.push_back(p);
				v3p.push_back(p + c->get_point_in(i));
				sec_handles.push_back(p + c->get_point_in(i));
			}

			if (i < c->get_point_count() - 1) {
				v3p.push_back(p);
				v3p.push_back(p + c->get_point_out(i));
				sec_handles.push_back(p + c->get_point_out(i));
			}
		}

		if (v3p.size() > 1) {
			add_lines(v3p, PathEditorPlugin::singleton->path_thin_material);
		}
		if (handles.size()) {
			add_handles(handles);
		}
		if (sec_handles.size()) {
			add_handles(sec_handles, false, true);
		}
	}
}

PathSpatialGizmo::PathSpatialGizmo(Path *p_path) {

	path = p_path;
	set_spatial_node(p_path);
}

Ref<SpatialEditorGizmo> PathEditorPlugin::create_spatial_gizmo(Spatial *p_spatial) {

	if (Object::cast_to<Path>(p_spatial)) {

		return memnew(PathSpatialGizmo(Object::cast_to<Path>(p_spatial)));
	}

	return Ref<SpatialEditorGizmo>();
}

bool PathEditorPlugin::forward_spatial_gui_input(Camera *p_camera, const Ref<InputEvent> &p_event) {

	if (!path)
		return false;
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return false;
	Transform gt = path->get_global_transform();
	Transform it = gt.affine_inverse();

	static const int click_dist = 10; //should make global

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {

		Point2 mbpos(mb->get_position().x, mb->get_position().y);

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT && (curve_create->is_pressed() || (curve_edit->is_pressed() && mb->get_control()))) {
			//click into curve, break it down
			PoolVector<Vector3> v3a = c->tessellate();
			int idx = 0;
			int rc = v3a.size();
			int closest_seg = -1;
			Vector3 closest_seg_point;
			float closest_d = 1e20;

			if (rc >= 2) {
				PoolVector<Vector3>::Read r = v3a.read();

				if (p_camera->unproject_position(gt.xform(c->get_point_position(0))).distance_to(mbpos) < click_dist)
					return false; //nope, existing

				for (int i = 0; i < c->get_point_count() - 1; i++) {
					//find the offset and point index of the place to break up
					int j = idx;
					if (p_camera->unproject_position(gt.xform(c->get_point_position(i + 1))).distance_to(mbpos) < click_dist)
						return false; //nope, existing

					while (j < rc && c->get_point_position(i + 1) != r[j]) {

						Vector3 from = r[j];
						Vector3 to = r[j + 1];
						real_t cdist = from.distance_to(to);
						from = gt.xform(from);
						to = gt.xform(to);
						if (cdist > 0) {
							Vector2 s[2];
							s[0] = p_camera->unproject_position(from);
							s[1] = p_camera->unproject_position(to);
							Vector2 inters = Geometry::get_closest_point_to_segment_2d(mbpos, s);
							float d = inters.distance_to(mbpos);

							if (d < 10 && d < closest_d) {

								closest_d = d;
								closest_seg = i;
								Vector3 ray_from = p_camera->project_ray_origin(mbpos);
								Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

								Vector3 ra, rb;
								Geometry::get_closest_points_between_segments(ray_from, ray_from + ray_dir * 4096, from, to, ra, rb);

								closest_seg_point = it.xform(rb);
							}
						}
						j++;
					}
					if (idx == j)
						idx++; //force next
					else
						idx = j; //swap

					if (j == rc)
						break;
				}
			}

			UndoRedo *ur = editor->get_undo_redo();
			if (closest_seg != -1) {
				//subdivide

				ur->create_action(TTR("Split Path"));
				ur->add_do_method(c.ptr(), "add_point", closest_seg_point, Vector3(), Vector3(), closest_seg + 1);
				ur->add_undo_method(c.ptr(), "remove_point", closest_seg + 1);
				ur->commit_action();
				return true;

			} else {

				Vector3 org;
				if (c->get_point_count() == 0)
					org = path->get_transform().get_origin();
				else
					org = gt.xform(c->get_point_position(c->get_point_count() - 1));
				Plane p(org, p_camera->get_transform().basis.get_axis(2));
				Vector3 ray_from = p_camera->project_ray_origin(mbpos);
				Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

				Vector3 inters;
				if (p.intersects_ray(ray_from, ray_dir, &inters)) {

					ur->create_action(TTR("Add Point to Curve"));
					ur->add_do_method(c.ptr(), "add_point", it.xform(inters), Vector3(), Vector3(), -1);
					ur->add_undo_method(c.ptr(), "remove_point", c->get_point_count());
					ur->commit_action();
					return true;
				}

				//add new at pos
			}

		} else if (mb->is_pressed() && ((mb->get_button_index() == BUTTON_LEFT && curve_del->is_pressed()) || (mb->get_button_index() == BUTTON_RIGHT && curve_edit->is_pressed()))) {

			for (int i = 0; i < c->get_point_count(); i++) {
				real_t dist_to_p = p_camera->unproject_position(gt.xform(c->get_point_position(i))).distance_to(mbpos);
				real_t dist_to_p_out = p_camera->unproject_position(gt.xform(c->get_point_position(i) + c->get_point_out(i))).distance_to(mbpos);
				real_t dist_to_p_in = p_camera->unproject_position(gt.xform(c->get_point_position(i) + c->get_point_in(i))).distance_to(mbpos);

				// Find the offset and point index of the place to break up.
				// Also check for the control points.
				if (dist_to_p < click_dist) {

					UndoRedo *ur = editor->get_undo_redo();
					ur->create_action(TTR("Remove Path Point"));
					ur->add_do_method(c.ptr(), "remove_point", i);
					ur->add_undo_method(c.ptr(), "add_point", c->get_point_position(i), c->get_point_in(i), c->get_point_out(i), i);
					ur->commit_action();
					return true;
				} else if (dist_to_p_out < click_dist) {

					UndoRedo *ur = editor->get_undo_redo();
					ur->create_action(TTR("Remove Out-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_out", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_out", i, c->get_point_out(i));
					ur->commit_action();
					return true;
				} else if (dist_to_p_in < click_dist) {

					UndoRedo *ur = editor->get_undo_redo();
					ur->create_action(TTR("Remove In-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_in", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_in", i, c->get_point_in(i));
					ur->commit_action();
					return true;
				}
			}
		}
	}

	return false;
}

void PathEditorPlugin::edit(Object *p_object) {

	if (p_object) {
		path = Object::cast_to<Path>(p_object);
		if (path) {

			if (path->get_curve().is_valid()) {
				path->get_curve()->emit_signal("changed");
			}
		}
	} else {
		Path *pre = path;
		path = NULL;
		if (pre) {
			pre->get_curve()->emit_signal("changed");
		}
	}
	//collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool PathEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Path");
}

void PathEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		curve_create->show();
		curve_edit->show();
		curve_del->show();
		curve_close->show();
		sep->show();
	} else {

		curve_create->hide();
		curve_edit->hide();
		curve_del->hide();
		curve_close->hide();
		sep->hide();

		{
			Path *pre = path;
			path = NULL;
			if (pre && pre->get_curve().is_valid()) {
				pre->get_curve()->emit_signal("changed");
			}
		}
	}
}

void PathEditorPlugin::_mode_changed(int p_idx) {

	curve_create->set_pressed(p_idx == 0);
	curve_edit->set_pressed(p_idx == 1);
	curve_del->set_pressed(p_idx == 2);
}

void PathEditorPlugin::_close_curve() {

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null())
		return;
	if (c->get_point_count() < 2)
		return;
	c->add_point(c->get_point_position(0), c->get_point_in(0), c->get_point_out(0));
}

void PathEditorPlugin::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		curve_create->connect("pressed", this, "_mode_changed", make_binds(0));
		curve_edit->connect("pressed", this, "_mode_changed", make_binds(1));
		curve_del->connect("pressed", this, "_mode_changed", make_binds(2));
		curve_close->connect("pressed", this, "_close_curve");
	}
}

void PathEditorPlugin::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_mode_changed"), &PathEditorPlugin::_mode_changed);
	ClassDB::bind_method(D_METHOD("_close_curve"), &PathEditorPlugin::_close_curve);
}

PathEditorPlugin *PathEditorPlugin::singleton = NULL;

PathEditorPlugin::PathEditorPlugin(EditorNode *p_node) {

	path = NULL;
	editor = p_node;
	singleton = this;

	path_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	path_material->set_albedo(Color(0.5, 0.5, 1.0, 0.8));
	path_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	path_material->set_line_width(3);
	path_material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	path_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);

	path_thin_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));
	path_thin_material->set_albedo(Color(0.5, 0.5, 1.0, 0.4));
	path_thin_material->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	path_thin_material->set_line_width(1);
	path_thin_material->set_cull_mode(SpatialMaterial::CULL_DISABLED);
	path_thin_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);

	//SpatialEditor::get_singleton()->add_gizmo_plugin(this);

	sep = memnew(VSeparator);
	sep->hide();
	SpatialEditor::get_singleton()->add_control_to_menu_panel(sep);
	curve_edit = memnew(ToolButton);
	curve_edit->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveEdit", "EditorIcons"));
	curve_edit->set_toggle_mode(true);
	curve_edit->hide();
	curve_edit->set_focus_mode(Control::FOCUS_NONE);
	curve_edit->set_tooltip(TTR("Select Points") + "\n" + TTR("Shift+Drag: Select Control Points") + "\n" + keycode_get_string(KEY_MASK_CMD) + TTR("Click: Add Point") + "\n" + TTR("Right Click: Delete Point"));
	SpatialEditor::get_singleton()->add_control_to_menu_panel(curve_edit);
	curve_create = memnew(ToolButton);
	curve_create->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveCreate", "EditorIcons"));
	curve_create->set_toggle_mode(true);
	curve_create->hide();
	curve_create->set_focus_mode(Control::FOCUS_NONE);
	curve_create->set_tooltip(TTR("Add Point (in empty space)") + "\n" + TTR("Split Segment (in curve)"));
	SpatialEditor::get_singleton()->add_control_to_menu_panel(curve_create);
	curve_del = memnew(ToolButton);
	curve_del->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveDelete", "EditorIcons"));
	curve_del->set_toggle_mode(true);
	curve_del->hide();
	curve_del->set_focus_mode(Control::FOCUS_NONE);
	curve_del->set_tooltip(TTR("Delete Point"));
	SpatialEditor::get_singleton()->add_control_to_menu_panel(curve_del);
	curve_close = memnew(ToolButton);
	curve_close->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveClose", "EditorIcons"));
	curve_close->hide();
	curve_close->set_focus_mode(Control::FOCUS_NONE);
	curve_close->set_tooltip(TTR("Close Curve"));
	SpatialEditor::get_singleton()->add_control_to_menu_panel(curve_close);

	curve_edit->set_pressed(true);
	/*
	collision_polygon_editor = memnew( PathEditor(p_node) );
	editor->get_viewport()->add_child(collision_polygon_editor);

	collision_polygon_editor->set_margin(MARGIN_LEFT,200);
	collision_polygon_editor->set_margin(MARGIN_RIGHT,230);
	collision_polygon_editor->set_margin(MARGIN_TOP,0);
	collision_polygon_editor->set_margin(MARGIN_BOTTOM,10);


	collision_polygon_editor->hide();
	*/
}

PathEditorPlugin::~PathEditorPlugin() {
}
