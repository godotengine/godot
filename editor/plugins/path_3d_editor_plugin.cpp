/*************************************************************************/
/*  path_3d_editor_plugin.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "path_3d_editor_plugin.h"

#include "core/math/geometry_2d.h"
#include "core/math/geometry_3d.h"
#include "core/os/keyboard.h"
#include "node_3d_editor_plugin.h"
#include "scene/resources/curve.h"

String Path3DGizmo::get_handle_name(int p_id) const {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return "";
	}

	if (p_id < c->get_point_count()) {
		return TTR("Curve Point #") + itos(p_id);
	}

	p_id = p_id - c->get_point_count() + 1;

	int idx = p_id / 2;
	int t = p_id % 2;
	String n = TTR("Curve Point #") + itos(idx);
	if (t == 0) {
		n += " In";
	} else {
		n += " Out";
	}

	return n;
}

Variant Path3DGizmo::get_handle_value(int p_id) const {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return Variant();
	}

	if (p_id < c->get_point_count()) {
		original = c->get_point_position(p_id);
		return original;
	}

	p_id = p_id - c->get_point_count() + 1;

	int idx = p_id / 2;
	int t = p_id % 2;

	Vector3 ofs;
	if (t == 0) {
		ofs = c->get_point_in(idx);
	} else {
		ofs = c->get_point_out(idx);
	}

	original = ofs + c->get_point_position(idx);

	return ofs;
}

void Path3DGizmo::set_handle(int p_id, Camera3D *p_camera, const Point2 &p_point) {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	Transform3D gt = path->get_global_transform();
	Transform3D gi = gt.affine_inverse();
	Vector3 ray_from = p_camera->project_ray_origin(p_point);
	Vector3 ray_dir = p_camera->project_ray_normal(p_point);

	// Setting curve point positions
	if (p_id < c->get_point_count()) {
		const Plane p = Plane(p_camera->get_transform().basis.get_axis(2), gt.xform(original));

		Vector3 inters;

		if (p.intersects_ray(ray_from, ray_dir, &inters)) {
			if (Node3DEditor::get_singleton()->is_snap_enabled()) {
				float snap = Node3DEditor::get_singleton()->get_translate_snap();
				inters.snap(Vector3(snap, snap, snap));
			}

			Vector3 local = gi.xform(inters);
			c->set_point_position(p_id, local);
		}

		return;
	}

	p_id = p_id - c->get_point_count() + 1;

	int idx = p_id / 2;
	int t = p_id % 2;

	Vector3 base = c->get_point_position(idx);

	Plane p(p_camera->get_transform().basis.get_axis(2), gt.xform(original));

	Vector3 inters;

	// Setting curve in/out positions
	if (p.intersects_ray(ray_from, ray_dir, &inters)) {
		if (!Path3DEditorPlugin::singleton->is_handle_clicked()) {
			orig_in_length = c->get_point_in(idx).length();
			orig_out_length = c->get_point_out(idx).length();
			Path3DEditorPlugin::singleton->set_handle_clicked(true);
		}

		Vector3 local = gi.xform(inters) - base;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			float snap = Node3DEditor::get_singleton()->get_translate_snap();
			local.snap(Vector3(snap, snap, snap));
		}

		if (t == 0) {
			c->set_point_in(idx, local);
			if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
				c->set_point_out(idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -local : (-local.normalized() * orig_out_length));
			}
		} else {
			c->set_point_out(idx, local);
			if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
				c->set_point_in(idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -local : (-local.normalized() * orig_in_length));
			}
		}
	}
}

void Path3DGizmo::commit_handle(int p_id, const Variant &p_restore, bool p_cancel) {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	UndoRedo *ur = Node3DEditor::get_singleton()->get_undo_redo();

	if (p_id < c->get_point_count()) {
		if (p_cancel) {
			c->set_point_position(p_id, p_restore);
			return;
		}
		ur->create_action(TTR("Set Curve Point Position"));
		ur->add_do_method(c.ptr(), "set_point_position", p_id, c->get_point_position(p_id));
		ur->add_undo_method(c.ptr(), "set_point_position", p_id, p_restore);
		ur->commit_action();

		return;
	}

	p_id = p_id - c->get_point_count() + 1;

	int idx = p_id / 2;
	int t = p_id % 2;

	if (t == 0) {
		if (p_cancel) {
			c->set_point_in(p_id, p_restore);
			return;
		}

		ur->create_action(TTR("Set Curve In Position"));
		ur->add_do_method(c.ptr(), "set_point_in", idx, c->get_point_in(idx));
		ur->add_undo_method(c.ptr(), "set_point_in", idx, p_restore);

		if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
			ur->add_do_method(c.ptr(), "set_point_out", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -c->get_point_in(idx) : (-c->get_point_in(idx).normalized() * orig_out_length));
			ur->add_undo_method(c.ptr(), "set_point_out", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -static_cast<Vector3>(p_restore) : (-static_cast<Vector3>(p_restore).normalized() * orig_out_length));
		}
		ur->commit_action();

	} else {
		if (p_cancel) {
			c->set_point_out(idx, p_restore);

			return;
		}

		ur->create_action(TTR("Set Curve Out Position"));
		ur->add_do_method(c.ptr(), "set_point_out", idx, c->get_point_out(idx));
		ur->add_undo_method(c.ptr(), "set_point_out", idx, p_restore);

		if (Path3DEditorPlugin::singleton->mirror_angle_enabled()) {
			ur->add_do_method(c.ptr(), "set_point_in", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -c->get_point_out(idx) : (-c->get_point_out(idx).normalized() * orig_in_length));
			ur->add_undo_method(c.ptr(), "set_point_in", idx, Path3DEditorPlugin::singleton->mirror_length_enabled() ? -static_cast<Vector3>(p_restore) : (-static_cast<Vector3>(p_restore).normalized() * orig_in_length));
		}
		ur->commit_action();
	}
}

void Path3DGizmo::redraw() {
	clear();

	Ref<StandardMaterial3D> path_material = gizmo_plugin->get_material("path_material", this);
	Ref<StandardMaterial3D> path_thin_material = gizmo_plugin->get_material("path_thin_material", this);
	Ref<StandardMaterial3D> handles_material = gizmo_plugin->get_material("handles");
	Ref<StandardMaterial3D> sec_handles_material = gizmo_plugin->get_material("sec_handles");

	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}

	Vector<Vector3> v3a = c->tessellate();
	//Vector<Vector3> v3a=c->get_baked_points();

	int v3s = v3a.size();
	if (v3s == 0) {
		return;
	}
	Vector<Vector3> v3p;
	const Vector3 *r = v3a.ptr();

	// BUG: the following won't work when v3s, avoid drawing as a temporary workaround.
	for (int i = 0; i < v3s - 1; i++) {
		v3p.push_back(r[i]);
		v3p.push_back(r[i + 1]);
		//v3p.push_back(r[i]);
		//v3p.push_back(r[i]+Vector3(0,0.2,0));
	}

	if (v3p.size() > 1) {
		add_lines(v3p, path_material);
		add_collision_segments(v3p);
	}

	if (Path3DEditorPlugin::singleton->get_edited_path() == path) {
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
			add_lines(v3p, path_thin_material);
		}
		if (handles.size()) {
			add_handles(handles, handles_material);
		}
		if (sec_handles.size()) {
			add_handles(sec_handles, sec_handles_material, Vector<int>(), false, true);
		}
	}
}

Path3DGizmo::Path3DGizmo(Path3D *p_path) {
	path = p_path;
	set_spatial_node(p_path);
	orig_in_length = 0;
	orig_out_length = 0;
}

EditorPlugin::AfterGUIInput Path3DEditorPlugin::forward_spatial_gui_input(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!path) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}
	Transform3D gt = path->get_global_transform();
	Transform3D it = gt.affine_inverse();

	static const int click_dist = 10; //should make global

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		Point2 mbpos(mb->get_position().x, mb->get_position().y);

		if (!mb->is_pressed()) {
			set_handle_clicked(false);
		}

		if (mb->is_pressed() && mb->get_button_index() == MOUSE_BUTTON_LEFT && (curve_create->is_pressed() || (curve_edit->is_pressed() && mb->is_ctrl_pressed()))) {
			//click into curve, break it down
			Vector<Vector3> v3a = c->tessellate();
			int idx = 0;
			int rc = v3a.size();
			int closest_seg = -1;
			Vector3 closest_seg_point;
			float closest_d = 1e20;

			if (rc >= 2) {
				const Vector3 *r = v3a.ptr();

				if (p_camera->unproject_position(gt.xform(c->get_point_position(0))).distance_to(mbpos) < click_dist) {
					return EditorPlugin::AFTER_GUI_INPUT_PASS; //nope, existing
				}

				for (int i = 0; i < c->get_point_count() - 1; i++) {
					//find the offset and point index of the place to break up
					int j = idx;
					if (p_camera->unproject_position(gt.xform(c->get_point_position(i + 1))).distance_to(mbpos) < click_dist) {
						return EditorPlugin::AFTER_GUI_INPUT_PASS; //nope, existing
					}

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
							Vector2 inters = Geometry2D::get_closest_point_to_segment(mbpos, s);
							float d = inters.distance_to(mbpos);

							if (d < 10 && d < closest_d) {
								closest_d = d;
								closest_seg = i;
								Vector3 ray_from = p_camera->project_ray_origin(mbpos);
								Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

								Vector3 ra, rb;
								Geometry3D::get_closest_points_between_segments(ray_from, ray_from + ray_dir * 4096, from, to, ra, rb);

								closest_seg_point = it.xform(rb);
							}
						}
						j++;
					}
					if (idx == j) {
						idx++; //force next
					} else {
						idx = j; //swap
					}

					if (j == rc) {
						break;
					}
				}
			}

			UndoRedo *ur = editor->get_undo_redo();
			if (closest_seg != -1) {
				//subdivide

				ur->create_action(TTR("Split Path"));
				ur->add_do_method(c.ptr(), "add_point", closest_seg_point, Vector3(), Vector3(), closest_seg + 1);
				ur->add_undo_method(c.ptr(), "remove_point", closest_seg + 1);
				ur->commit_action();
				return EditorPlugin::AFTER_GUI_INPUT_STOP;

			} else {
				Vector3 origin;
				if (c->get_point_count() == 0) {
					origin = path->get_transform().get_origin();
				} else {
					origin = gt.xform(c->get_point_position(c->get_point_count() - 1));
				}
				Plane p(p_camera->get_transform().basis.get_axis(2), origin);
				Vector3 ray_from = p_camera->project_ray_origin(mbpos);
				Vector3 ray_dir = p_camera->project_ray_normal(mbpos);

				Vector3 inters;
				if (p.intersects_ray(ray_from, ray_dir, &inters)) {
					ur->create_action(TTR("Add Point to Curve"));
					ur->add_do_method(c.ptr(), "add_point", it.xform(inters), Vector3(), Vector3(), -1);
					ur->add_undo_method(c.ptr(), "remove_point", c->get_point_count());
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}

				//add new at pos
			}

		} else if (mb->is_pressed() && ((mb->get_button_index() == MOUSE_BUTTON_LEFT && curve_del->is_pressed()) || (mb->get_button_index() == MOUSE_BUTTON_RIGHT && curve_edit->is_pressed()))) {
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
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (dist_to_p_out < click_dist) {
					UndoRedo *ur = editor->get_undo_redo();
					ur->create_action(TTR("Remove Out-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_out", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_out", i, c->get_point_out(i));
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (dist_to_p_in < click_dist) {
					UndoRedo *ur = editor->get_undo_redo();
					ur->create_action(TTR("Remove In-Control Point"));
					ur->add_do_method(c.ptr(), "set_point_in", i, Vector3());
					ur->add_undo_method(c.ptr(), "set_point_in", i, c->get_point_in(i));
					ur->commit_action();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}
			}
		}
	}

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

void Path3DEditorPlugin::edit(Object *p_object) {
	if (p_object) {
		path = Object::cast_to<Path3D>(p_object);
		if (path) {
			if (path->get_curve().is_valid()) {
				path->get_curve()->emit_signal(SNAME("changed"));
			}
		}
	} else {
		Path3D *pre = path;
		path = nullptr;
		if (pre) {
			pre->get_curve()->emit_signal(SNAME("changed"));
		}
	}
	//collision_polygon_editor->edit(Object::cast_to<Node>(p_object));
}

bool Path3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("Path3D");
}

void Path3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		curve_create->show();
		curve_edit->show();
		curve_del->show();
		curve_close->show();
		handle_menu->show();
		sep->show();
	} else {
		curve_create->hide();
		curve_edit->hide();
		curve_del->hide();
		curve_close->hide();
		handle_menu->hide();
		sep->hide();

		{
			Path3D *pre = path;
			path = nullptr;
			if (pre && pre->get_curve().is_valid()) {
				pre->get_curve()->emit_signal(SNAME("changed"));
			}
		}
	}
}

void Path3DEditorPlugin::_mode_changed(int p_idx) {
	curve_create->set_pressed(p_idx == 0);
	curve_edit->set_pressed(p_idx == 1);
	curve_del->set_pressed(p_idx == 2);
}

void Path3DEditorPlugin::_close_curve() {
	Ref<Curve3D> c = path->get_curve();
	if (c.is_null()) {
		return;
	}
	if (c->get_point_count() < 2) {
		return;
	}
	if (c->get_point_position(0) == c->get_point_position(c->get_point_count() - 1)) {
		return;
	}
	UndoRedo *ur = editor->get_undo_redo();
	ur->create_action(TTR("Close Curve"));
	ur->add_do_method(c.ptr(), "add_point", c->get_point_position(0), c->get_point_in(0), c->get_point_out(0), -1);
	ur->add_undo_method(c.ptr(), "remove_point", c->get_point_count());
	ur->commit_action();
}

void Path3DEditorPlugin::_handle_option_pressed(int p_option) {
	PopupMenu *pm;
	pm = handle_menu->get_popup();

	switch (p_option) {
		case HANDLE_OPTION_ANGLE: {
			bool is_checked = pm->is_item_checked(HANDLE_OPTION_ANGLE);
			mirror_handle_angle = !is_checked;
			pm->set_item_checked(HANDLE_OPTION_ANGLE, mirror_handle_angle);
			pm->set_item_disabled(HANDLE_OPTION_LENGTH, !mirror_handle_angle);
		} break;
		case HANDLE_OPTION_LENGTH: {
			bool is_checked = pm->is_item_checked(HANDLE_OPTION_LENGTH);
			mirror_handle_length = !is_checked;
			pm->set_item_checked(HANDLE_OPTION_LENGTH, mirror_handle_length);
		} break;
	}
}

void Path3DEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		curve_create->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed), make_binds(0));
		curve_edit->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed), make_binds(1));
		curve_del->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_mode_changed), make_binds(2));
		curve_close->connect("pressed", callable_mp(this, &Path3DEditorPlugin::_close_curve));
	}
}

void Path3DEditorPlugin::_bind_methods() {
}

Path3DEditorPlugin *Path3DEditorPlugin::singleton = nullptr;

Path3DEditorPlugin::Path3DEditorPlugin(EditorNode *p_node) {
	path = nullptr;
	editor = p_node;
	singleton = this;
	mirror_handle_angle = true;
	mirror_handle_length = true;

	Ref<Path3DGizmoPlugin> gizmo_plugin;
	gizmo_plugin.instantiate();
	Node3DEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);

	sep = memnew(VSeparator);
	sep->hide();
	Node3DEditor::get_singleton()->add_control_to_menu_panel(sep);
	curve_edit = memnew(Button);
	curve_edit->set_flat(true);
	curve_edit->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon(SNAME("CurveEdit"), SNAME("EditorIcons")));
	curve_edit->set_toggle_mode(true);
	curve_edit->hide();
	curve_edit->set_focus_mode(Control::FOCUS_NONE);
	curve_edit->set_tooltip(TTR("Select Points") + "\n" + TTR("Shift+Drag: Select Control Points") + "\n" + keycode_get_string(KEY_MASK_CMD) + TTR("Click: Add Point") + "\n" + TTR("Right Click: Delete Point"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(curve_edit);
	curve_create = memnew(Button);
	curve_create->set_flat(true);
	curve_create->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon(SNAME("CurveCreate"), SNAME("EditorIcons")));
	curve_create->set_toggle_mode(true);
	curve_create->hide();
	curve_create->set_focus_mode(Control::FOCUS_NONE);
	curve_create->set_tooltip(TTR("Add Point (in empty space)") + "\n" + TTR("Split Segment (in curve)"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(curve_create);
	curve_del = memnew(Button);
	curve_del->set_flat(true);
	curve_del->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon(SNAME("CurveDelete"), SNAME("EditorIcons")));
	curve_del->set_toggle_mode(true);
	curve_del->hide();
	curve_del->set_focus_mode(Control::FOCUS_NONE);
	curve_del->set_tooltip(TTR("Delete Point"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(curve_del);
	curve_close = memnew(Button);
	curve_close->set_flat(true);
	curve_close->set_icon(EditorNode::get_singleton()->get_gui_base()->get_theme_icon(SNAME("CurveClose"), SNAME("EditorIcons")));
	curve_close->hide();
	curve_close->set_focus_mode(Control::FOCUS_NONE);
	curve_close->set_tooltip(TTR("Close Curve"));
	Node3DEditor::get_singleton()->add_control_to_menu_panel(curve_close);

	PopupMenu *menu;

	handle_menu = memnew(MenuButton);
	handle_menu->set_text(TTR("Options"));
	handle_menu->hide();
	Node3DEditor::get_singleton()->add_control_to_menu_panel(handle_menu);

	menu = handle_menu->get_popup();
	menu->add_check_item(TTR("Mirror Handle Angles"));
	menu->set_item_checked(HANDLE_OPTION_ANGLE, mirror_handle_angle);
	menu->add_check_item(TTR("Mirror Handle Lengths"));
	menu->set_item_checked(HANDLE_OPTION_LENGTH, mirror_handle_length);
	menu->connect("id_pressed", callable_mp(this, &Path3DEditorPlugin::_handle_option_pressed));

	curve_edit->set_pressed(true);
}

Path3DEditorPlugin::~Path3DEditorPlugin() {
}

Ref<EditorNode3DGizmo> Path3DGizmoPlugin::create_gizmo(Node3D *p_spatial) {
	Ref<Path3DGizmo> ref;

	Path3D *path = Object::cast_to<Path3D>(p_spatial);
	if (path) {
		ref = Ref<Path3DGizmo>(memnew(Path3DGizmo(path)));
	}

	return ref;
}

String Path3DGizmoPlugin::get_gizmo_name() const {
	return "Path3D";
}

int Path3DGizmoPlugin::get_priority() const {
	return -1;
}

Path3DGizmoPlugin::Path3DGizmoPlugin() {
	Color path_color = EDITOR_DEF("editors/3d_gizmos/gizmo_colors/path", Color(0.5, 0.5, 1.0, 0.8));
	create_material("path_material", path_color);
	create_material("path_thin_material", Color(0.5, 0.5, 0.5));
	create_handle_material("handles", false, Node3DEditor::get_singleton()->get_theme_icon(SNAME("EditorPathSmoothHandle"), SNAME("EditorIcons")));
	create_handle_material("sec_handles", false, Node3DEditor::get_singleton()->get_theme_icon(SNAME("EditorCurveHandle"), SNAME("EditorIcons")));
}
