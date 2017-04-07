/*************************************************************************/
/*  path_2d_editor_plugin.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "path_2d_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_settings.h"
#include "os/file_access.h"
#include "os/keyboard.h"
void Path2DEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			//button_create->set_icon( get_icon("Edit","EditorIcons"));
			//button_edit->set_icon( get_icon("MovePoint","EditorIcons"));
			//set_pressed_button(button_edit);
			//button_edit->set_pressed(true);

		} break;
		case NOTIFICATION_FIXED_PROCESS: {

		} break;
	}
}
void Path2DEditor::_node_removed(Node *p_node) {

	if (p_node == node) {
		node = NULL;
		hide();
	}
}

bool Path2DEditor::forward_gui_input(const InputEvent &p_event) {

	if (!node)
		return false;

	if (!node->is_visible_in_tree())
		return false;

	if (!node->get_curve().is_valid())
		return false;

	switch (p_event.type) {

		case InputEvent::MOUSE_BUTTON: {

			const InputEventMouseButton &mb = p_event.mouse_button;

			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

			Vector2 gpoint = Point2(mb.x, mb.y);
			Vector2 cpoint = !mb.mod.alt ? canvas_item_editor->snap_point(xform.affine_inverse().xform(gpoint)) : node->get_global_transform().affine_inverse().xform(canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint)));

			//first check if a point is to be added (segment split)
			real_t grab_treshold = EDITOR_DEF("editors/poly_editor/point_grab_radius", 8);

			// Test move point!!

			if (mb.pressed && action == ACTION_NONE) {

				Ref<Curve2D> curve = node->get_curve();

				for (int i = 0; i < curve->get_point_count(); i++) {

					bool pointunder = false;

					{
						Point2 p = xform.xform(curve->get_point_pos(i));
						if (gpoint.distance_to(p) < grab_treshold) {

							if (mb.button_index == BUTTON_LEFT && !mb.mod.shift && mode == MODE_EDIT) {

								action = ACTION_MOVING_POINT;
								action_point = i;
								moving_from = curve->get_point_pos(i);
								moving_screen_from = gpoint;
								return true;
							} else if ((mb.button_index == BUTTON_RIGHT && mode == MODE_EDIT) || (mb.button_index == BUTTON_LEFT && mode == MODE_DELETE)) {

								undo_redo->create_action(TTR("Remove Point from Curve"));
								undo_redo->add_do_method(curve.ptr(), "remove_point", i);
								undo_redo->add_undo_method(curve.ptr(), "add_point", curve->get_point_pos(i), curve->get_point_in(i), curve->get_point_out(i), i);
								undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
								undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
								undo_redo->commit_action();
								return true;
							} else
								pointunder = true;
						}
					}

					if (mb.button_index == BUTTON_LEFT && i < (curve->get_point_count() - 1)) {
						Point2 p = xform.xform(curve->get_point_pos(i) + curve->get_point_out(i));
						if (gpoint.distance_to(p) < grab_treshold && (mode == MODE_EDIT || mode == MODE_EDIT_CURVE)) {

							action = ACTION_MOVING_OUT;
							action_point = i;
							moving_from = curve->get_point_out(i);
							moving_screen_from = gpoint;
							return true;
						}
					}

					if (mb.button_index == BUTTON_LEFT && i > 0) {
						Point2 p = xform.xform(curve->get_point_pos(i) + curve->get_point_in(i));
						if (gpoint.distance_to(p) < grab_treshold && (mode == MODE_EDIT || mode == MODE_EDIT_CURVE)) {

							action = ACTION_MOVING_IN;
							action_point = i;
							moving_from = curve->get_point_in(i);
							moving_screen_from = gpoint;
							return true;
						}
					}

					if (pointunder)
						return true;
				}
			}

			// Test add point in empty space!

			if (mb.pressed && mb.button_index == BUTTON_LEFT && ((mb.mod.command && mode == MODE_EDIT) || mode == MODE_CREATE)) {

				Ref<Curve2D> curve = node->get_curve();

				undo_redo->create_action(TTR("Add Point to Curve"));
				undo_redo->add_do_method(curve.ptr(), "add_point", cpoint);
				undo_redo->add_undo_method(curve.ptr(), "remove_point", curve->get_point_count());
				undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
				undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
				undo_redo->commit_action();

				action = ACTION_MOVING_POINT;
				action_point = curve->get_point_count() - 1;
				moving_from = curve->get_point_pos(action_point);
				moving_screen_from = gpoint;

				canvas_item_editor->get_viewport_control()->update();

				return true;
			}

			if (!mb.pressed && mb.button_index == BUTTON_LEFT && action != ACTION_NONE) {

				Ref<Curve2D> curve = node->get_curve();

				Vector2 new_pos = moving_from + xform.affine_inverse().basis_xform(gpoint - moving_screen_from);
				switch (action) {

					case ACTION_MOVING_POINT: {

						undo_redo->create_action(TTR("Move Point in Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_pos", action_point, cpoint);
						undo_redo->add_undo_method(curve.ptr(), "set_point_pos", action_point, moving_from);
						undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->commit_action();

					} break;
					case ACTION_MOVING_IN: {

						undo_redo->create_action(TTR("Move In-Control in Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_in", action_point, new_pos);
						undo_redo->add_undo_method(curve.ptr(), "set_point_in", action_point, moving_from);
						undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->commit_action();

					} break;
					case ACTION_MOVING_OUT: {

						undo_redo->create_action(TTR("Move Out-Control in Curve"));
						undo_redo->add_do_method(curve.ptr(), "set_point_out", action_point, new_pos);
						undo_redo->add_undo_method(curve.ptr(), "set_point_out", action_point, moving_from);
						undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
						undo_redo->commit_action();

					} break;
				}

				action = ACTION_NONE;

				return true;
			}

#if 0
			switch(mode) {


				case MODE_CREATE: {

					if (mb.button_index==BUTTON_LEFT && mb.pressed) {


						if (!wip_active) {

							wip.clear();
							wip.push_back( canvas_item_editor->snap_point(cpoint) );
							wip_active=true;
							edited_point_pos=canvas_item_editor->snap_point(cpoint);
							canvas_item_editor->update();
							edited_point=1;
							return true;
						} else {

							if (wip.size()>1 && xform.xform(wip[0]).distance_to(gpoint)<grab_treshold) {
								//wip closed
								_wip_close();

								return true;
							} else {

								wip.push_back( canvas_item_editor->snap_point(cpoint) );
								edited_point=wip.size();
								canvas_item_editor->update();
								return true;

								//add wip point
							}
						}
					} else if (mb.button_index==BUTTON_RIGHT && mb.pressed && wip_active) {
						_wip_close();
					}



				} break;

				case MODE_EDIT: {

					if (mb.button_index==BUTTON_LEFT) {
						if (mb.pressed) {

							if (mb.mod.control) {


								if (poly.size() < 3) {

									undo_redo->create_action(TTR("Edit Poly"));
									undo_redo->add_undo_method(node,"set_polygon",poly);
									poly.push_back(cpoint);
									undo_redo->add_do_method(node,"set_polygon",poly);
									undo_redo->add_do_method(canvas_item_editor->get_viewport_control(),"update");
									undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(),"update");
									undo_redo->commit_action();
									return true;
								}

								//search edges
								int closest_idx=-1;
								Vector2 closest_pos;
								real_t closest_dist=1e10;
								for(int i=0;i<poly.size();i++) {

									Vector2 points[2] ={ xform.xform(poly[i]),
										xform.xform(poly[(i+1)%poly.size()]) };

									Vector2 cp = Geometry::get_closest_point_to_segment_2d(gpoint,points);
									if (cp.distance_squared_to(points[0])<CMP_EPSILON2 || cp.distance_squared_to(points[1])<CMP_EPSILON2)
										continue; //not valid to reuse point

									real_t d = cp.distance_to(gpoint);
									if (d<closest_dist && d<grab_treshold) {
										closest_dist=d;
										closest_pos=cp;
										closest_idx=i;
									}


								}

								if (closest_idx>=0) {

									pre_move_edit=poly;
									poly.insert(closest_idx+1,canvas_item_editor->snap_point(xform.affine_inverse().xform(closest_pos)));
									edited_point=closest_idx+1;
									edited_point_pos=canvas_item_editor->snap_point(xform.affine_inverse().xform(closest_pos));
									node->set_polygon(poly);
									canvas_item_editor->update();
									return true;
								}
							} else {

								//look for points to move

								int closest_idx=-1;
								Vector2 closest_pos;
								real_t closest_dist=1e10;
								for(int i=0;i<poly.size();i++) {

									Vector2 cp =xform.xform(poly[i]);

									real_t d = cp.distance_to(gpoint);
									if (d<closest_dist && d<grab_treshold) {
										closest_dist=d;
										closest_pos=cp;
										closest_idx=i;
									}

								}

								if (closest_idx>=0) {

									pre_move_edit=poly;
									edited_point=closest_idx;
									edited_point_pos=xform.affine_inverse().xform(closest_pos);
									canvas_item_editor->update();
									return true;
								}
							}
						} else {

							if (edited_point!=-1) {

								//apply

								ERR_FAIL_INDEX_V(edited_point,poly.size(),false);
								poly[edited_point]=edited_point_pos;
								undo_redo->create_action(TTR("Edit Poly"));
								undo_redo->add_do_method(node,"set_polygon",poly);
								undo_redo->add_undo_method(node,"set_polygon",pre_move_edit);
								undo_redo->add_do_method(canvas_item_editor->get_viewport_control(),"update");
								undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(),"update");
								undo_redo->commit_action();

								edited_point=-1;
								return true;
							}
						}
					} if (mb.button_index==BUTTON_RIGHT && mb.pressed && edited_point==-1) {



						int closest_idx=-1;
						Vector2 closest_pos;
						real_t closest_dist=1e10;
						for(int i=0;i<poly.size();i++) {

							Vector2 cp =xform.xform(poly[i]);

							real_t d = cp.distance_to(gpoint);
							if (d<closest_dist && d<grab_treshold) {
								closest_dist=d;
								closest_pos=cp;
								closest_idx=i;
							}

						}

						if (closest_idx>=0) {


							undo_redo->create_action(TTR("Edit Poly (Remove Point)"));
							undo_redo->add_undo_method(node,"set_polygon",poly);
							poly.remove(closest_idx);
							undo_redo->add_do_method(node,"set_polygon",poly);
							undo_redo->add_do_method(canvas_item_editor->get_viewport_control(),"update");
							undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(),"update");
							undo_redo->commit_action();
							return true;
						}

					}



				} break;
			}

#endif
		} break;
		case InputEvent::MOUSE_MOTION: {

			const InputEventMouseMotion &mm = p_event.mouse_motion;

			if (action != ACTION_NONE) {

				Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
				Vector2 gpoint = Point2(mm.x, mm.y);
				Vector2 cpoint = !mm.mod.alt ? canvas_item_editor->snap_point(xform.affine_inverse().xform(gpoint)) : node->get_global_transform().affine_inverse().xform(canvas_item_editor->snap_point(canvas_item_editor->get_canvas_transform().affine_inverse().xform(gpoint)));

				Ref<Curve2D> curve = node->get_curve();

				Vector2 new_pos = moving_from + xform.affine_inverse().basis_xform(gpoint - moving_screen_from);

				switch (action) {

					case ACTION_MOVING_POINT: {

						curve->set_point_pos(action_point, cpoint);
					} break;
					case ACTION_MOVING_IN: {

						curve->set_point_in(action_point, new_pos);

					} break;
					case ACTION_MOVING_OUT: {

						curve->set_point_out(action_point, new_pos);

					} break;
				}

				canvas_item_editor->get_viewport_control()->update();
				return true;
			}

#if 0
			if (edited_point!=-1 && (wip_active || mm.button_mask&BUTTON_MASK_LEFT)) {


				Matrix32 xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();

				Vector2 gpoint = Point2(mm.x,mm.y);
				edited_point_pos = canvas_item_editor->snap_point(xform.affine_inverse().xform(gpoint));
				canvas_item_editor->update();

			}
#endif
		} break;
	}

	return false;
}
void Path2DEditor::_canvas_draw() {

	if (!node)
		return;

	if (!node->is_visible_in_tree())
		return;

	if (!node->get_curve().is_valid())
		return;

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
	Ref<Texture> handle = get_icon("EditorHandle", "EditorIcons");
	Size2 handle_size = handle->get_size();

	Ref<Curve2D> curve = node->get_curve();

	int len = curve->get_point_count();
	Control *vpc = canvas_item_editor->get_viewport_control();

	for (int i = 0; i < len; i++) {

		Vector2 point = xform.xform(curve->get_point_pos(i));
		vpc->draw_texture_rect(handle, Rect2(point - handle_size * 0.5, handle_size), false, Color(1, 1, 1, 1));

		if (i < len - 1) {
			Vector2 pointout = xform.xform(curve->get_point_pos(i) + curve->get_point_out(i));
			vpc->draw_line(point, pointout, Color(0.5, 0.5, 1.0, 0.8), 1.0);
			vpc->draw_texture_rect(handle, Rect2(pointout - handle_size * 0.5, handle_size), false, Color(1, 0.5, 1, 0.3));
		}

		if (i > 0) {
			Vector2 pointin = xform.xform(curve->get_point_pos(i) + curve->get_point_in(i));
			vpc->draw_line(point, pointin, Color(0.5, 0.5, 1.0, 0.8), 1.0);
			vpc->draw_texture_rect(handle, Rect2(pointin - handle_size * 0.5, handle_size), false, Color(1, 0.5, 1, 0.3));
		}
	}
}

void Path2DEditor::_node_visibility_changed() {
	if (!node)
		return;

	canvas_item_editor->get_viewport_control()->update();
}

void Path2DEditor::edit(Node *p_path2d) {

	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton();
	}

	if (p_path2d) {

		node = p_path2d->cast_to<Path2D>();
		if (!canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->connect("draw", this, "_canvas_draw");
		if (!node->is_connected("visibility_changed", this, "_node_visibility_changed"))
			node->connect("visibility_changed", this, "_node_visibility_changed");

	} else {

		if (canvas_item_editor->get_viewport_control()->is_connected("draw", this, "_canvas_draw"))
			canvas_item_editor->get_viewport_control()->disconnect("draw", this, "_canvas_draw");

		// node may have been deleted at this point
		if (node && node->is_connected("visibility_changed", this, "_node_visibility_changed"))
			node->disconnect("visibility_changed", this, "_node_visibility_changed");
		node = NULL;
	}
}

void Path2DEditor::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("_menu_option"),&Path2DEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_canvas_draw"), &Path2DEditor::_canvas_draw);
	ClassDB::bind_method(D_METHOD("_node_visibility_changed"), &Path2DEditor::_node_visibility_changed);
	ClassDB::bind_method(D_METHOD("_mode_selected"), &Path2DEditor::_mode_selected);
}

void Path2DEditor::_mode_selected(int p_mode) {

	if (p_mode == MODE_CREATE) {

		curve_create->set_pressed(true);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_EDIT) {

		curve_create->set_pressed(false);
		curve_edit->set_pressed(true);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_EDIT_CURVE) {

		curve_create->set_pressed(false);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(true);
		curve_del->set_pressed(false);
	} else if (p_mode == MODE_DELETE) {

		curve_create->set_pressed(false);
		curve_edit->set_pressed(false);
		curve_edit_curve->set_pressed(false);
		curve_del->set_pressed(true);
	} else if (p_mode == ACTION_CLOSE) {

		//?

		if (!node->get_curve().is_valid())
			return;
		if (node->get_curve()->get_point_count() < 3)
			return;

		Vector2 begin = node->get_curve()->get_point_pos(0);
		Vector2 end = node->get_curve()->get_point_pos(node->get_curve()->get_point_count() - 1);
		if (begin.distance_to(end) < CMP_EPSILON)
			return;

		undo_redo->create_action(TTR("Remove Point from Curve"));
		undo_redo->add_do_method(node->get_curve().ptr(), "add_point", begin);
		undo_redo->add_undo_method(node->get_curve().ptr(), "remove_point", node->get_curve()->get_point_count());
		undo_redo->add_do_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->add_undo_method(canvas_item_editor->get_viewport_control(), "update");
		undo_redo->commit_action();
		return;
	}

	mode = Mode(p_mode);
}

Path2DEditor::Path2DEditor(EditorNode *p_editor) {

	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	mode = MODE_EDIT;

	action = ACTION_NONE;
#if 0
	options = memnew( MenuButton );
	add_child(options);
	options->set_area_as_parent_rect();
	options->set_text("Polygon");
	//options->get_popup()->add_item("Parse BBCode",PARSE_BBCODE);
	options->get_popup()->connect("id_pressed", this,"_menu_option");
#endif

	base_hb = memnew(HBoxContainer);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(base_hb);

	sep = memnew(VSeparator);
	base_hb->add_child(sep);
	curve_edit = memnew(ToolButton);
	curve_edit->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveEdit", "EditorIcons"));
	curve_edit->set_toggle_mode(true);
	curve_edit->set_focus_mode(Control::FOCUS_NONE);
	curve_edit->set_tooltip(TTR("Select Points") + "\n" + TTR("Shift+Drag: Select Control Points") + "\n" + keycode_get_string(KEY_MASK_CMD) + TTR("Click: Add Point") + "\n" + TTR("Right Click: Delete Point"));
	curve_edit->connect("pressed", this, "_mode_selected", varray(MODE_EDIT));
	base_hb->add_child(curve_edit);
	curve_edit_curve = memnew(ToolButton);
	curve_edit_curve->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveCurve", "EditorIcons"));
	curve_edit_curve->set_toggle_mode(true);
	curve_edit_curve->set_focus_mode(Control::FOCUS_NONE);
	curve_edit_curve->set_tooltip(TTR("Select Control Points (Shift+Drag)"));
	curve_edit_curve->connect("pressed", this, "_mode_selected", varray(MODE_EDIT_CURVE));
	base_hb->add_child(curve_edit_curve);
	curve_create = memnew(ToolButton);
	curve_create->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveCreate", "EditorIcons"));
	curve_create->set_toggle_mode(true);
	curve_create->set_focus_mode(Control::FOCUS_NONE);
	curve_create->set_tooltip(TTR("Add Point (in empty space)") + "\n" + TTR("Split Segment (in curve)"));
	curve_create->connect("pressed", this, "_mode_selected", varray(MODE_CREATE));
	base_hb->add_child(curve_create);
	curve_del = memnew(ToolButton);
	curve_del->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveDelete", "EditorIcons"));
	curve_del->set_toggle_mode(true);
	curve_del->set_focus_mode(Control::FOCUS_NONE);
	curve_del->set_tooltip(TTR("Delete Point"));
	curve_del->connect("pressed", this, "_mode_selected", varray(MODE_DELETE));
	base_hb->add_child(curve_del);
	curve_close = memnew(ToolButton);
	curve_close->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("CurveClose", "EditorIcons"));
	curve_close->set_focus_mode(Control::FOCUS_NONE);
	curve_close->set_tooltip(TTR("Close Curve"));
	curve_close->connect("pressed", this, "_mode_selected", varray(ACTION_CLOSE));
	base_hb->add_child(curve_close);
	base_hb->hide();

	curve_edit->set_pressed(true);
}

void Path2DEditorPlugin::edit(Object *p_object) {

	path2d_editor->edit(p_object->cast_to<Node>());
}

bool Path2DEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("Path2D");
}

void Path2DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		path2d_editor->show();
		path2d_editor->base_hb->show();

	} else {

		path2d_editor->hide();
		path2d_editor->base_hb->hide();
		path2d_editor->edit(NULL);
	}
}

Path2DEditorPlugin::Path2DEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	path2d_editor = memnew(Path2DEditor(p_node));
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(path2d_editor);
	path2d_editor->hide();
}

Path2DEditorPlugin::~Path2DEditorPlugin() {
}
