/**************************************************************************/
/*  grid_map_editor_plugin.cpp                                            */
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

#include "grid_map_editor_plugin.h"

#ifdef TOOLS_ENABLED

#include "core/os/keyboard.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/themes/editor_scale.h"
#include "scene/3d/camera_3d.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/main/window.h"
#include "scene/resources/3d/primitive_meshes.h"

void GridMapEditor::_configure() {
	if (!node) {
		return;
	}

	update_grid();
}

void GridMapEditor::_menu_option(int p_option) {
	switch (p_option) {
		case MENU_OPTION_PREV_LEVEL: {
			floor->set_value(floor->get_value() - 1);
			if (selection.active && input_action == INPUT_SELECT) {
				_update_selection();
			}
		} break;

		case MENU_OPTION_NEXT_LEVEL: {
			floor->set_value(floor->get_value() + 1);
			if (selection.active && input_action == INPUT_SELECT) {
				_update_selection();
			}
		} break;

		case MENU_OPTION_X_AXIS:
			edit_axis = AXIS_X;
			update_grid();
			break;
		case MENU_OPTION_Y_AXIS:
			edit_axis = AXIS_Y;
			update_grid();
			break;
		case MENU_OPTION_Z_AXIS:
			edit_axis = AXIS_Z;
			update_grid();
			break;
		case MENU_OPTION_Q_AXIS:
			edit_axis = AXIS_Q;
			update_grid();
			break;
		case MENU_OPTION_R_AXIS:
			edit_axis = AXIS_R;
			update_grid();
			break;
		case MENU_OPTION_S_AXIS:
			edit_axis = AXIS_S;
			update_grid();
			break;
		case MENU_OPTION_ROTATE_AXIS_CW:
			switch (edit_axis) {
				case AXIS_R:
					edit_axis = AXIS_Q;
					break;
				case AXIS_Q:
					edit_axis = AXIS_X;
					break;
				case AXIS_X:
					edit_axis = AXIS_S;
					break;
				default:
					edit_axis = AXIS_R;
					break;
			}
			update_grid();
			break;
		case MENU_OPTION_ROTATE_AXIS_CCW:
			switch (edit_axis) {
				case AXIS_X:
					edit_axis = AXIS_Q;
					break;
				case AXIS_Q:
					edit_axis = AXIS_R;
					break;
				case AXIS_R:
					edit_axis = AXIS_S;
					break;
				default:
					edit_axis = AXIS_X;
					break;
			}
			update_grid();
			break;

		case MENU_OPTION_CURSOR_ROTATE_Y: {
			Basis r;
			real_t rotation = Math_PI / (node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? 2.0 : 3.0);

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), -rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), -rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_X: {
			Basis r;
			real_t rotation = node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? (Math_PI / 2.0) : Math_PI;

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), -rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), -rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_Z: {
			Basis r;
			real_t rotation = node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? (Math_PI / 2.0) : Math_PI;

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), -rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), -rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Y: {
			Basis r;
			real_t rotation = Math_PI / (node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? 2.0 : 3.0);

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_X: {
			Basis r;
			real_t rotation = node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? (Math_PI / 2.0) : Math_PI;

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Z: {
			Basis r;
			real_t rotation = node->get_cell_shape() == GridMap::CELL_SHAPE_SQUARE ? (Math_PI / 2.0) : Math_PI;

			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), rotation);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), rotation);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_CLEAR_ROTATION: {
			if (input_action == INPUT_PASTE) {
				paste_indicator.orientation = 0;
				_update_paste_indicator();
				break;
			}

			cursor_rot = 0;
			_update_cursor_transform();
		} break;

		case MENU_OPTION_PASTE_SELECTS: {
			int idx = options->get_popup()->get_item_index(MENU_OPTION_PASTE_SELECTS);
			options->get_popup()->set_item_checked(idx, !options->get_popup()->is_item_checked(idx));
		} break;

		case MENU_OPTION_SELECTION_DUPLICATE:
		case MENU_OPTION_SELECTION_CUT: {
			if (!(selection.active && input_action == INPUT_NONE)) {
				break;
			}

			_set_clipboard_data();

			if (p_option == MENU_OPTION_SELECTION_CUT) {
				_delete_selection();
			}

			input_action = INPUT_PASTE;
			paste_indicator.current_cell = selection.begin;
			paste_indicator.orientation = 0;
			_update_paste_indicator();
		} break;
		case MENU_OPTION_SELECTION_CLEAR: {
			if (!selection.active) {
				break;
			}

			_delete_selection();

		} break;
		case MENU_OPTION_SELECTION_FILL: {
			if (!selection.active) {
				return;
			}

			_fill_selection();

		} break;
		case MENU_OPTION_GRIDMAP_SETTINGS: {
			settings_dialog->popup_centered(settings_vbc->get_combined_minimum_size() + Size2(50, 50) * EDSCALE);
		} break;
	}
}

void GridMapEditor::_update_cursor_transform() {
	cursor_transform = Transform3D();
	cursor_transform.origin = node->map_to_local(cursor_cell);
	cursor_transform.basis = node->get_basis_with_orthogonal_index(cursor_rot);
	cursor_transform.basis *= node->get_cell_scale();
	cursor_transform = node->get_global_transform() * cursor_transform;

	if (selected_palette >= 0) {
		if (node && !node->get_mesh_library().is_null()) {
			cursor_transform *= node->get_mesh_library()->get_item_mesh_transform(selected_palette);
		}
	}

	if (cursor_instance.is_valid()) {
		RenderingServer::get_singleton()->instance_set_transform(cursor_instance, cursor_transform);
		RenderingServer::get_singleton()->instance_set_visible(cursor_instance, cursor_visible);
	}
}

void GridMapEditor::_update_selection() {
	RenderingServer *rs = RS::get_singleton();

	if (selection_instance.is_valid()) {
		rs->free(selection_instance);
	}

	if (!selection.active) {
		return;
	}

	// general scaling and translation for the cell mesh.  The translation is
	// necessary because the meshes are centered around the origin, and we
	// need to shift the cell up.
	//
	// XXX need to accommodate center_x/center_z also?
	Transform3D cell_transform = Transform3D()
										 .scaled_local(node->get_cell_size())
										 .translated(Vector3(0, node->get_center_y() ? 0 : (node->get_cell_size().y / 2.0), 0));

	// when performing selection on the Q & S axis (hex shaped cells) we need
	// the begin cell index to limit the selection to the desired plane.
	Vector3i begin = node->local_to_map(selection.begin);

	// get the cells in our selection area
	TypedArray<Vector3i> cells = node->local_region_to_map(selection.begin, selection.end);

	// add the cells to our selection multimesh
	rs->multimesh_allocate_data(selection_multimesh, cells.size(), RS::MULTIMESH_TRANSFORM_3D);
	for (int i = 0; i < cells.size(); i++) {
		Vector3i cell = cells[i];
		switch (edit_axis) {
			case AXIS_Q:
				if (cell.x != begin.x) {
					continue;
				}
				break;
			case AXIS_S:
				if (-cell.x - cell.z != -begin.x - begin.z) {
					continue;
				}
				break;

			default:
				break;
		}
		rs->multimesh_instance_set_transform(selection_multimesh, i,
				cell_transform.translated(node->map_to_local(cell)));
	}

	// create an instance of the multimesh with the transform of our node
	selection_instance = rs->instance_create2(selection_multimesh, get_tree()->get_root()->get_world_3d()->get_scenario());
	rs->instance_set_transform(selection_instance, node->get_global_transform());
	rs->instance_set_layer_mask(selection_instance, Node3DEditorViewport::MISC_TOOL_LAYER);
}

void GridMapEditor::_set_selection(bool p_active, const Vector3 &p_begin, const Vector3 &p_end) {
	selection.active = p_active;
	selection.begin = p_begin;
	selection.end = p_end;

	if (is_visible_in_tree()) {
		_update_selection();
	}

	options->get_popup()->set_item_disabled(options->get_popup()->get_item_index(MENU_OPTION_SELECTION_CLEAR), !selection.active);
	options->get_popup()->set_item_disabled(options->get_popup()->get_item_index(MENU_OPTION_SELECTION_CUT), !selection.active);
	options->get_popup()->set_item_disabled(options->get_popup()->get_item_index(MENU_OPTION_SELECTION_DUPLICATE), !selection.active);
	options->get_popup()->set_item_disabled(options->get_popup()->get_item_index(MENU_OPTION_SELECTION_FILL), !selection.active);
}

bool GridMapEditor::do_input_action(Camera3D *p_camera, const Point2 &p_point, bool p_click) {
	if (!spatial_editor) {
		return false;
	}

	if (selected_palette < 0 && input_action != INPUT_PICK && input_action != INPUT_SELECT && input_action != INPUT_PASTE) {
		return false;
	}
	if (mesh_library.is_null()) {
		return false;
	}
	if (input_action != INPUT_PICK && input_action != INPUT_SELECT && input_action != INPUT_PASTE && !mesh_library->has_item(selected_palette)) {
		return false;
	}

	Camera3D *camera = p_camera;
	Vector3 from = camera->project_ray_origin(p_point);
	Vector3 normal = camera->project_ray_normal(p_point);
	Transform3D local_xform = node->get_global_transform().affine_inverse();
	Vector<Plane> planes = camera->get_frustum();
	from = local_xform.xform(from);
	normal = local_xform.basis.xform(normal).normalized();

	Vector3 inters;
	if (!edit_plane.intersects_segment(from, from + normal * settings_pick_distance->get_value(), &inters)) {
		return false;
	}

	// Make sure the intersection is inside the frustum planes, to avoid
	// Painting on invisible regions.
	for (int i = 0; i < planes.size(); i++) {
		Plane fp = local_xform.xform(planes[i]);
		if (fp.is_point_over(inters)) {
			return false;
		}
	}

	Vector3i cell = node->local_to_map(inters);

	if (cursor_instance.is_valid()) {
		cursor_cell = cell;
		cursor_visible = true;

		if (input_action == INPUT_SELECT || input_action == INPUT_PASTE) {
			cursor_visible = false;
		}

		_update_cursor_transform();
	}

	if (input_action == INPUT_PASTE) {
		paste_indicator.current_cell = cell;
		_update_paste_indicator();

	} else if (input_action == INPUT_SELECT) {
		if (p_click) {
			selection.begin = inters;
		}
		selection.end = inters;
		selection.active = true;
		_update_selection();

		return true;
	} else if (input_action == INPUT_PICK) {
		int item = node->get_cell_item(cell);
		if (item >= 0) {
			selected_palette = item;

			// Clear the filter if picked an item that's filtered out.
			int index = mesh_library_palette->find_metadata(item);
			if (index == -1) {
				search_box->clear();
			}

			// This will select `selected_palette` in the ItemList when possible.
			update_palette();

			_update_cursor_instance();
		}
		return true;
	}

	if (input_action == INPUT_PAINT) {
		SetItem si;
		si.position = cell;
		si.new_value = selected_palette;
		si.new_orientation = cursor_rot;
		si.old_value = node->get_cell_item(cell);
		si.old_orientation = node->get_cell_item_orientation(cell);
		set_items.push_back(si);
		node->set_cell_item(cell, selected_palette, cursor_rot);
		return true;
	} else if (input_action == INPUT_ERASE) {
		SetItem si;
		si.position = Vector3i(cell[0], cell[1], cell[2]);
		si.new_value = -1;
		si.new_orientation = 0;
		si.old_value = node->get_cell_item(cell);
		si.old_orientation = node->get_cell_item_orientation(cell);
		set_items.push_back(si);
		node->set_cell_item(cell, -1);
		return true;
	}

	return false;
}

void GridMapEditor::_delete_selection() {
	if (!selection.active) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("GridMap Delete Selection"));

	TypedArray<Vector3i> cells = node->local_region_to_map(selection.begin, selection.end);
	for (const Vector3i cell : cells) {
		undo_redo->add_do_method(node, "set_cell_item", cell, GridMap::INVALID_CELL_ITEM);
		undo_redo->add_undo_method(node, "set_cell_item", cell, node->get_cell_item(cell), node->get_cell_item_orientation(cell));
	}

	undo_redo->add_do_method(this, "_set_selection", !selection.active, selection.begin, selection.end);
	undo_redo->add_undo_method(this, "_set_selection", selection.active, selection.begin, selection.end);
	undo_redo->commit_action();
}

void GridMapEditor::_fill_selection() {
	if (!selection.active) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("GridMap Fill Selection"));

	TypedArray<Vector3i> cells = node->local_region_to_map(selection.begin, selection.end);
	for (const Vector3i cell : cells) {
		undo_redo->add_do_method(node, "set_cell_item", cell, selected_palette, cursor_rot);
		undo_redo->add_undo_method(node, "set_cell_item", cell, node->get_cell_item(cell), node->get_cell_item_orientation(cell));
	}

	undo_redo->add_do_method(this, "_set_selection", !selection.active, selection.begin, selection.end);
	undo_redo->add_undo_method(this, "_set_selection", selection.active, selection.begin, selection.end);
	undo_redo->commit_action();
}

void GridMapEditor::_clear_clipboard_data() {
	for (const ClipboardItem &E : clipboard_items) {
		RenderingServer::get_singleton()->free(E.instance);
	}

	clipboard_items.clear();
}

void GridMapEditor::_set_clipboard_data() {
	_clear_clipboard_data();

	Ref<MeshLibrary> meshLibrary = node->get_mesh_library();

	RID root = get_tree()->get_root()->get_world_3d()->get_scenario();

	Vector3 begin = node->map_to_local(node->local_to_map(selection.begin));
	Vector3 end = node->map_to_local(node->local_to_map(selection.end));
	Vector3 selection_center = (end + begin) / 2.0;
	Vector3 offset = node->map_to_local(node->local_to_map(selection_center));

	TypedArray<Vector3i> cells = node->local_region_to_map(selection.begin, selection.end);
	for (const Vector3i cell : cells) {
		int id = node->get_cell_item(cell);
		if (id == GridMap::INVALID_CELL_ITEM) {
			continue;
		}

		RID mesh = meshLibrary->get_item_mesh(id)->get_rid();
		RID instance = RS::get_singleton()->instance_create2(mesh, root);

		ClipboardItem item;
		item.cell_item = id;
		item.grid_offset = node->map_to_local(cell) - offset;
		item.orientation = node->get_cell_item_orientation(cell);
		item.instance = instance;
		clipboard_items.push_back(item);
	}
}

void GridMapEditor::_update_paste_indicator() {
	if (input_action != INPUT_PASTE) {
		_clear_clipboard_data();
		return;
	}

	Vector3 cursor = node->map_to_local(paste_indicator.current_cell);
	Basis paste_rotation = node->get_basis_with_orthogonal_index(paste_indicator.orientation);

	for (const ClipboardItem &item : clipboard_items) {
		// move the item to the cursor, then apply paste rotation, then
		// translate by the item's cell offset.
		Transform3D xf;
		xf.origin = cursor;
		xf.basis = paste_rotation * xf.basis;
		xf.translate_local(item.grid_offset);

		xf.basis = node->get_basis_with_orthogonal_index(item.orientation) * xf.basis;

		RS::get_singleton()->instance_set_transform(item.instance, xf);
	}
}

void GridMapEditor::_do_paste() {
	Vector3 cursor = node->map_to_local(paste_indicator.current_cell);
	Basis paste_rotation = node->get_basis_with_orthogonal_index(paste_indicator.orientation);

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("GridMap Paste Selection"));

	// track the bounds of the paste region in case we need to select it below
	AABB bounds;
	bounds.set_position(cursor);

	for (const ClipboardItem &item : clipboard_items) {
		// apply paste rotation & convert it to a cell index
		Vector3 position = paste_rotation.xform(item.grid_offset) + cursor;
		Vector3i cell = node->local_to_map(position);
		bounds.expand_to(position);

		// apply paste rotation to existing cell rotation to get cell orientation
		Basis cell_rotation = paste_rotation *
				node->get_basis_with_orthogonal_index(item.orientation);
		int cell_orientation = node->get_orthogonal_index_from_basis(cell_rotation);

		undo_redo->add_do_method(node, "set_cell_item", cell, item.cell_item, cell_orientation);
		undo_redo->add_undo_method(node, "set_cell_item", cell, node->get_cell_item(cell), node->get_cell_item_orientation(cell));
	}

	// if "Paste Selects" option is checked, update the selection to reflect
	// the pasted region.
	int option_index = options->get_popup()->get_item_index(MENU_OPTION_PASTE_SELECTS);
	if (options->get_popup()->is_item_checked(option_index)) {
		Vector3 begin = bounds.position, end = bounds.get_end();
		undo_redo->add_do_method(this, "_set_selection", true, begin, end);
		undo_redo->add_undo_method(this, "_set_selection", selection.active, selection.begin, selection.end);
	}
	undo_redo->commit_action();

	_clear_clipboard_data();
}

EditorPlugin::AfterGUIInput GridMapEditor::forward_spatial_input_event(Camera3D *p_camera, const Ref<InputEvent> &p_event) {
	if (!node) {
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::WHEEL_UP && (mb->is_command_or_control_pressed())) {
			if (mb->is_pressed()) {
				floor->set_value(floor->get_value() + mb->get_factor());
			}

			return EditorPlugin::AFTER_GUI_INPUT_STOP; // Eaten.
		} else if (mb->get_button_index() == MouseButton::WHEEL_DOWN && (mb->is_command_or_control_pressed())) {
			if (mb->is_pressed()) {
				floor->set_value(floor->get_value() - mb->get_factor());
			}
			return EditorPlugin::AFTER_GUI_INPUT_STOP;
		}

		if (mb->is_pressed()) {
			Node3DEditorViewport::NavigationScheme nav_scheme = (Node3DEditorViewport::NavigationScheme)EDITOR_GET("editors/3d/navigation/navigation_scheme").operator int();
			if ((nav_scheme == Node3DEditorViewport::NAVIGATION_MAYA || nav_scheme == Node3DEditorViewport::NAVIGATION_MODO) && mb->is_alt_pressed()) {
				input_action = INPUT_NONE;
			} else if (mb->get_button_index() == MouseButton::LEFT) {
				bool can_edit = (node && node->get_mesh_library().is_valid());
				if (input_action == INPUT_PASTE) {
					_do_paste();
					input_action = INPUT_NONE;
					_update_paste_indicator();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (mb->is_shift_pressed() && can_edit) {
					input_action = INPUT_SELECT;
					last_selection = selection;
				} else if (mb->is_command_or_control_pressed() && can_edit) {
					input_action = INPUT_PICK;
				} else {
					input_action = INPUT_PAINT;
					set_items.clear();
				}
			} else if (mb->get_button_index() == MouseButton::RIGHT) {
				if (input_action == INPUT_PASTE) {
					_clear_clipboard_data();
					input_action = INPUT_NONE;
					_update_paste_indicator();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (selection.active) {
					_set_selection(false);
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else {
					input_action = INPUT_ERASE;
					set_items.clear();
				}
			} else {
				return EditorPlugin::AFTER_GUI_INPUT_PASS;
			}

			if (do_input_action(p_camera, Point2(mb->get_position().x, mb->get_position().y), true)) {
				return EditorPlugin::AFTER_GUI_INPUT_STOP;
			}
			return EditorPlugin::AFTER_GUI_INPUT_PASS;
		} else {
			if ((mb->get_button_index() == MouseButton::RIGHT && input_action == INPUT_ERASE) || (mb->get_button_index() == MouseButton::LEFT && input_action == INPUT_PAINT)) {
				if (set_items.size()) {
					EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
					undo_redo->create_action(TTR("GridMap Paint"));
					for (const SetItem &si : set_items) {
						undo_redo->add_do_method(node, "set_cell_item", si.position, si.new_value, si.new_orientation);
					}
					for (List<SetItem>::Element *E = set_items.back(); E; E = E->prev()) {
						const SetItem &si = E->get();
						undo_redo->add_undo_method(node, "set_cell_item", si.position, si.old_value, si.old_orientation);
					}

					undo_redo->commit_action();
				}
				set_items.clear();
				input_action = INPUT_NONE;

				if (set_items.size() > 0) {
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}
				return EditorPlugin::AFTER_GUI_INPUT_PASS;
			}

			if (mb->get_button_index() == MouseButton::LEFT && input_action == INPUT_SELECT) {
				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				undo_redo->create_action(TTR("GridMap Selection"));
				undo_redo->add_do_method(this, "_set_selection", selection.active, selection.begin, selection.end);
				undo_redo->add_undo_method(this, "_set_selection", last_selection.active, last_selection.begin, last_selection.end);
				undo_redo->commit_action();
			}

			if (mb->get_button_index() == MouseButton::LEFT && input_action != INPUT_NONE) {
				set_items.clear();
				input_action = INPUT_NONE;
				return EditorPlugin::AFTER_GUI_INPUT_STOP;
			}
			if (mb->get_button_index() == MouseButton::RIGHT && (input_action == INPUT_ERASE || input_action == INPUT_PASTE)) {
				input_action = INPUT_NONE;
				return EditorPlugin::AFTER_GUI_INPUT_STOP;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (do_input_action(p_camera, mm->get_position(), false)) {
			return EditorPlugin::AFTER_GUI_INPUT_STOP;
		}
		return EditorPlugin::AFTER_GUI_INPUT_PASS;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_pressed()) {
			if (k->get_keycode() == Key::ESCAPE) {
				if (input_action == INPUT_PASTE) {
					_clear_clipboard_data();
					input_action = INPUT_NONE;
					_update_paste_indicator();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else if (selection.active) {
					_set_selection(false);
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				} else {
					selected_palette = -1;
					mesh_library_palette->deselect_all();
					update_palette();
					_update_cursor_instance();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}
			}

			// Consume input to avoid conflicts with other plugins.
			if (k.is_valid() && k->is_pressed() && !k->is_echo()) {
				for (int i = 0; i < options->get_popup()->get_item_count(); ++i) {
					const Ref<Shortcut> &shortcut = options->get_popup()->get_item_shortcut(i);
					if (shortcut.is_valid() && shortcut->matches_event(p_event)) {
						accept_event();
						_menu_option(options->get_popup()->get_item_id(i));
						return EditorPlugin::AFTER_GUI_INPUT_STOP;
					}
				}
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		if (pan_gesture->is_alt_pressed() && pan_gesture->is_command_or_control_pressed()) {
			const real_t delta = pan_gesture->get_delta().y * 0.5;
			accumulated_floor_delta += delta;
			int step = 0;
			if (ABS(accumulated_floor_delta) > 1.0) {
				step = SIGN(accumulated_floor_delta);
				accumulated_floor_delta -= step;
			}
			if (step) {
				floor->set_value(floor->get_value() + step);
			}
			return EditorPlugin::AFTER_GUI_INPUT_STOP;
		}
	}
	accumulated_floor_delta = 0.0;

	return EditorPlugin::AFTER_GUI_INPUT_PASS;
}

struct _CGMEItemSort {
	String name;
	int id = 0;
	_FORCE_INLINE_ bool operator<(const _CGMEItemSort &r_it) const { return name < r_it.name; }
};

void GridMapEditor::_set_display_mode(int p_mode) {
	if (display_mode == p_mode) {
		return;
	}

	if (p_mode == DISPLAY_LIST) {
		mode_list->set_pressed(true);
		mode_thumbnail->set_pressed(false);
	} else if (p_mode == DISPLAY_THUMBNAIL) {
		mode_list->set_pressed(false);
		mode_thumbnail->set_pressed(true);
	}

	display_mode = p_mode;

	update_palette();
}

void GridMapEditor::_text_changed(const String &p_text) {
	update_palette();
}

void GridMapEditor::_sbox_input(const Ref<InputEvent> &p_event) {
	// Redirect navigational key events to the item list.
	Ref<InputEventKey> key = p_event;
	if (key.is_valid()) {
		if (key->is_action("ui_up", true) || key->is_action("ui_down", true) || key->is_action("ui_page_up") || key->is_action("ui_page_down")) {
			mesh_library_palette->gui_input(key);
			search_box->accept_event();
		}
	}
}

void GridMapEditor::_mesh_library_palette_input(const Ref<InputEvent> &p_ie) {
	const Ref<InputEventMouseButton> mb = p_ie;

	// Zoom in/out using Ctrl + mouse wheel
	if (mb.is_valid() && mb->is_pressed() && mb->is_command_or_control_pressed()) {
		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_UP) {
			size_slider->set_value(size_slider->get_value() + 0.2);
		}

		if (mb->is_pressed() && mb->get_button_index() == MouseButton::WHEEL_DOWN) {
			size_slider->set_value(size_slider->get_value() - 0.2);
		}
	}
}

void GridMapEditor::_icon_size_changed(float p_value) {
	mesh_library_palette->set_icon_scale(p_value);
	update_palette();
}

void GridMapEditor::update_palette() {
	float min_size = EDITOR_GET("editors/grid_map/preview_size");
	min_size *= EDSCALE;

	mesh_library_palette->clear();
	if (display_mode == DISPLAY_THUMBNAIL) {
		mesh_library_palette->set_max_columns(0);
		mesh_library_palette->set_icon_mode(ItemList::ICON_MODE_TOP);
		mesh_library_palette->set_fixed_column_width(min_size * MAX(size_slider->get_value(), 1.5));
	} else if (display_mode == DISPLAY_LIST) {
		mesh_library_palette->set_max_columns(1);
		mesh_library_palette->set_icon_mode(ItemList::ICON_MODE_LEFT);
		mesh_library_palette->set_fixed_column_width(0);
	}

	mesh_library_palette->set_fixed_icon_size(Size2(min_size, min_size));
	mesh_library_palette->set_max_text_lines(2);

	if (mesh_library.is_null()) {
		search_box->set_text("");
		search_box->set_editable(false);
		info_message->show();
		return;
	}

	search_box->set_editable(true);
	info_message->hide();

	Vector<int> ids;
	ids = mesh_library->get_item_list();

	List<_CGMEItemSort> il;
	for (int i = 0; i < ids.size(); i++) {
		_CGMEItemSort is;
		is.id = ids[i];
		is.name = mesh_library->get_item_name(ids[i]);
		il.push_back(is);
	}
	il.sort();

	String filter = search_box->get_text().strip_edges();

	int item = 0;

	for (_CGMEItemSort &E : il) {
		int id = E.id;
		String name = mesh_library->get_item_name(id);
		Ref<Texture2D> preview = mesh_library->get_item_preview(id);

		if (name.is_empty()) {
			name = "#" + itos(id);
		}

		if (!filter.is_empty() && !filter.is_subsequence_ofn(name)) {
			continue;
		}

		mesh_library_palette->add_item("");
		if (!preview.is_null()) {
			mesh_library_palette->set_item_icon(item, preview);
			mesh_library_palette->set_item_tooltip(item, name);
		}
		mesh_library_palette->set_item_text(item, name);
		mesh_library_palette->set_item_metadata(item, id);

		if (selected_palette == id) {
			mesh_library_palette->select(item);
		}

		item++;
	}
}

void GridMapEditor::_update_mesh_library() {
	ERR_FAIL_NULL(node);

	Ref<MeshLibrary> new_mesh_library = node->get_mesh_library();
	if (new_mesh_library != mesh_library) {
		if (mesh_library.is_valid()) {
			mesh_library->disconnect_changed(callable_mp(this, &GridMapEditor::update_palette));
		}
		mesh_library = new_mesh_library;
	} else {
		return;
	}

	if (mesh_library.is_valid()) {
		mesh_library->connect_changed(callable_mp(this, &GridMapEditor::update_palette));
	}

	update_palette();
	// Update the cursor and grid in case the library is changed or removed.
	_update_cursor_instance();
	update_grid();
}

void GridMapEditor::edit(GridMap *p_gridmap) {
	if (node) {
		node->disconnect(SNAME("cell_size_changed"), callable_mp(this, &GridMapEditor::_draw_grids));
		node->disconnect(SNAME("cell_shape_changed"), callable_mp(this, &GridMapEditor::_update_cell_shape));
		node->disconnect(CoreStringName(changed), callable_mp(this, &GridMapEditor::_update_mesh_library));
		if (mesh_library.is_valid()) {
			mesh_library->disconnect_changed(callable_mp(this, &GridMapEditor::update_palette));
			mesh_library = Ref<MeshLibrary>();
		}
	}

	node = p_gridmap;

	input_action = INPUT_NONE;
	selection.active = false;
	_build_selection_meshes();
	_update_selection();
	_update_paste_indicator();
	_update_options_menu();

	spatial_editor = Object::cast_to<Node3DEditorPlugin>(EditorNode::get_singleton()->get_editor_main_screen()->get_selected_plugin());

	if (!node) {
		set_process(false);
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_visible(grid_instance[i], false);
		}

		if (cursor_instance.is_valid()) {
			RenderingServer::get_singleton()->instance_set_visible(cursor_instance, false);
		}

		return;
	}

	update_palette();
	_update_cursor_instance();

	set_process(true);

	// load any previous floor values
	TypedArray<int> floors = node->get_meta("_editor_floor_", TypedArray<int>());
	for (int i = 0; i < MIN(floors.size(), AXIS_MAX); i++) {
		edit_floor[i] = floors[i];
	}
	_draw_grids(node->get_cell_size());
	update_grid();

	node->connect(SNAME("cell_size_changed"), callable_mp(this, &GridMapEditor::_draw_grids));
	node->connect(SNAME("cell_shape_changed"), callable_mp(this, &GridMapEditor::_update_cell_shape));
	node->connect(CoreStringName(changed), callable_mp(this, &GridMapEditor::_update_mesh_library));
	_update_mesh_library();
}

void GridMapEditor::update_grid() {
	RenderingServer *rs = RS::get_singleton();
	Vector3 cell_size = node->get_cell_size();
	bool is_hex = node->get_cell_shape() == GridMap::CELL_SHAPE_HEXAGON;

	// Hex planes Q, R, and S need to offset the grid by half a cell on even
	// numbered floors.  We calculate this value here to simplify the code
	// later.
	int is_even_floor = (edit_floor[edit_axis] & 1) == 0;

	// hide the active grid
	rs->instance_set_visible(active_grid_instance, false);

	real_t cell_depth;
	Transform3D grid_transform;
	Menu menu_axis;

	// switch the edit plane and pick the new active grid and rotate if necessary
	switch (edit_axis) {
		case AXIS_X:
			// set which grid to display
			active_grid_instance = grid_instance[0];
			// set the edit plane normal, and cell depth (used by the plane)
			edit_plane.normal = Vector3(1, 0, 0);
			cell_depth = is_hex ? (SQRT3_2 * cell_size.x) : cell_size.x;
			// shift the edit grid based on which floor we are on
			if (is_hex && !is_even_floor) {
				grid_transform.translate_local(Vector3(0, 0, 1.5 * cell_size.x));
			}
			// update the menu
			menu_axis = MENU_OPTION_X_AXIS;
			break;
		case AXIS_Y:
			active_grid_instance = grid_instance[1];
			edit_plane.normal = Vector3(0, 1, 0);
			cell_depth = cell_size.y;
			menu_axis = MENU_OPTION_Y_AXIS;
			break;
		case AXIS_Z:
			active_grid_instance = grid_instance[2];
			edit_plane.normal = Vector3(0, 0, 1);
			cell_depth = cell_size.z;
			menu_axis = MENU_OPTION_Z_AXIS;
			break;
		case AXIS_Q: // hex plane, northwest to southeast
			active_grid_instance = grid_instance[2];
			edit_plane.normal = Vector3(SQRT3_2, 0, -0.5).normalized();
			cell_depth = 1.5 * cell_size.x;
			grid_transform.rotate(Vector3(0, 1, 0), -Math_PI / 3.0);
			// offset the edit grid on even numbered floors by half a cell
			grid_transform.translate_local(Vector3(is_even_floor * SQRT3_2 * cell_size.x, 0, 0));
			menu_axis = MENU_OPTION_Q_AXIS;
			break;
		case AXIS_R: // hex plane, east to west; same as AXIS_Z, but for hex
			active_grid_instance = grid_instance[2];
			edit_plane.normal = Vector3(0, 0, 1);
			cell_depth = 1.5 * cell_size.x;
			grid_transform.translate_local(Vector3(is_even_floor * SQRT3_2 * cell_size.x, 0, 0));
			menu_axis = MENU_OPTION_R_AXIS;
			break;
		case AXIS_S: // hex plane, southwest to northeast
			active_grid_instance = grid_instance[2];
			edit_plane.normal = Vector3(SQRT3_2, 0, 0.5).normalized();
			cell_depth = 1.5 * cell_size.x;
			grid_transform.rotate(Vector3(0, 1, 0), Math_PI / 3.0);
			grid_transform.translate_local(Vector3(is_even_floor * SQRT3_2 * cell_size.x, 0, 0));
			menu_axis = MENU_OPTION_S_AXIS;
			break;
		default:
			ERR_PRINT_ED("unsupported edit plane axis");
			return;
	}

	// update the depth of the edit plane so it matches the floor, and update
	// the grid transform for the depth.
	edit_plane.d = edit_floor[edit_axis] * cell_depth;
	grid_transform.origin += edit_plane.normal * edit_plane.d;

	// shift the edit plane a little into the cell to prevent floating point
	// errors from causing the raycast to fall into the lower cell.  Note we
	// only need to do this when the grid is drawn along the edge of a cell,
	// so the Y & X axis, or any square shape cell.  Hex cells draw the grid
	// through the middle of the cells for Q/R/S.
	if (edit_axis == AXIS_Y || edit_axis == AXIS_X || !is_hex) {
		edit_plane.d += cell_depth * 0.1;
	}

	// make the editing grid visible
	RenderingServer::get_singleton()
			->instance_set_visible(active_grid_instance, true);
	RenderingServer::get_singleton()->instance_set_transform(active_grid_instance,
			node->get_global_transform() * grid_transform);

	// update the UI floor indicator
	floor->set_value(edit_floor[edit_axis]);

	// update the option menu to show the correct axis is selected
	PopupMenu *popup = options->get_popup();
	for (int i = MENU_OPTION_X_AXIS; i <= MENU_OPTION_S_AXIS; i++) {
		int index = popup->get_item_index(i);
		if (index != -1) {
			popup->set_item_checked(index, menu_axis == i);
		}
	}
}

void GridMapEditor::_draw_hex_grid(RID p_mesh_id, const Vector3 &p_cell_size) {
	// create the points that make up the top of a hex cell
	Vector<Vector3> shape_points;
	shape_points.append(Vector3(0.0, 0, -1.0) * p_cell_size);
	shape_points.append(Vector3(-SQRT3_2, 0, -0.5) * p_cell_size);
	shape_points.append(Vector3(-SQRT3_2, 0, 0.5) * p_cell_size);
	shape_points.append(Vector3(0.0, 0, 1.0) * p_cell_size);
	shape_points.append(Vector3(SQRT3_2, 0, 0.5) * p_cell_size);
	shape_points.append(Vector3(SQRT3_2, 0, -0.5) * p_cell_size);

	Vector<Vector3> grid_points;
	TypedArray<Vector3i> cells = node->local_region_to_map(
			Vector3i(-GRID_CURSOR_SIZE * Math_SQRT3 * p_cell_size.x,
					0,
					-GRID_CURSOR_SIZE * 1.625 * p_cell_size.x),
			Vector3i(GRID_CURSOR_SIZE * Math_SQRT3 * p_cell_size.x,
					0,
					GRID_CURSOR_SIZE * 1.625 * p_cell_size.x));
	for (const Vector3i cell : cells) {
		Vector3 center = node->map_to_local(cell);

		for (int j = 1; j < shape_points.size(); j++) {
			grid_points.append(center + shape_points[j - 1]);
			grid_points.append(center + shape_points[j]);
		}
	}

	Array d;
	d.resize(RS::ARRAY_MAX);
	d[RS::ARRAY_VERTEX] = grid_points;
	RenderingServer::get_singleton()->mesh_add_surface_from_arrays(p_mesh_id, RenderingServer::PRIMITIVE_LINES, d);
	RenderingServer::get_singleton()->mesh_surface_set_material(p_mesh_id, 0, indicator_mat->get_rid());
}

void GridMapEditor::_draw_hex_x_axis_grid(RID p_mesh_id, const Vector3 &p_cell_size) {
	Vector<Vector3> grid_points;

	// draw horizontal lines
	for (int y_index = -GRID_CURSOR_SIZE; y_index <= GRID_CURSOR_SIZE; y_index++) {
		real_t y = y_index * p_cell_size.y;
		grid_points.append(Vector3(0, y, -GRID_CURSOR_SIZE * 1.625 * p_cell_size.x));
		grid_points.append(Vector3(0, y, GRID_CURSOR_SIZE * 1.625 * p_cell_size.x));
	}

	// for vertical lines, we'll need to know where the center of the cell is
	// for a line along the Z axis.
	TypedArray<Vector3i> cells = node->local_region_to_map(
			Vector3(0, 0.001, -GRID_CURSOR_SIZE * 1.625 * p_cell_size.x),
			Vector3(0, 0.002, GRID_CURSOR_SIZE * 1.625 * p_cell_size.x));

	// use the cell list to draw the vertical lines
	for (const Vector3i cell : cells) {
		// grab the z coordinate for the center of the cell
		real_t z = node->map_to_local(cell).z;

		// Adjust from the center of the cell to where the line should fall.
		// We're drawing lines at 1 radius, then 2 radius apart, alternating.
		if ((cell.z & 1) == 0) {
			z += p_cell_size.x;
		} else {
			z += p_cell_size.x / 2;
		}

		grid_points.append(Vector3(0, -GRID_CURSOR_SIZE * p_cell_size.y, z));
		grid_points.append(Vector3(0, GRID_CURSOR_SIZE * p_cell_size.y, z));
	}

	Array d;
	d.resize(RS::ARRAY_MAX);
	d[RS::ARRAY_VERTEX] = grid_points;
	RenderingServer::get_singleton()->mesh_add_surface_from_arrays(p_mesh_id, RenderingServer::PRIMITIVE_LINES, d);
	RenderingServer::get_singleton()->mesh_surface_set_material(p_mesh_id, 0, indicator_mat->get_rid());
}

void GridMapEditor::_draw_plane_grid(
		RID p_mesh_id,
		const Vector3 &p_axis_n1,
		const Vector3 &p_axis_n2,
		const Vector3 &cell_size) {
	Vector<Vector3> grid_points;

	Vector3 axis_n1 = p_axis_n1 * cell_size;
	Vector3 axis_n2 = p_axis_n2 * cell_size;

	for (int j = -GRID_CURSOR_SIZE; j <= GRID_CURSOR_SIZE; j++) {
		for (int k = -GRID_CURSOR_SIZE; k <= GRID_CURSOR_SIZE; k++) {
			Vector3 p = axis_n1 * j + axis_n2 * k;

			Vector3 pj = axis_n1 * (j + 1) + axis_n2 * k;

			Vector3 pk = axis_n1 * j + axis_n2 * (k + 1);

			grid_points.push_back(p);
			grid_points.push_back(pk);

			grid_points.push_back(p);
			grid_points.push_back(pj);
		}
	}

	Array d;
	d.resize(RS::ARRAY_MAX);
	d[RS::ARRAY_VERTEX] = grid_points;
	RenderingServer::get_singleton()->mesh_add_surface_from_arrays(p_mesh_id, RenderingServer::PRIMITIVE_LINES, d);
	RenderingServer::get_singleton()->mesh_surface_set_material(p_mesh_id, 0, indicator_mat->get_rid());
}

void GridMapEditor::_draw_grids(const Vector3 &p_cell_size) {
	for (int i = 0; i < 3; i++) {
		RS::get_singleton()->mesh_clear(grid_mesh[i]);
	}

	switch (node->get_cell_shape()) {
		case GridMap::CELL_SHAPE_SQUARE:
			_draw_plane_grid(grid_mesh[0], Vector3(0, 1, 0), Vector3(0, 0, 1), p_cell_size);
			_draw_plane_grid(grid_mesh[1], Vector3(1, 0, 0), Vector3(0, 0, 1), p_cell_size);
			_draw_plane_grid(grid_mesh[2], Vector3(1, 0, 0), Vector3(0, 1, 0), p_cell_size);
			break;
		case GridMap::CELL_SHAPE_HEXAGON: {
			real_t radius = p_cell_size.x;
			Vector3 cell_size = Vector3(Math_SQRT3 * radius, p_cell_size.y, Math_SQRT3 * radius);
			_draw_hex_x_axis_grid(grid_mesh[0], p_cell_size);
			_draw_hex_grid(grid_mesh[1], p_cell_size);
			_draw_plane_grid(grid_mesh[2], Vector3(1, 0, 0), Vector3(0, 1, 0), cell_size);
			break;
		}
		default:
			ERR_PRINT_ED("unsupported cell shape");
			return;
	}
}

void GridMapEditor::_update_cell_shape(const GridMap::CellShape cell_shape) {
	_draw_grids(node->get_cell_size());
	_build_selection_meshes();
	edit_axis = AXIS_Y;
	_update_options_menu();
	selection.active = false;
	_update_selection();
}

void GridMapEditor::_build_selection_meshes() {
	if (selection_tile_mesh.is_valid()) {
		RS::get_singleton()->free(selection_tile_mesh);
		selection_tile_mesh = RID();
	}
	if (selection_multimesh.is_valid()) {
		RS::get_singleton()->free(selection_multimesh);
		selection_multimesh = RID();
	}

	// we can get called when node is null
	if (node == NULL) {
		return;
	}

	Array mesh_array;
	mesh_array.resize(RS::ARRAY_MAX);
	Array lines_array;
	lines_array.resize(RS::ARRAY_MAX);

	switch (node->get_cell_shape()) {
		case GridMap::CELL_SHAPE_SQUARE: {
			BoxMesh::create_mesh_array(mesh_array, Vector3(1, 1, 1));

			/*
			 *     (2)-----(3)               Y
			 *      | \     | \              |
			 *      |  (1)-----(0)           o---X
			 *      |   |   |   |             \
			 *     (6)--|--(7)  |              Z
			 *        \ |     \ |
			 *         (5)-----(4)
			 */
			lines_array[RS::ARRAY_VERTEX] = Vector<Vector3>({
					Vector3(0.5, 0.5, 0.5), // 0
					Vector3(-0.5, 0.5, 0.5), // 1
					Vector3(-0.5, 0.5, -0.5), // 2
					Vector3(0.5, 0.5, -0.5), // 3
					Vector3(0.5, -0.5, 0.5), // 4
					Vector3(-0.5, -0.5, 0.5), // 5
					Vector3(-0.5, -0.5, -0.5), // 6
					Vector3(0.5, -0.5, -0.5) // 7
			});
			lines_array[RS::ARRAY_INDEX] = Vector<int>({
					0, 1, 2, 3, // top
					7, 4, 5, 6, // bottom
					7, 3, 0, 4, // right
					5, 1, 2, 6, // left
			});
			break;
		}
		case GridMap::CELL_SHAPE_HEXAGON:
			CylinderMesh::create_mesh_array(mesh_array, 1.0, 1.0, 1, 6, 1);

			/*
			 *               (0)             Y
			 *              /   \            |
			 *           (1)     (5)         o---X
			 *            |       |           \
			 *           (2)     (4)           Z
			 *            | \   / |
			 *            |  (3)  |
			 *            |   |   |
			 *            |  (6)  |
			 *            | / | \ |
			 *           (7)  |  (b)
			 *            |   |   |
			 *           (8)  |  (a)
			 *              \ | /
			 *               (9)
			 */

			lines_array[RS::ARRAY_VERTEX] = Vector<Vector3>({
					Vector3(0.0, 0.5, -1.0), // 0
					Vector3(-SQRT3_2, 0.5, -0.5), // 1
					Vector3(-SQRT3_2, 0.5, 0.5), // 2
					Vector3(0.0, 0.5, 1.0), // 3
					Vector3(SQRT3_2, 0.5, 0.5), // 4
					Vector3(SQRT3_2, 0.5, -0.5), // 5
					Vector3(0.0, -0.5, -1.0), // 6
					Vector3(-SQRT3_2, -0.5, -0.5), // 7
					Vector3(-SQRT3_2, -0.5, 0.5), // 8
					Vector3(0.0, -0.5, 1.0), // 9
					Vector3(SQRT3_2, -0.5, 0.5), // 10 (0xa)
					Vector3(SQRT3_2, -0.5, -0.5), // 11 (0xb)
			});
			lines_array[RS::ARRAY_INDEX] = Vector<int>({
					0, 1, 2, 3, 4, 5, // top
					11, 6, 7, 8, 9, 10, // bottom
					11, 5, 0, 6, // northeast face
					7, 1, 2, 8, // west face
					9, 3, 4, 10, // southeast face
			});
			break;
		default:
			ERR_PRINT_ED("unsupported cell shape");
			return;
	}

	RenderingServer *rs = RS::get_singleton();
	selection_tile_mesh = rs->mesh_create();
	rs->mesh_add_surface_from_arrays(selection_tile_mesh, RS::PRIMITIVE_TRIANGLES, mesh_array);
	rs->mesh_surface_set_material(selection_tile_mesh, 0, inner_mat->get_rid());

	// add lines around the cell
	rs->mesh_add_surface_from_arrays(selection_tile_mesh, RS::PRIMITIVE_LINE_STRIP, lines_array);
	rs->mesh_surface_set_material(selection_tile_mesh, 1, outer_mat->get_rid());

	// create the multimesh for rendering the tile mesh in multiple locations.
	selection_multimesh = rs->multimesh_create();
	rs->multimesh_set_mesh(selection_multimesh, selection_tile_mesh);
}

void GridMapEditor::_update_theme() {
	options->set_icon(get_theme_icon(SNAME("GridMap"), EditorStringName(EditorIcons)));
	search_box->set_right_icon(get_theme_icon(SNAME("Search"), EditorStringName(EditorIcons)));
	mode_thumbnail->set_icon(get_theme_icon(SNAME("FileThumbnail"), EditorStringName(EditorIcons)));
	mode_list->set_icon(get_theme_icon(SNAME("FileList"), EditorStringName(EditorIcons)));
}

void GridMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			mesh_library_palette->connect(SceneStringName(item_selected), callable_mp(this, &GridMapEditor::_item_selected_cbk));
			for (int i = 0; i < 3; i++) {
				grid_mesh[i] = RS::get_singleton()->mesh_create();
				grid_instance[i] = RS::get_singleton()->instance_create2(grid_mesh[i], get_tree()->get_root()->get_world_3d()->get_scenario());
				RenderingServer::get_singleton()->instance_set_layer_mask(grid_instance[i], 1 << Node3DEditorViewport::MISC_TOOL_LAYER);
				RenderingServer::get_singleton()->instance_set_visible(grid_instance[i], false);
			}
			_update_selection();
			_update_paste_indicator();
			_update_theme();
			_update_options_menu();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_clear_clipboard_data();

			for (int i = 0; i < 3; i++) {
				RS::get_singleton()->free(grid_instance[i]);
				RS::get_singleton()->free(grid_mesh[i]);
				grid_instance[i] = RID();
				grid_mesh[i] = RID();
			}

			selection.active = false;
			_update_selection();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!node) {
				return;
			}

			// if the transform of our GridMap node has been changed, update
			// the grid.
			Transform3D transform = node->get_global_transform();
			if (transform != node_global_transform) {
				node_global_transform = transform;
				update_grid();
				_update_selection();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			if (input_action == INPUT_PAINT) {
				// Simulate mouse released event to stop drawing when editor focus exists.
				Ref<InputEventMouseButton> release;
				release.instantiate();
				release->set_button_index(MouseButton::LEFT);
				forward_spatial_input_event(nullptr, release);
			}
		} break;
	}
}

void GridMapEditor::_update_cursor_instance() {
	if (!node) {
		return;
	}

	if (cursor_instance.is_valid()) {
		RenderingServer::get_singleton()->free(cursor_instance);
	}
	cursor_instance = RID();

	if (selected_palette >= 0) {
		if (node && !node->get_mesh_library().is_null()) {
			Ref<Mesh> mesh = node->get_mesh_library()->get_item_mesh(selected_palette);
			if (!mesh.is_null() && mesh->get_rid().is_valid()) {
				cursor_instance = RenderingServer::get_singleton()->instance_create2(mesh->get_rid(), get_tree()->get_root()->get_world_3d()->get_scenario());
				RenderingServer::get_singleton()->instance_set_transform(cursor_instance, cursor_transform);
			}
		}
	}
}

void GridMapEditor::_item_selected_cbk(int p_idx) {
	selected_palette = mesh_library_palette->get_item_metadata(p_idx);

	_update_cursor_instance();
}

void GridMapEditor::_floor_changed(float p_value) {
	// update the floor number for the current plane we're editing
	edit_floor[edit_axis] = p_value;

	// save off editor floor numbers so the user can jump in and out of the
	// gridmap editor without losing their place.
	TypedArray<int> floors;
	for (int i = 0; i < AXIS_MAX; i++) {
		floors.push_back(edit_floor[i]);
	}
	node->set_meta("_editor_floor_", floors);

	update_grid();
	_update_selection();
}

void GridMapEditor::_floor_mouse_exited() {
	floor->get_line_edit()->release_focus();
}

void GridMapEditor::_bind_methods() {
	ClassDB::bind_method("_configure", &GridMapEditor::_configure);
	ClassDB::bind_method("_set_selection", &GridMapEditor::_set_selection);
}

void GridMapEditor::_update_options_menu() {
	PopupMenu *popup = options->get_popup();

	// save off the current settings
	bool paste_selects = false;
	if (int index = popup->get_item_index(MENU_OPTION_PASTE_SELECTS) != -1) {
		popup->is_item_checked(index);
	}

	// clear the menu
	popup->clear();

	// rebuild the menu
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/previous_floor"), MENU_OPTION_PREV_LEVEL);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/next_floor"), MENU_OPTION_NEXT_LEVEL);
	popup->add_separator();

	// shape-specific edit axis options
	if (node && node->get_cell_shape() == GridMap::CELL_SHAPE_HEXAGON) {
		// hex cells have five edit axis; we add shortcuts for Y, two more for
		// rotating a vertical plane clockwise and counter clockwise.
		popup->add_radio_check_item(TTR("Edit X Axis"), MENU_OPTION_X_AXIS);
		popup->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_y_axis"), MENU_OPTION_Y_AXIS);
		popup->add_radio_check_item(TTR("Edit Q Axis"), MENU_OPTION_Q_AXIS);
		popup->add_radio_check_item(TTR("Edit R Axis (Z Axis)"), MENU_OPTION_R_AXIS);
		popup->add_radio_check_item(TTR("Edit S Axis"), MENU_OPTION_S_AXIS);
		popup->add_shortcut(ED_GET_SHORTCUT("grid_map/edit_plane_rotate_cw"), MENU_OPTION_ROTATE_AXIS_CW);
		popup->add_shortcut(ED_GET_SHORTCUT("grid_map/edit_plane_rotate_ccw"), MENU_OPTION_ROTATE_AXIS_CCW);
	} else {
		// square cell shape only uses XYZ with per-plane shortcuts
		popup->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_x_axis"), MENU_OPTION_X_AXIS);
		popup->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_y_axis"), MENU_OPTION_Y_AXIS);
		popup->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_z_axis"), MENU_OPTION_Z_AXIS);
	}
	popup->set_item_checked(popup->get_item_index(MENU_OPTION_Y_AXIS), true);

	popup->add_separator();
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_x"), MENU_OPTION_CURSOR_ROTATE_X);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_y"), MENU_OPTION_CURSOR_ROTATE_Y);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_z"), MENU_OPTION_CURSOR_ROTATE_Z);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_x"), MENU_OPTION_CURSOR_BACK_ROTATE_X);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_y"), MENU_OPTION_CURSOR_BACK_ROTATE_Y);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_z"), MENU_OPTION_CURSOR_BACK_ROTATE_Z);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_clear_rotation"), MENU_OPTION_CURSOR_CLEAR_ROTATION);
	popup->add_separator();
	// TRANSLATORS: This is a toggle to select after pasting the new content.
	popup->add_check_shortcut(ED_GET_SHORTCUT("grid_map/paste_selects"), MENU_OPTION_PASTE_SELECTS);
	popup->set_item_checked(popup->get_item_index(MENU_OPTION_PASTE_SELECTS), paste_selects);
	popup->add_separator();
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/duplicate_selection"), MENU_OPTION_SELECTION_DUPLICATE);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/cut_selection"), MENU_OPTION_SELECTION_CUT);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/clear_selection"), MENU_OPTION_SELECTION_CLEAR);
	popup->add_shortcut(ED_GET_SHORTCUT("grid_map/fill_selection"), MENU_OPTION_SELECTION_FILL);

	popup->add_separator();
	popup->add_item(TTR("Settings..."), MENU_OPTION_GRIDMAP_SETTINGS);
}

GridMapEditor::GridMapEditor() {
	ED_SHORTCUT("grid_map/previous_floor", TTR("Previous Floor"), Key::Q, true);
	ED_SHORTCUT("grid_map/next_floor", TTR("Next Floor"), Key::E, true);
	ED_SHORTCUT("grid_map/edit_x_axis", TTR("Edit X Axis"), Key::Z, true);
	ED_SHORTCUT("grid_map/edit_y_axis", TTR("Edit Y Axis"), Key::X, true);
	ED_SHORTCUT("grid_map/edit_z_axis", TTR("Edit Z Axis"), Key::C, true);

	// TRANSLATORS: These two shortcuts are only used with hex-shaped cells to rotate an edit plane about the Y axis clockwise or counter-clockwise.
	ED_SHORTCUT("grid_map/edit_plane_rotate_cw", TTR("Rotate Edit Plane Clockwise"), Key::C, true);
	ED_SHORTCUT("grid_map/edit_plane_rotate_ccw", TTR("Rotate Edit Plane Counter-Clockwise"), Key::Z, true);

	ED_SHORTCUT("grid_map/cursor_rotate_x", TTR("Cursor Rotate X"), Key::A, true);
	ED_SHORTCUT("grid_map/cursor_rotate_y", TTR("Cursor Rotate Y"), Key::S, true);
	ED_SHORTCUT("grid_map/cursor_rotate_z", TTR("Cursor Rotate Z"), Key::D, true);
	ED_SHORTCUT("grid_map/cursor_back_rotate_x", TTR("Cursor Back Rotate X"), KeyModifierMask::SHIFT + Key::A, true);
	ED_SHORTCUT("grid_map/cursor_back_rotate_y", TTR("Cursor Back Rotate Y"), KeyModifierMask::SHIFT + Key::S, true);
	ED_SHORTCUT("grid_map/cursor_back_rotate_z", TTR("Cursor Back Rotate Z"), KeyModifierMask::SHIFT + Key::D, true);
	ED_SHORTCUT("grid_map/cursor_clear_rotation", TTR("Cursor Clear Rotation"), Key::W, true);
	ED_SHORTCUT("grid_map/paste_selects", TTR("Paste Selects"));
	ED_SHORTCUT("grid_map/duplicate_selection", TTR("Duplicate Selection"), KeyModifierMask::CTRL + Key::C);
	ED_SHORTCUT("grid_map/cut_selection", TTR("Cut Selection"), KeyModifierMask::CTRL + Key::X);
	ED_SHORTCUT("grid_map/clear_selection", TTR("Clear Selection"), Key::KEY_DELETE);
	ED_SHORTCUT("grid_map/fill_selection", TTR("Fill Selection"), KeyModifierMask::CTRL + Key::F);

	int mw = EDITOR_GET("editors/grid_map/palette_min_width");
	Control *ec = memnew(Control);
	ec->set_custom_minimum_size(Size2(mw, 0) * EDSCALE);
	add_child(ec);

	spatial_editor_hb = memnew(HBoxContainer);
	spatial_editor_hb->set_h_size_flags(SIZE_EXPAND_FILL);
	spatial_editor_hb->set_alignment(BoxContainer::ALIGNMENT_END);
	Node3DEditor::get_singleton()->add_control_to_menu_panel(spatial_editor_hb);

	spin_box_label = memnew(Label);
	spin_box_label->set_text(TTR("Floor:"));
	spatial_editor_hb->add_child(spin_box_label);

	floor = memnew(SpinBox);
	floor->set_min(-32767);
	floor->set_max(32767);
	floor->set_step(1);
	floor->get_line_edit()->add_theme_constant_override("minimum_character_width", 16);

	spatial_editor_hb->add_child(floor);
	floor->connect(SceneStringName(value_changed), callable_mp(this, &GridMapEditor::_floor_changed));
	floor->connect(SceneStringName(mouse_exited), callable_mp(this, &GridMapEditor::_floor_mouse_exited));
	floor->get_line_edit()->connect(SceneStringName(mouse_exited), callable_mp(this, &GridMapEditor::_floor_mouse_exited));

	spatial_editor_hb->add_child(memnew(VSeparator));

	options = memnew(MenuButton);
	spatial_editor_hb->add_child(options);
	spatial_editor_hb->hide();

	options->set_text(TTR("Grid Map"));
	_update_options_menu();

	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Settings..."), MENU_OPTION_GRIDMAP_SETTINGS);

	settings_dialog = memnew(ConfirmationDialog);
	settings_dialog->set_title(TTR("GridMap Settings"));
	add_child(settings_dialog);
	settings_vbc = memnew(VBoxContainer);
	settings_vbc->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	settings_dialog->add_child(settings_vbc);

	settings_pick_distance = memnew(SpinBox);
	settings_pick_distance->set_max(10000.0f);
	settings_pick_distance->set_min(500.0f);
	settings_pick_distance->set_step(1.0f);
	settings_pick_distance->set_value(EDITOR_GET("editors/grid_map/pick_distance"));
	settings_vbc->add_margin_child(TTR("Pick Distance:"), settings_pick_distance);

	options->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &GridMapEditor::_menu_option));

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_h_size_flags(SIZE_EXPAND_FILL);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Filter Meshes"));
	search_box->set_clear_button_enabled(true);
	hb->add_child(search_box);
	search_box->connect(SceneStringName(text_changed), callable_mp(this, &GridMapEditor::_text_changed));
	search_box->connect(SceneStringName(gui_input), callable_mp(this, &GridMapEditor::_sbox_input));

	mode_thumbnail = memnew(Button);
	mode_thumbnail->set_theme_type_variation("FlatButton");
	mode_thumbnail->set_toggle_mode(true);
	mode_thumbnail->set_pressed(true);
	hb->add_child(mode_thumbnail);
	mode_thumbnail->connect(SceneStringName(pressed), callable_mp(this, &GridMapEditor::_set_display_mode).bind(DISPLAY_THUMBNAIL));

	mode_list = memnew(Button);
	mode_list->set_theme_type_variation("FlatButton");
	mode_list->set_toggle_mode(true);
	mode_list->set_pressed(false);
	hb->add_child(mode_list);
	mode_list->connect(SceneStringName(pressed), callable_mp(this, &GridMapEditor::_set_display_mode).bind(DISPLAY_LIST));

	size_slider = memnew(HSlider);
	size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
	size_slider->set_min(0.2f);
	size_slider->set_max(4.0f);
	size_slider->set_step(0.1f);
	size_slider->set_value(1.0f);
	size_slider->connect(SceneStringName(value_changed), callable_mp(this, &GridMapEditor::_icon_size_changed));
	add_child(size_slider);

	mesh_library_palette = memnew(ItemList);
	mesh_library_palette->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	add_child(mesh_library_palette);
	mesh_library_palette->set_v_size_flags(SIZE_EXPAND_FILL);
	mesh_library_palette->connect(SceneStringName(gui_input), callable_mp(this, &GridMapEditor::_mesh_library_palette_input));

	info_message = memnew(Label);
	info_message->set_text(TTR("Give a MeshLibrary resource to this GridMap to use its meshes."));
	info_message->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	info_message->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	info_message->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	info_message->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	info_message->set_anchors_and_offsets_preset(PRESET_FULL_RECT, PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);
	mesh_library_palette->add_child(info_message);

	edit_floor[0] = -1;
	edit_floor[1] = -1;
	edit_floor[2] = -1;

	edit_axis = AXIS_Y;
	edit_plane = Plane();

	inner_mat.instantiate();
	inner_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.2));
	inner_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	inner_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	inner_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

	outer_mat.instantiate();
	outer_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.8));
	outer_mat->set_on_top_of_alpha();

	outer_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	outer_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	outer_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);

	_set_selection(false);

	indicator_mat.instantiate();
	indicator_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
	indicator_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
	indicator_mat->set_flag(StandardMaterial3D::FLAG_SRGB_VERTEX_COLOR, true);
	indicator_mat->set_flag(StandardMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	indicator_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
	indicator_mat->set_albedo(Color(0.8, 0.5, 0.1));
}

GridMapEditor::~GridMapEditor() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	_clear_clipboard_data();

	for (int i = 0; i < 3; i++) {
		if (grid_mesh[i].is_valid()) {
			RenderingServer::get_singleton()->free(grid_mesh[i]);
		}
		if (grid_instance[i].is_valid()) {
			RenderingServer::get_singleton()->free(grid_instance[i]);
		}
		if (cursor_instance.is_valid()) {
			RenderingServer::get_singleton()->free(cursor_instance);
		}
	}
	if (selection_multimesh.is_valid()) {
		RenderingServer::get_singleton()->free(selection_multimesh);
	}
	if (selection_tile_mesh.is_valid()) {
		RenderingServer::get_singleton()->free(selection_tile_mesh);
	}
}

void GridMapEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (!EditorSettings::get_singleton()->check_changed_settings_in_group("editors/grid_map")) {
				break;
			}
			switch ((int)EDITOR_GET("editors/grid_map/editor_side")) {
				case 0: { // Left.
					Node3DEditor::get_singleton()->move_control_to_left_panel(grid_map_editor);
				} break;
				case 1: { // Right.
					Node3DEditor::get_singleton()->move_control_to_right_panel(grid_map_editor);
				} break;
			}
		} break;
	}
}

void GridMapEditorPlugin::edit(Object *p_object) {
	grid_map_editor->edit(Object::cast_to<GridMap>(p_object));
}

bool GridMapEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("GridMap");
}

void GridMapEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		grid_map_editor->show();
		grid_map_editor->spatial_editor_hb->show();
		grid_map_editor->set_process(true);
	} else {
		grid_map_editor->spatial_editor_hb->hide();
		grid_map_editor->hide();
		grid_map_editor->set_process(false);
	}
}

GridMapEditorPlugin::GridMapEditorPlugin() {
	grid_map_editor = memnew(GridMapEditor);
	switch ((int)EDITOR_GET("editors/grid_map/editor_side")) {
		case 0: { // Left.
			Node3DEditor::get_singleton()->add_control_to_left_panel(grid_map_editor);
		} break;
		case 1: { // Right.
			Node3DEditor::get_singleton()->add_control_to_right_panel(grid_map_editor);
		} break;
	}
	grid_map_editor->hide();
}

GridMapEditorPlugin::~GridMapEditorPlugin() {
}

#endif // TOOLS_ENABLED
