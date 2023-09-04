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

#include "core/core_string_names.h"
#include "core/input/input.h"
#include "core/os/keyboard.h"
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
		} break;

		case MENU_OPTION_NEXT_LEVEL: {
			floor->set_value(floor->get_value() + 1);
		} break;

		case MENU_OPTION_X_AXIS:
		case MENU_OPTION_Y_AXIS:
		case MENU_OPTION_Z_AXIS: {
			int new_axis = p_option - MENU_OPTION_X_AXIS;
			for (int i = 0; i < 3; i++) {
				int idx = options->get_popup()->get_item_index(MENU_OPTION_X_AXIS + i);
				options->get_popup()->set_item_checked(idx, i == new_axis);
			}

			if (edit_axis != new_axis) {
				int item1 = options->get_popup()->get_item_index(MENU_OPTION_NEXT_LEVEL);
				int item2 = options->get_popup()->get_item_index(MENU_OPTION_PREV_LEVEL);
				if (edit_axis == Vector3::AXIS_Y) {
					options->get_popup()->set_item_text(item1, TTR("Next Plane"));
					options->get_popup()->set_item_text(item2, TTR("Previous Plane"));
					spin_box_label->set_text(TTR("Plane:"));
				} else if (new_axis == Vector3::AXIS_Y) {
					options->get_popup()->set_item_text(item1, TTR("Next Floor"));
					options->get_popup()->set_item_text(item2, TTR("Previous Floor"));
					spin_box_label->set_text(TTR("Floor:"));
				}
			}
			edit_axis = Vector3::Axis(new_axis);
			update_grid();

		} break;
		case MENU_OPTION_CURSOR_ROTATE_Y: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), -Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), -Math_PI / 2.0);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_X: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), -Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), -Math_PI / 2.0);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_Z: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), -Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), -Math_PI / 2.0);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Y: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), Math_PI / 2.0);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_X: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
			cursor_rot = node->get_orthogonal_index_from_basis(r);
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Z: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), Math_PI / 2.0);
				paste_indicator.orientation = node->get_orthogonal_index_from_basis(r);
				_update_paste_indicator();
				break;
			}

			r = node->get_basis_with_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), Math_PI / 2.0);
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
			paste_indicator.click = selection.begin;
			paste_indicator.current = selection.begin;
			paste_indicator.begin = selection.begin;
			paste_indicator.end = selection.end;
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
	cursor_transform.origin = cursor_origin;
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

void GridMapEditor::_update_selection_transform() {
	Transform3D xf_zero;
	xf_zero.basis.set_zero();

	if (!selection.active) {
		RenderingServer::get_singleton()->instance_set_transform(selection_instance, xf_zero);
		for (int i = 0; i < 3; i++) {
			RenderingServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf_zero);
		}
		return;
	}

	Transform3D xf;
	xf.scale((Vector3(1, 1, 1) + (selection.end - selection.begin)) * node->get_cell_size());
	xf.origin = selection.begin * node->get_cell_size();

	RenderingServer::get_singleton()->instance_set_transform(selection_instance, node->get_global_transform() * xf);

	for (int i = 0; i < 3; i++) {
		if (i != edit_axis || (edit_floor[edit_axis] < selection.begin[edit_axis]) || (edit_floor[edit_axis] > selection.end[edit_axis] + 1)) {
			RenderingServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf_zero);
		} else {
			Vector3 scale = (selection.end - selection.begin + Vector3(1, 1, 1));
			scale[edit_axis] = 1.0;
			Vector3 position = selection.begin;
			position[edit_axis] = edit_floor[edit_axis];

			scale *= node->get_cell_size();
			position *= node->get_cell_size();

			Transform3D xf2;
			xf2.basis.scale(scale);
			xf2.origin = position;

			RenderingServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf2);
		}
	}
}

void GridMapEditor::_validate_selection() {
	if (!selection.active) {
		return;
	}
	selection.begin = selection.click;
	selection.end = selection.current;

	if (selection.begin.x > selection.end.x) {
		SWAP(selection.begin.x, selection.end.x);
	}
	if (selection.begin.y > selection.end.y) {
		SWAP(selection.begin.y, selection.end.y);
	}
	if (selection.begin.z > selection.end.z) {
		SWAP(selection.begin.z, selection.end.z);
	}

	_update_selection_transform();
}

void GridMapEditor::_set_selection(bool p_active, const Vector3 &p_begin, const Vector3 &p_end) {
	selection.active = p_active;
	selection.begin = p_begin;
	selection.end = p_end;
	selection.click = p_begin;
	selection.current = p_end;

	if (is_visible_in_tree()) {
		_update_selection_transform();
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

	Plane p;
	p.normal[edit_axis] = 1.0;
	p.d = edit_floor[edit_axis] * node->get_cell_size()[edit_axis];

	Vector3 inters;
	if (!p.intersects_segment(from, from + normal * settings_pick_distance->get_value(), &inters)) {
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

	int cell[3];
	Vector3 cell_size = node->get_cell_size();

	for (int i = 0; i < 3; i++) {
		if (i == edit_axis) {
			cell[i] = edit_floor[i];
		} else {
			cell[i] = inters[i] / cell_size[i];
			if (inters[i] < 0) {
				cell[i] -= 1; // Compensate negative.
			}
			grid_ofs[i] = cell[i] * cell_size[i];
		}
	}

	RS::get_singleton()->instance_set_transform(grid_instance[edit_axis], node->get_global_transform() * edit_grid_xform);

	if (cursor_instance.is_valid()) {
		cursor_origin = (Vector3(cell[0], cell[1], cell[2]) + Vector3(0.5 * node->get_center_x(), 0.5 * node->get_center_y(), 0.5 * node->get_center_z())) * node->get_cell_size();
		cursor_visible = true;

		if (input_action == INPUT_SELECT || input_action == INPUT_PASTE) {
			cursor_visible = false;
		}

		_update_cursor_transform();
	}

	if (input_action == INPUT_PASTE) {
		paste_indicator.current = Vector3i(cell[0], cell[1], cell[2]);
		_update_paste_indicator();

	} else if (input_action == INPUT_SELECT) {
		selection.current = Vector3i(cell[0], cell[1], cell[2]);
		if (p_click) {
			selection.click = selection.current;
		}
		selection.active = true;
		_validate_selection();

		return true;
	} else if (input_action == INPUT_PICK) {
		int item = node->get_cell_item(Vector3i(cell[0], cell[1], cell[2]));
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
		si.position = Vector3i(cell[0], cell[1], cell[2]);
		si.new_value = selected_palette;
		si.new_orientation = cursor_rot;
		si.old_value = node->get_cell_item(Vector3i(cell[0], cell[1], cell[2]));
		si.old_orientation = node->get_cell_item_orientation(Vector3i(cell[0], cell[1], cell[2]));
		set_items.push_back(si);
		node->set_cell_item(Vector3i(cell[0], cell[1], cell[2]), selected_palette, cursor_rot);
		return true;
	} else if (input_action == INPUT_ERASE) {
		SetItem si;
		si.position = Vector3i(cell[0], cell[1], cell[2]);
		si.new_value = -1;
		si.new_orientation = 0;
		si.old_value = node->get_cell_item(Vector3i(cell[0], cell[1], cell[2]));
		si.old_orientation = node->get_cell_item_orientation(Vector3i(cell[0], cell[1], cell[2]));
		set_items.push_back(si);
		node->set_cell_item(Vector3i(cell[0], cell[1], cell[2]), -1);
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
	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				Vector3i selected = Vector3i(i, j, k);
				undo_redo->add_do_method(node, "set_cell_item", selected, GridMap::INVALID_CELL_ITEM);
				undo_redo->add_undo_method(node, "set_cell_item", selected, node->get_cell_item(selected), node->get_cell_item_orientation(selected));
			}
		}
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
	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				Vector3i selected = Vector3i(i, j, k);
				undo_redo->add_do_method(node, "set_cell_item", selected, selected_palette, cursor_rot);
				undo_redo->add_undo_method(node, "set_cell_item", selected, node->get_cell_item(selected), node->get_cell_item_orientation(selected));
			}
		}
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

	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				Vector3i selected = Vector3i(i, j, k);
				int itm = node->get_cell_item(selected);
				if (itm == GridMap::INVALID_CELL_ITEM) {
					continue;
				}

				Ref<Mesh> mesh = meshLibrary->get_item_mesh(itm);

				ClipboardItem item;
				item.cell_item = itm;
				item.grid_offset = Vector3(selected) - selection.begin;
				item.orientation = node->get_cell_item_orientation(selected);
				item.instance = RenderingServer::get_singleton()->instance_create2(mesh->get_rid(), get_tree()->get_root()->get_world_3d()->get_scenario());

				clipboard_items.push_back(item);
			}
		}
	}
}

void GridMapEditor::_update_paste_indicator() {
	if (input_action != INPUT_PASTE) {
		Transform3D xf;
		xf.basis.set_zero();
		RenderingServer::get_singleton()->instance_set_transform(paste_instance, xf);
		return;
	}

	Vector3 center = 0.5 * Vector3(real_t(node->get_center_x()), real_t(node->get_center_y()), real_t(node->get_center_z()));
	Vector3 scale = (Vector3(1, 1, 1) + (paste_indicator.end - paste_indicator.begin)) * node->get_cell_size();
	Transform3D xf;
	xf.scale(scale);
	xf.origin = (paste_indicator.begin + (paste_indicator.current - paste_indicator.click) + center) * node->get_cell_size();
	Basis rot;
	rot = node->get_basis_with_orthogonal_index(paste_indicator.orientation);
	xf.basis = rot * xf.basis;
	xf.translate_local((-center * node->get_cell_size()) / scale);

	RenderingServer::get_singleton()->instance_set_transform(paste_instance, node->get_global_transform() * xf);

	for (const ClipboardItem &item : clipboard_items) {
		xf = Transform3D();
		xf.origin = (paste_indicator.begin + (paste_indicator.current - paste_indicator.click) + center) * node->get_cell_size();
		xf.basis = rot * xf.basis;
		xf.translate_local(item.grid_offset * node->get_cell_size());

		Basis item_rot;
		item_rot = node->get_basis_with_orthogonal_index(item.orientation);
		xf.basis = item_rot * xf.basis * node->get_cell_scale();

		RenderingServer::get_singleton()->instance_set_transform(item.instance, node->get_global_transform() * xf);
	}
}

void GridMapEditor::_do_paste() {
	int idx = options->get_popup()->get_item_index(MENU_OPTION_PASTE_SELECTS);
	bool reselect = options->get_popup()->is_item_checked(idx);

	Basis rot;
	rot = node->get_basis_with_orthogonal_index(paste_indicator.orientation);

	Vector3 ofs = paste_indicator.current - paste_indicator.click;
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("GridMap Paste Selection"));

	for (const ClipboardItem &item : clipboard_items) {
		Vector3 position = rot.xform(item.grid_offset) + paste_indicator.begin + ofs;

		Basis orm;
		orm = node->get_basis_with_orthogonal_index(item.orientation);
		orm = rot * orm;

		undo_redo->add_do_method(node, "set_cell_item", position, item.cell_item, node->get_orthogonal_index_from_basis(orm));
		undo_redo->add_undo_method(node, "set_cell_item", position, node->get_cell_item(position), node->get_cell_item_orientation(position));
	}

	if (reselect) {
		undo_redo->add_do_method(this, "_set_selection", true, paste_indicator.begin + ofs, paste_indicator.end + ofs);
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
		// Update the grid, to check if the grid needs to be moved to a tile cursor.
		update_grid();

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

			if (k->is_shift_pressed() && selection.active && input_action != INPUT_PASTE) {
				if (k->get_keycode() == (Key)options->get_popup()->get_item_accelerator(options->get_popup()->get_item_index(MENU_OPTION_PREV_LEVEL))) {
					selection.click[edit_axis]--;
					_validate_selection();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
				}
				if (k->get_keycode() == (Key)options->get_popup()->get_item_accelerator(options->get_popup()->get_item_index(MENU_OPTION_NEXT_LEVEL))) {
					selection.click[edit_axis]++;
					_validate_selection();
					return EditorPlugin::AFTER_GUI_INPUT_STOP;
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

void GridMapEditor::_sbox_input(const Ref<InputEvent> &p_ie) {
	const Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_keycode() == Key::UP || k->get_keycode() == Key::DOWN || k->get_keycode() == Key::PAGEUP || k->get_keycode() == Key::PAGEDOWN)) {
		// Forward the key input to the ItemList so it can be scrolled
		mesh_library_palette->gui_input(k);
		search_box->accept_event();
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
		node->disconnect(CoreStringName(changed), callable_mp(this, &GridMapEditor::_update_mesh_library));
		if (mesh_library.is_valid()) {
			mesh_library->disconnect_changed(callable_mp(this, &GridMapEditor::update_palette));
			mesh_library = Ref<MeshLibrary>();
		}
	}

	node = p_gridmap;

	input_action = INPUT_NONE;
	selection.active = false;
	_update_selection_transform();
	_update_paste_indicator();

	spatial_editor = Object::cast_to<Node3DEditorPlugin>(EditorNode::get_singleton()->get_editor_plugin_screen());

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

	_draw_grids(node->get_cell_size());
	update_grid();

	node->connect(SNAME("cell_size_changed"), callable_mp(this, &GridMapEditor::_draw_grids));
	node->connect(CoreStringName(changed), callable_mp(this, &GridMapEditor::_update_mesh_library));
	_update_mesh_library();
}

void GridMapEditor::update_grid() {
	grid_xform.origin.x -= 1; // Force update in hackish way.

	grid_ofs[edit_axis] = edit_floor[edit_axis] * node->get_cell_size()[edit_axis];

	// If there's a valid tile cursor, offset the grid, otherwise move it back to the node.
	edit_grid_xform.origin = cursor_instance.is_valid() ? grid_ofs : Vector3();
	edit_grid_xform.basis = Basis();

	for (int i = 0; i < 3; i++) {
		RenderingServer::get_singleton()->instance_set_visible(grid_instance[i], i == edit_axis);
	}

	updating = true;
	floor->set_value(edit_floor[edit_axis]);
	updating = false;
}

void GridMapEditor::_draw_grids(const Vector3 &cell_size) {
	Vector3 edited_floor = node->get_meta("_editor_floor_", Vector3());

	for (int i = 0; i < 3; i++) {
		RS::get_singleton()->mesh_clear(grid[i]);
		edit_floor[i] = edited_floor[i];
	}

	Vector<Vector3> grid_points[3];
	Vector<Color> grid_colors[3];

	for (int i = 0; i < 3; i++) {
		Vector3 axis;
		axis[i] = 1;
		Vector3 axis_n1;
		axis_n1[(i + 1) % 3] = cell_size[(i + 1) % 3];
		Vector3 axis_n2;
		axis_n2[(i + 2) % 3] = cell_size[(i + 2) % 3];

		for (int j = -GRID_CURSOR_SIZE; j <= GRID_CURSOR_SIZE; j++) {
			for (int k = -GRID_CURSOR_SIZE; k <= GRID_CURSOR_SIZE; k++) {
				Vector3 p = axis_n1 * j + axis_n2 * k;
				float trans = Math::pow(MAX(0, 1.0 - (Vector2(j, k).length() / GRID_CURSOR_SIZE)), 2);

				Vector3 pj = axis_n1 * (j + 1) + axis_n2 * k;
				float transj = Math::pow(MAX(0, 1.0 - (Vector2(j + 1, k).length() / GRID_CURSOR_SIZE)), 2);

				Vector3 pk = axis_n1 * j + axis_n2 * (k + 1);
				float transk = Math::pow(MAX(0, 1.0 - (Vector2(j, k + 1).length() / GRID_CURSOR_SIZE)), 2);

				grid_points[i].push_back(p);
				grid_points[i].push_back(pk);
				grid_colors[i].push_back(Color(1, 1, 1, trans));
				grid_colors[i].push_back(Color(1, 1, 1, transk));

				grid_points[i].push_back(p);
				grid_points[i].push_back(pj);
				grid_colors[i].push_back(Color(1, 1, 1, trans));
				grid_colors[i].push_back(Color(1, 1, 1, transj));
			}
		}

		Array d;
		d.resize(RS::ARRAY_MAX);
		d[RS::ARRAY_VERTEX] = grid_points[i];
		d[RS::ARRAY_COLOR] = grid_colors[i];
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(grid[i], RenderingServer::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(grid[i], 0, indicator_mat->get_rid());
	}
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
			mesh_library_palette->connect("item_selected", callable_mp(this, &GridMapEditor::_item_selected_cbk));
			for (int i = 0; i < 3; i++) {
				grid[i] = RS::get_singleton()->mesh_create();
				grid_instance[i] = RS::get_singleton()->instance_create2(grid[i], get_tree()->get_root()->get_world_3d()->get_scenario());
				RenderingServer::get_singleton()->instance_set_layer_mask(grid_instance[i], 1 << Node3DEditorViewport::MISC_TOOL_LAYER);
				selection_level_instance[i] = RenderingServer::get_singleton()->instance_create2(selection_level_mesh[i], get_tree()->get_root()->get_world_3d()->get_scenario());
				RenderingServer::get_singleton()->instance_set_layer_mask(selection_level_instance[i], 1 << Node3DEditorViewport::MISC_TOOL_LAYER);
			}

			selection_instance = RenderingServer::get_singleton()->instance_create2(selection_mesh, get_tree()->get_root()->get_world_3d()->get_scenario());
			RenderingServer::get_singleton()->instance_set_layer_mask(selection_instance, 1 << Node3DEditorViewport::MISC_TOOL_LAYER);
			paste_instance = RenderingServer::get_singleton()->instance_create2(paste_mesh, get_tree()->get_root()->get_world_3d()->get_scenario());
			RenderingServer::get_singleton()->instance_set_layer_mask(paste_instance, 1 << Node3DEditorViewport::MISC_TOOL_LAYER);

			_update_selection_transform();
			_update_paste_indicator();
			_update_theme();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_clear_clipboard_data();

			for (int i = 0; i < 3; i++) {
				RS::get_singleton()->free(grid_instance[i]);
				RS::get_singleton()->free(grid[i]);
				grid_instance[i] = RID();
				grid[i] = RID();
				RenderingServer::get_singleton()->free(selection_level_instance[i]);
			}

			RenderingServer::get_singleton()->free(selection_instance);
			RenderingServer::get_singleton()->free(paste_instance);
			selection_instance = RID();
			paste_instance = RID();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!node) {
				return;
			}

			Transform3D xf = node->get_global_transform();

			if (xf != grid_xform) {
				for (int i = 0; i < 3; i++) {
					RS::get_singleton()->instance_set_transform(grid_instance[i], xf * edit_grid_xform);
				}
				grid_xform = xf;
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

void GridMapEditor::_item_selected_cbk(int idx) {
	selected_palette = mesh_library_palette->get_item_metadata(idx);

	_update_cursor_instance();
}

void GridMapEditor::_floor_changed(float p_value) {
	if (updating) {
		return;
	}

	edit_floor[edit_axis] = p_value;
	node->set_meta("_editor_floor_", Vector3(edit_floor[0], edit_floor[1], edit_floor[2]));
	update_grid();
	_update_selection_transform();
}

void GridMapEditor::_floor_mouse_exited() {
	floor->get_line_edit()->release_focus();
}

void GridMapEditor::_bind_methods() {
	ClassDB::bind_method("_configure", &GridMapEditor::_configure);
	ClassDB::bind_method("_set_selection", &GridMapEditor::_set_selection);
}

GridMapEditor::GridMapEditor() {
	ED_SHORTCUT("grid_map/previous_floor", TTR("Previous Floor"), Key::Q, true);
	ED_SHORTCUT("grid_map/next_floor", TTR("Next Floor"), Key::E, true);
	ED_SHORTCUT("grid_map/edit_x_axis", TTR("Edit X Axis"), Key::Z, true);
	ED_SHORTCUT("grid_map/edit_y_axis", TTR("Edit Y Axis"), Key::X, true);
	ED_SHORTCUT("grid_map/edit_z_axis", TTR("Edit Z Axis"), Key::C, true);
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

	int mw = EDITOR_DEF("editors/grid_map/palette_min_width", 230);
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
	floor->connect("value_changed", callable_mp(this, &GridMapEditor::_floor_changed));
	floor->connect("mouse_exited", callable_mp(this, &GridMapEditor::_floor_mouse_exited));
	floor->get_line_edit()->connect("mouse_exited", callable_mp(this, &GridMapEditor::_floor_mouse_exited));

	spatial_editor_hb->add_child(memnew(VSeparator));

	options = memnew(MenuButton);
	spatial_editor_hb->add_child(options);
	spatial_editor_hb->hide();

	options->set_text(TTR("Grid Map"));
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/previous_floor"), MENU_OPTION_PREV_LEVEL);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/next_floor"), MENU_OPTION_NEXT_LEVEL);
	options->get_popup()->add_separator();
	options->get_popup()->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_x_axis"), MENU_OPTION_X_AXIS);
	options->get_popup()->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_y_axis"), MENU_OPTION_Y_AXIS);
	options->get_popup()->add_radio_check_shortcut(ED_GET_SHORTCUT("grid_map/edit_z_axis"), MENU_OPTION_Z_AXIS);
	options->get_popup()->set_item_checked(options->get_popup()->get_item_index(MENU_OPTION_Y_AXIS), true);
	options->get_popup()->add_separator();
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_x"), MENU_OPTION_CURSOR_ROTATE_X);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_y"), MENU_OPTION_CURSOR_ROTATE_Y);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_rotate_z"), MENU_OPTION_CURSOR_ROTATE_Z);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_x"), MENU_OPTION_CURSOR_BACK_ROTATE_X);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_y"), MENU_OPTION_CURSOR_BACK_ROTATE_Y);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_back_rotate_z"), MENU_OPTION_CURSOR_BACK_ROTATE_Z);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cursor_clear_rotation"), MENU_OPTION_CURSOR_CLEAR_ROTATION);
	options->get_popup()->add_separator();
	// TRANSLATORS: This is a toggle to select after pasting the new content.
	options->get_popup()->add_check_shortcut(ED_GET_SHORTCUT("grid_map/paste_selects"), MENU_OPTION_PASTE_SELECTS);
	options->get_popup()->add_separator();
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/duplicate_selection"), MENU_OPTION_SELECTION_DUPLICATE);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/cut_selection"), MENU_OPTION_SELECTION_CUT);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/clear_selection"), MENU_OPTION_SELECTION_CLEAR);
	options->get_popup()->add_shortcut(ED_GET_SHORTCUT("grid_map/fill_selection"), MENU_OPTION_SELECTION_FILL);

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

	options->get_popup()->connect("id_pressed", callable_mp(this, &GridMapEditor::_menu_option));

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_h_size_flags(SIZE_EXPAND_FILL);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Filter Meshes"));
	search_box->set_clear_button_enabled(true);
	hb->add_child(search_box);
	search_box->connect("text_changed", callable_mp(this, &GridMapEditor::_text_changed));
	search_box->connect("gui_input", callable_mp(this, &GridMapEditor::_sbox_input));

	mode_thumbnail = memnew(Button);
	mode_thumbnail->set_theme_type_variation("FlatButton");
	mode_thumbnail->set_toggle_mode(true);
	mode_thumbnail->set_pressed(true);
	hb->add_child(mode_thumbnail);
	mode_thumbnail->connect("pressed", callable_mp(this, &GridMapEditor::_set_display_mode).bind(DISPLAY_THUMBNAIL));

	mode_list = memnew(Button);
	mode_list->set_theme_type_variation("FlatButton");
	mode_list->set_toggle_mode(true);
	mode_list->set_pressed(false);
	hb->add_child(mode_list);
	mode_list->connect("pressed", callable_mp(this, &GridMapEditor::_set_display_mode).bind(DISPLAY_LIST));

	size_slider = memnew(HSlider);
	size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
	size_slider->set_min(0.2f);
	size_slider->set_max(4.0f);
	size_slider->set_step(0.1f);
	size_slider->set_value(1.0f);
	size_slider->connect("value_changed", callable_mp(this, &GridMapEditor::_icon_size_changed));
	add_child(size_slider);

	EDITOR_DEF("editors/grid_map/preview_size", 64);

	mesh_library_palette = memnew(ItemList);
	mesh_library_palette->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	add_child(mesh_library_palette);
	mesh_library_palette->set_v_size_flags(SIZE_EXPAND_FILL);
	mesh_library_palette->connect("gui_input", callable_mp(this, &GridMapEditor::_mesh_library_palette_input));

	info_message = memnew(Label);
	info_message->set_text(TTR("Give a MeshLibrary resource to this GridMap to use its meshes."));
	info_message->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	info_message->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	info_message->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	info_message->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	info_message->set_anchors_and_offsets_preset(PRESET_FULL_RECT, PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);
	mesh_library_palette->add_child(info_message);

	edit_axis = Vector3::AXIS_Y;
	edit_floor[0] = -1;
	edit_floor[1] = -1;
	edit_floor[2] = -1;

	selection_mesh = RenderingServer::get_singleton()->mesh_create();
	paste_mesh = RenderingServer::get_singleton()->mesh_create();

	{
		// Selection mesh create.

		Vector<Vector3> lines;
		Vector<Vector3> triangles;
		Vector<Vector3> square[3];

		for (int i = 0; i < 6; i++) {
			Vector3 face_points[4];

			for (int j = 0; j < 4; j++) {
				float v[3];
				v[0] = 1.0;
				v[1] = 1 - 2 * ((j >> 1) & 1);
				v[2] = v[1] * (1 - 2 * (j & 1));

				for (int k = 0; k < 3; k++) {
					if (i < 3) {
						face_points[j][(i + k) % 3] = v[k];
					} else {
						face_points[3 - j][(i + k) % 3] = -v[k];
					}
				}
			}

			triangles.push_back(face_points[0] * 0.5 + Vector3(0.5, 0.5, 0.5));
			triangles.push_back(face_points[1] * 0.5 + Vector3(0.5, 0.5, 0.5));
			triangles.push_back(face_points[2] * 0.5 + Vector3(0.5, 0.5, 0.5));

			triangles.push_back(face_points[2] * 0.5 + Vector3(0.5, 0.5, 0.5));
			triangles.push_back(face_points[3] * 0.5 + Vector3(0.5, 0.5, 0.5));
			triangles.push_back(face_points[0] * 0.5 + Vector3(0.5, 0.5, 0.5));
		}

		for (int i = 0; i < 12; i++) {
			AABB base(Vector3(0, 0, 0), Vector3(1, 1, 1));
			Vector3 a, b;
			base.get_edge(i, a, b);
			lines.push_back(a);
			lines.push_back(b);
		}

		for (int i = 0; i < 3; i++) {
			Vector3 points[4];
			for (int j = 0; j < 4; j++) {
				static const bool orderx[4] = { false, true, true, false };
				static const bool ordery[4] = { false, false, true, true };

				Vector3 sp;
				if (orderx[j]) {
					sp[(i + 1) % 3] = 1.0;
				}
				if (ordery[j]) {
					sp[(i + 2) % 3] = 1.0;
				}

				points[j] = sp;
			}

			for (int j = 0; j < 4; j++) {
				Vector3 ofs;
				ofs[i] += 0.01;
				square[i].push_back(points[j] - ofs);
				square[i].push_back(points[(j + 1) % 4] - ofs);
				square[i].push_back(points[j] + ofs);
				square[i].push_back(points[(j + 1) % 4] + ofs);
			}
		}

		Array d;
		d.resize(RS::ARRAY_MAX);

		inner_mat.instantiate();
		inner_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.2));
		inner_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		inner_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);
		inner_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);

		d[RS::ARRAY_VERTEX] = triangles;
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(selection_mesh, RS::PRIMITIVE_TRIANGLES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(selection_mesh, 0, inner_mat->get_rid());

		outer_mat.instantiate();
		outer_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.8));
		outer_mat->set_on_top_of_alpha();

		outer_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		outer_mat->set_transparency(StandardMaterial3D::TRANSPARENCY_ALPHA);
		outer_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);

		selection_floor_mat.instantiate();
		selection_floor_mat->set_albedo(Color(0.80, 0.80, 1.0, 1));
		selection_floor_mat->set_on_top_of_alpha();
		selection_floor_mat->set_shading_mode(StandardMaterial3D::SHADING_MODE_UNSHADED);
		selection_floor_mat->set_flag(StandardMaterial3D::FLAG_DISABLE_FOG, true);

		d[RS::ARRAY_VERTEX] = lines;
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(selection_mesh, RS::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(selection_mesh, 1, outer_mat->get_rid());

		d[RS::ARRAY_VERTEX] = triangles;
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(paste_mesh, RS::PRIMITIVE_TRIANGLES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(paste_mesh, 0, inner_mat->get_rid());

		d[RS::ARRAY_VERTEX] = lines;
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(paste_mesh, RS::PRIMITIVE_LINES, d);
		RenderingServer::get_singleton()->mesh_surface_set_material(paste_mesh, 1, outer_mat->get_rid());

		for (int i = 0; i < 3; i++) {
			d[RS::ARRAY_VERTEX] = square[i];
			selection_level_mesh[i] = RS::get_singleton()->mesh_create();
			RenderingServer::get_singleton()->mesh_add_surface_from_arrays(selection_level_mesh[i], RS::PRIMITIVE_LINES, d);
			RenderingServer::get_singleton()->mesh_surface_set_material(selection_level_mesh[i], 0, selection_floor_mat->get_rid());
		}
	}

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
		if (grid[i].is_valid()) {
			RenderingServer::get_singleton()->free(grid[i]);
		}
		if (grid_instance[i].is_valid()) {
			RenderingServer::get_singleton()->free(grid_instance[i]);
		}
		if (cursor_instance.is_valid()) {
			RenderingServer::get_singleton()->free(cursor_instance);
		}
		if (selection_level_instance[i].is_valid()) {
			RenderingServer::get_singleton()->free(selection_level_instance[i]);
		}
		if (selection_level_mesh[i].is_valid()) {
			RenderingServer::get_singleton()->free(selection_level_mesh[i]);
		}
	}

	RenderingServer::get_singleton()->free(selection_mesh);
	if (selection_instance.is_valid()) {
		RenderingServer::get_singleton()->free(selection_instance);
	}

	RenderingServer::get_singleton()->free(paste_mesh);
	if (paste_instance.is_valid()) {
		RenderingServer::get_singleton()->free(paste_instance);
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
	EDITOR_DEF("editors/grid_map/editor_side", 1);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "editors/grid_map/editor_side", PROPERTY_HINT_ENUM, "Left,Right"));

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
