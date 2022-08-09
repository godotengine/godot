/*************************************************************************/
/*  grid_map_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "grid_map_editor_plugin.h"
#include "core/os/input.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "scene/3d/camera.h"

#include "core/math/geometry.h"
#include "core/os/keyboard.h"

void GridMapEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
	}
}

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

		case MENU_OPTION_LOCK_VIEW: {
			int index = options->get_popup()->get_item_index(MENU_OPTION_LOCK_VIEW);
			lock_view = !options->get_popup()->is_item_checked(index);

			options->get_popup()->set_item_checked(index, lock_view);
		} break;
		case MENU_OPTION_CLIP_DISABLED:
		case MENU_OPTION_CLIP_ABOVE:
		case MENU_OPTION_CLIP_BELOW: {
			clip_mode = ClipMode(p_option - MENU_OPTION_CLIP_DISABLED);
			for (int i = 0; i < 3; i++) {
				int index = options->get_popup()->get_item_index(MENU_OPTION_CLIP_DISABLED + i);
				options->get_popup()->set_item_checked(index, i == clip_mode);
			}

			_update_clip();
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
			_update_clip();

		} break;
		case MENU_OPTION_CURSOR_ROTATE_Y: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), -Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), -Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_X: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), -Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), -Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_ROTATE_Z: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), -Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), -Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Y: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 1, 0), Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 1, 0), Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_X: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(1, 0, 0), Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
			_update_cursor_transform();
		} break;
		case MENU_OPTION_CURSOR_BACK_ROTATE_Z: {
			Basis r;
			if (input_action == INPUT_PASTE) {
				r.set_orthogonal_index(paste_indicator.orientation);
				r.rotate(Vector3(0, 0, 1), Math_PI / 2.0);
				paste_indicator.orientation = r.get_orthogonal_index();
				_update_paste_indicator();
				break;
			}

			r.set_orthogonal_index(cursor_rot);
			r.rotate(Vector3(0, 0, 1), Math_PI / 2.0);
			cursor_rot = r.get_orthogonal_index();
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
	cursor_transform = Transform();
	cursor_transform.origin = cursor_origin;
	cursor_transform.basis.set_orthogonal_index(cursor_rot);
	cursor_transform.basis *= node->get_cell_scale();
	cursor_transform = node->get_global_transform() * cursor_transform;

	if (selected_palette >= 0) {
		if (node && !node->get_mesh_library().is_null()) {
			cursor_transform *= node->get_mesh_library()->get_item_mesh_transform(selected_palette);
		}
	}

	if (cursor_instance.is_valid()) {
		VisualServer::get_singleton()->instance_set_transform(cursor_instance, cursor_transform);
		VisualServer::get_singleton()->instance_set_visible(cursor_instance, cursor_visible);
	}
}

void GridMapEditor::_update_selection_transform() {
	Transform xf_zero;
	xf_zero.basis.set_zero();

	if (!selection.active) {
		VisualServer::get_singleton()->instance_set_transform(selection_instance, xf_zero);
		for (int i = 0; i < 3; i++) {
			VisualServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf_zero);
		}
		return;
	}

	Transform xf;
	xf.scale((Vector3(1, 1, 1) + (selection.end - selection.begin)) * node->get_cell_size());
	xf.origin = selection.begin * node->get_cell_size();

	VisualServer::get_singleton()->instance_set_transform(selection_instance, node->get_global_transform() * xf);

	for (int i = 0; i < 3; i++) {
		if (i != edit_axis || (edit_floor[edit_axis] < selection.begin[edit_axis]) || (edit_floor[edit_axis] > selection.end[edit_axis] + 1)) {
			VisualServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf_zero);
		} else {
			Vector3 scale = (selection.end - selection.begin + Vector3(1, 1, 1));
			scale[edit_axis] = 1.0;
			Vector3 pos = selection.begin;
			pos[edit_axis] = edit_floor[edit_axis];

			scale *= node->get_cell_size();
			pos *= node->get_cell_size();

			Transform xf2;
			xf2.basis.scale(scale);
			xf2.origin = pos;

			VisualServer::get_singleton()->instance_set_transform(selection_level_instance[i], xf2);
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

bool GridMapEditor::do_input_action(Camera *p_camera, const Point2 &p_point, bool p_click) {
	if (!spatial_editor) {
		return false;
	}

	if (selected_palette < 0 && input_action != INPUT_PICK && input_action != INPUT_SELECT && input_action != INPUT_PASTE) {
		return false;
	}
	Ref<MeshLibrary> mesh_library = node->get_mesh_library();
	if (mesh_library.is_null()) {
		return false;
	}
	if (input_action != INPUT_PICK && input_action != INPUT_SELECT && input_action != INPUT_PASTE && !mesh_library->has_item(selected_palette)) {
		return false;
	}

	Camera *camera = p_camera;
	Vector3 from = camera->project_ray_origin(p_point);
	Vector3 normal = camera->project_ray_normal(p_point);
	Transform local_xform = node->get_global_transform().affine_inverse();
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
	float cell_size[3] = { node->get_cell_size().x, node->get_cell_size().y, node->get_cell_size().z };

	for (int i = 0; i < 3; i++) {
		if (i == edit_axis) {
			cell[i] = edit_floor[i];
		} else {
			cell[i] = inters[i] / node->get_cell_size()[i];
			if (inters[i] < 0) {
				cell[i] -= 1; // Compensate negative.
			}
			grid_ofs[i] = cell[i] * cell_size[i];
		}
	}

	VS::get_singleton()->instance_set_transform(grid_instance[edit_axis], node->get_global_transform() * edit_grid_xform);

	if (cursor_instance.is_valid()) {
		cursor_origin = (Vector3(cell[0], cell[1], cell[2]) + Vector3(0.5 * node->get_center_x(), 0.5 * node->get_center_y(), 0.5 * node->get_center_z())) * node->get_cell_size();
		cursor_visible = true;

		if (input_action == INPUT_SELECT || input_action == INPUT_PASTE) {
			cursor_visible = false;
		}

		_update_cursor_transform();
	}

	if (input_action == INPUT_PASTE) {
		paste_indicator.current = Vector3(cell[0], cell[1], cell[2]);
		_update_paste_indicator();

	} else if (input_action == INPUT_SELECT) {
		selection.current = Vector3(cell[0], cell[1], cell[2]);
		if (p_click) {
			selection.click = selection.current;
		}
		selection.active = true;
		_validate_selection();

		return true;
	} else if (input_action == INPUT_PICK) {
		int item = node->get_cell_item(cell[0], cell[1], cell[2]);
		if (item >= 0) {
			selected_palette = item;
			mesh_library_palette->set_current(item);
			update_palette();
			_update_cursor_instance();
		}
		return true;
	}
	if (input_action == INPUT_PAINT) {
		SetItem si;
		si.pos = Vector3(cell[0], cell[1], cell[2]);
		si.new_value = selected_palette;
		si.new_orientation = cursor_rot;
		si.old_value = node->get_cell_item(cell[0], cell[1], cell[2]);
		si.old_orientation = node->get_cell_item_orientation(cell[0], cell[1], cell[2]);
		set_items.push_back(si);
		node->set_cell_item(cell[0], cell[1], cell[2], selected_palette, cursor_rot);
		return true;
	} else if (input_action == INPUT_ERASE) {
		SetItem si;
		si.pos = Vector3(cell[0], cell[1], cell[2]);
		si.new_value = -1;
		si.new_orientation = 0;
		si.old_value = node->get_cell_item(cell[0], cell[1], cell[2]);
		si.old_orientation = node->get_cell_item_orientation(cell[0], cell[1], cell[2]);
		set_items.push_back(si);
		node->set_cell_item(cell[0], cell[1], cell[2], -1);
		return true;
	}

	return false;
}

void GridMapEditor::_delete_selection() {
	if (!selection.active) {
		return;
	}

	undo_redo->create_action(TTR("GridMap Delete Selection"));
	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				undo_redo->add_do_method(node, "set_cell_item", i, j, k, GridMap::INVALID_CELL_ITEM);
				undo_redo->add_undo_method(node, "set_cell_item", i, j, k, node->get_cell_item(i, j, k), node->get_cell_item_orientation(i, j, k));
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

	undo_redo->create_action(TTR("GridMap Fill Selection"));
	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				undo_redo->add_do_method(node, "set_cell_item", i, j, k, selected_palette, cursor_rot);
				undo_redo->add_undo_method(node, "set_cell_item", i, j, k, node->get_cell_item(i, j, k), node->get_cell_item_orientation(i, j, k));
			}
		}
	}
	undo_redo->add_do_method(this, "_set_selection", !selection.active, selection.begin, selection.end);
	undo_redo->add_undo_method(this, "_set_selection", selection.active, selection.begin, selection.end);
	undo_redo->commit_action();
}

void GridMapEditor::_clear_clipboard_data() {
	for (List<ClipboardItem>::Element *E = clipboard_items.front(); E; E = E->next()) {
		if (E->get().instance.is_valid()) {
			VisualServer::get_singleton()->free(E->get().instance);
		}
	}

	clipboard_items.clear();
}

void GridMapEditor::_set_clipboard_data() {
	_clear_clipboard_data();

	Ref<MeshLibrary> meshLibrary = node->get_mesh_library();

	for (int i = selection.begin.x; i <= selection.end.x; i++) {
		for (int j = selection.begin.y; j <= selection.end.y; j++) {
			for (int k = selection.begin.z; k <= selection.end.z; k++) {
				int itm = node->get_cell_item(i, j, k);
				if (itm == GridMap::INVALID_CELL_ITEM) {
					continue;
				}

				Ref<Mesh> mesh = meshLibrary->get_item_mesh(itm);

				ClipboardItem item;
				item.cell_item = itm;
				item.grid_offset = Vector3(i, j, k) - selection.begin;
				item.orientation = node->get_cell_item_orientation(i, j, k);
				item.instance = VisualServer::get_singleton()->instance_create2(mesh->get_rid(), get_tree()->get_root()->get_world()->get_scenario());

				clipboard_items.push_back(item);
			}
		}
	}
}

void GridMapEditor::_update_paste_indicator() {
	if (input_action != INPUT_PASTE) {
		Transform xf;
		xf.basis.set_zero();
		VisualServer::get_singleton()->instance_set_transform(paste_instance, xf);
		return;
	}

	Vector3 center = 0.5 * Vector3(float(node->get_center_x()), float(node->get_center_y()), float(node->get_center_z()));
	Vector3 scale = (Vector3(1, 1, 1) + (paste_indicator.end - paste_indicator.begin)) * node->get_cell_size();
	Transform xf;
	xf.scale(scale);
	xf.origin = (paste_indicator.begin + (paste_indicator.current - paste_indicator.click) + center) * node->get_cell_size();
	Basis rot;
	rot.set_orthogonal_index(paste_indicator.orientation);
	xf.basis = rot * xf.basis;
	xf.translate((-center * node->get_cell_size()) / scale);

	VisualServer::get_singleton()->instance_set_transform(paste_instance, node->get_global_transform() * xf);

	for (List<ClipboardItem>::Element *E = clipboard_items.front(); E; E = E->next()) {
		ClipboardItem &item = E->get();

		xf = Transform();
		xf.origin = (paste_indicator.begin + (paste_indicator.current - paste_indicator.click) + center) * node->get_cell_size();
		xf.basis = rot * xf.basis;
		xf.translate(item.grid_offset * node->get_cell_size());

		Basis item_rot;
		item_rot.set_orthogonal_index(item.orientation);
		xf.basis = item_rot * xf.basis * node->get_cell_scale();

		VisualServer::get_singleton()->instance_set_transform(item.instance, node->get_global_transform() * xf);
	}
}

void GridMapEditor::_do_paste() {
	int idx = options->get_popup()->get_item_index(MENU_OPTION_PASTE_SELECTS);
	bool reselect = options->get_popup()->is_item_checked(idx);

	Basis rot;
	rot.set_orthogonal_index(paste_indicator.orientation);

	Vector3 ofs = paste_indicator.current - paste_indicator.click;
	undo_redo->create_action(TTR("GridMap Paste Selection"));

	for (List<ClipboardItem>::Element *E = clipboard_items.front(); E; E = E->next()) {
		ClipboardItem &item = E->get();

		Vector3 pos = rot.xform(item.grid_offset) + paste_indicator.begin + ofs;

		Basis orm;
		orm.set_orthogonal_index(item.orientation);
		orm = rot * orm;

		undo_redo->add_do_method(node, "set_cell_item", pos.x, pos.y, pos.z, item.cell_item, orm.get_orthogonal_index());
		undo_redo->add_undo_method(node, "set_cell_item", pos.x, pos.y, pos.z, node->get_cell_item(pos.x, pos.y, pos.z), node->get_cell_item_orientation(pos.x, pos.y, pos.z));
	}

	if (reselect) {
		undo_redo->add_do_method(this, "_set_selection", true, paste_indicator.begin + ofs, paste_indicator.end + ofs);
		undo_redo->add_undo_method(this, "_set_selection", selection.active, selection.begin, selection.end);
	}

	undo_redo->commit_action();

	_clear_clipboard_data();
}

bool GridMapEditor::forward_spatial_input_event(Camera *p_camera, const Ref<InputEvent> &p_event) {
	if (!node) {
		return false;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_WHEEL_UP && (mb->get_command() || mb->get_shift())) {
			if (mb->is_pressed()) {
				floor->set_value(floor->get_value() + mb->get_factor());
			}

			return true; // Eaten.
		} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN && (mb->get_command() || mb->get_shift())) {
			if (mb->is_pressed()) {
				floor->set_value(floor->get_value() - mb->get_factor());
			}
			return true;
		}

		if (mb->is_pressed()) {
			SpatialEditorViewport::NavigationScheme nav_scheme = (SpatialEditorViewport::NavigationScheme)EditorSettings::get_singleton()->get("editors/3d/navigation/navigation_scheme").operator int();
			if ((nav_scheme == SpatialEditorViewport::NAVIGATION_MAYA || nav_scheme == SpatialEditorViewport::NAVIGATION_MODO) && mb->get_alt()) {
				input_action = INPUT_NONE;
			} else if (mb->get_button_index() == BUTTON_LEFT) {
				bool can_edit = (node && node->get_mesh_library().is_valid());
				if (input_action == INPUT_PASTE) {
					_do_paste();
					input_action = INPUT_NONE;
					_update_paste_indicator();
				} else if (mb->get_shift() && can_edit) {
					input_action = INPUT_SELECT;
					last_selection = selection;
				} else if (mb->get_command() && can_edit) {
					input_action = INPUT_PICK;
				} else {
					input_action = INPUT_PAINT;
					set_items.clear();
				}
			} else if (mb->get_button_index() == BUTTON_RIGHT) {
				if (input_action == INPUT_PASTE) {
					_clear_clipboard_data();
					input_action = INPUT_NONE;
					_update_paste_indicator();
					return true;
				} else if (selection.active) {
					_set_selection(false);
					return true;
				} else {
					input_action = INPUT_ERASE;
					set_items.clear();
				}
			} else {
				return false;
			}

			return do_input_action(p_camera, Point2(mb->get_position().x, mb->get_position().y), true);
		} else {
			if ((mb->get_button_index() == BUTTON_RIGHT && input_action == INPUT_ERASE) || (mb->get_button_index() == BUTTON_LEFT && input_action == INPUT_PAINT)) {
				if (set_items.size()) {
					undo_redo->create_action(TTR("GridMap Paint"));
					for (List<SetItem>::Element *E = set_items.front(); E; E = E->next()) {
						const SetItem &si = E->get();
						undo_redo->add_do_method(node, "set_cell_item", si.pos.x, si.pos.y, si.pos.z, si.new_value, si.new_orientation);
					}
					for (List<SetItem>::Element *E = set_items.back(); E; E = E->prev()) {
						const SetItem &si = E->get();
						undo_redo->add_undo_method(node, "set_cell_item", si.pos.x, si.pos.y, si.pos.z, si.old_value, si.old_orientation);
					}

					undo_redo->commit_action();
				}
				set_items.clear();
				input_action = INPUT_NONE;
				return set_items.size() > 0;
			}

			if (mb->get_button_index() == BUTTON_LEFT && input_action == INPUT_SELECT) {
				undo_redo->create_action(TTR("GridMap Selection"));
				undo_redo->add_do_method(this, "_set_selection", selection.active, selection.begin, selection.end);
				undo_redo->add_undo_method(this, "_set_selection", last_selection.active, last_selection.begin, last_selection.end);
				undo_redo->commit_action();
			}

			if (mb->get_button_index() == BUTTON_LEFT && input_action != INPUT_NONE) {
				set_items.clear();
				input_action = INPUT_NONE;
				return true;
			}
			if (mb->get_button_index() == BUTTON_RIGHT && (input_action == INPUT_ERASE || input_action == INPUT_PASTE)) {
				input_action = INPUT_NONE;
				return true;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		return do_input_action(p_camera, mm->get_position(), false);
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->is_pressed()) {
			if (k->get_scancode() == KEY_ESCAPE) {
				if (input_action == INPUT_PASTE) {
					_clear_clipboard_data();
					input_action = INPUT_NONE;
					_update_paste_indicator();
					return true;
				} else if (selection.active) {
					_set_selection(false);
					return true;
				} else {
					selected_palette = -1;
					mesh_library_palette->unselect_all();
					update_palette();
					_update_cursor_instance();
					return true;
				}
			}

			if (k->get_shift() && selection.active && input_action != INPUT_PASTE) {
				if (k->get_scancode() == options->get_popup()->get_item_accelerator(options->get_popup()->get_item_index(MENU_OPTION_PREV_LEVEL))) {
					selection.click[edit_axis]--;
					_validate_selection();
					return true;
				}
				if (k->get_scancode() == options->get_popup()->get_item_accelerator(options->get_popup()->get_item_index(MENU_OPTION_NEXT_LEVEL))) {
					selection.click[edit_axis]++;
					_validate_selection();
					return true;
				}
			}
		}
	}

	Ref<InputEventPanGesture> pan_gesture = p_event;
	if (pan_gesture.is_valid()) {
		if (pan_gesture->get_alt() && (pan_gesture->get_command() || pan_gesture->get_shift())) {
			const real_t delta = pan_gesture->get_delta().y * 0.5;
			accumulated_floor_delta += delta;
			int step = 0;
			if (ABS(accumulated_floor_delta) > 1.0) {
				step = SGN(accumulated_floor_delta);
				accumulated_floor_delta -= step;
			}
			if (step) {
				floor->set_value(floor->get_value() + step);
			}
			return true;
		}
	}
	accumulated_floor_delta = 0.0;

	return false;
}

struct _CGMEItemSort {
	String name;
	int id;
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

	if (k.is_valid() && (k->get_scancode() == KEY_UP || k->get_scancode() == KEY_DOWN || k->get_scancode() == KEY_PAGEUP || k->get_scancode() == KEY_PAGEDOWN)) {
		// Forward the key input to the ItemList so it can be scrolled
		mesh_library_palette->call("_gui_input", k);
		search_box->accept_event();
	}
}

void GridMapEditor::_mesh_library_palette_input(const Ref<InputEvent> &p_ie) {
	const Ref<InputEventMouseButton> mb = p_ie;

	// Zoom in/out using Ctrl + mouse wheel
	if (mb.is_valid() && mb->is_pressed() && mb->get_command()) {
		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_UP) {
			size_slider->set_value(size_slider->get_value() + 0.2);
		}

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_DOWN) {
			size_slider->set_value(size_slider->get_value() - 0.2);
		}
	}
}

void GridMapEditor::_icon_size_changed(float p_value) {
	mesh_library_palette->set_icon_scale(p_value);
	update_palette();
}

void GridMapEditor::update_palette() {
	int selected = mesh_library_palette->get_current();

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

	Ref<MeshLibrary> mesh_library = node->get_mesh_library();

	if (mesh_library.is_null()) {
		last_mesh_library = nullptr;
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

	for (List<_CGMEItemSort>::Element *E = il.front(); E; E = E->next()) {
		int id = E->get().id;
		String name = mesh_library->get_item_name(id);
		Ref<Texture> preview = mesh_library->get_item_preview(id);

		if (name == "") {
			name = "#" + itos(id);
		}

		if (filter != "" && !filter.is_subsequence_ofi(name)) {
			continue;
		}

		mesh_library_palette->add_item("");
		if (!preview.is_null()) {
			mesh_library_palette->set_item_icon(item, preview);
			mesh_library_palette->set_item_tooltip(item, name);
		}
		mesh_library_palette->set_item_text(item, name);
		mesh_library_palette->set_item_metadata(item, id);

		item++;
	}

	if (selected != -1 && mesh_library_palette->get_item_count() > 0) {
		// Make sure that this variable is set correctly.
		selected_palette = MIN(selected, mesh_library_palette->get_item_count() - 1);
		mesh_library_palette->select(selected_palette);
	}

	last_mesh_library = *mesh_library;
}

void GridMapEditor::edit(GridMap *p_gridmap) {
	if (!p_gridmap && node) {
		node->disconnect("cell_size_changed", this, "_draw_grids");
	}

	node = p_gridmap;

	input_action = INPUT_NONE;
	selection.active = false;
	_update_selection_transform();
	_update_paste_indicator();

	spatial_editor = Object::cast_to<SpatialEditorPlugin>(editor->get_editor_plugin_screen());

	if (!node) {
		set_process(false);
		for (int i = 0; i < 3; i++) {
			VisualServer::get_singleton()->instance_set_visible(grid_instance[i], false);
		}

		if (cursor_instance.is_valid()) {
			VisualServer::get_singleton()->instance_set_visible(cursor_instance, false);
		}

		return;
	}

	update_palette();
	_update_cursor_instance();

	set_process(true);

	clip_mode = p_gridmap->has_meta("_editor_clip_") ? ClipMode(p_gridmap->get_meta("_editor_clip_").operator int()) : CLIP_DISABLED;

	_draw_grids(node->get_cell_size());
	update_grid();
	_update_clip();

	node->connect("cell_size_changed", this, "_draw_grids");
}

void GridMapEditor::_update_clip() {
	node->set_meta("_editor_clip_", clip_mode);
	if (clip_mode == CLIP_DISABLED) {
		node->set_clip(false);
	} else {
		node->set_clip(true, clip_mode == CLIP_ABOVE, edit_floor[edit_axis], edit_axis);
	}
}

void GridMapEditor::update_grid() {
	grid_xform.origin.x -= 1; // Force update in hackish way.

	grid_ofs[edit_axis] = edit_floor[edit_axis] * node->get_cell_size()[edit_axis];

	edit_grid_xform.origin = grid_ofs;
	edit_grid_xform.basis = Basis();

	for (int i = 0; i < 3; i++) {
		VisualServer::get_singleton()->instance_set_visible(grid_instance[i], i == edit_axis);
	}

	updating = true;
	floor->set_value(edit_floor[edit_axis]);
	updating = false;
}

void GridMapEditor::_draw_grids(const Vector3 &cell_size) {
	Vector3 edited_floor = node->has_meta("_editor_floor_") ? node->get_meta("_editor_floor_") : Variant();

	for (int i = 0; i < 3; i++) {
		if (VS::get_singleton()->mesh_get_surface_count(grid[i]) > 0) {
			VS::get_singleton()->mesh_remove_surface(grid[i], 0);
		}
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
		d.resize(VS::ARRAY_MAX);
		d[VS::ARRAY_VERTEX] = grid_points[i];
		d[VS::ARRAY_COLOR] = grid_colors[i];
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(grid[i], VisualServer::PRIMITIVE_LINES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(grid[i], 0, indicator_mat->get_rid());
	}
}

void GridMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", this, "_node_removed");
			mesh_library_palette->connect("item_selected", this, "_item_selected_cbk");
			for (int i = 0; i < 3; i++) {
				grid[i] = RID_PRIME(VS::get_singleton()->mesh_create());
				grid_instance[i] = VS::get_singleton()->instance_create2(grid[i], get_tree()->get_root()->get_world()->get_scenario());
				VS::get_singleton()->instance_set_layer_mask(grid_instance[i], 1 << SpatialEditorViewport::MISC_TOOL_LAYER);
				selection_level_instance[i] = VisualServer::get_singleton()->instance_create2(selection_level_mesh[i], get_tree()->get_root()->get_world()->get_scenario());
				VS::get_singleton()->instance_set_layer_mask(selection_level_instance[i], 1 << SpatialEditorViewport::MISC_TOOL_LAYER);
			}

			selection_instance = VisualServer::get_singleton()->instance_create2(selection_mesh, get_tree()->get_root()->get_world()->get_scenario());
			VS::get_singleton()->instance_set_layer_mask(selection_instance, 1 << SpatialEditorViewport::MISC_TOOL_LAYER);
			paste_instance = VisualServer::get_singleton()->instance_create2(paste_mesh, get_tree()->get_root()->get_world()->get_scenario());
			VS::get_singleton()->instance_set_layer_mask(paste_instance, 1 << SpatialEditorViewport::MISC_TOOL_LAYER);

			_update_selection_transform();
			_update_paste_indicator();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", this, "_node_removed");
			_clear_clipboard_data();

			for (int i = 0; i < 3; i++) {
				if (grid_instance[i].is_valid()) {
					VS::get_singleton()->free(grid_instance[i]);
					grid_instance[i] = RID();
				}

				if (grid[i].is_valid()) {
					VS::get_singleton()->free(grid[i]);
					grid[i] = RID();
				}

				if (selection_level_instance[i].is_valid()) {
					VS::get_singleton()->free(selection_level_instance[i]);
					selection_level_instance[i] = RID();
				}
			}

			if (selection_instance.is_valid()) {
				VS::get_singleton()->free(selection_instance);
				selection_instance = RID();
			}

			if (paste_instance.is_valid()) {
				VS::get_singleton()->free(paste_instance);
				paste_instance = RID();
			}
		} break;

		case NOTIFICATION_PROCESS: {
			if (!node) {
				return;
			}

			Transform xf = node->get_global_transform();

			if (xf != grid_xform) {
				for (int i = 0; i < 3; i++) {
					VS::get_singleton()->instance_set_transform(grid_instance[i], xf * edit_grid_xform);
				}
				grid_xform = xf;
			}
			Ref<MeshLibrary> cgmt = node->get_mesh_library();
			if (cgmt.operator->() != last_mesh_library) {
				update_palette();
			}

			if (lock_view) {
				EditorNode *editor = Object::cast_to<EditorNode>(get_tree()->get_root()->get_child(0));

				Plane p;
				p.normal[edit_axis] = 1.0;
				p.d = edit_floor[edit_axis] * node->get_cell_size()[edit_axis];
				p = node->get_transform().xform(p); // plane to snap

				SpatialEditorPlugin *sep = Object::cast_to<SpatialEditorPlugin>(editor->get_editor_plugin_screen());
				if (sep) {
					sep->snap_cursor_to_plane(p);
				}
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			options->set_icon(get_icon("GridMap", "EditorIcons"));
			search_box->set_right_icon(get_icon("Search", "EditorIcons"));
		} break;

		case NOTIFICATION_WM_FOCUS_OUT: {
			if (input_action == INPUT_PAINT) {
				// Simulate mouse released event to stop drawing when editor focus exists.
				Ref<InputEventMouseButton> release;
				release.instance();
				release->set_button_index(BUTTON_LEFT);
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
		VisualServer::get_singleton()->free(cursor_instance);
		cursor_instance = RID();
	}

	if (selected_palette >= 0) {
		if (node && !node->get_mesh_library().is_null()) {
			Ref<Mesh> mesh = node->get_mesh_library()->get_item_mesh(selected_palette);
			if (!mesh.is_null() && mesh->get_rid().is_valid()) {
				cursor_instance = VisualServer::get_singleton()->instance_create2(mesh->get_rid(), get_tree()->get_root()->get_world()->get_scenario());
				VisualServer::get_singleton()->instance_set_transform(cursor_instance, cursor_transform);
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
	_update_clip();
	_update_selection_transform();
}

void GridMapEditor::_floor_mouse_exited() {
	floor->get_line_edit()->release_focus();
}

void GridMapEditor::_bind_methods() {
	ClassDB::bind_method("_text_changed", &GridMapEditor::_text_changed);
	ClassDB::bind_method("_sbox_input", &GridMapEditor::_sbox_input);
	ClassDB::bind_method("_mesh_library_palette_input", &GridMapEditor::_mesh_library_palette_input);
	ClassDB::bind_method("_icon_size_changed", &GridMapEditor::_icon_size_changed);
	ClassDB::bind_method("_menu_option", &GridMapEditor::_menu_option);
	ClassDB::bind_method("_configure", &GridMapEditor::_configure);
	ClassDB::bind_method("_item_selected_cbk", &GridMapEditor::_item_selected_cbk);
	ClassDB::bind_method("_floor_changed", &GridMapEditor::_floor_changed);
	ClassDB::bind_method("_floor_mouse_exited", &GridMapEditor::_floor_mouse_exited);
	ClassDB::bind_method("_set_selection", &GridMapEditor::_set_selection);
	ClassDB::bind_method("_node_removed", &GridMapEditor::_node_removed);

	ClassDB::bind_method(D_METHOD("_set_display_mode", "mode"), &GridMapEditor::_set_display_mode);
	ClassDB::bind_method("_draw_grids", &GridMapEditor::_draw_grids);
}

GridMapEditor::GridMapEditor(EditorNode *p_editor) {
	input_action = INPUT_NONE;
	node = nullptr;
	editor = p_editor;
	undo_redo = p_editor->get_undo_redo();

	int mw = EDITOR_DEF("editors/grid_map/palette_min_width", 230);
	Control *ec = memnew(Control);
	ec->set_custom_minimum_size(Size2(mw, 0) * EDSCALE);
	add_child(ec);

	spatial_editor_hb = memnew(HBoxContainer);
	spatial_editor_hb->set_h_size_flags(SIZE_EXPAND_FILL);
	spatial_editor_hb->set_alignment(BoxContainer::ALIGN_END);
	SpatialEditor::get_singleton()->add_control_to_menu_panel(spatial_editor_hb);

	spin_box_label = memnew(Label);
	spin_box_label->set_text(TTR("Floor:"));
	spatial_editor_hb->add_child(spin_box_label);

	floor = memnew(SpinBox);
	floor->set_min(-32767);
	floor->set_max(32767);
	floor->set_step(1);
	floor->get_line_edit()->add_constant_override("minimum_spaces", 16);

	spatial_editor_hb->add_child(floor);
	floor->connect("value_changed", this, "_floor_changed");
	floor->connect("mouse_exited", this, "_floor_mouse_exited");
	floor->get_line_edit()->connect("mouse_exited", this, "_floor_mouse_exited");

	spatial_editor_hb->add_child(memnew(VSeparator));

	options = memnew(MenuButton);
	spatial_editor_hb->add_child(options);
	spatial_editor_hb->hide();

	options->set_text(TTR("Grid Map"));
	options->get_popup()->add_check_item(TTR("Snap View"), MENU_OPTION_LOCK_VIEW);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Previous Floor"), MENU_OPTION_PREV_LEVEL, KEY_Q);
	options->get_popup()->add_item(TTR("Next Floor"), MENU_OPTION_NEXT_LEVEL, KEY_E);
	options->get_popup()->add_separator();
	options->get_popup()->add_radio_check_item(TTR("Clip Disabled"), MENU_OPTION_CLIP_DISABLED);
	options->get_popup()->set_item_checked(options->get_popup()->get_item_index(MENU_OPTION_CLIP_DISABLED), true);
	options->get_popup()->add_radio_check_item(TTR("Clip Above"), MENU_OPTION_CLIP_ABOVE);
	options->get_popup()->add_radio_check_item(TTR("Clip Below"), MENU_OPTION_CLIP_BELOW);
	options->get_popup()->add_separator();
	options->get_popup()->add_radio_check_item(TTR("Edit X Axis"), MENU_OPTION_X_AXIS, KEY_Z);
	options->get_popup()->add_radio_check_item(TTR("Edit Y Axis"), MENU_OPTION_Y_AXIS, KEY_X);
	options->get_popup()->add_radio_check_item(TTR("Edit Z Axis"), MENU_OPTION_Z_AXIS, KEY_C);
	options->get_popup()->set_item_checked(options->get_popup()->get_item_index(MENU_OPTION_Y_AXIS), true);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Cursor Rotate X"), MENU_OPTION_CURSOR_ROTATE_X, KEY_A);
	options->get_popup()->add_item(TTR("Cursor Rotate Y"), MENU_OPTION_CURSOR_ROTATE_Y, KEY_S);
	options->get_popup()->add_item(TTR("Cursor Rotate Z"), MENU_OPTION_CURSOR_ROTATE_Z, KEY_D);
	options->get_popup()->add_item(TTR("Cursor Back Rotate X"), MENU_OPTION_CURSOR_BACK_ROTATE_X, KEY_MASK_SHIFT + KEY_A);
	options->get_popup()->add_item(TTR("Cursor Back Rotate Y"), MENU_OPTION_CURSOR_BACK_ROTATE_Y, KEY_MASK_SHIFT + KEY_S);
	options->get_popup()->add_item(TTR("Cursor Back Rotate Z"), MENU_OPTION_CURSOR_BACK_ROTATE_Z, KEY_MASK_SHIFT + KEY_D);
	options->get_popup()->add_item(TTR("Cursor Clear Rotation"), MENU_OPTION_CURSOR_CLEAR_ROTATION, KEY_W);
	options->get_popup()->add_separator();
	options->get_popup()->add_check_item(TTR("Paste Selects"), MENU_OPTION_PASTE_SELECTS);
	options->get_popup()->add_separator();
	options->get_popup()->add_item(TTR("Duplicate Selection"), MENU_OPTION_SELECTION_DUPLICATE, KEY_MASK_CTRL + KEY_C);
	options->get_popup()->add_item(TTR("Cut Selection"), MENU_OPTION_SELECTION_CUT, KEY_MASK_CTRL + KEY_X);
	options->get_popup()->add_item(TTR("Clear Selection"), MENU_OPTION_SELECTION_CLEAR, KEY_DELETE);
	options->get_popup()->add_item(TTR("Fill Selection"), MENU_OPTION_SELECTION_FILL, KEY_MASK_CTRL + KEY_F);

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
	settings_pick_distance->set_value(EDITOR_DEF("editors/grid_map/pick_distance", 5000.0));
	settings_vbc->add_margin_child(TTR("Pick Distance:"), settings_pick_distance);

	clip_mode = CLIP_DISABLED;
	options->get_popup()->connect("id_pressed", this, "_menu_option");

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);
	hb->set_h_size_flags(SIZE_EXPAND_FILL);

	search_box = memnew(LineEdit);
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_box->set_placeholder(TTR("Filter meshes"));
	hb->add_child(search_box);
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");

	mode_thumbnail = memnew(ToolButton);
	mode_thumbnail->set_toggle_mode(true);
	mode_thumbnail->set_pressed(true);
	mode_thumbnail->set_icon(p_editor->get_gui_base()->get_icon("FileThumbnail", "EditorIcons"));
	hb->add_child(mode_thumbnail);
	mode_thumbnail->connect("pressed", this, "_set_display_mode", varray(DISPLAY_THUMBNAIL));

	mode_list = memnew(ToolButton);
	mode_list->set_toggle_mode(true);
	mode_list->set_pressed(false);
	mode_list->set_icon(p_editor->get_gui_base()->get_icon("FileList", "EditorIcons"));
	hb->add_child(mode_list);
	mode_list->connect("pressed", this, "_set_display_mode", varray(DISPLAY_LIST));

	size_slider = memnew(HSlider);
	size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
	size_slider->set_min(0.2f);
	size_slider->set_max(4.0f);
	size_slider->set_step(0.1f);
	size_slider->set_value(1.0f);
	size_slider->connect("value_changed", this, "_icon_size_changed");
	add_child(size_slider);

	display_mode = DISPLAY_THUMBNAIL;

	mesh_library_palette = memnew(ItemList);
	add_child(mesh_library_palette);
	mesh_library_palette->set_v_size_flags(SIZE_EXPAND_FILL);
	mesh_library_palette->connect("gui_input", this, "_mesh_library_palette_input");

	info_message = memnew(Label);
	info_message->set_text(TTR("Give a MeshLibrary resource to this GridMap to use its meshes."));
	info_message->set_valign(Label::VALIGN_CENTER);
	info_message->set_align(Label::ALIGN_CENTER);
	info_message->set_autowrap(true);
	info_message->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	info_message->set_anchors_and_margins_preset(PRESET_WIDE, PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);
	mesh_library_palette->add_child(info_message);

	edit_axis = Vector3::AXIS_Y;
	edit_floor[0] = -1;
	edit_floor[1] = -1;
	edit_floor[2] = -1;

	cursor_visible = false;
	selected_palette = -1;
	lock_view = false;
	cursor_rot = 0;

	selection_mesh = RID_PRIME(VisualServer::get_singleton()->mesh_create());
	paste_mesh = RID_PRIME(VisualServer::get_singleton()->mesh_create());

	{
		// Selection mesh create.

		PoolVector<Vector3> lines;
		PoolVector<Vector3> triangles;
		PoolVector<Vector3> square[3];

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
		d.resize(VS::ARRAY_MAX);

		inner_mat.instance();
		inner_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.2));
		inner_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		inner_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);

		d[VS::ARRAY_VERTEX] = triangles;
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(selection_mesh, VS::PRIMITIVE_TRIANGLES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(selection_mesh, 0, inner_mat->get_rid());

		outer_mat.instance();
		outer_mat->set_albedo(Color(0.7, 0.7, 1.0, 0.8));
		outer_mat->set_on_top_of_alpha();
		outer_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		outer_mat->set_line_width(3.0);
		outer_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);

		selection_floor_mat.instance();
		selection_floor_mat->set_albedo(Color(0.80, 0.80, 1.0, 1));
		selection_floor_mat->set_on_top_of_alpha();
		selection_floor_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		selection_floor_mat->set_line_width(3.0);

		d[VS::ARRAY_VERTEX] = lines;
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(selection_mesh, VS::PRIMITIVE_LINES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(selection_mesh, 1, outer_mat->get_rid());

		d[VS::ARRAY_VERTEX] = triangles;
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(paste_mesh, VS::PRIMITIVE_TRIANGLES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(paste_mesh, 0, inner_mat->get_rid());

		d[VS::ARRAY_VERTEX] = lines;
		VisualServer::get_singleton()->mesh_add_surface_from_arrays(paste_mesh, VS::PRIMITIVE_LINES, d);
		VisualServer::get_singleton()->mesh_surface_set_material(paste_mesh, 1, outer_mat->get_rid());

		for (int i = 0; i < 3; i++) {
			d[VS::ARRAY_VERTEX] = square[i];
			selection_level_mesh[i] = RID_PRIME(VS::get_singleton()->mesh_create());
			VisualServer::get_singleton()->mesh_add_surface_from_arrays(selection_level_mesh[i], VS::PRIMITIVE_LINES, d);
			VisualServer::get_singleton()->mesh_surface_set_material(selection_level_mesh[i], 0, selection_floor_mat->get_rid());
		}
	}

	_set_selection(false);
	updating = false;
	accumulated_floor_delta = 0.0;

	indicator_mat.instance();
	indicator_mat->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
	indicator_mat->set_feature(SpatialMaterial::FEATURE_TRANSPARENT, true);
	indicator_mat->set_flag(SpatialMaterial::FLAG_SRGB_VERTEX_COLOR, true);
	indicator_mat->set_flag(SpatialMaterial::FLAG_ALBEDO_FROM_VERTEX_COLOR, true);
	indicator_mat->set_albedo(Color(0.8, 0.5, 0.1));
}

GridMapEditor::~GridMapEditor() {
	_clear_clipboard_data();

	for (int i = 0; i < 3; i++) {
		if (grid[i].is_valid()) {
			VisualServer::get_singleton()->free(grid[i]);
		}
		if (grid_instance[i].is_valid()) {
			VisualServer::get_singleton()->free(grid_instance[i]);
		}
		if (cursor_instance.is_valid()) {
			VisualServer::get_singleton()->free(cursor_instance);
		}
		if (selection_level_instance[i].is_valid()) {
			VisualServer::get_singleton()->free(selection_level_instance[i]);
		}
		if (selection_level_mesh[i].is_valid()) {
			VisualServer::get_singleton()->free(selection_level_mesh[i]);
		}
	}

	if (selection_mesh.is_valid()) {
		VisualServer::get_singleton()->free(selection_mesh);
	}
	if (selection_instance.is_valid()) {
		VisualServer::get_singleton()->free(selection_instance);
	}
	if (paste_mesh.is_valid()) {
		VisualServer::get_singleton()->free(paste_mesh);
	}
	if (paste_instance.is_valid()) {
		VisualServer::get_singleton()->free(paste_instance);
	}
}

void GridMapEditorPlugin::_notification(int p_what) {
	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		switch ((int)EditorSettings::get_singleton()->get("editors/grid_map/editor_side")) {
			case 0: { // Left.
				SpatialEditor::get_singleton()->move_control_to_left_panel(grid_map_editor);
			} break;
			case 1: { // Right.
				SpatialEditor::get_singleton()->move_control_to_right_panel(grid_map_editor);
			} break;
		}
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
		grid_map_editor->edit(nullptr);
		grid_map_editor->set_process(false);
	}
}

GridMapEditorPlugin::GridMapEditorPlugin(EditorNode *p_node) {
	editor = p_node;

	EDITOR_DEF("editors/grid_map/editor_side", 1);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "editors/grid_map/editor_side", PROPERTY_HINT_ENUM, "Left,Right"));

	grid_map_editor = memnew(GridMapEditor(editor));
	switch ((int)EditorSettings::get_singleton()->get("editors/grid_map/editor_side")) {
		case 0: { // Left.
			SpatialEditor::get_singleton()->add_control_to_left_panel(grid_map_editor);
		} break;
		case 1: { // Right.
			SpatialEditor::get_singleton()->add_control_to_right_panel(grid_map_editor);
		} break;
	}
	grid_map_editor->hide();
}

GridMapEditorPlugin::~GridMapEditorPlugin() {
}
