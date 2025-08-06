/**************************************************************************/
/*  scene_paint_editor_plugin.cpp                                         */
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

#include "scene_paint_editor_plugin.h"

#include "core/os/keyboard.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/gui/editor_zoom_widget.h"
#include "editor/inspector/editor_resource_preview.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/tile_map_layer.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/main/window.h"

void ScenePaintEditor::_draw() {
	if (!node || !grid || !is_tool_selected) {
		return;
	}

	_draw_grid();
	_draw_grid_highlight();
}

void ScenePaintEditor::_draw_grid() {
	Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
	CanvasItemEditor *canvas = CanvasItemEditor::get_singleton();
	Transform2D xform = canvas->get_canvas_transform().affine_inverse();
	Size2 size = viewport->get_size();

	int step_x = palette->get_grid_step().x;
	int step_y = palette->get_grid_step().y;

	Vector2 offset = Vector2();
	if (use_local_grid) {
		offset = node->get_global_position();
	}

	int last_cell = 0;

	if (step_x > 0) {
		for (int i = 0; i < size.width; i++) {
			int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(i, 0)).x - offset.x) / step_x));

			if (i == 0) {
				last_cell = cell;
			}
			if (last_cell != cell) {
				Vector2 from = node->to_local(xform.xform(Vector2(i, 0)));
				Vector2 to = node->to_local(xform.xform(Vector2(i, size.height)));

				node->draw_line(from, to, grid_color);
			}
			last_cell = cell;
		}
	}

	if (step_y > 0) {
		for (int i = 0; i < size.height; i++) {
			int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(0, i)).y - offset.y) / step_y));

			if (i == 0) {
				last_cell = cell;
			}
			if (last_cell != cell) {
				Vector2 from = node->to_local(xform.xform(Vector2(0, i)));
				Vector2 to = node->to_local(xform.xform(Vector2(size.width, i)));

				node->draw_line(from, to, grid_color);
			}
			last_cell = cell;
		}
	}
}

void ScenePaintEditor::_draw_grid_highlight() {
	Color fill_color = !is_erasing ? Color(1, 1, 1, 0.3) : Color(0, 0, 0, 0.3);
	node->draw_rect(paint_rect, fill_color);
}

void ScenePaintEditor::_update_draw() {
	if (!node) {
		return;
	}
	node->queue_redraw();
}

void ScenePaintEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !is_tool_selected) {
		return;
	}

	CanvasItemEditor *canvas = CanvasItemEditor::get_singleton();
	Transform2D canvas_xform = canvas->get_canvas_transform();

	use_local_grid = Input::get_singleton()->is_key_pressed(Key::SHIFT);

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		is_painting = mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT;
		is_erasing = mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT;
		if (is_painting && !is_erasing) {
			_add_node_at_position(mb->get_position(), canvas_xform);
			accept_event();
		}
		if (is_erasing && !is_painting) {
			_remove_node_at_position();
			accept_event();
		}
		if (!is_painting || !is_erasing) {
			last_paint_pos = Vector2();
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Vector2 cell_pos = _get_mouse_grid_cell();
		if (grid) {
			cell_pos = !snap_grid ? cell_pos : (cell_pos - palette->get_grid_step() / 2.0);
		}
		paint_rect = Rect2(cell_pos, palette->get_grid_step());

		if (is_painting && !is_erasing) {
			_add_node_at_position(mm->get_position(), canvas_xform);
			accept_event();
		}
		if (is_erasing && !is_painting) {
			_remove_node_at_position();
			accept_event();
		}
	}
	_update_draw();
}

void ScenePaintEditor::_add_node_at_position(const Vector2 &mouse_pos, const Transform2D &canvas_xform) {
	if (selected_scene.is_null() || !node) {
		return;
	}

	Vector2 cell_pos = _get_mouse_grid_cell();
	Vector2 pos = canvas_xform.affine_inverse().xform(mouse_pos);

	if (grid) {
		pos = snap_grid ? cell_pos : (cell_pos + palette->get_grid_step() / 2.0);
	} else {
		pos.x = Math::round(pos.x - node->get_global_position().x);
		pos.y = Math::round(pos.y - node->get_global_position().y);
	}

	if (last_paint_pos == pos) {
		return;
	}
	last_paint_pos = pos;

	Node2D *node_2d = Object::cast_to<Node2D>(selected_scene->instantiate());
	if (!node_2d) {
		return;
	}

	Node *parent = EditorNode::get_singleton()->get_edited_scene();
	if (parent) {
		node->add_child(node_2d, true);
		node_2d->set_owner(parent);
		node_2d->set_meta("_scene_painted", true);
		node_2d->set_position(pos);
	}
}

void ScenePaintEditor::_remove_node_at_position() {
	if (!node) {
		return;
	}

	for (int i = 0; i < node->get_child_count(); i++) {
		Node2D *node_2d = Object::cast_to<Node2D>(node->get_child(i));
		if (!node_2d || !node_2d->has_meta("_scene_painted")) {
			continue;
		}

		if (paint_rect.has_point(node_2d->get_position())) {
			node_2d->queue_free();
		}
	}
}

Vector2 ScenePaintEditor::_get_mouse_grid_cell() {
	if (!node) {
		return Vector2();
	}

	CanvasItemEditor *canvas = CanvasItemEditor::get_singleton();
	Transform2D xform = canvas->get_canvas_transform().affine_inverse();

	Vector2 mouse_canvas = canvas->get_viewport_control()->get_local_mouse_position();

	Vector2 mouse_world = xform.xform(mouse_canvas);

	if (use_local_grid) {
		mouse_world -= node->get_global_position();
	}

	int step_x = palette->get_grid_step().x;
	int step_y = palette->get_grid_step().y;

	int cell_x = Math::floor(mouse_world.x / step_x);
	int cell_y = Math::floor(mouse_world.y / step_y);

	Vector2 cell_pos = Vector2(cell_x * step_x, cell_y * step_y);

	if (!use_local_grid) {
		cell_pos -= node->get_global_position();
	}

	return cell_pos;
}

void ScenePaintEditor::_add_folder() {
	if (!folders->get_root()) {
		TreeItem *root = folders->create_item();
	}
	TreeItem *item = folders->create_item(folders->get_root());
	String name = "new_folder";
	int counter = 0;
	while (palette->has_folder(name)) {
		counter++;
		name = vformat("new_folder_%d", counter);
	}
	item->set_text(0, name);
	item->set_metadata(0, name);
	item->set_editable(0, true);
	palette->add_folder(name);
}

void ScenePaintEditor::_edit_folder() {
	TreeItem *edited = folders->get_edited();
	if (!edited) {
		return;
	}

	String new_name = edited->get_text(0);

	if (new_name == String(edited_folder)) {
		return;
	}

	if (new_name.is_empty()) {
		new_name = "new_folder";
	}

	new_name = new_name.replace_char('/', '_').replace_char(',', ' ');
	String name = new_name;
	int counter = 0;
	while (palette->has_folder(name)) {
		if (name == String(edited_folder)) {
			edited->set_text(0, name); // The name didn't change, just updated the column text to name.
			return;
		}
		counter++;
		name = new_name + "_" + itos(counter);
	}
	edited->set_text(0, name);
	palette->edit_folder(edited_folder, name);
}

void ScenePaintEditor::_remove_folder() {
	TreeItem *selected = folders->get_selected();
	if (!selected) {
		return;
	}
	String folder_name = selected->get_text(0);
	if (palette->has_folder(folder_name)) {
		palette->remove_folder(folder_name);
		if (edited_folder == folder_name) {
			edited_folder = "";
			_update_scene_list();
			_update_toolbar_buttons();
		}
		memdelete(selected);
	}
}

void ScenePaintEditor::_add_scene() {
	file->clear_filters();
	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	for (const String &extension : Vector<String>{ "tscn", "scn" }) {
		file->add_filter("*." + extension, extension.to_upper());
	}
	file->connect("file_selected", callable_mp(this, &ScenePaintEditor::_add_scene_request), CONNECT_ONE_SHOT);
	file->popup_file_dialog();
}

void ScenePaintEditor::_open_scene() {
	if (selected_scene_index < 0 || selected_scene_index >= item_list->get_item_count()) {
		return;
	}
	String scene_path = palette->get_scene(edited_folder, selected_scene_index).scene_path;
	EditorNode::get_singleton()->load_scene(scene_path);
}

void ScenePaintEditor::_remove_scene() {
	if (selected_scene_index < 0 || selected_scene_index >= item_list->get_item_count()) {
		return;
	}
	palette->remove_scene(edited_folder, selected_scene_index);
	_update_scene_list();
	_update_toolbar_buttons();
}

void ScenePaintEditor::_grid_toggle_toggled(bool p_toggled) {
	grid = p_toggled;
	_update_draw();
}

void ScenePaintEditor::_snap_grid_toggled(bool p_toggled) {
	snap_grid = p_toggled;
	_update_draw();
}

void ScenePaintEditor::_update_grid_step() {
	Size2i grid_step = palette->get_grid_step();
	if (grid_step.x != grid_step_x->get_value()) {
		grid_step.x = grid_step_x->get_value();
	}
	if (grid_step.y != grid_step_y->get_value()) {
		grid_step.y = grid_step_y->get_value();
	}
	palette->set_grid_step(grid_step);
	_update_draw();
}

void ScenePaintEditor::_folder_selected() {
	TreeItem *selected = folders->get_selected();
	if (selected) {
		edited_folder = selected->get_text(0);
	}
	_update_scene_list();
	_update_toolbar_buttons();
}

void ScenePaintEditor::_folder_deselected() {
	edited_folder = "";
	folders->deselect_all();
	_update_scene_list();
	_update_toolbar_buttons();
}

void ScenePaintEditor::_scene_deselected() {
	selected_scene_index = -1;
	selected_scene.unref();
	item_list->deselect_all();
	_update_toolbar_buttons();
}

void ScenePaintEditor::_scene_selected(const int p_index) {
	selected_scene_index = p_index;
	selected_scene = ResourceLoader::load(palette->get_scene(edited_folder, selected_scene_index).scene_path);
	_update_toolbar_buttons();
}

void ScenePaintEditor::_add_scene_request(const String &p_path) {
	Ref<PackedScene> scene = ResourceLoader::load(p_path);
	if (scene.is_valid()) {
		// Check if it extends CanvasItem.
		Ref<SceneState> scene_state = scene->get_state();
		String type;
		while (scene_state.is_valid() && type.is_empty()) {
			// Make sure we have a root node. Supposed to be at 0 index because find_node_by_path() does not seem to work.
			ERR_FAIL_COND(scene_state->get_node_count() < 1);

			type = scene_state->get_node_type(0);
			scene_state = scene_state->get_base_scene_state();
		}
		ERR_FAIL_COND_EDMSG(type.is_empty(), vformat("Invalid PackedScene for ScenePaint: %s. Could not get the type of the root node.", scene->get_path()));
		bool extends_correct_class = ClassDB::is_parent_class(type, "Node2D");
		ERR_FAIL_COND_EDMSG(!extends_correct_class, vformat("Invalid PackedScene for ScenePaint: %s. Root node should extend Node2D. Found %s instead.", scene->get_path(), type));
		String scene_path = scene->get_path();
		String display_name = scene_path.get_file().get_basename();
		palette->add_scene(edited_folder, scene_path, display_name);
		_update_scene_list();
		_update_toolbar_buttons();
	}
}

void ScenePaintEditor::_update_scene_list() {
	selected_scene_index = -1;
	item_list->clear();
	const Vector<ScenePalette::SceneData> &scenes = palette->get_folder(edited_folder);

	for (const ScenePalette::SceneData &E : scenes) {
		int udata = item_list->add_item(E.display_name);
		Ref<PackedScene> scene = ResourceLoader::load(E.scene_path);
		EditorResourcePreview::get_singleton()->queue_edited_resource_preview(scene, this, "_scene_thumbnail_done", udata);
	}
}

void ScenePaintEditor::_scene_thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, const Variant &p_ud) {
	int index = p_ud;

	if (index >= 0 && index < item_list->get_item_count()) {
		item_list->set_item_icon(index, p_preview);
	}
}

void ScenePaintEditor::_update_toolbar_buttons() {
	bool disabled;
	disabled = edited_folder.is_empty();
	remove_folder_button->set_disabled(disabled);
	disabled = selected_scene_index < 0 || selected_scene_index >= item_list->get_item_count();
	open_scene_button->set_disabled(disabled);
	remove_scene_button->set_disabled(disabled);
}

bool ScenePaintEditor::_can_drop_data_fw(const Point2 &p_point, const Variant &p_data) const {
	if (p_data.get_type() != Variant::DICTIONARY) {
		return false;
	}

	Dictionary d = p_data;
	if (!d.has("type") || !d.has("files")) {
		return false;
	}

	if (String(d["type"]) != "files") {
		return false;
	}

	Array files = d["files"];
	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		String ext = path.get_extension().to_lower();
		if (ext != "tscn" && ext != "scn") {
			return false;
		}
	}

	return true;
}

void ScenePaintEditor::_drop_data_fw(const Point2 &p_point, const Variant &p_data) {
	Dictionary d = p_data;
	Array files = d["files"];

	for (int i = 0; i < files.size(); i++) {
		String path = files[i];
		_add_scene_request(path);
	}
}

void ScenePaintEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			new_palette_button->set_button_icon(get_editor_theme_icon(SNAME("New")));
			load_palette_button->set_button_icon(get_editor_theme_icon(SNAME("Load")));
			save_palette_button->set_button_icon(get_editor_theme_icon(SNAME("Save")));
			add_folder_button->set_button_icon(get_editor_theme_icon(SNAME("FolderCreate")));
			remove_folder_button->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
			add_scene_button->set_button_icon(get_editor_theme_icon(SNAME("Add")));
			open_scene_button->set_button_icon(get_editor_theme_icon(SNAME("PlayScene")));
			remove_scene_button->set_button_icon(get_editor_theme_icon(SNAME("Remove")));
			grid_toggle_button->set_button_icon(get_editor_theme_icon(SNAME("GridToggle")));
			snap_grid_button->set_button_icon(get_editor_theme_icon(SNAME("SnapGrid")));
			_update_toolbar_buttons();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			is_painting = false;
			is_erasing = false;
		} break;
	}
}

void ScenePaintEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_scene_thumbnail_done"), &ScenePaintEditor::_scene_thumbnail_done);
}

void ScenePaintEditor::set_grid_step(const Size2i p_size) {
	palette->set_grid_step(p_size);
	grid_step_x->set_value(p_size.x);
	grid_step_y->set_value(p_size.y);
}

Size2i ScenePaintEditor::get_grid_step() const {
	return palette->get_grid_step();
}

ScenePaintEditor::ScenePaintEditor() {
	palette.instantiate();
	palette->add_folder("");

	toolbar = memnew(HBoxContainer);
	toolbar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(toolbar);

	new_palette_button = memnew(Button);
	new_palette_button->set_accessibility_name(TTRC("New Palette"));
	new_palette_button->set_theme_type_variation(SceneStringName(FlatButton));
	new_palette_button->set_tooltip_text(TTRC("Create a new scene paint palette."));
	toolbar->add_child(new_palette_button);

	load_palette_button = memnew(Button);
	load_palette_button->set_accessibility_name(TTRC("Load Palette"));
	load_palette_button->set_theme_type_variation(SceneStringName(FlatButton));
	load_palette_button->set_tooltip_text(TTRC("Load a scene paint palette from file."));
	toolbar->add_child(load_palette_button);

	save_palette_button = memnew(Button);
	save_palette_button->set_accessibility_name(TTRC("Save Palette"));
	save_palette_button->set_theme_type_variation(SceneStringName(FlatButton));
	save_palette_button->set_tooltip_text(TTRC("Save the current scene paint palette to file."));
	toolbar->add_child(save_palette_button);

	VSeparator *v_separator = memnew(VSeparator);
	toolbar->add_child(v_separator);

	add_folder_button = memnew(Button);
	add_folder_button->set_accessibility_name(TTRC("Add Folder"));
	add_folder_button->set_theme_type_variation(SceneStringName(FlatButton));
	add_folder_button->set_tooltip_text(TTRC("Add a folder to the current palette."));
	add_folder_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_add_folder));
	toolbar->add_child(add_folder_button);

	remove_folder_button = memnew(Button);
	remove_folder_button->set_accessibility_name(TTRC("Remove Folder"));
	remove_folder_button->set_theme_type_variation(SceneStringName(FlatButton));
	remove_folder_button->set_tooltip_text(TTRC("Remove a folder from the current palette."));
	remove_folder_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_remove_folder));
	toolbar->add_child(remove_folder_button);

	v_separator = memnew(VSeparator);
	toolbar->add_child(v_separator);

	add_scene_button = memnew(Button);
	add_scene_button->set_accessibility_name(TTRC("Add Scene"));
	add_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	add_scene_button->set_tooltip_text(TTRC("Add a scene to the current palette."));
	add_scene_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_add_scene));
	toolbar->add_child(add_scene_button);

	open_scene_button = memnew(Button);
	open_scene_button->set_accessibility_name(TTRC("Open Scene"));
	open_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	open_scene_button->set_tooltip_text(TTRC("Open a scene from the current palette."));
	open_scene_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_open_scene));
	toolbar->add_child(open_scene_button);

	remove_scene_button = memnew(Button);
	remove_scene_button->set_accessibility_name(TTRC("Remove Scene"));
	remove_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	remove_scene_button->set_tooltip_text(TTRC("Remove a scene from the current palette."));
	remove_scene_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_remove_scene));
	toolbar->add_child(remove_scene_button);

	v_separator = memnew(VSeparator);
	toolbar->add_child(v_separator);

	grid_toggle_button = memnew(Button);
	grid_toggle_button->set_toggle_mode(true);
	grid_toggle_button->set_accessibility_name(TTRC("Grid"));
	grid_toggle_button->set_theme_type_variation(SceneStringName(FlatButton));
	grid_toggle_button->set_tooltip_text(TTRC("Toggle grid visibility."));
	grid_toggle_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_grid_toggle_toggled));
	toolbar->add_child(grid_toggle_button);

	snap_grid_button = memnew(Button);
	snap_grid_button->set_toggle_mode(true);
	snap_grid_button->set_accessibility_name(TTRC("Snap Grid"));
	snap_grid_button->set_theme_type_variation(SceneStringName(FlatButton));
	snap_grid_button->set_tooltip_text(TTRC("Toggle grid snapping for scene placement."));
	snap_grid_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_snap_grid_toggled));
	toolbar->add_child(snap_grid_button);

	Label *grid_label = memnew(Label);
	grid_label->set_text(TTRC("Grid Step:"));
	toolbar->add_child(grid_label);

	grid_step_x = memnew(SpinBox);
	grid_step_x->set_min(1);
	grid_step_x->set_step(1);
	grid_step_x->set_value(palette->get_grid_step().x);
	grid_step_x->set_suffix("px");
	grid_step_x->set_allow_greater(true);
	grid_step_x->set_select_all_on_focus(true);
	grid_step_x->set_custom_minimum_size(Vector2(64, 0) * EDSCALE);
	grid_step_x->connect("value_changed", callable_mp(this, &ScenePaintEditor::_update_grid_step).unbind(1));
	toolbar->add_child(grid_step_x);

	grid_step_y = memnew(SpinBox);
	grid_step_y->set_min(1);
	grid_step_y->set_step(1);
	grid_step_y->set_value(palette->get_grid_step().y);
	grid_step_y->set_suffix("px");
	grid_step_y->set_allow_greater(true);
	grid_step_y->set_select_all_on_focus(true);
	grid_step_y->set_custom_minimum_size(Vector2(64, 0) * EDSCALE);
	grid_step_y->connect("value_changed", callable_mp(this, &ScenePaintEditor::_update_grid_step).unbind(1));
	toolbar->add_child(grid_step_y);

	HSplitContainer *split_container = memnew(HSplitContainer);
	split_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	split_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(split_container);

	folders = memnew(Tree);
	folders->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	folders->set_custom_minimum_size(Size2(200, 0) * EDSCALE);
	folders->set_theme_type_variation("TreeSecondary");
	folders->set_allow_reselect(true);
	folders->set_hide_root(true);
	folders->connect("empty_clicked", callable_mp(this, &ScenePaintEditor::_folder_deselected).unbind(2));
	folders->connect("cell_selected", callable_mp(this, &ScenePaintEditor::_folder_selected), CONNECT_DEFERRED);
	folders->connect("item_edited", callable_mp(this, &ScenePaintEditor::_edit_folder));
	split_container->add_child(folders);

	item_list = memnew(ItemList);
	item_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	item_list->set_icon_mode(ItemList::ICON_MODE_TOP);
	item_list->set_fixed_icon_size(Size2(64, 64) * EDSCALE);
	item_list->set_fixed_column_width(96 * EDSCALE);
	item_list->set_max_text_lines(2);
	item_list->set_max_columns(0);
	item_list->connect("item_activated", callable_mp(this, &ScenePaintEditor::_open_scene).unbind(1));
	item_list->connect("empty_clicked", callable_mp(this, &ScenePaintEditor::_scene_deselected).unbind(2));
	item_list->connect("item_selected", callable_mp(this, &ScenePaintEditor::_scene_selected));
	item_list->set_drag_forwarding(Callable(),
			callable_mp(this, &ScenePaintEditor::_can_drop_data_fw),
			callable_mp(this, &ScenePaintEditor::_drop_data_fw));
	split_container->add_child(item_list);

	file = memnew(EditorFileDialog);
	add_child(file);
}

ScenePaintEditor::~ScenePaintEditor() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
}

void ScenePaintEditorPlugin::_canvas_item_tool_changed(int p_tool) {
	scene_paint_editor->is_tool_selected = (CanvasItemEditor::Tool)p_tool == CanvasItemEditor::TOOL_SCENE_PAINT;
	make_visible(true);
}

void ScenePaintEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			CanvasItemEditor::get_singleton()->connect("canvas_item_tool_changed", callable_mp(this, &ScenePaintEditorPlugin::_canvas_item_tool_changed));
			scene_paint_editor = memnew(ScenePaintEditor);
			scene_paint_editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			scene_paint_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			scene_paint_editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
			scene_paint_editor->hide();

			scene_paint_editor->viewport = CanvasItemEditor::get_singleton()->get_viewport_control();
			scene_paint_editor->viewport->connect("draw", callable_mp(scene_paint_editor, &ScenePaintEditor::_update_draw));
			scene_paint_editor->viewport->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_gui_input_viewport));

			panel_button = EditorNode::get_bottom_panel()->add_item(TTRC("ScenePaint"), scene_paint_editor, ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_scene_paint_bottom_panel", TTRC("Toggle ScenePaint Bottom Panel")));
			panel_button->hide();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			CanvasItemEditor::get_singleton()->disconnect("canvas_item_tool_changed", callable_mp(this, &ScenePaintEditorPlugin::_canvas_item_tool_changed));
			EditorNode::get_bottom_panel()->remove_item(scene_paint_editor);
			memdelete_notnull(scene_paint_editor);
			scene_paint_editor = nullptr;
			panel_button = nullptr;
		} break;
	}
}

void ScenePaintEditorPlugin::_bind_methods() {
}

void ScenePaintEditorPlugin::edit(Object *p_object) {
	if (scene_paint_editor->node) {
		scene_paint_editor->node->disconnect("draw", callable_mp(scene_paint_editor, &ScenePaintEditor::_draw));
		scene_paint_editor->node->queue_redraw();
	}

	scene_paint_editor->node = Object::cast_to<Node2D>(p_object);
	if (scene_paint_editor->node) {
		scene_paint_editor->node->connect("draw", callable_mp(scene_paint_editor, &ScenePaintEditor::_draw));
		scene_paint_editor->node->queue_redraw();
	}

	TileMapLayer *edited_layer = Object::cast_to<TileMapLayer>(p_object);
	if (edited_layer) {
		Ref<TileSet> tile_set = edited_layer->get_tile_set();
		if (tile_set.is_valid()) {
			scene_paint_editor->set_grid_step(tile_set->get_tile_size());
		}
		call_deferred("make_visible", true);
	}
}

bool ScenePaintEditorPlugin::handles(Object *p_object) const {
	is_node_2d = p_object->is_class("Node2D");
	return is_node_2d;
}

void ScenePaintEditorPlugin::make_visible(bool p_visible) {
	ERR_FAIL_NULL(scene_paint_editor);
	if (p_visible && is_node_2d && scene_paint_editor->is_tool_selected && scene_paint_editor->node) {
		panel_button->show();
		EditorNode::get_bottom_panel()->make_item_visible(scene_paint_editor);
		scene_paint_editor->set_process(true);
	} else {
		panel_button->hide();
		if (scene_paint_editor->is_visible_in_tree()) {
			EditorNode::get_bottom_panel()->hide_bottom_panel();
		}
		scene_paint_editor->set_process(false);
	}
}

void ScenePalette::set_grid_step(const Size2i &p_size) {
	_grid_step = p_size;
	emit_changed();
}

void ScenePalette::add_folder(const StringName &p_name) {
	ERR_FAIL_COND_MSG(_data.has(p_name), vformat("Folder with name '%s' already exists in the palette.", p_name));
	_data[p_name] = Vector<SceneData>();
	emit_changed();
}

void ScenePalette::edit_folder(const StringName &p_name, const String &p_new_name) {
	ERR_FAIL_COND_MSG(!_data.has(p_name), vformat("Folder with name '%s' does not exist in the palette.", p_name));
	_data[p_new_name] = _data[p_name];
	_data.erase(p_name);
	emit_changed();
}

void ScenePalette::remove_folder(const StringName &p_name) {
	ERR_FAIL_COND_MSG(!_data.has(p_name), vformat("Folder with name '%s' does not exist in the palette.", p_name));
	_data.erase(p_name);
	emit_changed();
}

Vector<ScenePalette::SceneData> ScenePalette::get_folder(const StringName &p_name) {
	return _data[p_name];
}

void ScenePalette::add_scene(const StringName &p_folder, const String &p_scene_path, const String &p_display_name) {
	ERR_FAIL_COND_EDMSG(has_scene(p_folder, p_scene_path), vformat("Scene '%s' already exists in folder '%s'.", p_display_name, p_folder));
	_data[p_folder].push_back(SceneData(p_scene_path, p_display_name));
	emit_changed();
}

bool ScenePalette::has_scene(const StringName &p_folder, const String p_scene_path) {
	if (!_data.has(p_folder)) {
		return false;
	}

	const Vector<SceneData> &scenes = _data[p_folder];
	for (int i = 0; i < scenes.size(); i++) {
		if (scenes[i].scene_path == p_scene_path) {
			return true;
		}
	}

	return false;
}

ScenePalette::SceneData ScenePalette::get_scene(const StringName &p_folder, const int p_index) const {
	ERR_FAIL_COND_V_MSG(!_data.has(p_folder), SceneData(), vformat("Folder '%s' does not exist in the palette.", p_folder));
	const Vector<SceneData> &scenes = _data[p_folder];
	ERR_FAIL_INDEX_V_MSG(p_index, scenes.size(), SceneData(), vformat("Scene index %d is out of range for folder '%s'.", p_index, p_folder));
	return scenes[p_index];
}

void ScenePalette::remove_scene(const StringName &p_folder, const int p_index) {
	ERR_FAIL_COND_MSG(!_data.has(p_folder), vformat("Folder '%s' does not exist in the palette.", p_folder));

	Vector<SceneData> &scenes = _data[p_folder];
	ERR_FAIL_INDEX_MSG(p_index, scenes.size(), vformat("Scene index %d is out of range for folder '%s'.", p_index, p_folder));

	scenes.remove_at(p_index);

	emit_changed();
}
