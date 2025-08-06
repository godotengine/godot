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

#include "core/config/project_settings.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/docks/inspector_dock.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/tile_map_layer.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"

void ScenePaintEditor::_edit(Object *p_object) {
	TileMapLayer *edited_layer = Object::cast_to<TileMapLayer>(p_object);
	if (edited_layer) {
		Ref<TileSet> tile_set = edited_layer->get_tile_set();
		if (tile_set.is_valid()) {
			set_grid_step(tile_set->get_tile_size());
		}
	}

	cache_node = Object::cast_to<Node2D>(p_object);
	if (node && (pinned || scene_picker)) {
		if (scene_picker) {
			SceneTreeDock::get_singleton()->set_selection(Vector<Node *>{ node });
		}
		return;
	}

	_update_node();
}

void ScenePaintEditor::_draw() {
	if (!is_visible_in_tree() || !is_tool_selected || !node || !grid) {
		return;
	}

	_draw_grid();
	_draw_grid_highlight();
}

void ScenePaintEditor::_draw_grid() {
	Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
	Transform2D xform = canvas->get_canvas_transform().affine_inverse();
	Size2 size = viewport->get_size();
	Size2i step = get_grid_step();

	if (step.x <= 0 || step.y <= 0) {
		return;
	}

	Vector2 corners_world[4] = {
		xform.xform(Vector2(0, 0)),
		xform.xform(Vector2(size.width, 0)),
		xform.xform(Vector2(size.width, size.height)),
		xform.xform(Vector2(0, size.height)),
	};

	Vector2 corners_local[4];
	for (int i = 0; i < 4; i++) {
		corners_local[i] = node->to_local(corners_world[i]);
	}

	Rect2 bounds = Rect2(corners_local[0], Vector2());
	for (int i = 1; i < 4; i++) {
		bounds.expand_to(corners_local[i]);
	}

	int start_x = Math::floor(bounds.position.x / step.x) * step.x;
	int end_x = Math::ceil((bounds.position.x + bounds.size.x) / step.x) * step.x;
	for (int x = start_x; x <= end_x; x += step.x) {
		Vector2 from(x, bounds.position.y);
		Vector2 to(x, bounds.position.y + bounds.size.y);
		node->draw_line(from, to, grid_color);
	}

	int start_y = Math::floor(bounds.position.y / step.y) * step.y;
	int end_y = Math::ceil((bounds.position.y + bounds.size.y) / step.y) * step.y;
	for (int y = start_y; y <= end_y; y += step.y) {
		Vector2 from(bounds.position.x, y);
		Vector2 to(bounds.position.x + bounds.size.x, y);
		node->draw_line(from, to, grid_color);
	}
}

void ScenePaintEditor::_draw_grid_highlight() {
	if (preview && !is_erasing) {
		return;
	}
	Color fill_color = !is_erasing ? Color(1, 1, 1, 0.3) : Color(0, 0, 0, 0.3);
	node->draw_rect(paint_rect, fill_color);
}

void ScenePaintEditor::_update_node() {
	if (node) {
		node->disconnect("draw", callable_mp(this, &ScenePaintEditor::_draw));
		node->queue_redraw();
	}

	node = cache_node;
	if (node) {
		node->connect("draw", callable_mp(this, &ScenePaintEditor::_draw));
	}
	_update_draw();
	callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
}

void ScenePaintEditor::_update_draw() {
	if (!node) {
		return;
	}
	node->queue_redraw();
}

void ScenePaintEditor::_update_preview() {
	if (!is_visible_in_tree() || !is_tool_selected || !selected_scene || !node || is_erasing) {
		if (preview) {
			preview->queue_free();
			preview = nullptr;
			instance = nullptr;
		}
		return;
	}
	if (node) {
		if (edit_properties) {
			InspectorDock::get_inspector_singleton()->edit(node);
			InspectorDock::get_singleton()->update(node);
		}
		if (instance) {
			instance->queue_free();
			instance = nullptr;
		}
		if (preview && preview->get_parent() != node) {
			preview->reparent(node);
		}
		if (!preview) {
			preview = memnew(Node2D);
			preview->set_modulate(Color(1, 1, 1, 0.5));
			node->add_child(preview);
		}
	}
	if (preview && !scene_picker) {
		if (selected_scene && !instance) {
			instance = Object::cast_to<Node2D>(selected_scene->duplicate());
			preview->add_child(instance, true);
			preview->set_transform(Transform2D());
			instance->set_transform(Transform2D());
		}
		_update_preview_position();
	}
	_edit_properties();
}

void ScenePaintEditor::_update_paint_rect() {
	if (!node) {
		return;
	}
	Vector2 cell_pos = _get_mouse_grid_cell() - node->get_global_position();
	Vector2 pos = !snap_grid ? cell_pos : (cell_pos - get_grid_step() / 2.0);

	paint_rect = Rect2(pos, get_grid_step());
}

void ScenePaintEditor::_update_preview_position() {
	if (!preview || !node) {
		return;
	}
	Vector2 cell_pos = _get_mouse_grid_cell() - node->get_global_position();
	Vector2 pos = canvas_xform.affine_inverse().xform(mouse_pos);

	if (grid || snap_grid) {
		pos = snap_grid ? cell_pos : (cell_pos + get_grid_step() / 2.0);
		preview->set_position(pos);
	} else {
		preview->set_global_position(pos);
	}
}

void ScenePaintEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !is_tool_selected) {
		return;
	}

	canvas_xform = canvas->get_canvas_transform();

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		mouse_pos = mb->get_position();

		if (scene_picker) {
			if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
				callable_mp(this, &ScenePaintEditor::_update_scene_picker).call_deferred(PICK_CANVAS_ITEM);
			}
		} else {
			if (mb->is_pressed()) {
				if (mb->get_button_index() == MouseButton::LEFT) {
					if (!node) {
						EditorNode::get_singleton()->show_warning(
								"No target node selected. Please select a Node2D (or compatible node) in the SceneTree where painted scenes will be added.");
						return;
					} else if (!selected_scene) {
						EditorNode::get_singleton()->show_warning(
								"No scene selected for painting. Use the Scene Picker tool (from the FileSystem or SceneTree) to choose a scene before painting.");
						return;
					}

					is_painting = true;
					undo_redo->create_action("Paint Node(s)");
					_add_node_at_pos();
					accept_event();
				} else if (mb->get_button_index() == MouseButton::RIGHT) {
					is_erasing = true;
					undo_redo->create_action("Erase Node(s)");
					_remove_node_at_pos();
					accept_event();
				}
			} else if (mb->is_released()) {
				if (mb->get_button_index() == MouseButton::LEFT) {
					is_painting = false;
					undo_redo->commit_action(false);
				} else if (mb->get_button_index() == MouseButton::RIGHT) {
					is_erasing = false;
					undo_redo->commit_action(false);
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		mouse_pos = mm->get_position();

		if (!preview) {
			callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
		}
		_update_paint_rect();
		_update_preview_position();

		if (!scene_picker) {
			if (is_painting) {
				_add_node_at_pos();
				accept_event();
			} else if (is_erasing) {
				if (preview) {
					callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
				}
				_remove_node_at_pos();
				accept_event();
			}
		}
	}
	_update_draw();
}

void ScenePaintEditor::_add_node_at_pos() {
	if (!node || !instance) {
		return;
	}

	Vector2 cell_pos = _get_mouse_grid_cell() - node->get_global_position();
	Vector2 pos = canvas_xform.affine_inverse().xform(mouse_pos);

	if (grid || snap_grid) {
		pos = snap_grid ? cell_pos : (cell_pos + get_grid_step() / 2.0);
		List<CanvasItem *> results;
		canvas->find_canvas_items_in_rect(paint_rect, node, &results);
		for (int i = 0; i < results.size(); i++) {
			Node2D *node_2d = Object::cast_to<Node2D>(results.get(i));
			if (!node_2d || !node_2d->has_meta("_scene_painted") || node_2d->get_parent() != node) {
				continue;
			}
			return;
		}
	} else {
		Vector<CanvasItemEditor::_SelectResult> results;
		canvas->find_canvas_items_at_pos(pos, node, results);
		for (int i = 0; i < results.size(); i++) {
			Node2D *node_2d = Object::cast_to<Node2D>(results[i].item);
			if (!node_2d || !node_2d->has_meta("_scene_painted") || node_2d->get_parent() != node) {
				continue;
			}
			return;
		}
	}

	Node2D *node_2d = Object::cast_to<Node2D>(instance->duplicate());
	if (!node_2d) {
		return;
	}

	Node *parent = EditorNode::get_singleton()->get_edited_scene();
	if (parent) {
		node->add_child(node_2d, true);
		node_2d->set_owner(parent);
		node_2d->set_meta("_scene_painted", true);
		node_2d->set_global_position(pos);

		undo_redo->add_do_reference(node_2d);
		undo_redo->add_do_method(node, "add_child", node_2d);
		undo_redo->add_do_method(node_2d, "set_owner", parent);
		undo_redo->add_do_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_do_method(node_2d, "set_global_position", pos);
		undo_redo->add_undo_method(node, "remove_child", node_2d);
	}
}

void ScenePaintEditor::_remove_node_at_pos() {
	if (!node) {
		return;
	}

	Vector2 mouse_canvas = viewport->get_local_mouse_position();
	Vector2 mouse_world = canvas->get_canvas_transform().affine_inverse().xform(mouse_canvas);
	List<CanvasItem *> selection;

	if (grid || snap_grid) {
		canvas->find_canvas_items_in_rect(paint_rect, node, &selection);
	} else {
		Vector<CanvasItemEditor::_SelectResult> results;
		canvas->find_canvas_items_at_pos(mouse_world, node, results);
		for (int i = 0; i < results.size(); i++) {
			selection.push_back(results[i].item);
		}
	}

	for (int i = 0; i < selection.size(); i++) {
		Node2D *node_2d = Object::cast_to<Node2D>(selection.get(i));
		if (!node_2d || !node_2d->has_meta("_scene_painted") || node_2d->get_parent() != node) {
			continue;
		}
		Node *parent = EditorNode::get_singleton()->get_edited_scene();
		node->remove_child(node_2d);

		undo_redo->add_do_method(node, "remove_child", node_2d);
		undo_redo->add_undo_method(node, "add_child", node_2d);
		undo_redo->add_undo_reference(node_2d);
		undo_redo->add_undo_method(node_2d, "set_owner", parent);
		undo_redo->add_undo_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_undo_method(node_2d, "set_global_position", node_2d->get_global_position());
	}
}

Vector2 ScenePaintEditor::_get_mouse_grid_cell() {
	if (!node) {
		return Vector2();
	}

	Transform2D viewport_to_world = canvas->get_canvas_transform().affine_inverse();
	Vector2 mouse_canvas = viewport->get_local_mouse_position();
	Vector2 mouse_world = viewport_to_world.xform(mouse_canvas);
	Size2i step = get_grid_step();
	Vector2 origin = node->get_global_position();
	Vector2 local = mouse_world - origin;
	Vector2 snapped = Vector2(
			Math::floor(local.x / step.x) * step.x,
			Math::floor(local.y / step.y) * step.y);

	return snapped + origin;
}

void ScenePaintEditor::_edit_properties() {
	if (!instance || !edit_properties) {
		if (node) {
			InspectorDock::get_inspector_singleton()->edit(node);
			InspectorDock::get_singleton()->update(node);
		}
		return;
	}
	InspectorDock::get_inspector_singleton()->edit(instance);
	InspectorDock::get_singleton()->update(instance);
}

void ScenePaintEditor::_scene_picker_toggled(bool p_pressed) {
	scene_picker = p_pressed;
	callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
}

void ScenePaintEditor::_file_system_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaintEditor::_update_scene_picker).call_deferred(PICK_FILE_SYSTEM);
	}
}

void ScenePaintEditor::_scene_tree_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_released() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaintEditor::_update_scene_picker).call_deferred(PICK_SCENE_TREE);
	}
}

void ScenePaintEditor::_pinned_toggled(bool p_pressed) {
	pinned = p_pressed;
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	if (pinned && !pinned_nodes.has(scene)) {
		pinned_nodes.insert(scene, node);
	} else {
		pinned_nodes.erase(scene);
		_update_node();
	}
}

void ScenePaintEditor::_scene_changed() {
	is_painting = false;
	is_erasing = false;
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	Node *node = Object::cast_to<Node>(EditorNode::get_singleton()->get_editor_selection()->get_top_selected_nodes().pop_front());
	if (pinned_nodes.has(scene)) {
		pinned = true;
		pin_node_button->set_pressed_no_signal(true);
		cache_node = Object::cast_to<Node2D>(pinned_nodes[scene]);
	} else {
		pinned = false;
		pin_node_button->set_pressed_no_signal(false);
		cache_node = Object::cast_to<Node2D>(node);
	}
	_update_node();
}

void ScenePaintEditor::_update_scene_picker(int p_mode) {
	if (!scene_picker) {
		return;
	}
	if (preview) {
		memdelete(preview);
		preview = nullptr;
		instance = nullptr;
	}
	bool success = false;
	String scene_path;
	Node2D *node_2d = nullptr;
	switch (p_mode) {
		case PICK_FILE_SYSTEM: {
			scene_path = FileSystemDock::get_singleton()->get_current_path();
			Ref<PackedScene> scene = ResourceLoader::load(scene_path);
			if (scene.is_valid()) {
				node_2d = Object::cast_to<Node2D>(scene->instantiate());
			}
		} break;
		case PICK_SCENE_TREE: {
			node_2d = Object::cast_to<Node2D>(SceneTreeDock::get_singleton()->get_tree_editor()->get_selected());
			if (selected_scene == node_2d && preview) {
			}
		} break;
		case PICK_CANVAS_ITEM: {
			Vector2 mouse_canvas = viewport->get_local_mouse_position();
			Vector2 mouse_world = canvas->get_canvas_transform().affine_inverse().xform(mouse_canvas);
			Vector<CanvasItemEditor::_SelectResult> results;
			Node *scene = EditorNode::get_singleton()->get_edited_scene();
			canvas->find_canvas_items_at_pos(mouse_world, scene, results);

			for (int i = 0; i < results.size(); i++) {
				node_2d = Object::cast_to<Node2D>(results[i].item);
				if (node_2d && !node_2d->get_scene_file_path().is_empty()) {
					break;
				}
			}
		} break;
	}
	if (node_2d && !node_2d->get_scene_file_path().is_empty() && node_2d != EditorNode::get_singleton()->get_edited_scene()) {
		selected_scene = node_2d;
		success = true;
	}
	callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
	scene_picker_button->set_pressed(!success);
}

void ScenePaintEditor::_edit_properties_toggled(bool p_pressed) {
	edit_properties = p_pressed;
	_edit_properties();
}

void ScenePaintEditor::_grid_settings_pressed() {
	Vector2 pos = grid_settings_button->get_screen_position() + grid_settings_button->get_size();
	grid_settings_popup->set_position(pos - Vector2(grid_settings_popup->get_contents_minimum_size().width / 2, 0));
	grid_settings_popup->reset_size();
	grid_settings_popup->popup();
	grid_settings_popup->grab_focus();
}

void ScenePaintEditor::_grid_toggled(bool p_toggled) {
	grid = p_toggled;
	_update_draw();
}

void ScenePaintEditor::_snap_grid_toggled(bool p_toggled) {
	snap_grid = p_toggled;
	_update_draw();
}

void ScenePaintEditor::_grid_step_changed() {
	Size2i size = get_grid_step();
	if (size.x != grid_step_x->get_value()) {
		size.x = grid_step_x->get_value();
	}
	if (size.y != grid_step_y->get_value()) {
		size.y = grid_step_y->get_value();
	}
	set_grid_step(size);
}

void ScenePaintEditor::_update_grid_step() {
	Size2i size = get_grid_step();
	if (size.x != grid_step_x->get_value()) {
		grid_step_x->set_value(size.x);
	}
	if (size.y != grid_step_y->get_value()) {
		grid_step_y->set_value(size.y);
	}
	_update_draw();
}

void ScenePaintEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_node_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			scene_picker_button->set_button_icon(get_editor_theme_icon(SNAME("ColorPick")));
			edit_properties_button->set_button_icon(get_editor_theme_icon(SNAME("Tools")));
			grid_settings_button->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_draw();
			callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			is_painting = false;
			is_erasing = false;
		} break;
	}
}

void ScenePaintEditor::_bind_methods() {
}

void ScenePaintEditor::set_grid_step(const Size2i p_size) {
	grid_step = p_size;
	EditorSettings::get_singleton()->set_project_metadata("scene_paint_editor", "grid_step", grid_step);
	_update_grid_step();
}

Size2i ScenePaintEditor::get_grid_step() const {
	return grid_step;
}

ScenePaintEditor::ScenePaintEditor() {
	undo_redo = EditorUndoRedoManager::get_singleton();
	canvas = CanvasItemEditor::get_singleton();

	toolbar = memnew(HBoxContainer);
	canvas->add_control_to_menu_panel(toolbar);

	pin_node_button = memnew(Button);
	pin_node_button->set_toggle_mode(true);
	pin_node_button->set_accessibility_name(TTRC("Pin Node"));
	pin_node_button->set_theme_type_variation(SceneStringName(FlatButton));
	pin_node_button->set_tooltip_text(TTRC("Pin the current node.\nWhen enabled, the paiting node won't change when selecting other nodes in the SceneTree."));
	pin_node_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_pinned_toggled));
	pin_node_button->set_shortcut(ED_SHORTCUT("scene_painter/pin_node", TTRC("Pin Node"), Key::P));
	pin_node_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(pin_node_button);

	scene_picker_button = memnew(Button);
	scene_picker_button->set_toggle_mode(true);
	scene_picker_button->set_accessibility_name(TTRC("Scene Picker"));
	scene_picker_button->set_theme_type_variation(SceneStringName(FlatButton));
	scene_picker_button->set_tooltip_text(TTRC("Toggle scene picker mode.\nWhen enabled, you can select scenes from the FileSystem dock, SceneTree dock, or Viewport."));
	scene_picker_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_scene_picker_toggled));
	scene_picker_button->set_shortcut(ED_SHORTCUT("scene_painter/scene_picker", TTRC("Scene Picker"), Key::I));
	scene_picker_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(scene_picker_button);

	edit_properties_button = memnew(Button);
	edit_properties_button->set_toggle_mode(true);
	edit_properties_button->set_accessibility_name(TTRC("Edit Properties"));
	edit_properties_button->set_theme_type_variation(SceneStringName(FlatButton));
	edit_properties_button->set_tooltip_text(TTRC("Edit properties of the selected scene."));
	edit_properties_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_edit_properties_toggled));
	edit_properties_button->set_shortcut(ED_SHORTCUT("scene_painter/edit_properties", TTRC("Edit Properties"), Key::O));
	edit_properties_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(edit_properties_button);

	grid_settings_button = memnew(Button);
	grid_settings_button->set_accessibility_name(TTRC("Grid Settings"));
	grid_settings_button->set_theme_type_variation(SceneStringName(FlatButton));
	grid_settings_button->set_tooltip_text(TTRC("Open grid settings."));
	grid_settings_button->connect("pressed", callable_mp(this, &ScenePaintEditor::_grid_settings_pressed));
	toolbar->add_child(grid_settings_button);

	grid_settings_popup = memnew(PopupPanel);
	add_child(grid_settings_popup);

	VBoxContainer *vbc = memnew(VBoxContainer);
	grid_settings_popup->add_child(vbc);

	grid_toggle_button = memnew(CheckBox);
	grid_toggle_button->set_text(TTRC("Show Grid"));
	grid_toggle_button->set_accessibility_name(TTRC("Show Grid"));
	grid_toggle_button->set_theme_type_variation(SceneStringName(FlatButton));
	grid_toggle_button->set_tooltip_text(TTRC("Toggle grid visibility in the viewport."));
	grid_toggle_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_grid_toggled));
	vbc->add_child(grid_toggle_button);

	snap_grid_button = memnew(CheckBox);
	snap_grid_button->set_text(TTRC("Snap to Grid"));
	snap_grid_button->set_accessibility_name(TTRC("Snap to Grid"));
	snap_grid_button->set_theme_type_variation(SceneStringName(FlatButton));
	snap_grid_button->set_tooltip_text(TTRC("Toggle snapping to grid when painting."));
	snap_grid_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_snap_grid_toggled));
	vbc->add_child(snap_grid_button);

	Label *grid_label = memnew(Label);
	grid_label->set_text(TTRC("Grid Step:"));
	vbc->add_child(grid_label);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	grid_step_x = memnew(SpinBox);
	grid_step_x->set_min(1);
	grid_step_x->set_step(1);
	grid_step_x->set_suffix("px");
	grid_step_x->set_allow_greater(true);
	grid_step_x->set_select_all_on_focus(true);
	grid_step_x->set_custom_minimum_size(Vector2(64, 0) * EDSCALE);
	grid_step_x->connect("value_changed", callable_mp(this, &ScenePaintEditor::_grid_step_changed).unbind(1));
	hbc->add_child(grid_step_x);

	grid_step_y = memnew(SpinBox);
	grid_step_y->set_min(1);
	grid_step_y->set_step(1);
	grid_step_y->set_suffix("px");
	grid_step_y->set_allow_greater(true);
	grid_step_y->set_select_all_on_focus(true);
	grid_step_y->set_custom_minimum_size(Vector2(64, 0) * EDSCALE);
	grid_step_y->connect("value_changed", callable_mp(this, &ScenePaintEditor::_grid_step_changed).unbind(1));
	hbc->add_child(grid_step_y);

	set_grid_step(EditorSettings::get_singleton()->get_project_metadata("scene_paint_editor", "grid_step", Size2i(16, 16)));
}

ScenePaintEditor::~ScenePaintEditor() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
}

void ScenePaintEditorPlugin::_canvas_item_tool_changed(int p_tool) {
	scene_paint_editor->is_painting = false;
	scene_paint_editor->is_erasing = false;
	scene_paint_editor->is_tool_selected = (CanvasItemEditor::Tool)p_tool == CanvasItemEditor::TOOL_SCENE_PAINT;
	TypedArray<Node> selected_nodes = EditorNode::get_singleton()->get_editor_selection()->get_top_selected_nodes();
	if (selected_nodes.size() == 0) {
		make_visible(false);
		return;
	}
	scene_paint_editor->_edit(Object::cast_to<Node2D>(selected_nodes[0]));
	make_visible(true);
}

void ScenePaintEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			CanvasItemEditor::get_singleton()->connect("canvas_item_tool_changed", callable_mp(this, &ScenePaintEditorPlugin::_canvas_item_tool_changed));
			FileSystemDock::get_singleton()->get_tree_control()->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_file_system_input));
			FileSystemDock::get_singleton()->get_list_control()->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_file_system_input));
			SceneTreeDock::get_singleton()->get_tree_editor()->get_scene_tree()->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_scene_tree_input));
			EditorNode::get_singleton()->connect("scene_changed", callable_mp(scene_paint_editor, &ScenePaintEditor::_scene_changed));
			scene_paint_editor->viewport = CanvasItemEditor::get_singleton()->get_viewport_control();
			scene_paint_editor->viewport->connect("draw", callable_mp(scene_paint_editor, &ScenePaintEditor::_update_draw));
			scene_paint_editor->viewport->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_gui_input_viewport));
		} break;
		case NOTIFICATION_EXIT_TREE: {
			CanvasItemEditor::get_singleton()->disconnect("canvas_item_tool_changed", callable_mp(this, &ScenePaintEditorPlugin::_canvas_item_tool_changed));
			FileSystemDock::get_singleton()->get_tree_control()->disconnect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_file_system_input));
			FileSystemDock::get_singleton()->get_list_control()->disconnect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_file_system_input));
			SceneTreeDock::get_singleton()->get_tree_editor()->get_scene_tree()->disconnect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_scene_tree_input));
			EditorNode::get_singleton()->disconnect("scene_changed", callable_mp(scene_paint_editor, &ScenePaintEditor::_scene_changed));
		} break;
	}
}

void ScenePaintEditorPlugin::_bind_methods() {
}

void ScenePaintEditorPlugin::edit(Object *p_object) {
	ERR_FAIL_NULL(scene_paint_editor);
	scene_paint_editor->_edit(p_object);
}

bool ScenePaintEditorPlugin::handles(Object *p_object) const {
	is_node_2d = p_object->is_class("Node2D");
	return is_node_2d;
}

void ScenePaintEditorPlugin::make_visible(bool p_visible) {
	ERR_FAIL_NULL(scene_paint_editor);
	if (p_visible && is_node_2d && scene_paint_editor->is_tool_selected && scene_paint_editor->node) {
		scene_paint_editor->toolbar->show();
	} else {
		scene_paint_editor->toolbar->hide();
		scene_paint_editor->_edit(nullptr);
	}
}

ScenePaintEditorPlugin::ScenePaintEditorPlugin() {
	scene_paint_editor = memnew(ScenePaintEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(scene_paint_editor);

	make_visible(false);
}
