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
#include "core/templates/hash_map.h"
#include "editor/docks/filesystem_dock.h"
#include "editor/docks/inspector_dock.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/tile_map_layer.h"
#include "scene/gui/box_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tree.h"

void ScenePaintEditor::_can_handle(bool p_is_node_2d, bool p_edit) {
	if ((p_is_node_2d || pinned) && is_tool_selected && node) {
		toolbar->show();
	} else {
		toolbar->hide();
		if (p_edit) {
			_edit(nullptr);
		}
	}
}

void ScenePaintEditor::_edit(Object *p_object) {
	if (is_tool_selected) {
		TileMapLayer *edited_layer = Object::cast_to<TileMapLayer>(p_object);
		if (edited_layer) {
			Ref<TileSet> tile_set = edited_layer->get_tile_set();
			if (tile_set.is_valid()) {
				set_grid_step(tile_set->get_tile_size());
			}
		}
	}

	cache_node = Object::cast_to<Node2D>(p_object);
	if (node && (pinned || scene_picker)) {
		if (scene_picker) {
			SceneTreeDock::get_singleton()->set_selection(Vector<Node *>{ node });
		}
		_set_input_tool(false, false, true);
		return;
	}

	_update_node(p_object);
}

void ScenePaintEditor::_update_node(Object *p_object) {
	if (p_object == nullptr) {
		p_object = cache_node;
	}

	// If the object is not a Node2D, hide the toolbar unless pinned.
	Node2D *node_2d = Object::cast_to<Node2D>(p_object);
	_can_handle(node_2d, false);

	node = cache_node;
	_update_draw_overlay();
	_set_input_tool(false, false, true);
}

void ScenePaintEditor::_draw_overlay() {
	if (!is_visible_in_tree() || !is_tool_selected || !node || !grid) {
		return;
	}
	// Draw grid
	{
		Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
		CanvasItemEditor *canvas_editor = CanvasItemEditor::get_singleton();

		Transform2D xform = canvas_editor->get_canvas_transform() * node->get_global_transform();
		Transform2D xform_inv = xform.affine_inverse();

		Size2 viewport_size = custom_overlay->get_size();

		Vector2 corners[4] = {
			xform_inv.xform(Vector2(0, 0)),
			xform_inv.xform(Vector2(viewport_size.x, 0)),
			xform_inv.xform(Vector2(viewport_size.x, viewport_size.y)),
			xform_inv.xform(Vector2(0, viewport_size.y))
		};

		Rect2 bounds(corners[0], Vector2());
		for (int i = 1; i < 4; i++) {
			bounds.expand_to(corners[i]);
		}

		int start_x = Math::floor(bounds.position.x / grid_step.x) * grid_step.x;
		int end_x = Math::ceil((bounds.position.x + bounds.size.x) / grid_step.x) * grid_step.x;
		int start_y = Math::floor(bounds.position.y / grid_step.y) * grid_step.y;
		int end_y = Math::ceil((bounds.position.y + bounds.size.y) / grid_step.y) * grid_step.y;

		Vector2 hint_distance = xform.get_scale() * grid_step;
		float scale_fade = MIN(1.0, (MIN(hint_distance.x, hint_distance.y) - 5) / 5);
		if (scale_fade <= 0) {
			return;
		}
		grid_color.a *= scale_fade;

		for (int x = start_x; x <= end_x; x += grid_step.x) {
			Vector2 from = xform.xform(Vector2(x, start_y));
			Vector2 to = xform.xform(Vector2(x, end_y));
			custom_overlay->draw_line(from, to, grid_color);
		}

		for (int y = start_y; y <= end_y; y += grid_step.y) {
			Vector2 from = xform.xform(Vector2(start_x, y));
			Vector2 to = xform.xform(Vector2(end_x, y));
			custom_overlay->draw_line(from, to, grid_color);
		}
	}
	// Draw preview cell
	{
		if ((preview && !is_erasing) || (!snap_cell && !snap_grid)) {
			return;
		}
		Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
		Vector2 mouse_canvas = viewport->get_local_mouse_position();
		Vector2 mouse_local = xform.affine_inverse().xform(mouse_canvas);
		Vector2 snapped_local(Math::floor(mouse_local.x / grid_step.x) * grid_step.x, Math::floor(mouse_local.y / grid_step.y) * grid_step.y);
		Vector2 corners[4] = { snapped_local, snapped_local + Vector2(grid_step.x, 0), snapped_local + Vector2(grid_step.x, grid_step.y), snapped_local + Vector2(0, grid_step.y) };
		for (int i = 0; i < 4; i++) {
			corners[i] = xform.xform(corners[i]);
		}
		Color rect_color = is_erasing ? Color(0, 0, 0, 0.3) : Color(1, 1, 1, 0.3);
		Vector<Vector2> points;
		points.push_back(corners[0]);
		points.push_back(corners[1]);
		points.push_back(corners[2]);
		points.push_back(corners[3]);
		custom_overlay->draw_polygon(points, Vector<Color>({ rect_color }));
	}
}

void ScenePaintEditor::_update_draw_overlay() {
	if (custom_overlay) {
		custom_overlay->queue_redraw();
	}
}

void ScenePaintEditor::_update_preview() {
	if (!is_visible_in_tree() || !is_tool_selected || !selected_scene || !node || !node->is_inside_tree() || is_erasing) {
		if (is_erasing && preview && preview->is_inside_tree() && instance->get_parent() == preview) {
			preview->remove_child(instance);
		} else if (preview) {
			memdelete(preview);
			preview = nullptr;
			instance = nullptr;
		}
		return;
	}
	if (instance) {
		if (instance->get_scene_file_path() != selected_scene->get_scene_file_path()) {
			memdelete(instance);
			instance = nullptr;
		} else if (instance->get_scene_file_path() == selected_scene->get_scene_file_path() && instance->get_parent() == preview) {
			preview->remove_child(instance);
		}
	}
	if (preview) {
		if (preview->get_parent() && preview->get_parent() != node) {
			preview->reparent(node);
		} else if (!preview->get_parent()) {
			memdelete(preview);
			preview = nullptr;
		}
	} else if (!preview) {
		preview = memnew(Node2D);
		preview->set_modulate(Color(1, 1, 1, 0.5));
		node->add_child(preview);
	}
	if (preview && !scene_picker) {
		Node *scene = EditorNode::get_singleton()->get_edited_scene();
		if (!instance) {
			HashMap<const Node *, Node *> duplimap;
			instance = Object::cast_to<Node2D>(selected_scene->duplicate_from_editor(duplimap));
			ERR_FAIL_NULL(instance);
		}
		preview->add_child(instance, true);
		instance->set_owner(scene);
		preview->set_global_position(Vector2());
		instance->set_global_position(Vector2());
		_update_preview_position();
	}
	_edit_properties();
}

void ScenePaintEditor::_update_preview_position() {
	if (!preview || !node) {
		return;
	}

	Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
	Vector2 mouse_canvas = viewport->get_local_mouse_position();
	Vector2 mouse_local = xform.affine_inverse().xform(mouse_canvas);
	Vector2 snapped_local(
			Math::floor(mouse_local.x / grid_step.x) * grid_step.x,
			Math::floor(mouse_local.y / grid_step.y) * grid_step.y);

	Vector2 final_local = (snap_grid || snap_cell)
			? (snap_grid ? snapped_local : (snapped_local + grid_step / 2.0))
			: mouse_local;

	Vector2 final_global = node->get_global_transform().xform(final_local);

	preview->set_global_position(final_global);
}

void ScenePaintEditor::_set_input_tool(bool p_painting, bool p_erasing, bool p_force) {
	is_painting = p_painting;
	is_erasing = p_erasing;
	if (p_force) {
		callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
	}
}

void ScenePaintEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !is_tool_selected) {
		return;
	}

	// Hack: Ignore accidentally painting while panning with 'pan_view'. When holding 'pan_view', the tool doesn't change.
	if (ED_IS_SHORTCUT("canvas_item_editor/pan_view", p_event)) {
		is_panning = p_event->is_pressed();
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid()) {
		if (k->get_keycode() == Key::CMD_OR_CTRL) {
			viewport_scene_picker = k->is_pressed();
			scene_picker_button->set_pressed(viewport_scene_picker);
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && !is_panning) {
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
					_set_input_tool(true, false, true);
					_add_node_at_pos();
					accept_event();
				} else if (mb->get_button_index() == MouseButton::RIGHT) {
					_set_input_tool(false, true, true);
					_remove_node_at_pos();
					accept_event();
				}
			} else if (mb->is_released()) {
				if (mb->get_button_index() == MouseButton::LEFT) {
					_set_input_tool(false, false, true);
				} else if (mb->get_button_index() == MouseButton::RIGHT) {
					_set_input_tool(false, false, true);
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && !is_panning) {
		_update_preview_position();

		if (!scene_picker) {
			if (is_painting) {
				_add_node_at_pos();
				accept_event();
			} else if (is_erasing) {
				_remove_node_at_pos();
				accept_event();
			}
		}
	}
	_update_draw_overlay();
}

void ScenePaintEditor::_add_node_at_pos() {
	if (!node || !instance) {
		return;
	}

	Vector2 cell_pos = _get_mouse_grid_cell();
	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());

	if (snap_grid || snap_cell) {
		Vector2 offset = node->get_global_transform().basis_xform(grid_step / 2.0);
		pos = snap_grid ? cell_pos : (cell_pos + offset);
	}

	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	Vector<CanvasItemEditor::SelectResult> results;
	canvas_item_editor->find_canvas_items_at_pos(pos, node, results);
	for (CanvasItemEditor::SelectResult &result : results) {
		Node2D *root = _get_node_root(result.item);
		if (_is_scene_painted(root)) {
			return;
		}
	}

	HashMap<const Node *, Node *> duplimap;
	Node2D *node_2d = Object::cast_to<Node2D>(instance->duplicate_from_editor(duplimap));
	if (!node_2d) {
		return;
	}

	if (scene) {
		node->add_child(node_2d, true);
		node_2d->set_owner(scene);
		node_2d->set_meta("_scene_painted", true);
		node_2d->set_global_position(pos);

		undo_redo->create_action(TTR("Paint Node(s)"), UndoRedo::MERGE_ALL);
		undo_redo->add_do_reference(node_2d);
		undo_redo->add_do_method(node, "add_child", node_2d);
		undo_redo->add_do_method(node_2d, "set_owner", scene);
		undo_redo->add_do_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_do_method(node_2d, "set_global_position", pos);
		undo_redo->add_undo_method(node, "remove_child", node_2d);
		undo_redo->commit_action(false);
	}
}

void ScenePaintEditor::_remove_node_at_pos() {
	if (!node) {
		return;
	}

	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());

	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	Vector<CanvasItemEditor::SelectResult> results;
	canvas_item_editor->find_canvas_items_at_pos(pos, node, results);

	for (CanvasItemEditor::SelectResult &result : results) {
		Node2D *root = _get_node_root(result.item);
		if (!_is_scene_painted(root)) {
			continue;
		}

		Node2D *node_2d = root;
		Vector2 node_pos = node_2d->get_global_position();
		node->remove_child(node_2d);

		undo_redo->create_action(TTR("Erase Node(s)"), UndoRedo::MERGE_ALL);
		undo_redo->add_do_method(node, "remove_child", node_2d);
		undo_redo->add_undo_reference(node_2d);
		undo_redo->add_undo_method(node, "add_child", node_2d);
		undo_redo->add_undo_method(node_2d, "set_owner", scene);
		undo_redo->add_undo_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_undo_method(node_2d, "set_global_position", node_pos);
		undo_redo->commit_action(false);
	}
}

Vector2 ScenePaintEditor::_get_mouse_grid_cell() {
	if (!node) {
		return Vector2();
	}

	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());
	Vector2 local = node->get_global_transform().affine_inverse().xform(pos);
	Vector2 snapped = Vector2(
			Math::floor(local.x / grid_step.x) * grid_step.x,
			Math::floor(local.y / grid_step.y) * grid_step.y);

	return node->get_global_transform().xform(snapped);
}

void ScenePaintEditor::_edit_properties() {
	if (!instance || !edit_properties) {
		return;
	}
	InspectorDock::get_inspector_singleton()->edit(instance);
}

void ScenePaintEditor::_scene_picker_toggled(bool p_pressed) {
	scene_picker = p_pressed;
	_set_input_tool(false, false, true);
}

void ScenePaintEditor::_file_system_input(const Ref<InputEvent> &p_event) {
	if (!scene_picker || viewport_scene_picker) {
		viewport_scene_picker = false;
		scene_picker_button->set_pressed(viewport_scene_picker);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaintEditor::_update_scene_picker).call_deferred(PICK_FILE_SYSTEM);
	}
}

void ScenePaintEditor::_scene_tree_input(const Ref<InputEvent> &p_event) {
	if (!scene_picker || viewport_scene_picker) {
		viewport_scene_picker = false;
		scene_picker_button->set_pressed(viewport_scene_picker);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_released() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaintEditor::_update_scene_picker).call_deferred(PICK_SCENE_TREE);
	}
}

bool ScenePaintEditor::_is_instance_valid(Node2D *p_node) const {
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	return p_node && p_node->is_instance() && p_node != scene;
}

bool ScenePaintEditor::_is_scene_painted(Node2D *p_node) const {
	return p_node && p_node->has_meta("_scene_painted") && p_node->get_parent() == node;
}

Node2D *ScenePaintEditor::_get_node_root(Node *p_node) const {
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	Node2D *item = Object::cast_to<Node2D>(p_node);
	if (!item) {
		return nullptr;
	}
	Node2D *root = item;
	while (root && root->get_owner() != scene) {
		root = Object::cast_to<Node2D>(root->get_parent());
	}
	return root;
}

void ScenePaintEditor::_pinned_toggled(bool p_pressed) {
	pinned = p_pressed;
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	if (pinned && !pinned_nodes.has(scene)) {
		pinned_nodes[scene] = node;
		_update_node();
	} else {
		pinned_nodes.erase(scene);
		Node *selected_node = SceneTreeDock::get_singleton()->get_tree_editor()->get_selected();
		_update_node(selected_node);
	}
}

void ScenePaintEditor::_scene_changed() {
	_set_input_tool(false, false, false);
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	if (pinned_nodes.has(scene)) {
		pinned = true;
		pin_node_button->set_pressed_no_signal(true);
		cache_node = Object::cast_to<Node2D>(pinned_nodes[scene]);
	} else {
		pinned = false;
		pin_node_button->set_pressed_no_signal(false);
		cache_node = Object::cast_to<Node2D>(SceneTreeDock::get_singleton()->get_tree_editor()->get_selected());
	}
	_update_node();
}

void ScenePaintEditor::_update_scene_picker(PickMode p_mode) {
	if (!is_tool_selected || !scene_picker) {
		return;
	}

	Node2D *node_2d = nullptr;
	switch (p_mode) {
		case PICK_FILE_SYSTEM: {
			String scene_path = FileSystemDock::get_singleton()->get_current_path();
			Ref<PackedScene> scene = ResourceLoader::load(scene_path);
			if (scene.is_valid()) {
				node_2d = Object::cast_to<Node2D>(scene->instantiate());
			}
		} break;
		case PICK_SCENE_TREE: {
			node_2d = Object::cast_to<Node2D>(SceneTreeDock::get_singleton()->get_tree_editor()->get_selected());
		} break;
		case PICK_CANVAS_ITEM: {
			Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());
			Vector<CanvasItemEditor::SelectResult> results;
			Node *scene = EditorNode::get_singleton()->get_edited_scene();
			canvas_item_editor->find_canvas_items_at_pos(pos, scene, results);
			for (const CanvasItemEditor::SelectResult &result : results) {
				Node2D *root = _get_node_root(result.item);
				if (_is_instance_valid(root)) {
					node_2d = root;
					break;
				}
			}
		} break;
	}

	if (_is_instance_valid(node_2d)) {
		undo_redo->create_action(TTR("Pick Scene"));
		undo_redo->add_do_reference(node_2d);
		undo_redo->add_do_method(this, "set_picked_scene", node_2d);
		if (selected_scene) {
			undo_redo->add_undo_reference(selected_scene);
			undo_redo->add_undo_method(this, "set_picked_scene", selected_scene);
		} else {
			undo_redo->add_undo_method(this, "set_picked_scene", (Object *)nullptr);
		}
		undo_redo->commit_action();
	}
}

void ScenePaintEditor::_edit_properties_toggled(bool p_pressed) {
	edit_properties = p_pressed;
	if (!edit_properties) {
		if (node && pinned) {
			InspectorDock::get_inspector_singleton()->edit(node);
		} else {
			Node *selected_node = SceneTreeDock::get_singleton()->get_tree_editor()->get_selected();
			if (selected_node) {
				InspectorDock::get_inspector_singleton()->edit(selected_node);
			}
		}
	}
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
	_update_draw_overlay();
}

void ScenePaintEditor::_paint_mode_changed(PaintMode p_mode) {
	snap_cell = p_mode == PAINT_MODE_SNAP_CELL;
	snap_grid = p_mode == PAINT_MODE_SNAP_GRID;
	_update_draw_overlay();
}

void ScenePaintEditor::_grid_step_changed() {
	if (grid_step.x != grid_step_x->get_value()) {
		grid_step.x = grid_step_x->get_value();
	}
	if (grid_step.y != grid_step_y->get_value()) {
		grid_step.y = grid_step_y->get_value();
	}
	set_grid_step(grid_step);
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
			_update_draw_overlay();
			callable_mp(this, &ScenePaintEditor::_update_preview).call_deferred();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			_set_input_tool(false, false, true);
		} break;
	}
}

void ScenePaintEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_picked_scene", "scene"), &ScenePaintEditor::set_picked_scene);
}

void ScenePaintEditor::set_picked_scene(Node2D *p_scene) {
	selected_scene = p_scene;
	scene_picker_button->set_pressed(false);
	_set_input_tool(false, false, true);
}

void ScenePaintEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!custom_overlay) {
		custom_overlay = memnew(Control);
		custom_overlay->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
		custom_overlay->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		custom_overlay->set_clip_contents(true);
		custom_overlay->set_draw_behind_parent(true);
		p_overlay->add_child(custom_overlay);
		custom_overlay->connect(SceneStringName(draw), callable_mp(this, &ScenePaintEditor::_draw_overlay));
	}
}

void ScenePaintEditor::set_grid_step(const Size2i p_size) {
	grid_step = p_size;
	if (grid_step.x != grid_step_x->get_value()) {
		grid_step_x->set_value_no_signal(grid_step.x);
	}
	if (grid_step.y != grid_step_y->get_value()) {
		grid_step_y->set_value_no_signal(grid_step.y);
	}
	_update_draw_overlay();
	EditorSettings::get_singleton()->set_project_metadata("scene_paint_editor", "grid_step", grid_step);
}

ScenePaintEditor::ScenePaintEditor() {
	undo_redo = EditorUndoRedoManager::get_singleton();
	canvas_item_editor = CanvasItemEditor::get_singleton();

	toolbar = memnew(HBoxContainer);
	canvas_item_editor->add_control_to_menu_panel(toolbar);
	toolbar->hide();

	pin_node_button = memnew(Button);
	pin_node_button->set_toggle_mode(true);
	pin_node_button->set_accessibility_name(TTRC("Pin Node"));
	pin_node_button->set_theme_type_variation(SceneStringName(FlatButton));
	pin_node_button->set_tooltip_text(TTRC("Pin the current node.\nWhen enabled, the painting parent node won't change when selecting other nodes in the scene."));
	pin_node_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_pinned_toggled));
	pin_node_button->set_shortcut(ED_SHORTCUT("scene_painter/pin_node", TTRC("Pin Node"), Key::P));
	pin_node_button->set_shortcut_context(canvas_item_editor);
	toolbar->add_child(pin_node_button);

	scene_picker_button = memnew(Button);
	scene_picker_button->set_toggle_mode(true);
	scene_picker_button->set_accessibility_name(TTRC("Scene Picker"));
	scene_picker_button->set_theme_type_variation(SceneStringName(FlatButton));
	scene_picker_button->set_tooltip_text(TTRC("Toggle scene picker mode.\nHolding Ctrl enables the scene picker tool. \nWhen enabled, you can select scenes from the FileSystem dock, Scene dock, or 2D editor's viewport."));
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
	grid_toggle_button->set_tooltip_text(TTRC("Toggle grid visibility in the viewport."));
	grid_toggle_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_grid_toggled));
	vbc->add_child(grid_toggle_button);

	HSeparator *hsep = memnew(HSeparator);
	hsep->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hsep);

	button_group.instantiate();

	free_paint_button = memnew(CheckBox);
	free_paint_button->set_text(TTRC("Free Paint"));
	free_paint_button->set_accessibility_name(TTRC("Free Paint"));
	free_paint_button->set_tooltip_text(TTRC("Toggle free painting without snapping."));
	free_paint_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_paint_mode_changed).bind(PAINT_MODE_FREE).unbind(1));
	free_paint_button->set_pressed(true);
	free_paint_button->set_button_group(button_group);
	vbc->add_child(free_paint_button);

	snap_cell_button = memnew(CheckBox);
	snap_cell_button->set_text(TTRC("Snap to Cell"));
	snap_cell_button->set_accessibility_name(TTRC("Snap to Cell"));
	snap_cell_button->set_tooltip_text(TTRC("Toggle snapping to cell when painting."));
	snap_cell_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_paint_mode_changed).bind(PAINT_MODE_SNAP_CELL).unbind(1));
	snap_cell_button->set_button_group(button_group);
	vbc->add_child(snap_cell_button);

	snap_grid_button = memnew(CheckBox);
	snap_grid_button->set_text(TTRC("Snap to Grid"));
	snap_grid_button->set_accessibility_name(TTRC("Snap to Grid"));
	snap_grid_button->set_tooltip_text(TTRC("Toggle snapping to grid when painting."));
	snap_grid_button->connect("toggled", callable_mp(this, &ScenePaintEditor::_paint_mode_changed).bind(PAINT_MODE_SNAP_GRID).unbind(1));
	snap_grid_button->set_button_group(button_group);
	vbc->add_child(snap_grid_button);

	hsep = memnew(HSeparator);
	hsep->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hsep);

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

void ScenePaintEditorPlugin::_canvas_item_tool_changed(int p_tool) {
	scene_paint_editor->is_painting = false;
	scene_paint_editor->is_erasing = false;
	scene_paint_editor->selected_scene = nullptr;
	scene_paint_editor->is_tool_selected = (CanvasItemEditor::Tool)p_tool == CanvasItemEditor::TOOL_SCENE_PAINT;
	List<Node *> selected_nodes = EditorNode::get_singleton()->get_editor_selection()->get_top_selected_node_list();
	if (selected_nodes.size() == 0) {
		make_visible(false);
		return;
	}
	scene_paint_editor->_edit(Object::cast_to<Node2D>(selected_nodes.front()->get()));
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
			scene_paint_editor->viewport->connect("draw", callable_mp(scene_paint_editor, &ScenePaintEditor::_update_draw_overlay));
			scene_paint_editor->viewport->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_gui_input_viewport));
		} break;
	}
}

void ScenePaintEditorPlugin::edit(Object *p_object) {
	scene_paint_editor->_edit(p_object);
}

bool ScenePaintEditorPlugin::handles(Object *p_object) const {
	is_node_2d = bool(Object::cast_to<Node2D>(p_object));
	return is_node_2d;
}

void ScenePaintEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		scene_paint_editor->_can_handle(is_node_2d, false);
	} else {
		scene_paint_editor->_can_handle(is_node_2d, true);
	}
}

void ScenePaintEditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	scene_paint_editor->forward_canvas_draw_over_viewport(p_overlay);
}

ScenePaintEditorPlugin::ScenePaintEditorPlugin() {
	scene_paint_editor = memnew(ScenePaintEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(scene_paint_editor);
	make_visible(false);
}
