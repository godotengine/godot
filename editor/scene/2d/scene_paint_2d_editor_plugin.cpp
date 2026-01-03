/**************************************************************************/
/*  scene_paint_2d_editor_plugin.cpp                                      */
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

#include "scene_paint_2d_editor_plugin.h"

#include "core/config/project_settings.h"
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
#include "scene/gui/subviewport_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"

void ScenePaint2DEditor::_can_handle(bool p_is_node_2d, bool p_edit) {
	if ((p_is_node_2d || pinned) && is_tool_selected && _is_node_valid()) {
		toolbar->show();
	} else {
		toolbar->hide();
		if (p_edit) {
			_edit(nullptr);
		}
	}
}

void ScenePaint2DEditor::_edit(Object *p_object) {
	if (is_tool_selected) {
		TileMapLayer *edited_layer = Object::cast_to<TileMapLayer>(p_object);
		if (edited_layer) {
			Ref<TileSet> tile_set = edited_layer->get_tile_set();
			if (tile_set.is_valid()) {
				grid_step = tile_set->get_tile_size();
			}
		} else {
			grid_step = CanvasItemEditor::get_singleton()->get_grid_step();
		}
	}

	cache_node = Object::cast_to<Node2D>(p_object);
	if (_is_node_valid() && (pinned || input_tool == INPUT_TOOL_PICK)) {
		if (input_tool == INPUT_TOOL_PICK) {
			SceneTreeDock::get_singleton()->set_selection(Vector<Node *>{ node });
		}
		return;
	}

	_update_node(p_object);
}

void ScenePaint2DEditor::_update_node(Object *p_object) {
	if (p_object == nullptr) {
		p_object = cache_node;
	}

	// If the object is not a Node2D, hide the toolbar unless pinned.
	Node2D *node_2d = Object::cast_to<Node2D>(p_object);
	_can_handle(node_2d, false);

	node = cache_node;
	_update_draw_overlay();
	input_tool = INPUT_TOOL_NONE;
}

bool ScenePaint2DEditor::_is_node_valid() {
	return node && node->is_inside_tree();
}

void ScenePaint2DEditor::_draw_overlay() {
	if (!is_visible_in_tree() || !is_tool_selected || !_is_node_valid()) {
		_clear_instance(true);
		return;
	}
	if (grid) {
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
			if (scale_fade > 0) {
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
		}
		// Draw preview cell
		if ((!instance || !instance->is_visible()) && paint_mode != PAINT_MODE_FREE) {
			CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
			Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
			Vector2 mouse_canvas = viewport->get_local_mouse_position();
			Vector2 mouse_local = xform.affine_inverse().xform(mouse_canvas);
			Vector2 snapped_local = Vector2(
					Math::floor(mouse_local.x / grid_step.x) * grid_step.x,
					Math::floor(mouse_local.y / grid_step.y) * grid_step.y);
			Vector2 corners[4] = {
				snapped_local, snapped_local + Vector2(grid_step.x, 0),
				snapped_local + Vector2(grid_step.x, grid_step.y),
				snapped_local + Vector2(0, grid_step.y)
			};
			for (int i = 0; i < 4; i++) {
				corners[i] = xform.xform(corners[i]);
			}
			Color rect_color = input_tool == INPUT_TOOL_ERASE ? Color(0, 0, 0, 0.3) : Color(1, 1, 1, 0.3);
			Vector<Vector2> points;
			points.push_back(corners[0]);
			points.push_back(corners[1]);
			points.push_back(corners[2]);
			points.push_back(corners[3]);
			custom_overlay->draw_polygon(points, Vector<Color>({ rect_color }));
		}
	}
	// instance
	if (_is_instance_valid() && input_tool != INPUT_TOOL_ERASE && input_tool != INPUT_TOOL_PICK && input_tool != INPUT_TOOL_QUICK_PICK) {
		_add_instance(true);
	} else if (!_is_instance_valid() || input_tool == INPUT_TOOL_ERASE || input_tool == INPUT_TOOL_PICK || input_tool == INPUT_TOOL_QUICK_PICK) {
		_clear_instance(true);
	}
	if (instance) {
		CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
		Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
		Vector2 mouse_canvas = viewport->get_local_mouse_position();
		Vector2 mouse_local = xform.affine_inverse().xform(mouse_canvas);
		Vector2 final_local;
		if (paint_mode == PAINT_MODE_FREE) {
			final_local = mouse_local;
		} else {
			Vector2 snapped_local;
			if (paint_mode == PAINT_MODE_SNAP_CELL) {
				snapped_local = Vector2(
						Math::floor(mouse_local.x / grid_step.x) * grid_step.x,
						Math::floor(mouse_local.y / grid_step.y) * grid_step.y);
			} else if (paint_mode == PAINT_MODE_SNAP_GRID) {
				snapped_local = Vector2(
						Math::round(mouse_local.x / grid_step.x) * grid_step.x,
						Math::round(mouse_local.y / grid_step.y) * grid_step.y);
			}
			final_local = (paint_mode != PAINT_MODE_FREE)
					? (paint_mode == PAINT_MODE_SNAP_GRID ? snapped_local : (snapped_local + grid_step / 2.0))
					: mouse_local;
		}
		instance_container->set_position(xform.xform(final_local));
		instance_container->set_rotation(xform.get_rotation());
		instance_container->set_scale(xform.get_scale());
		_edit_properties();
	}
}

void ScenePaint2DEditor::_add_instance(bool p_show) {
	if (p_show && instance_container) {
		instance_container->show();
	} else if (selected_scene && instance_container && !instance) {
		HashMap<const Node *, Node *> duplimap;
		instance = Object::cast_to<Node2D>(selected_scene->duplicate_from_editor(duplimap));
		instance_container->add_child(instance, true);
		instance->set_position(Point2());
	}
}

void ScenePaint2DEditor::_clear_instance(bool p_hide) {
	if (p_hide && instance_container) {
		instance_container->hide();
	} else if (instance) {
		instance->queue_free();
		instance = nullptr;
	}
}

void ScenePaint2DEditor::_update_instance() {
	if (_is_instance_valid()) {
		_clear_instance();
	}
	if (!_is_instance_valid()) {
		_add_instance();
	}
}

bool ScenePaint2DEditor::_is_instance_valid() {
	return instance && instance->is_inside_tree();
}

void ScenePaint2DEditor::_update_draw_overlay() {
	if (custom_overlay) {
		custom_overlay->queue_redraw();
	}
}

void ScenePaint2DEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !is_tool_selected) {
		return;
	}

	// Hack: Ignore accidentally painting while panning with 'pan_view'. When holding 'pan_view', the tool doesn't change.
	if (ED_IS_SHORTCUT("canvas_item_editor/pan_view", p_event) && p_event->is_pressed()) {
		input_tool = INPUT_TOOL_PAN;
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->get_keycode() == Key::CMD_OR_CTRL) {
		if (k->is_pressed()) {
			input_tool = INPUT_TOOL_QUICK_PICK;
		}
		scene_picker_button->set_pressed(input_tool == INPUT_TOOL_QUICK_PICK);
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && input_tool != INPUT_TOOL_PAN) {
		if (input_tool == INPUT_TOOL_PICK || input_tool == INPUT_TOOL_QUICK_PICK) {
			if (mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
				callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_CANVAS_ITEM);
			}
		} else if (mb->is_pressed()) {
			if (mb->get_button_index() == MouseButton::LEFT) {
				if (!node) {
					EditorNode::get_singleton()->show_warning(
							"No target node selected. Please select a Node2D compatible node in the SceneTree where painted scenes will be added.");
					return;
				} else if (!selected_scene) {
					EditorNode::get_singleton()->show_warning(
							"No scene selected for painting. Use the Scene Picker from the toolbar to choose a scene before painting.");
					return;
				} else if (!FileAccess::exists(selected_scene->get_scene_file_path())) {
					EditorNode::get_singleton()->show_warning(
							vformat("The selected scene '%s' no longer exists on disk. Please select a valid scene to paint.",
									selected_scene->get_scene_file_path()));
					_set_picked_scene(nullptr);
					return;
				}
				input_tool = INPUT_TOOL_PAINT;
				_add_node_at_pos();
				accept_event();
			} else if (mb->get_button_index() == MouseButton::RIGHT) {
				if (mb->is_double_click()) {
					_set_picked_scene(nullptr);
					return;
				}
				input_tool = INPUT_TOOL_ERASE;
				_remove_node_at_pos();
				accept_event();
			}
		} else if (mb->is_released()) {
			if (mb->get_button_index() == MouseButton::LEFT || mb->get_button_index() == MouseButton::RIGHT) {
				input_tool = INPUT_TOOL_NONE;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (input_tool == INPUT_TOOL_PAINT) {
			_add_node_at_pos();
			accept_event();
		} else if (input_tool == INPUT_TOOL_ERASE) {
			_remove_node_at_pos();
			accept_event();
		}
	}
	_update_draw_overlay();
}

void ScenePaint2DEditor::_add_node_at_pos() {
	if (!_is_node_valid() || !_is_instance_valid()) {
		return;
	}

	Vector2 cell_pos = _get_mouse_grid_cell();
	CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());

	if (paint_mode != PAINT_MODE_FREE) {
		Vector2 offset = node->get_global_transform().basis_xform(grid_step / 2.0);
		pos = paint_mode == PAINT_MODE_SNAP_GRID ? cell_pos : (cell_pos + offset);
	}

	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	if (!allow_overlapping) {
		Vector<CanvasItemEditor::SelectResult> results;
		canvas_item_editor->find_canvas_items_at_pos(pos, node, results);
		for (const CanvasItemEditor::SelectResult &result : results) {
			Node2D *root = _get_node_root(result.item);
			if (_is_scene_painted(root)) {
				if (paint_mode == PAINT_MODE_FREE) {
					return;
				} else if (paint_mode != PAINT_MODE_FREE && pos == root->get_position()) {
					return;
				}
			}
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

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Paint Node(s)"), UndoRedo::MERGE_ALL);
		undo_redo->add_do_reference(node_2d);
		undo_redo->add_do_method(node, "add_child", node_2d, true);
		undo_redo->add_do_method(node_2d, "set_owner", scene);
		undo_redo->add_do_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_do_method(node_2d, "set_global_position", pos);
		undo_redo->add_undo_method(node, "remove_child", node_2d);
		undo_redo->commit_action(false);
	}
}

void ScenePaint2DEditor::_remove_node_at_pos() {
	if (!_is_node_valid()) {
		return;
	}

	Vector2 cell_pos = _get_mouse_grid_cell();
	CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());

	if (paint_mode != PAINT_MODE_FREE) {
		Vector2 offset = node->get_global_transform().basis_xform(grid_step / 2.0);
		pos = paint_mode == PAINT_MODE_SNAP_GRID ? cell_pos : (cell_pos + offset);
	}

	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	Vector<CanvasItemEditor::SelectResult> results;
	canvas_item_editor->find_canvas_items_at_pos(pos, node, results);

	for (const CanvasItemEditor::SelectResult &result : results) {
		Node2D *root = _get_node_root(result.item);
		if (!_is_scene_painted(root)) {
			continue;
		}

		Node2D *node_2d = root;
		Vector2 node_pos = node_2d->get_global_position();
		node->remove_child(node_2d);

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Erase Node(s)"), UndoRedo::MERGE_ALL);
		undo_redo->add_do_method(node, "remove_child", node_2d);
		undo_redo->add_undo_reference(node_2d);
		undo_redo->add_undo_method(node, "add_child", node_2d, true);
		undo_redo->add_undo_method(node_2d, "set_owner", scene);
		undo_redo->add_undo_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_undo_method(node_2d, "set_global_position", node_pos);
		undo_redo->commit_action(false);
	}
}

Vector2 ScenePaint2DEditor::_get_mouse_grid_cell() {
	if (!_is_node_valid()) {
		return Vector2();
	}
	CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());
	Vector2 local = node->get_global_transform().affine_inverse().xform(pos);
	Vector2 snapped;
	if (paint_mode == PAINT_MODE_SNAP_CELL) {
		snapped = Vector2(
				Math::floor(local.x / grid_step.x) * grid_step.x,
				Math::floor(local.y / grid_step.y) * grid_step.y);
	} else if (paint_mode == PAINT_MODE_SNAP_GRID) {
		snapped = Vector2(
				Math::round(local.x / grid_step.x) * grid_step.x,
				Math::round(local.y / grid_step.y) * grid_step.y);
	}

	return node->get_global_transform().xform(snapped);
}

void ScenePaint2DEditor::_edit_properties() {
	if (!_is_instance_valid() || !edit_properties) {
		return;
	}
	InspectorDock::get_inspector_singleton()->edit(instance);
}

void ScenePaint2DEditor::_scene_picker_toggled(bool p_pressed) {
	input_tool = p_pressed ? INPUT_TOOL_PICK : INPUT_TOOL_NONE;
}

void ScenePaint2DEditor::_file_system_input(const Ref<InputEvent> &p_event) {
	if (input_tool != INPUT_TOOL_PICK || input_tool == INPUT_TOOL_QUICK_PICK) {
		input_tool = INPUT_TOOL_NONE;
		scene_picker_button->set_pressed(false);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_FILE_SYSTEM);
	}
}

void ScenePaint2DEditor::_scene_tree_input(const Ref<InputEvent> &p_event) {
	if (input_tool != INPUT_TOOL_PICK || input_tool == INPUT_TOOL_QUICK_PICK) {
		input_tool = INPUT_TOOL_NONE;
		scene_picker_button->set_pressed(false);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_released() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_SCENE_TREE);
	}
}

void ScenePaint2DEditor::_recent_item_selected(int p_idx) {
	recent_idx = p_idx;
	if (recent_idx == recent_scenes_button->get_item_count() - 1) {
		_set_picked_scene(nullptr);
		recent_scenes_button->select(-1);
		EditorSettings::get_singleton()->set_project_metadata("scene_paint_2d_editor", "recent_scenes", PackedStringArray());
		return;
	}
	callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_RECENT_LIST);
}

bool ScenePaint2DEditor::_is_selected_scene_valid(Node2D *p_node) const {
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	return p_node && p_node->is_instance() && p_node != scene;
}

bool ScenePaint2DEditor::_is_scene_painted(Node2D *p_node) const {
	return p_node && p_node->has_meta("_scene_painted") && p_node->get_parent() == node;
}

Node2D *ScenePaint2DEditor::_get_node_root(Node *p_node) const {
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

void ScenePaint2DEditor::_set_pinned(bool p_pinned) {
	pinned = p_pinned;
	pin_node_button->set_pressed_no_signal(pinned);
}

void ScenePaint2DEditor::_pinned_toggled(bool p_pressed) {
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

void ScenePaint2DEditor::_scene_changed() {
	input_tool = INPUT_TOOL_NONE;
	Node *scene = EditorNode::get_singleton()->get_edited_scene();
	if (pinned_nodes.has(scene)) {
		Node *pinned_node = pinned_nodes[scene];
		if (pinned_node && pinned_node->is_inside_tree()) {
			_set_pinned(true);
			cache_node = Object::cast_to<Node2D>(pinned_node);
		} else {
			pinned_nodes.erase(scene);
			_set_pinned(false);
		}
	} else {
		_set_pinned(false);
		cache_node = Object::cast_to<Node2D>(SceneTreeDock::get_singleton()->get_tree_editor()->get_selected());
	}
	_update_node();
}

void ScenePaint2DEditor::_update_scene_picker(int p_mode) {
	if (!is_tool_selected) {
		return;
	}

	Node2D *node_2d = nullptr;
	PickMode pick_mode = (PickMode)p_mode;
	switch (pick_mode) {
		case PICK_FILE_SYSTEM: {
			String scene_path = FileSystemDock::get_singleton()->get_current_path();
			Ref<PackedScene> scene = ResourceLoader::load(scene_path);
			if (scene.is_null()) {
				return;
			}
			Ref<SceneState> scene_state = scene->get_state();
			String type;
			while (scene_state.is_valid() && type.is_empty()) {
				ERR_FAIL_COND(scene_state->get_node_count() < 1);
				type = scene_state->get_node_type(0);
				scene_state = scene_state->get_base_scene_state();
			}
			ERR_FAIL_COND_EDMSG(type.is_empty(), "The selected scene is invalid.");
			bool extends_current_class = ClassDB::is_parent_class(type, "Node2D");
			if (scene.is_valid() && extends_current_class) {
				node_2d = Object::cast_to<Node2D>(scene->instantiate());
			}
		} break;
		case PICK_SCENE_TREE: {
			node_2d = Object::cast_to<Node2D>(SceneTreeDock::get_singleton()->get_tree_editor()->get_selected());
		} break;
		case PICK_CANVAS_ITEM: {
			CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
			Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());
			Vector<CanvasItemEditor::SelectResult> results;
			Node *scene = EditorNode::get_singleton()->get_edited_scene();
			canvas_item_editor->find_canvas_items_at_pos(pos, scene, results);
			for (const CanvasItemEditor::SelectResult &result : results) {
				Node2D *root = _get_node_root(result.item);
				if (_is_selected_scene_valid(root)) {
					node_2d = root;
					break;
				}
			}
		} break;
		case PICK_RECENT_LIST: {
			String scene_path = recent_scenes_button->get_item_metadata(recent_idx);
			if (!ResourceLoader::exists(scene_path)) {
				EditorNode::get_singleton()->show_accept(
						TTR("The selected scene could not be found. It may have been moved or deleted."),
						TTR("OK"));
				PackedStringArray rc = EditorSettings::get_singleton()->get_project_metadata("scene_paint_2d_editor", "recent_scenes", PackedStringArray());
				rc.erase(scene_path);
				EditorSettings::get_singleton()->set_project_metadata("scene_paint_2d_editor", "recent_scenes", rc);
				callable_mp(this, &ScenePaint2DEditor::_update_recent_scenes).call_deferred();
				return;
			}
			Ref<PackedScene> scene = ResourceLoader::load(scene_path);
			if (scene.is_valid()) {
				node_2d = Object::cast_to<Node2D>(scene->instantiate());
			}
		}
	}

	if (_is_selected_scene_valid(node_2d)) {
		_set_picked_scene(node_2d);
	}
}

void ScenePaint2DEditor::_edit_properties_toggled(bool p_pressed) {
	edit_properties = p_pressed;
	edit_properties_button->set_pressed_no_signal(edit_properties);
	if (!edit_properties) {
		Node *selected_node = SceneTreeDock::get_singleton()->get_tree_editor()->get_selected();
		if (selected_node) {
			InspectorDock::get_inspector_singleton()->edit(selected_node);
		} else if (node) {
			InspectorDock::get_inspector_singleton()->edit(node);
		}
	}
	_edit_properties();
}

void ScenePaint2DEditor::_advanced_settings_pressed() {
	Vector2 pos = advanced_settings_button->get_screen_position() + advanced_settings_button->get_size();
	advanced_settings_popup->set_position(pos - Vector2(advanced_settings_popup->get_contents_minimum_size().width / 2, 0));
	advanced_settings_popup->reset_size();
	advanced_settings_popup->popup();
	advanced_settings_popup->grab_focus();
}

void ScenePaint2DEditor::_grid_toggled(bool p_toggled) {
	grid = p_toggled;
	_update_draw_overlay();
}

void ScenePaint2DEditor::_paint_mode_changed(int p_mode) {
	paint_mode = (PaintMode)p_mode;
	_update_draw_overlay();
}

void ScenePaint2DEditor::_grid_step_changed() {
	grid_step = CanvasItemEditor::get_singleton()->get_grid_step();
}

void ScenePaint2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_node_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			scene_picker_button->set_button_icon(get_editor_theme_icon(SNAME("ColorPick")));
			edit_properties_button->set_button_icon(get_editor_theme_icon(SNAME("Tools")));
			advanced_settings_button->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
			advanced_settings_popup->add_theme_style_override(SceneStringName(panel),
					get_theme_stylebox(SceneStringName(panel), SNAME("AcceptDialog")));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_draw_overlay();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			input_tool = INPUT_TOOL_NONE;
		} break;
	}
}

void ScenePaint2DEditor::_set_picked_scene(Node2D *p_scene) {
	selected_scene = p_scene;
	scene_picker_button->set_pressed(false);
	input_tool = INPUT_TOOL_NONE;
	if (!selected_scene) {
		recent_scenes_button->select(-1);
	}
	String scene_path = selected_scene ? selected_scene->get_scene_file_path() : String();
	callable_mp(this, &ScenePaint2DEditor::_add_to_recent_scenes).call_deferred(scene_path);
	callable_mp(this, &ScenePaint2DEditor::_update_instance).call_deferred();
}

void ScenePaint2DEditor::_add_to_recent_scenes(const String &p_scene) {
	if (p_scene.is_empty()) {
		return;
	}
	PackedStringArray rc = EditorSettings::get_singleton()->get_project_metadata("scene_paint_2d_editor", "recent_scenes", PackedStringArray());
	String uid = ResourceUID::path_to_uid(p_scene);
	rc.erase(uid);
	rc.insert(0, uid);
	if (rc.size() > 10) {
		rc.resize(10);
	}

	EditorSettings::get_singleton()->set_project_metadata("scene_paint_2d_editor", "recent_scenes", rc);
	_update_recent_scenes();
}

void ScenePaint2DEditor::_update_recent_scenes() {
	PackedStringArray rc = EditorSettings::get_singleton()->get_project_metadata("scene_paint_2d_editor", "recent_scenes", PackedStringArray());
	recent_scenes_button->clear();

	if (rc.is_empty()) {
		recent_scenes_button->add_item(TTRC("No Recent Scenes"), -1);
		recent_scenes_button->set_item_disabled(-1, true);
		recent_scenes_button->get_popup()->set_item_as_radio_checkable(-1, false);
	} else {
		for (const String &uid : rc) {
			String path = ResourceUID::ensure_path(uid);
			recent_scenes_button->add_item(path.get_file());
			recent_scenes_button->set_item_tooltip(-1, path);
			recent_scenes_button->set_item_metadata(-1, uid);
			recent_scenes_button->get_popup()->set_item_as_radio_checkable(-1, false);
		}
		recent_scenes_button->add_separator();
		recent_scenes_button->add_item(TTRC("Clear Recent Scenes"), -1);
		recent_scenes_button->get_popup()->set_item_as_radio_checkable(-1, false);
	}
	recent_scenes_button->set_item_auto_translate_mode(-1, AUTO_TRANSLATE_MODE_ALWAYS);
	if (!selected_scene) {
		recent_scenes_button->select(-1);
	}
}

void ScenePaint2DEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!custom_overlay) {
		custom_overlay = memnew(Control);
		custom_overlay->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
		custom_overlay->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
		custom_overlay->set_clip_contents(true);
		custom_overlay->set_draw_behind_parent(true);
		p_overlay->add_child(custom_overlay);
		custom_overlay->connect(SceneStringName(draw), callable_mp(this, &ScenePaint2DEditor::_draw_overlay));
	}
	if (!instance_container) {
		instance_container = memnew(Node2D);
		instance_container->set_modulate(Color(1, 1, 1, 0.3));
		custom_overlay->add_child(instance_container);
	}
}

ScenePaint2DEditor::ScenePaint2DEditor() {
	toolbar = memnew(HBoxContainer);
	CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
	canvas_item_editor->add_control_to_menu_panel(toolbar);
	toolbar->hide();

	pin_node_button = memnew(Button);
	pin_node_button->set_toggle_mode(true);
	pin_node_button->set_accessibility_name(TTRC("Pin Node"));
	pin_node_button->set_theme_type_variation(SceneStringName(FlatButton));
	pin_node_button->set_tooltip_text(TTRC("Pin the current node.\nWhen enabled, the painting parent node won't change when selecting other nodes in the scene."));
	pin_node_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_pinned_toggled));
	pin_node_button->set_shortcut(ED_SHORTCUT("scene_painter/pin_node", TTRC("Pin Node"), Key::P));
	pin_node_button->set_shortcut_context(canvas_item_editor);
	toolbar->add_child(pin_node_button);

	scene_picker_button = memnew(Button);
	scene_picker_button->set_toggle_mode(true);
	scene_picker_button->set_accessibility_name(TTRC("Scene Picker"));
	scene_picker_button->set_theme_type_variation(SceneStringName(FlatButton));
	scene_picker_button->set_tooltip_text(TTRC("Toggle scene picker mode.\nWhen enabled, you can select scenes from the FileSystem dock, Scene dock, or 2D editor's viewport.\nHolding Ctrl enables picking from the 2D editor's viewport.\nRight-click the button to open a list of recently used scenes."));
	scene_picker_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_scene_picker_toggled));
	scene_picker_button->set_shortcut(ED_SHORTCUT("scene_painter/scene_picker", TTRC("Scene Picker"), Key::I));
	scene_picker_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(scene_picker_button);

	recent_scenes_button = memnew(OptionButton);
	recent_scenes_button->set_theme_type_variation(SceneStringName(FlatButton));
	recent_scenes_button->set_custom_minimum_size(Vector2(128 * EDSCALE, 0));
	recent_scenes_button->set_fit_to_longest_item(false);
	recent_scenes_button->connect(SceneStringName(item_selected), callable_mp(this, &ScenePaint2DEditor::_recent_item_selected));
	toolbar->add_child(recent_scenes_button);

	PopupMenu *popup = recent_scenes_button->get_popup();
	popup->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	popup->connect("about_to_popup", callable_mp(this, &ScenePaint2DEditor::_update_recent_scenes));

	edit_properties_button = memnew(Button);
	edit_properties_button->set_toggle_mode(true);
	edit_properties_button->set_accessibility_name(TTRC("Edit Properties"));
	edit_properties_button->set_theme_type_variation(SceneStringName(FlatButton));
	edit_properties_button->set_tooltip_text(TTRC("Edit properties of the selected scene."));
	edit_properties_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_edit_properties_toggled));
	edit_properties_button->set_shortcut(ED_SHORTCUT("scene_painter/edit_properties", TTRC("Edit Properties"), Key::O));
	edit_properties_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(edit_properties_button);

	advanced_settings_button = memnew(Button);
	advanced_settings_button->set_accessibility_name(TTRC("Advanced Settings"));
	advanced_settings_button->set_theme_type_variation(SceneStringName(FlatButton));
	advanced_settings_button->set_tooltip_text(TTRC("Open advanced settings."));
	advanced_settings_button->connect(SceneStringName(pressed), callable_mp(this, &ScenePaint2DEditor::_advanced_settings_pressed));
	toolbar->add_child(advanced_settings_button);

	advanced_settings_popup = memnew(PopupPanel);
	add_child(advanced_settings_popup);

	VBoxContainer *vbc = memnew(VBoxContainer);
	advanced_settings_popup->add_child(vbc);

	grid_toggle_button = memnew(CheckBox);
	grid_toggle_button->set_text(TTRC("Show Grid"));
	grid_toggle_button->set_accessibility_name(TTRC("Show Grid"));
	grid_toggle_button->set_tooltip_text(TTRC(U"Toggle the grid visibility in the viewport.\nThe grid spacing is determined by the Snapping Options → Configure Snap → Grid Step or based on the Tile Size of the selected TileSet in the TileMapLayer."));
	grid_toggle_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_grid_toggled));
	vbc->add_child(grid_toggle_button);

	HSeparator *hsep = memnew(HSeparator);
	hsep->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hsep);

	button_group.instantiate();

	free_paint_button = memnew(CheckBox);
	free_paint_button->set_text(TTRC("Free Paint"));
	free_paint_button->set_accessibility_name(TTRC("Free Paint"));
	free_paint_button->set_tooltip_text(TTRC("Toggle free painting without snapping."));
	free_paint_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_paint_mode_changed).bind(PAINT_MODE_FREE).unbind(1));
	free_paint_button->set_pressed(true);
	free_paint_button->set_button_group(button_group);
	vbc->add_child(free_paint_button);

	snap_cell_button = memnew(CheckBox);
	snap_cell_button->set_text(TTRC("Snap to Cell"));
	snap_cell_button->set_accessibility_name(TTRC("Snap to Cell"));
	snap_cell_button->set_tooltip_text(TTRC("Toggle snapping to cell when painting."));
	snap_cell_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_paint_mode_changed).bind(PAINT_MODE_SNAP_CELL).unbind(1));
	snap_cell_button->set_button_group(button_group);
	vbc->add_child(snap_cell_button);

	snap_grid_button = memnew(CheckBox);
	snap_grid_button->set_text(TTRC("Snap to Grid"));
	snap_grid_button->set_accessibility_name(TTRC("Snap to Grid"));
	snap_grid_button->set_tooltip_text(TTRC("Toggle snapping to grid when painting."));
	snap_grid_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_paint_mode_changed).bind(PAINT_MODE_SNAP_GRID).unbind(1));
	snap_grid_button->set_button_group(button_group);
	vbc->add_child(snap_grid_button);

	hsep = memnew(HSeparator);
	hsep->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hsep);

	allow_overlapping_button = memnew(CheckBox);
	allow_overlapping_button->set_text(TTRC("Allow Overlapping"));
	allow_overlapping_button->set_accessibility_name(TTRC("Allow Overlapping"));
	allow_overlapping_button->set_tooltip_text(TTRC("Allow painting over existing painted scenes."));
	allow_overlapping_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_allow_overlapping_toggled));
	vbc->add_child(allow_overlapping_button);
}

void ScenePaint2DEditorPlugin::_canvas_item_tool_changed(int p_tool) {
	scene_paint_2d_editor->is_tool_selected = (CanvasItemEditor::Tool)p_tool == CanvasItemEditor::TOOL_SCENE_PAINT;
	Node *selected_node = SceneTreeDock::get_singleton()->get_tree_editor()->get_selected();
	if (!selected_node) {
		make_visible(false);
		return;
	}
	scene_paint_2d_editor->_set_picked_scene(nullptr);
	scene_paint_2d_editor->_edit(Object::cast_to<Node2D>(selected_node));
	scene_paint_2d_editor->input_tool = ScenePaint2DEditor::InputTool::INPUT_TOOL_NONE;
	scene_paint_2d_editor->_edit_properties_toggled(false);
	make_visible(true);
}

void ScenePaint2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			CanvasItemEditor::get_singleton()->connect("canvas_item_tool_changed", callable_mp(this, &ScenePaint2DEditorPlugin::_canvas_item_tool_changed));
			CanvasItemEditor::get_singleton()->connect("snap_changed", callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_grid_step_changed));
			FileSystemDock::get_singleton()->get_tree_control()->connect(SceneStringName(gui_input), callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_file_system_input));
			FileSystemDock::get_singleton()->get_list_control()->connect(SceneStringName(gui_input), callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_file_system_input));
			SceneTreeDock::get_singleton()->get_tree_editor()->get_scene_tree()->connect(SceneStringName(gui_input), callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_scene_tree_input));
			EditorNode::get_singleton()->connect("scene_changed", callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_scene_changed));
			scene_paint_2d_editor->viewport = CanvasItemEditor::get_singleton()->get_viewport_control();
			scene_paint_2d_editor->viewport->connect(SceneStringName(draw), callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_update_draw_overlay));
			scene_paint_2d_editor->viewport->connect(SceneStringName(gui_input), callable_mp(scene_paint_2d_editor, &ScenePaint2DEditor::_gui_input_viewport));
		} break;
	}
}

void ScenePaint2DEditorPlugin::edit(Object *p_object) {
	scene_paint_2d_editor->_edit(p_object);
}

bool ScenePaint2DEditorPlugin::handles(Object *p_object) const {
	is_node_2d = bool(Object::cast_to<Node2D>(p_object));
	return is_node_2d;
}

void ScenePaint2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		scene_paint_2d_editor->_can_handle(is_node_2d, false);
	} else {
		scene_paint_2d_editor->_can_handle(is_node_2d, true);
	}
}

void ScenePaint2DEditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	scene_paint_2d_editor->forward_canvas_draw_over_viewport(p_overlay);
}

ScenePaint2DEditorPlugin::ScenePaint2DEditorPlugin() {
	scene_paint_2d_editor = memnew(ScenePaint2DEditor);
	EditorNode::get_singleton()->get_gui_base()->add_child(scene_paint_2d_editor);
	make_visible(false);
}
