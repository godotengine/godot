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
				set_grid_step(tile_set->get_tile_size());
			}
		}
	}

	cache_node = Object::cast_to<Node2D>(p_object);
	if (_is_node_valid() && (pinned || scene_picker)) {
		if (scene_picker) {
			SceneTreeDock::get_singleton()->set_selection(Vector<Node *>{ node });
		}
		_set_input_tool(false, false);
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
	_set_input_tool(false, false);
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
			Color rect_color = is_erasing ? Color(0, 0, 0, 0.3) : Color(1, 1, 1, 0.3);
			Vector<Vector2> points;
			points.push_back(corners[0]);
			points.push_back(corners[1]);
			points.push_back(corners[2]);
			points.push_back(corners[3]);
			custom_overlay->draw_polygon(points, Vector<Color>({ rect_color }));
		}
	}
	// instance
	if (_is_instance_valid() && !is_erasing && !scene_picker && !viewport_scene_picker) {
		_add_instance(true);
	} else if (!_is_instance_valid() || is_erasing || scene_picker || viewport_scene_picker) {
		_clear_instance(true);
	}
	if (instance) {
		CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
		Transform2D xform = canvas_item_editor->get_canvas_transform() * node->get_global_transform();
		Vector2 mouse_canvas = viewport->get_local_mouse_position();
		Vector2 mouse_local = xform.affine_inverse().xform(mouse_canvas);
		Vector2 final_local;
		if (paint_mode != PAINT_MODE_FREE) {
			Vector2 snapped_local = Vector2(
					Math::floor(mouse_local.x / grid_step.x) * grid_step.x,
					Math::floor(mouse_local.y / grid_step.y) * grid_step.y);
			final_local = (paint_mode != PAINT_MODE_FREE)
					? (paint_mode == PAINT_MODE_SNAP_GRID ? snapped_local : (snapped_local + grid_step / 2.0))
					: mouse_local;
		} else {
			final_local = mouse_local;
		}
		instance_container->set_position(xform.xform(final_local + paint_offset));
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

void ScenePaint2DEditor::_set_input_tool(bool p_painting, bool p_erasing) {
	is_painting = p_painting;
	is_erasing = p_erasing;
}

void ScenePaint2DEditor::_gui_input_viewport(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree() || !is_tool_selected) {
		return;
	}

	// Hack: Ignore accidentally painting while panning with 'pan_view'. When holding 'pan_view', the tool doesn't change.
	if (ED_IS_SHORTCUT("canvas_item_editor/pan_view", p_event)) {
		is_panning = p_event->is_pressed();
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->get_keycode() == Key::CMD_OR_CTRL) {
		viewport_scene_picker = k->is_pressed();
		scene_picker_button->set_pressed(viewport_scene_picker);
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && !is_panning) {
		if (scene_picker) {
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
				}
				_set_input_tool(true, false);
				_add_node_at_pos();
				accept_event();
			} else if (mb->get_button_index() == MouseButton::RIGHT) {
				if (mb->is_double_click()) {
					_set_picked_scene(nullptr);
					return;
				}
				_set_input_tool(false, true);
				_remove_node_at_pos();
				accept_event();
			}
		} else if (mb->is_released()) {
			if (mb->get_button_index() == MouseButton::LEFT) {
				_set_input_tool(false, false);
			} else if (mb->get_button_index() == MouseButton::RIGHT) {
				_set_input_tool(false, false);
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && !is_panning) {
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
	Vector<CanvasItemEditor::SelectResult> results;
	canvas_item_editor->find_canvas_items_at_pos(pos, node, results);
	for (const CanvasItemEditor::SelectResult &result : results) {
		Node2D *root = _get_node_root(result.item);
		if (_is_scene_painted(root)) {
			return;
		}
	}

	if (!FileAccess::exists(selected_scene->get_scene_file_path())) {
		EditorNode::get_singleton()->show_warning(
				vformat("The selected scene '%s' no longer exists on disk. Please select a valid scene to paint.",
						selected_scene->get_scene_file_path()));
		_set_picked_scene(nullptr);
		return;
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
		node_2d->set_global_position(pos + paint_offset);

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Paint Node(s)"), UndoRedo::MERGE_ALL);
		undo_redo->add_do_reference(node_2d);
		undo_redo->add_do_method(node, "add_child", node_2d, true);
		undo_redo->add_do_method(node_2d, "set_owner", scene);
		undo_redo->add_do_method(node_2d, "set_meta", "_scene_painted", true);
		undo_redo->add_do_method(node_2d, "set_global_position", pos + paint_offset);
		undo_redo->add_undo_method(node, "remove_child", node_2d);
		undo_redo->commit_action(false);
	}
}

void ScenePaint2DEditor::_remove_node_at_pos() {
	if (!_is_node_valid() || is_painting) {
		return;
	}
	CanvasItemEditor *canvas_item_editor = CanvasItemEditor::get_singleton();
	Vector2 pos = canvas_item_editor->get_canvas_transform().affine_inverse().xform(viewport->get_local_mouse_position());

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
	Vector2 snapped = Vector2(
			Math::floor(local.x / grid_step.x) * grid_step.x,
			Math::floor(local.y / grid_step.y) * grid_step.y);

	return node->get_global_transform().xform(snapped);
}

void ScenePaint2DEditor::_edit_properties() {
	if (!_is_instance_valid() || !edit_properties) {
		return;
	}
	InspectorDock::get_inspector_singleton()->edit(instance);
}

void ScenePaint2DEditor::_scene_picker_toggled(bool p_pressed) {
	scene_picker = p_pressed;
	_set_input_tool(false, false);
}

void ScenePaint2DEditor::_file_system_input(const Ref<InputEvent> &p_event) {
	if (!scene_picker || viewport_scene_picker) {
		viewport_scene_picker = false;
		scene_picker_button->set_pressed(viewport_scene_picker);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_FILE_SYSTEM);
	}
}

void ScenePaint2DEditor::_scene_tree_input(const Ref<InputEvent> &p_event) {
	if (!scene_picker || viewport_scene_picker) {
		viewport_scene_picker = false;
		scene_picker_button->set_pressed(viewport_scene_picker);
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_released() && mb->get_button_index() == MouseButton::LEFT) {
		callable_mp(this, &ScenePaint2DEditor::_update_scene_picker).call_deferred(PICK_SCENE_TREE);
	}
}

void ScenePaint2DEditor::_recent_id_pressed(int p_id) {
	recent_id = p_id;
	if (recent_id == recent_scenes_popup->get_item_count() - 1) {
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
	_set_input_tool(false, false);
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
			String scene_path = recent_scenes_popup->get_item_metadata(recent_id);
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

void ScenePaint2DEditor::_grid_settings_pressed() {
	Vector2 pos = grid_settings_button->get_screen_position() + grid_settings_button->get_size();
	grid_settings_popup->set_position(pos - Vector2(grid_settings_popup->get_contents_minimum_size().width / 2, 0));
	grid_settings_popup->reset_size();
	grid_settings_popup->popup();
	grid_settings_popup->grab_focus();
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
	grid_step.x = grid_step_x->get_value();
	grid_step.y = grid_step_y->get_value();
	set_grid_step(grid_step);
}

void ScenePaint2DEditor::_offset_changed() {
	paint_offset.x = offset_x->get_value();
	paint_offset.y = offset_y->get_value();
	set_offset(paint_offset);
}

void ScenePaint2DEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			pin_node_button->set_button_icon(get_editor_theme_icon(SNAME("Pin")));
			scene_picker_button->set_button_icon(get_editor_theme_icon(SNAME("ColorPick")));
			edit_properties_button->set_button_icon(get_editor_theme_icon(SNAME("Tools")));
			grid_settings_button->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
			grid_settings_popup->add_theme_style_override(SceneStringName(panel),
					get_theme_stylebox(SceneStringName(panel), SNAME("AcceptDialog")));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			_update_draw_overlay();
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			_set_input_tool(false, false);
		} break;
	}
}

void ScenePaint2DEditor::_set_picked_scene(Node2D *p_scene) {
	selected_scene = p_scene;
	scene_picker_button->set_pressed(false);
	recent_scenes_popup->set_visible(false);
	_set_input_tool(false, false);
	String scene_path = selected_scene ? selected_scene->get_scene_file_path() : String();
	callable_mp(this, &ScenePaint2DEditor::_add_to_recent_scenes).call_deferred(scene_path);
	callable_mp(this, &ScenePaint2DEditor::_update_instance).call_deferred();
}

void ScenePaint2DEditor::_picker_button_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
		_update_recent_scenes();
		Vector2 pos = scene_picker_button->get_screen_position() + scene_picker_button->get_size();
		recent_scenes_popup->set_position(pos - Vector2(recent_scenes_popup->get_contents_minimum_size().width / 2, 0));
		recent_scenes_popup->reset_size();
		recent_scenes_popup->popup();
		recent_scenes_popup->grab_focus();
		accept_event();
	}
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
	recent_scenes_popup->clear();

	if (rc.is_empty()) {
		recent_scenes_popup->add_item(TTRC("No Recent Scenes"), -1);
		recent_scenes_popup->set_item_disabled(-1, true);
	} else {
		for (const String &uid : rc) {
			String path = ResourceUID::ensure_path(uid);
			recent_scenes_popup->add_item(path.get_file());
			recent_scenes_popup->set_item_tooltip(-1, path);
			recent_scenes_popup->set_item_metadata(-1, uid);
		}
		recent_scenes_popup->add_separator();
		recent_scenes_popup->add_item(TTRC("Clear Recent Scenes"), -1);
	}
	recent_scenes_popup->set_item_auto_translate_mode(-1, AUTO_TRANSLATE_MODE_ALWAYS);
	recent_scenes_popup->reset_size();
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

void ScenePaint2DEditor::set_grid_step(const Size2i &p_size) {
	grid_step = p_size;
	grid_step_x->set_value_no_signal(grid_step.x);
	grid_step_y->set_value_no_signal(grid_step.y);
	_update_draw_overlay();
	EditorSettings::get_singleton()->set_project_metadata("scene_paint_2d_editor", "grid_step", grid_step);
}

void ScenePaint2DEditor::set_offset(const Point2 &p_offset) {
	paint_offset = p_offset;
	offset_x->set_value_no_signal(paint_offset.x);
	offset_y->set_value_no_signal(paint_offset.y);
	EditorSettings::get_singleton()->set_project_metadata("scene_paint_2d_editor", "offset", paint_offset);
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
	scene_picker_button->connect(SceneStringName(gui_input), callable_mp(this, &ScenePaint2DEditor::_picker_button_gui_input));
	scene_picker_button->set_shortcut(ED_SHORTCUT("scene_painter/scene_picker", TTRC("Scene Picker"), Key::I));
	scene_picker_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(scene_picker_button);

	recent_scenes_popup = memnew(PopupMenu);
	recent_scenes_popup->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	recent_scenes_popup->connect(SceneStringName(id_pressed), callable_mp(this, &ScenePaint2DEditor::_recent_id_pressed));
	add_child(recent_scenes_popup);

	edit_properties_button = memnew(Button);
	edit_properties_button->set_toggle_mode(true);
	edit_properties_button->set_accessibility_name(TTRC("Edit Properties"));
	edit_properties_button->set_theme_type_variation(SceneStringName(FlatButton));
	edit_properties_button->set_tooltip_text(TTRC("Edit properties of the selected scene."));
	edit_properties_button->connect(SceneStringName(toggled), callable_mp(this, &ScenePaint2DEditor::_edit_properties_toggled));
	edit_properties_button->set_shortcut(ED_SHORTCUT("scene_painter/edit_properties", TTRC("Edit Properties"), Key::O));
	edit_properties_button->set_shortcut_context(CanvasItemEditor::get_singleton());
	toolbar->add_child(edit_properties_button);

	grid_settings_button = memnew(Button);
	grid_settings_button->set_accessibility_name(TTRC("Grid Settings"));
	grid_settings_button->set_theme_type_variation(SceneStringName(FlatButton));
	grid_settings_button->set_tooltip_text(TTRC("Open grid settings."));
	grid_settings_button->connect(SceneStringName(pressed), callable_mp(this, &ScenePaint2DEditor::_grid_settings_pressed));
	toolbar->add_child(grid_settings_button);

	grid_settings_popup = memnew(PopupPanel);
	add_child(grid_settings_popup);

	VBoxContainer *vbc = memnew(VBoxContainer);
	grid_settings_popup->add_child(vbc);

	grid_toggle_button = memnew(CheckBox);
	grid_toggle_button->set_text(TTRC("Show Grid"));
	grid_toggle_button->set_accessibility_name(TTRC("Show Grid"));
	grid_toggle_button->set_tooltip_text(TTRC("Toggle grid visibility in the viewport."));
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

	Label *grid_label = memnew(Label);
	grid_label->set_text(TTRC("Grid Step:"));
	vbc->add_child(grid_label);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	grid_step_x = memnew(SpinBox);
	grid_step_x->set_min(1);
	grid_step_x->set_suffix("px");
	grid_step_x->set_allow_greater(true);
	grid_step_x->set_select_all_on_focus(true);
	grid_step_x->set_custom_minimum_size(Vector2(64 * EDSCALE, 0));
	grid_step_x->connect(SceneStringName(value_changed), callable_mp(this, &ScenePaint2DEditor::_grid_step_changed).unbind(1));
	hbc->add_child(grid_step_x);

	grid_step_y = memnew(SpinBox);
	grid_step_y->set_min(1);
	grid_step_y->set_suffix("px");
	grid_step_y->set_allow_greater(true);
	grid_step_y->set_select_all_on_focus(true);
	grid_step_y->set_custom_minimum_size(Vector2(64 * EDSCALE, 0));
	grid_step_y->connect(SceneStringName(value_changed), callable_mp(this, &ScenePaint2DEditor::_grid_step_changed).unbind(1));
	hbc->add_child(grid_step_y);

	hsep = memnew(HSeparator);
	hsep->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vbc->add_child(hsep);

	Label *offset_label = memnew(Label);
	offset_label->set_text(TTRC("Offset:"));
	vbc->add_child(offset_label);

	hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	offset_x = memnew(SpinBox);
	offset_x->set_suffix("px");
	offset_x->set_allow_lesser(true);
	offset_x->set_allow_greater(true);
	offset_x->set_select_all_on_focus(true);
	offset_x->set_custom_minimum_size(Vector2(64 * EDSCALE, 0));
	offset_x->connect(SceneStringName(value_changed), callable_mp(this, &ScenePaint2DEditor::_offset_changed).unbind(1));
	hbc->add_child(offset_x);

	offset_y = memnew(SpinBox);
	offset_y->set_suffix("px");
	offset_y->set_allow_lesser(true);
	offset_y->set_allow_greater(true);
	offset_y->set_select_all_on_focus(true);
	offset_y->set_custom_minimum_size(Vector2(64 * EDSCALE, 0));
	offset_y->connect(SceneStringName(value_changed), callable_mp(this, &ScenePaint2DEditor::_offset_changed).unbind(1));
	hbc->add_child(offset_y);

	set_grid_step(EditorSettings::get_singleton()->get_project_metadata("scene_paint_2d_editor", "grid_step", Size2i(16, 16)));
	set_offset(EditorSettings::get_singleton()->get_project_metadata("scene_paint_2d_editor", "offset", Size2i()));
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
	scene_paint_2d_editor->_set_input_tool(false, false);
	scene_paint_2d_editor->_edit_properties_toggled(false);
	make_visible(true);
}

void ScenePaint2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			CanvasItemEditor::get_singleton()->connect("canvas_item_tool_changed", callable_mp(this, &ScenePaint2DEditorPlugin::_canvas_item_tool_changed));
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
