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
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/2d/node_2d.h"
#include "scene/2d/tile_map_layer.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/separator.h"
#include "scene/gui/spin_box.h"
#include "scene/main/window.h"

void ScenePaintEditor::_draw() {
	if (!node || !snap_grid || !is_tool_selected) {
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

	int step_x = get_grid_step().x;
	int step_y = get_grid_step().y;

	int last_cell = 0;

	if (step_x > 0) {
		for (int i = 0; i < size.width; i++) {
			int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(i, 0)).x) / step_x));

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
			int cell = Math::fast_ftoi(Math::floor((xform.xform(Vector2(0, i)).y) / step_y));

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
	Vector2 cell_pos = _get_mouse_grid_cell();

	int step_x = get_grid_step().x;
	int step_y = get_grid_step().y;

	node->draw_rect(Rect2(cell_pos, Size2(step_x, step_y)), Color(1, 1, 1, 0.3));
}

void ScenePaintEditor::_update_draw() {
	if (!node || !snap_grid) {
		return;
	}
	node->queue_redraw();
}

Vector2 ScenePaintEditor::_get_mouse_grid_cell() {
	CanvasItemEditor *canvas = CanvasItemEditor::get_singleton();
	Transform2D xform = canvas->get_canvas_transform().affine_inverse();

	Vector2 mouse_canvas = canvas->get_viewport_control()->get_local_mouse_position();

	Vector2 mouse_world = xform.xform(mouse_canvas);

	Vector2 mouse_local = node->to_local(mouse_world);

	int step_x = get_grid_step().x;
	int step_y = get_grid_step().y;

	int cell_x = Math::floor(mouse_local.x / step_x);
	int cell_y = Math::floor(mouse_local.y / step_y);

	return Vector2(cell_x * step_x, cell_y * step_y);
}

void ScenePaintEditor::_snap_grid_toggled(bool p_toggled) {
	snap_grid = p_toggled;
	if (!node) {
		return;
	}
	node->queue_redraw();
}

void ScenePaintEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
		} break;

		case NOTIFICATION_EXIT_TREE: {
		} break;

		case NOTIFICATION_PROCESS: {
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			new_palette_button->set_button_icon(get_editor_theme_icon("New"));
			load_palette_button->set_button_icon(get_editor_theme_icon("Load"));
			save_palette_button->set_button_icon(get_editor_theme_icon("Save"));
			add_scene_button->set_button_icon(get_editor_theme_icon("Add"));
			open_scene_button->set_button_icon(get_editor_theme_icon("PlayScene"));
			remove_scene_button->set_button_icon(get_editor_theme_icon("Remove"));
			snap_grid_button->set_button_icon(get_editor_theme_icon("SnapGrid"));
		} break;

		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
		} break;
	}
}

void ScenePaintEditor::_bind_methods() {
}

void ScenePaintEditor::set_grid_step(const Size2i size) {
	grid_step_x->set_value(size.x);
	grid_step_y->set_value(size.y);
}

Size2i ScenePaintEditor::get_grid_step() const {
	return Size2i(grid_step_x->get_value(), grid_step_y->get_value());
}

ScenePaintEditor::ScenePaintEditor() {
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

	add_scene_button = memnew(Button);
	add_scene_button->set_accessibility_name(TTRC("Add Scene"));
	add_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	add_scene_button->set_tooltip_text(TTRC("Add a scene to the current palette."));
	toolbar->add_child(add_scene_button);

	open_scene_button = memnew(Button);
	open_scene_button->set_accessibility_name(TTRC("Open Scene"));
	open_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	open_scene_button->set_tooltip_text(TTRC("Open a scene from the current palette."));
	toolbar->add_child(open_scene_button);

	remove_scene_button = memnew(Button);
	remove_scene_button->set_accessibility_name(TTRC("Remove Scene"));
	remove_scene_button->set_theme_type_variation(SceneStringName(FlatButton));
	remove_scene_button->set_tooltip_text(TTRC("Remove a scene from the current palette."));
	toolbar->add_child(remove_scene_button);

	v_separator = memnew(VSeparator);
	toolbar->add_child(v_separator);

	snap_grid_button = memnew(Button);
	snap_grid_button->set_toggle_mode(true);
	snap_grid_button->set_accessibility_name(TTRC("Grid"));
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
	grid_step_x->set_value(8);
	grid_step_x->set_suffix("px");
	grid_step_x->set_allow_greater(true);
	grid_step_x->set_custom_minimum_size(Vector2(64, 0));
	grid_step_x->connect("value_changed", callable_mp(this, &ScenePaintEditor::_update_draw).unbind(1));
	toolbar->add_child(grid_step_x);

	grid_step_y = memnew(SpinBox);
	grid_step_y->set_min(1);
	grid_step_y->set_step(1);
	grid_step_y->set_value(8);
	grid_step_y->set_suffix("px");
	grid_step_y->set_allow_greater(true);
	grid_step_y->set_custom_minimum_size(Vector2(64, 0));
	grid_step_y->connect("value_changed", callable_mp(this, &ScenePaintEditor::_update_draw).unbind(1));
	toolbar->add_child(grid_step_y);

	item_list = memnew(ItemList);
	item_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	add_child(item_list);
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
			scene_paint_editor->viewport->connect("gui_input", callable_mp(scene_paint_editor, &ScenePaintEditor::_update_draw).unbind(1));

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
	}
}

bool ScenePaintEditorPlugin::handles(Object *p_object) const {
	is_node_2d = p_object->is_class("Node2D");
	return is_node_2d;
}

void ScenePaintEditorPlugin::make_visible(bool p_visible) {
	ERR_FAIL_NULL(scene_paint_editor);
	if (p_visible && is_node_2d && scene_paint_editor->is_tool_selected) {
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
