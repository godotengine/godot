/**************************************************************************/
/*  tile_map_editor.cpp                                                   */
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

#include "tile_map_editor.h"

#include "tiles_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_resource_preview.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/canvas_item_editor_plugin.h"

#include "scene/2d/camera_2d.h"
#include "scene/gui/center_container.h"
#include "scene/gui/split_container.h"

#include "core/input/input.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"

void TileMapEditorTilesPlugin::tile_set_changed() {
	_update_fix_selected_and_hovered();
	_update_tile_set_sources_list();
	_update_source_display();
	_update_patterns_list();
}

void TileMapEditorTilesPlugin::_on_random_tile_checkbox_toggled(bool p_pressed) {
	scatter_controls_container->set_visible(p_pressed);
}

void TileMapEditorTilesPlugin::_on_scattering_spinbox_changed(double p_value) {
	scattering = p_value;
}

void TileMapEditorTilesPlugin::_update_toolbar() {
	// Stop draggig if needed.
	_stop_dragging();

	// Hide all settings.
	for (int i = 0; i < tools_settings->get_child_count(); i++) {
		Object::cast_to<CanvasItem>(tools_settings->get_child(i))->hide();
	}

	// Show only the correct settings.
	if (tool_buttons_group->get_pressed_button() == select_tool_button) {
		transform_toolbar->show();
	} else if (tool_buttons_group->get_pressed_button() != bucket_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		transform_toolbar->show();
		tools_settings_vsep_2->show();
		random_tile_toggle->show();
		scatter_label->show();
		scatter_spinbox->show();
	} else {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		transform_toolbar->show();
		tools_settings_vsep_2->show();
		bucket_contiguous_checkbox->show();
		random_tile_toggle->show();
		scatter_label->show();
		scatter_spinbox->show();
	}
}

void TileMapEditorTilesPlugin::_update_transform_buttons() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (tile_set.is_null() || selection_pattern.is_null()) {
		return;
	}

	bool has_scene_tile = false;
	for (const KeyValue<Vector2i, TileMapCell> &E : selection_pattern->get_pattern()) {
		if (Object::cast_to<TileSetScenesCollectionSource>(tile_set->get_source(E.value.source_id).ptr())) {
			has_scene_tile = true;
			break;
		}
	}

	if (has_scene_tile) {
		_set_transform_buttons_state({}, { transform_button_rotate_left, transform_button_rotate_right, transform_button_flip_h, transform_button_flip_v },
				TTR("Can't transform scene tiles."));
	} else if (tile_set->get_tile_shape() != TileSet::TILE_SHAPE_SQUARE && selection_pattern->get_size() != Vector2i(1, 1)) {
		_set_transform_buttons_state({ transform_button_flip_h, transform_button_flip_v }, { transform_button_rotate_left, transform_button_rotate_right },
				TTR("Can't rotate patterns when using non-square tile grid."));
	} else {
		_set_transform_buttons_state({ transform_button_rotate_left, transform_button_rotate_right, transform_button_flip_h, transform_button_flip_v }, {}, "");
	}
}

void TileMapEditorTilesPlugin::_set_transform_buttons_state(const Vector<Button *> &p_enabled_buttons, const Vector<Button *> &p_disabled_buttons, const String &p_why_disabled) {
	for (Button *button : p_enabled_buttons) {
		button->set_disabled(false);
		button->set_tooltip_text("");
	}
	for (Button *button : p_disabled_buttons) {
		button->set_disabled(true);
		button->set_tooltip_text(p_why_disabled);
	}
}

Vector<TileMapSubEditorPlugin::TabData> TileMapEditorTilesPlugin::get_tabs() const {
	Vector<TileMapSubEditorPlugin::TabData> tabs;
	tabs.push_back({ toolbar, tiles_bottom_panel });
	tabs.push_back({ toolbar, patterns_bottom_panel });
	return tabs;
}

void TileMapEditorTilesPlugin::_tab_changed() {
	if (tiles_bottom_panel->is_visible_in_tree()) {
		_update_selection_pattern_from_tileset_tiles_selection();
	} else if (patterns_bottom_panel->is_visible_in_tree()) {
		_update_selection_pattern_from_tileset_pattern_selection();
	}
}

void TileMapEditorTilesPlugin::_update_tile_set_sources_list() {
	// Update the sources.
	int old_current = sources_list->get_current();
	int old_source = -1;
	if (old_current > -1) {
		old_source = sources_list->get_item_metadata(old_current);
	}
	sources_list->clear();

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (!tile_set->has_source(old_source)) {
		old_source = -1;
	}

	List<int> source_ids = TilesEditorUtils::get_singleton()->get_sorted_sources(tile_set);
	for (const int &source_id : source_ids) {
		TileSetSource *source = *tile_set->get_source(source_id);

		Ref<Texture2D> texture;
		String item_text;

		// Common to all type of sources.
		if (!source->get_name().is_empty()) {
			item_text = source->get_name();
		}

		// Atlas source.
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			texture = atlas_source->get_texture();
			if (item_text.is_empty()) {
				if (texture.is_valid()) {
					item_text = texture->get_path().get_file();
				} else {
					item_text = vformat(TTR("No Texture Atlas Source (ID: %d)"), source_id);
				}
			}
		}

		// Scene collection source.
		TileSetScenesCollectionSource *scene_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
		if (scene_collection_source) {
			texture = tiles_bottom_panel->get_editor_theme_icon(SNAME("PackedScene"));
			if (item_text.is_empty()) {
				if (scene_collection_source->get_scene_tiles_count() > 0) {
					item_text = vformat(TTR("Scene Collection Source (ID: %d)"), source_id);
				} else {
					item_text = vformat(TTR("Empty Scene Collection Source (ID: %d)"), source_id);
				}
			}
		}

		// Use default if not valid.
		if (item_text.is_empty()) {
			item_text = vformat(TTR("Unknown Type Source (ID: %d)"), source_id);
		}
		if (!texture.is_valid()) {
			texture = missing_atlas_texture_icon;
		}

		sources_list->add_item(item_text, texture);
		sources_list->set_item_metadata(-1, source_id);
	}

	if (sources_list->get_item_count() > 0) {
		if (old_source >= 0) {
			for (int i = 0; i < sources_list->get_item_count(); i++) {
				if ((int)sources_list->get_item_metadata(i) == old_source) {
					sources_list->set_current(i);
					sources_list->ensure_current_is_visible();
					break;
				}
			}
		} else {
			sources_list->set_current(0);
		}
		sources_list->emit_signal(SNAME("item_selected"), sources_list->get_current());
	}

	// Synchronize the lists.
	TilesEditorUtils::get_singleton()->set_sources_lists_current(sources_list->get_current());
}

void TileMapEditorTilesPlugin::_update_source_display() {
	// Update the atlas display.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index >= 0 && source_index < sources_list->get_item_count()) {
		atlas_sources_split_container->show();
		missing_source_label->hide();

		int source_id = sources_list->get_item_metadata(source_index);
		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);

		if (atlas_source) {
			tile_atlas_view->show();
			scene_tiles_list->hide();
			invalid_source_label->hide();
			_update_atlas_view();
		} else if (scenes_collection_source) {
			tile_atlas_view->hide();
			scene_tiles_list->show();
			invalid_source_label->hide();
			_update_scenes_collection_view();
		} else {
			tile_atlas_view->hide();
			scene_tiles_list->hide();
			invalid_source_label->show();
		}
	} else {
		atlas_sources_split_container->hide();
		missing_source_label->show();

		tile_atlas_view->hide();
		scene_tiles_list->hide();
		invalid_source_label->hide();
	}
}

void TileMapEditorTilesPlugin::_patterns_item_list_gui_input(const Ref<InputEvent> &p_event) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid() || EditorNode::get_singleton()->is_resource_read_only(tile_set)) {
		return;
	}

	if (ED_IS_SHORTCUT("tiles_editor/paste", p_event) && p_event->is_pressed() && !p_event->is_echo()) {
		select_last_pattern = true;
		int new_pattern_index = tile_set->get_patterns_count();
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Add TileSet pattern"));
		undo_redo->add_do_method(*tile_set, "add_pattern", tile_map_clipboard, new_pattern_index);
		undo_redo->add_undo_method(*tile_set, "remove_pattern", new_pattern_index);
		undo_redo->commit_action();
		patterns_item_list->accept_event();
	}

	if (ED_IS_SHORTCUT("tiles_editor/delete", p_event) && p_event->is_pressed() && !p_event->is_echo()) {
		Vector<int> selected = patterns_item_list->get_selected_items();
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Remove TileSet patterns"));
		for (int i = 0; i < selected.size(); i++) {
			int pattern_index = selected[i];
			undo_redo->add_do_method(*tile_set, "remove_pattern", pattern_index);
			undo_redo->add_undo_method(*tile_set, "add_pattern", tile_set->get_pattern(pattern_index), pattern_index);
		}
		undo_redo->commit_action();
		patterns_item_list->accept_event();
	}
}

void TileMapEditorTilesPlugin::_pattern_preview_done(Ref<TileMapPattern> p_pattern, Ref<Texture2D> p_texture) {
	// TODO optimize ?
	for (int i = 0; i < patterns_item_list->get_item_count(); i++) {
		if (patterns_item_list->get_item_metadata(i) == p_pattern) {
			patterns_item_list->set_item_icon(i, p_texture);
			break;
		}
	}
}

void TileMapEditorTilesPlugin::_update_patterns_list() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Recreate the items.
	patterns_item_list->clear();
	for (int i = 0; i < tile_set->get_patterns_count(); i++) {
		int id = patterns_item_list->add_item("");
		patterns_item_list->set_item_metadata(id, tile_set->get_pattern(i));
		patterns_item_list->set_item_tooltip(id, vformat(TTR("Index: %d"), i));
		TilesEditorUtils::get_singleton()->queue_pattern_preview(tile_set, tile_set->get_pattern(i), callable_mp(this, &TileMapEditorTilesPlugin::_pattern_preview_done));
	}

	// Update the label visibility.
	patterns_help_label->set_visible(patterns_item_list->get_item_count() == 0);

	// Added a new pattern, thus select the last one.
	if (select_last_pattern) {
		patterns_item_list->select(tile_set->get_patterns_count() - 1);
		patterns_item_list->grab_focus();
		_update_selection_pattern_from_tileset_pattern_selection();
	}
	select_last_pattern = false;
}

void TileMapEditorTilesPlugin::_update_atlas_view() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(sources_list->get_current());
	TileSetSource *source = *tile_set->get_source(source_id);
	TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
	ERR_FAIL_NULL(atlas_source);

	tile_atlas_view->set_atlas_source(*tile_map->get_tileset(), atlas_source, source_id);
	TilesEditorUtils::get_singleton()->synchronize_atlas_view(tile_atlas_view);
	tile_atlas_control->queue_redraw();
}

void TileMapEditorTilesPlugin::_update_scenes_collection_view() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(sources_list->get_current());
	TileSetSource *source = *tile_set->get_source(source_id);
	TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
	ERR_FAIL_NULL(scenes_collection_source);

	// Clear the list.
	scene_tiles_list->clear();

	// Rebuild the list.
	for (int i = 0; i < scenes_collection_source->get_scene_tiles_count(); i++) {
		int scene_id = scenes_collection_source->get_scene_tile_id(i);

		Ref<PackedScene> scene = scenes_collection_source->get_scene_tile_scene(scene_id);

		int item_index = 0;
		if (scene.is_valid()) {
			item_index = scene_tiles_list->add_item(vformat("%s (Path: %s, ID: %d)", scene->get_path().get_file().get_basename(), scene->get_path(), scene_id));
			Variant udata = i;
			EditorResourcePreview::get_singleton()->queue_edited_resource_preview(scene, this, "_scene_thumbnail_done", udata);
		} else {
			item_index = scene_tiles_list->add_item(TTR("Tile with Invalid Scene"), tiles_bottom_panel->get_editor_theme_icon(SNAME("PackedScene")));
		}
		scene_tiles_list->set_item_metadata(item_index, scene_id);

		// Check if in selection.
		if (tile_set_selection.has(TileMapCell(source_id, Vector2i(), scene_id))) {
			scene_tiles_list->select(item_index, false);
		}
	}
	if (scene_tiles_list->get_item_count() == 0) {
		scene_tiles_list->add_item(TTR("The selected scene collection source has no scenes. Add scenes in the TileSet bottom tab."));
		scene_tiles_list->set_item_disabled(-1, true);
	}

	// Icon size update.
	int int_size = int(EDITOR_GET("filesystem/file_dialog/thumbnail_size")) * EDSCALE;
	scene_tiles_list->set_fixed_icon_size(Vector2(int_size, int_size));
}

void TileMapEditorTilesPlugin::_scene_thumbnail_done(const String &p_path, const Ref<Texture2D> &p_preview, const Ref<Texture2D> &p_small_preview, Variant p_ud) {
	int index = p_ud;

	if (index >= 0 && index < scene_tiles_list->get_item_count()) {
		scene_tiles_list->set_item_icon(index, p_preview);
	}
}

void TileMapEditorTilesPlugin::_scenes_list_multi_selected(int p_index, bool p_selected) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Add or remove the Tile form the selection.
	int scene_id = scene_tiles_list->get_item_metadata(p_index);
	int source_id = sources_list->get_item_metadata(sources_list->get_current());
	TileSetSource *source = *tile_set->get_source(source_id);
	TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
	ERR_FAIL_NULL(scenes_collection_source);

	TileMapCell selected = TileMapCell(source_id, Vector2i(), scene_id);

	// Clear the selection if shift is not pressed.
	if (!Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
		tile_set_selection.clear();
	}

	if (p_selected) {
		tile_set_selection.insert(selected);
	} else {
		if (tile_set_selection.has(selected)) {
			tile_set_selection.erase(selected);
		}
	}

	_update_selection_pattern_from_tileset_tiles_selection();
}

void TileMapEditorTilesPlugin::_scenes_list_lmb_empty_clicked(const Vector2 &p_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index != MouseButton::LEFT) {
		return;
	}

	scene_tiles_list->deselect_all();
	tile_set_selection.clear();
	tile_map_selection.clear();
	selection_pattern.instantiate();
	_update_selection_pattern_from_tileset_tiles_selection();
}

void TileMapEditorTilesPlugin::_update_theme() {
	source_sort_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Sort")));
	select_tool_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("ToolSelect")));
	paint_tool_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Edit")));
	line_tool_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Line")));
	rect_tool_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Rectangle")));
	bucket_tool_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Bucket")));

	picker_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("ColorPick")));
	erase_button->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("Eraser")));
	random_tile_toggle->set_icon(tiles_bottom_panel->get_editor_theme_icon(SNAME("RandomNumberGenerator")));

	transform_button_rotate_left->set_icon(tiles_bottom_panel->get_editor_theme_icon("RotateLeft"));
	transform_button_rotate_right->set_icon(tiles_bottom_panel->get_editor_theme_icon("RotateRight"));
	transform_button_flip_h->set_icon(tiles_bottom_panel->get_editor_theme_icon("MirrorX"));
	transform_button_flip_v->set_icon(tiles_bottom_panel->get_editor_theme_icon("MirrorY"));

	missing_atlas_texture_icon = tiles_bottom_panel->get_editor_theme_icon(SNAME("TileSet"));
	_update_tile_set_sources_list();
}

bool TileMapEditorTilesPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!(tiles_bottom_panel->is_visible_in_tree() || patterns_bottom_panel->is_visible_in_tree())) {
		// If the bottom editor is not visible, we ignore inputs.
		return false;
	}

	if (CanvasItemEditor::get_singleton()->get_current_tool() != CanvasItemEditor::TOOL_SELECT) {
		return false;
	}

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return false;
	}

	if (tile_map_layer < 0) {
		return false;
	}
	ERR_FAIL_INDEX_V(tile_map_layer, tile_map->get_layers_count(), false);

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return false;
	}

	// Shortcuts
	if (ED_IS_SHORTCUT("tiles_editor/cut", p_event) || ED_IS_SHORTCUT("tiles_editor/copy", p_event)) {
		// Fill in the clipboard.
		if (!tile_map_selection.is_empty()) {
			tile_map_clipboard.instantiate();
			TypedArray<Vector2i> coords_array;
			for (const Vector2i &E : tile_map_selection) {
				coords_array.push_back(E);
			}
			tile_map_clipboard = tile_map->get_pattern(tile_map_layer, coords_array);
		}

		if (ED_IS_SHORTCUT("tiles_editor/cut", p_event)) {
			// Delete selected tiles.
			if (!tile_map_selection.is_empty()) {
				EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
				undo_redo->create_action(TTR("Delete tiles"));
				for (const Vector2i &E : tile_map_selection) {
					undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
					undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E, tile_map->get_cell_source_id(tile_map_layer, E), tile_map->get_cell_atlas_coords(tile_map_layer, E), tile_map->get_cell_alternative_tile(tile_map_layer, E));
				}
				undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());
				tile_map_selection.clear();
				undo_redo->add_do_method(this, "_set_tile_map_selection", _get_tile_map_selection());
				undo_redo->commit_action();
			}
		}

		return true;
	}
	if (ED_IS_SHORTCUT("tiles_editor/paste", p_event)) {
		if (drag_type == DRAG_TYPE_NONE) {
			drag_type = DRAG_TYPE_CLIPBOARD_PASTE;
		}
		CanvasItemEditor::get_singleton()->update_viewport();
		return true;
	}
	if (ED_IS_SHORTCUT("tiles_editor/cancel", p_event)) {
		if (drag_type == DRAG_TYPE_CLIPBOARD_PASTE) {
			drag_type = DRAG_TYPE_NONE;
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
	}
	if (ED_IS_SHORTCUT("tiles_editor/delete", p_event)) {
		// Delete selected tiles.
		if (!tile_map_selection.is_empty()) {
			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Delete tiles"));
			for (const Vector2i &E : tile_map_selection) {
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E, tile_map->get_cell_source_id(tile_map_layer, E), tile_map->get_cell_atlas_coords(tile_map_layer, E), tile_map->get_cell_alternative_tile(tile_map_layer, E));
			}
			undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());
			tile_map_selection.clear();
			undo_redo->add_do_method(this, "_set_tile_map_selection", _get_tile_map_selection());
			undo_redo->commit_action();
		}
		return true;
	}

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !k->is_echo()) {
		for (BaseButton *b : viewport_shortcut_buttons) {
			if (b->is_disabled()) {
				continue;
			}

			if (b->get_shortcut().is_valid() && b->get_shortcut()->matches_event(p_event)) {
				if (b->is_toggle_mode()) {
					b->set_pressed(b->get_button_group().is_valid() || !b->is_pressed());
				} else {
					// Can't press a button without toggle mode, so just emit the signal directly.
					b->emit_signal(SNAME("pressed"));
				}
				return true;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		has_mouse = true;
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
		Vector2 mpos = xform.affine_inverse().xform(mm->get_position());

		switch (drag_type) {
			case DRAG_TYPE_PAINT: {
				HashMap<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, drag_last_mouse_pos, mpos, drag_erasing);
				for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
					Vector2i coords = E.key;
					if (!drag_modified.has(coords)) {
						drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
						if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
							continue;
						}
						tile_map->set_cell(tile_map_layer, coords, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
					}
				}
				_fix_invalid_tiles_in_tile_map_selection();
			} break;
			case DRAG_TYPE_BUCKET: {
				Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->local_to_map(drag_last_mouse_pos), tile_map->local_to_map(mpos));
				for (int i = 0; i < line.size(); i++) {
					if (!drag_modified.has(line[i])) {
						HashMap<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], bucket_contiguous_checkbox->is_pressed(), drag_erasing);
						for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
							Vector2i coords = E.key;
							if (!drag_modified.has(coords)) {
								drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
								if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
									continue;
								}
								tile_map->set_cell(tile_map_layer, coords, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
							}
						}
					}
				}
				_fix_invalid_tiles_in_tile_map_selection();
			} break;
			default:
				break;
		}
		drag_last_mouse_pos = mpos;
		CanvasItemEditor::get_singleton()->update_viewport();

		return true;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		has_mouse = true;
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
		Vector2 mpos = xform.affine_inverse().xform(mb->get_position());

		if (mb->get_button_index() == MouseButton::LEFT || mb->get_button_index() == MouseButton::RIGHT) {
			if (mb->is_pressed()) {
				// Pressed
				if (erase_button->is_pressed() || mb->get_button_index() == MouseButton::RIGHT) {
					drag_erasing = true;
				}

				if (drag_type == DRAG_TYPE_CLIPBOARD_PASTE) {
					// Cancel tile pasting on right-click
					if (mb->get_button_index() == MouseButton::RIGHT) {
						drag_type = DRAG_TYPE_NONE;
					}
				} else if (tool_buttons_group->get_pressed_button() == select_tool_button) {
					drag_start_mouse_pos = mpos;
					if (tile_map_selection.has(tile_map->local_to_map(drag_start_mouse_pos)) && !mb->is_shift_pressed() && !mb->is_command_or_control_pressed()) {
						// Move the selection
						_update_selection_pattern_from_tilemap_selection(); // Make sure the pattern is up to date before moving.
						drag_type = DRAG_TYPE_MOVE;
						drag_modified.clear();
						for (const Vector2i &E : tile_map_selection) {
							Vector2i coords = E;
							drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
							tile_map->set_cell(tile_map_layer, coords, TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
						}
					} else {
						// Select tiles
						drag_type = DRAG_TYPE_SELECT;
					}
				} else {
					// Check if we are picking a tile.
					if (picker_button->is_pressed() || (Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT))) {
						drag_type = DRAG_TYPE_PICK;
						drag_start_mouse_pos = mpos;
					} else {
						// Paint otherwise.
						if (tool_buttons_group->get_pressed_button() == paint_tool_button && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
							drag_type = DRAG_TYPE_PAINT;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
							HashMap<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, mpos, mpos, drag_erasing);
							for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
								if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
									continue;
								}
								Vector2i coords = E.key;
								if (!drag_modified.has(coords)) {
									drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
								}
								tile_map->set_cell(tile_map_layer, coords, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
							}
							_fix_invalid_tiles_in_tile_map_selection();
						} else if (tool_buttons_group->get_pressed_button() == line_tool_button || (tool_buttons_group->get_pressed_button() == paint_tool_button && Input::get_singleton()->is_key_pressed(Key::SHIFT) && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL))) {
							drag_type = DRAG_TYPE_LINE;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
						} else if (tool_buttons_group->get_pressed_button() == rect_tool_button || (tool_buttons_group->get_pressed_button() == paint_tool_button && Input::get_singleton()->is_key_pressed(Key::SHIFT) && Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL))) {
							drag_type = DRAG_TYPE_RECT;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
						} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button) {
							drag_type = DRAG_TYPE_BUCKET;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
							Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->local_to_map(drag_last_mouse_pos), tile_map->local_to_map(mpos));
							for (int i = 0; i < line.size(); i++) {
								if (!drag_modified.has(line[i])) {
									HashMap<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], bucket_contiguous_checkbox->is_pressed(), drag_erasing);
									for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
										Vector2i coords = E.key;
										if (!drag_modified.has(coords)) {
											drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
											if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
												continue;
											}
											tile_map->set_cell(tile_map_layer, coords, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
										}
									}
								}
							}
							_fix_invalid_tiles_in_tile_map_selection();
						}
					}
				}

			} else {
				// Released
				_stop_dragging();
				drag_erasing = false;
			}

			CanvasItemEditor::get_singleton()->update_viewport();

			return true;
		}
		drag_last_mouse_pos = mpos;
	}

	return false;
}

void TileMapEditorTilesPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	if (tile_map_layer < 0) {
		return;
	}
	ERR_FAIL_INDEX(tile_map_layer, tile_map->get_layers_count());

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (!tile_map->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
	Vector2 mpos = tile_map->get_local_mouse_position();
	Vector2i tile_shape_size = tile_set->get_tile_size();

	// Draw the selection.
	if ((tiles_bottom_panel->is_visible_in_tree() || patterns_bottom_panel->is_visible_in_tree()) && tool_buttons_group->get_pressed_button() == select_tool_button) {
		// In select mode, we only draw the current selection if we are modifying it (pressing control or shift).
		if (drag_type == DRAG_TYPE_MOVE || (drag_type == DRAG_TYPE_SELECT && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT))) {
			// Do nothing.
		} else {
			Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
			Color selection_color = Color().from_hsv(Math::fposmod(grid_color.get_h() + 0.5, 1.0), grid_color.get_s(), grid_color.get_v(), 1.0);
			tile_map->draw_cells_outline(p_overlay, tile_map_selection, selection_color, xform);
		}
	}

	// Handle the preview of the tiles to be placed.
	if ((tiles_bottom_panel->is_visible_in_tree() || patterns_bottom_panel->is_visible_in_tree()) && CanvasItemEditor::get_singleton()->get_current_tool() == CanvasItemEditor::TOOL_SELECT && has_mouse) { // Only if the tilemap editor is opened and the viewport is hovered.
		HashMap<Vector2i, TileMapCell> preview;
		Rect2i drawn_grid_rect;

		if (drag_type == DRAG_TYPE_PICK) {
			// Draw the area being picked.
			Rect2i rect = Rect2i(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos) - tile_map->local_to_map(drag_start_mouse_pos)).abs();
			rect.size += Vector2i(1, 1);
			for (int x = rect.position.x; x < rect.get_end().x; x++) {
				for (int y = rect.position.y; y < rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
						Transform2D tile_xform(0, tile_shape_size, 0, tile_map->map_to_local(coords));
						tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0), false);
					}
				}
			}
		} else if (drag_type == DRAG_TYPE_SELECT) {
			// Draw the area being selected.
			Rect2i rect = Rect2i(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos) - tile_map->local_to_map(drag_start_mouse_pos)).abs();
			rect.size += Vector2i(1, 1);
			RBSet<Vector2i> to_draw;
			for (int x = rect.position.x; x < rect.get_end().x; x++) {
				for (int y = rect.position.y; y < rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
						to_draw.insert(coords);
					}
					Transform2D tile_xform(0, tile_shape_size, 0, tile_map->map_to_local(coords));
					tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0, 0.2), true);
				}
			}
			tile_map->draw_cells_outline(p_overlay, to_draw, Color(1.0, 1.0, 1.0), xform);
		} else if (drag_type == DRAG_TYPE_MOVE) {
			if (!(patterns_item_list->is_visible_in_tree() && patterns_item_list->has_point(patterns_item_list->get_local_mouse_position()))) {
				// Preview when moving.
				Vector2i top_left;
				if (!tile_map_selection.is_empty()) {
					top_left = tile_map_selection.front()->get();
				}
				for (const Vector2i &E : tile_map_selection) {
					top_left = top_left.min(E);
				}
				Vector2i offset = drag_start_mouse_pos - tile_map->map_to_local(top_left);
				offset = tile_map->local_to_map(mpos - offset) - tile_map->local_to_map(drag_start_mouse_pos - offset);

				TypedArray<Vector2i> selection_used_cells = selection_pattern->get_used_cells();
				for (int i = 0; i < selection_used_cells.size(); i++) {
					Vector2i coords = tile_map->map_pattern(offset + top_left, selection_used_cells[i], selection_pattern);
					preview[coords] = TileMapCell(selection_pattern->get_cell_source_id(selection_used_cells[i]), selection_pattern->get_cell_atlas_coords(selection_used_cells[i]), selection_pattern->get_cell_alternative_tile(selection_used_cells[i]));
				}
			}
		} else if (drag_type == DRAG_TYPE_CLIPBOARD_PASTE) {
			// Preview when pasting.
			Vector2 mouse_offset = (Vector2(tile_map_clipboard->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
			TypedArray<Vector2i> clipboard_used_cells = tile_map_clipboard->get_used_cells();
			for (int i = 0; i < clipboard_used_cells.size(); i++) {
				Vector2i coords = tile_map->map_pattern(tile_map->local_to_map(mpos - mouse_offset), clipboard_used_cells[i], tile_map_clipboard);
				preview[coords] = TileMapCell(tile_map_clipboard->get_cell_source_id(clipboard_used_cells[i]), tile_map_clipboard->get_cell_atlas_coords(clipboard_used_cells[i]), tile_map_clipboard->get_cell_alternative_tile(clipboard_used_cells[i]));
			}
		} else if (!picker_button->is_pressed() && !(drag_type == DRAG_TYPE_NONE && Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT))) {
			bool expand_grid = false;
			if (tool_buttons_group->get_pressed_button() == paint_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a single pattern.
				preview = _draw_line(mpos, mpos, mpos, erase_button->is_pressed());
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == line_tool_button || drag_type == DRAG_TYPE_LINE) {
				if (drag_type == DRAG_TYPE_NONE) {
					// Preview for a single pattern.
					preview = _draw_line(mpos, mpos, mpos, erase_button->is_pressed());
					expand_grid = true;
				} else if (drag_type == DRAG_TYPE_LINE) {
					// Preview for a line pattern.
					preview = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, mpos, drag_erasing);
					expand_grid = true;
				}
			} else if (drag_type == DRAG_TYPE_RECT) {
				// Preview for a rect pattern.
				preview = _draw_rect(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos), drag_erasing);
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a fill pattern.
				preview = _draw_bucket_fill(tile_map->local_to_map(mpos), bucket_contiguous_checkbox->is_pressed(), erase_button->is_pressed());
			}

			// Expand the grid if needed
			if (expand_grid && !preview.is_empty()) {
				drawn_grid_rect = Rect2i(preview.begin()->key, Vector2i(0, 0));
				for (const KeyValue<Vector2i, TileMapCell> &E : preview) {
					drawn_grid_rect.expand_to(E.key);
				}
				drawn_grid_rect.size += Vector2i(1, 1);
			}
		}

		if (!preview.is_empty()) {
			const int fading = 5;

			// Draw the lines of the grid behind the preview.
			bool display_grid = EDITOR_GET("editors/tiles_editor/display_grid");
			if (display_grid) {
				Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
				if (drawn_grid_rect.size.x > 0 && drawn_grid_rect.size.y > 0) {
					drawn_grid_rect = drawn_grid_rect.grow(fading);
					for (int x = drawn_grid_rect.position.x; x < (drawn_grid_rect.position.x + drawn_grid_rect.size.x); x++) {
						for (int y = drawn_grid_rect.position.y; y < (drawn_grid_rect.position.y + drawn_grid_rect.size.y); y++) {
							Vector2i pos_in_rect = Vector2i(x, y) - drawn_grid_rect.position;

							// Fade out the border of the grid.
							float left_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.x), 0.0f, 1.0f);
							float right_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.x, (float)(drawn_grid_rect.size.x - fading), (float)(pos_in_rect.x + 1)), 0.0f, 1.0f);
							float top_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.y), 0.0f, 1.0f);
							float bottom_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.y, (float)(drawn_grid_rect.size.y - fading), (float)(pos_in_rect.y + 1)), 0.0f, 1.0f);
							float opacity = CLAMP(MIN(left_opacity, MIN(right_opacity, MIN(top_opacity, bottom_opacity))) + 0.1, 0.0f, 1.0f);

							Transform2D tile_xform;
							tile_xform.set_origin(tile_map->map_to_local(Vector2(x, y)));
							tile_xform.set_scale(tile_shape_size);
							Color color = grid_color;
							color.a = color.a * opacity;
							tile_set->draw_tile_shape(p_overlay, xform * tile_xform, color, false);
						}
					}
				}
			}

			// Draw the preview.
			for (const KeyValue<Vector2i, TileMapCell> &E : preview) {
				Transform2D tile_xform;
				tile_xform.set_origin(tile_map->map_to_local(E.key));
				tile_xform.set_scale(tile_set->get_tile_size());
				if (!(drag_erasing || erase_button->is_pressed()) && random_tile_toggle->is_pressed()) {
					tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0, 0.5), true);
				} else {
					if (tile_set->has_source(E.value.source_id)) {
						TileSetSource *source = *tile_set->get_source(E.value.source_id);
						TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
						if (atlas_source) {
							// Get tile data.
							TileData *tile_data = atlas_source->get_tile_data(E.value.get_atlas_coords(), E.value.alternative_tile);
							if (!tile_data) {
								continue;
							}

							// Compute the offset
							Rect2i source_rect = atlas_source->get_tile_texture_region(E.value.get_atlas_coords());
							Vector2i tile_offset = tile_data->get_texture_origin();

							// Compute the destination rectangle in the CanvasItem.
							Rect2 dest_rect;
							dest_rect.size = source_rect.size;

							bool transpose = tile_data->get_transpose() ^ bool(E.value.alternative_tile & TileSetAtlasSource::TRANSFORM_TRANSPOSE);
							if (transpose) {
								dest_rect.position = (tile_map->map_to_local(E.key) - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
							} else {
								dest_rect.position = (tile_map->map_to_local(E.key) - dest_rect.size / 2 - tile_offset);
							}

							if (tile_data->get_flip_h() ^ bool(E.value.alternative_tile & TileSetAtlasSource::TRANSFORM_FLIP_H)) {
								dest_rect.size.x = -dest_rect.size.x;
							}

							if (tile_data->get_flip_v() ^ bool(E.value.alternative_tile & TileSetAtlasSource::TRANSFORM_FLIP_V)) {
								dest_rect.size.y = -dest_rect.size.y;
							}

							// Get the tile modulation.
							Color modulate = tile_data->get_modulate();
							Color self_modulate = tile_map->get_modulate_in_tree() * tile_map->get_self_modulate();
							modulate *= self_modulate;
							modulate *= tile_map->get_layer_modulate(tile_map_layer);

							// Draw the tile.
							p_overlay->draw_set_transform_matrix(xform);
							p_overlay->draw_texture_rect_region(atlas_source->get_texture(), dest_rect, source_rect, modulate * Color(1.0, 1.0, 1.0, 0.5), transpose, tile_set->is_uv_clipping());
							p_overlay->draw_set_transform_matrix(Transform2D());
						} else {
							tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0, 0.5), true);
						}
					} else {
						tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(0.0, 0.0, 0.0, 0.5), true);
					}
				}
			}
		}

		Ref<Font> font = p_overlay->get_theme_font(SNAME("font"), SNAME("Label"));
		int font_size = p_overlay->get_theme_font_size(SNAME("font_size"), SNAME("Label"));
		Point2 msgpos = Point2(20 * EDSCALE, p_overlay->get_size().y - 20 * EDSCALE);

		String text = tile_map->local_to_map(tile_map->get_local_mouse_position());
		if (drag_type == DRAG_TYPE_RECT) {
			Vector2i size = tile_map->local_to_map(tile_map->get_local_mouse_position()) - tile_map->local_to_map(drag_start_mouse_pos);
			text += vformat(" %s (%dx%d)", TTR("Drawing Rect:"), ABS(size.x) + 1, ABS(size.y) + 1);
		}

		p_overlay->draw_string(font, msgpos + Point2(1, 1), text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0, 0, 0, 0.8));
		p_overlay->draw_string(font, msgpos + Point2(-1, -1), text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(0, 0, 0, 0.8));
		p_overlay->draw_string(font, msgpos, text, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, Color(1, 1, 1, 1));
	}
}

void TileMapEditorTilesPlugin::_mouse_exited_viewport() {
	has_mouse = false;
	CanvasItemEditor::get_singleton()->update_viewport();
}

TileMapCell TileMapEditorTilesPlugin::_pick_random_tile(Ref<TileMapPattern> p_pattern) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return TileMapCell();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return TileMapCell();
	}

	TypedArray<Vector2i> used_cells = p_pattern->get_used_cells();
	double sum = 0.0;
	for (int i = 0; i < used_cells.size(); i++) {
		int source_id = p_pattern->get_cell_source_id(used_cells[i]);
		Vector2i atlas_coords = p_pattern->get_cell_atlas_coords(used_cells[i]);
		int alternative_tile = p_pattern->get_cell_alternative_tile(used_cells[i]);

		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			TileData *tile_data = atlas_source->get_tile_data(atlas_coords, alternative_tile);
			ERR_FAIL_NULL_V(tile_data, TileMapCell());
			sum += tile_data->get_probability();
		} else {
			sum += 1.0;
		}
	}

	double empty_probability = sum * scattering;
	double current = 0.0;
	double rand = Math::random(0.0, sum + empty_probability);
	for (int i = 0; i < used_cells.size(); i++) {
		int source_id = p_pattern->get_cell_source_id(used_cells[i]);
		Vector2i atlas_coords = p_pattern->get_cell_atlas_coords(used_cells[i]);
		int alternative_tile = p_pattern->get_cell_alternative_tile(used_cells[i]);

		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			current += atlas_source->get_tile_data(atlas_coords, alternative_tile)->get_probability();
		} else {
			current += 1.0;
		}

		if (current >= rand) {
			return TileMapCell(source_id, atlas_coords, alternative_tile);
		}
	}
	return TileMapCell();
}

HashMap<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_line(Vector2 p_start_drag_mouse_pos, Vector2 p_from_mouse_pos, Vector2 p_to_mouse_pos, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	Ref<TileMapPattern> erase_pattern;
	erase_pattern.instantiate();
	erase_pattern->set_cell(Vector2i(0, 0), TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
	Ref<TileMapPattern> pattern = p_erase ? erase_pattern : selection_pattern;

	HashMap<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		// Paint the tiles on the tile map.
		if (!p_erase && random_tile_toggle->is_pressed()) {
			// Paint a random tile.
			Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->local_to_map(p_from_mouse_pos), tile_map->local_to_map(p_to_mouse_pos));
			for (int i = 0; i < line.size(); i++) {
				output.insert(line[i], _pick_random_tile(pattern));
			}
		} else {
			// Paint the pattern.
			// If we paint several tiles, we virtually move the mouse as if it was in the center of the "brush"
			Vector2 mouse_offset = (Vector2(pattern->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
			Vector2i last_hovered_cell = tile_map->local_to_map(p_from_mouse_pos - mouse_offset);
			Vector2i new_hovered_cell = tile_map->local_to_map(p_to_mouse_pos - mouse_offset);
			Vector2i drag_start_cell = tile_map->local_to_map(p_start_drag_mouse_pos - mouse_offset);

			TypedArray<Vector2i> used_cells = pattern->get_used_cells();
			Vector2i offset = Vector2i(Math::posmod(drag_start_cell.x, pattern->get_size().x), Math::posmod(drag_start_cell.y, pattern->get_size().y)); // Note: no posmodv for Vector2i for now. Meh.s
			Vector<Vector2i> line = TileMapEditor::get_line(tile_map, (last_hovered_cell - offset) / pattern->get_size(), (new_hovered_cell - offset) / pattern->get_size());
			for (int i = 0; i < line.size(); i++) {
				Vector2i top_left = line[i] * pattern->get_size() + offset;
				for (int j = 0; j < used_cells.size(); j++) {
					Vector2i coords = tile_map->map_pattern(top_left, used_cells[j], pattern);
					output.insert(coords, TileMapCell(pattern->get_cell_source_id(used_cells[j]), pattern->get_cell_atlas_coords(used_cells[j]), pattern->get_cell_alternative_tile(used_cells[j])));
				}
			}
		}
	}
	return output;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_rect(Vector2i p_start_cell, Vector2i p_end_cell, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	// Create the rect to draw.
	Rect2i rect = Rect2i(p_start_cell, p_end_cell - p_start_cell).abs();
	rect.size += Vector2i(1, 1);

	// Get or create the pattern.
	Ref<TileMapPattern> erase_pattern;
	erase_pattern.instantiate();
	erase_pattern->set_cell(Vector2i(0, 0), TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
	Ref<TileMapPattern> pattern = p_erase ? erase_pattern : selection_pattern;

	HashMap<Vector2i, TileMapCell> err_output;
	ERR_FAIL_COND_V(pattern->is_empty(), err_output);

	// Compute the offset to align things to the bottom or right.
	bool aligned_right = p_end_cell.x < p_start_cell.x;
	bool valigned_bottom = p_end_cell.y < p_start_cell.y;
	Vector2i offset = Vector2i(aligned_right ? -(pattern->get_size().x - (rect.get_size().x % pattern->get_size().x)) : 0, valigned_bottom ? -(pattern->get_size().y - (rect.get_size().y % pattern->get_size().y)) : 0);

	HashMap<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		if (!p_erase && random_tile_toggle->is_pressed()) {
			// Paint a random tile.
			for (int x = 0; x < rect.size.x; x++) {
				for (int y = 0; y < rect.size.y; y++) {
					Vector2i coords = rect.position + Vector2i(x, y);
					output.insert(coords, _pick_random_tile(pattern));
				}
			}
		} else {
			// Paint the pattern.
			TypedArray<Vector2i> used_cells = pattern->get_used_cells();
			for (int x = 0; x <= rect.size.x / pattern->get_size().x; x++) {
				for (int y = 0; y <= rect.size.y / pattern->get_size().y; y++) {
					Vector2i pattern_coords = rect.position + Vector2i(x, y) * pattern->get_size() + offset;
					for (int j = 0; j < used_cells.size(); j++) {
						Vector2i coords = pattern_coords + used_cells[j];
						if (rect.has_point(coords)) {
							output.insert(coords, TileMapCell(pattern->get_cell_source_id(used_cells[j]), pattern->get_cell_atlas_coords(used_cells[j]), pattern->get_cell_alternative_tile(used_cells[j])));
						}
					}
				}
			}
		}
	}

	return output;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_bucket_fill(Vector2i p_coords, bool p_contiguous, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	if (tile_map_layer < 0) {
		return HashMap<Vector2i, TileMapCell>();
	}
	HashMap<Vector2i, TileMapCell> output;
	ERR_FAIL_INDEX_V(tile_map_layer, tile_map->get_layers_count(), output);

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	Ref<TileMapPattern> erase_pattern;
	erase_pattern.instantiate();
	erase_pattern->set_cell(Vector2i(0, 0), TileSet::INVALID_SOURCE, TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE);
	Ref<TileMapPattern> pattern = p_erase ? erase_pattern : selection_pattern;

	if (!pattern->is_empty()) {
		TileMapCell source_cell = tile_map->get_cell(tile_map_layer, p_coords);

		// If we are filling empty tiles, compute the tilemap boundaries.
		Rect2i boundaries;
		if (source_cell.source_id == TileSet::INVALID_SOURCE) {
			boundaries = tile_map->get_used_rect();
		}

		if (p_contiguous) {
			// Replace continuous tiles like the source.
			RBSet<Vector2i> already_checked;
			List<Vector2i> to_check;
			to_check.push_back(p_coords);
			while (!to_check.is_empty()) {
				Vector2i coords = to_check.back()->get();
				to_check.pop_back();
				if (!already_checked.has(coords)) {
					if (source_cell.source_id == tile_map->get_cell_source_id(tile_map_layer, coords) &&
							source_cell.get_atlas_coords() == tile_map->get_cell_atlas_coords(tile_map_layer, coords) &&
							source_cell.alternative_tile == tile_map->get_cell_alternative_tile(tile_map_layer, coords) &&
							(source_cell.source_id != TileSet::INVALID_SOURCE || boundaries.has_point(coords))) {
						if (!p_erase && random_tile_toggle->is_pressed()) {
							// Paint a random tile.
							output.insert(coords, _pick_random_tile(pattern));
						} else {
							// Paint the pattern.
							Vector2i pattern_coords = (coords - p_coords) % pattern->get_size(); // Note: it would be good to have posmodv for Vector2i.
							pattern_coords.x = pattern_coords.x < 0 ? pattern_coords.x + pattern->get_size().x : pattern_coords.x;
							pattern_coords.y = pattern_coords.y < 0 ? pattern_coords.y + pattern->get_size().y : pattern_coords.y;
							if (pattern->has_cell(pattern_coords)) {
								output.insert(coords, TileMapCell(pattern->get_cell_source_id(pattern_coords), pattern->get_cell_atlas_coords(pattern_coords), pattern->get_cell_alternative_tile(pattern_coords)));
							} else {
								output.insert(coords, TileMapCell());
							}
						}

						// Get surrounding tiles (handles different tile shapes).
						TypedArray<Vector2i> around = tile_map->get_surrounding_cells(coords);
						for (int i = 0; i < around.size(); i++) {
							to_check.push_back(around[i]);
						}
					}
					already_checked.insert(coords);
				}
			}
		} else {
			// Replace all tiles like the source.
			TypedArray<Vector2i> to_check;
			if (source_cell.source_id == TileSet::INVALID_SOURCE) {
				Rect2i rect = tile_map->get_used_rect();
				if (!rect.has_area()) {
					rect = Rect2i(p_coords, Vector2i(1, 1));
				}
				for (int x = boundaries.position.x; x < boundaries.get_end().x; x++) {
					for (int y = boundaries.position.y; y < boundaries.get_end().y; y++) {
						to_check.append(Vector2i(x, y));
					}
				}
			} else {
				to_check = tile_map->get_used_cells(tile_map_layer);
			}
			for (int i = 0; i < to_check.size(); i++) {
				Vector2i coords = to_check[i];
				if (source_cell.source_id == tile_map->get_cell_source_id(tile_map_layer, coords) &&
						source_cell.get_atlas_coords() == tile_map->get_cell_atlas_coords(tile_map_layer, coords) &&
						source_cell.alternative_tile == tile_map->get_cell_alternative_tile(tile_map_layer, coords) &&
						(source_cell.source_id != TileSet::INVALID_SOURCE || boundaries.has_point(coords))) {
					if (!p_erase && random_tile_toggle->is_pressed()) {
						// Paint a random tile.
						output.insert(coords, _pick_random_tile(pattern));
					} else {
						// Paint the pattern.
						Vector2i pattern_coords = (coords - p_coords) % pattern->get_size(); // Note: it would be good to have posmodv for Vector2i.
						pattern_coords.x = pattern_coords.x < 0 ? pattern_coords.x + pattern->get_size().x : pattern_coords.x;
						pattern_coords.y = pattern_coords.y < 0 ? pattern_coords.y + pattern->get_size().y : pattern_coords.y;
						if (pattern->has_cell(pattern_coords)) {
							output.insert(coords, TileMapCell(pattern->get_cell_source_id(pattern_coords), pattern->get_cell_atlas_coords(pattern_coords), pattern->get_cell_alternative_tile(pattern_coords)));
						} else {
							output.insert(coords, TileMapCell());
						}
					}
				}
			}
		}
	}
	return output;
}

void TileMapEditorTilesPlugin::_stop_dragging() {
	if (drag_type == DRAG_TYPE_NONE) {
		return;
	}

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	if (tile_map_layer < 0) {
		return;
	}
	ERR_FAIL_INDEX(tile_map_layer, tile_map->get_layers_count());

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
	Vector2 mpos = xform.affine_inverse().xform(CanvasItemEditor::get_singleton()->get_viewport_control()->get_local_mouse_position());

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	switch (drag_type) {
		case DRAG_TYPE_SELECT: {
			undo_redo->create_action(TTR("Change selection"));
			undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());

			if (!Input::get_singleton()->is_key_pressed(Key::SHIFT) && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
				tile_map_selection.clear();
			}
			Rect2i rect = Rect2i(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos) - tile_map->local_to_map(drag_start_mouse_pos)).abs();
			for (int x = rect.position.x; x <= rect.get_end().x; x++) {
				for (int y = rect.position.y; y <= rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL)) {
						if (tile_map_selection.has(coords)) {
							tile_map_selection.erase(coords);
						}
					} else {
						if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
							tile_map_selection.insert(coords);
						}
					}
				}
			}
			undo_redo->add_do_method(this, "_set_tile_map_selection", _get_tile_map_selection());
			undo_redo->commit_action(false);

			_update_selection_pattern_from_tilemap_selection();
			_update_tileset_selection_from_selection_pattern();
		} break;
		case DRAG_TYPE_MOVE: {
			if (patterns_item_list->is_visible_in_tree() && patterns_item_list->has_point(patterns_item_list->get_local_mouse_position())) {
				// Restore the cells.
				for (KeyValue<Vector2i, TileMapCell> kv : drag_modified) {
					tile_map->set_cell(tile_map_layer, kv.key, kv.value.source_id, kv.value.get_atlas_coords(), kv.value.alternative_tile);
				}

				if (!EditorNode::get_singleton()->is_resource_read_only(tile_set)) {
					// Creating a pattern in the pattern list.
					select_last_pattern = true;
					int new_pattern_index = tile_set->get_patterns_count();
					undo_redo->create_action(TTR("Add TileSet pattern"));
					undo_redo->add_do_method(*tile_set, "add_pattern", selection_pattern, new_pattern_index);
					undo_redo->add_undo_method(*tile_set, "remove_pattern", new_pattern_index);
					undo_redo->commit_action();
				}
			} else {
				// Get the top-left cell.
				Vector2i top_left;
				if (!tile_map_selection.is_empty()) {
					top_left = tile_map_selection.front()->get();
				}
				for (const Vector2i &E : tile_map_selection) {
					top_left = top_left.min(E);
				}

				// Get the offset from the mouse.
				Vector2i offset = drag_start_mouse_pos - tile_map->map_to_local(top_left);
				offset = tile_map->local_to_map(mpos - offset) - tile_map->local_to_map(drag_start_mouse_pos - offset);

				TypedArray<Vector2i> selection_used_cells = selection_pattern->get_used_cells();

				// Build the list of cells to undo.
				Vector2i coords;
				HashMap<Vector2i, TileMapCell> cells_undo;
				for (int i = 0; i < selection_used_cells.size(); i++) {
					coords = tile_map->map_pattern(top_left, selection_used_cells[i], selection_pattern);
					cells_undo[coords] = TileMapCell(drag_modified[coords].source_id, drag_modified[coords].get_atlas_coords(), drag_modified[coords].alternative_tile);
					coords = tile_map->map_pattern(top_left + offset, selection_used_cells[i], selection_pattern);
					cells_undo[coords] = TileMapCell(tile_map->get_cell_source_id(tile_map_layer, coords), tile_map->get_cell_atlas_coords(tile_map_layer, coords), tile_map->get_cell_alternative_tile(tile_map_layer, coords));
				}

				// Build the list of cells to do.
				HashMap<Vector2i, TileMapCell> cells_do;
				for (int i = 0; i < selection_used_cells.size(); i++) {
					coords = tile_map->map_pattern(top_left, selection_used_cells[i], selection_pattern);
					cells_do[coords] = TileMapCell();
				}
				for (int i = 0; i < selection_used_cells.size(); i++) {
					coords = tile_map->map_pattern(top_left + offset, selection_used_cells[i], selection_pattern);
					cells_do[coords] = TileMapCell(selection_pattern->get_cell_source_id(selection_used_cells[i]), selection_pattern->get_cell_atlas_coords(selection_used_cells[i]), selection_pattern->get_cell_alternative_tile(selection_used_cells[i]));
				}

				// Move the tiles.
				undo_redo->create_action(TTR("Move tiles"));
				for (const KeyValue<Vector2i, TileMapCell> &E : cells_do) {
					undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				}
				for (const KeyValue<Vector2i, TileMapCell> &E : cells_undo) {
					undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				}

				// Update the selection.
				undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());
				tile_map_selection.clear();
				for (int i = 0; i < selection_used_cells.size(); i++) {
					coords = tile_map->map_pattern(top_left + offset, selection_used_cells[i], selection_pattern);
					tile_map_selection.insert(coords);
				}
				undo_redo->add_do_method(this, "_set_tile_map_selection", _get_tile_map_selection());
				undo_redo->commit_action();
			}
		} break;
		case DRAG_TYPE_PICK: {
			Rect2i rect = Rect2i(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos) - tile_map->local_to_map(drag_start_mouse_pos)).abs();
			rect.size += Vector2i(1, 1);

			int picked_source = -1;
			TypedArray<Vector2i> coords_array;
			for (int x = rect.position.x; x < rect.get_end().x; x++) {
				for (int y = rect.position.y; y < rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);

					int source = tile_map->get_cell_source_id(tile_map_layer, coords);
					if (source != TileSet::INVALID_SOURCE) {
						coords_array.push_back(coords);
						if (picked_source == -1) {
							picked_source = source;
						} else if (picked_source != source) {
							picked_source = -2;
						}
					}
				}
			}

			if (picked_source >= 0) {
				for (int i = 0; i < sources_list->get_item_count(); i++) {
					if (int(sources_list->get_item_metadata(i)) == picked_source) {
						sources_list->set_current(i);
						TilesEditorUtils::get_singleton()->set_sources_lists_current(i);
						break;
					}
				}
				sources_list->ensure_current_is_visible();
			}

			Ref<TileMapPattern> new_selection_pattern = tile_map->get_pattern(tile_map_layer, coords_array);
			if (!new_selection_pattern->is_empty()) {
				selection_pattern = new_selection_pattern;
				_update_tileset_selection_from_selection_pattern();
			}
			picker_button->set_pressed(false);
		} break;
		case DRAG_TYPE_PAINT: {
			undo_redo->create_action(TTR("Paint tiles"));
			for (const KeyValue<Vector2i, TileMapCell> &E : drag_modified) {
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
			}
			undo_redo->commit_action(false);
		} break;
		case DRAG_TYPE_LINE: {
			HashMap<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, mpos, drag_erasing);
			undo_redo->create_action(TTR("Paint tiles"));
			for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
				if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
					continue;
				}
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_RECT: {
			HashMap<Vector2i, TileMapCell> to_draw = _draw_rect(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos), drag_erasing);
			undo_redo->create_action(TTR("Paint tiles"));
			for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
				if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
					continue;
				}
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_BUCKET: {
			undo_redo->create_action(TTR("Paint tiles"));
			for (const KeyValue<Vector2i, TileMapCell> &E : drag_modified) {
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
			}
			undo_redo->commit_action(false);
		} break;
		case DRAG_TYPE_CLIPBOARD_PASTE: {
			Vector2 mouse_offset = (Vector2(tile_map_clipboard->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
			undo_redo->create_action(TTR("Paste tiles"));
			TypedArray<Vector2i> used_cells = tile_map_clipboard->get_used_cells();
			for (int i = 0; i < used_cells.size(); i++) {
				Vector2i coords = tile_map->map_pattern(tile_map->local_to_map(mpos - mouse_offset), used_cells[i], tile_map_clipboard);
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, coords, tile_map_clipboard->get_cell_source_id(used_cells[i]), tile_map_clipboard->get_cell_atlas_coords(used_cells[i]), tile_map_clipboard->get_cell_alternative_tile(used_cells[i]));
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, coords, tile_map->get_cell_source_id(tile_map_layer, coords), tile_map->get_cell_atlas_coords(tile_map_layer, coords), tile_map->get_cell_alternative_tile(tile_map_layer, coords));
			}
			undo_redo->commit_action();
		} break;
		default:
			break;
	}
	drag_type = DRAG_TYPE_NONE;
}

void TileMapEditorTilesPlugin::_apply_transform(int p_type) {
	if (selection_pattern.is_null() || selection_pattern->is_empty()) {
		return;
	}

	Ref<TileMapPattern> transformed_pattern;
	transformed_pattern.instantiate();
	bool keep_shape = selection_pattern->get_size() == Vector2i(1, 1);

	Vector2i size = selection_pattern->get_size();
	for (int y = 0; y < size.y; y++) {
		for (int x = 0; x < size.x; x++) {
			Vector2i src_coords = Vector2i(x, y);
			if (!selection_pattern->has_cell(src_coords)) {
				continue;
			}

			Vector2i dst_coords;

			if (keep_shape) {
				dst_coords = src_coords;
			} else if (p_type == TRANSFORM_ROTATE_LEFT) {
				dst_coords = Vector2i(y, size.x - x - 1);
			} else if (p_type == TRANSFORM_ROTATE_RIGHT) {
				dst_coords = Vector2i(size.y - y - 1, x);
			} else if (p_type == TRANSFORM_FLIP_H) {
				dst_coords = Vector2i(size.x - x - 1, y);
			} else if (p_type == TRANSFORM_FLIP_V) {
				dst_coords = Vector2i(x, size.y - y - 1);
			}

			transformed_pattern->set_cell(dst_coords,
					selection_pattern->get_cell_source_id(src_coords), selection_pattern->get_cell_atlas_coords(src_coords),
					_get_transformed_alternative(selection_pattern->get_cell_alternative_tile(src_coords), p_type));
		}
	}
	selection_pattern = transformed_pattern;
	CanvasItemEditor::get_singleton()->update_viewport();
}

int TileMapEditorTilesPlugin::_get_transformed_alternative(int p_alternative_id, int p_transform) {
	bool transform_flip_h = p_alternative_id & TileSetAtlasSource::TRANSFORM_FLIP_H;
	bool transform_flip_v = p_alternative_id & TileSetAtlasSource::TRANSFORM_FLIP_V;
	bool transform_transpose = p_alternative_id & TileSetAtlasSource::TRANSFORM_TRANSPOSE;

	switch (p_transform) {
		case TRANSFORM_ROTATE_LEFT:
		case TRANSFORM_ROTATE_RIGHT: {
			// A matrix with every possible flip/transpose combination, sorted by what comes next when you rotate.
			const LocalVector<bool> rotation_matrix = {
				0, 0, 0,
				0, 1, 1,
				1, 1, 0,
				1, 0, 1,
				1, 0, 0,
				0, 0, 1,
				0, 1, 0,
				1, 1, 1
			};

			for (int i = 0; i < 8; i++) {
				if (transform_flip_h == rotation_matrix[i * 3] && transform_flip_v == rotation_matrix[i * 3 + 1] && transform_transpose == rotation_matrix[i * 3 + 2]) {
					if (p_transform == TRANSFORM_ROTATE_LEFT) {
						i = i / 4 * 4 + (i + 1) % 4;
					} else {
						i = i / 4 * 4 + Math::posmod(i - 1, 4);
					}
					transform_flip_h = rotation_matrix[i * 3];
					transform_flip_v = rotation_matrix[i * 3 + 1];
					transform_transpose = rotation_matrix[i * 3 + 2];
					break;
				}
			}
		} break;
		case TRANSFORM_FLIP_H: {
			transform_flip_h = !transform_flip_h;
		} break;
		case TRANSFORM_FLIP_V: {
			transform_flip_v = !transform_flip_v;
		} break;
	}

	return TileSetAtlasSource::alternative_no_transform(p_alternative_id) |
			int(transform_flip_h) * TileSetAtlasSource::TRANSFORM_FLIP_H |
			int(transform_flip_v) * TileSetAtlasSource::TRANSFORM_FLIP_V |
			int(transform_transpose) * TileSetAtlasSource::TRANSFORM_TRANSPOSE;
}

void TileMapEditorTilesPlugin::_update_fix_selected_and_hovered() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		hovered_tile.source_id = TileSet::INVALID_SOURCE;
		hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		patterns_item_list->deselect_all();
		tile_map_selection.clear();
		selection_pattern.instantiate();
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		hovered_tile.source_id = TileSet::INVALID_SOURCE;
		hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		patterns_item_list->deselect_all();
		tile_map_selection.clear();
		selection_pattern.instantiate();
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		hovered_tile.source_id = TileSet::INVALID_SOURCE;
		hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		patterns_item_list->deselect_all();
		tile_map_selection.clear();
		selection_pattern.instantiate();
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);

	// Clear hovered if needed.
	if (source_id != hovered_tile.source_id ||
			!tile_set->has_source(hovered_tile.source_id) ||
			!tile_set->get_source(hovered_tile.source_id)->has_tile(hovered_tile.get_atlas_coords()) ||
			!tile_set->get_source(hovered_tile.source_id)->has_alternative_tile(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile)) {
		hovered_tile.source_id = TileSet::INVALID_SOURCE;
		hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	}

	// Selection if needed.
	for (RBSet<TileMapCell>::Element *E = tile_set_selection.front(); E;) {
		RBSet<TileMapCell>::Element *N = E->next();
		const TileMapCell *selected = &(E->get());
		if (!tile_set->has_source(selected->source_id) ||
				!tile_set->get_source(selected->source_id)->has_tile(selected->get_atlas_coords()) ||
				!tile_set->get_source(selected->source_id)->has_alternative_tile(selected->get_atlas_coords(), selected->alternative_tile)) {
			tile_set_selection.erase(E);
		}
		E = N;
	}

	if (!tile_map_selection.is_empty()) {
		_update_selection_pattern_from_tilemap_selection();
	} else if (tiles_bottom_panel->is_visible_in_tree()) {
		_update_selection_pattern_from_tileset_tiles_selection();
	} else {
		_update_selection_pattern_from_tileset_pattern_selection();
	}
}

void TileMapEditorTilesPlugin::_fix_invalid_tiles_in_tile_map_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	RBSet<Vector2i> to_remove;
	for (Vector2i selected : tile_map_selection) {
		TileMapCell cell = tile_map->get_cell(tile_map_layer, selected);
		if (cell.source_id == TileSet::INVALID_SOURCE && cell.get_atlas_coords() == TileSetSource::INVALID_ATLAS_COORDS && cell.alternative_tile == TileSetAtlasSource::INVALID_TILE_ALTERNATIVE) {
			to_remove.insert(selected);
		}
	}

	for (Vector2i cell : to_remove) {
		tile_map_selection.erase(cell);
	}
}
void TileMapEditorTilesPlugin::patterns_item_list_empty_clicked(const Vector2 &p_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index == MouseButton::LEFT) {
		_update_selection_pattern_from_tileset_pattern_selection();
	}
}

void TileMapEditorTilesPlugin::_update_selection_pattern_from_tilemap_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	ERR_FAIL_INDEX(tile_map_layer, tile_map->get_layers_count());

	selection_pattern.instantiate();

	TypedArray<Vector2i> coords_array;
	for (const Vector2i &E : tile_map_selection) {
		coords_array.push_back(E);
	}
	selection_pattern = tile_map->get_pattern(tile_map_layer, coords_array);
	_update_transform_buttons();
}

void TileMapEditorTilesPlugin::_update_selection_pattern_from_tileset_tiles_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Clear the tilemap selection.
	tile_map_selection.clear();

	// Clear the selected pattern.
	selection_pattern.instantiate();

	// Group per source.
	HashMap<int, List<const TileMapCell *>> per_source;
	for (const TileMapCell &E : tile_set_selection) {
		per_source[E.source_id].push_back(&(E));
	}

	int vertical_offset = 0;
	for (const KeyValue<int, List<const TileMapCell *>> &E_source : per_source) {
		// Per source.
		List<const TileMapCell *> unorganized;
		Rect2i encompassing_rect_coords;
		HashMap<Vector2i, const TileMapCell *> organized_pattern;

		TileSetSource *source = *tile_set->get_source(E_source.key);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			// Organize using coordinates.
			for (const TileMapCell *current : E_source.value) {
				if (current->alternative_tile == 0) {
					organized_pattern[current->get_atlas_coords()] = current;
				} else {
					unorganized.push_back(current);
				}
			}

			// Compute the encompassing rect for the organized pattern.
			HashMap<Vector2i, const TileMapCell *>::Iterator E_cell = organized_pattern.begin();
			if (E_cell) {
				encompassing_rect_coords = Rect2i(E_cell->key, Vector2i(1, 1));
				for (; E_cell; ++E_cell) {
					encompassing_rect_coords.expand_to(E_cell->key + Vector2i(1, 1));
					encompassing_rect_coords.expand_to(E_cell->key);
				}
			}
		} else {
			// Add everything unorganized.
			for (const TileMapCell *cell : E_source.value) {
				unorganized.push_back(cell);
			}
		}

		// Now add everything to the output pattern.
		for (const KeyValue<Vector2i, const TileMapCell *> &E_cell : organized_pattern) {
			selection_pattern->set_cell(E_cell.key - encompassing_rect_coords.position + Vector2i(0, vertical_offset), E_cell.value->source_id, E_cell.value->get_atlas_coords(), E_cell.value->alternative_tile);
		}
		Vector2i organized_size = selection_pattern->get_size();
		int unorganized_index = 0;
		for (const TileMapCell *cell : unorganized) {
			selection_pattern->set_cell(Vector2(organized_size.x + unorganized_index, vertical_offset), cell->source_id, cell->get_atlas_coords(), cell->alternative_tile);
			unorganized_index++;
		}
		vertical_offset += MAX(organized_size.y, 1);
	}
	CanvasItemEditor::get_singleton()->update_viewport();
	_update_transform_buttons();
}

void TileMapEditorTilesPlugin::_update_selection_pattern_from_tileset_pattern_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Clear the tilemap selection.
	tile_map_selection.clear();

	// Clear the selected pattern.
	selection_pattern.instantiate();

	if (patterns_item_list->get_selected_items().size() >= 1) {
		selection_pattern = patterns_item_list->get_item_metadata(patterns_item_list->get_selected_items()[0]);
	}

	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditorTilesPlugin::_update_tileset_selection_from_selection_pattern() {
	tile_set_selection.clear();
	TypedArray<Vector2i> used_cells = selection_pattern->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = used_cells[i];
		if (selection_pattern->get_cell_source_id(coords) != TileSet::INVALID_SOURCE) {
			tile_set_selection.insert(TileMapCell(selection_pattern->get_cell_source_id(coords), selection_pattern->get_cell_atlas_coords(coords), selection_pattern->get_cell_alternative_tile(coords)));
		}
	}
	_update_source_display();
	tile_atlas_control->queue_redraw();
	alternative_tiles_control->queue_redraw();
	_update_transform_buttons();
}

void TileMapEditorTilesPlugin::_tile_atlas_control_draw() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);
	if (!tile_set->has_source(source_id)) {
		return;
	}

	TileSetAtlasSource *atlas = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(source_id));
	if (!atlas) {
		return;
	}

	// Draw the selection.
	Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
	Color selection_color = Color().from_hsv(Math::fposmod(grid_color.get_h() + 0.5, 1.0), grid_color.get_s(), grid_color.get_v(), 1.0);
	for (const TileMapCell &E : tile_set_selection) {
		if (E.source_id == source_id && E.alternative_tile == 0) {
			for (int frame = 0; frame < atlas->get_tile_animation_frames_count(E.get_atlas_coords()); frame++) {
				Color color = selection_color;
				if (frame > 0) {
					color.a *= 0.3;
				}
				TilesEditorUtils::draw_selection_rect(tile_atlas_control, atlas->get_tile_texture_region(E.get_atlas_coords(), frame), color);
			}
		}
	}

	// Draw the hovered tile.
	if (hovered_tile.get_atlas_coords() != TileSetSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0 && !tile_set_dragging_selection) {
		for (int frame = 0; frame < atlas->get_tile_animation_frames_count(hovered_tile.get_atlas_coords()); frame++) {
			Color color = Color(1.0, 0.8, 0.0, frame == 0 ? 0.6 : 0.3);
			TilesEditorUtils::draw_selection_rect(tile_atlas_control, atlas->get_tile_texture_region(hovered_tile.get_atlas_coords(), frame), color);
		}
	}

	// Draw the selection rect.
	if (tile_set_dragging_selection) {
		Vector2i start_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_set_drag_start_mouse_pos, true);
		Vector2i end_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position(), true);

		Rect2i region = Rect2i(start_tile, end_tile - start_tile).abs();
		region.size += Vector2i(1, 1);

		RBSet<Vector2i> to_draw;
		for (int x = region.position.x; x < region.get_end().x; x++) {
			for (int y = region.position.y; y < region.get_end().y; y++) {
				Vector2i tile = atlas->get_tile_at_coords(Vector2i(x, y));
				if (tile != TileSetSource::INVALID_ATLAS_COORDS) {
					to_draw.insert(tile);
				}
			}
		}
		for (const Vector2i &E : to_draw) {
			TilesEditorUtils::draw_selection_rect(tile_atlas_control, atlas->get_tile_texture_region(E));
		}
	}
}

void TileMapEditorTilesPlugin::_tile_atlas_control_mouse_exited() {
	hovered_tile.source_id = TileSet::INVALID_SOURCE;
	hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	tile_atlas_control->queue_redraw();
}

void TileMapEditorTilesPlugin::_tile_atlas_control_gui_input(const Ref<InputEvent> &p_event) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);
	if (!tile_set->has_source(source_id)) {
		return;
	}

	TileSetAtlasSource *atlas = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(source_id));
	if (!atlas) {
		return;
	}

	// Update the hovered tile
	hovered_tile.source_id = source_id;
	hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
	if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
		coords = atlas->get_tile_at_coords(coords);
		if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
			hovered_tile.set_atlas_coords(coords);
			hovered_tile.alternative_tile = 0;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->queue_redraw();
		alternative_tiles_control->queue_redraw();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) { // Pressed
			tile_set_dragging_selection = true;
			tile_set_drag_start_mouse_pos = tile_atlas_control->get_local_mouse_position();
			if (!mb->is_shift_pressed()) {
				tile_set_selection.clear();
			}

			if (hovered_tile.get_atlas_coords() != TileSetSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0) {
				if (mb->is_shift_pressed() && tile_set_selection.has(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0))) {
					tile_set_selection.erase(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0));
				} else {
					tile_set_selection.insert(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0));
				}
			}
			_update_selection_pattern_from_tileset_tiles_selection();
		} else { // Released
			if (tile_set_dragging_selection) {
				if (!mb->is_shift_pressed()) {
					tile_set_selection.clear();
				}
				// Compute the covered area.
				Vector2i start_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_set_drag_start_mouse_pos, true);
				Vector2i end_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position(), true);
				if (start_tile != TileSetSource::INVALID_ATLAS_COORDS && end_tile != TileSetSource::INVALID_ATLAS_COORDS) {
					Rect2i region = Rect2i(start_tile, end_tile - start_tile).abs();
					region.size += Vector2i(1, 1);

					// To update the selection, we copy the selected/not selected status of the tiles we drag from.
					Vector2i start_coords = atlas->get_tile_at_coords(start_tile);
					if (mb->is_shift_pressed() && start_coords != TileSetSource::INVALID_ATLAS_COORDS && !tile_set_selection.has(TileMapCell(source_id, start_coords, 0))) {
						// Remove from the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileSetSource::INVALID_ATLAS_COORDS && tile_set_selection.has(TileMapCell(source_id, tile_coords, 0))) {
									tile_set_selection.erase(TileMapCell(source_id, tile_coords, 0));
								}
							}
						}
					} else {
						// Insert in the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileSetSource::INVALID_ATLAS_COORDS) {
									tile_set_selection.insert(TileMapCell(source_id, tile_coords, 0));
								}
							}
						}
					}
				}
				_update_selection_pattern_from_tileset_tiles_selection();
			}
			tile_set_dragging_selection = false;
		}
		tile_atlas_control->queue_redraw();
	}
}

void TileMapEditorTilesPlugin::_tile_alternatives_control_draw() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);
	if (!tile_set->has_source(source_id)) {
		return;
	}

	TileSetAtlasSource *atlas = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(source_id));
	if (!atlas) {
		return;
	}

	// Draw the selection.
	for (const TileMapCell &E : tile_set_selection) {
		if (E.source_id == source_id && E.get_atlas_coords() != TileSetSource::INVALID_ATLAS_COORDS && E.alternative_tile > 0) {
			Rect2i rect = tile_atlas_view->get_alternative_tile_rect(E.get_atlas_coords(), E.alternative_tile);
			if (rect != Rect2i()) {
				TilesEditorUtils::draw_selection_rect(alternative_tiles_control, rect);
			}
		}
	}

	// Draw hovered tile.
	if (hovered_tile.get_atlas_coords() != TileSetSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile > 0) {
		Rect2i rect = tile_atlas_view->get_alternative_tile_rect(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile);
		if (rect != Rect2i()) {
			TilesEditorUtils::draw_selection_rect(alternative_tiles_control, rect, Color(1.0, 0.8, 0.0, 0.5));
		}
	}
}

void TileMapEditorTilesPlugin::_tile_alternatives_control_mouse_exited() {
	hovered_tile.source_id = TileSet::INVALID_SOURCE;
	hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	alternative_tiles_control->queue_redraw();
}

void TileMapEditorTilesPlugin::_tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);
	if (!tile_set->has_source(source_id)) {
		return;
	}

	TileSetAtlasSource *atlas = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(source_id));
	if (!atlas) {
		return;
	}

	// Update the hovered tile
	hovered_tile.source_id = source_id;
	hovered_tile.set_atlas_coords(TileSetSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetSource::INVALID_TILE_ALTERNATIVE;
	Vector3i alternative_coords = tile_atlas_view->get_alternative_tile_at_pos(alternative_tiles_control->get_local_mouse_position());
	Vector2i coords = Vector2i(alternative_coords.x, alternative_coords.y);
	int alternative = alternative_coords.z;
	if (coords != TileSetSource::INVALID_ATLAS_COORDS && alternative != TileSetSource::INVALID_TILE_ALTERNATIVE) {
		hovered_tile.set_atlas_coords(coords);
		hovered_tile.alternative_tile = alternative;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->queue_redraw();
		alternative_tiles_control->queue_redraw();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT) {
		if (mb->is_pressed()) { // Pressed
			// Left click pressed.
			if (!mb->is_shift_pressed()) {
				tile_set_selection.clear();
			}

			if (coords != TileSetSource::INVALID_ATLAS_COORDS && alternative != TileSetAtlasSource::INVALID_TILE_ALTERNATIVE) {
				if (mb->is_shift_pressed() && tile_set_selection.has(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile))) {
					tile_set_selection.erase(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile));
				} else {
					tile_set_selection.insert(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile));
				}
			}
			_update_selection_pattern_from_tileset_tiles_selection();
		}
		tile_atlas_control->queue_redraw();
		alternative_tiles_control->queue_redraw();
	}
}

void TileMapEditorTilesPlugin::_set_tile_map_selection(const TypedArray<Vector2i> &p_selection) {
	tile_map_selection.clear();
	for (int i = 0; i < p_selection.size(); i++) {
		tile_map_selection.insert(p_selection[i]);
	}
	_update_selection_pattern_from_tilemap_selection();
	_update_tileset_selection_from_selection_pattern();
	CanvasItemEditor::get_singleton()->update_viewport();
}

TypedArray<Vector2i> TileMapEditorTilesPlugin::_get_tile_map_selection() const {
	TypedArray<Vector2i> output;
	for (const Vector2i &E : tile_map_selection) {
		output.push_back(E);
	}
	return output;
}

void TileMapEditorTilesPlugin::edit(ObjectID p_tile_map_id, int p_tile_map_layer) {
	_stop_dragging(); // Avoids staying in a wrong drag state.

	// Disable sort button if the tileset is read-only
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		Ref<TileSet> tile_set = tile_map->get_tileset();
		if (tile_set.is_valid()) {
			source_sort_button->set_disabled(EditorNode::get_singleton()->is_resource_read_only(tile_set));
		}
	}

	if (tile_map_id != p_tile_map_id) {
		tile_map_id = p_tile_map_id;

		// Clear the selection.
		tile_set_selection.clear();
		patterns_item_list->deselect_all();
		tile_map_selection.clear();
		selection_pattern.instantiate();
	}

	tile_map_layer = p_tile_map_layer;
}

void TileMapEditorTilesPlugin::_set_source_sort(int p_sort) {
	for (int i = 0; i != TilesEditorUtils::SOURCE_SORT_MAX; i++) {
		source_sort_button->get_popup()->set_item_checked(i, (i == (int)p_sort));
	}
	TilesEditorUtils::get_singleton()->set_sorting_option(p_sort);
	_update_tile_set_sources_list();
	EditorSettings::get_singleton()->set_project_metadata("editor_metadata", "tile_source_sort", p_sort);
}

void TileMapEditorTilesPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_scene_thumbnail_done"), &TileMapEditorTilesPlugin::_scene_thumbnail_done);
	ClassDB::bind_method(D_METHOD("_set_tile_map_selection", "selection"), &TileMapEditorTilesPlugin::_set_tile_map_selection);
	ClassDB::bind_method(D_METHOD("_get_tile_map_selection"), &TileMapEditorTilesPlugin::_get_tile_map_selection);
}

TileMapEditorTilesPlugin::TileMapEditorTilesPlugin() {
	CanvasItemEditor::get_singleton()
			->get_viewport_control()
			->connect("mouse_exited", callable_mp(this, &TileMapEditorTilesPlugin::_mouse_exited_viewport));

	// --- Shortcuts ---
	ED_SHORTCUT("tiles_editor/cut", TTR("Cut"), KeyModifierMask::CMD_OR_CTRL | Key::X);
	ED_SHORTCUT("tiles_editor/copy", TTR("Copy"), KeyModifierMask::CMD_OR_CTRL | Key::C);
	ED_SHORTCUT("tiles_editor/paste", TTR("Paste"), KeyModifierMask::CMD_OR_CTRL | Key::V);
	ED_SHORTCUT("tiles_editor/cancel", TTR("Cancel"), Key::ESCAPE);
	ED_SHORTCUT("tiles_editor/delete", TTR("Delete"), Key::KEY_DELETE);

	// --- Initialize references ---
	tile_map_clipboard.instantiate();
	selection_pattern.instantiate();

	// --- Toolbar ---
	toolbar = memnew(HBoxContainer);

	HBoxContainer *tilemap_tiles_tools_buttons = memnew(HBoxContainer);

	tool_buttons_group.instantiate();

	select_tool_button = memnew(Button);
	select_tool_button->set_theme_type_variation("FlatButton");
	select_tool_button->set_toggle_mode(true);
	select_tool_button->set_button_group(tool_buttons_group);
	select_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/selection_tool", TTR("Selection"), Key::S));
	select_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(select_tool_button);
	viewport_shortcut_buttons.push_back(select_tool_button);

	paint_tool_button = memnew(Button);
	paint_tool_button->set_theme_type_variation("FlatButton");
	paint_tool_button->set_toggle_mode(true);
	paint_tool_button->set_button_group(tool_buttons_group);
	paint_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/paint_tool", TTR("Paint"), Key::D));
	paint_tool_button->set_tooltip_text(TTR("Shift: Draw line.") + "\n" + keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Shift: Draw rectangle."));
	paint_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(paint_tool_button);
	viewport_shortcut_buttons.push_back(paint_tool_button);

	line_tool_button = memnew(Button);
	line_tool_button->set_theme_type_variation("FlatButton");
	line_tool_button->set_toggle_mode(true);
	line_tool_button->set_button_group(tool_buttons_group);
	// TRANSLATORS: This refers to the line tool in the tilemap editor.
	line_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/line_tool", TTR("Line", "Tool"), Key::L));
	line_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(line_tool_button);
	viewport_shortcut_buttons.push_back(line_tool_button);

	rect_tool_button = memnew(Button);
	rect_tool_button->set_theme_type_variation("FlatButton");
	rect_tool_button->set_toggle_mode(true);
	rect_tool_button->set_button_group(tool_buttons_group);
	rect_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/rect_tool", TTR("Rect"), Key::R));
	rect_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(rect_tool_button);
	viewport_shortcut_buttons.push_back(rect_tool_button);

	bucket_tool_button = memnew(Button);
	bucket_tool_button->set_theme_type_variation("FlatButton");
	bucket_tool_button->set_toggle_mode(true);
	bucket_tool_button->set_button_group(tool_buttons_group);
	bucket_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/bucket_tool", TTR("Bucket"), Key::B));
	bucket_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(bucket_tool_button);
	toolbar->add_child(tilemap_tiles_tools_buttons);
	viewport_shortcut_buttons.push_back(bucket_tool_button);

	// -- TileMap tool settings --
	tools_settings = memnew(HBoxContainer);
	toolbar->add_child(tools_settings);

	tools_settings_vsep = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep);

	// Picker
	picker_button = memnew(Button);
	picker_button->set_theme_type_variation("FlatButton");
	picker_button->set_toggle_mode(true);
	picker_button->set_shortcut(ED_SHORTCUT("tiles_editor/picker", TTR("Picker"), Key::P));
	Key key = (OS::get_singleton()->has_feature("macos") || OS::get_singleton()->has_feature("web_macos") || OS::get_singleton()->has_feature("web_ios")) ? Key::META : Key::CTRL;
	picker_button->set_tooltip_text(vformat(TTR("Alternatively hold %s with other tools to pick tile."), find_keycode_name(key)));
	picker_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(picker_button);
	viewport_shortcut_buttons.push_back(picker_button);

	// Erase button.
	erase_button = memnew(Button);
	erase_button->set_theme_type_variation("FlatButton");
	erase_button->set_toggle_mode(true);
	erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", TTR("Eraser"), Key::E));
	erase_button->set_tooltip_text(TTR("Alternatively use RMB to erase tiles."));
	erase_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(erase_button);
	viewport_shortcut_buttons.push_back(erase_button);

	// Transform toolbar.
	transform_toolbar = memnew(HBoxContainer);
	tools_settings->add_child(transform_toolbar);
	transform_toolbar->add_child(memnew(VSeparator));

	transform_button_rotate_left = memnew(Button);
	transform_button_rotate_left->set_theme_type_variation("FlatButton");
	transform_button_rotate_left->set_shortcut(ED_SHORTCUT("tiles_editor/rotate_tile_left", TTR("Rotate Tile Left"), Key::Z));
	transform_toolbar->add_child(transform_button_rotate_left);
	transform_button_rotate_left->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_apply_transform).bind(TRANSFORM_ROTATE_LEFT));
	viewport_shortcut_buttons.push_back(transform_button_rotate_left);

	transform_button_rotate_right = memnew(Button);
	transform_button_rotate_right->set_theme_type_variation("FlatButton");
	transform_button_rotate_right->set_shortcut(ED_SHORTCUT("tiles_editor/rotate_tile_right", TTR("Rotate Tile Right"), Key::X));
	transform_toolbar->add_child(transform_button_rotate_right);
	transform_button_rotate_right->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_apply_transform).bind(TRANSFORM_ROTATE_RIGHT));
	viewport_shortcut_buttons.push_back(transform_button_rotate_right);

	transform_button_flip_h = memnew(Button);
	transform_button_flip_h->set_theme_type_variation("FlatButton");
	transform_button_flip_h->set_shortcut(ED_SHORTCUT("tiles_editor/flip_tile_horizontal", TTR("Flip Tile Horizontally"), Key::C));
	transform_toolbar->add_child(transform_button_flip_h);
	transform_button_flip_h->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_apply_transform).bind(TRANSFORM_FLIP_H));
	viewport_shortcut_buttons.push_back(transform_button_flip_h);

	transform_button_flip_v = memnew(Button);
	transform_button_flip_v->set_theme_type_variation("FlatButton");
	transform_button_flip_v->set_shortcut(ED_SHORTCUT("tiles_editor/flip_tile_vertical", TTR("Flip Tile Vertically"), Key::V));
	transform_toolbar->add_child(transform_button_flip_v);
	transform_button_flip_v->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_apply_transform).bind(TRANSFORM_FLIP_V));
	viewport_shortcut_buttons.push_back(transform_button_flip_v);

	// Separator 2.
	tools_settings_vsep_2 = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep_2);

	// Continuous checkbox.
	bucket_contiguous_checkbox = memnew(CheckBox);
	bucket_contiguous_checkbox->set_flat(true);
	bucket_contiguous_checkbox->set_text(TTR("Contiguous"));
	bucket_contiguous_checkbox->set_pressed(true);
	tools_settings->add_child(bucket_contiguous_checkbox);

	// Random tile checkbox.
	random_tile_toggle = memnew(Button);
	random_tile_toggle->set_theme_type_variation("FlatButton");
	random_tile_toggle->set_toggle_mode(true);
	random_tile_toggle->set_tooltip_text(TTR("Place Random Tile"));
	random_tile_toggle->connect("toggled", callable_mp(this, &TileMapEditorTilesPlugin::_on_random_tile_checkbox_toggled));
	tools_settings->add_child(random_tile_toggle);

	// Random tile scattering.
	scatter_controls_container = memnew(HBoxContainer);

	scatter_label = memnew(Label);
	scatter_label->set_tooltip_text(TTR("Modifies the chance of painting nothing instead of a randomly selected tile."));
	scatter_label->set_text(TTR("Scattering:"));
	scatter_controls_container->add_child(scatter_label);

	scatter_spinbox = memnew(SpinBox);
	scatter_spinbox->set_min(0.0);
	scatter_spinbox->set_max(1000);
	scatter_spinbox->set_step(0.001);
	scatter_spinbox->set_tooltip_text(TTR("Modifies the chance of painting nothing instead of a randomly selected tile."));
	scatter_spinbox->get_line_edit()->add_theme_constant_override("minimum_character_width", 4);
	scatter_spinbox->connect("value_changed", callable_mp(this, &TileMapEditorTilesPlugin::_on_scattering_spinbox_changed));
	scatter_controls_container->add_child(scatter_spinbox);
	tools_settings->add_child(scatter_controls_container);

	_on_random_tile_checkbox_toggled(false);

	// Default tool.
	paint_tool_button->set_pressed(true);
	_update_toolbar();

	// --- Bottom panel tiles ---
	tiles_bottom_panel = memnew(VBoxContainer);
	tiles_bottom_panel->connect("tree_entered", callable_mp(this, &TileMapEditorTilesPlugin::_update_theme));
	tiles_bottom_panel->connect("theme_changed", callable_mp(this, &TileMapEditorTilesPlugin::_update_theme));
	tiles_bottom_panel->connect("visibility_changed", callable_mp(this, &TileMapEditorTilesPlugin::_stop_dragging));
	tiles_bottom_panel->connect("visibility_changed", callable_mp(this, &TileMapEditorTilesPlugin::_tab_changed));
	tiles_bottom_panel->set_name(TTR("Tiles"));

	missing_source_label = memnew(Label);
	missing_source_label->set_text(TTR("This TileMap's TileSet has no source configured. Go to the TileSet bottom tab to add one."));
	missing_source_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	missing_source_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	missing_source_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	missing_source_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	missing_source_label->hide();
	tiles_bottom_panel->add_child(missing_source_label);

	atlas_sources_split_container = memnew(HSplitContainer);
	atlas_sources_split_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	atlas_sources_split_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tiles_bottom_panel->add_child(atlas_sources_split_container);

	VBoxContainer *split_container_left_side = memnew(VBoxContainer);
	split_container_left_side->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	split_container_left_side->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	split_container_left_side->set_stretch_ratio(0.25);
	split_container_left_side->set_custom_minimum_size(Size2(70, 0) * EDSCALE);
	atlas_sources_split_container->add_child(split_container_left_side);

	HBoxContainer *sources_bottom_actions = memnew(HBoxContainer);
	sources_bottom_actions->set_alignment(HBoxContainer::ALIGNMENT_END);

	source_sort_button = memnew(MenuButton);
	source_sort_button->set_flat(false);
	source_sort_button->set_theme_type_variation("FlatMenuButton");
	source_sort_button->set_tooltip_text(TTR("Sort sources"));

	PopupMenu *p = source_sort_button->get_popup();
	p->connect("id_pressed", callable_mp(this, &TileMapEditorTilesPlugin::_set_source_sort));
	p->add_radio_check_item(TTR("Sort by ID (Ascending)"), TilesEditorUtils::SOURCE_SORT_ID);
	p->add_radio_check_item(TTR("Sort by ID (Descending)"), TilesEditorUtils::SOURCE_SORT_ID_REVERSE);
	p->add_radio_check_item(TTR("Sort by Name (Ascending)"), TilesEditorUtils::SOURCE_SORT_NAME);
	p->add_radio_check_item(TTR("Sort by Name (Descending)"), TilesEditorUtils::SOURCE_SORT_NAME_REVERSE);
	p->set_item_checked(TilesEditorUtils::SOURCE_SORT_ID, true);
	sources_bottom_actions->add_child(source_sort_button);

	sources_list = memnew(ItemList);
	sources_list->set_fixed_icon_size(Size2(60, 60) * EDSCALE);
	sources_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	sources_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	sources_list->set_stretch_ratio(0.25);
	sources_list->set_custom_minimum_size(Size2(70, 0) * EDSCALE);
	sources_list->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditorTilesPlugin::_update_fix_selected_and_hovered).unbind(1));
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditorTilesPlugin::_update_source_display).unbind(1));
	sources_list->connect("item_selected", callable_mp(TilesEditorUtils::get_singleton(), &TilesEditorUtils::set_sources_lists_current));
	sources_list->connect("item_activated", callable_mp(TilesEditorUtils::get_singleton(), &TilesEditorUtils::display_tile_set_editor_panel).unbind(1));
	sources_list->connect("visibility_changed", callable_mp(TilesEditorUtils::get_singleton(), &TilesEditorUtils::synchronize_sources_list).bind(sources_list, source_sort_button));
	sources_list->add_user_signal(MethodInfo("sort_request"));
	sources_list->connect("sort_request", callable_mp(this, &TileMapEditorTilesPlugin::_update_tile_set_sources_list));
	split_container_left_side->add_child(sources_list);
	split_container_left_side->add_child(sources_bottom_actions);

	// Tile atlas source.
	tile_atlas_view = memnew(TileAtlasView);
	tile_atlas_view->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tile_atlas_view->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	tile_atlas_view->set_texture_grid_visible(false);
	tile_atlas_view->set_tile_shape_grid_visible(false);
	tile_atlas_view->connect("transform_changed", callable_mp(TilesEditorUtils::get_singleton(), &TilesEditorUtils::set_atlas_view_transform));
	atlas_sources_split_container->add_child(tile_atlas_view);

	tile_atlas_control = memnew(Control);
	tile_atlas_control->connect("draw", callable_mp(this, &TileMapEditorTilesPlugin::_tile_atlas_control_draw));
	tile_atlas_control->connect("mouse_exited", callable_mp(this, &TileMapEditorTilesPlugin::_tile_atlas_control_mouse_exited));
	tile_atlas_control->connect("gui_input", callable_mp(this, &TileMapEditorTilesPlugin::_tile_atlas_control_gui_input));
	tile_atlas_view->add_control_over_atlas_tiles(tile_atlas_control);

	alternative_tiles_control = memnew(Control);
	alternative_tiles_control->connect("draw", callable_mp(this, &TileMapEditorTilesPlugin::_tile_alternatives_control_draw));
	alternative_tiles_control->connect("mouse_exited", callable_mp(this, &TileMapEditorTilesPlugin::_tile_alternatives_control_mouse_exited));
	alternative_tiles_control->connect("gui_input", callable_mp(this, &TileMapEditorTilesPlugin::_tile_alternatives_control_gui_input));
	tile_atlas_view->add_control_over_alternative_tiles(alternative_tiles_control);

	// Scenes collection source.
	scene_tiles_list = memnew(ItemList);
	scene_tiles_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	scene_tiles_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	scene_tiles_list->set_select_mode(ItemList::SELECT_MULTI);
	scene_tiles_list->connect("multi_selected", callable_mp(this, &TileMapEditorTilesPlugin::_scenes_list_multi_selected));
	scene_tiles_list->connect("empty_clicked", callable_mp(this, &TileMapEditorTilesPlugin::_scenes_list_lmb_empty_clicked));
	scene_tiles_list->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
	atlas_sources_split_container->add_child(scene_tiles_list);

	// Invalid source label.
	invalid_source_label = memnew(Label);
	invalid_source_label->set_text(TTR("Invalid source selected."));
	invalid_source_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	invalid_source_label->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	invalid_source_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	invalid_source_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	invalid_source_label->hide();
	atlas_sources_split_container->add_child(invalid_source_label);

	// --- Bottom panel patterns ---
	patterns_bottom_panel = memnew(VBoxContainer);
	patterns_bottom_panel->set_name(TTR("Patterns"));
	patterns_bottom_panel->connect("visibility_changed", callable_mp(this, &TileMapEditorTilesPlugin::_tab_changed));

	int thumbnail_size = 64;
	patterns_item_list = memnew(ItemList);
	patterns_item_list->set_max_columns(0);
	patterns_item_list->set_icon_mode(ItemList::ICON_MODE_TOP);
	patterns_item_list->set_fixed_column_width(thumbnail_size * 3 / 2);
	patterns_item_list->set_max_text_lines(2);
	patterns_item_list->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));
	patterns_item_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	patterns_item_list->connect("gui_input", callable_mp(this, &TileMapEditorTilesPlugin::_patterns_item_list_gui_input));
	patterns_item_list->connect("item_selected", callable_mp(this, &TileMapEditorTilesPlugin::_update_selection_pattern_from_tileset_pattern_selection).unbind(1));
	patterns_item_list->connect("item_activated", callable_mp(this, &TileMapEditorTilesPlugin::_update_selection_pattern_from_tileset_pattern_selection).unbind(1));
	patterns_item_list->connect("empty_clicked", callable_mp(this, &TileMapEditorTilesPlugin::patterns_item_list_empty_clicked));
	patterns_bottom_panel->add_child(patterns_item_list);

	patterns_help_label = memnew(Label);
	patterns_help_label->set_text(TTR("Drag and drop or paste a TileMap selection here to store a pattern."));
	patterns_help_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	patterns_help_label->set_anchors_and_offsets_preset(Control::PRESET_CENTER);
	patterns_item_list->add_child(patterns_help_label);

	// Update.
	_update_source_display();
}

TileMapEditorTilesPlugin::~TileMapEditorTilesPlugin() {
}

void TileMapEditorTerrainsPlugin::tile_set_changed() {
	_update_terrains_cache();
	_update_terrains_tree();
	_update_tiles_list();
}

void TileMapEditorTerrainsPlugin::_update_toolbar() {
	// Hide all settings.
	for (int i = 0; i < tools_settings->get_child_count(); i++) {
		Object::cast_to<CanvasItem>(tools_settings->get_child(i))->hide();
	}

	// Show only the correct settings.
	if (tool_buttons_group->get_pressed_button() != bucket_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
	} else {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		tools_settings_vsep_2->show();
		bucket_contiguous_checkbox->show();
	}
}

Vector<TileMapSubEditorPlugin::TabData> TileMapEditorTerrainsPlugin::get_tabs() const {
	Vector<TileMapSubEditorPlugin::TabData> tabs;
	tabs.push_back({ toolbar, main_vbox_container });
	return tabs;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTerrainsPlugin::_draw_terrain_path_or_connect(const Vector<Vector2i> &p_to_paint, int p_terrain_set, int p_terrain, bool p_connect) const {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output;
	if (p_connect) {
		terrain_fill_output = tile_map->terrain_fill_connect(tile_map_layer, p_to_paint, p_terrain_set, p_terrain, false);
	} else {
		terrain_fill_output = tile_map->terrain_fill_path(tile_map_layer, p_to_paint, p_terrain_set, p_terrain, false);
	}

	// Make the painted path a set for faster lookups
	HashSet<Vector2i> painted_set;
	for (Vector2i coords : p_to_paint) {
		painted_set.insert(coords);
	}

	HashMap<Vector2i, TileMapCell> output;
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			output[kv.key] = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = tile_map->get_cell(tile_map_layer, kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				output[kv.key] = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			}
		}
	}
	return output;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTerrainsPlugin::_draw_terrain_pattern(const Vector<Vector2i> &p_to_paint, int p_terrain_set, TileSet::TerrainsPattern p_terrains_pattern) const {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	HashMap<Vector2i, TileSet::TerrainsPattern> terrain_fill_output = tile_map->terrain_fill_pattern(tile_map_layer, p_to_paint, p_terrain_set, p_terrains_pattern, false);

	// Make the painted path a set for faster lookups
	HashSet<Vector2i> painted_set;
	for (Vector2i coords : p_to_paint) {
		painted_set.insert(coords);
	}

	HashMap<Vector2i, TileMapCell> output;
	for (const KeyValue<Vector2i, TileSet::TerrainsPattern> &kv : terrain_fill_output) {
		if (painted_set.has(kv.key)) {
			// Paint a random tile with the correct terrain for the painted path.
			output[kv.key] = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
		} else {
			// Avoids updating the painted path from the output if the new pattern is the same as before.
			TileSet::TerrainsPattern in_map_terrain_pattern = TileSet::TerrainsPattern(*tile_set, p_terrain_set);
			TileMapCell cell = tile_map->get_cell(tile_map_layer, kv.key);
			if (cell.source_id != TileSet::INVALID_SOURCE) {
				TileSetSource *source = *tile_set->get_source(cell.source_id);
				TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
				if (atlas_source) {
					// Get tile data.
					TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
					if (tile_data && tile_data->get_terrain_set() == p_terrain_set) {
						in_map_terrain_pattern = tile_data->get_terrains_pattern();
					}
				}
			}
			if (in_map_terrain_pattern != kv.value) {
				output[kv.key] = tile_set->get_random_tile_from_terrains_pattern(p_terrain_set, kv.value);
			}
		}
	}
	return output;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTerrainsPlugin::_draw_line(Vector2i p_start_cell, Vector2i p_end_cell, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	if (p_erase) {
		return _draw_terrain_pattern(TileMapEditor::get_line(tile_map, p_start_cell, p_end_cell), selected_terrain_set, TileSet::TerrainsPattern(*tile_set, selected_terrain_set));
	} else {
		if (selected_type == SELECTED_TYPE_CONNECT) {
			return _draw_terrain_path_or_connect(TileMapEditor::get_line(tile_map, p_start_cell, p_end_cell), selected_terrain_set, selected_terrain, true);
		} else if (selected_type == SELECTED_TYPE_PATH) {
			return _draw_terrain_path_or_connect(TileMapEditor::get_line(tile_map, p_start_cell, p_end_cell), selected_terrain_set, selected_terrain, false);
		} else { // SELECTED_TYPE_PATTERN
			return _draw_terrain_pattern(TileMapEditor::get_line(tile_map, p_start_cell, p_end_cell), selected_terrain_set, selected_terrains_pattern);
		}
	}
}

HashMap<Vector2i, TileMapCell> TileMapEditorTerrainsPlugin::_draw_rect(Vector2i p_start_cell, Vector2i p_end_cell, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Rect2i rect;
	rect.set_position(p_start_cell);
	rect.set_end(p_end_cell);
	rect = rect.abs();

	Vector<Vector2i> to_draw;
	for (int x = rect.position.x; x <= rect.get_end().x; x++) {
		for (int y = rect.position.y; y <= rect.get_end().y; y++) {
			to_draw.append(Vector2i(x, y));
		}
	}

	if (p_erase) {
		return _draw_terrain_pattern(to_draw, selected_terrain_set, TileSet::TerrainsPattern(*tile_set, selected_terrain_set));
	} else {
		if (selected_type == SELECTED_TYPE_CONNECT || selected_type == SELECTED_TYPE_PATH) {
			return _draw_terrain_path_or_connect(to_draw, selected_terrain_set, selected_terrain, true);
		} else { // SELECTED_TYPE_PATTERN
			return _draw_terrain_pattern(to_draw, selected_terrain_set, selected_terrains_pattern);
		}
	}
}

RBSet<Vector2i> TileMapEditorTerrainsPlugin::_get_cells_for_bucket_fill(Vector2i p_coords, bool p_contiguous) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return RBSet<Vector2i>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return RBSet<Vector2i>();
	}

	TileMapCell source_cell = tile_map->get_cell(tile_map_layer, p_coords);

	TileSet::TerrainsPattern source_pattern(*tile_set, selected_terrain_set);
	if (source_cell.source_id != TileSet::INVALID_SOURCE) {
		TileData *tile_data = nullptr;
		Ref<TileSetSource> source = tile_set->get_source(source_cell.source_id);
		Ref<TileSetAtlasSource> atlas_source = source;
		if (atlas_source.is_valid()) {
			tile_data = atlas_source->get_tile_data(source_cell.get_atlas_coords(), source_cell.alternative_tile);
		}
		if (!tile_data) {
			return RBSet<Vector2i>();
		}
		source_pattern = tile_data->get_terrains_pattern();
	}

	// If we are filling empty tiles, compute the tilemap boundaries.
	Rect2i boundaries;
	if (source_cell.source_id == TileSet::INVALID_SOURCE) {
		boundaries = tile_map->get_used_rect();
	}

	RBSet<Vector2i> output;
	if (p_contiguous) {
		// Replace continuous tiles like the source.
		RBSet<Vector2i> already_checked;
		List<Vector2i> to_check;
		to_check.push_back(p_coords);
		while (!to_check.is_empty()) {
			Vector2i coords = to_check.back()->get();
			to_check.pop_back();
			if (!already_checked.has(coords)) {
				// Get the candidate cell pattern.
				TileSet::TerrainsPattern candidate_pattern(*tile_set, selected_terrain_set);
				if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
					TileData *tile_data = nullptr;
					Ref<TileSetSource> source = tile_set->get_source(tile_map->get_cell_source_id(tile_map_layer, coords));
					Ref<TileSetAtlasSource> atlas_source = source;
					if (atlas_source.is_valid()) {
						tile_data = atlas_source->get_tile_data(tile_map->get_cell_atlas_coords(tile_map_layer, coords), tile_map->get_cell_alternative_tile(tile_map_layer, coords));
					}
					if (tile_data) {
						candidate_pattern = tile_data->get_terrains_pattern();
					}
				}

				// Draw.
				if (candidate_pattern == source_pattern && (!source_pattern.is_erase_pattern() || boundaries.has_point(coords))) {
					output.insert(coords);

					// Get surrounding tiles (handles different tile shapes).
					TypedArray<Vector2i> around = tile_map->get_surrounding_cells(coords);
					for (int i = 0; i < around.size(); i++) {
						to_check.push_back(around[i]);
					}
				}
				already_checked.insert(coords);
			}
		}
	} else {
		// Replace all tiles like the source.
		TypedArray<Vector2i> to_check;
		if (source_cell.source_id == TileSet::INVALID_SOURCE) {
			Rect2i rect = tile_map->get_used_rect();
			if (!rect.has_area()) {
				rect = Rect2i(p_coords, Vector2i(1, 1));
			}
			for (int x = boundaries.position.x; x < boundaries.get_end().x; x++) {
				for (int y = boundaries.position.y; y < boundaries.get_end().y; y++) {
					to_check.append(Vector2i(x, y));
				}
			}
		} else {
			to_check = tile_map->get_used_cells(tile_map_layer);
		}
		for (int i = 0; i < to_check.size(); i++) {
			Vector2i coords = to_check[i];
			// Get the candidate cell pattern.
			TileSet::TerrainsPattern candidate_pattern;
			if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
				TileData *tile_data = nullptr;
				Ref<TileSetSource> source = tile_set->get_source(tile_map->get_cell_source_id(tile_map_layer, coords));
				Ref<TileSetAtlasSource> atlas_source = source;
				if (atlas_source.is_valid()) {
					tile_data = atlas_source->get_tile_data(tile_map->get_cell_atlas_coords(tile_map_layer, coords), tile_map->get_cell_alternative_tile(tile_map_layer, coords));
				}
				if (tile_data) {
					candidate_pattern = tile_data->get_terrains_pattern();
				}
			}

			// Draw.
			if (candidate_pattern == source_pattern && (!source_pattern.is_erase_pattern() || boundaries.has_point(coords))) {
				output.insert(coords);
			}
		}
	}
	return output;
}

HashMap<Vector2i, TileMapCell> TileMapEditorTerrainsPlugin::_draw_bucket_fill(Vector2i p_coords, bool p_contiguous, bool p_erase) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return HashMap<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return HashMap<Vector2i, TileMapCell>();
	}

	RBSet<Vector2i> cells_to_draw = _get_cells_for_bucket_fill(p_coords, p_contiguous);
	Vector<Vector2i> cells_to_draw_as_vector;
	for (Vector2i cell : cells_to_draw) {
		cells_to_draw_as_vector.append(cell);
	}

	if (p_erase) {
		return _draw_terrain_pattern(cells_to_draw_as_vector, selected_terrain_set, TileSet::TerrainsPattern(*tile_set, selected_terrain_set));
	} else {
		if (selected_type == SELECTED_TYPE_CONNECT || selected_type == SELECTED_TYPE_PATH) {
			return _draw_terrain_path_or_connect(cells_to_draw_as_vector, selected_terrain_set, selected_terrain, true);
		} else { // SELECTED_TYPE_PATTERN
			return _draw_terrain_pattern(cells_to_draw_as_vector, selected_terrain_set, selected_terrains_pattern);
		}
	}
}

void TileMapEditorTerrainsPlugin::_stop_dragging() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
	Vector2 mpos = xform.affine_inverse().xform(CanvasItemEditor::get_singleton()->get_viewport_control()->get_local_mouse_position());

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	switch (drag_type) {
		case DRAG_TYPE_PICK: {
			Vector2i coords = tile_map->local_to_map(mpos);
			TileMapCell cell = tile_map->get_cell(tile_map_layer, coords);
			TileData *tile_data = nullptr;

			Ref<TileSetSource> source = tile_set->get_source(cell.source_id);
			Ref<TileSetAtlasSource> atlas_source = source;
			if (atlas_source.is_valid()) {
				tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
			}

			if (tile_data) {
				TileSet::TerrainsPattern terrains_pattern = tile_data->get_terrains_pattern();

				// Find the tree item for the right terrain set.
				bool need_tree_item_switch = true;
				TreeItem *tree_item = terrains_tree->get_selected();
				int new_terrain_set = -1;
				if (tree_item) {
					Dictionary metadata_dict = tree_item->get_metadata(0);
					if (metadata_dict.has("terrain_set") && metadata_dict.has("terrain_id")) {
						int terrain_set = metadata_dict["terrain_set"];
						int terrain_id = metadata_dict["terrain_id"];
						if (per_terrain_terrains_patterns[terrain_set][terrain_id].has(terrains_pattern)) {
							new_terrain_set = terrain_set;
							need_tree_item_switch = false;
						}
					}
				}

				if (need_tree_item_switch) {
					for (tree_item = terrains_tree->get_root()->get_first_child(); tree_item; tree_item = tree_item->get_next_visible()) {
						Dictionary metadata_dict = tree_item->get_metadata(0);
						if (metadata_dict.has("terrain_set") && metadata_dict.has("terrain_id")) {
							int terrain_set = metadata_dict["terrain_set"];
							int terrain_id = metadata_dict["terrain_id"];
							if (per_terrain_terrains_patterns[terrain_set][terrain_id].has(terrains_pattern)) {
								// Found
								new_terrain_set = terrain_set;
								tree_item->select(0);
								_update_tiles_list();
								break;
							}
						}
					}
				}

				// Find the list item for the given tile.
				if (tree_item) {
					for (int i = 0; i < terrains_tile_list->get_item_count(); i++) {
						Dictionary metadata_dict = terrains_tile_list->get_item_metadata(i);
						if (int(metadata_dict["type"]) == SELECTED_TYPE_PATTERN) {
							TileSet::TerrainsPattern in_meta_terrains_pattern(*tile_set, new_terrain_set);
							in_meta_terrains_pattern.from_array(metadata_dict["terrains_pattern"]);
							if (in_meta_terrains_pattern == terrains_pattern) {
								terrains_tile_list->select(i);
								break;
							}
						}
					}
				} else {
					ERR_PRINT("Terrain tile not found.");
				}
			}
			picker_button->set_pressed(false);
		} break;
		case DRAG_TYPE_PAINT: {
			undo_redo->create_action(TTR("Paint terrain"));
			for (const KeyValue<Vector2i, TileMapCell> &E : drag_modified) {
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
			}
			undo_redo->commit_action(false);
		} break;
		case DRAG_TYPE_LINE: {
			HashMap<Vector2i, TileMapCell> to_draw = _draw_line(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos), drag_erasing);
			undo_redo->create_action(TTR("Paint terrain"));
			for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
				if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
					continue;
				}
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_RECT: {
			HashMap<Vector2i, TileMapCell> to_draw = _draw_rect(tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos), drag_erasing);
			undo_redo->create_action(TTR("Paint terrain"));
			for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
				if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
					continue;
				}
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_BUCKET: {
			undo_redo->create_action(TTR("Paint terrain"));
			for (const KeyValue<Vector2i, TileMapCell> &E : drag_modified) {
				undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, E.key, tile_map->get_cell_source_id(tile_map_layer, E.key), tile_map->get_cell_atlas_coords(tile_map_layer, E.key), tile_map->get_cell_alternative_tile(tile_map_layer, E.key));
				undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
			}
			undo_redo->commit_action(false);
		} break;

		default:
			break;
	}
	drag_type = DRAG_TYPE_NONE;
}

void TileMapEditorTerrainsPlugin::_mouse_exited_viewport() {
	has_mouse = false;
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditorTerrainsPlugin::_update_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Get the selected terrain.
	selected_terrain_set = -1;
	selected_terrains_pattern = TileSet::TerrainsPattern();

	TreeItem *selected_tree_item = terrains_tree->get_selected();
	if (selected_tree_item && selected_tree_item->get_metadata(0)) {
		Dictionary metadata_dict = selected_tree_item->get_metadata(0);
		// Selected terrain
		selected_terrain_set = metadata_dict["terrain_set"];
		selected_terrain = metadata_dict["terrain_id"];

		// Selected mode/terrain pattern
		if (erase_button->is_pressed()) {
			selected_type = SELECTED_TYPE_PATTERN;
			selected_terrains_pattern = TileSet::TerrainsPattern(*tile_set, selected_terrain_set);
		} else if (terrains_tile_list->is_anything_selected()) {
			metadata_dict = terrains_tile_list->get_item_metadata(terrains_tile_list->get_selected_items()[0]);
			if (int(metadata_dict["type"]) == SELECTED_TYPE_CONNECT) {
				selected_type = SELECTED_TYPE_CONNECT;
			} else if (int(metadata_dict["type"]) == SELECTED_TYPE_PATH) {
				selected_type = SELECTED_TYPE_PATH;
			} else if (int(metadata_dict["type"]) == SELECTED_TYPE_PATTERN) {
				selected_type = SELECTED_TYPE_PATTERN;
				selected_terrains_pattern = TileSet::TerrainsPattern(*tile_set, selected_terrain_set);
				selected_terrains_pattern.from_array(metadata_dict["terrains_pattern"]);
			} else {
				ERR_FAIL();
			}
		}
	}
}

bool TileMapEditorTerrainsPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!main_vbox_container->is_visible_in_tree()) {
		// If the bottom editor is not visible, we ignore inputs.
		return false;
	}

	if (CanvasItemEditor::get_singleton()->get_current_tool() != CanvasItemEditor::TOOL_SELECT) {
		return false;
	}

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return false;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return false;
	}

	if (tile_map_layer < 0) {
		return false;
	}
	ERR_FAIL_COND_V(tile_map_layer >= tile_map->get_layers_count(), false);

	_update_selection();

	Ref<InputEventKey> k = p_event;
	if (k.is_valid() && k->is_pressed() && !k->is_echo()) {
		for (BaseButton *b : viewport_shortcut_buttons) {
			if (b->get_shortcut().is_valid() && b->get_shortcut()->matches_event(p_event)) {
				b->set_pressed(b->get_button_group().is_valid() || !b->is_pressed());
				return true;
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		has_mouse = true;
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
		Vector2 mpos = xform.affine_inverse().xform(mm->get_position());

		switch (drag_type) {
			case DRAG_TYPE_PAINT: {
				if (selected_terrain_set >= 0) {
					HashMap<Vector2i, TileMapCell> to_draw = _draw_line(tile_map->local_to_map(drag_last_mouse_pos), tile_map->local_to_map(mpos), drag_erasing);
					for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
						if (!drag_modified.has(E.key)) {
							drag_modified[E.key] = tile_map->get_cell(tile_map_layer, E.key);
						}
						tile_map->set_cell(tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
					}
				}
			} break;
			default:
				break;
		}
		drag_last_mouse_pos = mpos;
		CanvasItemEditor::get_singleton()->update_viewport();

		return true;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		has_mouse = true;
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
		Vector2 mpos = xform.affine_inverse().xform(mb->get_position());

		if (mb->get_button_index() == MouseButton::LEFT || mb->get_button_index() == MouseButton::RIGHT) {
			if (mb->is_pressed()) {
				// Pressed
				if (erase_button->is_pressed() || mb->get_button_index() == MouseButton::RIGHT) {
					drag_erasing = true;
				}

				if (picker_button->is_pressed()) {
					drag_type = DRAG_TYPE_PICK;
				} else {
					// Paint otherwise.
					if (tool_buttons_group->get_pressed_button() == paint_tool_button && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
						if (selected_terrain_set < 0 || selected_terrain < 0 || (selected_type == SELECTED_TYPE_PATTERN && !selected_terrains_pattern.is_valid())) {
							return true;
						}

						drag_type = DRAG_TYPE_PAINT;
						drag_start_mouse_pos = mpos;

						drag_modified.clear();
						Vector2i cell = tile_map->local_to_map(mpos);
						HashMap<Vector2i, TileMapCell> to_draw = _draw_line(cell, cell, drag_erasing);
						for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
							drag_modified[E.key] = tile_map->get_cell(tile_map_layer, E.key);
							tile_map->set_cell(tile_map_layer, E.key, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
						}
					} else if (tool_buttons_group->get_pressed_button() == line_tool_button || (tool_buttons_group->get_pressed_button() == paint_tool_button && Input::get_singleton()->is_key_pressed(Key::SHIFT) && !Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL))) {
						if (selected_terrain_set < 0 || selected_terrain < 0 || (selected_type == SELECTED_TYPE_PATTERN && !selected_terrains_pattern.is_valid())) {
							return true;
						}
						drag_type = DRAG_TYPE_LINE;
						drag_start_mouse_pos = mpos;
						drag_modified.clear();
					} else if (tool_buttons_group->get_pressed_button() == rect_tool_button || (tool_buttons_group->get_pressed_button() == paint_tool_button && Input::get_singleton()->is_key_pressed(Key::SHIFT) && Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL))) {
						if (selected_terrain_set < 0 || selected_terrain < 0 || (selected_type == SELECTED_TYPE_PATTERN && !selected_terrains_pattern.is_valid())) {
							return true;
						}
						drag_type = DRAG_TYPE_RECT;
						drag_start_mouse_pos = mpos;
						drag_modified.clear();
					} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button) {
						if (selected_terrain_set < 0 || selected_terrain < 0 || (selected_type == SELECTED_TYPE_PATTERN && !selected_terrains_pattern.is_valid())) {
							return true;
						}
						drag_type = DRAG_TYPE_BUCKET;
						drag_start_mouse_pos = mpos;
						drag_modified.clear();
						Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->local_to_map(drag_last_mouse_pos), tile_map->local_to_map(mpos));
						for (int i = 0; i < line.size(); i++) {
							if (!drag_modified.has(line[i])) {
								HashMap<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], bucket_contiguous_checkbox->is_pressed(), drag_erasing);
								for (const KeyValue<Vector2i, TileMapCell> &E : to_draw) {
									if (!drag_erasing && E.value.source_id == TileSet::INVALID_SOURCE) {
										continue;
									}
									Vector2i coords = E.key;
									if (!drag_modified.has(coords)) {
										drag_modified.insert(coords, tile_map->get_cell(tile_map_layer, coords));
									}
									tile_map->set_cell(tile_map_layer, coords, E.value.source_id, E.value.get_atlas_coords(), E.value.alternative_tile);
								}
							}
						}
					}
				}
			} else {
				// Released
				_stop_dragging();
				drag_erasing = false;
			}

			CanvasItemEditor::get_singleton()->update_viewport();

			return true;
		}
		drag_last_mouse_pos = mpos;
	}

	return false;
}

void TileMapEditorTerrainsPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	if (tile_map_layer < 0) {
		return;
	}
	ERR_FAIL_INDEX(tile_map_layer, tile_map->get_layers_count());

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (!tile_map->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
	Vector2 mpos = tile_map->get_local_mouse_position();
	Vector2i tile_shape_size = tile_set->get_tile_size();

	// Handle the preview of the tiles to be placed.
	if (main_vbox_container->is_visible_in_tree() && has_mouse) { // Only if the tilemap editor is opened and the viewport is hovered.
		RBSet<Vector2i> preview;
		Rect2i drawn_grid_rect;

		if (drag_type == DRAG_TYPE_PICK) {
			// Draw the area being picked.
			Vector2i coords = tile_map->local_to_map(mpos);
			if (tile_map->get_cell_source_id(tile_map_layer, coords) != TileSet::INVALID_SOURCE) {
				Transform2D tile_xform;
				tile_xform.set_origin(tile_map->map_to_local(coords));
				tile_xform.set_scale(tile_shape_size);
				tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0), false);
			}
		} else if (!picker_button->is_pressed() && !(drag_type == DRAG_TYPE_NONE && Input::get_singleton()->is_key_pressed(Key::CMD_OR_CTRL) && !Input::get_singleton()->is_key_pressed(Key::SHIFT))) {
			bool expand_grid = false;
			if (tool_buttons_group->get_pressed_button() == paint_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a single tile.
				preview.insert(tile_map->local_to_map(mpos));
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == line_tool_button || drag_type == DRAG_TYPE_LINE) {
				if (drag_type == DRAG_TYPE_NONE) {
					// Preview for a single tile.
					preview.insert(tile_map->local_to_map(mpos));
				} else if (drag_type == DRAG_TYPE_LINE) {
					// Preview for a line.
					Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->local_to_map(drag_start_mouse_pos), tile_map->local_to_map(mpos));
					for (int i = 0; i < line.size(); i++) {
						preview.insert(line[i]);
					}
					expand_grid = true;
				}
			} else if (drag_type == DRAG_TYPE_RECT) {
				// Preview for a rect.
				Rect2i rect;
				rect.set_position(tile_map->local_to_map(drag_start_mouse_pos));
				rect.set_end(tile_map->local_to_map(mpos));
				rect = rect.abs();

				HashMap<Vector2i, TileSet::TerrainsPattern> to_draw;
				for (int x = rect.position.x; x <= rect.get_end().x; x++) {
					for (int y = rect.position.y; y <= rect.get_end().y; y++) {
						preview.insert(Vector2i(x, y));
					}
				}
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a fill.
				preview = _get_cells_for_bucket_fill(tile_map->local_to_map(mpos), bucket_contiguous_checkbox->is_pressed());
			}

			// Expand the grid if needed
			if (expand_grid && !preview.is_empty()) {
				drawn_grid_rect = Rect2i(preview.front()->get(), Vector2i(1, 1));
				for (const Vector2i &E : preview) {
					drawn_grid_rect.expand_to(E);
				}
			}
		}

		if (!preview.is_empty()) {
			const int fading = 5;

			// Draw the lines of the grid behind the preview.
			bool display_grid = EDITOR_GET("editors/tiles_editor/display_grid");
			if (display_grid) {
				Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
				if (drawn_grid_rect.size.x > 0 && drawn_grid_rect.size.y > 0) {
					drawn_grid_rect = drawn_grid_rect.grow(fading);
					for (int x = drawn_grid_rect.position.x; x < (drawn_grid_rect.position.x + drawn_grid_rect.size.x); x++) {
						for (int y = drawn_grid_rect.position.y; y < (drawn_grid_rect.position.y + drawn_grid_rect.size.y); y++) {
							Vector2i pos_in_rect = Vector2i(x, y) - drawn_grid_rect.position;

							// Fade out the border of the grid.
							float left_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.x), 0.0f, 1.0f);
							float right_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.x, (float)(drawn_grid_rect.size.x - fading), (float)(pos_in_rect.x + 1)), 0.0f, 1.0f);
							float top_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.y), 0.0f, 1.0f);
							float bottom_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.y, (float)(drawn_grid_rect.size.y - fading), (float)(pos_in_rect.y + 1)), 0.0f, 1.0f);
							float opacity = CLAMP(MIN(left_opacity, MIN(right_opacity, MIN(top_opacity, bottom_opacity))) + 0.1, 0.0f, 1.0f);

							Transform2D tile_xform;
							tile_xform.set_origin(tile_map->map_to_local(Vector2(x, y)));
							tile_xform.set_scale(tile_shape_size);
							Color color = grid_color;
							color.a = color.a * opacity;
							tile_set->draw_tile_shape(p_overlay, xform * tile_xform, color, false);
						}
					}
				}
			}

			// Draw the preview.
			for (const Vector2i &E : preview) {
				Transform2D tile_xform;
				tile_xform.set_origin(tile_map->map_to_local(E));
				tile_xform.set_scale(tile_set->get_tile_size());
				if (drag_erasing || erase_button->is_pressed()) {
					tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(0.0, 0.0, 0.0, 0.5), true);
				} else {
					tile_set->draw_tile_shape(p_overlay, xform * tile_xform, Color(1.0, 1.0, 1.0, 0.5), true);
				}
			}
		}
	}
}

void TileMapEditorTerrainsPlugin::_update_terrains_cache() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Organizes tiles into structures.
	per_terrain_terrains_patterns.resize(tile_set->get_terrain_sets_count());
	for (int i = 0; i < tile_set->get_terrain_sets_count(); i++) {
		per_terrain_terrains_patterns[i].resize(tile_set->get_terrains_count(i));
		for (RBSet<TileSet::TerrainsPattern> &pattern : per_terrain_terrains_patterns[i]) {
			pattern.clear();
		}
	}

	for (int source_index = 0; source_index < tile_set->get_source_count(); source_index++) {
		int source_id = tile_set->get_source_id(source_index);
		Ref<TileSetSource> source = tile_set->get_source(source_id);

		Ref<TileSetAtlasSource> atlas_source = source;
		if (atlas_source.is_valid()) {
			for (int tile_index = 0; tile_index < source->get_tiles_count(); tile_index++) {
				Vector2i tile_id = source->get_tile_id(tile_index);
				for (int alternative_index = 0; alternative_index < source->get_alternative_tiles_count(tile_id); alternative_index++) {
					int alternative_id = source->get_alternative_tile_id(tile_id, alternative_index);

					TileData *tile_data = atlas_source->get_tile_data(tile_id, alternative_id);
					int terrain_set = tile_data->get_terrain_set();
					if (terrain_set >= 0) {
						ERR_FAIL_INDEX(terrain_set, (int)per_terrain_terrains_patterns.size());

						TileMapCell cell;
						cell.source_id = source_id;
						cell.set_atlas_coords(tile_id);
						cell.alternative_tile = alternative_id;

						TileSet::TerrainsPattern terrains_pattern = tile_data->get_terrains_pattern();

						// Terrain center bit
						int terrain = terrains_pattern.get_terrain();
						if (terrain >= 0 && terrain < (int)per_terrain_terrains_patterns[terrain_set].size()) {
							per_terrain_terrains_patterns[terrain_set][terrain].insert(terrains_pattern);
						}

						// Terrain bits.
						for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
							TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
							if (tile_set->is_valid_terrain_peering_bit(terrain_set, bit)) {
								terrain = terrains_pattern.get_terrain_peering_bit(bit);
								if (terrain >= 0 && terrain < (int)per_terrain_terrains_patterns[terrain_set].size()) {
									per_terrain_terrains_patterns[terrain_set][terrain].insert(terrains_pattern);
								}
							}
						}
					}
				}
			}
		}
	}
}

void TileMapEditorTerrainsPlugin::_update_terrains_tree() {
	terrains_tree->clear();
	terrains_tree->create_item();

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Fill in the terrain list.
	Vector<Vector<Ref<Texture2D>>> icons = tile_set->generate_terrains_icons(Size2(16, 16) * EDSCALE);
	for (int terrain_set_index = 0; terrain_set_index < tile_set->get_terrain_sets_count(); terrain_set_index++) {
		// Add an item for the terrain set.
		TreeItem *terrain_set_tree_item = terrains_tree->create_item();
		String matches;
		if (tile_set->get_terrain_set_mode(terrain_set_index) == TileSet::TERRAIN_MODE_MATCH_CORNERS_AND_SIDES) {
			terrain_set_tree_item->set_icon(0, main_vbox_container->get_editor_theme_icon(SNAME("TerrainMatchCornersAndSides")));
			matches = String(TTR("Matches Corners and Sides"));
		} else if (tile_set->get_terrain_set_mode(terrain_set_index) == TileSet::TERRAIN_MODE_MATCH_CORNERS) {
			terrain_set_tree_item->set_icon(0, main_vbox_container->get_editor_theme_icon(SNAME("TerrainMatchCorners")));
			matches = String(TTR("Matches Corners Only"));
		} else {
			terrain_set_tree_item->set_icon(0, main_vbox_container->get_editor_theme_icon(SNAME("TerrainMatchSides")));
			matches = String(TTR("Matches Sides Only"));
		}
		terrain_set_tree_item->set_text(0, vformat(TTR("Terrain Set %d (%s)"), terrain_set_index, matches));
		terrain_set_tree_item->set_selectable(0, false);

		for (int terrain_index = 0; terrain_index < tile_set->get_terrains_count(terrain_set_index); terrain_index++) {
			// Add the item to the terrain list.
			TreeItem *terrain_tree_item = terrains_tree->create_item(terrain_set_tree_item);
			terrain_tree_item->set_text(0, tile_set->get_terrain_name(terrain_set_index, terrain_index));
			terrain_tree_item->set_icon_max_width(0, 32 * EDSCALE);
			terrain_tree_item->set_icon(0, icons[terrain_set_index][terrain_index]);

			Dictionary metadata_dict;
			metadata_dict["terrain_set"] = terrain_set_index;
			metadata_dict["terrain_id"] = terrain_index;
			terrain_tree_item->set_metadata(0, metadata_dict);
		}
	}
}

void TileMapEditorTerrainsPlugin::_update_tiles_list() {
	terrains_tile_list->clear();

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	TreeItem *selected_tree_item = terrains_tree->get_selected();
	if (selected_tree_item && selected_tree_item->get_metadata(0)) {
		Dictionary metadata_dict = selected_tree_item->get_metadata(0);
		int sel_terrain_set = metadata_dict["terrain_set"];
		int sel_terrain_id = metadata_dict["terrain_id"];
		ERR_FAIL_INDEX(sel_terrain_set, tile_set->get_terrain_sets_count());
		ERR_FAIL_INDEX(sel_terrain_id, tile_set->get_terrains_count(sel_terrain_set));

		// Add the two first generic modes
		int item_index = terrains_tile_list->add_icon_item(main_vbox_container->get_editor_theme_icon(SNAME("TerrainConnect")));
		terrains_tile_list->set_item_tooltip(item_index, TTR("Connect mode: paints a terrain, then connects it with the surrounding tiles with the same terrain."));
		Dictionary list_metadata_dict;
		list_metadata_dict["type"] = SELECTED_TYPE_CONNECT;
		terrains_tile_list->set_item_metadata(item_index, list_metadata_dict);

		item_index = terrains_tile_list->add_icon_item(main_vbox_container->get_editor_theme_icon(SNAME("TerrainPath")));
		terrains_tile_list->set_item_tooltip(item_index, TTR("Path mode: paints a terrain, thens connects it to the previous tile painted within the same stroke."));
		list_metadata_dict = Dictionary();
		list_metadata_dict["type"] = SELECTED_TYPE_PATH;
		terrains_tile_list->set_item_metadata(item_index, list_metadata_dict);

		// Sort the items in a map by the number of corresponding terrains.
		RBMap<int, RBSet<TileSet::TerrainsPattern>> sorted;

		for (const TileSet::TerrainsPattern &E : per_terrain_terrains_patterns[sel_terrain_set][sel_terrain_id]) {
			// Count the number of matching sides/terrains.
			int count = 0;

			for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
				TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
				if (tile_set->is_valid_terrain_peering_bit(sel_terrain_set, bit) && E.get_terrain_peering_bit(bit) == sel_terrain_id) {
					count++;
				}
			}
			sorted[count].insert(E);
		}

		for (RBMap<int, RBSet<TileSet::TerrainsPattern>>::Element *E_set = sorted.back(); E_set; E_set = E_set->prev()) {
			for (const TileSet::TerrainsPattern &E : E_set->get()) {
				TileSet::TerrainsPattern terrains_pattern = E;

				// Get the icon.
				Ref<Texture2D> icon;
				Rect2 region;
				bool transpose = false;

				double max_probability = -1.0;
				for (const TileMapCell &cell : tile_set->get_tiles_for_terrains_pattern(sel_terrain_set, terrains_pattern)) {
					Ref<TileSetSource> source = tile_set->get_source(cell.source_id);

					Ref<TileSetAtlasSource> atlas_source = source;
					if (atlas_source.is_valid()) {
						TileData *tile_data = atlas_source->get_tile_data(cell.get_atlas_coords(), cell.alternative_tile);
						if (tile_data->get_probability() > max_probability) {
							icon = atlas_source->get_texture();
							region = atlas_source->get_tile_texture_region(cell.get_atlas_coords());
							if (tile_data->get_flip_h()) {
								region.position.x += region.size.x;
								region.size.x = -region.size.x;
							}
							if (tile_data->get_flip_v()) {
								region.position.y += region.size.y;
								region.size.y = -region.size.y;
							}
							transpose = tile_data->get_transpose();
							max_probability = tile_data->get_probability();
						}
					}
				}

				// Create the ItemList's item.
				item_index = terrains_tile_list->add_item("");
				terrains_tile_list->set_item_icon(item_index, icon);
				terrains_tile_list->set_item_icon_region(item_index, region);
				terrains_tile_list->set_item_icon_transposed(item_index, transpose);
				list_metadata_dict = Dictionary();
				list_metadata_dict["type"] = SELECTED_TYPE_PATTERN;
				list_metadata_dict["terrains_pattern"] = terrains_pattern.as_array();
				terrains_tile_list->set_item_metadata(item_index, list_metadata_dict);
			}
		}
		if (terrains_tile_list->get_item_count() > 0) {
			terrains_tile_list->select(0);
		}
	}
}

void TileMapEditorTerrainsPlugin::_update_theme() {
	paint_tool_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("Edit")));
	line_tool_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("Line")));
	rect_tool_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("Rectangle")));
	bucket_tool_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("Bucket")));

	picker_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("ColorPick")));
	erase_button->set_icon(main_vbox_container->get_editor_theme_icon(SNAME("Eraser")));

	_update_tiles_list();
}

void TileMapEditorTerrainsPlugin::edit(ObjectID p_tile_map_id, int p_tile_map_layer) {
	_stop_dragging(); // Avoids staying in a wrong drag state.

	if (tile_map_id != p_tile_map_id) {
		tile_map_id = p_tile_map_id;

		// Clear the selection.
		_update_terrains_cache();
		_update_terrains_tree();
		_update_tiles_list();
	}

	tile_map_layer = p_tile_map_layer;
}

TileMapEditorTerrainsPlugin::TileMapEditorTerrainsPlugin() {
	main_vbox_container = memnew(VBoxContainer);
	main_vbox_container->connect("tree_entered", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_theme));
	main_vbox_container->connect("theme_changed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_theme));
	main_vbox_container->set_name(TTR("Terrains"));

	HSplitContainer *tilemap_tab_terrains = memnew(HSplitContainer);
	tilemap_tab_terrains->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tilemap_tab_terrains->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vbox_container->add_child(tilemap_tab_terrains);

	terrains_tree = memnew(Tree);
	terrains_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	terrains_tree->set_stretch_ratio(0.25);
	terrains_tree->set_custom_minimum_size(Size2(70, 0) * EDSCALE);
	terrains_tree->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
	terrains_tree->set_hide_root(true);
	terrains_tree->connect("item_selected", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_tiles_list));
	tilemap_tab_terrains->add_child(terrains_tree);

	terrains_tile_list = memnew(ItemList);
	terrains_tile_list->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	terrains_tile_list->set_max_columns(0);
	terrains_tile_list->set_same_column_width(true);
	terrains_tile_list->set_fixed_icon_size(Size2(32, 32) * EDSCALE);
	terrains_tile_list->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
	tilemap_tab_terrains->add_child(terrains_tile_list);

	// --- Toolbar ---
	toolbar = memnew(HBoxContainer);

	HBoxContainer *tilemap_tiles_tools_buttons = memnew(HBoxContainer);

	tool_buttons_group.instantiate();

	paint_tool_button = memnew(Button);
	paint_tool_button->set_theme_type_variation("FlatButton");
	paint_tool_button->set_toggle_mode(true);
	paint_tool_button->set_button_group(tool_buttons_group);
	paint_tool_button->set_pressed(true);
	paint_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/paint_tool", TTR("Paint"), Key::D));
	paint_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(paint_tool_button);
	viewport_shortcut_buttons.push_back(paint_tool_button);

	line_tool_button = memnew(Button);
	line_tool_button->set_theme_type_variation("FlatButton");
	line_tool_button->set_toggle_mode(true);
	line_tool_button->set_button_group(tool_buttons_group);
	line_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/line_tool", TTR("Line"), Key::L));
	line_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(line_tool_button);
	viewport_shortcut_buttons.push_back(line_tool_button);

	rect_tool_button = memnew(Button);
	rect_tool_button->set_theme_type_variation("FlatButton");
	rect_tool_button->set_toggle_mode(true);
	rect_tool_button->set_button_group(tool_buttons_group);
	rect_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/rect_tool", TTR("Rect"), Key::R));
	rect_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(rect_tool_button);
	viewport_shortcut_buttons.push_back(rect_tool_button);

	bucket_tool_button = memnew(Button);
	bucket_tool_button->set_theme_type_variation("FlatButton");
	bucket_tool_button->set_toggle_mode(true);
	bucket_tool_button->set_button_group(tool_buttons_group);
	bucket_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/bucket_tool", TTR("Bucket"), Key::B));
	bucket_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(bucket_tool_button);
	viewport_shortcut_buttons.push_back(bucket_tool_button);

	toolbar->add_child(tilemap_tiles_tools_buttons);

	// -- TileMap tool settings --
	tools_settings = memnew(HBoxContainer);
	toolbar->add_child(tools_settings);

	tools_settings_vsep = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep);

	// Picker
	picker_button = memnew(Button);
	picker_button->set_theme_type_variation("FlatButton");
	picker_button->set_toggle_mode(true);
	picker_button->set_shortcut(ED_SHORTCUT("tiles_editor/picker", TTR("Picker"), Key::P));
	picker_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(picker_button);
	viewport_shortcut_buttons.push_back(picker_button);

	// Erase button.
	erase_button = memnew(Button);
	erase_button->set_theme_type_variation("FlatButton");
	erase_button->set_toggle_mode(true);
	erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", TTR("Eraser"), Key::E));
	erase_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(erase_button);
	viewport_shortcut_buttons.push_back(erase_button);

	// Separator 2.
	tools_settings_vsep_2 = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep_2);

	// Continuous checkbox.
	bucket_contiguous_checkbox = memnew(CheckBox);
	bucket_contiguous_checkbox->set_flat(true);
	bucket_contiguous_checkbox->set_text(TTR("Contiguous"));
	bucket_contiguous_checkbox->set_pressed(true);
	tools_settings->add_child(bucket_contiguous_checkbox);
}

TileMapEditorTerrainsPlugin::~TileMapEditorTerrainsPlugin() {
}

void TileMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			missing_tile_texture = get_editor_theme_icon(SNAME("StatusWarning"));
			warning_pattern_texture = get_editor_theme_icon(SNAME("WarningPattern"));
			advanced_menu_button->set_icon(get_editor_theme_icon(SNAME("Tools")));
			toggle_grid_button->set_icon(get_editor_theme_icon(SNAME("Grid")));
			toggle_grid_button->set_pressed(EDITOR_GET("editors/tiles_editor/display_grid"));
			toggle_highlight_selected_layer_button->set_icon(get_editor_theme_icon(SNAME("TileMapHighlightSelected")));
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (is_visible_in_tree() && tileset_changed_needs_update) {
				_update_bottom_panel();
				_update_layers_selection();
				tabs_plugins[tabs_bar->get_current_tab()]->tile_set_changed();
				CanvasItemEditor::get_singleton()->update_viewport();
				tileset_changed_needs_update = false;
			}
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			toggle_grid_button->set_pressed(EDITOR_GET("editors/tiles_editor/display_grid"));
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
			if (tile_map) {
				if (is_visible_in_tree()) {
					tile_map->set_selected_layer(tile_map_layer);
				} else {
					tile_map->set_selected_layer(-1);
				}
			}
		} break;
	}
}

void TileMapEditor::_on_grid_toggled(bool p_pressed) {
	EditorSettings::get_singleton()->set("editors/tiles_editor/display_grid", p_pressed);
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_layers_selection_item_selected(int p_index) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map || tile_map->get_layers_count() <= 0) {
		return;
	}

	tile_map_layer = p_index;
	_update_layers_selection();
}

void TileMapEditor::_advanced_menu_button_id_pressed(int p_id) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (p_id == 0) { // Replace Tile Proxies
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Replace Tiles with Proxies"));
		for (int layer_index = 0; layer_index < tile_map->get_layers_count(); layer_index++) {
			TypedArray<Vector2i> used_cells = tile_map->get_used_cells(layer_index);
			for (int i = 0; i < used_cells.size(); i++) {
				Vector2i cell_coords = used_cells[i];
				TileMapCell from = tile_map->get_cell(layer_index, cell_coords);
				Array to_array = tile_set->map_tile_proxy(from.source_id, from.get_atlas_coords(), from.alternative_tile);
				TileMapCell to;
				to.source_id = to_array[0];
				to.set_atlas_coords(to_array[1]);
				to.alternative_tile = to_array[2];
				if (from != to) {
					undo_redo->add_do_method(tile_map, "set_cell", tile_map_layer, cell_coords, to.source_id, to.get_atlas_coords(), to.alternative_tile);
					undo_redo->add_undo_method(tile_map, "set_cell", tile_map_layer, cell_coords, from.source_id, from.get_atlas_coords(), from.alternative_tile);
				}
			}
		}
		undo_redo->commit_action();
	}
}

void TileMapEditor::_update_bottom_panel() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}
	Ref<TileSet> tile_set = tile_map->get_tileset();

	// Update the visibility of controls.
	missing_tileset_label->set_visible(!tile_set.is_valid());
	for (TileMapSubEditorPlugin::TabData &tab_data : tabs_data) {
		tab_data.panel->hide();
	}
	if (tile_set.is_valid()) {
		tabs_data[tabs_bar->get_current_tab()].panel->show();
	}
}

Vector<Vector2i> TileMapEditor::get_line(TileMap *p_tile_map, Vector2i p_from_cell, Vector2i p_to_cell) {
	ERR_FAIL_NULL_V(p_tile_map, Vector<Vector2i>());

	Ref<TileSet> tile_set = p_tile_map->get_tileset();
	ERR_FAIL_COND_V(!tile_set.is_valid(), Vector<Vector2i>());

	if (tile_set->get_tile_shape() == TileSet::TILE_SHAPE_SQUARE) {
		return Geometry2D::bresenham_line(p_from_cell, p_to_cell);
	} else {
		// Adapt the bresenham line algorithm to half-offset shapes.
		// See this blog post: http://zvold.blogspot.com/2010/01/bresenhams-line-drawing-algorithm-on_26.html
		Vector<Point2i> points;

		bool transposed = tile_set->get_tile_offset_axis() == TileSet::TILE_OFFSET_AXIS_VERTICAL;
		p_from_cell = TileMap::transform_coords_layout(p_from_cell, tile_set->get_tile_offset_axis(), tile_set->get_tile_layout(), TileSet::TILE_LAYOUT_STACKED);
		p_to_cell = TileMap::transform_coords_layout(p_to_cell, tile_set->get_tile_offset_axis(), tile_set->get_tile_layout(), TileSet::TILE_LAYOUT_STACKED);
		if (transposed) {
			SWAP(p_from_cell.x, p_from_cell.y);
			SWAP(p_to_cell.x, p_to_cell.y);
		}

		Vector2i delta = p_to_cell - p_from_cell;
		delta = Vector2i(2 * delta.x + ABS(p_to_cell.y % 2) - ABS(p_from_cell.y % 2), delta.y);
		Vector2i sign = delta.sign();

		Vector2i current = p_from_cell;
		points.push_back(TileMap::transform_coords_layout(transposed ? Vector2i(current.y, current.x) : current, tile_set->get_tile_offset_axis(), TileSet::TILE_LAYOUT_STACKED, tile_set->get_tile_layout()));

		int err = 0;
		if (ABS(delta.y) < ABS(delta.x)) {
			Vector2i err_step = 3 * delta.abs();
			while (current != p_to_cell) {
				err += err_step.y;
				if (err > ABS(delta.x)) {
					if (sign.x == 0) {
						current += Vector2(sign.y, 0);
					} else {
						current += Vector2(bool(current.y % 2) ^ (sign.x < 0) ? sign.x : 0, sign.y);
					}
					err -= err_step.x;
				} else {
					current += Vector2i(sign.x, 0);
					err += err_step.y;
				}
				points.push_back(TileMap::transform_coords_layout(transposed ? Vector2i(current.y, current.x) : current, tile_set->get_tile_offset_axis(), TileSet::TILE_LAYOUT_STACKED, tile_set->get_tile_layout()));
			}
		} else {
			Vector2i err_step = delta.abs();
			while (current != p_to_cell) {
				err += err_step.x;
				if (err > 0) {
					if (sign.x == 0) {
						current += Vector2(0, sign.y);
					} else {
						current += Vector2(bool(current.y % 2) ^ (sign.x < 0) ? sign.x : 0, sign.y);
					}
					err -= err_step.y;
				} else {
					if (sign.x == 0) {
						current += Vector2(0, sign.y);
					} else {
						current += Vector2(bool(current.y % 2) ^ (sign.x > 0) ? -sign.x : 0, sign.y);
					}
					err += err_step.y;
				}
				points.push_back(TileMap::transform_coords_layout(transposed ? Vector2i(current.y, current.x) : current, tile_set->get_tile_offset_axis(), TileSet::TILE_LAYOUT_STACKED, tile_set->get_tile_layout()));
			}
		}

		return points;
	}
}

void TileMapEditor::_tile_map_changed() {
	tileset_changed_needs_update = true;
}

void TileMapEditor::_tab_changed(int p_tab_id) {
	// Make the plugin edit the correct tilemap.
	tabs_plugins[tabs_bar->get_current_tab()]->edit(tile_map_id, tile_map_layer);

	// Update toolbar.
	for (TileMapSubEditorPlugin::TabData &tab_data : tabs_data) {
		tab_data.toolbar->hide();
	}
	tabs_data[p_tab_id].toolbar->show();

	// Update visible panel.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	for (TileMapSubEditorPlugin::TabData &tab_data : tabs_data) {
		tab_data.panel->hide();
	}
	if (tile_map && tile_map->get_tileset().is_valid()) {
		tabs_data[tabs_bar->get_current_tab()].panel->show();
	}

	// Graphical update.
	tabs_data[tabs_bar->get_current_tab()].panel->queue_redraw();
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_layers_select_next_or_previous(bool p_next) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	if (tile_map->get_layers_count() < 1) {
		return;
	}

	if (tile_map_layer < 0) {
		tile_map_layer = 0;
	}

	int inc = p_next ? 1 : -1;
	int origin_layer = tile_map_layer;
	tile_map_layer = Math::posmod((tile_map_layer + inc), tile_map->get_layers_count());
	while (tile_map_layer != origin_layer) {
		if (tile_map->is_layer_enabled(tile_map_layer)) {
			break;
		}
		tile_map_layer = Math::posmod((tile_map_layer + inc), tile_map->get_layers_count());
	}

	_update_layers_selection();
}

void TileMapEditor::_update_layers_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	// Update the selected layer.
	if (is_visible_in_tree() && tile_map->get_layers_count() >= 1) {
		tile_map_layer = CLAMP(tile_map_layer, 0, tile_map->get_layers_count() - 1);

		// Search for an enabled layer if the current one is not.
		int origin_layer = tile_map_layer;
		while (tile_map_layer >= 0 && !tile_map->is_layer_enabled(tile_map_layer)) {
			tile_map_layer--;
		}
		if (tile_map_layer < 0) {
			tile_map_layer = origin_layer;
			while (tile_map_layer < tile_map->get_layers_count() && !tile_map->is_layer_enabled(tile_map_layer)) {
				tile_map_layer++;
			}
		}
		if (tile_map_layer >= tile_map->get_layers_count()) {
			tile_map_layer = -1;
		}
	} else {
		tile_map_layer = -1;
	}
	tile_map->set_selected_layer(toggle_highlight_selected_layer_button->is_pressed() ? tile_map_layer : -1);
	tileset_changed_needs_update = false; // Update is not needed here and actually causes problems.

	layers_selection_button->clear();
	if (tile_map->get_layers_count() > 0) {
		// Build the list of layers.
		for (int i = 0; i < tile_map->get_layers_count(); i++) {
			String name = tile_map->get_layer_name(i);
			layers_selection_button->add_item(name.is_empty() ? vformat(TTR("Layer %d"), i) : name, i);
			layers_selection_button->set_item_disabled(i, !tile_map->is_layer_enabled(i));
		}

		layers_selection_button->set_disabled(false);
		layers_selection_button->select(tile_map_layer);
	} else {
		layers_selection_button->set_disabled(true);
		layers_selection_button->set_text(TTR("No Layers"));
	}

	tabs_plugins[tabs_bar->get_current_tab()]->edit(tile_map_id, tile_map_layer);
}

void TileMapEditor::_move_tile_map_array_element(Object *p_undo_redo, Object *p_edited, String p_array_prefix, int p_from_index, int p_to_pos) {
	EditorUndoRedoManager *undo_redo_man = Object::cast_to<EditorUndoRedoManager>(p_undo_redo);
	ERR_FAIL_NULL(undo_redo_man);

	TileMap *tile_map = Object::cast_to<TileMap>(p_edited);
	if (!tile_map) {
		return;
	}

	// Compute the array indices to save.
	int begin = 0;
	int end;
	if (p_array_prefix == "layer_") {
		end = tile_map->get_layers_count();
	} else {
		ERR_FAIL_MSG("Invalid array prefix for TileSet.");
	}
	if (p_from_index < 0) {
		// Adding new.
		if (p_to_pos >= 0) {
			begin = p_to_pos;
		} else {
			end = 0; // Nothing to save when adding at the end.
		}
	} else if (p_to_pos < 0) {
		// Removing.
		begin = p_from_index;
	} else {
		// Moving.
		begin = MIN(p_from_index, p_to_pos);
		end = MIN(MAX(p_from_index, p_to_pos) + 1, end);
	}

#define ADD_UNDO(obj, property) undo_redo_man->add_undo_property(obj, property, obj->get(property));
	// Save layers' properties.
	if (p_from_index < 0) {
		undo_redo_man->add_undo_method(tile_map, "remove_layer", p_to_pos < 0 ? tile_map->get_layers_count() : p_to_pos);
	} else if (p_to_pos < 0) {
		undo_redo_man->add_undo_method(tile_map, "add_layer", p_from_index);
	}

	List<PropertyInfo> properties;
	tile_map->get_property_list(&properties);
	for (PropertyInfo pi : properties) {
		if (pi.name.begins_with(p_array_prefix)) {
			String str = pi.name.trim_prefix(p_array_prefix);
			int to_char_index = 0;
			while (to_char_index < str.length()) {
				if (!is_digit(str[to_char_index])) {
					break;
				}
				to_char_index++;
			}
			if (to_char_index > 0) {
				int array_index = str.left(to_char_index).to_int();
				if (array_index >= begin && array_index < end) {
					ADD_UNDO(tile_map, pi.name);
				}
			}
		}
	}
#undef ADD_UNDO

	if (p_from_index < 0) {
		undo_redo_man->add_do_method(tile_map, "add_layer", p_to_pos);
	} else if (p_to_pos < 0) {
		undo_redo_man->add_do_method(tile_map, "remove_layer", p_from_index);
	} else {
		undo_redo_man->add_do_method(tile_map, "move_layer", p_from_index, p_to_pos);
	}
}

bool TileMapEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (ED_IS_SHORTCUT("tiles_editor/select_next_layer", p_event) && p_event->is_pressed()) {
		_layers_select_next_or_previous(true);
		return true;
	}

	if (ED_IS_SHORTCUT("tiles_editor/select_previous_layer", p_event) && p_event->is_pressed()) {
		_layers_select_next_or_previous(false);
		return true;
	}

	return tabs_plugins[tabs_bar->get_current_tab()]->forward_canvas_gui_input(p_event);
}

void TileMapEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (!tile_map->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform_with_canvas();
	Transform2D xform_inv = xform.affine_inverse();
	Vector2i tile_shape_size = tile_set->get_tile_size();

	// Draw tiles with invalid IDs in the grid.
	if (tile_map_layer >= 0) {
		ERR_FAIL_COND(tile_map_layer >= tile_map->get_layers_count());
		TypedArray<Vector2i> used_cells = tile_map->get_used_cells(tile_map_layer);
		for (int i = 0; i < used_cells.size(); i++) {
			Vector2i coords = used_cells[i];
			int tile_source_id = tile_map->get_cell_source_id(tile_map_layer, coords);
			if (tile_source_id >= 0) {
				Vector2i tile_atlas_coords = tile_map->get_cell_atlas_coords(tile_map_layer, coords);
				int tile_alternative_tile = tile_map->get_cell_alternative_tile(tile_map_layer, coords);

				TileSetSource *source = nullptr;
				if (tile_set->has_source(tile_source_id)) {
					source = *tile_set->get_source(tile_source_id);
				}

				if (!source || !source->has_tile(tile_atlas_coords) || !source->has_alternative_tile(tile_atlas_coords, tile_alternative_tile)) {
					// Generate a random color from the hashed values of the tiles.
					Array a = tile_set->map_tile_proxy(tile_source_id, tile_atlas_coords, tile_alternative_tile);
					if (int(a[0]) == tile_source_id && Vector2i(a[1]) == tile_atlas_coords && int(a[2]) == tile_alternative_tile) {
						// Only display the pattern if we have no proxy tile.
						Array to_hash;
						to_hash.push_back(tile_source_id);
						to_hash.push_back(tile_atlas_coords);
						to_hash.push_back(tile_alternative_tile);
						uint32_t hash = RandomPCG(to_hash.hash()).rand();

						Color color;
						color = color.from_hsv(
								(float)((hash >> 24) & 0xFF) / 256.0,
								Math::lerp(0.5, 1.0, (float)((hash >> 16) & 0xFF) / 256.0),
								Math::lerp(0.5, 1.0, (float)((hash >> 8) & 0xFF) / 256.0),
								0.8);

						// Draw the scaled tile.
						Transform2D tile_xform;
						tile_xform.set_origin(tile_map->map_to_local(coords));
						tile_xform.set_scale(tile_shape_size);
						tile_set->draw_tile_shape(p_overlay, xform * tile_xform, color, true, warning_pattern_texture);
					}

					// Draw the warning icon.
					Vector2::Axis min_axis = missing_tile_texture->get_size().min_axis_index();
					Vector2 icon_size;
					icon_size[min_axis] = tile_set->get_tile_size()[min_axis] / 3;
					icon_size[(min_axis + 1) % 2] = (icon_size[min_axis] * missing_tile_texture->get_size()[(min_axis + 1) % 2] / missing_tile_texture->get_size()[min_axis]);
					Rect2 rect = Rect2(xform.xform(tile_map->map_to_local(coords)) - (icon_size * xform.get_scale() / 2), icon_size * xform.get_scale());
					p_overlay->draw_texture_rect(missing_tile_texture, rect);
				}
			}
		}
	}

	// Fading on the border.
	const int fading = 5;

	// Determine the drawn area.
	Size2 screen_size = p_overlay->get_size();
	Rect2i screen_rect;
	screen_rect.position = tile_map->local_to_map(xform_inv.xform(Vector2()));
	screen_rect.expand_to(tile_map->local_to_map(xform_inv.xform(Vector2(0, screen_size.height))));
	screen_rect.expand_to(tile_map->local_to_map(xform_inv.xform(Vector2(screen_size.width, 0))));
	screen_rect.expand_to(tile_map->local_to_map(xform_inv.xform(screen_size)));
	screen_rect = screen_rect.grow(1);

	Rect2i tilemap_used_rect = tile_map->get_used_rect();

	Rect2i displayed_rect = tilemap_used_rect.intersection(screen_rect);
	displayed_rect = displayed_rect.grow(fading);

	// Reduce the drawn area to avoid crashes if needed.
	int max_size = 100;
	if (displayed_rect.size.x > max_size) {
		displayed_rect = displayed_rect.grow_individual(-(displayed_rect.size.x - max_size) / 2, 0, -(displayed_rect.size.x - max_size) / 2, 0);
	}
	if (displayed_rect.size.y > max_size) {
		displayed_rect = displayed_rect.grow_individual(0, -(displayed_rect.size.y - max_size) / 2, 0, -(displayed_rect.size.y - max_size) / 2);
	}

	// Draw the grid.
	bool display_grid = EDITOR_GET("editors/tiles_editor/display_grid");
	if (display_grid) {
		Color grid_color = EDITOR_GET("editors/tiles_editor/grid_color");
		for (int x = displayed_rect.position.x; x < (displayed_rect.position.x + displayed_rect.size.x); x++) {
			for (int y = displayed_rect.position.y; y < (displayed_rect.position.y + displayed_rect.size.y); y++) {
				Vector2i pos_in_rect = Vector2i(x, y) - displayed_rect.position;

				// Fade out the border of the grid.
				float left_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.x), 0.0f, 1.0f);
				float right_opacity = CLAMP(Math::inverse_lerp((float)displayed_rect.size.x, (float)(displayed_rect.size.x - fading), (float)(pos_in_rect.x + 1)), 0.0f, 1.0f);
				float top_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.y), 0.0f, 1.0f);
				float bottom_opacity = CLAMP(Math::inverse_lerp((float)displayed_rect.size.y, (float)(displayed_rect.size.y - fading), (float)(pos_in_rect.y + 1)), 0.0f, 1.0f);
				float opacity = CLAMP(MIN(left_opacity, MIN(right_opacity, MIN(top_opacity, bottom_opacity))) + 0.1, 0.0f, 1.0f);

				Transform2D tile_xform;
				tile_xform.set_origin(tile_map->map_to_local(Vector2(x, y)));
				tile_xform.set_scale(tile_shape_size);
				Color color = grid_color;
				color.a = color.a * opacity;
				tile_set->draw_tile_shape(p_overlay, xform * tile_xform, color, false);
			}
		}
	}

	// Draw the IDs for debug.
	/*Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	for (int x = displayed_rect.position.x; x < (displayed_rect.position.x + displayed_rect.size.x); x++) {
		for (int y = displayed_rect.position.y; y < (displayed_rect.position.y + displayed_rect.size.y); y++) {
			p_overlay->draw_string(font, xform.xform(tile_map->map_to_local(Vector2(x, y))) + Vector2i(-tile_shape_size.x / 2, 0), vformat("%s", Vector2(x, y)));
		}
	}*/

	// Draw the plugins.
	tabs_plugins[tabs_bar->get_current_tab()]->forward_canvas_draw_over_viewport(p_overlay);
}

void TileMapEditor::edit(TileMap *p_tile_map) {
	if (p_tile_map && p_tile_map->get_instance_id() == tile_map_id) {
		return;
	}

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		// Unselect layer if we are changing tile_map.
		if (tile_map != p_tile_map) {
			tile_map->set_selected_layer(-1);
		}

		// Disconnect to changes.
		tile_map->disconnect("changed", callable_mp(this, &TileMapEditor::_tile_map_changed));
	}

	if (p_tile_map) {
		// Change the edited object.
		tile_map_id = p_tile_map->get_instance_id();
		tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
		// Connect to changes.
		if (!tile_map->is_connected("changed", callable_mp(this, &TileMapEditor::_tile_map_changed))) {
			tile_map->connect("changed", callable_mp(this, &TileMapEditor::_tile_map_changed));
		}
	} else {
		tile_map_id = ObjectID();
	}

	_update_layers_selection();

	// Call the plugins.
	tabs_plugins[tabs_bar->get_current_tab()]->edit(tile_map_id, tile_map_layer);

	_tile_map_changed();
}

TileMapEditor::TileMapEditor() {
	set_process_internal(true);

	// Shortcuts.
	ED_SHORTCUT("tiles_editor/select_next_layer", TTR("Select Next Tile Map Layer"), Key::PAGEUP);
	ED_SHORTCUT("tiles_editor/select_previous_layer", TTR("Select Previous Tile Map Layer"), Key::PAGEDOWN);

	// TileMap editor plugins
	tile_map_editor_plugins.push_back(memnew(TileMapEditorTilesPlugin));
	tile_map_editor_plugins.push_back(memnew(TileMapEditorTerrainsPlugin));

	// TabBar.
	tabs_bar = memnew(TabBar);
	tabs_bar->set_clip_tabs(false);
	for (int plugin_index = 0; plugin_index < tile_map_editor_plugins.size(); plugin_index++) {
		Vector<TileMapSubEditorPlugin::TabData> tabs_vector = tile_map_editor_plugins[plugin_index]->get_tabs();
		for (int tab_index = 0; tab_index < tabs_vector.size(); tab_index++) {
			tabs_bar->add_tab(tabs_vector[tab_index].panel->get_name());
			tabs_data.push_back(tabs_vector[tab_index]);
			tabs_plugins.push_back(tile_map_editor_plugins[plugin_index]);
		}
	}
	tabs_bar->connect("tab_changed", callable_mp(this, &TileMapEditor::_tab_changed));

	// --- TileMap toolbar ---
	tile_map_toolbar = memnew(HFlowContainer);
	tile_map_toolbar->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(tile_map_toolbar);

	// Tabs.
	tile_map_toolbar->add_child(tabs_bar);

	// Tabs toolbars.
	for (TileMapSubEditorPlugin::TabData &tab_data : tabs_data) {
		tab_data.toolbar->hide();
		if (!tab_data.toolbar->get_parent()) {
			tile_map_toolbar->add_child(tab_data.toolbar);
		}
	}

	// Wide empty separation control. (like BoxContainer::add_spacer())
	Control *c = memnew(Control);
	c->set_mouse_filter(MOUSE_FILTER_PASS);
	c->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_map_toolbar->add_child(c);

	// Layer selector.
	layers_selection_button = memnew(OptionButton);
	layers_selection_button->set_custom_minimum_size(Size2(200, 0));
	layers_selection_button->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	layers_selection_button->set_tooltip_text(TTR("TileMap Layers"));
	layers_selection_button->connect("item_selected", callable_mp(this, &TileMapEditor::_layers_selection_item_selected));
	tile_map_toolbar->add_child(layers_selection_button);

	toggle_highlight_selected_layer_button = memnew(Button);
	toggle_highlight_selected_layer_button->set_theme_type_variation("FlatButton");
	toggle_highlight_selected_layer_button->set_toggle_mode(true);
	toggle_highlight_selected_layer_button->set_pressed(true);
	toggle_highlight_selected_layer_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_layers_selection));
	toggle_highlight_selected_layer_button->set_tooltip_text(TTR("Highlight Selected TileMap Layer"));
	tile_map_toolbar->add_child(toggle_highlight_selected_layer_button);

	tile_map_toolbar->add_child(memnew(VSeparator));

	// Grid toggle.
	toggle_grid_button = memnew(Button);
	toggle_grid_button->set_theme_type_variation("FlatButton");
	toggle_grid_button->set_toggle_mode(true);
	toggle_grid_button->set_tooltip_text(TTR("Toggle grid visibility."));
	toggle_grid_button->connect("toggled", callable_mp(this, &TileMapEditor::_on_grid_toggled));
	tile_map_toolbar->add_child(toggle_grid_button);

	// Advanced settings menu button.
	advanced_menu_button = memnew(MenuButton);
	advanced_menu_button->set_flat(false);
	advanced_menu_button->set_theme_type_variation("FlatButton");
	advanced_menu_button->get_popup()->add_item(TTR("Automatically Replace Tiles with Proxies"));
	advanced_menu_button->get_popup()->connect("id_pressed", callable_mp(this, &TileMapEditor::_advanced_menu_button_id_pressed));
	tile_map_toolbar->add_child(advanced_menu_button);

	missing_tileset_label = memnew(Label);
	missing_tileset_label->set_text(TTR("The edited TileMap node has no TileSet resource.\nCreate or load a TileSet resource in the Tile Set property in the inspector."));
	missing_tileset_label->set_h_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_v_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	missing_tileset_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	missing_tileset_label->hide();
	add_child(missing_tileset_label);

	for (unsigned int tab_index = 0; tab_index < tabs_data.size(); tab_index++) {
		add_child(tabs_data[tab_index].panel);
		tabs_data[tab_index].panel->set_v_size_flags(SIZE_EXPAND_FILL);
		tabs_data[tab_index].panel->set_visible(tab_index == 0);
		tabs_data[tab_index].panel->set_h_size_flags(SIZE_EXPAND_FILL);
	}

	_tab_changed(0);

	// Registers UndoRedo inspector callback.
	EditorNode::get_editor_data().add_move_array_element_function(SNAME("TileMap"), callable_mp(this, &TileMapEditor::_move_tile_map_array_element));
}

TileMapEditor::~TileMapEditor() {
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		memdelete(tile_map_editor_plugins[i]);
	}
}
