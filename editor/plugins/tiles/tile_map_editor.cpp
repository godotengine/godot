/*************************************************************************/
/*  tile_map_editor.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "tile_map_editor.h"

#include "tiles_editor_plugin.h"

#include "editor/editor_scale.h"
#include "editor/plugins/canvas_item_editor_plugin.h"

#include "scene/gui/center_container.h"
#include "scene/gui/split_container.h"

#include "core/input/input.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"

void TileMapEditorTilesPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			select_tool_button->set_icon(get_theme_icon("ToolSelect", "EditorIcons"));
			paint_tool_button->set_icon(get_theme_icon("Edit", "EditorIcons"));
			line_tool_button->set_icon(get_theme_icon("CurveLinear", "EditorIcons"));
			rect_tool_button->set_icon(get_theme_icon("Rectangle", "EditorIcons"));
			bucket_tool_button->set_icon(get_theme_icon("Bucket", "EditorIcons"));

			picker_button->set_icon(get_theme_icon("ColorPick", "EditorIcons"));
			erase_button->set_icon(get_theme_icon("Eraser", "EditorIcons"));

			missing_texture_texture = get_theme_icon("TileSet", "EditorIcons");
			break;
	}
}

void TileMapEditorTilesPlugin::tile_set_changed() {
	_update_fix_selected_and_hovered();
	_update_bottom_panel();
	_update_tile_set_sources_list();
	_update_atlas_view();
}

void TileMapEditorTilesPlugin::_on_random_tile_checkbox_toggled(bool p_pressed) {
	scatter_spinbox->set_editable(p_pressed);
}

void TileMapEditorTilesPlugin::_on_scattering_spinbox_changed(double p_value) {
	scattering = p_value;
}

void TileMapEditorTilesPlugin::_update_toolbar() {
	// Hide all settings.
	for (int i = 0; i < tools_settings->get_child_count(); i++) {
		Object::cast_to<CanvasItem>(tools_settings->get_child(i))->hide();
	}

	// Show only the correct settings.
	if (tool_buttons_group->get_pressed_button() == select_tool_button) {
	} else if (tool_buttons_group->get_pressed_button() == paint_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		tools_settings_vsep_2->show();
		random_tile_checkbox->show();
		scatter_label->show();
		scatter_spinbox->show();
	} else if (tool_buttons_group->get_pressed_button() == line_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		tools_settings_vsep_2->show();
		random_tile_checkbox->show();
		scatter_label->show();
		scatter_spinbox->show();
	} else if (tool_buttons_group->get_pressed_button() == rect_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		tools_settings_vsep_2->show();
		random_tile_checkbox->show();
		scatter_label->show();
		scatter_spinbox->show();
	} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
		tools_settings_vsep_2->show();
		bucket_continuous_checkbox->show();
		random_tile_checkbox->show();
		scatter_label->show();
		scatter_spinbox->show();
	}
}

Control *TileMapEditorTilesPlugin::get_toolbar() const {
	return toolbar;
}

void TileMapEditorTilesPlugin::_update_tile_set_sources_list() {
	// Update the sources.
	int old_current = sources_list->get_current();
	sources_list->clear();

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	for (int i = 0; i < tile_set->get_source_count(); i++) {
		int source_id = tile_set->get_source_id(i);

		// TODO: handle with virtual functions
		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			Ref<Texture2D> texture = atlas_source->get_texture();
			if (texture.is_valid()) {
				sources_list->add_item(vformat("%s - (id:%d)", texture->get_path().get_file(), source_id), texture);
			} else {
				sources_list->add_item(vformat("No texture atlas source - (id:%d)", source_id), missing_texture_texture);
			}
		} else {
			sources_list->add_item(vformat("Unknown type source - (id:%d)", source_id), missing_texture_texture);
		}
		sources_list->set_item_metadata(i, source_id);
	}

	if (sources_list->get_item_count() > 0) {
		if (old_current > 0) {
			// Keep the current selected item if needed.
			sources_list->set_current(CLAMP(old_current, 0, sources_list->get_item_count() - 1));
		} else {
			sources_list->set_current(0);
		}
		sources_list->emit_signal("item_selected", sources_list->get_current());
	}

	// Synchronize
	TilesEditor::get_singleton()->set_atlas_sources_lists_current(sources_list->get_current());
}

void TileMapEditorTilesPlugin::_update_atlas_view() {
	// Update the atlas display.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		tile_atlas_view->hide();
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index >= 0 && source_index < sources_list->get_item_count()) {
		int source_id = sources_list->get_item_metadata(source_index);
		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			tile_atlas_view->set_atlas_source(*tile_map->get_tileset(), atlas_source, source_id);
			tile_atlas_view->show();
		}
	} else {
		tile_atlas_view->hide();
	}

	// Synchronize atlas view.
	TilesEditor::get_singleton()->synchronize_atlas_view(tile_atlas_view);

	tile_atlas_control->update();
}

void TileMapEditorTilesPlugin::_update_bottom_panel() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}
	Ref<TileSet> tile_set = tile_map->get_tileset();

	// Update the tabs.
	bool valid_sources = tile_set.is_valid() && tile_set->get_source_count() > 0;
	missing_source_label->set_visible(!valid_sources);
	atlas_sources_split_container->set_visible(valid_sources);
}

bool TileMapEditorTilesPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree()) {
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

	// Shortcuts
	if (ED_IS_SHORTCUT("tiles_editor/cut", p_event) || ED_IS_SHORTCUT("tiles_editor/copy", p_event)) {
		// Fill in the clipboard.
		if (!tile_map_selection.is_empty()) {
			memdelete(tile_map_clipboard);
			TypedArray<Vector2i> coords_array;
			for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
				coords_array.push_back(E->get());
			}
			tile_map_clipboard = tile_map->get_pattern(coords_array);
		}

		if (ED_IS_SHORTCUT("tiles_editor/cut", p_event)) {
			// Delete selected tiles.
			if (!tile_map_selection.is_empty()) {
				undo_redo->create_action(TTR("Delete tiles"));
				for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
					undo_redo->add_do_method(tile_map, "set_cell", E->get(), -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
					undo_redo->add_undo_method(tile_map, "set_cell", E->get(), tile_map->get_cell_source_id(E->get()), tile_map->get_cell_atlas_coords(E->get()), tile_map->get_cell_alternative_tile(E->get()));
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
			undo_redo->create_action(TTR("Delete tiles"));
			for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
				undo_redo->add_do_method(tile_map, "set_cell", E->get(), -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
				undo_redo->add_undo_method(tile_map, "set_cell", E->get(), tile_map->get_cell_source_id(E->get()), tile_map->get_cell_atlas_coords(E->get()), tile_map->get_cell_alternative_tile(E->get()));
			}
			undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());
			tile_map_selection.clear();
			undo_redo->add_do_method(this, "_set_tile_map_selection", _get_tile_map_selection());
			undo_redo->commit_action();
		}
		return true;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		has_mouse = true;
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform();
		Vector2 mpos = xform.affine_inverse().xform(mm->get_position());

		switch (drag_type) {
			case DRAG_TYPE_PAINT: {
				Map<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, drag_last_mouse_pos, mpos);
				for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
					if (!erase_button->is_pressed() && E->get().source_id == -1) {
						continue;
					}
					Vector2i coords = E->key();
					if (!drag_modified.has(coords)) {
						drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
					}
					tile_map->set_cell(coords, E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
				}
			} break;
			case DRAG_TYPE_BUCKET: {
				Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->world_to_map(drag_last_mouse_pos), tile_map->world_to_map(mpos));
				for (int i = 0; i < line.size(); i++) {
					if (!drag_modified.has(line[i])) {
						Map<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], bucket_continuous_checkbox->is_pressed());
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							Vector2i coords = E->key();
							if (!drag_modified.has(coords)) {
								drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
							}
							tile_map->set_cell(coords, E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
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
		Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform();
		Vector2 mpos = xform.affine_inverse().xform(mb->get_position());

		if (mb->get_button_index() == MOUSE_BUTTON_LEFT) {
			if (mb->is_pressed()) {
				// Pressed
				if (tool_buttons_group->get_pressed_button() == select_tool_button) {
					drag_start_mouse_pos = mpos;
					if (tile_map_selection.has(tile_map->world_to_map(drag_start_mouse_pos)) && !mb->get_shift()) {
						// Move the selection
						drag_type = DRAG_TYPE_MOVE;
						drag_modified.clear();
						for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
							Vector2i coords = E->get();
							drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
							tile_map->set_cell(coords, -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
						}
					} else {
						// Select tiles
						drag_type = DRAG_TYPE_SELECT;
					}
				} else {
					// Check if we are picking a tile.
					if (picker_button->is_pressed()) {
						drag_type = DRAG_TYPE_PICK;
						drag_start_mouse_pos = mpos;
					} else {
						// Paint otherwise.
						if (tool_buttons_group->get_pressed_button() == paint_tool_button) {
							drag_type = DRAG_TYPE_PAINT;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
							Map<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, mpos, mpos);
							for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
								if (!erase_button->is_pressed() && E->get().source_id == -1) {
									continue;
								}
								Vector2i coords = E->key();
								if (!drag_modified.has(coords)) {
									drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
								}
								tile_map->set_cell(coords, E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
							}
						} else if (tool_buttons_group->get_pressed_button() == line_tool_button) {
							drag_type = DRAG_TYPE_LINE;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
						} else if (tool_buttons_group->get_pressed_button() == rect_tool_button) {
							drag_type = DRAG_TYPE_RECT;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
						} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button) {
							drag_type = DRAG_TYPE_BUCKET;
							drag_start_mouse_pos = mpos;
							drag_modified.clear();
							Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->world_to_map(drag_last_mouse_pos), tile_map->world_to_map(mpos));
							for (int i = 0; i < line.size(); i++) {
								if (!drag_modified.has(line[i])) {
									Map<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], bucket_continuous_checkbox->is_pressed());
									for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
										if (!erase_button->is_pressed() && E->get().source_id == -1) {
											continue;
										}
										Vector2i coords = E->key();
										if (!drag_modified.has(coords)) {
											drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
										}
										tile_map->set_cell(coords, E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
									}
								}
							}
						}
					}
				}

			} else {
				// Released
				switch (drag_type) {
					case DRAG_TYPE_SELECT: {
						undo_redo->create_action(TTR("Change selection"));
						undo_redo->add_undo_method(this, "_set_tile_map_selection", _get_tile_map_selection());

						if (!mb->get_shift() && !mb->get_control()) {
							tile_map_selection.clear();
						}
						Rect2i rect = Rect2i(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(mpos) - tile_map->world_to_map(drag_start_mouse_pos)).abs();
						for (int x = rect.position.x; x <= rect.get_end().x; x++) {
							for (int y = rect.position.y; y <= rect.get_end().y; y++) {
								Vector2i coords = Vector2i(x, y);
								if (mb->get_control()) {
									if (tile_map_selection.has(coords)) {
										tile_map_selection.erase(coords);
									}
								} else {
									if (tile_map->get_cell_source_id(coords) != -1) {
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
						Vector2i top_left;
						if (!tile_map_selection.is_empty()) {
							top_left = tile_map_selection.front()->get();
						}
						for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
							top_left = top_left.min(E->get());
						}

						Vector2i offset = drag_start_mouse_pos - tile_map->map_to_world(top_left);
						offset = tile_map->world_to_map(mpos - offset) - tile_map->world_to_map(drag_start_mouse_pos - offset);

						TypedArray<Vector2i> selection_used_cells = selection_pattern->get_used_cells();

						Vector2i coords;
						Map<Vector2i, TileMapCell> cells_undo;
						for (int i = 0; i < selection_used_cells.size(); i++) {
							coords = tile_map->map_pattern(top_left, selection_used_cells[i], selection_pattern);
							cells_undo[coords] = TileMapCell(drag_modified[coords].source_id, drag_modified[coords].get_atlas_coords(), drag_modified[coords].alternative_tile);
							coords = tile_map->map_pattern(top_left + offset, selection_used_cells[i], selection_pattern);
							cells_undo[coords] = TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords));
						}

						Map<Vector2i, TileMapCell> cells_do;
						for (int i = 0; i < selection_used_cells.size(); i++) {
							coords = tile_map->map_pattern(top_left, selection_used_cells[i], selection_pattern);
							cells_do[coords] = TileMapCell();
						}
						for (int i = 0; i < selection_used_cells.size(); i++) {
							coords = tile_map->map_pattern(top_left + offset, selection_used_cells[i], selection_pattern);
							cells_do[coords] = TileMapCell(selection_pattern->get_cell_source_id(selection_used_cells[i]), selection_pattern->get_cell_atlas_coords(selection_used_cells[i]), selection_pattern->get_cell_alternative_tile(selection_used_cells[i]));
						}
						undo_redo->create_action(TTR("Move tiles"));
						// Move the tiles.
						for (Map<Vector2i, TileMapCell>::Element *E = cells_do.front(); E; E = E->next()) {
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
						for (Map<Vector2i, TileMapCell>::Element *E = cells_undo.front(); E; E = E->next()) {
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
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
					} break;
					case DRAG_TYPE_PICK: {
						Rect2i rect = Rect2i(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(mpos) - tile_map->world_to_map(drag_start_mouse_pos)).abs();
						rect.size += Vector2i(1, 1);
						memdelete(selection_pattern);
						TypedArray<Vector2i> coords_array;
						for (int x = rect.position.x; x < rect.get_end().x; x++) {
							for (int y = rect.position.y; y < rect.get_end().y; y++) {
								Vector2i coords = Vector2i(x, y);
								if (tile_map->get_cell_source_id(coords) != -1) {
									coords_array.push_back(coords);
								}
							}
						}
						selection_pattern = tile_map->get_pattern(coords_array);
						if (!selection_pattern->is_empty()) {
							_update_tileset_selection_from_selection_pattern();
						} else {
							_update_selection_pattern_from_tileset_selection();
						}
						picker_button->set_pressed(false);
					} break;
					case DRAG_TYPE_PAINT: {
						undo_redo->create_action(TTR("Paint tiles"));
						for (Map<Vector2i, TileMapCell>::Element *E = drag_modified.front(); E; E = E->next()) {
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
						undo_redo->commit_action(false);
					} break;
					case DRAG_TYPE_LINE: {
						Map<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, mpos);
						undo_redo->create_action(TTR("Paint tiles"));
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
						}
						undo_redo->commit_action();
					} break;
					case DRAG_TYPE_RECT: {
						Map<Vector2i, TileMapCell> to_draw = _draw_rect(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(mpos));
						undo_redo->create_action(TTR("Paint tiles"));
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
						}
						undo_redo->commit_action();
					} break;
					case DRAG_TYPE_BUCKET: {
						undo_redo->create_action(TTR("Paint tiles"));
						for (Map<Vector2i, TileMapCell>::Element *E = drag_modified.front(); E; E = E->next()) {
							if (!erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
						undo_redo->commit_action(false);
					} break;
					case DRAG_TYPE_CLIPBOARD_PASTE: {
						Vector2 mouse_offset = (Vector2(tile_map_clipboard->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
						undo_redo->create_action(TTR("Paste tiles"));
						TypedArray<Vector2i> used_cells = tile_map_clipboard->get_used_cells();
						for (int i = 0; i < used_cells.size(); i++) {
							Vector2i coords = tile_map->map_pattern(tile_map->world_to_map(mpos - mouse_offset), used_cells[i], tile_map_clipboard);
							undo_redo->add_do_method(tile_map, "set_cell", coords, tile_map_clipboard->get_cell_source_id(used_cells[i]), tile_map_clipboard->get_cell_atlas_coords(used_cells[i]), tile_map_clipboard->get_cell_alternative_tile(used_cells[i]));
							undo_redo->add_undo_method(tile_map, "set_cell", coords, tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords));
						}
						undo_redo->commit_action();
					} break;
					default:
						break;
				}
				drag_type = DRAG_TYPE_NONE;
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

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	if (!tile_map->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform();
	Vector2i tile_shape_size = tile_set->get_tile_size();

	// Draw the selection.
	if (is_visible_in_tree() && tool_buttons_group->get_pressed_button() == select_tool_button) {
		// In select mode, we only draw the current selection if we are modifying it (pressing control or shift).
		if (drag_type == DRAG_TYPE_MOVE || (drag_type == DRAG_TYPE_SELECT && !Input::get_singleton()->is_key_pressed(KEY_CONTROL) && !Input::get_singleton()->is_key_pressed(KEY_SHIFT))) {
			// Do nothing
		} else {
			tile_map->draw_cells_outline(p_overlay, tile_map_selection, Color(0.0, 0.0, 1.0), xform);
		}
	}

	// handle the preview of the tiles to be placed.
	if (is_visible_in_tree() && has_mouse) { // Only if the tilemap editor is opened and the viewport is hovered.
		Map<Vector2i, TileMapCell> preview;
		Rect2i drawn_grid_rect;

		if (drag_type == DRAG_TYPE_PICK) {
			// Draw the area being picvked.
			Rect2i rect = Rect2i(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(drag_last_mouse_pos) - tile_map->world_to_map(drag_start_mouse_pos)).abs();
			rect.size += Vector2i(1, 1);
			for (int x = rect.position.x; x < rect.get_end().x; x++) {
				for (int y = rect.position.y; y < rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (tile_map->get_cell_source_id(coords) != -1) {
						Rect2 cell_region = xform.xform(Rect2(tile_map->map_to_world(coords) - tile_shape_size / 2, tile_shape_size));
						tile_set->draw_tile_shape(p_overlay, cell_region, Color(1.0, 1.0, 1.0), false);
					}
				}
			}
		} else if (drag_type == DRAG_TYPE_SELECT) {
			// Draw the area being selected.
			Rect2i rect = Rect2i(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(drag_last_mouse_pos) - tile_map->world_to_map(drag_start_mouse_pos)).abs();
			rect.size += Vector2i(1, 1);
			Set<Vector2i> to_draw;
			for (int x = rect.position.x; x < rect.get_end().x; x++) {
				for (int y = rect.position.y; y < rect.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (tile_map->get_cell_source_id(coords) != -1) {
						to_draw.insert(coords);
					}
				}
			}
			tile_map->draw_cells_outline(p_overlay, to_draw, Color(1.0, 1.0, 1.0), xform);
		} else if (drag_type == DRAG_TYPE_MOVE) {
			// Preview when moving.
			Vector2i top_left;
			if (!tile_map_selection.is_empty()) {
				top_left = tile_map_selection.front()->get();
			}
			for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
				top_left = top_left.min(E->get());
			}
			Vector2i offset = drag_start_mouse_pos - tile_map->map_to_world(top_left);
			offset = tile_map->world_to_map(drag_last_mouse_pos - offset) - tile_map->world_to_map(drag_start_mouse_pos - offset);

			TypedArray<Vector2i> selection_used_cells = selection_pattern->get_used_cells();
			for (int i = 0; i < selection_used_cells.size(); i++) {
				Vector2i coords = tile_map->map_pattern(offset + top_left, selection_used_cells[i], selection_pattern);
				preview[coords] = TileMapCell(selection_pattern->get_cell_source_id(selection_used_cells[i]), selection_pattern->get_cell_atlas_coords(selection_used_cells[i]), selection_pattern->get_cell_alternative_tile(selection_used_cells[i]));
			}
		} else if (drag_type == DRAG_TYPE_CLIPBOARD_PASTE) {
			// Preview when pasting.
			Vector2 mouse_offset = (Vector2(tile_map_clipboard->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
			TypedArray<Vector2i> clipboard_used_cells = tile_map_clipboard->get_used_cells();
			for (int i = 0; i < clipboard_used_cells.size(); i++) {
				Vector2i coords = tile_map->map_pattern(tile_map->world_to_map(drag_last_mouse_pos - mouse_offset), clipboard_used_cells[i], tile_map_clipboard);
				preview[coords] = TileMapCell(tile_map_clipboard->get_cell_source_id(clipboard_used_cells[i]), tile_map_clipboard->get_cell_atlas_coords(clipboard_used_cells[i]), tile_map_clipboard->get_cell_alternative_tile(clipboard_used_cells[i]));
			}
		} else if (!picker_button->is_pressed()) {
			bool expand_grid = false;
			if (tool_buttons_group->get_pressed_button() == paint_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a single pattern.
				preview = _draw_line(drag_last_mouse_pos, drag_last_mouse_pos, drag_last_mouse_pos);
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == line_tool_button) {
				if (drag_type == DRAG_TYPE_NONE) {
					// Preview for a single pattern.
					preview = _draw_line(drag_last_mouse_pos, drag_last_mouse_pos, drag_last_mouse_pos);
					expand_grid = true;
				} else if (drag_type == DRAG_TYPE_LINE) {
					// Preview for a line pattern.
					preview = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, drag_last_mouse_pos);
					expand_grid = true;
				}
			} else if (tool_buttons_group->get_pressed_button() == rect_tool_button && drag_type == DRAG_TYPE_RECT) {
				// Preview for a line pattern.
				preview = _draw_rect(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(drag_last_mouse_pos));
				expand_grid = true;
			} else if (tool_buttons_group->get_pressed_button() == bucket_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a line pattern.
				preview = _draw_bucket_fill(tile_map->world_to_map(drag_last_mouse_pos), bucket_continuous_checkbox->is_pressed());
			}

			// Expand the grid if needed
			if (expand_grid && !preview.is_empty()) {
				drawn_grid_rect = Rect2i(preview.front()->key(), Vector2i(1, 1));
				for (Map<Vector2i, TileMapCell>::Element *E = preview.front(); E; E = E->next()) {
					drawn_grid_rect.expand_to(E->key());
				}
			}
		}

		if (!preview.is_empty()) {
			const int fading = 5;

			// Draw the lines of the grid behind the preview.
			if (drawn_grid_rect.size.x > 0 && drawn_grid_rect.size.y > 0) {
				drawn_grid_rect = drawn_grid_rect.grow(fading);
				for (int x = drawn_grid_rect.position.x; x < (drawn_grid_rect.position.x + drawn_grid_rect.size.x); x++) {
					for (int y = drawn_grid_rect.position.y; y < (drawn_grid_rect.position.y + drawn_grid_rect.size.y); y++) {
						Vector2i pos_in_rect = Vector2i(x, y) - drawn_grid_rect.position;

						// Fade out the border of the grid.
						float left_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.x), 0.0f, 1.0f);
						float right_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.x, (float)(drawn_grid_rect.size.x - fading), (float)pos_in_rect.x), 0.0f, 1.0f);
						float top_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.y), 0.0f, 1.0f);
						float bottom_opacity = CLAMP(Math::inverse_lerp((float)drawn_grid_rect.size.y, (float)(drawn_grid_rect.size.y - fading), (float)pos_in_rect.y), 0.0f, 1.0f);
						float opacity = CLAMP(MIN(left_opacity, MIN(right_opacity, MIN(top_opacity, bottom_opacity))) + 0.1, 0.0f, 1.0f);

						Rect2 cell_region = xform.xform(Rect2(tile_map->map_to_world(Vector2(x, y)) - tile_shape_size / 2, tile_shape_size));
						tile_set->draw_tile_shape(p_overlay, cell_region, Color(1.0, 0.5, 0.2, 0.5 * opacity), false);
					}
				}
			}

			// Draw the preview.
			for (Map<Vector2i, TileMapCell>::Element *E = preview.front(); E; E = E->next()) {
				if (!erase_button->is_pressed() && random_tile_checkbox->is_pressed()) {
					Vector2i size = tile_set->get_tile_size();
					Vector2 position = tile_map->map_to_world(E->key()) - size / 2;
					Rect2 cell_region = xform.xform(Rect2(position, size));

					tile_set->draw_tile_shape(p_overlay, cell_region, Color(1.0, 1.0, 1.0, 0.5), true);
				} else {
					if (tile_set->has_source(E->get().source_id)) {
						TileSetSource *source = *tile_set->get_source(E->get().source_id);
						TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
						if (atlas_source) {
							// Get tile data.
							TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(E->get().get_atlas_coords(), E->get().alternative_tile));

							// Compute the offset
							Rect2i source_rect = atlas_source->get_tile_texture_region(E->get().get_atlas_coords());
							Vector2i tile_offset = tile_set->get_tile_effective_texture_offset(E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);

							// Compute the destination rectangle in the CanvasItem.
							Rect2 dest_rect;
							dest_rect.size = source_rect.size;

							bool transpose = tile_data->get_transpose();
							if (transpose) {
								dest_rect.position = (tile_map->map_to_world(E->key()) - Vector2(dest_rect.size.y, dest_rect.size.x) / 2 - tile_offset);
							} else {
								dest_rect.position = (tile_map->map_to_world(E->key()) - dest_rect.size / 2 - tile_offset);
							}

							dest_rect = xform.xform(dest_rect);

							if (tile_data->get_flip_h()) {
								dest_rect.size.x = -dest_rect.size.x;
							}

							if (tile_data->get_flip_v()) {
								dest_rect.size.y = -dest_rect.size.y;
							}

							// Get the tile modulation.
							Color modulate = tile_data->get_modulate();
							Color self_modulate = tile_map->get_self_modulate();
							modulate = Color(modulate.r * self_modulate.r, modulate.g * self_modulate.g, modulate.b * self_modulate.b, modulate.a * self_modulate.a);

							// Draw the tile.
							p_overlay->draw_texture_rect_region(atlas_source->get_texture(), dest_rect, source_rect, modulate * Color(1.0, 1.0, 1.0, 0.5), transpose, tile_set->is_uv_clipping());
						}
					} else {
						Vector2i size = tile_set->get_tile_size();
						Vector2 position = tile_map->map_to_world(E->key()) - size / 2;
						Rect2 cell_region = xform.xform(Rect2(position, size));

						tile_set->draw_tile_shape(p_overlay, cell_region, Color(0.0, 0.0, 0.0, 0.5), true);
					}
				}
			}
		}
	}
}

void TileMapEditorTilesPlugin::_mouse_exited_viewport() {
	has_mouse = false;
	CanvasItemEditor::get_singleton()->update_viewport();
}

TileMapCell TileMapEditorTilesPlugin::_pick_random_tile(const TileMapPattern *p_pattern) {
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
			sum += Object::cast_to<TileData>(atlas_source->get_tile_data(atlas_coords, alternative_tile))->get_probability();
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
			current += Object::cast_to<TileData>(atlas_source->get_tile_data(atlas_coords, alternative_tile))->get_probability();
		} else {
			current += 1.0;
		}

		if (current >= rand) {
			return TileMapCell(source_id, atlas_coords, alternative_tile);
		}
	}
	return TileMapCell();
}

Map<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_line(Vector2 p_start_drag_mouse_pos, Vector2 p_from_mouse_pos, Vector2i p_to_mouse_pos) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return Map<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	TileMapPattern erase_pattern;
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = erase_button->is_pressed() ? &erase_pattern : selection_pattern;

	Map<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		// Paint the tiles on the tile map.
		if (!erase_button->is_pressed() && random_tile_checkbox->is_pressed()) {
			// Paint a random tile.
			Vector<Vector2i> line = TileMapEditor::get_line(tile_map, tile_map->world_to_map(p_from_mouse_pos), tile_map->world_to_map(p_to_mouse_pos));
			for (int i = 0; i < line.size(); i++) {
				output.insert(line[i], _pick_random_tile(pattern));
			}
		} else {
			// Paint the pattern.
			// If we paint several tiles, we virtually move the mouse as if it was in the center of the "brush"
			Vector2 mouse_offset = (Vector2(pattern->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
			Vector2i last_hovered_cell = tile_map->world_to_map(p_from_mouse_pos - mouse_offset);
			Vector2i new_hovered_cell = tile_map->world_to_map(p_to_mouse_pos - mouse_offset);
			Vector2i drag_start_cell = tile_map->world_to_map(p_start_drag_mouse_pos - mouse_offset);

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

Map<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_rect(Vector2i p_start_cell, Vector2i p_end_cell) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return Map<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileMapCell>();
	}

	// Create the rect to draw.
	Rect2i rect = Rect2i(p_start_cell, p_end_cell - p_start_cell).abs();
	rect.size += Vector2i(1, 1);

	// Get or create the pattern.
	TileMapPattern erase_pattern;
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = erase_button->is_pressed() ? &erase_pattern : selection_pattern;

	// Compute the offset to align things to the bottom or right.
	bool aligned_right = p_end_cell.x < p_start_cell.x;
	bool valigned_bottom = p_end_cell.y < p_start_cell.y;
	Vector2i offset = Vector2i(aligned_right ? -(pattern->get_size().x - (rect.get_size().x % pattern->get_size().x)) : 0, valigned_bottom ? -(pattern->get_size().y - (rect.get_size().y % pattern->get_size().y)) : 0);

	Map<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		if (!erase_button->is_pressed() && random_tile_checkbox->is_pressed()) {
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

Map<Vector2i, TileMapCell> TileMapEditorTilesPlugin::_draw_bucket_fill(Vector2i p_coords, bool p_contiguous) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return Map<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	TileMapPattern erase_pattern;
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileSetAtlasSource::INVALID_ATLAS_COORDS, TileSetAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = erase_button->is_pressed() ? &erase_pattern : selection_pattern;

	Map<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		TileMapCell source = tile_map->get_cell(p_coords);

		// If we are filling empty tiles, compute the tilemap boundaries.
		Rect2i boundaries;
		if (source.source_id == -1) {
			boundaries = tile_map->get_used_rect();
		}

		if (p_contiguous) {
			// Replace continuous tiles like the source.
			Set<Vector2i> already_checked;
			List<Vector2i> to_check;
			to_check.push_back(p_coords);
			while (!to_check.is_empty()) {
				Vector2i coords = to_check.back()->get();
				to_check.pop_back();
				if (!already_checked.has(coords)) {
					if (source.source_id == tile_map->get_cell_source_id(coords) &&
							source.get_atlas_coords() == tile_map->get_cell_atlas_coords(coords) &&
							source.alternative_tile == tile_map->get_cell_alternative_tile(coords) &&
							(source.source_id != -1 || boundaries.has_point(coords))) {
						if (!erase_button->is_pressed() && random_tile_checkbox->is_pressed()) {
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
						TypedArray<Vector2i> around = tile_map->get_surrounding_tiles(coords);
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
			if (source.source_id == -1) {
				Rect2i rect = tile_map->get_used_rect();
				if (rect.size.x <= 0 || rect.size.y <= 0) {
					rect = Rect2i(p_coords, Vector2i(1, 1));
				}
				for (int x = boundaries.position.x; x < boundaries.get_end().x; x++) {
					for (int y = boundaries.position.y; y < boundaries.get_end().y; y++) {
						to_check.append(Vector2i(x, y));
					}
				}
			} else {
				to_check = tile_map->get_used_cells();
			}
			for (int i = 0; i < to_check.size(); i++) {
				Vector2i coords = to_check[i];
				if (source.source_id == tile_map->get_cell_source_id(coords) &&
						source.get_atlas_coords() == tile_map->get_cell_atlas_coords(coords) &&
						source.alternative_tile == tile_map->get_cell_alternative_tile(coords) &&
						(source.source_id != -1 || boundaries.has_point(coords))) {
					if (!erase_button->is_pressed() && random_tile_checkbox->is_pressed()) {
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

void TileMapEditorTilesPlugin::_update_fix_selected_and_hovered() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);

	// Clear hovered if needed.
	if (source_id != hovered_tile.source_id ||
			!tile_set->has_source(hovered_tile.source_id) ||
			!tile_set->get_source(hovered_tile.source_id)->has_tile(hovered_tile.get_atlas_coords()) ||
			!tile_set->get_source(hovered_tile.source_id)->has_alternative_tile(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile)) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
	}

	// Selection if needed.
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		const TileMapCell *selected = &(E->get());
		if (!tile_set->has_source(selected->source_id) ||
				!tile_set->get_source(selected->source_id)->has_tile(selected->get_atlas_coords()) ||
				!tile_set->get_source(selected->source_id)->has_alternative_tile(selected->get_atlas_coords(), selected->alternative_tile)) {
			tile_set_selection.erase(E);
		}
	}

	if (!tile_map_selection.is_empty()) {
		_update_selection_pattern_from_tilemap_selection();
	} else {
		_update_selection_pattern_from_tileset_selection();
	}
}

void TileMapEditorTilesPlugin::_update_selection_pattern_from_tilemap_selection() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	memdelete(selection_pattern);

	TypedArray<Vector2i> coords_array;
	for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
		coords_array.push_back(E->get());
	}
	selection_pattern = tile_map->get_pattern(coords_array);
}

void TileMapEditorTilesPlugin::_update_selection_pattern_from_tileset_selection() {
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
	selection_pattern->clear();

	// Group per source.
	Map<int, List<const TileMapCell *>> per_source;
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		per_source[E->get().source_id].push_back(&(E->get()));
	}

	for (Map<int, List<const TileMapCell *>>::Element *E_source = per_source.front(); E_source; E_source = E_source->next()) {
		// Per source.
		List<const TileMapCell *> unorganized;
		Rect2i encompassing_rect_coords;
		Map<Vector2i, const TileMapCell *> organized_pattern;

		TileSetSource *source = *tile_set->get_source(E_source->key());
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			// Organize using coordinates.
			for (List<const TileMapCell *>::Element *E_cell = E_source->get().front(); E_cell; E_cell = E_cell->next()) {
				const TileMapCell *current = E_cell->get();
				if (organized_pattern.has(current->get_atlas_coords())) {
					if (current->alternative_tile < organized_pattern[current->get_atlas_coords()]->alternative_tile) {
						unorganized.push_back(organized_pattern[current->get_atlas_coords()]);
						organized_pattern[current->get_atlas_coords()] = current;
					} else {
						unorganized.push_back(current);
					}
				} else {
					organized_pattern[current->get_atlas_coords()] = current;
				}
			}

			// Compute the encompassing rect for the organized pattern.
			Map<Vector2i, const TileMapCell *>::Element *E_cell = organized_pattern.front();
			encompassing_rect_coords = Rect2i(E_cell->key(), Vector2i(1, 1));
			for (; E_cell; E_cell = E_cell->next()) {
				encompassing_rect_coords.expand_to(E_cell->key() + Vector2i(1, 1));
				encompassing_rect_coords.expand_to(E_cell->key());
			}
		} else {
			// Add everything unorganized.
			for (List<const TileMapCell *>::Element *E_cell = E_source->get().front(); E_cell; E_cell = E_cell->next()) {
				unorganized.push_back(E_cell->get());
			}
		}

		// Now add everything to the output pattern.
		for (Map<Vector2i, const TileMapCell *>::Element *E_cell = organized_pattern.front(); E_cell; E_cell = E_cell->next()) {
			selection_pattern->set_cell(E_cell->key() - encompassing_rect_coords.position, E_cell->get()->source_id, E_cell->get()->get_atlas_coords(), E_cell->get()->alternative_tile);
		}
		Vector2i organized_size = selection_pattern->get_size();
		for (List<const TileMapCell *>::Element *E_cell = unorganized.front(); E_cell; E_cell = E_cell->next()) {
			selection_pattern->set_cell(Vector2(organized_size.x, 0), E_cell->get()->source_id, E_cell->get()->get_atlas_coords(), E_cell->get()->alternative_tile);
		}
	}
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditorTilesPlugin::_update_tileset_selection_from_selection_pattern() {
	tile_set_selection.clear();
	TypedArray<Vector2i> used_cells = selection_pattern->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = used_cells[i];
		if (selection_pattern->get_cell_source_id(coords) != -1) {
			tile_set_selection.insert(TileMapCell(selection_pattern->get_cell_source_id(coords), selection_pattern->get_cell_atlas_coords(coords), selection_pattern->get_cell_alternative_tile(coords)));
		}
	}
	_update_atlas_view();
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
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		if (E->get().source_id == source_id && E->get().alternative_tile == 0) {
			tile_atlas_control->draw_rect(atlas->get_tile_texture_region(E->get().get_atlas_coords()), Color(0.0, 0.0, 1.0), false);
		}
	}

	// Draw the hovered tile.
	if (hovered_tile.get_atlas_coords() != TileSetAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0 && !tile_set_dragging_selection) {
		tile_atlas_control->draw_rect(atlas->get_tile_texture_region(hovered_tile.get_atlas_coords()), Color(1.0, 1.0, 1.0), false);
	}

	// Draw the selection rect.
	if (tile_set_dragging_selection) {
		Vector2i start_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_set_drag_start_mouse_pos);
		Vector2i end_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());

		Rect2i region = Rect2i(start_tile, end_tile - start_tile).abs();
		region.size += Vector2i(1, 1);

		Set<Vector2i> to_draw;
		for (int x = region.position.x; x < region.get_end().x; x++) {
			for (int y = region.position.y; y < region.get_end().y; y++) {
				Vector2i tile = atlas->get_tile_at_coords(Vector2i(x, y));
				if (tile != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
					to_draw.insert(tile);
				}
			}
		}

		for (Set<Vector2i>::Element *E = to_draw.front(); E; E = E->next()) {
			tile_atlas_control->draw_rect(atlas->get_tile_texture_region(E->get()), Color(0.8, 0.8, 1.0), false);
		}
	}
}

void TileMapEditorTilesPlugin::_tile_atlas_control_mouse_exited() {
	hovered_tile.source_id = -1;
	hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
	tile_set_dragging_selection = false;
	tile_atlas_control->update();
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
	hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
	Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
	if (coords != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
		coords = atlas->get_tile_at_coords(coords);
		if (coords != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
			hovered_tile.set_atlas_coords(coords);
			hovered_tile.alternative_tile = 0;
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->update();
		alternative_tiles_control->update();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MOUSE_BUTTON_LEFT) {
		if (mb->is_pressed()) { // Pressed
			tile_set_dragging_selection = true;
			tile_set_drag_start_mouse_pos = tile_atlas_control->get_local_mouse_position();
			if (!mb->get_shift()) {
				tile_set_selection.clear();
			}

			if (hovered_tile.get_atlas_coords() != TileSetAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0) {
				if (mb->get_shift() && tile_set_selection.has(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0))) {
					tile_set_selection.erase(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0));
				} else {
					tile_set_selection.insert(TileMapCell(source_id, hovered_tile.get_atlas_coords(), 0));
				}
			}
			_update_selection_pattern_from_tileset_selection();
		} else { // Released
			if (tile_set_dragging_selection) {
				if (!mb->get_shift()) {
					tile_set_selection.clear();
				}
				// Compute the covered area.
				Vector2i start_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_set_drag_start_mouse_pos);
				Vector2i end_tile = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
				if (start_tile != TileSetAtlasSource::INVALID_ATLAS_COORDS && end_tile != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
					Rect2i region = Rect2i(start_tile, end_tile - start_tile).abs();
					region.size += Vector2i(1, 1);

					// To update the selection, we copy the selected/not selected status of the tiles we drag from.
					Vector2i start_coords = atlas->get_tile_at_coords(start_tile);
					if (mb->get_shift() && start_coords != TileSetAtlasSource::INVALID_ATLAS_COORDS && !tile_set_selection.has(TileMapCell(source_id, start_coords, 0))) {
						// Remove from the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileSetAtlasSource::INVALID_ATLAS_COORDS && tile_set_selection.has(TileMapCell(source_id, tile_coords, 0))) {
									tile_set_selection.erase(TileMapCell(source_id, tile_coords, 0));
								}
							}
						}
					} else {
						// Insert in the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileSetAtlasSource::INVALID_ATLAS_COORDS) {
									tile_set_selection.insert(TileMapCell(source_id, tile_coords, 0));
								}
							}
						}
					}
				}
				_update_selection_pattern_from_tileset_selection();
			}
			tile_set_dragging_selection = false;
		}
		tile_atlas_control->update();
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
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		if (E->get().source_id == source_id && E->get().get_atlas_coords() != TileSetAtlasSource::INVALID_ATLAS_COORDS && E->get().alternative_tile > 0) {
			Rect2i rect = tile_atlas_view->get_alternative_tile_rect(E->get().get_atlas_coords(), E->get().alternative_tile);
			if (rect != Rect2i()) {
				alternative_tiles_control->draw_rect(rect, Color(0.2, 0.2, 1.0), false);
			}
		}
	}

	// Draw hovered tile.
	if (hovered_tile.get_atlas_coords() != TileSetAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile > 0) {
		Rect2i rect = tile_atlas_view->get_alternative_tile_rect(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile);
		if (rect != Rect2i()) {
			alternative_tiles_control->draw_rect(rect, Color(1.0, 1.0, 1.0), false);
		}
	}
}

void TileMapEditorTilesPlugin::_tile_alternatives_control_mouse_exited() {
	hovered_tile.source_id = -1;
	hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
	tile_set_dragging_selection = false;
	alternative_tiles_control->update();
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
	hovered_tile.set_atlas_coords(TileSetAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileSetAtlasSource::INVALID_TILE_ALTERNATIVE;
	Vector3i alternative_coords = tile_atlas_view->get_alternative_tile_at_pos(alternative_tiles_control->get_local_mouse_position());
	Vector2i coords = Vector2i(alternative_coords.x, alternative_coords.y);
	int alternative = alternative_coords.z;
	if (coords != TileSetAtlasSource::INVALID_ATLAS_COORDS && alternative != TileSetAtlasSource::INVALID_TILE_ALTERNATIVE) {
		hovered_tile.set_atlas_coords(coords);
		hovered_tile.alternative_tile = alternative;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->update();
		alternative_tiles_control->update();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == MOUSE_BUTTON_LEFT) {
		if (mb->is_pressed()) { // Pressed
			// Left click pressed.
			if (!mb->get_shift()) {
				tile_set_selection.clear();
			}

			if (coords != TileSetAtlasSource::INVALID_ATLAS_COORDS && alternative != TileSetAtlasSource::INVALID_TILE_ALTERNATIVE) {
				if (mb->get_shift() && tile_set_selection.has(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile))) {
					tile_set_selection.erase(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile));
				} else {
					tile_set_selection.insert(TileMapCell(source_id, hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile));
				}
			}
			_update_selection_pattern_from_tileset_selection();
		}
		tile_atlas_control->update();
		alternative_tiles_control->update();
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
	for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
		output.push_back(E->get());
	}
	return output;
}

void TileMapEditorTilesPlugin::edit(ObjectID p_tile_map_id) {
	tile_map_id = p_tile_map_id;

	// Clean the selection.
	tile_set_selection.clear();
	tile_map_selection.clear();
	selection_pattern->clear();
}

void TileMapEditorTilesPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_tile_map_selection", "selection"), &TileMapEditorTilesPlugin::_set_tile_map_selection);
	ClassDB::bind_method(D_METHOD("_get_tile_map_selection"), &TileMapEditorTilesPlugin::_get_tile_map_selection);
}

TileMapEditorTilesPlugin::TileMapEditorTilesPlugin() {
	CanvasItemEditor::get_singleton()->get_viewport_control()->connect("mouse_exited", callable_mp(this, &TileMapEditorTilesPlugin::_mouse_exited_viewport));

	// --- Shortcuts ---
	ED_SHORTCUT("tiles_editor/cut", TTR("Cut"), KEY_MASK_CMD | KEY_X);
	ED_SHORTCUT("tiles_editor/copy", TTR("Copy"), KEY_MASK_CMD | KEY_C);
	ED_SHORTCUT("tiles_editor/paste", TTR("Paste"), KEY_MASK_CMD | KEY_V);
	ED_SHORTCUT("tiles_editor/cancel", TTR("Cancel"), KEY_ESCAPE);
	ED_SHORTCUT("tiles_editor/delete", TTR("Delete"), KEY_DELETE);

	// --- Toolbar ---
	toolbar = memnew(HBoxContainer);

	HBoxContainer *tilemap_tiles_tools_buttons = memnew(HBoxContainer);

	tool_buttons_group.instance();

	select_tool_button = memnew(Button);
	select_tool_button->set_flat(true);
	select_tool_button->set_toggle_mode(true);
	select_tool_button->set_button_group(tool_buttons_group);
	select_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/selection_tool", "Selection", KEY_S));
	select_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(select_tool_button);

	paint_tool_button = memnew(Button);
	paint_tool_button->set_flat(true);
	paint_tool_button->set_toggle_mode(true);
	paint_tool_button->set_button_group(tool_buttons_group);
	paint_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/paint_tool", "Paint", KEY_E));
	paint_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(paint_tool_button);

	line_tool_button = memnew(Button);
	line_tool_button->set_flat(true);
	line_tool_button->set_toggle_mode(true);
	line_tool_button->set_button_group(tool_buttons_group);
	line_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/line_tool", "Line", KEY_L));
	line_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(line_tool_button);

	rect_tool_button = memnew(Button);
	rect_tool_button->set_flat(true);
	rect_tool_button->set_toggle_mode(true);
	rect_tool_button->set_button_group(tool_buttons_group);
	rect_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/rect_tool", "Rect", KEY_R));
	rect_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(rect_tool_button);

	bucket_tool_button = memnew(Button);
	bucket_tool_button->set_flat(true);
	bucket_tool_button->set_toggle_mode(true);
	bucket_tool_button->set_button_group(tool_buttons_group);
	bucket_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/bucket_tool", "Bucket", KEY_B));
	bucket_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTilesPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(bucket_tool_button);

	toolbar->add_child(tilemap_tiles_tools_buttons);

	// -- TileMap tool settings --
	tools_settings = memnew(HBoxContainer);
	toolbar->add_child(tools_settings);

	tools_settings_vsep = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep);

	// Picker
	picker_button = memnew(Button);
	picker_button->set_flat(true);
	picker_button->set_toggle_mode(true);
	picker_button->set_shortcut(ED_SHORTCUT("tiles_editor/picker", "Picker", KEY_P));
	picker_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(picker_button);

	// Erase button.
	erase_button = memnew(Button);
	erase_button->set_flat(true);
	erase_button->set_toggle_mode(true);
	erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", "Eraser", KEY_E));
	erase_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(erase_button);

	// Separator 2.
	tools_settings_vsep_2 = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep_2);

	// Continuous checkbox.
	bucket_continuous_checkbox = memnew(CheckBox);
	bucket_continuous_checkbox->set_flat(true);
	bucket_continuous_checkbox->set_text(TTR("Contiguous"));
	tools_settings->add_child(bucket_continuous_checkbox);

	// Random tile checkbox.
	random_tile_checkbox = memnew(CheckBox);
	random_tile_checkbox->set_flat(true);
	random_tile_checkbox->set_text(TTR("Place random tile"));
	random_tile_checkbox->connect("toggled", callable_mp(this, &TileMapEditorTilesPlugin::_on_random_tile_checkbox_toggled));
	tools_settings->add_child(random_tile_checkbox);

	// Random tile scattering.
	scatter_label = memnew(Label);
	scatter_label->set_tooltip(TTR("Defines the probability of painting nothing instead of a randomly selected tile."));
	scatter_label->set_text(TTR("Scattering:"));
	tools_settings->add_child(scatter_label);

	scatter_spinbox = memnew(SpinBox);
	scatter_spinbox->set_min(0.0);
	scatter_spinbox->set_max(1000);
	scatter_spinbox->set_step(0.001);
	scatter_spinbox->set_tooltip(TTR("Defines the probability of painting nothing instead of a randomly selected tile."));
	scatter_spinbox->get_line_edit()->add_theme_constant_override("minimum_character_width", 4);
	scatter_spinbox->connect("value_changed", callable_mp(this, &TileMapEditorTilesPlugin::_on_scattering_spinbox_changed));
	tools_settings->add_child(scatter_spinbox);

	_on_random_tile_checkbox_toggled(false);

	// Default tool.
	paint_tool_button->set_pressed(true);
	_update_toolbar();

	// --- Bottom panel ---
	set_name("Tiles");

	missing_source_label = memnew(Label);
	missing_source_label->set_text(TTR("This TileMap's TileSet has no source configured. Edit the TileSet resource to add one."));
	missing_source_label->set_h_size_flags(SIZE_EXPAND_FILL);
	missing_source_label->set_v_size_flags(SIZE_EXPAND_FILL);
	missing_source_label->set_align(Label::ALIGN_CENTER);
	missing_source_label->set_valign(Label::VALIGN_CENTER);
	missing_source_label->hide();
	add_child(missing_source_label);

	atlas_sources_split_container = memnew(HSplitContainer);
	atlas_sources_split_container->set_h_size_flags(SIZE_EXPAND_FILL);
	atlas_sources_split_container->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(atlas_sources_split_container);

	sources_list = memnew(ItemList);
	sources_list->set_fixed_icon_size(Size2i(60, 60) * EDSCALE);
	sources_list->set_h_size_flags(SIZE_EXPAND_FILL);
	sources_list->set_stretch_ratio(0.25);
	sources_list->set_custom_minimum_size(Size2i(70, 0) * EDSCALE);
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditorTilesPlugin::_update_fix_selected_and_hovered).unbind(1));
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditorTilesPlugin::_update_atlas_view).unbind(1));
	sources_list->connect("item_selected", callable_mp(TilesEditor::get_singleton(), &TilesEditor::set_atlas_sources_lists_current));
	sources_list->connect("visibility_changed", callable_mp(TilesEditor::get_singleton(), &TilesEditor::synchronize_atlas_sources_list), varray(sources_list));
	//sources_list->set_drag_forwarding(this);
	atlas_sources_split_container->add_child(sources_list);

	tile_atlas_view = memnew(TileAtlasView);
	tile_atlas_view->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view->set_texture_grid_visible(false);
	tile_atlas_view->set_tile_shape_grid_visible(false);
	tile_atlas_view->connect("transform_changed", callable_mp(TilesEditor::get_singleton(), &TilesEditor::set_atlas_view_transform));
	//tile_atlas_view->connect("visibility_changed", callable_mp(TilesEditor::get_singleton(), &TilesEditor::synchronize_atlas_view), varray(tile_atlas_view));
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

	_update_bottom_panel();
}

TileMapEditorTilesPlugin::~TileMapEditorTilesPlugin() {
	memdelete(selection_pattern);
	memdelete(tile_map_clipboard);
}

void TileMapEditorTerrainsPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			paint_tool_button->set_icon(get_theme_icon("Edit", "EditorIcons"));
			picker_button->set_icon(get_theme_icon("ColorPick", "EditorIcons"));
			erase_button->set_icon(get_theme_icon("Eraser", "EditorIcons"));
			break;
	}
}

void TileMapEditorTerrainsPlugin::tile_set_changed() {
	_update_terrains();
}

void TileMapEditorTerrainsPlugin::_update_toolbar() {
	// Hide all settings.
	for (int i = 0; i < tools_settings->get_child_count(); i++) {
		Object::cast_to<CanvasItem>(tools_settings->get_child(i))->hide();
	}

	// Show only the correct settings.
	if (tool_buttons_group->get_pressed_button() == paint_tool_button) {
		tools_settings_vsep->show();
		picker_button->show();
		erase_button->show();
	}
}

Control *TileMapEditorTerrainsPlugin::get_toolbar() const {
	return toolbar;
}

void TileMapEditorTerrainsPlugin::_update_terrains() {
	per_terrain_tiles.clear();
	tilemap_tab_terrains_list->clear();

	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	// Group tiles per terrain.
	per_terrain_tiles.resize(tile_set->get_terrains_count());
	for (int source_index = 0; source_index < tile_set->get_source_count(); source_index++) {
		int source_id = tile_set->get_source_id(source_index);
		Ref<TileSetSource> source = tile_set->get_source(source_id);

		Ref<TileSetAtlasSource> atlas_source = source;
		if (atlas_source.is_valid()) {
			for (int tile_index = 0; tile_index < source->get_tiles_count(); tile_index++) {
				Vector2i tile_id = source->get_tile_id(tile_index);
				for (int alternative_index = 0; alternative_index < source->get_alternative_tiles_count(tile_id); alternative_index++) {
					int alternative_id = source->get_alternative_tile_id(tile_id, alternative_index);

					TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(tile_id, alternative_id));

					// Central terrain bit.
					int terrain = tile_data->get_terrain();
					if (terrain >= 0 && terrain < per_terrain_tiles.size()) {
						TileMapCell c;
						c.source_id = source_id;
						c.set_atlas_coords(tile_id);
						c.alternative_tile = alternative_id;
						per_terrain_tiles.write[terrain].insert(c);
						continue;
					}

					// Other bits.
					for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
						terrain = tile_data->get_peering_bit_terrain((TileSet::CellNeighbor)i);
						if (terrain >= 0 && terrain < per_terrain_tiles.size()) {
							TileMapCell c;
							c.source_id = source_id;
							c.set_atlas_coords(tile_id);
							c.alternative_tile = alternative_id;
							per_terrain_tiles.write[terrain].insert(c);
							break;
						}
					}
				}
			}
		}
	}

	// Fill in the terrain list.
	for (int terrain_index = 0; terrain_index < tile_set->get_terrains_count(); terrain_index++) {
		// Compute the cell used for terrain preview (whenever possible).
		TileMapCell c;
		int max_bit_count = -1;
		for (Set<TileMapCell>::Element *E = per_terrain_tiles[terrain_index].front(); E; E = E->next()) {
			Ref<TileSetSource> source = tile_set->get_source(E->get().source_id);

			Ref<TileSetAtlasSource> atlas_source = source;
			if (atlas_source.is_valid()) {
				TileData *tile_data = Object::cast_to<TileData>(atlas_source->get_tile_data(E->get().get_atlas_coords(), E->get().alternative_tile));
				int count = 0;
				if (tile_data->get_terrain() == terrain_index) {
					for (int peering_bit = 0; peering_bit < TileSet::CELL_NEIGHBOR_MAX; peering_bit++) {
						if (tile_data->get_peering_bit_terrain((TileSet::CellNeighbor)peering_bit) == terrain_index) {
							count++;
						}
					}
					if (count > max_bit_count) {
						c = E->get();
						max_bit_count = count;
					}
				}
			}
		}

		// Get the preview.
		Ref<Texture2D> icon;
		Rect2 region;
		if (max_bit_count >= 0) {
			Ref<TileSetSource> source = tile_set->get_source(c.source_id);

			Ref<TileSetAtlasSource> atlas_source = source;
			if (atlas_source.is_valid()) {
				icon = atlas_source->get_texture();
				region = atlas_source->get_tile_texture_region(c.get_atlas_coords());
			}
		} else {
			Ref<Image> image;
			image.instance();
			image->create(1, 1, false, Image::FORMAT_RGBA8);
			image->set_pixel(0, 0, tile_set->get_terrain_color(terrain_index));
			Ref<ImageTexture> image_texture;
			image_texture.instance();
			image_texture->create_from_image(image);
			icon = image_texture;
		}

		// Add the item to the terrain list.
		tilemap_tab_terrains_list->add_item(tile_set->get_terrain_name(terrain_index), icon);
		tilemap_tab_terrains_list->set_item_icon_region(terrain_index, region);
	}
}

void TileMapEditorTerrainsPlugin::edit(ObjectID p_tile_map_id) {
	tile_map_id = p_tile_map_id;
	_update_terrains();
}

TileMapEditorTerrainsPlugin::TileMapEditorTerrainsPlugin() {
	set_name("Terrains");

	HSplitContainer *tilemap_tab_terrains = memnew(HSplitContainer);
	tilemap_tab_terrains->set_h_size_flags(SIZE_EXPAND_FILL);
	tilemap_tab_terrains->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(tilemap_tab_terrains);

	tilemap_tab_terrains_list = memnew(ItemList);
	tilemap_tab_terrains_list->set_fixed_icon_size(Size2i(30, 30) * EDSCALE);
	tilemap_tab_terrains_list->set_h_size_flags(SIZE_EXPAND_FILL);
	tilemap_tab_terrains_list->set_stretch_ratio(0.25);
	tilemap_tab_terrains_list->set_custom_minimum_size(Size2i(70, 0) * EDSCALE);
	tilemap_tab_terrains->add_child(tilemap_tab_terrains_list);

	// --- Toolbar ---
	toolbar = memnew(HBoxContainer);

	HBoxContainer *tilemap_tiles_tools_buttons = memnew(HBoxContainer);

	tool_buttons_group.instance();

	paint_tool_button = memnew(Button);
	paint_tool_button->set_flat(true);
	paint_tool_button->set_toggle_mode(true);
	paint_tool_button->set_button_group(tool_buttons_group);
	paint_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/paint_tool", "Paint", KEY_E));
	paint_tool_button->connect("pressed", callable_mp(this, &TileMapEditorTerrainsPlugin::_update_toolbar));
	tilemap_tiles_tools_buttons->add_child(paint_tool_button);

	toolbar->add_child(tilemap_tiles_tools_buttons);

	// -- TileMap tool settings --
	tools_settings = memnew(HBoxContainer);
	toolbar->add_child(tools_settings);

	tools_settings_vsep = memnew(VSeparator);
	tools_settings->add_child(tools_settings_vsep);

	// Picker
	picker_button = memnew(Button);
	picker_button->set_flat(true);
	picker_button->set_toggle_mode(true);
	picker_button->set_shortcut(ED_SHORTCUT("tiles_editor/picker", "Picker", KEY_P));
	picker_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(picker_button);

	// Erase button.
	erase_button = memnew(Button);
	erase_button->set_flat(true);
	erase_button->set_toggle_mode(true);
	erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", "Eraser", KEY_E));
	erase_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tools_settings->add_child(erase_button);
}

TileMapEditorTerrainsPlugin::~TileMapEditorTerrainsPlugin() {
}

void TileMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			missing_tile_texture = get_theme_icon("StatusWarning", "EditorIcons");
			warning_pattern_texture = get_theme_icon("WarningPattern", "EditorIcons");
			break;
		case NOTIFICATION_INTERNAL_PROCESS:
			if (is_visible_in_tree() && tileset_changed_needs_update) {
				_update_bottom_panel();
				tile_map_editor_plugins[tabs->get_current_tab()]->tile_set_changed();
				CanvasItemEditor::get_singleton()->update_viewport();
				tileset_changed_needs_update = false;
			}
			break;
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
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		tile_map_editor_plugins[i]->set_visible(i == tabs->get_current_tab());
	}
}

Vector<Vector2i> TileMapEditor::get_line(TileMap *p_tile_map, Vector2i p_from_cell, Vector2i p_to_cell) {
	ERR_FAIL_COND_V(!p_tile_map, Vector<Vector2i>());

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
	tile_map_editor_plugins[tabs->get_current_tab()]->edit(tile_map_id);

	// Update toolbar.
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		tile_map_editor_plugins[i]->get_toolbar()->set_visible(i == p_tab_id);
	}

	// Update visible panel.
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		tile_map_editor_plugins[i]->set_visible(i == tabs->get_current_tab());
	}

	// Graphical update.
	tile_map_editor_plugins[tabs->get_current_tab()]->update();
	CanvasItemEditor::get_singleton()->update_viewport();
}

bool TileMapEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	return tile_map_editor_plugins[tabs->get_current_tab()]->forward_canvas_gui_input(p_event);
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

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * tile_map->get_global_transform();
	Transform2D xform_inv = xform.affine_inverse();
	Vector2i tile_shape_size = tile_set->get_tile_size();

	// Draw tiles with invalid IDs in the grid.
	float icon_ratio = MIN(missing_tile_texture->get_size().x / tile_set->get_tile_size().x, missing_tile_texture->get_size().y / tile_set->get_tile_size().y) / 3;
	TypedArray<Vector2i> used_cells = tile_map->get_used_cells();
	for (int i = 0; i < used_cells.size(); i++) {
		Vector2i coords = used_cells[i];
		int tile_source_id = tile_map->get_cell_source_id(coords);
		if (tile_source_id >= 0) {
			Vector2i tile_atlas_coords = tile_map->get_cell_atlas_coords(coords);
			int tile_alternative_tile = tile_map->get_cell_alternative_tile(coords);

			TileSetSource *source = nullptr;
			if (tile_set->has_source(tile_source_id)) {
				source = *tile_set->get_source(tile_source_id);
			}

			if (!source || !source->has_tile(tile_atlas_coords) || !source->has_alternative_tile(tile_atlas_coords, tile_alternative_tile)) {
				// Generate a random color from the hashed values of the tiles.
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
				Rect2 cell_region = xform.xform(Rect2(tile_map->map_to_world(coords) - Vector2(tile_shape_size) / 2, Vector2(tile_shape_size)));
				tile_set->draw_tile_shape(p_overlay, cell_region, color, true, warning_pattern_texture);

				// Draw the warning icon.
				Rect2 rect = Rect2(xform.xform(tile_map->map_to_world(coords)) - (icon_ratio * missing_tile_texture->get_size() * xform.get_scale() / 2), icon_ratio * missing_tile_texture->get_size() * xform.get_scale());
				p_overlay->draw_texture_rect(missing_tile_texture, rect);
			}
		}
	}

	// Fading on the border.
	const int fading = 5;

	// Determine the drawn area.
	Size2 screen_size = p_overlay->get_size();
	Rect2i screen_rect;
	screen_rect.position = tile_map->world_to_map(xform_inv.xform(Vector2()));
	screen_rect.expand_to(tile_map->world_to_map(xform_inv.xform(Vector2(0, screen_size.height))));
	screen_rect.expand_to(tile_map->world_to_map(xform_inv.xform(Vector2(screen_size.width, 0))));
	screen_rect.expand_to(tile_map->world_to_map(xform_inv.xform(screen_size)));
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
	for (int x = displayed_rect.position.x; x < (displayed_rect.position.x + displayed_rect.size.x); x++) {
		for (int y = displayed_rect.position.y; y < (displayed_rect.position.y + displayed_rect.size.y); y++) {
			Vector2i pos_in_rect = Vector2i(x, y) - displayed_rect.position;

			// Fade out the border of the grid.
			float left_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.x), 0.0f, 1.0f);
			float right_opacity = CLAMP(Math::inverse_lerp((float)displayed_rect.size.x, (float)(displayed_rect.size.x - fading), (float)pos_in_rect.x), 0.0f, 1.0f);
			float top_opacity = CLAMP(Math::inverse_lerp(0.0f, (float)fading, (float)pos_in_rect.y), 0.0f, 1.0f);
			float bottom_opacity = CLAMP(Math::inverse_lerp((float)displayed_rect.size.y, (float)(displayed_rect.size.y - fading), (float)pos_in_rect.y), 0.0f, 1.0f);
			float opacity = CLAMP(MIN(left_opacity, MIN(right_opacity, MIN(top_opacity, bottom_opacity))) + 0.1, 0.0f, 1.0f);

			Rect2 cell_region = xform.xform(Rect2(tile_map->map_to_world(Vector2(x, y)) - tile_shape_size / 2, tile_shape_size));
			tile_set->draw_tile_shape(p_overlay, cell_region, Color(1.0, 0.5, 0.2, 0.5 * opacity), false);
		}
	}

	// Draw the IDs for debug.
	/*Ref<Font> font = get_theme_font("font", "Label");
	for (int x = displayed_rect.position.x; x < (displayed_rect.position.x + displayed_rect.size.x); x++) {
		for (int y = displayed_rect.position.y; y < (displayed_rect.position.y + displayed_rect.size.y); y++) {
			p_overlay->draw_string(font, xform.xform(tile_map->map_to_world(Vector2(x, y))) + Vector2i(-tile_shape_size.x / 2, 0), vformat("%s", Vector2(x, y)));
		}
	}*/

	// Draw the plugins.
	tile_map_editor_plugins[tabs->get_current_tab()]->forward_canvas_draw_over_viewport(p_overlay);
}

void TileMapEditor::edit(TileMap *p_tile_map) {
	if (p_tile_map && p_tile_map->get_instance_id() == tile_map_id) {
		return;
	}

	// Disconnect to changes.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		tile_map->disconnect("changed", callable_mp(this, &TileMapEditor::_tile_map_changed));
	}

	// Change the edited object.
	if (p_tile_map) {
		tile_map_id = p_tile_map->get_instance_id();
		tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
		// Connect to changes.
		if (!tile_map->is_connected("changed", callable_mp(this, &TileMapEditor::_tile_map_changed))) {
			tile_map->connect("changed", callable_mp(this, &TileMapEditor::_tile_map_changed));
		}
	} else {
		tile_map_id = ObjectID();
	}

	// Call the plugins.
	tile_map_editor_plugins[tabs->get_current_tab()]->edit(tile_map_id);

	_tile_map_changed();
}

TileMapEditor::TileMapEditor() {
	set_process_internal(true);

	// TileMap editor plugins
	tile_map_editor_plugins.push_back(memnew(TileMapEditorTilesPlugin));
	tile_map_editor_plugins.push_back(memnew(TileMapEditorTerrainsPlugin));

	// Tabs.
	tabs = memnew(Tabs);
	tabs->set_clip_tabs(false);
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		tabs->add_tab(tile_map_editor_plugins[i]->get_name());
	}
	tabs->connect("tab_changed", callable_mp(this, &TileMapEditor::_tab_changed));

	// --- TileMap toolbar ---
	tilemap_toolbar = memnew(HBoxContainer);
	//tilemap_toolbar->add_child(memnew(VSeparator));
	tilemap_toolbar->add_child(tabs);
	//tilemap_toolbar->add_child(memnew(VSeparator));
	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		tile_map_editor_plugins[i]->get_toolbar()->hide();
		tilemap_toolbar->add_child(tile_map_editor_plugins[i]->get_toolbar());
	}

	missing_tileset_label = memnew(Label);
	missing_tileset_label->set_text(TTR("The edited TileMap node has no TileSet resource."));
	missing_tileset_label->set_h_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_v_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_align(Label::ALIGN_CENTER);
	missing_tileset_label->set_valign(Label::VALIGN_CENTER);
	missing_tileset_label->hide();
	add_child(missing_tileset_label);

	for (int i = 0; i < tile_map_editor_plugins.size(); i++) {
		add_child(tile_map_editor_plugins[i]);
		tile_map_editor_plugins[i]->set_h_size_flags(SIZE_EXPAND_FILL);
		tile_map_editor_plugins[i]->set_v_size_flags(SIZE_EXPAND_FILL);
		tile_map_editor_plugins[i]->set_visible(i == 0);
	}

	_tab_changed(0);
}

TileMapEditor::~TileMapEditor() {
}
