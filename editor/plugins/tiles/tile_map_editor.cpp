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

void TileMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			tilemap_select_tool_button->set_icon(get_theme_icon("ToolSelect", "EditorIcons"));
			tilemap_paint_tool_button->set_icon(get_theme_icon("Edit", "EditorIcons"));
			tilemap_line_tool_button->set_icon(get_theme_icon("CurveLinear", "EditorIcons"));
			tilemap_rect_tool_button->set_icon(get_theme_icon("Rectangle", "EditorIcons"));
			tilemap_bucket_tool_button->set_icon(get_theme_icon("Bucket", "EditorIcons"));

			tilemap_picker_button->set_icon(get_theme_icon("ColorPick", "EditorIcons"));
			tilemap_erase_button->set_icon(get_theme_icon("Eraser", "EditorIcons"));

			missing_tile_texture = get_theme_icon("StatusWarning", "EditorIcons");
			warning_pattern_texture = get_theme_icon("WarningPattern", "EditorIcons");

			missing_texture_texture = get_theme_icon("TileSet", "EditorIcons");
			break;
		case NOTIFICATION_INTERNAL_PROCESS:
			if (is_visible_in_tree() && tileset_changed_needs_update) {
				_update_fix_selected_and_hovered();
				_update_bottom_panel();
				_update_tile_set_atlas_sources_list();
				_update_atlas_view();
				tileset_changed_needs_update = false;
			}
			break;
	}
}

void TileMapEditor::_update_bottom_panel() {
	if (!tile_map) {
		return;
	}

	// Update the visibility of controls.
	Ref<TileSet> tile_set = tile_map->get_tileset();
	missing_tileset_label->set_visible(!tile_set.is_valid());
	tileset_tabs_container->set_visible(tile_set.is_valid());

	// Update the tabs.
	bool valid_sources = tile_set.is_valid() && tile_set->get_atlas_source_count() > 0;
	missing_atlas_source_label->set_visible(!valid_sources);
	atlas_sources_split_container->set_visible(valid_sources);
}

void TileMapEditor::_update_tile_set_atlas_sources_list() {
	// Update the atlas sources.

	int old_current = sources_list->get_current();
	sources_list->clear();

	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return;
	}

	for (int i = 0; i < tile_set->get_atlas_source_count(); i++) {
		int source_id = tile_set->get_atlas_source_id(i);
		Ref<Texture2D> texture = tile_set->get_atlas_source(source_id)->get_texture();
		if (texture.is_valid()) {
			sources_list->add_item(vformat("%s - (id:%d)", texture->get_path().get_file(), source_id), texture);
		} else {
			sources_list->add_item(vformat("No texture atlas source - (id:%d)", source_id), missing_texture_texture);
		}
		sources_list->set_item_metadata(sources_list->get_item_count() - 1, source_id);
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

void TileMapEditor::_update_atlas_view() {
	// Update the atlas display.
	if (!tile_map) {
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		tile_atlas_view->set_atlas_source(nullptr, -1);
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index >= 0 && source_index < sources_list->get_item_count()) {
		int source_id = sources_list->get_item_metadata(source_index);
		tile_atlas_view->set_atlas_source(*tile_map->get_tileset(), source_id);
	} else {
		tile_atlas_view->set_atlas_source(*tile_map->get_tileset(), -1);
	}

	// Synchronize atlas view.
	TilesEditor::get_singleton()->synchronize_atlas_view(tile_atlas_view);

	tile_atlas_control->update();
}

void TileMapEditor::_tile_map_changed() {
	tileset_changed_needs_update = true;
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_update_toolbar() {
	// Hide all settings.
	for (int i = 0; i < tilemap_tools_settings->get_child_count(); i++) {
		Object::cast_to<CanvasItem>(tilemap_tools_settings->get_child(i))->hide();
	}

	// Show only the correct settings.
	if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_select_tool_button) {
	} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_paint_tool_button) {
		tilemap_tools_settings_vsep->show();
		tilemap_picker_button->show();
		tilemap_erase_button->show();
		tilemap_tools_settings_vsep_2->show();
		tilemap_random_tile_checkbox->show();
		tilemap_scatter_label->show();
		tilemap_scatter_spinbox->show();
	} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_line_tool_button) {
		tilemap_tools_settings_vsep->show();
		tilemap_picker_button->show();
		tilemap_erase_button->show();
		tilemap_tools_settings_vsep_2->show();
		tilemap_random_tile_checkbox->show();
		tilemap_scatter_label->show();
		tilemap_scatter_spinbox->show();
	} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_rect_tool_button) {
		tilemap_tools_settings_vsep->show();
		tilemap_picker_button->show();
		tilemap_erase_button->show();
		tilemap_tools_settings_vsep_2->show();
		tilemap_random_tile_checkbox->show();
		tilemap_scatter_label->show();
		tilemap_scatter_spinbox->show();
	} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_bucket_tool_button) {
		tilemap_tools_settings_vsep->show();
		tilemap_picker_button->show();
		tilemap_erase_button->show();
		tilemap_tools_settings_vsep_2->show();
		tilemap_bucket_continuous_checkbox->show();
		tilemap_random_tile_checkbox->show();
		tilemap_scatter_label->show();
		tilemap_scatter_spinbox->show();
	}
}

void TileMapEditor::_mouse_exited_viewport() {
	has_mouse = false;
	CanvasItemEditor::get_singleton()->update_viewport();
}

Vector<Vector2i> TileMapEditor::_get_line(Vector2i p_from_cell, Vector2i p_to_cell) {
	ERR_FAIL_COND_V(!tile_map, Vector<Vector2i>());

	Ref<TileSet> tile_set = tile_map->get_tileset();
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

TileMapCell TileMapEditor::_pick_random_tile(const TileMapPattern *p_pattern) {
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

		if (tile_set->has_atlas_source(source_id)) {
			sum += tile_set->get_atlas_source(source_id)->get_tile_data(atlas_coords, alternative_tile)->tile_get_probability();
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
		if (tile_set->has_atlas_source(source_id)) {
			current += tile_set->get_atlas_source(source_id)->get_tile_data(atlas_coords, alternative_tile)->tile_get_probability();
		} else {
			sum += 1.0;
		}

		if (current >= rand) {
			return TileMapCell(source_id, atlas_coords, alternative_tile);
		}
	}
	return TileMapCell();
}

Map<Vector2i, TileMapCell> TileMapEditor::_draw_line(Vector2 p_start_drag_mouse_pos, Vector2 p_from_mouse_pos, Vector2i p_to_mouse_pos) {
	if (!tile_map) {
		return Map<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	TileMapPattern erase_pattern;
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = tilemap_erase_button->is_pressed() ? &erase_pattern : selection_pattern;

	Map<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		// Paint the tiles on the tile map.
		if (!tilemap_erase_button->is_pressed() && tilemap_random_tile_checkbox->is_pressed()) {
			// Paint a random tile.
			Vector<Vector2i> line = _get_line(tile_map->world_to_map(p_from_mouse_pos), tile_map->world_to_map(p_to_mouse_pos));
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
			Vector<Vector2i> line = _get_line((last_hovered_cell - offset) / pattern->get_size(), (new_hovered_cell - offset) / pattern->get_size());
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

Map<Vector2i, TileMapCell> TileMapEditor::_draw_rect(Vector2i p_start_cell, Vector2i p_end_cell) {
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
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = tilemap_erase_button->is_pressed() ? &erase_pattern : selection_pattern;

	Map<Vector2i, TileMapCell> output;
	if (!pattern->is_empty()) {
		if (!tilemap_erase_button->is_pressed() && tilemap_random_tile_checkbox->is_pressed()) {
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
					for (int j = 0; j < used_cells.size(); j++) {
						Vector2i coords = rect.position + Vector2i(x, y) * pattern->get_size() + used_cells[j];
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

Map<Vector2i, TileMapCell> TileMapEditor::_draw_bucket_fill(Vector2i p_coords, bool p_contiguous) {
	if (!tile_map) {
		return Map<Vector2i, TileMapCell>();
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		return Map<Vector2i, TileMapCell>();
	}

	// Get or create the pattern.
	TileMapPattern erase_pattern;
	erase_pattern.set_cell(Vector2i(0, 0), -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
	TileMapPattern *pattern = tilemap_erase_button->is_pressed() ? &erase_pattern : selection_pattern;

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
						if (!tilemap_erase_button->is_pressed() && tilemap_random_tile_checkbox->is_pressed()) {
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
					if (!tilemap_erase_button->is_pressed() && tilemap_random_tile_checkbox->is_pressed()) {
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

bool TileMapEditor::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	if (!is_visible_in_tree()) {
		// If the bottom editor is not visible, we ignore inputs.
		return false;
	}

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
				undo_redo->create_action("Delete tiles");
				for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
					undo_redo->add_do_method(tile_map, "set_cell", E->get(), -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
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
		return true;
	}
	if (ED_IS_SHORTCUT("tiles_editor/delete", p_event)) {
		// Delete selected tiles.
		if (!tile_map_selection.is_empty()) {
			undo_redo->create_action("Delete tiles");
			for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
				undo_redo->add_do_method(tile_map, "set_cell", E->get(), -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
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
					if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
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
				Vector<Vector2i> line = _get_line(tile_map->world_to_map(drag_last_mouse_pos), tile_map->world_to_map(mpos));
				for (int i = 0; i < line.size(); i++) {
					if (!drag_modified.has(line[i])) {
						Map<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], tilemap_bucket_continuous_checkbox->is_pressed());
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
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

		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed()) {
				// Pressed
				if (drag_type == DRAG_TYPE_NONE) {
					if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_select_tool_button) {
						drag_start_mouse_pos = mpos;
						if (tile_map_selection.has(tile_map->world_to_map(drag_start_mouse_pos)) && !mb->get_shift()) {
							// Move the selection
							drag_type = DRAG_TYPE_MOVE;
							drag_modified.clear();
							for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
								Vector2i coords = E->get();
								drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
								tile_map->set_cell(coords, -1, TileAtlasSource::INVALID_ATLAS_COORDS, TileAtlasSource::INVALID_TILE_ALTERNATIVE);
							}
						} else {
							// Select tiles
							drag_type = DRAG_TYPE_SELECT;
						}
					} else {
						// Check if we are picking a tile.
						if (tilemap_picker_button->is_pressed()) {
							drag_type = DRAG_TYPE_PICK;
							drag_start_mouse_pos = mpos;
						} else {
							// Paint otherwise.
							if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_paint_tool_button) {
								drag_type = DRAG_TYPE_PAINT;
								drag_start_mouse_pos = mpos;
								drag_modified.clear();
								Map<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, mpos, mpos);
								for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
									if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
										continue;
									}
									Vector2i coords = E->key();
									if (!drag_modified.has(coords)) {
										drag_modified.insert(coords, TileMapCell(tile_map->get_cell_source_id(coords), tile_map->get_cell_atlas_coords(coords), tile_map->get_cell_alternative_tile(coords)));
									}
									tile_map->set_cell(coords, E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
								}
							} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_line_tool_button) {
								drag_type = DRAG_TYPE_LINE;
								drag_start_mouse_pos = mpos;
								drag_modified.clear();
							} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_rect_tool_button) {
								drag_type = DRAG_TYPE_RECT;
								drag_start_mouse_pos = mpos;
								drag_modified.clear();
							} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_bucket_tool_button) {
								drag_type = DRAG_TYPE_BUCKET;
								drag_start_mouse_pos = mpos;
								drag_modified.clear();
								Vector<Vector2i> line = _get_line(tile_map->world_to_map(drag_last_mouse_pos), tile_map->world_to_map(mpos));
								for (int i = 0; i < line.size(); i++) {
									if (!drag_modified.has(line[i])) {
										Map<Vector2i, TileMapCell> to_draw = _draw_bucket_fill(line[i], tilemap_bucket_continuous_checkbox->is_pressed());
										for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
											if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
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
				}
			} else {
				// Released
				switch (drag_type) {
					case DRAG_TYPE_SELECT: {
						undo_redo->create_action("Change selection");
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
						undo_redo->create_action("Move tiles");
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
						tilemap_picker_button->set_pressed(false);
					} break;
					case DRAG_TYPE_PAINT: {
						undo_redo->create_action("Paint tiles");
						for (Map<Vector2i, TileMapCell>::Element *E = drag_modified.front(); E; E = E->next()) {
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
						undo_redo->commit_action(false);
					} break;
					case DRAG_TYPE_LINE: {
						Map<Vector2i, TileMapCell> to_draw = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, mpos);
						undo_redo->create_action("Paint tiles");
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
						}
						undo_redo->commit_action();
					} break;
					case DRAG_TYPE_RECT: {
						Map<Vector2i, TileMapCell> to_draw = _draw_rect(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(mpos));
						undo_redo->create_action("Paint tiles");
						for (Map<Vector2i, TileMapCell>::Element *E = to_draw.front(); E; E = E->next()) {
							if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
						}
						undo_redo->commit_action();
					} break;
					case DRAG_TYPE_BUCKET: {
						undo_redo->create_action("Paint tiles");
						for (Map<Vector2i, TileMapCell>::Element *E = drag_modified.front(); E; E = E->next()) {
							if (!tilemap_erase_button->is_pressed() && E->get().source_id == -1) {
								continue;
							}
							undo_redo->add_do_method(tile_map, "set_cell", E->key(), tile_map->get_cell_source_id(E->key()), tile_map->get_cell_atlas_coords(E->key()), tile_map->get_cell_alternative_tile(E->key()));
							undo_redo->add_undo_method(tile_map, "set_cell", E->key(), E->get().source_id, E->get().get_atlas_coords(), E->get().alternative_tile);
						}
						undo_redo->commit_action(false);
					} break;
					case DRAG_TYPE_CLIPBOARD_PASTE: {
						Vector2 mouse_offset = (Vector2(tile_map_clipboard->get_size()) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
						undo_redo->create_action("Paste tiles");
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

void TileMapEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
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
		if (tile_source_id != TileSet::SOURCE_TYPE_INVALID) {
			Vector2i tile_atlas_coords = tile_map->get_cell_atlas_coords(coords);
			int tile_alternative_tile = tile_map->get_cell_alternative_tile(coords);

			if (!tile_set->has_atlas_source(tile_source_id) || !tile_set->get_atlas_source(tile_source_id)->has_tile(tile_atlas_coords) || !tile_set->get_atlas_source(tile_source_id)->has_alternative_tile(tile_atlas_coords, tile_alternative_tile)) {
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
	int fading = 5;

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

	// Draw the selection.
	if (is_visible_in_tree() && tilemap_tool_buttons_group->get_pressed_button() == tilemap_select_tool_button) {
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
		} else if (!tilemap_picker_button->is_pressed()) {
			if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_paint_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a single pattern.
				preview = _draw_line(drag_last_mouse_pos, drag_last_mouse_pos, drag_last_mouse_pos);
				if (!preview.is_empty()) {
					drawn_grid_rect = Rect2i(preview.front()->key(), Vector2i(1, 1));
					for (Map<Vector2i, TileMapCell>::Element *E = preview.front(); E; E = E->next()) {
						drawn_grid_rect.expand_to(E->key());
					}
				}
			} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_line_tool_button && drag_type == DRAG_TYPE_LINE) {
				// Preview for a line pattern.
				preview = _draw_line(drag_start_mouse_pos, drag_start_mouse_pos, drag_last_mouse_pos);
				if (!preview.is_empty()) {
					drawn_grid_rect = Rect2i(preview.front()->key(), Vector2i(1, 1));
					for (Map<Vector2i, TileMapCell>::Element *E = preview.front(); E; E = E->next()) {
						drawn_grid_rect.expand_to(E->key());
					}
				}
			} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_rect_tool_button && drag_type == DRAG_TYPE_RECT) {
				// Preview for a line pattern.
				preview = _draw_rect(tile_map->world_to_map(drag_start_mouse_pos), tile_map->world_to_map(drag_last_mouse_pos));
				if (!preview.is_empty()) {
					drawn_grid_rect = Rect2i(preview.front()->key(), Vector2i(1, 1));
					for (Map<Vector2i, TileMapCell>::Element *E = preview.front(); E; E = E->next()) {
						drawn_grid_rect.expand_to(E->key());
					}
				}
			} else if (tilemap_tool_buttons_group->get_pressed_button() == tilemap_bucket_tool_button && drag_type == DRAG_TYPE_NONE) {
				// Preview for a line pattern.
				preview = _draw_bucket_fill(tile_map->world_to_map(drag_last_mouse_pos), tilemap_bucket_continuous_checkbox->is_pressed());
			}
		}

		if (!preview.is_empty()) {
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
				if (!tilemap_erase_button->is_pressed() && tilemap_random_tile_checkbox->is_pressed()) {
					Vector2i size = tile_set->get_tile_size();
					Vector2 position = tile_map->map_to_world(E->key()) - size / 2;
					Rect2 cell_region = xform.xform(Rect2(position, size));

					tile_set->draw_tile_shape(p_overlay, cell_region, Color(1.0, 1.0, 1.0, 0.5), true);
				} else {
					switch (tile_set->get_source_type(E->get().source_id)) {
						case TileSet::SOURCE_TYPE_ATLAS: {
							TileAtlasSource *atlas = tile_set->get_atlas_source(E->get().source_id);
							Rect2 tile_region = atlas->get_tile_texture_region(E->get().get_atlas_coords());
							Vector2i tile_offset = tile_set->get_tile_effective_texture_offset(E->get().source_id, E->get().get_atlas_coords(), 0);
							Vector2 position = tile_map->map_to_world(E->key()) - tile_region.size / 2 - tile_offset;
							Rect2 cell_region = xform.xform(Rect2(position, tile_region.size));

							p_overlay->draw_texture_rect_region(atlas->get_texture(), cell_region, tile_region, Color(1.0, 1.0, 1.0, 0.5));
						} break;
						case TileSet::SOURCE_TYPE_SCENE:
							// TODO
							break;
						default:
							Vector2i size = tile_set->get_tile_size();
							Vector2 position = tile_map->map_to_world(E->key()) - size / 2;
							Rect2 cell_region = xform.xform(Rect2(position, size));

							tile_set->draw_tile_shape(p_overlay, cell_region, Color(0.0, 0.0, 0.0, 0.5), true);
							break;
					}
				}
			}
		}
	}
}

void TileMapEditor::edit(TileMap *p_tile_map) {
	if (p_tile_map == tile_map) {
		return;
	}

	// Change the edited object.
	tile_map = p_tile_map;

	// Add the listener again.
	if (tile_map) {
		tile_map->connect("changed", callable_mp(this, &TileMapEditor::_tile_map_changed));
	}

	// Clean the selection.
	tile_set_selection.clear();
	tile_map_selection.clear();
	selection_pattern->clear();

	_tile_map_changed();
}

void TileMapEditor::_update_fix_selected_and_hovered() {
	if (!tile_map) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	Ref<TileSet> tile_set = tile_map->get_tileset();
	if (!tile_set.is_valid()) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	int source_index = sources_list->get_current();
	if (source_index < 0 || source_index >= sources_list->get_item_count()) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
		tile_set_selection.clear();
		tile_map_selection.clear();
		selection_pattern->clear();
		return;
	}

	int source_id = sources_list->get_item_metadata(source_index);

	// Clear hovered if needed.
	if (source_id != hovered_tile.source_id ||
			!tile_set->has_atlas_source(hovered_tile.source_id) ||
			!tile_set->get_atlas_source(hovered_tile.source_id)->has_tile(hovered_tile.get_atlas_coords()) ||
			!tile_set->get_atlas_source(hovered_tile.source_id)->has_alternative_tile(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile)) {
		hovered_tile.source_id = -1;
		hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
		hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
	}

	// Selection if needed.
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		const TileMapCell *selected = &(E->get());
		if (!tile_set->has_atlas_source(selected->source_id) ||
				!tile_set->get_atlas_source(selected->source_id)->has_tile(selected->get_atlas_coords()) ||
				!tile_set->get_atlas_source(selected->source_id)->has_alternative_tile(selected->get_atlas_coords(), selected->alternative_tile)) {
			tile_set_selection.erase(E);
		}
	}

	if (!tile_map_selection.is_empty()) {
		_update_selection_pattern_from_tilemap_selection();
	} else {
		_update_selection_pattern_from_tileset_selection();
	}
}

void TileMapEditor::_update_selection_pattern_from_tilemap_selection() {
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

void TileMapEditor::_update_selection_pattern_from_tileset_selection() {
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
		switch (tile_set->get_source_type(E_source->key())) {
			case TileSet::SOURCE_TYPE_ATLAS: {
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
			} break;
			case TileSet::SOURCE_TYPE_SCENE:
				// Add everything unorganized.
				for (List<const TileMapCell *>::Element *E_cell = E_source->get().front(); E_cell; E_cell = E_cell->next()) {
					unorganized.push_back(E_cell->get());
				}
				break;
			default:
				break;
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

void TileMapEditor::_update_tileset_selection_from_selection_pattern() {
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

void TileMapEditor::_tile_atlas_control_draw() {
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
	if (!tile_set->has_atlas_source(source_id)) {
		return;
	}

	TileAtlasSource *atlas = tile_set->get_atlas_source(source_id);

	// Draw the selection.
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		if (E->get().source_id == source_id && E->get().alternative_tile == 0) {
			tile_atlas_control->draw_rect(atlas->get_tile_texture_region(E->get().get_atlas_coords()), Color(0.0, 0.0, 1.0), false);
		}
	}

	// Draw the hovered tile.
	if (hovered_tile.get_atlas_coords() != TileAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0 && !tile_set_dragging_selection) {
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
				if (tile != TileAtlasSource::INVALID_ATLAS_COORDS) {
					to_draw.insert(tile);
				}
			}
		}

		for (Set<Vector2i>::Element *E = to_draw.front(); E; E = E->next()) {
			tile_atlas_control->draw_rect(atlas->get_tile_texture_region(E->get()), Color(0.8, 0.8, 1.0), false);
		}
	}
}

void TileMapEditor::_tile_atlas_control_mouse_exited() {
	hovered_tile.source_id = -1;
	hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
	tile_set_dragging_selection = false;
	tile_atlas_control->update();
}

void TileMapEditor::_tile_atlas_control_gui_input(const Ref<InputEvent> &p_event) {
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
	if (!tile_set->has_atlas_source(source_id)) {
		return;
	}

	TileAtlasSource *atlas = tile_set->get_atlas_source(source_id);

	// Update the hovered tile
	hovered_tile.source_id = source_id;
	hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
	Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
	if (coords != TileAtlasSource::INVALID_ATLAS_COORDS) {
		coords = atlas->get_tile_at_coords(coords);
		if (coords != TileAtlasSource::INVALID_ATLAS_COORDS) {
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
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {
		if (mb->is_pressed()) { // Pressed
			tile_set_dragging_selection = true;
			tile_set_drag_start_mouse_pos = tile_atlas_control->get_local_mouse_position();
			if (!mb->get_shift()) {
				tile_set_selection.clear();
			}

			if (hovered_tile.get_atlas_coords() != TileAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile == 0) {
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
				if (start_tile != TileAtlasSource::INVALID_ATLAS_COORDS && end_tile != TileAtlasSource::INVALID_ATLAS_COORDS) {
					Rect2i region = Rect2i(start_tile, end_tile - start_tile).abs();
					region.size += Vector2i(1, 1);

					// To update the selection, we copy the selected/not selected status of the tiles we drag from.
					Vector2i start_coords = atlas->get_tile_at_coords(start_tile);
					if (mb->get_shift() && start_coords != TileAtlasSource::INVALID_ATLAS_COORDS && !tile_set_selection.has(TileMapCell(source_id, start_coords, 0))) {
						// Remove from the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileAtlasSource::INVALID_ATLAS_COORDS && tile_set_selection.has(TileMapCell(source_id, tile_coords, 0))) {
									tile_set_selection.erase(TileMapCell(source_id, tile_coords, 0));
								}
							}
						}
					} else {
						// Insert in the selection.
						for (int x = region.position.x; x < region.get_end().x; x++) {
							for (int y = region.position.y; y < region.get_end().y; y++) {
								Vector2i tile_coords = atlas->get_tile_at_coords(Vector2i(x, y));
								if (tile_coords != TileAtlasSource::INVALID_ATLAS_COORDS) {
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

void TileMapEditor::_tile_alternatives_control_draw() {
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
	if (!tile_set->has_atlas_source(source_id)) {
		return;
	}

	// Draw the selection.
	for (Set<TileMapCell>::Element *E = tile_set_selection.front(); E; E = E->next()) {
		if (E->get().source_id == source_id && E->get().get_atlas_coords() != TileAtlasSource::INVALID_ATLAS_COORDS && E->get().alternative_tile > 0) {
			Rect2i rect = tile_atlas_view->get_alternative_tile_rect(E->get().get_atlas_coords(), E->get().alternative_tile);
			if (rect != Rect2i()) {
				alternative_tiles_control->draw_rect(rect, Color(0.2, 0.2, 1.0), false);
			}
		}
	}

	// Draw hovered tile.
	if (hovered_tile.get_atlas_coords() != TileAtlasSource::INVALID_ATLAS_COORDS && hovered_tile.alternative_tile > 0) {
		Rect2i rect = tile_atlas_view->get_alternative_tile_rect(hovered_tile.get_atlas_coords(), hovered_tile.alternative_tile);
		if (rect != Rect2i()) {
			alternative_tiles_control->draw_rect(rect, Color(1.0, 1.0, 1.0), false);
		}
	}
}

void TileMapEditor::_tile_alternatives_control_mouse_exited() {
	hovered_tile.source_id = -1;
	hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
	tile_set_dragging_selection = false;
	alternative_tiles_control->update();
}

void TileMapEditor::_tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event) {
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
	if (!tile_set->has_atlas_source(source_id)) {
		return;
	}

	// Update the hovered tile
	hovered_tile.source_id = source_id;
	hovered_tile.set_atlas_coords(TileAtlasSource::INVALID_ATLAS_COORDS);
	hovered_tile.alternative_tile = TileAtlasSource::INVALID_TILE_ALTERNATIVE;
	Vector3i alternative_coords = tile_atlas_view->get_alternative_tile_at_pos(alternative_tiles_control->get_local_mouse_position());
	Vector2i coords = Vector2i(alternative_coords.x, alternative_coords.y);
	int alternative = alternative_coords.z;
	if (coords != TileAtlasSource::INVALID_ATLAS_COORDS && alternative != TileAtlasSource::INVALID_TILE_ALTERNATIVE) {
		hovered_tile.set_atlas_coords(coords);
		hovered_tile.alternative_tile = alternative;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->update();
		alternative_tiles_control->update();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->get_button_index() == BUTTON_LEFT) {
		if (mb->is_pressed()) { // Pressed
			// Left click pressed.
			if (!mb->get_shift()) {
				tile_set_selection.clear();
			}

			if (coords != TileAtlasSource::INVALID_ATLAS_COORDS && alternative != TileAtlasSource::INVALID_TILE_ALTERNATIVE) {
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

void TileMapEditor::_set_tile_map_selection(const TypedArray<Vector2i> &p_selection) {
	tile_map_selection.clear();
	for (int i = 0; i < p_selection.size(); i++) {
		tile_map_selection.insert(p_selection[i]);
	}
	_update_selection_pattern_from_tilemap_selection();
	_update_tileset_selection_from_selection_pattern();
	CanvasItemEditor::get_singleton()->update_viewport();
}

TypedArray<Vector2i> TileMapEditor::_get_tile_map_selection() const {
	TypedArray<Vector2i> output;
	for (Set<Vector2i>::Element *E = tile_map_selection.front(); E; E = E->next()) {
		output.push_back(E->get());
	}
	return output;
}

void TileMapEditor::_on_random_tile_checkbox_toggled(bool p_pressed) {
	tilemap_scatter_spinbox->set_editable(p_pressed);
}

void TileMapEditor::_on_scattering_spinbox_changed(double p_value) {
	scattering = p_value;
}

void TileMapEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_set_tile_map_selection", "selection"), &TileMapEditor::_set_tile_map_selection);
	ClassDB::bind_method(D_METHOD("_get_tile_map_selection"), &TileMapEditor::_get_tile_map_selection);
}

TileMapEditor::TileMapEditor() {
	set_process_internal(true);

	CanvasItemEditor::get_singleton()->get_viewport_control()->connect("mouse_exited", callable_mp(this, &TileMapEditor::_mouse_exited_viewport));

	// Shortcuts
	ED_SHORTCUT("tiles_editor/cut", TTR("Cut"), KEY_MASK_CMD | KEY_X);
	ED_SHORTCUT("tiles_editor/copy", TTR("Copy"), KEY_MASK_CMD | KEY_C);
	ED_SHORTCUT("tiles_editor/paste", TTR("Paste"), KEY_MASK_CMD | KEY_V);
	ED_SHORTCUT("tiles_editor/delete", TTR("Delete"), KEY_DELETE);

	// --- TileMap toolbar ---
	tilemap_toolbar = memnew(HBoxContainer);
	tilemap_toolbar->add_child(memnew(VSeparator));

	// -- TileMap tool selection --
	HBoxContainer *tilemap_tools_buttons = memnew(HBoxContainer);
	tilemap_toolbar->add_child(tilemap_tools_buttons);

	tilemap_tool_buttons_group.instance();

	tilemap_select_tool_button = memnew(Button);
	tilemap_select_tool_button->set_flat(true);
	tilemap_select_tool_button->set_toggle_mode(true);
	tilemap_select_tool_button->set_button_group(tilemap_tool_buttons_group);
	tilemap_select_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/selection_tool", "Selection", KEY_S));
	tilemap_select_tool_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_toolbar));
	tilemap_tools_buttons->add_child(tilemap_select_tool_button);

	tilemap_paint_tool_button = memnew(Button);
	tilemap_paint_tool_button->set_flat(true);
	tilemap_paint_tool_button->set_toggle_mode(true);
	tilemap_paint_tool_button->set_button_group(tilemap_tool_buttons_group);
	tilemap_paint_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/paint_tool", "Paint", KEY_E));
	tilemap_paint_tool_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_toolbar));
	tilemap_tools_buttons->add_child(tilemap_paint_tool_button);

	tilemap_line_tool_button = memnew(Button);
	tilemap_line_tool_button->set_flat(true);
	tilemap_line_tool_button->set_toggle_mode(true);
	tilemap_line_tool_button->set_button_group(tilemap_tool_buttons_group);
	tilemap_line_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/line_tool", "Line", KEY_L));
	tilemap_line_tool_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_toolbar));
	tilemap_tools_buttons->add_child(tilemap_line_tool_button);

	tilemap_rect_tool_button = memnew(Button);
	tilemap_rect_tool_button->set_flat(true);
	tilemap_rect_tool_button->set_toggle_mode(true);
	tilemap_rect_tool_button->set_button_group(tilemap_tool_buttons_group);
	tilemap_rect_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/rect_tool", "Rect", KEY_R));
	tilemap_rect_tool_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_toolbar));
	tilemap_tools_buttons->add_child(tilemap_rect_tool_button);

	tilemap_bucket_tool_button = memnew(Button);
	tilemap_bucket_tool_button->set_flat(true);
	tilemap_bucket_tool_button->set_toggle_mode(true);
	tilemap_bucket_tool_button->set_button_group(tilemap_tool_buttons_group);
	tilemap_bucket_tool_button->set_shortcut(ED_SHORTCUT("tiles_editor/bucket_tool", "Bucket", KEY_B));
	tilemap_bucket_tool_button->connect("pressed", callable_mp(this, &TileMapEditor::_update_toolbar));
	tilemap_tools_buttons->add_child(tilemap_bucket_tool_button);

	// -- TileMap tool settings --
	tilemap_tools_settings = memnew(HBoxContainer);
	tilemap_toolbar->add_child(tilemap_tools_settings);

	tilemap_tools_settings_vsep = memnew(VSeparator);
	tilemap_tools_settings->add_child(tilemap_tools_settings_vsep);

	// Picker
	tilemap_picker_button = memnew(Button);
	tilemap_picker_button->set_flat(true);
	tilemap_picker_button->set_toggle_mode(true);
	tilemap_picker_button->set_shortcut(ED_SHORTCUT("tiles_editor/picker", "Picker", KEY_P));
	tilemap_picker_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tilemap_tools_settings->add_child(tilemap_picker_button);

	// Erase button.
	tilemap_erase_button = memnew(Button);
	tilemap_erase_button->set_flat(true);
	tilemap_erase_button->set_toggle_mode(true);
	tilemap_erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", "Eraser", KEY_E));
	tilemap_erase_button->connect("pressed", callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	tilemap_tools_settings->add_child(tilemap_erase_button);

	// Separator 2.
	tilemap_tools_settings_vsep_2 = memnew(VSeparator);
	tilemap_tools_settings->add_child(tilemap_tools_settings_vsep_2);

	// Continuous checkbox.
	tilemap_bucket_continuous_checkbox = memnew(CheckBox);
	tilemap_bucket_continuous_checkbox->set_flat(true);
	tilemap_bucket_continuous_checkbox->set_text(TTR("Contiguous"));
	tilemap_tools_settings->add_child(tilemap_bucket_continuous_checkbox);

	// Random tile checkbox.
	tilemap_random_tile_checkbox = memnew(CheckBox);
	tilemap_random_tile_checkbox->set_flat(true);
	tilemap_random_tile_checkbox->set_text(TTR("Place random tile"));
	tilemap_random_tile_checkbox->connect("toggled", callable_mp(this, &TileMapEditor::_on_random_tile_checkbox_toggled));
	tilemap_tools_settings->add_child(tilemap_random_tile_checkbox);

	// Random tile scattering.
	tilemap_scatter_label = memnew(Label);
	tilemap_scatter_label->set_tooltip(TTR("Defines the probability of painting nothing instead of a randomly selected tile."));
	tilemap_scatter_label->set_text(TTR("Scattering:"));
	tilemap_tools_settings->add_child(tilemap_scatter_label);

	tilemap_scatter_spinbox = memnew(SpinBox);
	tilemap_scatter_spinbox->set_min(0.0);
	tilemap_scatter_spinbox->set_max(1000);
	tilemap_scatter_spinbox->set_step(0.001);
	tilemap_scatter_spinbox->set_tooltip(TTR("Defines the probability of painting nothing instead of a randomly selected tile."));
	tilemap_scatter_spinbox->get_line_edit()->add_theme_constant_override("minimum_spaces", 3);
	tilemap_scatter_spinbox->connect("value_changed", callable_mp(this, &TileMapEditor::_on_scattering_spinbox_changed));
	tilemap_tools_settings->add_child(tilemap_scatter_spinbox);

	_on_random_tile_checkbox_toggled(false);

	// Default tool.
	tilemap_paint_tool_button->set_pressed(true);
	_update_toolbar();

	// --- Tabs ---
	missing_tileset_label = memnew(Label);
	missing_tileset_label->set_text(TTR("The edited TileMap node has no TileSet resource."));
	missing_tileset_label->set_h_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_v_size_flags(SIZE_EXPAND_FILL);
	missing_tileset_label->set_align(Label::ALIGN_CENTER);
	missing_tileset_label->set_valign(Label::VALIGN_CENTER);
	missing_tileset_label->hide();
	add_child(missing_tileset_label);

	tileset_tabs_container = memnew(TabContainer);
	tileset_tabs_container->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tabs_container->set_tab_align(TabContainer::ALIGN_LEFT);
	add_child(tileset_tabs_container);

	// -- Atlas tab --
	HBoxContainer *tileset_tab_atlas = memnew(HBoxContainer);
	tileset_tab_atlas->set_name("Atlases");
	tileset_tab_atlas->set_h_size_flags(SIZE_EXPAND_FILL);
	tileset_tab_atlas->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tabs_container->add_child(tileset_tab_atlas);

	missing_atlas_source_label = memnew(Label);
	missing_atlas_source_label->set_text(TTR("This TileMap's TileSet has no Atlas configured. Edit the TileSet resource to add one."));
	missing_atlas_source_label->set_h_size_flags(SIZE_EXPAND_FILL);
	missing_atlas_source_label->set_v_size_flags(SIZE_EXPAND_FILL);
	missing_atlas_source_label->set_align(Label::ALIGN_CENTER);
	missing_atlas_source_label->set_valign(Label::VALIGN_CENTER);
	missing_atlas_source_label->hide();
	tileset_tab_atlas->add_child(missing_atlas_source_label);

	atlas_sources_split_container = memnew(HSplitContainer);
	atlas_sources_split_container->set_h_size_flags(SIZE_EXPAND_FILL);
	atlas_sources_split_container->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tab_atlas->add_child(atlas_sources_split_container);

	sources_list = memnew(ItemList);
	sources_list->set_fixed_icon_size(Size2i(60, 60) * EDSCALE);
	sources_list->set_h_size_flags(SIZE_EXPAND_FILL);
	sources_list->set_stretch_ratio(0.25);
	sources_list->set_custom_minimum_size(Size2i(70, 0) * EDSCALE);
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditor::_update_fix_selected_and_hovered).unbind(1));
	sources_list->connect("item_selected", callable_mp(this, &TileMapEditor::_update_atlas_view).unbind(1));
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
	tile_atlas_control->connect("draw", callable_mp(this, &TileMapEditor::_tile_atlas_control_draw));
	tile_atlas_control->connect("mouse_exited", callable_mp(this, &TileMapEditor::_tile_atlas_control_mouse_exited));
	tile_atlas_control->connect("gui_input", callable_mp(this, &TileMapEditor::_tile_atlas_control_gui_input));
	tile_atlas_view->add_control_over_atlas_tiles(tile_atlas_control);

	alternative_tiles_control = memnew(Control);
	alternative_tiles_control->connect("draw", callable_mp(this, &TileMapEditor::_tile_alternatives_control_draw));
	alternative_tiles_control->connect("mouse_exited", callable_mp(this, &TileMapEditor::_tile_alternatives_control_mouse_exited));
	alternative_tiles_control->connect("gui_input", callable_mp(this, &TileMapEditor::_tile_alternatives_control_gui_input));
	tile_atlas_view->add_control_over_alternative_tiles(alternative_tiles_control);

	_update_bottom_panel();

	// -- Terrains tab --
	ItemList *tilemap_tab_terrains = memnew(ItemList);
	tilemap_tab_terrains->set_name("Terrains");
	tilemap_tab_terrains->set_h_size_flags(SIZE_EXPAND_FILL);
	tilemap_tab_terrains->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tabs_container->add_child(tilemap_tab_terrains);

	// -- Scenes tab --
	ItemList *tilemap_tab_scenes = memnew(ItemList);
	tilemap_tab_scenes->set_name("Scenes");
	tileset_tabs_container->add_child(tilemap_tab_scenes);

	// Disable unused tabs.
	tileset_tabs_container->set_tab_disabled(1, true);
	tileset_tabs_container->set_tab_disabled(2, true);
}
