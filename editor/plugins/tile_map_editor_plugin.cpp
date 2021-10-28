/*************************************************************************/
/*  tile_map_editor_plugin.cpp                                           */
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

#include "tile_map_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "core/math/math_funcs.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/gui/split_container.h"

void TileMapEditor::_node_removed(Node *p_node) {
	if (p_node == node) {
		node = nullptr;
	}
}

void TileMapEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			if (bucket_queue.size()) {
				CanvasItemEditor::get_singleton()->update_viewport();
			}

		} break;

		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", this, "_node_removed");
			FALLTHROUGH;
		}

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (is_visible_in_tree()) {
				_update_palette();
			}

			paint_button->set_icon(get_icon("Edit", "EditorIcons"));
			bucket_fill_button->set_icon(get_icon("Bucket", "EditorIcons"));
			picker_button->set_icon(get_icon("ColorPick", "EditorIcons"));
			select_button->set_icon(get_icon("ActionCopy", "EditorIcons"));

			rotate_left_button->set_icon(get_icon("RotateLeft", "EditorIcons"));
			rotate_right_button->set_icon(get_icon("RotateRight", "EditorIcons"));
			flip_horizontal_button->set_icon(get_icon("MirrorX", "EditorIcons"));
			flip_vertical_button->set_icon(get_icon("MirrorY", "EditorIcons"));
			clear_transform_button->set_icon(get_icon("Clear", "EditorIcons"));

			search_box->set_right_icon(get_icon("Search", "EditorIcons"));
			search_box->set_clear_button_enabled(true);

			PopupMenu *p = options->get_popup();
			p->set_item_icon(p->get_item_index(OPTION_CUT), get_icon("ActionCut", "EditorIcons"));
			p->set_item_icon(p->get_item_index(OPTION_COPY), get_icon("Duplicate", "EditorIcons"));
			p->set_item_icon(p->get_item_index(OPTION_ERASE_SELECTION), get_icon("Remove", "EditorIcons"));

		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", this, "_node_removed");
		} break;

		case NOTIFICATION_WM_FOCUS_OUT: {
			if (tool == TOOL_PAINTING) {
				Vector<int> ids = get_selected_tiles();

				if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
					_set_cell(over_tile, ids, flip_h, flip_v, transpose);
					_finish_undo();

					paint_undo.clear();
				}

				tool = TOOL_NONE;
				_update_button_tool();
			}

			// set flag to ignore over_tile on refocus
			refocus_over_tile = true;
		} break;
	}
}

void TileMapEditor::_update_button_tool() {
	ToolButton *tb[4] = { paint_button, bucket_fill_button, picker_button, select_button };
	// Unpress all buttons
	for (int i = 0; i < 4; i++) {
		tb[i]->set_pressed(false);
	}

	// Press the good button
	switch (tool) {
		case TOOL_NONE:
		case TOOL_PAINTING: {
			paint_button->set_pressed(true);
		} break;
		case TOOL_BUCKET: {
			bucket_fill_button->set_pressed(true);
		} break;
		case TOOL_PICKING: {
			picker_button->set_pressed(true);
		} break;
		case TOOL_SELECTING: {
			select_button->set_pressed(true);
		} break;
		default:
			break;
	}

	if (tool != TOOL_PICKING) {
		last_tool = tool;
	}
}

void TileMapEditor::_button_tool_select(int p_tool) {
	tool = (Tool)p_tool;
	_update_button_tool();
	switch (tool) {
		case TOOL_SELECTING: {
			selection_active = false;
		} break;
		default:
			break;
	}
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_menu_option(int p_option) {
	switch (p_option) {
		case OPTION_COPY: {
			_update_copydata();

			if (selection_active) {
				tool = TOOL_PASTING;

				CanvasItemEditor::get_singleton()->update_viewport();
			}
		} break;
		case OPTION_ERASE_SELECTION: {
			if (!selection_active) {
				return;
			}

			_start_undo(TTR("Erase Selection"));
			_erase_selection();
			_finish_undo();

			selection_active = false;
			copydata.clear();

			CanvasItemEditor::get_singleton()->update_viewport();
		} break;
		case OPTION_FIX_INVALID: {
			undo_redo->create_action(TTR("Fix Invalid Tiles"));
			undo_redo->add_undo_method(node, "set", "tile_data", node->get("tile_data"));
			node->fix_invalid_tiles();
			undo_redo->add_do_method(node, "set", "tile_data", node->get("tile_data"));
			undo_redo->commit_action();

		} break;
		case OPTION_CUT: {
			if (selection_active) {
				_update_copydata();

				_start_undo(TTR("Cut Selection"));
				_erase_selection();
				_finish_undo();

				selection_active = false;

				tool = TOOL_PASTING;

				CanvasItemEditor::get_singleton()->update_viewport();
			}
		} break;
	}
	_update_button_tool();
}

void TileMapEditor::_palette_selected(int index) {
	_update_palette();
}

void TileMapEditor::_palette_multi_selected(int index, bool selected) {
	_update_palette();
}

void TileMapEditor::_palette_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mb = p_event;

	// Zoom in/out using Ctrl + mouse wheel.
	if (mb.is_valid() && mb->is_pressed() && mb->get_command()) {
		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_UP) {
			size_slider->set_value(size_slider->get_value() + 0.2);
		}

		if (mb->is_pressed() && mb->get_button_index() == BUTTON_WHEEL_DOWN) {
			size_slider->set_value(size_slider->get_value() - 0.2);
		}
	}
}

void TileMapEditor::_canvas_mouse_enter() {
	mouse_over = true;
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_canvas_mouse_exit() {
	mouse_over = false;
	CanvasItemEditor::get_singleton()->update_viewport();
}

Vector<int> TileMapEditor::get_selected_tiles() const {
	Vector<int> items = palette->get_selected_items();

	if (items.size() == 0) {
		items.push_back(TileMap::INVALID_CELL);
		return items;
	}

	for (int i = items.size() - 1; i >= 0; i--) {
		items.write[i] = palette->get_item_metadata(items[i]);
	}
	return items;
}

void TileMapEditor::set_selected_tiles(Vector<int> p_tiles) {
	palette->unselect_all();

	for (int i = p_tiles.size() - 1; i >= 0; i--) {
		int idx = palette->find_metadata(p_tiles[i]);

		if (idx >= 0) {
			palette->select(idx, false);
		}
	}

	palette->ensure_current_is_visible();
}

Dictionary TileMapEditor::_create_cell_dictionary(int tile, bool flip_x, bool flip_y, bool transpose, Vector2 autotile_coord) {
	Dictionary cell;

	cell["id"] = tile;
	cell["flip_h"] = flip_x;
	cell["flip_y"] = flip_y;
	cell["transpose"] = transpose;
	cell["auto_coord"] = autotile_coord;

	return cell;
}

void TileMapEditor::_create_set_cell_undo_redo(const Vector2 &p_vec, const CellOp &p_cell_old, const CellOp &p_cell_new) {
	Dictionary cell_old = _create_cell_dictionary(p_cell_old.idx, p_cell_old.xf, p_cell_old.yf, p_cell_old.tr, p_cell_old.ac);
	Dictionary cell_new = _create_cell_dictionary(p_cell_new.idx, p_cell_new.xf, p_cell_new.yf, p_cell_new.tr, p_cell_new.ac);

	undo_redo->add_undo_method(node, "_set_celld", p_vec, cell_old);
	undo_redo->add_do_method(node, "_set_celld", p_vec, cell_new);
}

void TileMapEditor::_start_undo(const String &p_action) {
	undo_data.clear();
	undo_redo->create_action(p_action);
}

void TileMapEditor::_finish_undo() {
	if (undo_data.size()) {
		for (Map<Point2i, CellOp>::Element *E = undo_data.front(); E; E = E->next()) {
			_create_set_cell_undo_redo(E->key(), E->get(), _get_op_from_cell(E->key()));
		}

		undo_data.clear();
	}

	undo_redo->commit_action();
}

void TileMapEditor::_set_cell(const Point2i &p_pos, Vector<int> p_values, bool p_flip_h, bool p_flip_v, bool p_transpose, const Point2i &p_autotile_coord) {
	ERR_FAIL_COND(!node);

	if (p_values.size() == 0) {
		return;
	}

	int p_value = p_values[Math::rand() % p_values.size()];
	int prev_val = node->get_cell(p_pos.x, p_pos.y);

	bool prev_flip_h = node->is_cell_x_flipped(p_pos.x, p_pos.y);
	bool prev_flip_v = node->is_cell_y_flipped(p_pos.x, p_pos.y);
	bool prev_transpose = node->is_cell_transposed(p_pos.x, p_pos.y);
	Vector2 prev_position = node->get_cell_autotile_coord(p_pos.x, p_pos.y);

	Vector2 position;
	int current = manual_palette->get_current();
	if (current != -1) {
		if (tool != TOOL_PASTING) {
			position = manual_palette->get_item_metadata(current);
		} else {
			position = p_autotile_coord;
		}
	} else {
		// If there is no manual tile selected, that either means that
		// autotiling is enabled, or the given tile is not autotiling. Either
		// way, the coordinate of the tile does not matter, so assigning it to
		// the coordinate of the existing tile works fine.
		position = prev_position;
	}

	if (p_value == prev_val && p_flip_h == prev_flip_h && p_flip_v == prev_flip_v && p_transpose == prev_transpose && prev_position == position) {
		return; // Check that it's actually different.
	}

	for (int y = p_pos.y - 1; y <= p_pos.y + 1; y++) {
		for (int x = p_pos.x - 1; x <= p_pos.x + 1; x++) {
			Point2i p = Point2i(x, y);
			if (!undo_data.has(p)) {
				undo_data[p] = _get_op_from_cell(p);
			}
		}
	}

	node->_set_celld(p_pos, _create_cell_dictionary(p_value, p_flip_h, p_flip_v, p_transpose, p_autotile_coord));

	if (tool == TOOL_PASTING) {
		return;
	}

	if (manual_autotile || (p_value != -1 && node->get_tileset()->has_tile(p_value) && node->get_tileset()->tile_get_tile_mode(p_value) == TileSet::ATLAS_TILE)) {
		if (current != -1) {
			node->set_cell_autotile_coord(p_pos.x, p_pos.y, position);
		} else if (node->get_tileset()->tile_get_tile_mode(p_value) == TileSet::ATLAS_TILE && priority_atlastile) {
			// BIND_CENTER is used to indicate that bitmask should not update for this tile cell.
			node->get_tileset()->autotile_set_bitmask(p_value, Vector2(p_pos.x, p_pos.y), TileSet::BIND_CENTER);
			node->update_cell_bitmask(p_pos.x, p_pos.y);
		}
	} else {
		node->update_bitmask_area(Point2(p_pos));
	}
}

void TileMapEditor::_manual_toggled(bool p_enabled) {
	manual_autotile = p_enabled;
	_update_palette();
}

void TileMapEditor::_priority_toggled(bool p_enabled) {
	priority_atlastile = p_enabled;
	_update_palette();
}

void TileMapEditor::_text_entered(const String &p_text) {
	canvas_item_editor_viewport->grab_focus();
}

void TileMapEditor::_text_changed(const String &p_text) {
	_update_palette();
}

void TileMapEditor::_sbox_input(const Ref<InputEvent> &p_ie) {
	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_scancode() == KEY_UP || k->get_scancode() == KEY_DOWN || k->get_scancode() == KEY_PAGEUP || k->get_scancode() == KEY_PAGEDOWN)) {
		palette->call("_gui_input", k);
		search_box->accept_event();
	}
}

// Implementation detail of TileMapEditor::_update_palette();
// In modern C++ this could have been inside its body.
namespace {
struct _PaletteEntry {
	int id;
	String name;

	bool operator<(const _PaletteEntry &p_rhs) const {
		// Natural no case comparison will compare strings based on CharType
		// order (except digits) and on numbers that start on the same position.
		return name.naturalnocasecmp_to(p_rhs.name) < 0;
	}
};
} // namespace

void TileMapEditor::_update_palette() {
	if (!node) {
		return;
	}

	// Update the clear button.
	clear_transform_button->set_disabled(!flip_h && !flip_v && !transpose);

	// Update the palette.
	Vector<int> selected = get_selected_tiles();
	int selected_single = palette->get_current();
	int selected_manual = manual_palette->get_current();
	palette->clear();
	manual_palette->clear();
	manual_palette->hide();

	Ref<TileSet> tileset = node->get_tileset();
	if (tileset.is_null()) {
		search_box->set_text("");
		search_box->set_editable(false);
		info_message->show();
		return;
	}

	search_box->set_editable(true);
	info_message->hide();

	List<int> tiles;
	tileset->get_tile_list(&tiles);
	if (tiles.empty()) {
		return;
	}

	float min_size = EDITOR_DEF("editors/tile_map/preview_size", 64);
	min_size *= EDSCALE;
	int hseparation = EDITOR_DEF("editors/tile_map/palette_item_hseparation", 8);
	bool show_tile_names = bool(EDITOR_DEF("editors/tile_map/show_tile_names", true));
	bool show_tile_ids = bool(EDITOR_DEF("editors/tile_map/show_tile_ids", false));
	bool sort_by_name = bool(EDITOR_DEF("editors/tile_map/sort_tiles_by_name", true));

	palette->add_constant_override("hseparation", hseparation * EDSCALE);

	palette->set_fixed_icon_size(Size2(min_size, min_size));
	palette->set_fixed_column_width(min_size * MAX(size_slider->get_value(), 1));
	palette->set_same_column_width(true);
	manual_palette->set_fixed_icon_size(Size2(min_size, min_size));
	manual_palette->set_same_column_width(true);

	String filter = search_box->get_text().strip_edges();

	Vector<_PaletteEntry> entries;

	for (List<int>::Element *E = tiles.front(); E; E = E->next()) {
		String name = tileset->tile_get_name(E->get());

		if (name != "") {
			if (show_tile_ids) {
				if (sort_by_name) {
					name = name + " - " + itos(E->get());
				} else {
					name = itos(E->get()) + " - " + name;
				}
			}
		} else {
			name = "#" + itos(E->get());
		}

		if (filter != "" && !filter.is_subsequence_ofi(name)) {
			continue;
		}

		const _PaletteEntry entry = { E->get(), name };
		entries.push_back(entry);
	}

	if (sort_by_name) {
		entries.sort();
	}

	for (int i = 0; i < entries.size(); i++) {
		if (show_tile_names) {
			palette->add_item(entries[i].name);
		} else {
			palette->add_item(String());
		}

		Ref<Texture> tex = tileset->tile_get_texture(entries[i].id);

		if (tex.is_valid()) {
			Rect2 region = tileset->tile_get_region(entries[i].id);

			if (tileset->tile_get_tile_mode(entries[i].id) == TileSet::AUTO_TILE || tileset->tile_get_tile_mode(entries[i].id) == TileSet::ATLAS_TILE) {
				int spacing = tileset->autotile_get_spacing(entries[i].id);
				region.size = tileset->autotile_get_size(entries[i].id);
				region.position += (region.size + Vector2(spacing, spacing)) * tileset->autotile_get_icon_coordinate(entries[i].id);
			}

			// Transpose and flip.
			palette->set_item_icon_transposed(palette->get_item_count() - 1, transpose);
			if (flip_h) {
				region.size.x = -region.size.x;
			}
			if (flip_v) {
				region.size.y = -region.size.y;
			}

			// Set region.
			if (region.size != Size2()) {
				palette->set_item_icon_region(palette->get_item_count() - 1, region);
			}

			// Set icon.
			palette->set_item_icon(palette->get_item_count() - 1, tex);

			// Modulation.
			Color color = tileset->tile_get_modulate(entries[i].id);
			palette->set_item_icon_modulate(palette->get_item_count() - 1, color);
		}

		palette->set_item_metadata(palette->get_item_count() - 1, entries[i].id);
	}

	int sel_tile = selected.get(0);
	if (selected.get(0) != TileMap::INVALID_CELL) {
		set_selected_tiles(selected);
		sel_tile = selected.get(Math::rand() % selected.size());
	} else if (palette->get_item_count() > 0) {
		palette->select(0);
		sel_tile = palette->get_selected_items().get(0);
	}

	if (sel_tile != TileMap::INVALID_CELL && tileset->has_tile(sel_tile) && ((manual_autotile && tileset->tile_get_tile_mode(sel_tile) == TileSet::AUTO_TILE) || (!priority_atlastile && tileset->tile_get_tile_mode(sel_tile) == TileSet::ATLAS_TILE))) {
		const Map<Vector2, uint32_t> &tiles2 = tileset->autotile_get_bitmask_map(sel_tile);

		Vector<Vector2> entries2;
		for (const Map<Vector2, uint32_t>::Element *E = tiles2.front(); E; E = E->next()) {
			entries2.push_back(E->key());
		}
		// Sort tiles in row-major order.
		struct SwapComparator {
			_FORCE_INLINE_ bool operator()(const Vector2 &v_l, const Vector2 &v_r) const {
				return v_l.y != v_r.y ? v_l.y < v_r.y : v_l.x < v_r.x;
			}
		};
		entries2.sort_custom<SwapComparator>();

		Ref<Texture> tex = tileset->tile_get_texture(sel_tile);
		Color modulate = tileset->tile_get_modulate(sel_tile);

		for (int i = 0; i < entries2.size(); i++) {
			manual_palette->add_item(String());

			if (tex.is_valid()) {
				Rect2 region = tileset->tile_get_region(sel_tile);
				int spacing = tileset->autotile_get_spacing(sel_tile);
				region.size = tileset->autotile_get_size(sel_tile); // !!
				region.position += (region.size + Vector2(spacing, spacing)) * entries2[i];

				if (!region.has_no_area()) {
					manual_palette->set_item_icon_region(manual_palette->get_item_count() - 1, region);
				}

				manual_palette->set_item_icon(manual_palette->get_item_count() - 1, tex);
				manual_palette->set_item_icon_modulate(manual_palette->get_item_count() - 1, modulate);
			}

			manual_palette->set_item_metadata(manual_palette->get_item_count() - 1, entries2[i]);
		}
	}

	if (manual_palette->get_item_count() > 0) {
		// Only show the manual palette if at least tile exists in it.
		if (selected_manual == -1 || selected_single != palette->get_current()) {
			selected_manual = 0;
		}
		if (selected_manual < manual_palette->get_item_count()) {
			manual_palette->set_current(selected_manual);
		}
		manual_palette->show();
	}

	if (sel_tile != TileMap::INVALID_CELL && tileset->has_tile(sel_tile) && tileset->tile_get_tile_mode(sel_tile) == TileSet::AUTO_TILE) {
		manual_button->show();
		priority_button->hide();
	} else {
		manual_button->hide();
		priority_button->show();
	}
}

void TileMapEditor::_pick_tile(const Point2 &p_pos) {
	int id = node->get_cell(p_pos.x, p_pos.y);

	if (id == TileMap::INVALID_CELL || !node->get_tileset()->has_tile(id)) {
		return;
	}

	if (search_box->get_text() != "") {
		search_box->set_text("");
		_update_palette();
	}

	flip_h = node->is_cell_x_flipped(p_pos.x, p_pos.y);
	flip_v = node->is_cell_y_flipped(p_pos.x, p_pos.y);
	transpose = node->is_cell_transposed(p_pos.x, p_pos.y);
	autotile_coord = node->get_cell_autotile_coord(p_pos.x, p_pos.y);

	Vector<int> selected;
	selected.push_back(id);
	set_selected_tiles(selected);
	_update_palette();

	if ((manual_autotile && node->get_tileset()->tile_get_tile_mode(id) == TileSet::AUTO_TILE) || (!priority_atlastile && node->get_tileset()->tile_get_tile_mode(id) == TileSet::ATLAS_TILE)) {
		manual_palette->select(manual_palette->find_metadata((Point2)autotile_coord));
	}

	CanvasItemEditor::get_singleton()->update_viewport();
}

PoolVector<Vector2> TileMapEditor::_bucket_fill(const Point2i &p_start, bool erase, bool preview) {
	int prev_id = node->get_cell(p_start.x, p_start.y);
	Vector<int> ids;
	ids.push_back(TileMap::INVALID_CELL);
	if (!erase) {
		ids = get_selected_tiles();

		if (ids.size() == 0 || ids[0] == TileMap::INVALID_CELL) {
			return PoolVector<Vector2>();
		}
	} else if (prev_id == TileMap::INVALID_CELL) {
		return PoolVector<Vector2>();
	}

	// Check if the tile variation is the same
	if (ids.size() == 1 && ids[0] == prev_id) {
		int current = manual_palette->get_current();
		if (current == -1) {
			// Same ID, no variation selected, nothing to change
			return PoolVector<Vector2>();
		}
		Vector2 prev_autotile_coord = node->get_cell_autotile_coord(p_start.x, p_start.y);
		Vector2 autotile_coord = manual_palette->get_item_metadata(current);
		if (autotile_coord == prev_autotile_coord) {
			// Same ID and variation, nothing to change
			return PoolVector<Vector2>();
		}
	}

	Rect2i r = node->get_used_rect();

	int area = r.get_area();
	if (preview) {
		// Test if we can re-use the result from preview bucket fill
		bool invalidate_cache = false;
		// Area changed
		if (r != bucket_cache_rect) {
			_clear_bucket_cache();
		}
		// Cache grid is not initialized
		if (bucket_cache_visited == nullptr) {
			bucket_cache_visited = new bool[area];
			invalidate_cache = true;
		}
		// Tile ID changed or position wasn't visited by the previous fill
		const int loc = (p_start.x - r.position.x) + (p_start.y - r.position.y) * r.get_size().x;
		const bool in_range = 0 <= loc && loc < area;
		if (prev_id != bucket_cache_tile || (in_range && !bucket_cache_visited[loc])) {
			invalidate_cache = true;
		}
		if (invalidate_cache) {
			for (int i = 0; i < area; ++i) {
				bucket_cache_visited[i] = false;
			}
			bucket_cache = PoolVector<Vector2>();
			bucket_cache_tile = prev_id;
			bucket_cache_rect = r;
			bucket_queue.clear();
		}
	}

	PoolVector<Vector2> points;
	Vector<Vector2> non_preview_cache;
	int count = 0;
	int limit = 0;

	if (preview) {
		limit = 1024;
	} else {
		bucket_queue.clear();
	}

	bucket_queue.push_back(p_start);

	while (bucket_queue.size()) {
		Point2i n = bucket_queue.front()->get();
		bucket_queue.pop_front();

		if (!r.has_point(n)) {
			continue;
		}

		if (node->get_cell(n.x, n.y) == prev_id) {
			if (preview) {
				int loc = (n.x - r.position.x) + (n.y - r.position.y) * r.get_size().x;
				if (bucket_cache_visited[loc]) {
					continue;
				}
				bucket_cache_visited[loc] = true;
				bucket_cache.push_back(n);
			} else {
				if (non_preview_cache.find(n) >= 0) {
					continue;
				}
				points.push_back(n);
				non_preview_cache.push_back(n);
			}

			bucket_queue.push_back(Point2i(n.x, n.y + 1));
			bucket_queue.push_back(Point2i(n.x, n.y - 1));
			bucket_queue.push_back(Point2i(n.x + 1, n.y));
			bucket_queue.push_back(Point2i(n.x - 1, n.y));
			count++;
		}

		if (limit > 0 && count >= limit) {
			break;
		}
	}

	return preview ? bucket_cache : points;
}

void TileMapEditor::_fill_points(const PoolVector<Vector2> &p_points, const Dictionary &p_op) {
	int len = p_points.size();
	PoolVector<Vector2>::Read pr = p_points.read();

	Vector<int> ids = p_op["id"];
	bool xf = p_op["flip_h"];
	bool yf = p_op["flip_v"];
	bool tr = p_op["transpose"];

	for (int i = 0; i < len; i++) {
		_set_cell(pr[i], ids, xf, yf, tr);
		node->make_bitmask_area_dirty(pr[i]);
	}
	if (!manual_autotile) {
		node->update_dirty_bitmask();
	}
}

void TileMapEditor::_erase_points(const PoolVector<Vector2> &p_points) {
	int len = p_points.size();
	PoolVector<Vector2>::Read pr = p_points.read();

	for (int i = 0; i < len; i++) {
		_set_cell(pr[i], invalid_cell);
	}
}

void TileMapEditor::_select(const Point2i &p_from, const Point2i &p_to) {
	Point2i begin = p_from;
	Point2i end = p_to;

	if (begin.x > end.x) {
		SWAP(begin.x, end.x);
	}
	if (begin.y > end.y) {
		SWAP(begin.y, end.y);
	}

	rectangle.position = begin;
	rectangle.size = end - begin;

	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_erase_selection() {
	if (!selection_active) {
		return;
	}

	for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
		for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {
			_set_cell(Point2i(j, i), invalid_cell, false, false, false);
		}
	}
}

void TileMapEditor::_draw_cell(Control *p_viewport, int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Point2i &p_autotile_coord, const Transform2D &p_xform) {
	if (!node->get_tileset()->has_tile(p_cell)) {
		return;
	}

	Ref<Texture> t = node->get_tileset()->tile_get_texture(p_cell);
	if (t.is_null()) {
		return;
	}

	Vector2 tile_ofs = node->get_tileset()->tile_get_texture_offset(p_cell);

	Rect2 r = node->get_tileset()->tile_get_region(p_cell);
	if (node->get_tileset()->tile_get_tile_mode(p_cell) == TileSet::AUTO_TILE || node->get_tileset()->tile_get_tile_mode(p_cell) == TileSet::ATLAS_TILE) {
		Vector2 offset;
		if (tool != TOOL_PASTING) {
			int selected = manual_palette->get_current();
			if ((manual_autotile || (node->get_tileset()->tile_get_tile_mode(p_cell) == TileSet::ATLAS_TILE && !priority_atlastile)) && selected != -1) {
				offset = manual_palette->get_item_metadata(selected);
			} else {
				offset = node->get_tileset()->autotile_get_icon_coordinate(p_cell);
			}
		} else {
			offset = p_autotile_coord;
		}

		int spacing = node->get_tileset()->autotile_get_spacing(p_cell);
		r.size = node->get_tileset()->autotile_get_size(p_cell);
		r.position += (r.size + Vector2(spacing, spacing)) * offset;
	}
	Size2 cell_size = node->get_cell_size();
	bool centered_texture = node->is_centered_textures_enabled();
	bool compatibility_mode_enabled = node->is_compatibility_mode_enabled();
	Rect2 rect = Rect2();
	rect.position = node->map_to_world(p_point) + node->get_cell_draw_offset();

	if (r.has_no_area()) {
		rect.size = t->get_size();
	} else {
		rect.size = r.size;
	}

	if (compatibility_mode_enabled && !centered_texture) {
		if (rect.size.y > rect.size.x) {
			if ((p_flip_h && (p_flip_v || p_transpose)) || (p_flip_v && !p_transpose)) {
				tile_ofs.y += rect.size.y - rect.size.x;
			}
		} else if (rect.size.y < rect.size.x) {
			if ((p_flip_v && (p_flip_h || p_transpose)) || (p_flip_h && !p_transpose)) {
				tile_ofs.x += rect.size.x - rect.size.y;
			}
		}
	}

	if (p_transpose) {
		SWAP(tile_ofs.x, tile_ofs.y);
		if (centered_texture) {
			rect.position.x += cell_size.x / 2 - rect.size.y / 2;
			rect.position.y += cell_size.y / 2 - rect.size.x / 2;
		}
	} else if (centered_texture) {
		rect.position += cell_size / 2 - rect.size / 2;
	}

	if (p_flip_h) {
		rect.size.x *= -1.0;
		tile_ofs.x *= -1.0;
	}

	if (p_flip_v) {
		rect.size.y *= -1.0;
		tile_ofs.y *= -1.0;
	}

	if (compatibility_mode_enabled && !centered_texture) {
		if (node->get_tile_origin() == TileMap::TILE_ORIGIN_TOP_LEFT) {
			rect.position += tile_ofs;
		} else if (node->get_tile_origin() == TileMap::TILE_ORIGIN_BOTTOM_LEFT) {
			rect.position += tile_ofs;

			if (p_transpose) {
				if (p_flip_h) {
					rect.position.x -= cell_size.x;
				} else {
					rect.position.x += cell_size.x;
				}
			} else {
				if (p_flip_v) {
					rect.position.y -= cell_size.y;
				} else {
					rect.position.y += cell_size.y;
				}
			}

		} else if (node->get_tile_origin() == TileMap::TILE_ORIGIN_CENTER) {
			rect.position += tile_ofs;

			if (p_flip_h) {
				rect.position.x -= cell_size.x / 2;
			} else {
				rect.position.x += cell_size.x / 2;
			}

			if (p_flip_v) {
				rect.position.y -= cell_size.y / 2;
			} else {
				rect.position.y += cell_size.y / 2;
			}
		}
	} else {
		rect.position += tile_ofs;
	}

	Color modulate = node->get_tileset()->tile_get_modulate(p_cell);
	modulate.a = 0.5;

	Transform2D old_transform = p_viewport->get_viewport_transform();
	p_viewport->draw_set_transform_matrix(p_xform); // Take into account TileMap transformation when displaying cell
	if (r.has_no_area()) {
		p_viewport->draw_texture_rect(t, rect, false, modulate, p_transpose);
	} else {
		p_viewport->draw_texture_rect_region(t, rect, r, modulate, p_transpose);
	}
	p_viewport->draw_set_transform_matrix(old_transform);
}

void TileMapEditor::_draw_fill_preview(Control *p_viewport, int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Point2i &p_autotile_coord, const Transform2D &p_xform) {
	PoolVector<Vector2> points = _bucket_fill(p_point, false, true);
	PoolVector<Vector2>::Read pr = points.read();
	int len = points.size();

	for (int i = 0; i < len; ++i) {
		_draw_cell(p_viewport, p_cell, pr[i], p_flip_h, p_flip_v, p_transpose, p_autotile_coord, p_xform);
	}
}

void TileMapEditor::_clear_bucket_cache() {
	if (bucket_cache_visited) {
		delete[] bucket_cache_visited;
		bucket_cache_visited = nullptr;
	}
}

void TileMapEditor::_update_copydata() {
	copydata.clear();

	if (!selection_active) {
		return;
	}

	for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
		for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {
			TileData tcd;

			tcd.cell = node->get_cell(j, i);
			if (tcd.cell != TileMap::INVALID_CELL) {
				tcd.pos = Point2i(j, i);
				tcd.flip_h = node->is_cell_x_flipped(j, i);
				tcd.flip_v = node->is_cell_y_flipped(j, i);
				tcd.transpose = node->is_cell_transposed(j, i);
				tcd.autotile_coord = node->get_cell_autotile_coord(j, i);

				copydata.push_back(tcd);
			}
		}
	}
}

static inline Vector<Point2i> line(int x0, int x1, int y0, int y1) {
	Vector<Point2i> points;

	float dx = ABS(x1 - x0);
	float dy = ABS(y1 - y0);

	int x = x0;
	int y = y0;

	int sx = x0 > x1 ? -1 : 1;
	int sy = y0 > y1 ? -1 : 1;

	if (dx > dy) {
		float err = dx / 2;

		for (; x != x1; x += sx) {
			points.push_back(Vector2(x, y));

			err -= dy;
			if (err < 0) {
				y += sy;
				err += dx;
			}
		}
	} else {
		float err = dy / 2;

		for (; y != y1; y += sy) {
			points.push_back(Vector2(x, y));

			err -= dx;
			if (err < 0) {
				x += sx;
				err += dy;
			}
		}
	}

	points.push_back(Vector2(x, y));

	return points;
}

bool TileMapEditor::forward_gui_input(const Ref<InputEvent> &p_event) {
	if (!node || !node->get_tileset().is_valid() || !node->is_visible_in_tree() || CanvasItemEditor::get_singleton()->get_current_tool() != CanvasItemEditor::TOOL_SELECT) {
		return false;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Transform2D xform_inv = xform.affine_inverse();

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed()) {
				if (Input::get_singleton()->is_key_pressed(KEY_SPACE)) {
					return false; // Drag.
				}

				if (tool == TOOL_NONE) {
					if (mb->get_shift()) {
						if (mb->get_command()) {
							tool = TOOL_RECTANGLE_PAINT;
						} else {
							tool = TOOL_LINE_PAINT;
						}

						selection_active = false;
						rectangle_begin = over_tile;

						_update_button_tool();
						return true;
					}

					if (mb->get_command()) {
						tool = TOOL_PICKING;
						_pick_tile(over_tile);
						_update_button_tool();

						return true;
					}

					tool = TOOL_PAINTING;
					_update_button_tool();
				}

				if (tool == TOOL_PAINTING) {
					Vector<int> ids = get_selected_tiles();

					if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
						tool = TOOL_PAINTING;

						_start_undo(TTR("Paint TileMap"));
					}
				} else if (tool == TOOL_PICKING) {
					_pick_tile(over_tile);
				} else if (tool == TOOL_SELECTING) {
					selection_active = true;
					rectangle_begin = over_tile;
				}

				_update_button_tool();
				return true;

			} else {
				// Mousebutton was released.
				if (tool != TOOL_NONE) {
					if (tool == TOOL_PAINTING) {
						Vector<int> ids = get_selected_tiles();

						if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
							_set_cell(over_tile, ids, flip_h, flip_v, transpose);
							_finish_undo();

							paint_undo.clear();
						}
					} else if (tool == TOOL_LINE_PAINT) {
						Vector<int> ids = get_selected_tiles();

						if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
							_start_undo(TTR("Line Draw"));
							for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {
								_set_cell(E->key(), ids, flip_h, flip_v, transpose);
							}
							_finish_undo();

							paint_undo.clear();

							CanvasItemEditor::get_singleton()->update_viewport();
						}
					} else if (tool == TOOL_RECTANGLE_PAINT) {
						Vector<int> ids = get_selected_tiles();

						if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
							_start_undo(TTR("Rectangle Paint"));
							for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
								for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {
									_set_cell(Point2i(j, i), ids, flip_h, flip_v, transpose);
								}
							}
							_finish_undo();

							CanvasItemEditor::get_singleton()->update_viewport();
						}
					} else if (tool == TOOL_PASTING) {
						Point2 ofs = over_tile - rectangle.position;
						Vector<int> ids;

						_start_undo(TTR("Paste"));
						ids.push_back(0);
						for (List<TileData>::Element *E = copydata.front(); E; E = E->next()) {
							ids.write[0] = E->get().cell;
							_set_cell(E->get().pos + ofs, ids, E->get().flip_h, E->get().flip_v, E->get().transpose, E->get().autotile_coord);
						}
						_finish_undo();

						CanvasItemEditor::get_singleton()->update_viewport();

						return true; // We want to keep the Pasting tool.
					} else if (tool == TOOL_SELECTING) {
						CanvasItemEditor::get_singleton()->update_viewport();

					} else if (tool == TOOL_BUCKET) {
						PoolVector<Vector2> points = _bucket_fill(over_tile);

						if (points.size() == 0) {
							return false;
						}

						_start_undo(TTR("Bucket Fill"));

						Dictionary op;
						op["id"] = get_selected_tiles();
						op["flip_h"] = flip_h;
						op["flip_v"] = flip_v;
						op["transpose"] = transpose;

						_fill_points(points, op);

						_finish_undo();

						// So the fill preview is cleared right after the click.
						CanvasItemEditor::get_singleton()->update_viewport();

						// We want to keep the bucket-tool active.
						return true;
					}

					tool = TOOL_NONE;
					_update_button_tool();

					return true;
				}
			}
		} else if (mb->get_button_index() == BUTTON_RIGHT) {
			if (mb->is_pressed()) {
				if (tool == TOOL_SELECTING || selection_active) {
					tool = TOOL_NONE;
					selection_active = false;

					CanvasItemEditor::get_singleton()->update_viewport();

					_update_button_tool();
					return true;
				}

				if (tool == TOOL_PASTING) {
					tool = TOOL_NONE;
					copydata.clear();

					CanvasItemEditor::get_singleton()->update_viewport();

					_update_button_tool();
					return true;
				}

				if (tool == TOOL_NONE) {
					paint_undo.clear();

					Point2 local = node->world_to_map(xform_inv.xform(mb->get_position()));

					_start_undo(TTR("Erase TileMap"));

					if (mb->get_shift()) {
						if (mb->get_command()) {
							tool = TOOL_RECTANGLE_ERASE;
						} else {
							tool = TOOL_LINE_ERASE;
						}

						selection_active = false;
						rectangle_begin = local;
					} else {
						tool = TOOL_ERASING;

						_set_cell(local, invalid_cell);
					}

					_update_button_tool();
					return true;
				}

			} else {
				if (tool == TOOL_ERASING || tool == TOOL_RECTANGLE_ERASE || tool == TOOL_LINE_ERASE) {
					_finish_undo();

					if (tool == TOOL_RECTANGLE_ERASE || tool == TOOL_LINE_ERASE) {
						CanvasItemEditor::get_singleton()->update_viewport();
					}

					tool = TOOL_NONE;

					_update_button_tool();
					return true;

				} else if (tool == TOOL_BUCKET) {
					Vector<int> ids;
					ids.push_back(node->get_cell(over_tile.x, over_tile.y));
					Dictionary pop;
					pop["id"] = ids;
					pop["flip_h"] = node->is_cell_x_flipped(over_tile.x, over_tile.y);
					pop["flip_v"] = node->is_cell_y_flipped(over_tile.x, over_tile.y);
					pop["transpose"] = node->is_cell_transposed(over_tile.x, over_tile.y);

					PoolVector<Vector2> points = _bucket_fill(over_tile, true);

					if (points.size() == 0) {
						return false;
					}

					undo_redo->create_action(TTR("Bucket Fill"));

					undo_redo->add_do_method(this, "_erase_points", points);
					undo_redo->add_undo_method(this, "_fill_points", points, pop);

					undo_redo->commit_action();
				}
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		Point2i new_over_tile = node->world_to_map(xform_inv.xform(mm->get_position()));
		Point2i old_over_tile = over_tile;

		if (new_over_tile != over_tile) {
			over_tile = new_over_tile;
			CanvasItemEditor::get_singleton()->update_viewport();
		}

		if (refocus_over_tile) {
			// editor lost focus; forget last tile position
			old_over_tile = new_over_tile;
			refocus_over_tile = false;
		}

		int tile_under = node->get_cell(over_tile.x, over_tile.y);
		String tile_name = "none";

		if (node->get_tileset()->has_tile(tile_under)) {
			tile_name = node->get_tileset()->tile_get_name(tile_under);
		}
		tile_info->show();
		tile_info->set_text(String::num(over_tile.x) + ", " + String::num(over_tile.y) + " [" + tile_name + "]");

		if (tool == TOOL_PAINTING) {
			// Paint using bresenham line to prevent holes in painting if the user moves fast.

			Vector<Point2i> points = line(old_over_tile.x, over_tile.x, old_over_tile.y, over_tile.y);
			Vector<int> ids = get_selected_tiles();

			for (int i = 0; i < points.size(); ++i) {
				Point2i pos = points[i];

				if (!paint_undo.has(pos)) {
					paint_undo[pos] = _get_op_from_cell(pos);
				}

				_set_cell(pos, ids, flip_h, flip_v, transpose);
			}

			return true;
		}

		if (tool == TOOL_ERASING) {
			// Erase using bresenham line to prevent holes in painting if the user moves fast.

			Vector<Point2i> points = line(old_over_tile.x, over_tile.x, old_over_tile.y, over_tile.y);

			for (int i = 0; i < points.size(); ++i) {
				Point2i pos = points[i];

				_set_cell(pos, invalid_cell);
			}

			return true;
		}

		if (tool == TOOL_SELECTING) {
			_select(rectangle_begin, over_tile);

			return true;
		}

		if (tool == TOOL_LINE_PAINT || tool == TOOL_LINE_ERASE) {
			Vector<int> ids = get_selected_tiles();
			Vector<int> tmp_cell;
			bool erasing = (tool == TOOL_LINE_ERASE);

			tmp_cell.push_back(0);
			if (erasing && paint_undo.size()) {
				for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {
					tmp_cell.write[0] = E->get().idx;
					_set_cell(E->key(), tmp_cell, E->get().xf, E->get().yf, E->get().tr);
				}
			}

			paint_undo.clear();

			if (ids.size() > 0 && ids[0] != TileMap::INVALID_CELL) {
				Vector<Point2i> points = line(rectangle_begin.x, over_tile.x, rectangle_begin.y, over_tile.y);

				for (int i = 0; i < points.size(); i++) {
					paint_undo[points[i]] = _get_op_from_cell(points[i]);

					if (erasing) {
						_set_cell(points[i], invalid_cell);
					}
				}

				CanvasItemEditor::get_singleton()->update_viewport();
			}

			return true;
		}
		if (tool == TOOL_RECTANGLE_PAINT || tool == TOOL_RECTANGLE_ERASE) {
			Vector<int> tmp_cell;
			tmp_cell.push_back(0);

			_select(rectangle_begin, over_tile);

			if (tool == TOOL_RECTANGLE_ERASE) {
				if (paint_undo.size()) {
					for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {
						tmp_cell.write[0] = E->get().idx;
						_set_cell(E->key(), tmp_cell, E->get().xf, E->get().yf, E->get().tr);
					}
				}

				paint_undo.clear();

				for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
					for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {
						Point2i tile = Point2i(j, i);
						paint_undo[tile] = _get_op_from_cell(tile);

						_set_cell(tile, invalid_cell);
					}
				}
			}

			return true;
		}
		if (tool == TOOL_PICKING && Input::get_singleton()->is_mouse_button_pressed(BUTTON_LEFT)) {
			_pick_tile(over_tile);

			return true;
		}
	}

	Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed()) {
		if (last_tool == TOOL_NONE && tool == TOOL_PICKING && k->get_scancode() == KEY_SHIFT && k->get_command()) {
			// trying to draw a rectangle with the painting tool, so change to the correct tool
			tool = last_tool;

			CanvasItemEditor::get_singleton()->update_viewport();
			_update_button_tool();
		}

		if (k->get_scancode() == KEY_ESCAPE) {
			if (tool == TOOL_PASTING) {
				copydata.clear();
			} else if (tool == TOOL_SELECTING || selection_active) {
				selection_active = false;
			}

			tool = TOOL_NONE;

			CanvasItemEditor::get_singleton()->update_viewport();

			_update_button_tool();
			return true;
		}

		if (!mouse_over) {
			// Editor shortcuts should not fire if mouse not in viewport.
			return false;
		}

		if (ED_IS_SHORTCUT("tile_map_editor/paint_tile", p_event)) {
			// NOTE: We do not set tool = TOOL_PAINTING as this begins painting
			// immediately without pressing the left mouse button first.
			tool = TOOL_NONE;
			CanvasItemEditor::get_singleton()->update_viewport();

			_update_button_tool();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/bucket_fill", p_event)) {
			tool = TOOL_BUCKET;
			CanvasItemEditor::get_singleton()->update_viewport();

			_update_button_tool();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/erase_selection", p_event)) {
			_menu_option(OPTION_ERASE_SELECTION);

			_update_button_tool();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/select", p_event)) {
			tool = TOOL_SELECTING;
			selection_active = false;

			CanvasItemEditor::get_singleton()->update_viewport();

			_update_button_tool();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/copy_selection", p_event)) {
			_update_copydata();

			if (selection_active) {
				tool = TOOL_PASTING;

				CanvasItemEditor::get_singleton()->update_viewport();

				_update_button_tool();
				return true;
			}
		}
		if (ED_IS_SHORTCUT("tile_map_editor/cut_selection", p_event)) {
			if (selection_active) {
				_update_copydata();

				_start_undo(TTR("Cut Selection"));
				_erase_selection();
				_finish_undo();

				selection_active = false;

				tool = TOOL_PASTING;

				CanvasItemEditor::get_singleton()->update_viewport();
				_update_button_tool();
				return true;
			}
		}
		if (ED_IS_SHORTCUT("tile_map_editor/find_tile", p_event)) {
			search_box->select_all();
			search_box->grab_focus();

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/rotate_left", p_event)) {
			_rotate(-1);
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/rotate_right", p_event)) {
			_rotate(1);
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/flip_horizontal", p_event)) {
			_flip_horizontal();
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/flip_vertical", p_event)) {
			_flip_vertical();
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/clear_transform", p_event)) {
			_clear_transform();
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/transpose", p_event)) {
			transpose = !transpose;
			_update_palette();
			CanvasItemEditor::get_singleton()->update_viewport();
			return true;
		}
	} else if (k.is_valid()) { // Release event.

		if (tool == TOOL_NONE) {
			if (k->get_scancode() == KEY_SHIFT && k->get_command()) {
				tool = TOOL_PICKING;
				_update_button_tool();
			}
		} else if (tool == TOOL_PICKING) {
#ifdef APPLE_STYLE_KEYS
			if (k->get_scancode() == KEY_META) {
#else
			if (k->get_scancode() == KEY_CONTROL) {
#endif
				// Go back to that last tool if KEY_CONTROL was released.
				tool = last_tool;

				CanvasItemEditor::get_singleton()->update_viewport();
				_update_button_tool();
			}
		}
	}
	return false;
}

void TileMapEditor::forward_canvas_draw_over_viewport(Control *p_overlay) {
	if (!node || CanvasItemEditor::get_singleton()->get_current_tool() != CanvasItemEditor::TOOL_SELECT) {
		return;
	}

	Transform2D cell_xf = node->get_cell_transform();
	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Transform2D xform_inv = xform.affine_inverse();

	Size2 screen_size = p_overlay->get_size();
	{
		Rect2 aabb;
		aabb.position = node->world_to_map(xform_inv.xform(Vector2()));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(0, screen_size.height))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(screen_size.width, 0))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(screen_size)));
		Rect2i si = aabb.grow(1.0);

		if (node->get_half_offset() != TileMap::HALF_OFFSET_X && node->get_half_offset() != TileMap::HALF_OFFSET_NEGATIVE_X) {
			int max_lines = 2000; //avoid crash if size too small

			for (int i = (si.position.x) - 1; i <= (si.position.x + si.size.x); i++) {
				Vector2 from = xform.xform(node->map_to_world(Vector2(i, si.position.y)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(i, si.position.y + si.size.y + 1)));

				Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
				p_overlay->draw_line(from, to, col, 1);
				if (max_lines-- == 0) {
					break;
				}
			}
		} else {
			int max_lines = 10000; //avoid crash if size too small

			for (int i = (si.position.x) - 1; i <= (si.position.x + si.size.x); i++) {
				for (int j = (si.position.y) - 1; j <= (si.position.y + si.size.y); j++) {
					Vector2 ofs;
					if (ABS(j) & 1) {
						ofs = cell_xf[0] * (node->get_half_offset() == TileMap::HALF_OFFSET_X ? 0.5 : -0.5);
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(i, j), true) + ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(i, j + 1), true) + ofs);

					Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
					p_overlay->draw_line(from, to, col, 1);

					if (--max_lines == 0) {
						break;
					}
				}
				if (max_lines == 0) {
					break;
				}
			}
		}

		int max_lines = 10000; //avoid crash if size too small

		if (node->get_half_offset() != TileMap::HALF_OFFSET_Y && node->get_half_offset() != TileMap::HALF_OFFSET_NEGATIVE_Y) {
			for (int i = (si.position.y) - 1; i <= (si.position.y + si.size.y); i++) {
				Vector2 from = xform.xform(node->map_to_world(Vector2(si.position.x, i)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(si.position.x + si.size.x + 1, i)));

				Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
				p_overlay->draw_line(from, to, col, 1);

				if (max_lines-- == 0) {
					break;
				}
			}
		} else {
			for (int i = (si.position.y) - 1; i <= (si.position.y + si.size.y); i++) {
				for (int j = (si.position.x) - 1; j <= (si.position.x + si.size.x); j++) {
					Vector2 ofs;
					if (ABS(j) & 1) {
						ofs = cell_xf[1] * (node->get_half_offset() == TileMap::HALF_OFFSET_Y ? 0.5 : -0.5);
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(j, i), true) + ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(j + 1, i), true) + ofs);

					Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
					p_overlay->draw_line(from, to, col, 1);

					if (--max_lines == 0) {
						break;
					}
				}
				if (max_lines == 0) {
					break;
				}
			}
		}
	}

	if (selection_active) {
		Vector<Vector2> points;
		points.push_back(xform.xform(node->map_to_world((rectangle.position))));
		points.push_back(xform.xform(node->map_to_world((rectangle.position + Point2(rectangle.size.x + 1, 0)))));
		points.push_back(xform.xform(node->map_to_world((rectangle.position + Point2(rectangle.size.x + 1, rectangle.size.y + 1)))));
		points.push_back(xform.xform(node->map_to_world((rectangle.position + Point2(0, rectangle.size.y + 1)))));

		p_overlay->draw_colored_polygon(points, Color(0.2, 0.8, 1, 0.4));
	}

	if (mouse_over && node->get_tileset().is_valid()) {
		Vector2 endpoints[4] = {
			node->map_to_world(over_tile, true),
			node->map_to_world((over_tile + Point2(1, 0)), true),
			node->map_to_world((over_tile + Point2(1, 1)), true),
			node->map_to_world((over_tile + Point2(0, 1)), true)
		};

		for (int i = 0; i < 4; i++) {
			if (node->get_half_offset() == TileMap::HALF_OFFSET_X && ABS(over_tile.y) & 1) {
				endpoints[i] += cell_xf[0] * 0.5;
			}
			if (node->get_half_offset() == TileMap::HALF_OFFSET_NEGATIVE_X && ABS(over_tile.y) & 1) {
				endpoints[i] += cell_xf[0] * -0.5;
			}
			if (node->get_half_offset() == TileMap::HALF_OFFSET_Y && ABS(over_tile.x) & 1) {
				endpoints[i] += cell_xf[1] * 0.5;
			}
			if (node->get_half_offset() == TileMap::HALF_OFFSET_NEGATIVE_Y && ABS(over_tile.x) & 1) {
				endpoints[i] += cell_xf[1] * -0.5;
			}
			endpoints[i] = xform.xform(endpoints[i]);
		}
		Color col;
		if (node->get_cell(over_tile.x, over_tile.y) != TileMap::INVALID_CELL) {
			col = Color(0.2, 0.8, 1.0, 0.8);
		} else {
			col = Color(1.0, 0.4, 0.2, 0.8);
		}

		for (int i = 0; i < 4; i++) {
			p_overlay->draw_line(endpoints[i], endpoints[(i + 1) % 4], col, 2);
		}

		bool bucket_preview = EditorSettings::get_singleton()->get("editors/tile_map/bucket_fill_preview");
		if (tool == TOOL_SELECTING || tool == TOOL_PICKING || !bucket_preview) {
			return;
		}

		if (tool == TOOL_LINE_PAINT) {
			if (paint_undo.empty()) {
				return;
			}

			Vector<int> ids = get_selected_tiles();

			if (ids.size() == 1 && ids[0] == TileMap::INVALID_CELL) {
				return;
			}

			for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {
				_draw_cell(p_overlay, ids[0], E->key(), flip_h, flip_v, transpose, autotile_coord, xform);
			}

		} else if (tool == TOOL_RECTANGLE_PAINT) {
			Vector<int> ids = get_selected_tiles();

			if (ids.size() == 1 && ids[0] == TileMap::INVALID_CELL) {
				return;
			}

			for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
				for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {
					_draw_cell(p_overlay, ids[0], Point2i(j, i), flip_h, flip_v, transpose, autotile_coord, xform);
				}
			}
		} else if (tool == TOOL_PASTING) {
			if (copydata.empty()) {
				return;
			}

			Ref<TileSet> ts = node->get_tileset();

			if (ts.is_null()) {
				return;
			}

			Point2 ofs = over_tile - rectangle.position;

			for (List<TileData>::Element *E = copydata.front(); E; E = E->next()) {
				if (!ts->has_tile(E->get().cell)) {
					continue;
				}

				TileData tcd = E->get();

				_draw_cell(p_overlay, tcd.cell, tcd.pos + ofs, tcd.flip_h, tcd.flip_v, tcd.transpose, tcd.autotile_coord, xform);
			}

			Rect2i duplicate = rectangle;
			duplicate.position = over_tile;

			Vector<Vector2> points;
			points.push_back(xform.xform(node->map_to_world(duplicate.position)));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(duplicate.size.x + 1, 0)))));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(duplicate.size.x + 1, duplicate.size.y + 1)))));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(0, duplicate.size.y + 1)))));

			p_overlay->draw_colored_polygon(points, Color(0.2, 1.0, 0.8, 0.2));

		} else if (tool == TOOL_BUCKET) {
			Vector<int> tiles = get_selected_tiles();
			_draw_fill_preview(p_overlay, tiles[0], over_tile, flip_h, flip_v, transpose, autotile_coord, xform);

		} else {
			Vector<int> st = get_selected_tiles();

			if (st.size() == 1 && st[0] == TileMap::INVALID_CELL) {
				return;
			}

			_draw_cell(p_overlay, st[0], over_tile, flip_h, flip_v, transpose, autotile_coord, xform);
		}
	}
}

void TileMapEditor::edit(Node *p_tile_map) {
	search_box->set_text("");

	if (!canvas_item_editor_viewport) {
		canvas_item_editor_viewport = CanvasItemEditor::get_singleton()->get_viewport_control();
	}

	if (node && node->is_connected("settings_changed", this, "_tileset_settings_changed")) {
		node->disconnect("settings_changed", this, "_tileset_settings_changed");
	}

	if (p_tile_map) {
		node = Object::cast_to<TileMap>(p_tile_map);
		if (!canvas_item_editor_viewport->is_connected("mouse_entered", this, "_canvas_mouse_enter")) {
			canvas_item_editor_viewport->connect("mouse_entered", this, "_canvas_mouse_enter");
		}
		if (!canvas_item_editor_viewport->is_connected("mouse_exited", this, "_canvas_mouse_exit")) {
			canvas_item_editor_viewport->connect("mouse_exited", this, "_canvas_mouse_exit");
		}

		_update_palette();

	} else {
		node = nullptr;

		if (canvas_item_editor_viewport->is_connected("mouse_entered", this, "_canvas_mouse_enter")) {
			canvas_item_editor_viewport->disconnect("mouse_entered", this, "_canvas_mouse_enter");
		}
		if (canvas_item_editor_viewport->is_connected("mouse_exited", this, "_canvas_mouse_exit")) {
			canvas_item_editor_viewport->disconnect("mouse_exited", this, "_canvas_mouse_exit");
		}

		_update_palette();
	}

	if (node && !node->is_connected("settings_changed", this, "_tileset_settings_changed")) {
		node->connect("settings_changed", this, "_tileset_settings_changed");
	}

	_clear_bucket_cache();
}

void TileMapEditor::_tileset_settings_changed() {
	_update_palette();
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TileMapEditor::_icon_size_changed(float p_value) {
	if (node) {
		palette->set_icon_scale(p_value);
		manual_palette->set_icon_scale(p_value);
		_update_palette();
	}
}

void TileMapEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_manual_toggled"), &TileMapEditor::_manual_toggled);
	ClassDB::bind_method(D_METHOD("_priority_toggled"), &TileMapEditor::_priority_toggled);
	ClassDB::bind_method(D_METHOD("_text_entered"), &TileMapEditor::_text_entered);
	ClassDB::bind_method(D_METHOD("_text_changed"), &TileMapEditor::_text_changed);
	ClassDB::bind_method(D_METHOD("_sbox_input"), &TileMapEditor::_sbox_input);
	ClassDB::bind_method(D_METHOD("_button_tool_select"), &TileMapEditor::_button_tool_select);
	ClassDB::bind_method(D_METHOD("_menu_option"), &TileMapEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_canvas_mouse_enter"), &TileMapEditor::_canvas_mouse_enter);
	ClassDB::bind_method(D_METHOD("_canvas_mouse_exit"), &TileMapEditor::_canvas_mouse_exit);
	ClassDB::bind_method(D_METHOD("_tileset_settings_changed"), &TileMapEditor::_tileset_settings_changed);
	ClassDB::bind_method(D_METHOD("_rotate"), &TileMapEditor::_rotate);
	ClassDB::bind_method(D_METHOD("_flip_horizontal"), &TileMapEditor::_flip_horizontal);
	ClassDB::bind_method(D_METHOD("_flip_vertical"), &TileMapEditor::_flip_vertical);
	ClassDB::bind_method(D_METHOD("_clear_transform"), &TileMapEditor::_clear_transform);
	ClassDB::bind_method(D_METHOD("_palette_selected"), &TileMapEditor::_palette_selected);
	ClassDB::bind_method(D_METHOD("_palette_multi_selected"), &TileMapEditor::_palette_multi_selected);
	ClassDB::bind_method(D_METHOD("_palette_input"), &TileMapEditor::_palette_input);

	ClassDB::bind_method(D_METHOD("_fill_points"), &TileMapEditor::_fill_points);
	ClassDB::bind_method(D_METHOD("_erase_points"), &TileMapEditor::_erase_points);

	ClassDB::bind_method(D_METHOD("_icon_size_changed"), &TileMapEditor::_icon_size_changed);
	ClassDB::bind_method(D_METHOD("_node_removed"), &TileMapEditor::_node_removed);
}

TileMapEditor::CellOp TileMapEditor::_get_op_from_cell(const Point2i &p_pos) {
	CellOp op;
	op.idx = node->get_cell(p_pos.x, p_pos.y);
	if (op.idx != TileMap::INVALID_CELL) {
		if (node->is_cell_x_flipped(p_pos.x, p_pos.y)) {
			op.xf = true;
		}
		if (node->is_cell_y_flipped(p_pos.x, p_pos.y)) {
			op.yf = true;
		}
		if (node->is_cell_transposed(p_pos.x, p_pos.y)) {
			op.tr = true;
		}
		op.ac = node->get_cell_autotile_coord(p_pos.x, p_pos.y);
	}
	return op;
}

void TileMapEditor::_rotate(int steps) {
	const bool normal_rotation_matrix[][3] = {
		{ false, false, false },
		{ true, true, false },
		{ false, true, true },
		{ true, false, true }
	};

	const bool mirrored_rotation_matrix[][3] = {
		{ false, true, false },
		{ true, true, true },
		{ false, false, true },
		{ true, false, false }
	};

	if (transpose ^ flip_h ^ flip_v) {
		// Odd number of flags activated = mirrored rotation
		for (int i = 0; i < 4; i++) {
			if (transpose == mirrored_rotation_matrix[i][0] &&
					flip_h == mirrored_rotation_matrix[i][1] &&
					flip_v == mirrored_rotation_matrix[i][2]) {
				int new_id = Math::wrapi(i + steps, 0, 4);
				transpose = mirrored_rotation_matrix[new_id][0];
				flip_h = mirrored_rotation_matrix[new_id][1];
				flip_v = mirrored_rotation_matrix[new_id][2];
				break;
			}
		}
	} else {
		// Even number of flags activated = normal rotation
		for (int i = 0; i < 4; i++) {
			if (transpose == normal_rotation_matrix[i][0] &&
					flip_h == normal_rotation_matrix[i][1] &&
					flip_v == normal_rotation_matrix[i][2]) {
				int new_id = Math::wrapi(i + steps, 0, 4);
				transpose = normal_rotation_matrix[new_id][0];
				flip_h = normal_rotation_matrix[new_id][1];
				flip_v = normal_rotation_matrix[new_id][2];
				break;
			}
		}
	}

	_update_palette();
}

void TileMapEditor::_flip_horizontal() {
	flip_h = !flip_h;
	_update_palette();
}

void TileMapEditor::_flip_vertical() {
	flip_v = !flip_v;
	_update_palette();
}

void TileMapEditor::_clear_transform() {
	transpose = false;
	flip_h = false;
	flip_v = false;
	_update_palette();
}

TileMapEditor::TileMapEditor(EditorNode *p_editor) {
	node = nullptr;
	manual_autotile = false;
	priority_atlastile = false;
	manual_position = Vector2(0, 0);
	canvas_item_editor_viewport = nullptr;
	editor = p_editor;
	undo_redo = EditorNode::get_undo_redo();

	tool = TOOL_NONE;
	selection_active = false;
	mouse_over = false;

	flip_h = false;
	flip_v = false;
	transpose = false;

	bucket_cache_tile = -1;
	bucket_cache_visited = nullptr;

	invalid_cell.resize(1);
	invalid_cell.write[0] = TileMap::INVALID_CELL;

	ED_SHORTCUT("tile_map_editor/erase_selection", TTR("Erase Selection"), KEY_DELETE);
	ED_SHORTCUT("tile_map_editor/find_tile", TTR("Find Tile"), KEY_MASK_CMD + KEY_F);
	ED_SHORTCUT("tile_map_editor/transpose", TTR("Transpose"), KEY_T);

	HBoxContainer *tool_hb = memnew(HBoxContainer);
	add_child(tool_hb);

	manual_button = memnew(CheckBox);
	manual_button->set_text(TTR("Disable Autotile"));
	manual_button->connect("toggled", this, "_manual_toggled");
	add_child(manual_button);

	priority_button = memnew(CheckBox);
	priority_button->set_text(TTR("Enable Priority"));
	priority_button->connect("toggled", this, "_priority_toggled");
	add_child(priority_button);

	search_box = memnew(LineEdit);
	search_box->set_placeholder(TTR("Filter tiles"));
	search_box->set_h_size_flags(SIZE_EXPAND_FILL);
	search_box->connect("text_entered", this, "_text_entered");
	search_box->connect("text_changed", this, "_text_changed");
	search_box->connect("gui_input", this, "_sbox_input");
	add_child(search_box);

	size_slider = memnew(HSlider);
	size_slider->set_h_size_flags(SIZE_EXPAND_FILL);
	size_slider->set_min(0.1f);
	size_slider->set_max(4.0f);
	size_slider->set_step(0.1f);
	size_slider->set_value(1.0f);
	size_slider->connect("value_changed", this, "_icon_size_changed");
	add_child(size_slider);

	int mw = EDITOR_DEF("editors/tile_map/palette_min_width", 80);

	VSplitContainer *palette_container = memnew(VSplitContainer);
	palette_container->set_v_size_flags(SIZE_EXPAND_FILL);
	palette_container->set_custom_minimum_size(Size2(mw, 0));
	add_child(palette_container);

	// Add tile palette.
	palette = memnew(ItemList);
	palette->set_h_size_flags(SIZE_EXPAND_FILL);
	palette->set_v_size_flags(SIZE_EXPAND_FILL);
	palette->set_max_columns(0);
	palette->set_icon_mode(ItemList::ICON_MODE_TOP);
	palette->set_max_text_lines(2);
	palette->set_select_mode(ItemList::SELECT_MULTI);
	palette->add_constant_override("vseparation", 8 * EDSCALE);
	palette->connect("item_selected", this, "_palette_selected");
	palette->connect("multi_selected", this, "_palette_multi_selected");
	palette->connect("gui_input", this, "_palette_input");
	palette_container->add_child(palette);

	// Add message for when no texture is selected.
	info_message = memnew(Label);
	info_message->set_text(TTR("Give a TileSet resource to this TileMap to use its tiles."));
	info_message->set_valign(Label::VALIGN_CENTER);
	info_message->set_align(Label::ALIGN_CENTER);
	info_message->set_autowrap(true);
	info_message->set_custom_minimum_size(Size2(100 * EDSCALE, 0));
	info_message->set_anchors_and_margins_preset(PRESET_WIDE, PRESET_MODE_KEEP_SIZE, 8 * EDSCALE);
	palette->add_child(info_message);

	// Add autotile override palette.
	manual_palette = memnew(ItemList);
	manual_palette->set_h_size_flags(SIZE_EXPAND_FILL);
	manual_palette->set_v_size_flags(SIZE_EXPAND_FILL);
	manual_palette->set_max_columns(0);
	manual_palette->set_icon_mode(ItemList::ICON_MODE_TOP);
	manual_palette->set_max_text_lines(2);
	manual_palette->hide();
	palette_container->add_child(manual_palette);

	// Add menu items.
	toolbar = memnew(HBoxContainer);
	toolbar->hide();
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(toolbar);

	toolbar->add_child(memnew(VSeparator));

	// Tools.
	paint_button = memnew(ToolButton);
	paint_button->set_shortcut(ED_SHORTCUT("tile_map_editor/paint_tile", TTR("Paint Tile"), KEY_P));
#ifdef OSX_ENABLED
	paint_button->set_tooltip(TTR("Shift+LMB: Line Draw\nShift+Command+LMB: Rectangle Paint"));
#else
	paint_button->set_tooltip(TTR("Shift+LMB: Line Draw\nShift+Ctrl+LMB: Rectangle Paint"));
#endif
	paint_button->connect("pressed", this, "_button_tool_select", make_binds(TOOL_NONE));
	paint_button->set_toggle_mode(true);
	toolbar->add_child(paint_button);

	bucket_fill_button = memnew(ToolButton);
	bucket_fill_button->set_shortcut(ED_SHORTCUT("tile_map_editor/bucket_fill", TTR("Bucket Fill"), KEY_B));
	bucket_fill_button->connect("pressed", this, "_button_tool_select", make_binds(TOOL_BUCKET));
	bucket_fill_button->set_toggle_mode(true);
	toolbar->add_child(bucket_fill_button);

	picker_button = memnew(ToolButton);
	picker_button->set_shortcut(ED_SHORTCUT("tile_map_editor/pick_tile", TTR("Pick Tile"), KEY_I));
	picker_button->connect("pressed", this, "_button_tool_select", make_binds(TOOL_PICKING));
	picker_button->set_toggle_mode(true);
	toolbar->add_child(picker_button);

	select_button = memnew(ToolButton);
	select_button->set_shortcut(ED_SHORTCUT("tile_map_editor/select", TTR("Select"), KEY_M));
	select_button->connect("pressed", this, "_button_tool_select", make_binds(TOOL_SELECTING));
	select_button->set_toggle_mode(true);
	toolbar->add_child(select_button);

	_update_button_tool();

	// Container to the right of the toolbar.
	toolbar_right = memnew(HBoxContainer);
	toolbar_right->hide();
	toolbar_right->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbar_right->set_alignment(BoxContainer::ALIGN_END);
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(toolbar_right);

	// Tile position.
	tile_info = memnew(Label);
	tile_info->set_modulate(Color(1, 1, 1, 0.8));
	tile_info->set_mouse_filter(MOUSE_FILTER_IGNORE);
	tile_info->add_font_override("font", EditorNode::get_singleton()->get_gui_base()->get_font("main", "EditorFonts"));
	// The tile info is only displayed after a tile has been hovered.
	tile_info->hide();
	CanvasItemEditor::get_singleton()->add_control_to_info_overlay(tile_info);

	// Menu.
	options = memnew(MenuButton);
	options->set_text("TileMap");
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("TileMap", "EditorIcons"));
	options->set_process_unhandled_key_input(false);
	toolbar_right->add_child(options);

	PopupMenu *p = options->get_popup();
	p->add_shortcut(ED_SHORTCUT("tile_map_editor/cut_selection", TTR("Cut Selection"), KEY_MASK_CMD + KEY_X), OPTION_CUT);
	p->add_shortcut(ED_SHORTCUT("tile_map_editor/copy_selection", TTR("Copy Selection"), KEY_MASK_CMD + KEY_C), OPTION_COPY);
	p->add_shortcut(ED_GET_SHORTCUT("tile_map_editor/erase_selection"), OPTION_ERASE_SELECTION);
	p->add_separator();
	p->add_item(TTR("Fix Invalid Tiles"), OPTION_FIX_INVALID);
	p->connect("id_pressed", this, "_menu_option");

	rotate_left_button = memnew(ToolButton);
	rotate_left_button->set_tooltip(TTR("Rotate Left"));
	rotate_left_button->set_focus_mode(FOCUS_NONE);
	rotate_left_button->connect("pressed", this, "_rotate", varray(-1));
	rotate_left_button->set_shortcut(ED_SHORTCUT("tile_map_editor/rotate_left", TTR("Rotate Left"), KEY_A));
	tool_hb->add_child(rotate_left_button);

	rotate_right_button = memnew(ToolButton);
	rotate_right_button->set_tooltip(TTR("Rotate Right"));
	rotate_right_button->set_focus_mode(FOCUS_NONE);
	rotate_right_button->connect("pressed", this, "_rotate", varray(1));
	rotate_right_button->set_shortcut(ED_SHORTCUT("tile_map_editor/rotate_right", TTR("Rotate Right"), KEY_S));
	tool_hb->add_child(rotate_right_button);

	flip_horizontal_button = memnew(ToolButton);
	flip_horizontal_button->set_tooltip(TTR("Flip Horizontally"));
	flip_horizontal_button->set_focus_mode(FOCUS_NONE);
	flip_horizontal_button->connect("pressed", this, "_flip_horizontal");
	flip_horizontal_button->set_shortcut(ED_SHORTCUT("tile_map_editor/flip_horizontal", TTR("Flip Horizontally"), KEY_X));
	tool_hb->add_child(flip_horizontal_button);

	flip_vertical_button = memnew(ToolButton);
	flip_vertical_button->set_tooltip(TTR("Flip Vertically"));
	flip_vertical_button->set_focus_mode(FOCUS_NONE);
	flip_vertical_button->connect("pressed", this, "_flip_vertical");
	flip_vertical_button->set_shortcut(ED_SHORTCUT("tile_map_editor/flip_vertical", TTR("Flip Vertically"), KEY_Z));
	tool_hb->add_child(flip_vertical_button);

	clear_transform_button = memnew(ToolButton);
	clear_transform_button->set_tooltip(TTR("Clear Transform"));
	clear_transform_button->set_focus_mode(FOCUS_NONE);
	clear_transform_button->connect("pressed", this, "_clear_transform");
	clear_transform_button->set_shortcut(ED_SHORTCUT("tile_map_editor/clear_transform", TTR("Clear Transform"), KEY_W));
	tool_hb->add_child(clear_transform_button);

	clear_transform_button->set_disabled(true);
}

TileMapEditor::~TileMapEditor() {
	_clear_bucket_cache();
	copydata.clear();
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

void TileMapEditorPlugin::_notification(int p_what) {
	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		switch ((int)EditorSettings::get_singleton()->get("editors/tile_map/editor_side")) {
			case 0: { // Left.
				CanvasItemEditor::get_singleton()->get_palette_split()->move_child(tile_map_editor, 0);
			} break;
			case 1: { // Right.
				CanvasItemEditor::get_singleton()->get_palette_split()->move_child(tile_map_editor, 1);
			} break;
		}
	}
}

void TileMapEditorPlugin::edit(Object *p_object) {
	tile_map_editor->edit(Object::cast_to<Node>(p_object));
}

bool TileMapEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("TileMap");
}

void TileMapEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		tile_map_editor->show();
		tile_map_editor->get_toolbar()->show();
		tile_map_editor->get_toolbar_right()->show();
		// `tile_info` isn't shown here, as it's displayed after a tile has been hovered.
		// Otherwise, a translucent black rectangle would be visible as there would be an
		// empty Label in the CanvasItemEditor's info overlay.

		// Change to TOOL_SELECT when TileMap node is selected, to prevent accidental movement.
		CanvasItemEditor::get_singleton()->set_current_tool(CanvasItemEditor::TOOL_SELECT);
	} else {
		tile_map_editor->hide();
		tile_map_editor->get_toolbar()->hide();
		tile_map_editor->get_toolbar_right()->hide();
		tile_map_editor->get_tile_info()->hide();
		tile_map_editor->edit(nullptr);
	}
}

TileMapEditorPlugin::TileMapEditorPlugin(EditorNode *p_node) {
	EDITOR_DEF("editors/tile_map/preview_size", 64);
	EDITOR_DEF("editors/tile_map/palette_item_hseparation", 8);
	EDITOR_DEF("editors/tile_map/show_tile_names", true);
	EDITOR_DEF("editors/tile_map/show_tile_ids", false);
	EDITOR_DEF("editors/tile_map/sort_tiles_by_name", true);
	EDITOR_DEF("editors/tile_map/bucket_fill_preview", true);
	EDITOR_DEF("editors/tile_map/editor_side", 1);
	EditorSettings::get_singleton()->add_property_hint(PropertyInfo(Variant::INT, "editors/tile_map/editor_side", PROPERTY_HINT_ENUM, "Left,Right"));

	tile_map_editor = memnew(TileMapEditor(p_node));
	switch ((int)EditorSettings::get_singleton()->get("editors/tile_map/editor_side")) {
		case 0: { // Left.
			add_control_to_container(CONTAINER_CANVAS_EDITOR_SIDE_LEFT, tile_map_editor);
		} break;
		case 1: { // Right.
			add_control_to_container(CONTAINER_CANVAS_EDITOR_SIDE_RIGHT, tile_map_editor);
		} break;
	}
	tile_map_editor->hide();
}

TileMapEditorPlugin::~TileMapEditorPlugin() {
}
