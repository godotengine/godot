/*************************************************************************/
/*  tile_map_editor_plugin.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "os/input.h"
#include "os/keyboard.h"

void TileMapEditor::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_PROCESS: {

			if (bucket_queue.size() && canvas_item_editor) {
				canvas_item_editor->update();
			}

		} break;

		case NOTIFICATION_ENTER_TREE: {

			transp->set_icon(get_icon("Transpose", "EditorIcons"));
			mirror_x->set_icon(get_icon("MirrorX", "EditorIcons"));
			mirror_y->set_icon(get_icon("MirrorY", "EditorIcons"));
			rotate_0->set_icon(get_icon("Rotate0", "EditorIcons"));
			rotate_90->set_icon(get_icon("Rotate90", "EditorIcons"));
			rotate_180->set_icon(get_icon("Rotate180", "EditorIcons"));
			rotate_270->set_icon(get_icon("Rotate270", "EditorIcons"));

			search_box->add_icon_override("right_icon", get_icon("Search", "EditorIcons"));

		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {

			bool new_show_tile_info = EditorSettings::get_singleton()->get("editors/tile_map/show_tile_info_on_hover");
			if (new_show_tile_info != show_tile_info) {
				show_tile_info = new_show_tile_info;
				tile_info->set_visible(show_tile_info);
			}

			if (is_visible_in_tree()) {
				_update_palette();
			}
		} break;
	}
}

void TileMapEditor::_menu_option(int p_option) {

	switch (p_option) {

		case OPTION_PAINTING: {
			// NOTE: We do not set tool = TOOL_PAINTING as this begins painting
			// immediately without pressing the left mouse button first
			tool = TOOL_NONE;

			canvas_item_editor->update();

		} break;
		case OPTION_BUCKET: {

			tool = TOOL_BUCKET;

			canvas_item_editor->update();
		} break;
		case OPTION_PICK_TILE: {

			tool = TOOL_PICKING;

			canvas_item_editor->update();
		} break;
		case OPTION_SELECT: {

			tool = TOOL_SELECTING;
			selection_active = false;

			canvas_item_editor->update();
		} break;
		case OPTION_DUPLICATE: {

			_update_copydata();

			if (selection_active) {
				tool = TOOL_DUPLICATING;

				canvas_item_editor->update();
			}
		} break;
		case OPTION_ERASE_SELECTION: {

			if (!selection_active)
				return;

			undo_redo->create_action(TTR("Erase Selection"));
			for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
				for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {

					_set_cell(Point2i(j, i), TileMap::INVALID_CELL, false, false, false, true);
				}
			}
			undo_redo->commit_action();

			selection_active = false;
			copydata.clear();

			canvas_item_editor->update();
		} break;
	}
}

void TileMapEditor::_canvas_mouse_enter() {

	mouse_over = true;
	canvas_item_editor->update();
}

void TileMapEditor::_canvas_mouse_exit() {

	mouse_over = false;
	canvas_item_editor->update();
}

int TileMapEditor::get_selected_tile() const {

	int item = palette->get_current();

	if (item == -1)
		return TileMap::INVALID_CELL;

	return palette->get_item_metadata(item);
}

void TileMapEditor::set_selected_tile(int p_tile) {

	int idx = palette->find_metadata(p_tile);

	if (idx >= 0) {
		palette->select(idx, true);
		palette->ensure_current_is_visible();
	}
}

void TileMapEditor::_set_cell(const Point2i &p_pos, int p_value, bool p_flip_h, bool p_flip_v, bool p_transpose, bool p_with_undo) {

	ERR_FAIL_COND(!node);

	int prev_val = node->get_cell(p_pos.x, p_pos.y);

	bool prev_flip_h = node->is_cell_x_flipped(p_pos.x, p_pos.y);
	bool prev_flip_v = node->is_cell_y_flipped(p_pos.x, p_pos.y);
	bool prev_transpose = node->is_cell_transposed(p_pos.x, p_pos.y);

	if (p_value == prev_val && p_flip_h == prev_flip_h && p_flip_v == prev_flip_v && p_transpose == prev_transpose)
		return; //check that it's actually different

	if (p_with_undo) {

		undo_redo->add_do_method(node, "set_cellv", Point2(p_pos), p_value, p_flip_h, p_flip_v, p_transpose);
		undo_redo->add_undo_method(node, "set_cellv", Point2(p_pos), prev_val, prev_flip_h, prev_flip_v, prev_transpose);
	} else {

		node->set_cell(p_pos.x, p_pos.y, p_value, p_flip_h, p_flip_v, p_transpose);
	}
}

void TileMapEditor::_text_entered(const String &p_text) {

	canvas_item_editor->grab_focus();
}

void TileMapEditor::_text_changed(const String &p_text) {

	_update_palette();
}

void TileMapEditor::_sbox_input(const Ref<InputEvent> &p_ie) {

	Ref<InputEventKey> k = p_ie;

	if (k.is_valid() && (k->get_scancode() == KEY_UP ||
								k->get_scancode() == KEY_DOWN ||
								k->get_scancode() == KEY_PAGEUP ||
								k->get_scancode() == KEY_PAGEDOWN)) {

		palette->call("_gui_input", k);
		search_box->accept_event();
	}
}

// Implementation detail of TileMapEditor::_update_palette();
// in modern C++ this could have been inside its body
namespace {
struct _PaletteEntry {
	int id;
	String name;

	bool operator<(const _PaletteEntry &p_rhs) const {
		return name < p_rhs.name;
	}
};
} // namespace

void TileMapEditor::_update_palette() {

	if (!node)
		return;

	int selected = get_selected_tile();
	palette->clear();

	Ref<TileSet> tileset = node->get_tileset();
	if (tileset.is_null())
		return;

	List<int> tiles;
	tileset->get_tile_list(&tiles);

	if (tiles.empty())
		return;

	float min_size = EDITOR_DEF("editors/tile_map/preview_size", 64);
	min_size *= EDSCALE;
	int hseparation = EDITOR_DEF("editors/tile_map/palette_item_hseparation", 8);
	bool show_tile_names = bool(EDITOR_DEF("editors/tile_map/show_tile_names", true));
	bool show_tile_ids = bool(EDITOR_DEF("editors/tile_map/show_tile_ids", false));
	bool sort_by_name = bool(EDITOR_DEF("editors/tile_map/sort_tiles_by_name", true));

	palette->add_constant_override("hseparation", hseparation * EDSCALE);
	palette->add_constant_override("vseparation", 8 * EDSCALE);

	palette->set_fixed_icon_size(Size2(min_size, min_size));
	palette->set_fixed_column_width(min_size * MAX(size_slider->get_value(), 1));

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

		if (filter != "" && !filter.is_subsequence_ofi(name))
			continue;

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

			if (!region.has_no_area())
				palette->set_item_icon_region(palette->get_item_count() - 1, region);

			palette->set_item_icon(palette->get_item_count() - 1, tex);
		}

		palette->set_item_metadata(palette->get_item_count() - 1, entries[i].id);
	}

	palette->set_same_column_width(true);

	if (selected != -1)
		set_selected_tile(selected);
	else
		palette->select(0);
}

void TileMapEditor::_pick_tile(const Point2 &p_pos) {

	int id = node->get_cell(p_pos.x, p_pos.y);

	if (id == TileMap::INVALID_CELL)
		return;

	if (search_box->get_text().strip_edges() != "") {

		search_box->set_text("");
		_update_palette();
	}

	set_selected_tile(id);

	mirror_x->set_pressed(node->is_cell_x_flipped(p_pos.x, p_pos.y));
	mirror_y->set_pressed(node->is_cell_y_flipped(p_pos.x, p_pos.y));
	transp->set_pressed(node->is_cell_transposed(p_pos.x, p_pos.y));

	_update_transform_buttons();
	canvas_item_editor->update();
}

PoolVector<Vector2> TileMapEditor::_bucket_fill(const Point2i &p_start, bool erase, bool preview) {

	int prev_id = node->get_cell(p_start.x, p_start.y);
	int id = TileMap::INVALID_CELL;
	if (!erase) {
		id = get_selected_tile();

		if (id == TileMap::INVALID_CELL)
			return PoolVector<Vector2>();
	} else if (prev_id == TileMap::INVALID_CELL) {
		return PoolVector<Vector2>();
	}

	if (id == prev_id) {
		return PoolVector<Vector2>();
	}

	Rect2i r = node->get_item_rect();
	r.position = r.position / node->get_cell_size();
	r.size = r.size / node->get_cell_size();

	int area = r.get_area();
	if (preview) {
		// Test if we can re-use the result from preview bucket fill
		bool invalidate_cache = false;
		// Area changed
		if (r != bucket_cache_rect)
			_clear_bucket_cache();
		// Cache grid is not initialized
		if (bucket_cache_visited == 0) {
			bucket_cache_visited = new bool[area];
			invalidate_cache = true;
		}
		// Tile ID changed or position wasn't visited by the previous fill
		int loc = (p_start.x - r.position.x) + (p_start.y - r.position.y) * r.get_size().x;
		if (prev_id != bucket_cache_tile || !bucket_cache_visited[loc]) {
			invalidate_cache = true;
		}
		if (invalidate_cache) {
			for (int i = 0; i < area; ++i)
				bucket_cache_visited[i] = false;
			bucket_cache = PoolVector<Vector2>();
			bucket_cache_tile = prev_id;
			bucket_cache_rect = r;
			bucket_queue.clear();
		}
	}

	PoolVector<Vector2> points;
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

		if (!r.has_point(n))
			continue;

		if (node->get_cell(n.x, n.y) == prev_id) {

			if (preview) {
				int loc = (n.x - r.position.x) + (n.y - r.position.y) * r.get_size().x;
				if (bucket_cache_visited[loc])
					continue;
				bucket_cache_visited[loc] = true;
				bucket_cache.push_back(n);
			} else {
				node->set_cellv(n, id, flip_h, flip_v, transpose);
				points.push_back(n);
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

void TileMapEditor::_fill_points(const PoolVector<Vector2> p_points, const Dictionary &p_op) {

	int len = p_points.size();
	PoolVector<Vector2>::Read pr = p_points.read();

	int id = p_op["id"];
	bool xf = p_op["flip_h"];
	bool yf = p_op["flip_v"];
	bool tr = p_op["transpose"];

	for (int i = 0; i < len; i++) {

		_set_cell(pr[i], id, xf, yf, tr);
	}
}

void TileMapEditor::_erase_points(const PoolVector<Vector2> p_points) {

	int len = p_points.size();
	PoolVector<Vector2>::Read pr = p_points.read();

	for (int i = 0; i < len; i++) {

		_set_cell(pr[i], TileMap::INVALID_CELL);
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

	canvas_item_editor->update();
}

void TileMapEditor::_draw_cell(int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Transform2D &p_xform) {

	Ref<Texture> t = node->get_tileset()->tile_get_texture(p_cell);

	if (t.is_null())
		return;

	Vector2 tile_ofs = node->get_tileset()->tile_get_texture_offset(p_cell);

	Rect2 r = node->get_tileset()->tile_get_region(p_cell);
	Size2 sc = p_xform.get_scale();

	Rect2 rect = Rect2();
	rect.position = node->map_to_world(p_point) + node->get_cell_draw_offset();

	if (r.has_no_area()) {
		rect.size = t->get_size();
	} else {
		rect.size = r.size;
	}

	if (rect.size.y > rect.size.x) {
		if ((p_flip_h && (p_flip_v || p_transpose)) || (p_flip_v && !p_transpose))
			tile_ofs.y += rect.size.y - rect.size.x;
	} else if (rect.size.y < rect.size.x) {
		if ((p_flip_v && (p_flip_h || p_transpose)) || (p_flip_h && !p_transpose))
			tile_ofs.x += rect.size.x - rect.size.y;
	}

	if (p_transpose) {
		SWAP(tile_ofs.x, tile_ofs.y);
	}
	if (p_flip_h) {
		sc.x *= -1.0;
		tile_ofs.x *= -1.0;
	}
	if (p_flip_v) {
		sc.y *= -1.0;
		tile_ofs.y *= -1.0;
	}

	if (node->get_tile_origin() == TileMap::TILE_ORIGIN_TOP_LEFT) {

		rect.position += tile_ofs;
	} else if (node->get_tile_origin() == TileMap::TILE_ORIGIN_BOTTOM_LEFT) {
		Size2 cell_size = node->get_cell_size();

		rect.position += tile_ofs;

		if (p_transpose) {
			if (p_flip_h)
				rect.position.x -= cell_size.x;
			else
				rect.position.x += cell_size.x;
		} else {
			if (p_flip_v)
				rect.position.y -= cell_size.y;
			else
				rect.position.y += cell_size.y;
		}

	} else if (node->get_tile_origin() == TileMap::TILE_ORIGIN_CENTER) {
		rect.position += node->get_cell_size() / 2;
		Vector2 s = r.size;

		Vector2 center = (s / 2) - tile_ofs;

		if (p_flip_h)
			rect.position.x -= s.x - center.x;
		else
			rect.position.x -= center.x;

		if (p_flip_v)
			rect.position.y -= s.y - center.y;
		else
			rect.position.y -= center.y;
	}

	rect.position = p_xform.xform(rect.position);
	rect.size *= sc;

	if (r.has_no_area())
		canvas_item_editor->draw_texture_rect(t, rect, false, Color(1, 1, 1, 0.5), p_transpose);
	else
		canvas_item_editor->draw_texture_rect_region(t, rect, r, Color(1, 1, 1, 0.5), p_transpose);
}

void TileMapEditor::_draw_fill_preview(int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Transform2D &p_xform) {

	PoolVector<Vector2> points = _bucket_fill(p_point, false, true);
	PoolVector<Vector2>::Read pr = points.read();
	int len = points.size();

	for (int i = 0; i < len; ++i) {
		_draw_cell(p_cell, pr[i], p_flip_h, p_flip_v, p_transpose, p_xform);
	}
}

void TileMapEditor::_clear_bucket_cache() {
	if (bucket_cache_visited) {
		delete[] bucket_cache_visited;
		bucket_cache_visited = 0;
	}
}

void TileMapEditor::_update_copydata() {

	copydata.clear();

	if (!selection_active)
		return;

	for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {

		for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {

			TileData tcd;

			tcd.cell = node->get_cell(j, i);

			if (tcd.cell != TileMap::INVALID_CELL) {
				tcd.pos = Point2i(j, i);
				tcd.flip_h = node->is_cell_x_flipped(j, i);
				tcd.flip_v = node->is_cell_y_flipped(j, i);
				tcd.transpose = node->is_cell_transposed(j, i);
			}

			copydata.push_back(tcd);
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

	if (!node || !node->get_tileset().is_valid() || !node->is_visible_in_tree())
		return false;

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Transform2D xform_inv = xform.affine_inverse();

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_LEFT) {

			if (mb->is_pressed()) {

				if (Input::get_singleton()->is_key_pressed(KEY_SPACE))
					return false; //drag

				if (tool == TOOL_NONE) {

					if (mb->get_shift()) {

						if (mb->get_control())
							tool = TOOL_RECTANGLE_PAINT;
						else
							tool = TOOL_LINE_PAINT;

						selection_active = false;
						rectangle_begin = over_tile;

						return true;
					}

					if (mb->get_control()) {

						tool = TOOL_PICKING;
						_pick_tile(over_tile);

						return true;
					}

					tool = TOOL_PAINTING;
				}

				if (tool == TOOL_PAINTING) {

					int id = get_selected_tile();

					if (id != TileMap::INVALID_CELL) {

						tool = TOOL_PAINTING;

						paint_undo.clear();
						paint_undo[over_tile] = _get_op_from_cell(over_tile);

						_set_cell(over_tile, id, flip_h, flip_v, transpose);
					}
				} else if (tool == TOOL_PICKING) {

					_pick_tile(over_tile);
				} else if (tool == TOOL_SELECTING) {

					selection_active = true;
					rectangle_begin = over_tile;
				}

				return true;

			} else {
				// Mousebutton was released
				if (tool != TOOL_NONE) {

					if (tool == TOOL_PAINTING) {

						int id = get_selected_tile();

						if (id != TileMap::INVALID_CELL && paint_undo.size()) {

							undo_redo->create_action(TTR("Paint TileMap"));
							for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

								Point2 p = E->key();
								undo_redo->add_do_method(node, "set_cellv", p, id, flip_h, flip_v, transpose);
								undo_redo->add_undo_method(node, "set_cellv", p, E->get().idx, E->get().xf, E->get().yf, E->get().tr);
							}
							undo_redo->commit_action();

							paint_undo.clear();
						}
					} else if (tool == TOOL_LINE_PAINT) {

						int id = get_selected_tile();

						if (id != TileMap::INVALID_CELL) {

							undo_redo->create_action(TTR("Line Draw"));
							for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

								_set_cell(E->key(), id, flip_h, flip_v, transpose, true);
							}
							undo_redo->commit_action();

							paint_undo.clear();

							canvas_item_editor->update();
						}
					} else if (tool == TOOL_RECTANGLE_PAINT) {

						int id = get_selected_tile();

						if (id != TileMap::INVALID_CELL) {

							undo_redo->create_action(TTR("Rectangle Paint"));
							for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
								for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {

									_set_cell(Point2i(j, i), id, flip_h, flip_v, transpose, true);
								}
							}
							undo_redo->commit_action();

							canvas_item_editor->update();
						}
					} else if (tool == TOOL_DUPLICATING) {

						Point2 ofs = over_tile - rectangle.position;

						undo_redo->create_action(TTR("Duplicate"));
						for (List<TileData>::Element *E = copydata.front(); E; E = E->next()) {

							_set_cell(E->get().pos + ofs, E->get().cell, E->get().flip_h, E->get().flip_v, E->get().transpose, true);
						}
						undo_redo->commit_action();

						copydata.clear();

						canvas_item_editor->update();

					} else if (tool == TOOL_SELECTING) {

						canvas_item_editor->update();

					} else if (tool == TOOL_BUCKET) {

						Dictionary pop;
						pop["id"] = node->get_cell(over_tile.x, over_tile.y);
						pop["flip_h"] = node->is_cell_x_flipped(over_tile.x, over_tile.y);
						pop["flip_v"] = node->is_cell_y_flipped(over_tile.x, over_tile.y);
						pop["transpose"] = node->is_cell_transposed(over_tile.x, over_tile.y);

						PoolVector<Vector2> points = _bucket_fill(over_tile);

						if (points.size() == 0)
							return false;

						Dictionary op;
						op["id"] = get_selected_tile();
						op["flip_h"] = flip_h;
						op["flip_v"] = flip_v;
						op["transpose"] = transpose;

						undo_redo->create_action(TTR("Bucket Fill"));

						undo_redo->add_do_method(this, "_fill_points", points, op);
						undo_redo->add_undo_method(this, "_fill_points", points, pop);

						undo_redo->commit_action();

						// We want to keep the bucket-tool active
						return true;
					}

					tool = TOOL_NONE;

					return true;
				}
			}
		} else if (mb->get_button_index() == BUTTON_RIGHT) {

			if (mb->is_pressed()) {

				if (tool == TOOL_SELECTING || selection_active) {

					tool = TOOL_NONE;
					selection_active = false;

					canvas_item_editor->update();

					return true;
				}

				if (tool == TOOL_DUPLICATING) {

					tool = TOOL_NONE;
					copydata.clear();

					canvas_item_editor->update();

					return true;
				}

				if (tool == TOOL_NONE) {

					paint_undo.clear();

					Point2 local = node->world_to_map(xform_inv.xform(mb->get_position()));

					if (mb->get_shift()) {

						if (mb->get_control())
							tool = TOOL_RECTANGLE_ERASE;
						else
							tool = TOOL_LINE_ERASE;

						selection_active = false;
						rectangle_begin = local;
					} else {

						tool = TOOL_ERASING;

						paint_undo[local] = _get_op_from_cell(local);
						_set_cell(local, TileMap::INVALID_CELL);
					}

					return true;
				}

			} else {
				if (tool == TOOL_ERASING || tool == TOOL_RECTANGLE_ERASE || tool == TOOL_LINE_ERASE) {

					if (paint_undo.size()) {
						undo_redo->create_action(TTR("Erase TileMap"));
						for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

							Point2 p = E->key();
							undo_redo->add_do_method(node, "set_cellv", p, TileMap::INVALID_CELL, false, false, false);
							undo_redo->add_undo_method(node, "set_cellv", p, E->get().idx, E->get().xf, E->get().yf, E->get().tr);
						}

						undo_redo->commit_action();
						paint_undo.clear();
					}

					if (tool == TOOL_RECTANGLE_ERASE || tool == TOOL_LINE_ERASE) {
						canvas_item_editor->update();
					}

					tool = TOOL_NONE;

					return true;

				} else if (tool == TOOL_BUCKET) {

					Dictionary pop;
					pop["id"] = node->get_cell(over_tile.x, over_tile.y);
					pop["flip_h"] = node->is_cell_x_flipped(over_tile.x, over_tile.y);
					pop["flip_v"] = node->is_cell_y_flipped(over_tile.x, over_tile.y);
					pop["transpose"] = node->is_cell_transposed(over_tile.x, over_tile.y);

					PoolVector<Vector2> points = _bucket_fill(over_tile, true);

					if (points.size() == 0)
						return false;

					undo_redo->create_action("Bucket Fill");

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
			canvas_item_editor->update();
		}

		if (show_tile_info) {
			int tile_under = node->get_cell(over_tile.x, over_tile.y);
			String tile_name = "none";

			if (node->get_tileset()->has_tile(tile_under))
				tile_name = node->get_tileset()->tile_get_name(tile_under);
			tile_info->set_text(String::num(over_tile.x) + ", " + String::num(over_tile.y) + " [" + tile_name + "]");
		}

		if (tool == TOOL_PAINTING) {

			// Paint using bresenham line to prevent holes in painting if the user moves fast

			Vector<Point2i> points = line(old_over_tile.x, over_tile.x, old_over_tile.y, over_tile.y);
			int id = get_selected_tile();

			for (int i = 0; i < points.size(); ++i) {

				Point2i pos = points[i];

				if (!paint_undo.has(over_tile)) {
					paint_undo[pos] = _get_op_from_cell(pos);
				}

				_set_cell(pos, id, flip_h, flip_v, transpose);
			}

			return true;
		}

		if (tool == TOOL_ERASING) {

			// erase using bresenham line to prevent holes in painting if the user moves fast

			Vector<Point2i> points = line(old_over_tile.x, over_tile.x, old_over_tile.y, over_tile.y);

			for (int i = 0; i < points.size(); ++i) {

				Point2i pos = points[i];

				if (!paint_undo.has(over_tile)) {
					paint_undo[pos] = _get_op_from_cell(pos);
				}

				_set_cell(pos, TileMap::INVALID_CELL);
			}

			return true;
		}

		if (tool == TOOL_SELECTING) {

			_select(rectangle_begin, over_tile);

			return true;
		}

		if (tool == TOOL_LINE_PAINT || tool == TOOL_LINE_ERASE) {

			int id = get_selected_tile();
			bool erasing = (tool == TOOL_LINE_ERASE);

			if (erasing && paint_undo.size()) {

				for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

					_set_cell(E->key(), E->get().idx, E->get().xf, E->get().yf, E->get().tr);
				}
			}

			paint_undo.clear();

			if (id != TileMap::INVALID_CELL) {

				Vector<Point2i> points = line(rectangle_begin.x, over_tile.x, rectangle_begin.y, over_tile.y);

				for (int i = 0; i < points.size(); i++) {

					paint_undo[points[i]] = _get_op_from_cell(points[i]);

					if (erasing)
						_set_cell(points[i], TileMap::INVALID_CELL);
				}

				canvas_item_editor->update();
			}

			return true;
		}
		if (tool == TOOL_RECTANGLE_PAINT || tool == TOOL_RECTANGLE_ERASE) {

			_select(rectangle_begin, over_tile);

			if (tool == TOOL_RECTANGLE_ERASE) {

				if (paint_undo.size()) {

					for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

						_set_cell(E->key(), E->get().idx, E->get().xf, E->get().yf, E->get().tr);
					}
				}

				paint_undo.clear();

				for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
					for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {

						Point2i tile = Point2i(j, i);
						paint_undo[tile] = _get_op_from_cell(tile);

						_set_cell(tile, TileMap::INVALID_CELL);
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

		if (k->get_scancode() == KEY_ESCAPE) {

			if (tool == TOOL_DUPLICATING)
				copydata.clear();
			else if (tool == TOOL_SELECTING || selection_active)
				selection_active = false;

			tool = TOOL_NONE;

			canvas_item_editor->update();

			return true;
		}

		if (!mouse_over) {
			// Editor shortcuts should not fire if mouse not in viewport
			return false;
		}

		if (ED_IS_SHORTCUT("tile_map_editor/paint_tile", p_event)) {
			// NOTE: We do not set tool = TOOL_PAINTING as this begins painting
			// immediately without pressing the left mouse button first
			tool = TOOL_NONE;
			canvas_item_editor->update();

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/bucket_fill", p_event)) {
			tool = TOOL_BUCKET;
			canvas_item_editor->update();

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/erase_selection", p_event)) {
			_menu_option(OPTION_ERASE_SELECTION);

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/select", p_event)) {
			tool = TOOL_SELECTING;
			selection_active = false;

			canvas_item_editor->update();

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/duplicate_selection", p_event)) {
			_update_copydata();

			if (selection_active) {
				tool = TOOL_DUPLICATING;

				canvas_item_editor->update();

				return true;
			}
		}
		if (ED_IS_SHORTCUT("tile_map_editor/find_tile", p_event)) {
			search_box->select_all();
			search_box->grab_focus();

			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/mirror_x", p_event)) {
			flip_h = !flip_h;
			mirror_x->set_pressed(flip_h);
			canvas_item_editor->update();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/mirror_y", p_event)) {
			flip_v = !flip_v;
			mirror_y->set_pressed(flip_v);
			canvas_item_editor->update();
			return true;
		}
		if (ED_IS_SHORTCUT("tile_map_editor/transpose", p_event)) {
			transpose = !transpose;
			transp->set_pressed(transpose);
			canvas_item_editor->update();
			return true;
		}
	}

	return false;
}

void TileMapEditor::forward_draw_over_canvas(Control *p_canvas) {

	if (!node)
		return;

	Transform2D cell_xf = node->get_cell_transform();

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * node->get_global_transform();
	Transform2D xform_inv = xform.affine_inverse();

	Size2 screen_size = canvas_item_editor->get_size();
	{
		Rect2 aabb;
		aabb.position = node->world_to_map(xform_inv.xform(Vector2()));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(0, screen_size.height))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(Vector2(screen_size.width, 0))));
		aabb.expand_to(node->world_to_map(xform_inv.xform(screen_size)));
		Rect2i si = aabb.grow(1.0);

		if (node->get_half_offset() != TileMap::HALF_OFFSET_X) {

			int max_lines = 2000; //avoid crash if size too smal

			for (int i = (si.position.x) - 1; i <= (si.position.x + si.size.x); i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(i, si.position.y)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(i, si.position.y + si.size.y + 1)));

				Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
				canvas_item_editor->draw_line(from, to, col, 1);
				if (max_lines-- == 0)
					break;
			}
		} else {

			int max_lines = 10000; //avoid crash if size too smal

			for (int i = (si.position.x) - 1; i <= (si.position.x + si.size.x); i++) {

				for (int j = (si.position.y) - 1; j <= (si.position.y + si.size.y); j++) {

					Vector2 ofs;
					if (ABS(j) & 1) {
						ofs = cell_xf[0] * 0.5;
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(i, j), true) + ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(i, j + 1), true) + ofs);
					Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
					canvas_item_editor->draw_line(from, to, col, 1);

					if (max_lines-- == 0)
						break;
				}
			}
		}

		int max_lines = 10000; //avoid crash if size too smal

		if (node->get_half_offset() != TileMap::HALF_OFFSET_Y) {

			for (int i = (si.position.y) - 1; i <= (si.position.y + si.size.y); i++) {

				Vector2 from = xform.xform(node->map_to_world(Vector2(si.position.x, i)));
				Vector2 to = xform.xform(node->map_to_world(Vector2(si.position.x + si.size.x + 1, i)));

				Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
				canvas_item_editor->draw_line(from, to, col, 1);

				if (max_lines-- == 0)
					break;
			}
		} else {

			for (int i = (si.position.y) - 1; i <= (si.position.y + si.size.y); i++) {

				for (int j = (si.position.x) - 1; j <= (si.position.x + si.size.x); j++) {

					Vector2 ofs;
					if (ABS(j) & 1) {
						ofs = cell_xf[1] * 0.5;
					}

					Vector2 from = xform.xform(node->map_to_world(Vector2(j, i), true) + ofs);
					Vector2 to = xform.xform(node->map_to_world(Vector2(j + 1, i), true) + ofs);
					Color col = i == 0 ? Color(1, 0.8, 0.2, 0.5) : Color(1, 0.3, 0.1, 0.2);
					canvas_item_editor->draw_line(from, to, col, 1);

					if (max_lines-- == 0)
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

		canvas_item_editor->draw_colored_polygon(points, Color(0.2, 0.8, 1, 0.4));
	}

	if (mouse_over) {

		Vector2 endpoints[4] = {
			node->map_to_world(over_tile, true),
			node->map_to_world((over_tile + Point2(1, 0)), true),
			node->map_to_world((over_tile + Point2(1, 1)), true),
			node->map_to_world((over_tile + Point2(0, 1)), true)
		};

		for (int i = 0; i < 4; i++) {
			if (node->get_half_offset() == TileMap::HALF_OFFSET_X && ABS(over_tile.y) & 1)
				endpoints[i] += cell_xf[0] * 0.5;
			if (node->get_half_offset() == TileMap::HALF_OFFSET_Y && ABS(over_tile.x) & 1)
				endpoints[i] += cell_xf[1] * 0.5;
			endpoints[i] = xform.xform(endpoints[i]);
		}
		Color col;
		if (node->get_cell(over_tile.x, over_tile.y) != TileMap::INVALID_CELL)
			col = Color(0.2, 0.8, 1.0, 0.8);
		else
			col = Color(1.0, 0.4, 0.2, 0.8);

		for (int i = 0; i < 4; i++)
			canvas_item_editor->draw_line(endpoints[i], endpoints[(i + 1) % 4], col, 2);

		bool bucket_preview = EditorSettings::get_singleton()->get("editors/tile_map/bucket_fill_preview");
		if (tool == TOOL_SELECTING || tool == TOOL_PICKING || !bucket_preview) {
			return;
		}

		if (tool == TOOL_LINE_PAINT) {

			if (paint_undo.empty())
				return;

			int id = get_selected_tile();

			if (id == TileMap::INVALID_CELL)
				return;

			for (Map<Point2i, CellOp>::Element *E = paint_undo.front(); E; E = E->next()) {

				_draw_cell(id, E->key(), flip_h, flip_v, transpose, xform);
			}

		} else if (tool == TOOL_RECTANGLE_PAINT) {

			int id = get_selected_tile();

			if (id == TileMap::INVALID_CELL)
				return;

			for (int i = rectangle.position.y; i <= rectangle.position.y + rectangle.size.y; i++) {
				for (int j = rectangle.position.x; j <= rectangle.position.x + rectangle.size.x; j++) {

					_draw_cell(id, Point2i(j, i), flip_h, flip_v, transpose, xform);
				}
			}
		} else if (tool == TOOL_DUPLICATING) {

			if (copydata.empty())
				return;

			Ref<TileSet> ts = node->get_tileset();

			if (ts.is_null())
				return;

			Point2 ofs = over_tile - rectangle.position;

			for (List<TileData>::Element *E = copydata.front(); E; E = E->next()) {

				if (!ts->has_tile(E->get().cell))
					continue;

				TileData tcd = E->get();

				_draw_cell(tcd.cell, tcd.pos + ofs, tcd.flip_h, tcd.flip_v, tcd.transpose, xform);
			}

			Rect2i duplicate = rectangle;
			duplicate.position = over_tile;

			Vector<Vector2> points;
			points.push_back(xform.xform(node->map_to_world(duplicate.position)));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(duplicate.size.x + 1, 0)))));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(duplicate.size.x + 1, duplicate.size.y + 1)))));
			points.push_back(xform.xform(node->map_to_world((duplicate.position + Point2(0, duplicate.size.y + 1)))));

			canvas_item_editor->draw_colored_polygon(points, Color(0.2, 1.0, 0.8, 0.2));

		} else if (tool == TOOL_BUCKET) {

			int tile = get_selected_tile();
			_draw_fill_preview(tile, over_tile, flip_h, flip_v, transpose, xform);

		} else {

			int st = get_selected_tile();

			if (st == TileMap::INVALID_CELL)
				return;

			_draw_cell(st, over_tile, flip_h, flip_v, transpose, xform);
		}
	}
}

void TileMapEditor::edit(Node *p_tile_map) {

	search_box->set_text("");

	if (!canvas_item_editor) {
		canvas_item_editor = CanvasItemEditor::get_singleton()->get_viewport_control();
	}

	if (node)
		node->disconnect("settings_changed", this, "_tileset_settings_changed");
	if (p_tile_map) {

		node = Object::cast_to<TileMap>(p_tile_map);
		if (!canvas_item_editor->is_connected("mouse_entered", this, "_canvas_mouse_enter"))
			canvas_item_editor->connect("mouse_entered", this, "_canvas_mouse_enter");
		if (!canvas_item_editor->is_connected("mouse_exited", this, "_canvas_mouse_exit"))
			canvas_item_editor->connect("mouse_exited", this, "_canvas_mouse_exit");

		_update_palette();

	} else {
		node = NULL;

		if (canvas_item_editor->is_connected("mouse_entered", this, "_canvas_mouse_enter"))
			canvas_item_editor->disconnect("mouse_entered", this, "_canvas_mouse_enter");
		if (canvas_item_editor->is_connected("mouse_exited", this, "_canvas_mouse_exit"))
			canvas_item_editor->disconnect("mouse_exited", this, "_canvas_mouse_exit");

		_update_palette();
	}

	if (node)
		node->connect("settings_changed", this, "_tileset_settings_changed");

	_clear_bucket_cache();
}

void TileMapEditor::_tileset_settings_changed() {

	_update_palette();

	if (canvas_item_editor)
		canvas_item_editor->update();
}

void TileMapEditor::_icon_size_changed(float p_value) {
	if (node) {
		palette->set_icon_scale(p_value);
		_update_palette();
	}
}

void TileMapEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_text_entered"), &TileMapEditor::_text_entered);
	ClassDB::bind_method(D_METHOD("_text_changed"), &TileMapEditor::_text_changed);
	ClassDB::bind_method(D_METHOD("_sbox_input"), &TileMapEditor::_sbox_input);
	ClassDB::bind_method(D_METHOD("_menu_option"), &TileMapEditor::_menu_option);
	ClassDB::bind_method(D_METHOD("_canvas_mouse_enter"), &TileMapEditor::_canvas_mouse_enter);
	ClassDB::bind_method(D_METHOD("_canvas_mouse_exit"), &TileMapEditor::_canvas_mouse_exit);
	ClassDB::bind_method(D_METHOD("_tileset_settings_changed"), &TileMapEditor::_tileset_settings_changed);
	ClassDB::bind_method(D_METHOD("_update_transform_buttons"), &TileMapEditor::_update_transform_buttons);

	ClassDB::bind_method(D_METHOD("_fill_points"), &TileMapEditor::_fill_points);
	ClassDB::bind_method(D_METHOD("_erase_points"), &TileMapEditor::_erase_points);

	ClassDB::bind_method(D_METHOD("_icon_size_changed"), &TileMapEditor::_icon_size_changed);
}

TileMapEditor::CellOp TileMapEditor::_get_op_from_cell(const Point2i &p_pos) {
	CellOp op;
	op.idx = node->get_cell(p_pos.x, p_pos.y);
	if (op.idx != TileMap::INVALID_CELL) {
		if (node->is_cell_x_flipped(p_pos.x, p_pos.y))
			op.xf = true;
		if (node->is_cell_y_flipped(p_pos.x, p_pos.y))
			op.yf = true;
		if (node->is_cell_transposed(p_pos.x, p_pos.y))
			op.tr = true;
	}
	return op;
}

void TileMapEditor::_update_transform_buttons(Object *p_button) {
	//ERR_FAIL_NULL(p_button);
	ToolButton *b = Object::cast_to<ToolButton>(p_button);
	//ERR_FAIL_COND(!b);

	if (b == rotate_0) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(false);
		transp->set_pressed(false);
	} else if (b == rotate_90) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(false);
		transp->set_pressed(true);
	} else if (b == rotate_180) {
		mirror_x->set_pressed(true);
		mirror_y->set_pressed(true);
		transp->set_pressed(false);
	} else if (b == rotate_270) {
		mirror_x->set_pressed(false);
		mirror_y->set_pressed(true);
		transp->set_pressed(true);
	}

	flip_h = mirror_x->is_pressed();
	flip_v = mirror_y->is_pressed();
	transpose = transp->is_pressed();

	rotate_0->set_pressed(!flip_h && !flip_v && !transpose);
	rotate_90->set_pressed(flip_h && !flip_v && transpose);
	rotate_180->set_pressed(flip_h && flip_v && !transpose);
	rotate_270->set_pressed(!flip_h && flip_v && transpose);
}

TileMapEditor::TileMapEditor(EditorNode *p_editor) {

	node = NULL;
	canvas_item_editor = NULL;
	editor = p_editor;
	undo_redo = editor->get_undo_redo();

	tool = TOOL_NONE;
	selection_active = false;
	mouse_over = false;
	show_tile_info = true;

	flip_h = false;
	flip_v = false;
	transpose = false;

	bucket_cache_tile = -1;
	bucket_cache_visited = 0;

	ED_SHORTCUT("tile_map_editor/erase_selection", TTR("Erase selection"), KEY_DELETE);
	ED_SHORTCUT("tile_map_editor/find_tile", TTR("Find tile"), KEY_MASK_CMD + KEY_F);
	ED_SHORTCUT("tile_map_editor/transpose", TTR("Transpose"), KEY_T);
	ED_SHORTCUT("tile_map_editor/mirror_x", TTR("Mirror X"), KEY_A);
	ED_SHORTCUT("tile_map_editor/mirror_y", TTR("Mirror Y"), KEY_S);

	HBoxContainer *tool_hb1 = memnew(HBoxContainer);
	add_child(tool_hb1);
	HBoxContainer *tool_hb2 = memnew(HBoxContainer);
	add_child(tool_hb2);

	search_box = memnew(LineEdit);
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

	// Add tile palette
	palette = memnew(ItemList);
	palette->set_v_size_flags(SIZE_EXPAND_FILL);
	palette->set_custom_minimum_size(Size2(mw, 0));
	palette->set_max_columns(0);
	palette->set_icon_mode(ItemList::ICON_MODE_TOP);
	palette->set_max_text_lines(2);
	add_child(palette);

	// Add menu items
	toolbar = memnew(HBoxContainer);
	toolbar->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbar->set_alignment(BoxContainer::ALIGN_END);
	toolbar->hide();
	CanvasItemEditor::get_singleton()->add_control_to_menu_panel(toolbar);

	// Tile position
	tile_info = memnew(Label);
	toolbar->add_child(tile_info);

	options = memnew(MenuButton);
	options->set_text("Tile Map");
	options->set_icon(EditorNode::get_singleton()->get_gui_base()->get_icon("TileMap", "EditorIcons"));
	options->set_process_unhandled_key_input(false);

	PopupMenu *p = options->get_popup();

	p->add_shortcut(ED_SHORTCUT("tile_map_editor/paint_tile", TTR("Paint Tile"), KEY_P), OPTION_PAINTING);
	p->add_shortcut(ED_SHORTCUT("tile_map_editor/bucket_fill", TTR("Bucket Fill"), KEY_G), OPTION_BUCKET);
	p->add_separator();
	p->add_item(TTR("Pick Tile"), OPTION_PICK_TILE, KEY_CONTROL);
	p->add_separator();
	p->add_shortcut(ED_SHORTCUT("tile_map_editor/select", TTR("Select"), KEY_MASK_CMD + KEY_B), OPTION_SELECT);
	p->add_shortcut(ED_SHORTCUT("tile_map_editor/duplicate_selection", TTR("Duplicate Selection"), KEY_MASK_CMD + KEY_D), OPTION_DUPLICATE);
	p->add_shortcut(ED_GET_SHORTCUT("tile_map_editor/erase_selection"), OPTION_ERASE_SELECTION);

	p->connect("id_pressed", this, "_menu_option");

	toolbar->add_child(options);

	transp = memnew(ToolButton);
	transp->set_toggle_mode(true);
	transp->set_tooltip(TTR("Transpose") + " (" + ED_GET_SHORTCUT("tile_map_editor/transpose")->get_as_text() + ")");
	transp->set_focus_mode(FOCUS_NONE);
	transp->connect("pressed", this, "_update_transform_buttons", make_binds(transp));
	tool_hb1->add_child(transp);
	mirror_x = memnew(ToolButton);
	mirror_x->set_toggle_mode(true);
	mirror_x->set_tooltip(TTR("Mirror X") + " (" + ED_GET_SHORTCUT("tile_map_editor/mirror_x")->get_as_text() + ")");
	mirror_x->set_focus_mode(FOCUS_NONE);
	mirror_x->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_x));
	tool_hb1->add_child(mirror_x);
	mirror_y = memnew(ToolButton);
	mirror_y->set_toggle_mode(true);
	mirror_y->set_tooltip(TTR("Mirror Y") + " (" + ED_GET_SHORTCUT("tile_map_editor/mirror_y")->get_as_text() + ")");
	mirror_y->set_focus_mode(FOCUS_NONE);
	mirror_y->connect("pressed", this, "_update_transform_buttons", make_binds(mirror_y));
	tool_hb1->add_child(mirror_y);

	rotate_0 = memnew(ToolButton);
	rotate_0->set_toggle_mode(true);
	rotate_0->set_tooltip(TTR("Rotate 0 degrees"));
	rotate_0->set_focus_mode(FOCUS_NONE);
	rotate_0->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_0));
	tool_hb2->add_child(rotate_0);
	rotate_90 = memnew(ToolButton);
	rotate_90->set_toggle_mode(true);
	rotate_90->set_tooltip(TTR("Rotate 90 degrees"));
	rotate_90->set_focus_mode(FOCUS_NONE);
	rotate_90->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_90));
	tool_hb2->add_child(rotate_90);
	rotate_180 = memnew(ToolButton);
	rotate_180->set_toggle_mode(true);
	rotate_180->set_tooltip(TTR("Rotate 180 degrees"));
	rotate_180->set_focus_mode(FOCUS_NONE);
	rotate_180->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_180));
	tool_hb2->add_child(rotate_180);
	rotate_270 = memnew(ToolButton);
	rotate_270->set_toggle_mode(true);
	rotate_270->set_tooltip(TTR("Rotate 270 degrees"));
	rotate_270->set_focus_mode(FOCUS_NONE);
	rotate_270->connect("pressed", this, "_update_transform_buttons", make_binds(rotate_270));
	tool_hb2->add_child(rotate_270);

	rotate_0->set_pressed(true);
}

TileMapEditor::~TileMapEditor() {
	_clear_bucket_cache();
}

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

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
	} else {

		tile_map_editor->hide();
		tile_map_editor->get_toolbar()->hide();
		tile_map_editor->edit(NULL);
	}
}

TileMapEditorPlugin::TileMapEditorPlugin(EditorNode *p_node) {

	EDITOR_DEF("editors/tile_map/preview_size", 64);
	EDITOR_DEF("editors/tile_map/palette_item_hseparation", 8);
	EDITOR_DEF("editors/tile_map/show_tile_names", true);
	EDITOR_DEF("editors/tile_map/show_tile_ids", false);
	EDITOR_DEF("editors/tile_map/sort_tiles_by_name", true);
	EDITOR_DEF("editors/tile_map/bucket_fill_preview", true);
	EDITOR_DEF("editors/tile_map/show_tile_info_on_hover", true);

	tile_map_editor = memnew(TileMapEditor(p_node));
	add_control_to_container(CONTAINER_CANVAS_EDITOR_SIDE, tile_map_editor);
	tile_map_editor->hide();
	tile_map_editor->set_process(true);
}

TileMapEditorPlugin::~TileMapEditorPlugin() {
}
