/**************************************************************************/
/*  tile_map_editor_plugin.h                                              */
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

#ifndef TILE_MAP_EDITOR_PLUGIN_H
#define TILE_MAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

#include "scene/2d/tile_map.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tool_button.h"

class TileMapEditor : public VBoxContainer {
	GDCLASS(TileMapEditor, VBoxContainer);

	enum Tool {

		TOOL_NONE,
		TOOL_PAINTING,
		TOOL_ERASING,
		TOOL_RECTANGLE_PAINT,
		TOOL_RECTANGLE_ERASE,
		TOOL_LINE_PAINT,
		TOOL_LINE_ERASE,
		TOOL_SELECTING,
		TOOL_BUCKET,
		TOOL_PICKING,
		TOOL_PASTING
	};

	enum Options {

		OPTION_COPY,
		OPTION_ERASE_SELECTION,
		OPTION_FIX_INVALID,
		OPTION_CUT
	};

	TileMap *node;
	bool manual_autotile;
	bool priority_atlastile;
	Vector2 manual_position;

	EditorNode *editor;
	UndoRedo *undo_redo;
	Control *canvas_item_editor_viewport;

	LineEdit *search_box;
	HSlider *size_slider;
	ItemList *palette;
	ItemList *manual_palette;

	Label *info_message;

	HBoxContainer *toolbar;
	HBoxContainer *toolbar_right;

	Label *tile_info;
	MenuButton *options;

	ToolButton *paint_button;
	ToolButton *bucket_fill_button;
	ToolButton *picker_button;
	ToolButton *select_button;

	ToolButton *flip_horizontal_button;
	ToolButton *flip_vertical_button;
	ToolButton *rotate_left_button;
	ToolButton *rotate_right_button;
	ToolButton *clear_transform_button;

	CheckBox *manual_button;
	CheckBox *priority_button;

	Tool tool;
	Tool last_tool;

	bool selection_active;
	bool mouse_over;

	bool flip_h;
	bool flip_v;
	bool transpose;
	Point2i autotile_coord;

	Point2i rectangle_begin;
	Rect2i rectangle;

	Point2i over_tile;
	bool refocus_over_tile;

	bool *bucket_cache_visited;
	Rect2i bucket_cache_rect;
	int bucket_cache_tile;
	PoolVector<Vector2> bucket_cache;
	List<Point2i> bucket_queue;

	struct CellOp {
		int idx;
		bool xf;
		bool yf;
		bool tr;
		Vector2 ac;

		CellOp() :
				idx(TileMap::INVALID_CELL),
				xf(false),
				yf(false),
				tr(false) {}
	};

	Map<Point2i, CellOp> paint_undo;

	struct TileData {
		Point2i pos;
		int cell;
		bool flip_h;
		bool flip_v;
		bool transpose;
		Point2i autotile_coord;

		TileData() :
				cell(TileMap::INVALID_CELL),
				flip_h(false),
				flip_v(false),
				transpose(false) {}
	};

	List<TileData> copydata;

	Map<Point2i, CellOp> undo_data;
	Vector<int> invalid_cell;

	void _pick_tile(const Point2 &p_pos);

	PoolVector<Vector2> _bucket_fill(const Point2i &p_start, bool erase = false, bool preview = false);

	void _fill_points(const PoolVector<Vector2> &p_points, const Dictionary &p_op);
	void _erase_points(const PoolVector<Vector2> &p_points);

	void _select(const Point2i &p_from, const Point2i &p_to);
	void _erase_selection();

	void _draw_grid(Control *p_viewport, const Rect2 &p_rect) const;
	void _draw_cell(Control *p_viewport, int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Point2i &p_autotile_coord, const Transform2D &p_xform);
	void _draw_fill_preview(Control *p_viewport, int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Point2i &p_autotile_coord, const Transform2D &p_xform);
	void _clear_bucket_cache();

	void _update_copydata();

	Vector<int> get_selected_tiles() const;
	void set_selected_tiles(Vector<int> p_tile);

	void _manual_toggled(bool p_enabled);
	void _priority_toggled(bool p_enabled);
	void _text_entered(const String &p_text);
	void _text_changed(const String &p_text);
	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _update_palette();
	void _update_button_tool();
	void _button_tool_select(int p_tool);
	void _menu_option(int p_option);
	void _palette_selected(int index);
	void _palette_multi_selected(int index, bool selected);
	void _palette_input(const Ref<InputEvent> &p_event);

	Dictionary _create_cell_dictionary(int tile, bool flip_x, bool flip_y, bool transpose, Vector2 autotile_coord);
	void _start_undo(const String &p_action);
	void _finish_undo();
	void _create_set_cell_undo_redo(const Vector2 &p_vec, const CellOp &p_cell_old, const CellOp &p_cell_new);
	void _set_cell(const Point2i &p_pos, Vector<int> p_values, bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false, const Point2i &p_autotile_coord = Point2());

	void _canvas_mouse_enter();
	void _canvas_mouse_exit();
	void _tileset_settings_changed();
	void _icon_size_changed(float p_value);

	void _clear_transform();
	void _flip_horizontal();
	void _flip_vertical();
	void _rotate(int steps);

protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
	CellOp _get_op_from_cell(const Point2i &p_pos);

public:
	HBoxContainer *get_toolbar() const { return toolbar; }
	HBoxContainer *get_toolbar_right() const { return toolbar_right; }
	Label *get_tile_info() const { return tile_info; }

	bool forward_gui_input(const Ref<InputEvent> &p_event);
	void forward_canvas_draw_over_viewport(Control *p_overlay);

	void edit(Node *p_tile_map);

	TileMapEditor(EditorNode *p_editor);
	~TileMapEditor();
};

class TileMapEditorPlugin : public EditorPlugin {
	GDCLASS(TileMapEditorPlugin, EditorPlugin);

	TileMapEditor *tile_map_editor;

protected:
	void _notification(int p_what);

public:
	virtual bool forward_canvas_gui_input(const Ref<InputEvent> &p_event) { return tile_map_editor->forward_gui_input(p_event); }
	virtual void forward_canvas_draw_over_viewport(Control *p_overlay) { tile_map_editor->forward_canvas_draw_over_viewport(p_overlay); }

	virtual String get_name() const { return "TileMap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	TileMapEditorPlugin(EditorNode *p_node);
	~TileMapEditorPlugin();
};

#endif // TILE_MAP_EDITOR_PLUGIN_H
