/*************************************************************************/
/*  tile_map_editor_plugin.h                                             */
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
#ifndef TILE_MAP_EDITOR_PLUGIN_H
#define TILE_MAP_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"

#include "scene/2d/tile_map.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/tool_button.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

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
		TOOL_DUPLICATING,
	};

	enum Options {

		OPTION_BUCKET,
		OPTION_PICK_TILE,
		OPTION_SELECT,
		OPTION_DUPLICATE,
		OPTION_ERASE_SELECTION,
		OPTION_PAINTING,
	};

	TileMap *node;

	EditorNode *editor;
	UndoRedo *undo_redo;
	Control *canvas_item_editor;

	LineEdit *search_box;
	HSlider *size_slider;
	ItemList *palette;

	HBoxContainer *toolbar;

	Label *tile_info;
	MenuButton *options;
	ToolButton *transp;
	ToolButton *mirror_x;
	ToolButton *mirror_y;
	ToolButton *rotate_0;
	ToolButton *rotate_90;
	ToolButton *rotate_180;
	ToolButton *rotate_270;

	Tool tool;

	bool selection_active;
	bool mouse_over;
	bool show_tile_info;

	bool flip_h;
	bool flip_v;
	bool transpose;

	Point2i rectangle_begin;
	Rect2i rectangle;

	Point2i over_tile;

	bool *bucket_cache_visited;
	Rect2i bucket_cache_rect;
	int bucket_cache_tile;
	PoolVector<Vector2> bucket_cache;

	struct CellOp {
		int idx;
		bool xf;
		bool yf;
		bool tr;

		CellOp() {
			idx = -1;
			xf = false;
			yf = false;
			tr = false;
		}
	};

	Map<Point2i, CellOp> paint_undo;

	struct TileData {
		Point2i pos;
		int cell;
		bool flip_h;
		bool flip_v;
		bool transpose;
	};

	List<TileData> copydata;

	void _pick_tile(const Point2 &p_pos);

	PoolVector<Vector2> _bucket_fill(const Point2i &p_start, bool erase = false, bool preview = false);

	void _fill_points(const PoolVector<Vector2> p_points, const Dictionary &p_op);
	void _erase_points(const PoolVector<Vector2> p_points);

	void _select(const Point2i &p_from, const Point2i &p_to);

	void _draw_cell(int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Transform2D &p_xform);
	void _draw_fill_preview(int p_cell, const Point2i &p_point, bool p_flip_h, bool p_flip_v, bool p_transpose, const Transform2D &p_xform);
	void _clear_bucket_cache();

	void _update_copydata();

	int get_selected_tile() const;
	void set_selected_tile(int p_tile);

	void _text_entered(const String &p_text);
	void _text_changed(const String &p_text);
	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _update_palette();
	void _canvas_draw();
	void _menu_option(int p_option);

	void _set_cell(const Point2i &p_pos, int p_value, bool p_flip_h = false, bool p_flip_v = false, bool p_transpose = false, bool p_with_undo = false);

	void _canvas_mouse_enter();
	void _canvas_mouse_exit();
	void _tileset_settings_changed();
	void _icon_size_changed(float p_value);

protected:
	void _notification(int p_what);
	static void _bind_methods();
	CellOp _get_op_from_cell(const Point2i &p_pos);
	void _update_transform_buttons(Object *p_button = NULL);

public:
	HBoxContainer *get_toolbar() const { return toolbar; }

	bool forward_gui_input(const Ref<InputEvent> &p_event);
	void edit(Node *p_tile_map);

	TileMapEditor(EditorNode *p_editor);
	~TileMapEditor();
};

class TileMapEditorPlugin : public EditorPlugin {

	GDCLASS(TileMapEditorPlugin, EditorPlugin);

	TileMapEditor *tile_map_editor;

public:
	virtual bool forward_canvas_gui_input(const Transform2D &p_canvas_xform, const Ref<InputEvent> &p_event) { return tile_map_editor->forward_gui_input(p_event); }

	virtual String get_name() const { return "TileMap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);

	TileMapEditorPlugin(EditorNode *p_node);
	~TileMapEditorPlugin();
};

#endif // TILE_MAP_EDITOR_PLUGIN_H
