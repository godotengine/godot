/*************************************************************************/
/*  tile_map_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/tile_map.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/button_group.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class CanvasItemEditor;

class TileMapEditor : public VBoxContainer {

	OBJ_TYPE(TileMapEditor, VBoxContainer );

	UndoRedo *undo_redo;

	enum Tool {

		TOOL_NONE,
		TOOL_PAINTING,
		TOOL_SELECTING,
		TOOL_ERASING,
		TOOL_DUPLICATING,
		TOOL_PICKING
	};

	Tool tool;
	Control *canvas_item_editor;

	Tree *palette;
	EditorNode *editor;
	Panel *panel;
	TileMap *node;
	MenuButton *options;

	bool selection_active;
	Point2i selection_begin;
	Rect2i selection;
	Point2i over_tile;
	bool mouse_over;

	Label *mirror_label;
	ToolButton *mirror_x;
	ToolButton *mirror_y;

	HBoxContainer *canvas_item_editor_hb;


	struct CellOp {
		int idx;
		bool xf;
		bool yf;
		CellOp() { idx=-1; xf=false; yf=false; }
		CellOp(const CellOp& p_other) : idx(p_other.idx), xf(p_other.xf), yf(p_other.yf) {}
	};

	Map<Point2i,CellOp> paint_undo;

	int get_selected_tile() const;
	void set_selected_tile(int p_tile);

	void _update_palette();
	void _canvas_draw();
	void _menu_option(int p_option);

	void _set_cell(const Point2i& p_pos, int p_value, bool p_flip_h=false, bool p_flip_v=false, bool p_with_undo=false);

	void _canvas_mouse_enter();
	void _canvas_mouse_exit();
	void _tileset_settings_changed();


protected:
	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();
	CellOp _get_op_from_cell(const Point2i& p_pos);
public:

	HBoxContainer *get_canvas_item_editor_hb() const { return canvas_item_editor_hb; }
	Vector2 snap_point(const Vector2& p_point) const;
	bool forward_input_event(const InputEvent& p_event);
	void edit(Node *p_tile_map);
	TileMapEditor(EditorNode *p_editor);
};

class TileMapEditorPlugin : public EditorPlugin {

	OBJ_TYPE( TileMapEditorPlugin, EditorPlugin );

	TileMapEditor *tile_map_editor;
	EditorNode *editor;


public:

	virtual bool forward_input_event(const InputEvent& p_event) { return tile_map_editor->forward_input_event(p_event); }

	virtual String get_name() const { return "TileMap"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	TileMapEditorPlugin(EditorNode *p_node);
	~TileMapEditorPlugin();

};


#endif // TILE_MAP_EDITOR_PLUGIN_H
