/*************************************************************************/
/*  sprite_region_editor_plugin.h                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Author: Mariano Suligoy                                               */
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

#ifndef SPRITE_REGION_EDITOR_PLUGIN_H
#define SPRITE_REGION_EDITOR_PLUGIN_H

#include "canvas_item_editor_plugin.h"
#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/2d/sprite.h"

class SpriteRegionEditor : public HBoxContainer {

	OBJ_TYPE(SpriteRegionEditor, HBoxContainer );

	friend class SpriteRegionEditorPlugin;

	ToolButton *edit_node;
//	Button *use_region;
	ToolButton *b_snap_enable;
	ToolButton *b_snap_grid;
	TextureFrame *icon_zoom;
	HSlider *zoom;
	SpinBox *zoom_value;
	SpinBox *sb_step_y;
	SpinBox *sb_step_x;
	SpinBox *sb_off_y;
	SpinBox *sb_off_x;
	Control *edit_draw;

	VScrollBar *vscroll;
	HScrollBar *hscroll;

	Sprite *node;
	EditorNode *editor;
	AcceptDialog *dlg_editor;
	UndoRedo* undo_redo;

	Vector2 draw_ofs;
	float draw_zoom;
	bool updating_scroll;

	bool use_snap;
	bool snap_show_grid;
	Vector2 snap_offset;
	Vector2 snap_step;

	Rect2 rect;
	Rect2 rect_prev;
	bool drag;
	bool creating;
	Vector2 drag_from;
	int drag_index;

	AcceptDialog *error;

	void _set_use_snap(bool p_use);
	void _set_show_grid(bool p_show);
	void _set_snap_off_x(float p_val);
	void _set_snap_off_y(float p_val);
	void _set_snap_step_x(float p_val);
	void _set_snap_step_y(float p_val);

protected:

	void _notification(int p_what);
	void _node_removed(Node *p_node);
	static void _bind_methods();

	Vector2 snap_point(Vector2 p_target) const;

public:

	void edit();
	void _edit_node();
	void _region_draw();
	void _region_input(const InputEvent &p_input);
	void _scroll_changed(float);

	void edit(Node *p_sprite);
	SpriteRegionEditor(EditorNode* p_editor);

};

class SpriteRegionEditorPlugin : public EditorPlugin
{

	OBJ_TYPE( SpriteRegionEditorPlugin, EditorPlugin );

	SpriteRegionEditor *region_editor;
	EditorNode *editor;
public:

	virtual String get_name() const { return "SpriteRegion"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	SpriteRegionEditorPlugin(EditorNode *p_node);
};

#endif // SPRITE_REGION_EDITOR_PLUGIN_H
