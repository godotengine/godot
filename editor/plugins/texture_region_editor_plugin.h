/*************************************************************************/
/*  texture_region_editor_plugin.h                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef TEXTURE_REGION_EDITOR_PLUGIN_H
#define TEXTURE_REGION_EDITOR_PLUGIN_H

#include "canvas_item_editor_plugin.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/sprite.h"
#include "scene/gui/patch_9_rect.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"

class TextureRegionEditor : public Control {

	GDCLASS(TextureRegionEditor, Control);

	enum SnapMode {
		SNAP_NONE,
		SNAP_PIXEL,
		SNAP_GRID,
		SNAP_AUTOSLICE
	};

	friend class TextureRegionEditorPlugin;
	MenuButton *snap_mode_button;
	TextureRect *icon_zoom;
	ToolButton *zoom_in;
	ToolButton *zoom_reset;
	ToolButton *zoom_out;
	HBoxContainer *hb_grid; //For showing/hiding the grid controls when changing the SnapMode
	SpinBox *sb_step_y;
	SpinBox *sb_step_x;
	SpinBox *sb_off_y;
	SpinBox *sb_off_x;
	SpinBox *sb_sep_y;
	SpinBox *sb_sep_x;
	Control *edit_draw;

	VScrollBar *vscroll;
	HScrollBar *hscroll;

	EditorNode *editor;
	UndoRedo *undo_redo;

	Vector2 draw_ofs;
	float draw_zoom;
	bool updating_scroll;

	int snap_mode;
	Vector2 snap_offset;
	Vector2 snap_step;
	Vector2 snap_separation;

	NinePatchRect *node_patch9;
	Sprite *node_sprite;
	Ref<StyleBoxTexture> obj_styleBox;
	Ref<AtlasTexture> atlas_tex;

	Rect2 rect;
	Rect2 rect_prev;
	float prev_margin;
	int edited_margin;
	List<Rect2> autoslice_cache;

	bool drag;
	bool creating;
	Vector2 drag_from;
	int drag_index;

	void _set_snap_mode(int p_mode);
	void _set_snap_off_x(float p_val);
	void _set_snap_off_y(float p_val);
	void _set_snap_step_x(float p_val);
	void _set_snap_step_y(float p_val);
	void _set_snap_sep_x(float p_val);
	void _set_snap_sep_y(float p_val);
	void _zoom_in();
	void _zoom_reset();
	void _zoom_out();
	void apply_rect(const Rect2 &rect);

protected:
	void _notification(int p_what);
	void _node_removed(Object *p_obj);
	static void _bind_methods();

	Vector2 snap_point(Vector2 p_target) const;

	virtual void _changed_callback(Object *p_changed, const char *p_prop);

public:
	void _edit_region();
	void _region_draw();
	void _region_input(const InputEvent &p_input);
	void _scroll_changed(float);

	void edit(Object *p_obj);
	TextureRegionEditor(EditorNode *p_editor);
};

class TextureRegionEditorPlugin : public EditorPlugin {
	GDCLASS(TextureRegionEditorPlugin, EditorPlugin);

	Button *region_button;
	TextureRegionEditor *region_editor;
	EditorNode *editor;

public:
	virtual String get_name() const { return "TextureRegion"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	TextureRegionEditorPlugin(EditorNode *p_node);
};

#endif // TEXTURE_REGION_EDITOR_PLUGIN_H
