/**************************************************************************/
/*  texture_region_editor_plugin.h                                        */
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

#ifndef TEXTURE_REGION_EDITOR_PLUGIN_H
#define TEXTURE_REGION_EDITOR_PLUGIN_H

#include "canvas_item_editor_plugin.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/2d/sprite.h"
#include "scene/3d/sprite_3d.h"
#include "scene/gui/nine_patch_rect.h"
#include "scene/resources/style_box.h"
#include "scene/resources/texture.h"

/**
	@author Mariano Suligoy
*/

class TextureRegionEditor : public VBoxContainer {
	GDCLASS(TextureRegionEditor, VBoxContainer);

	enum SnapMode {
		SNAP_NONE,
		SNAP_PIXEL,
		SNAP_GRID,
		SNAP_AUTOSLICE
	};

	friend class TextureRegionEditorPlugin;
	OptionButton *snap_mode_button;
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
	Panel *edit_draw;

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

	Sprite *node_sprite;
	Sprite3D *node_sprite_3d;
	NinePatchRect *node_ninepatch;
	Ref<StyleBoxTexture> obj_styleBox;
	Ref<AtlasTexture> atlas_tex;

	Rect2 rect;
	Rect2 rect_prev;
	float prev_margin;
	int edited_margin;
	Map<RID, List<Rect2>> cache_map;
	List<Rect2> autoslice_cache;
	bool autoslice_is_dirty;

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
	void _zoom_on_position(float p_zoom, Point2 p_position = Point2());
	void _zoom_in();
	void _zoom_reset();
	void _zoom_out();
	void apply_rect(const Rect2 &p_rect);
	void _update_rect();
	void _update_autoslice();

protected:
	void _notification(int p_what);
	void _node_removed(Object *p_obj);
	static void _bind_methods();

	Vector2 snap_point(Vector2 p_target) const;

	virtual void _changed_callback(Object *p_changed, const char *p_prop);

public:
	void _edit_region();
	void _region_draw();
	void _region_input(const Ref<InputEvent> &p_input);
	void _scroll_changed(float);
	bool is_stylebox();
	bool is_atlas_texture();
	bool is_ninepatch();
	Sprite3D *get_sprite_3d();
	Sprite *get_sprite();

	void edit(Object *p_obj);
	TextureRegionEditor(EditorNode *p_editor);
};

class TextureRegionEditorPlugin : public EditorPlugin {
	GDCLASS(TextureRegionEditorPlugin, EditorPlugin);

	bool manually_hidden;
	Button *texture_region_button;
	TextureRegionEditor *region_editor;
	EditorNode *editor;

protected:
	static void _bind_methods();

	void _editor_visiblity_changed();

public:
	virtual String get_name() const { return "TextureRegion"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_object);
	virtual bool handles(Object *p_object) const;
	virtual void make_visible(bool p_visible);
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	TextureRegionEditorPlugin(EditorNode *p_node);
};

#endif // TEXTURE_REGION_EDITOR_PLUGIN_H
