/*************************************************************************/
/*  polygon_2d_editor_plugin.h                                           */
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

#ifndef POLYGON_2D_EDITOR_PLUGIN_H
#define POLYGON_2D_EDITOR_PLUGIN_H

#include "editor/plugins/abstract_polygon_2d_editor.h"
#include "scene/gui/scroll_container.h"

class Polygon2DEditor : public AbstractPolygon2DEditor {
	GDCLASS(Polygon2DEditor, AbstractPolygon2DEditor);

	enum Mode {
		MODE_EDIT_UV = MODE_CONT,
		UVEDIT_POLYGON_TO_UV,
		UVEDIT_UV_TO_POLYGON,
		UVEDIT_UV_CLEAR,
		UVEDIT_GRID_SETTINGS
	};

	enum UVMode {
		UV_MODE_CREATE,
		UV_MODE_CREATE_INTERNAL,
		UV_MODE_REMOVE_INTERNAL,
		UV_MODE_EDIT_POINT,
		UV_MODE_MOVE,
		UV_MODE_ROTATE,
		UV_MODE_SCALE,
		UV_MODE_ADD_POLYGON,
		UV_MODE_REMOVE_POLYGON,
		UV_MODE_PAINT_WEIGHT,
		UV_MODE_CLEAR_WEIGHT,
		UV_MODE_MAX
	};

	Button *uv_edit_mode[4];
	Ref<ButtonGroup> uv_edit_group;

	Polygon2D *node;

	UVMode uv_mode;
	AcceptDialog *uv_edit;
	Button *uv_button[UV_MODE_MAX];
	Button *b_snap_enable;
	Button *b_snap_grid;
	Panel *uv_edit_draw;
	HSlider *uv_zoom;
	SpinBox *uv_zoom_value;
	HScrollBar *uv_hscroll;
	VScrollBar *uv_vscroll;
	MenuButton *uv_menu;
	TextureRect *uv_icon_zoom;

	VBoxContainer *bone_scroll_main_vb;
	ScrollContainer *bone_scroll;
	VBoxContainer *bone_scroll_vb;
	Button *sync_bones;
	HSlider *bone_paint_strength;
	SpinBox *bone_paint_radius;
	Label *bone_paint_radius_label;
	bool bone_painting;
	int bone_painting_bone;
	Vector<float> prev_weights;
	Vector2 bone_paint_pos;
	AcceptDialog *grid_settings;

	void _sync_bones();
	void _update_bone_list();

	Vector2 uv_draw_ofs;
	float uv_draw_zoom;
	Vector<Vector2> points_prev;
	Vector<Vector2> uv_create_uv_prev;
	Vector<Vector2> uv_create_poly_prev;
	Vector<Color> uv_create_colors_prev;
	int uv_create_prev_internal_vertices;
	Array uv_create_bones_prev;
	Array polygons_prev;

	Vector2 uv_create_to;
	int point_drag_index;
	bool uv_drag;
	bool uv_create;
	Vector<int> polygon_create;
	UVMode uv_move_current;
	Vector2 uv_drag_from;
	bool updating_uv_scroll;

	AcceptDialog *error;

	Button *button_uv;

	bool use_snap;
	bool snap_show_grid;
	Vector2 snap_offset;
	Vector2 snap_step;

	virtual void _menu_option(int p_option) override;

	void _cancel_editing();
	void _update_polygon_editing_state();

	void _uv_scroll_changed(float);
	void _uv_input(const Ref<InputEvent> &p_input);
	void _uv_draw();
	void _uv_mode(int p_mode);

	void _set_use_snap(bool p_use);
	void _set_show_grid(bool p_show);
	void _set_snap_off_x(float p_val);
	void _set_snap_off_y(float p_val);
	void _set_snap_step_x(float p_val);
	void _set_snap_step_y(float p_val);

	void _uv_edit_mode_select(int p_mode);
	void _uv_edit_popup_hide();
	void _bone_paint_selected(int p_index);

	int _get_polygon_count() const override;

protected:
	virtual Node2D *_get_node() const override;
	virtual void _set_node(Node *p_polygon) override;

	virtual Vector2 _get_offset(int p_idx) const override;

	virtual bool _has_uv() const override { return true; };
	virtual void _commit_action() override;

	void _notification(int p_what);
	static void _bind_methods();

	Vector2 snap_point(Vector2 p_target) const;

public:
	Polygon2DEditor(EditorNode *p_editor);
};

class Polygon2DEditorPlugin : public AbstractPolygon2DEditorPlugin {
	GDCLASS(Polygon2DEditorPlugin, AbstractPolygon2DEditorPlugin);

public:
	Polygon2DEditorPlugin(EditorNode *p_node);
};

#endif // POLYGON_2D_EDITOR_PLUGIN_H
