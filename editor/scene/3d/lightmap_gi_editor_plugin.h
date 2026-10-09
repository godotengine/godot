/**************************************************************************/
/*  lightmap_gi_editor_plugin.h                                           */
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

#pragma once

#include "core/math/rect2.h"
#include "editor/docks/editor_dock.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/3d/lightmap_gi.h"
#include "scene/gui/check_button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/view_panner.h"

struct EditorProgress;
class EditorFileDialog;
class LightmapGI;

class LightmapGIBakeEditor : public Control {
	GDCLASS(LightmapGIBakeEditor, Control);

	friend class LightmapGIEditorPlugin;

	LightmapGI *lightmap = nullptr;

	Button *bake = nullptr;

	EditorFileDialog *file_dialog = nullptr;
	static EditorProgress *tmp_progress;
	static bool bake_func_step(float p_progress, const String &p_description, void *, bool p_refresh);
	static void bake_func_end(uint64_t p_time_started);

	void _bake_select_file(const String &p_file);
	void _bake();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void edit(LightmapGI *p_lightmap);
	LightmapGIBakeEditor();
};

class LightmapGITextureEditor : public EditorDock {
	GDCLASS(LightmapGITextureEditor, EditorDock);

	Button *zoom_in = nullptr;
	Button *zoom_reset = nullptr;
	Button *zoom_out = nullptr;
	Button *zoom_to_fit = nullptr;

	CheckButton *cb_show_shadowmask = nullptr;
	CheckButton *cb_show_uv = nullptr;

	Label *lbl_islands_title = nullptr;

	PanelContainer *texture_preview = nullptr;
	Control *texture_overlay = nullptr;

	SpinBox *sb_layer = nullptr;
	ItemList *user_list = nullptr;

	VScrollBar *vscroll = nullptr;
	HScrollBar *hscroll = nullptr;

	Vector2 draw_ofs;
	float draw_zoom = 1.0;
	float min_draw_zoom = 1.0;
	float max_draw_zoom = 1.0;
	bool updating_scroll = false;

	bool drag = false;
	bool creating = false;
	bool moving = false;
	Vector2 drag_from;
	int drag_index = -1;
	bool request_center = false;

	bool show_uv = false;
	bool show_shadowmask = false;

	Ref<Texture2D> current_texture;
	LightmapGI *lightmap = nullptr;
	Node *selected_node = nullptr;
	Ref<LightmapGIData> lightmap_data;

	int current_layer = 0;
	int total_layers = 0;

	struct UserEntry {
		int slice = -1;
		int user_idx = -1;
		int subinstance = -1;
	};
	UserEntry selected_user = {};

	Vector<UserEntry> users_in_list;

	bool internal_selection = false;

	Ref<ViewPanner> panner;
	void _pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event);
	void _zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event);
	void _scroll_changed(float);
	Transform2D _get_offset_transform() const;

	void _zoom_on_position(float p_zoom, Point2 p_position = Point2());
	void _zoom_in();
	void _zoom_reset();
	void _zoom_out();
	void _zoom_to_fit();
	float _get_zoom_to_fit(const Point2 p_size);
	void _update_zoom_label();

	void _texture_preview_draw();
	void _texture_overlay_draw();
	void _texture_overlay_input(const Ref<InputEvent> &p_input);

	bool _update_current_texture(int p_layer, bool p_shadowmask);

	void _set_layer(int p_layer);

	void _show_shadowmask_pressed();
	void _show_uv_pressed();

	void _update_list();
	Rect2 _get_user_uv_rect(int p_user);

	enum SelectMode {
		NONE,
		SELECT,
		FOCUS
	};

	void _select_user(UserEntry p_user, SelectMode p_select_mode);
	void _on_user_list_item_selected(int p_index);
	void _on_user_list_item_activated(int p_index);

	void _open();

protected:
	void _notification(int p_what);

public:
	void edit(LightmapGI *p_lightmap, Node3D *p_selected_node = nullptr, int user_idx = -1);
	bool is_internal_selection() const { return internal_selection; }
	void clear_internal_selection() { internal_selection = false; }
	LightmapGITextureEditor();
};

class LightmapGIEditorPlugin : public EditorPlugin {
	GDCLASS(LightmapGIEditorPlugin, EditorPlugin);

	LightmapGIBakeEditor *bake_editor = nullptr;
	LightmapGITextureEditor *texture_editor = nullptr;

public:
	virtual String get_plugin_name() const override { return "LightmapGI"; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;

	LightmapGIEditorPlugin();
};
