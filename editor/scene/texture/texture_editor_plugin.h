/**************************************************************************/
/*  texture_editor_plugin.h                                               */
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

#include "editor/inspector/editor_inspector.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/view_panner.h"
#include "scene/resources/texture.h"

class TextureRect;
class ShaderMaterial;
class Button;
class ColorChannelSelector;
class SpinBox;

class TexturePreview : public MarginContainer {
	GDCLASS(TexturePreview, MarginContainer);

private:
	struct ThemeCache {
		Color outline_color;
	} theme_cache;

	Control *texture_display = nullptr;
	Control *outline_overlay = nullptr;
	TextureRect *checkerboard = nullptr;
	VScrollBar *vscroll = nullptr;
	HScrollBar *hscroll = nullptr;

	HBoxContainer *top_bar = nullptr;
	Button *zoom_out_button = nullptr;
	Button *zoom_reset_button = nullptr;
	Button *zoom_in_button = nullptr;
	Button *popout_button = nullptr;
	Label *metadata_label = nullptr;
	Button *metadata_toggle = nullptr;
	ColorChannelSelector *channel_selector = nullptr;
	SpinBox *mipmap_spinbox = nullptr;

	Ref<Texture2D> preview_texture;
	Ref<ViewPanner> panner;
	Vector2 draw_ofs;
	float draw_zoom = 1.0;
	float min_draw_zoom = 1.0;
	float max_draw_zoom = 1.0;
	bool updating_scroll = false;

	static inline Ref<ShaderMaterial> texture_material;

	void _pan_callback(Vector2 p_scroll_vec, Ref<InputEvent> p_event);
	void _zoom_callback(float p_zoom_factor, Vector2 p_origin, Ref<InputEvent> p_event);
	void _scroll_changed(float);
	void _zoom_on_position(float p_zoom, Point2 p_position = Point2());
	void _zoom_in();
	void _zoom_out();
	void _fit_to_view();
	void _clamp_draw_ofs();
	void _update_scrollbars();
	Transform2D _get_offset_transform() const;

	void _update_metadata_label_text();
	void _toggle_metadata_label();

protected:
	void _notification(int p_what);
	void _texture_display_gui_input(const Ref<InputEvent> &p_event);
	void _texture_display_draw();
	void _outline_overlay_draw();

	void on_selected_channels_changed();
	void on_selected_mipmap_changed(double p_value);
	void on_popout_pressed();
	void on_popout_closed(AcceptDialog *p_dialog);

public:
	static void init_shaders();
	static void finish_shaders();

	TexturePreview(Ref<Texture2D> p_texture, bool p_show_metadata, bool p_popout = false);
};

class EditorInspectorPluginTexture : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginTexture, EditorInspectorPlugin);

	Ref<Image> this_image;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class TextureEditorPlugin : public EditorPlugin {
	GDCLASS(TextureEditorPlugin, EditorPlugin);

public:
	virtual String get_plugin_name() const override { return "Texture2D"; }

	TextureEditorPlugin();
};
