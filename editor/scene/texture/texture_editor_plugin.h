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
#include "scene/resources/texture.h"

class AspectRatioContainer;
class ColorRect;
class TextureRect;
class ShaderMaterial;
class ColorChannelSelector;

class TexturePreview : public MarginContainer {
	GDCLASS(TexturePreview, MarginContainer);

private:
	struct ThemeCache {
		Color outline_color;
	} theme_cache;

	TextureRect *texture_display = nullptr;

	MarginContainer *margin_container = nullptr;
	Control *outline_overlay = nullptr;
	AspectRatioContainer *centering_container = nullptr;
	ColorRect *bg_rect = nullptr;
	TextureRect *checkerboard = nullptr;
	Label *metadata_label = nullptr;
	HBoxContainer *right_upper_corner_container = nullptr;
	Button *zoom_out_button = nullptr;
	Button *zoom_reset_button = nullptr;
	Button *zoom_in_button = nullptr;
	Button *popout_button = nullptr;
	Vector2 drag_start;
	Vector2 pan;
	bool panning = false;

	static inline Ref<ShaderMaterial> texture_material;

	ColorChannelSelector *channel_selector = nullptr;

	void _draw_outline();
	void _update_metadata_label_text();
	void _update_pan();

protected:
	virtual void _texture_display_gui_input(const Ref<InputEvent> &p_event);
	void _notification(int p_what);
	void _update_texture_display_ratio();

	void on_selected_channels_changed();
	void on_popout_pressed();
	void on_popout_closed(AcceptDialog *p_dialog);

	void on_zoom_out_pressed();
	void on_zoom_reset_pressed();
	void on_zoom_in_pressed();

	float zoom_level = 1.0;

public:
	static void init_shaders();
	static void finish_shaders();
	virtual CursorShape get_cursor_shape(const Point2 &p_pos) const override;

	TextureRect *get_texture_display();
	TexturePreview(Ref<Texture2D> p_texture, bool p_show_metadata, bool p_popout);
};

class EditorInspectorPluginTexture : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginTexture, EditorInspectorPlugin);

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
