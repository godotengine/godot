/**************************************************************************/
/*  virtual_button.h                                                      */
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

#ifndef VIRTUAL_BUTTON_H
#define VIRTUAL_BUTTON_H

#include "scene/gui/virtual_device.h"

class TextParagraph;

class VirtualButton : public VirtualDevice {
	GDCLASS(VirtualButton, VirtualDevice);

public:
	enum StretchMode {
		STRETCH_SCALE,
		STRETCH_TILE,
		STRETCH_KEEP,
		STRETCH_KEEP_CENTERED,
		STRETCH_KEEP_ASPECT,
		STRETCH_KEEP_ASPECT_CENTERED,
		STRETCH_KEEP_ASPECT_COVERED,
	};

private:
	String text;
	String xl_text;
	Ref<TextParagraph> text_buf;
	HorizontalAlignment alignment = HORIZONTAL_ALIGNMENT_CENTER;
	bool flat = false;

	Ref<Font> font;
	int font_size = 0;

	struct ThemeCache {
		Ref<StyleBox> normal_style;
		Ref<StyleBox> pressed_style;
		Ref<StyleBox> hover_style;
		Ref<StyleBox> disabled_style;
		Ref<StyleBox> focus_style;

		Ref<Font> font_theme;
		int font_size_theme = 0;
		Color font_color;
		Color font_pressed_color;
		Color font_hover_color;
		Color font_focus_color;
		Color font_disabled_color;
		Color font_outline_color;
		int outline_size = 0;

		Ref<Texture2D> icon_theme;
		Color icon_normal_color;
		Color icon_pressed_color;
		Color icon_hover_color;
		Color icon_disabled_color;
		Color icon_focus_color;
	} theme_cache;

	void _shape();

	int button_index = 0; // The virtual button index (0, 1, 2...)

	// Similar to TextureButton
	Ref<Texture2D> texture_normal;
	Ref<Texture2D> texture_pressed;
	Ref<Texture2D> texture_hover;
	Ref<Texture2D> texture_disabled;
	Ref<Texture2D> texture_focused;
	Ref<Texture2D> icon;

	bool ignore_texture_size = false;
	StretchMode stretch_mode = STRETCH_SCALE;

protected:
	virtual void _update_theme_item_cache() override;
	void _notification(int p_what);
	static void _bind_methods();

	virtual void pressed_state_changed() override;
	virtual Size2 get_minimum_size() const override;

public:
	void set_text(const String &p_text);
	String get_text() const;

	void set_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_alignment() const;

	void set_button_index(int p_index);
	int get_button_index() const;

	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;

	void set_font_size(int p_size);
	int get_font_size() const;

	void set_texture_normal(const Ref<Texture2D> &p_normal);
	Ref<Texture2D> get_texture_normal() const;

	void set_icon(const Ref<Texture2D> &p_icon);
	Ref<Texture2D> get_icon() const;

	void set_texture_pressed(const Ref<Texture2D> &p_pressed);
	Ref<Texture2D> get_texture_pressed() const;

	void set_texture_hover(const Ref<Texture2D> &p_hover);
	Ref<Texture2D> get_texture_hover() const;

	void set_texture_disabled(const Ref<Texture2D> &p_disabled);
	Ref<Texture2D> get_texture_disabled() const;

	void set_texture_focused(const Ref<Texture2D> &p_focused);
	Ref<Texture2D> get_texture_focused() const;

	void set_ignore_texture_size(bool p_ignore);
	bool get_ignore_texture_size() const;

	void set_stretch_mode(StretchMode p_mode);
	StretchMode get_stretch_mode() const;

	VirtualButton();
};

VARIANT_ENUM_CAST(VirtualButton::StretchMode);

#endif // VIRTUAL_BUTTON_H
