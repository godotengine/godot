/**************************************************************************/
/*  foldable_container.h                                                  */
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

#ifndef FOLDABLE_CONTAINER_H
#define FOLDABLE_CONTAINER_H

#include "scene/gui/container.h"
#include "scene/resources/text_line.h"

class FoldableContainer : public Container {
	GDCLASS(FoldableContainer, Container);

private:
	bool expanded = true;
	String title;
	Ref<TextLine> text_buf;
	String language;
	Control::TextDirection text_direction = Control::TEXT_DIRECTION_INHERITED;
	HorizontalAlignment title_alignment = HORIZONTAL_ALIGNMENT_LEFT;

	bool is_hovering = false;
	int title_panel_height = 0;

	struct ThemeCache {
		Ref<StyleBox> title_style;
		Ref<StyleBox> title_hover_style;
		Ref<StyleBox> title_collapsed_style;
		Ref<StyleBox> title_collapsed_hover_style;
		Ref<StyleBox> panel_style;
		Ref<StyleBox> focus_style;

		Ref<Font> title_font;
		int title_font_size = 0;
		int title_font_outline_size = 0;

		Color title_font_color;
		Color title_hover_font_color;
		Color title_collapsed_font_color;
		Color title_font_outline_color;

		Ref<Texture2D> arrow;
		Ref<Texture2D> arrow_collapsed;
		Ref<Texture2D> arrow_collapsed_mirrored;

		int h_separation = 0;
	} theme_cache;

	Ref<StyleBox> _get_title_style() const;
	Ref<Texture2D> _get_title_icon() const;
	Size2 _get_title_panel_min_size() const;
	void _shape();

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_expanded(bool p_expanded);
	bool is_expanded() const;

	void set_title(const String &p_title);
	String get_title() const;

	void set_title_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_title_alignment() const;

	void set_language(const String &p_language);
	String get_language() const;

	void set_text_direction(TextDirection p_text_direction);
	TextDirection get_text_direction() const;

	virtual Size2 get_minimum_size() const override;

	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

	FoldableContainer(const String &p_title = String());
};

#endif // FOLDABLE_CONTAINER_H
