/**************************************************************************/
/*  spin_box.h                                                            */
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

#ifndef SPIN_BOX_H
#define SPIN_BOX_H

#include "scene/gui/line_edit.h"
#include "scene/gui/range.h"
#include "scene/main/timer.h"

class SpinBox : public Range {
	GDCLASS(SpinBox, Range);

	LineEdit *line_edit = nullptr;
	int last_w = 0;
	bool update_on_text_changed = false;

	Timer *range_click_timer = nullptr;
	void _range_click_timeout();
	void _release_mouse();

	void _update_text(bool p_keep_line_edit = false);
	void _text_submitted(const String &p_string);
	void _text_changed(const String &p_string);

	String prefix;
	String suffix;
	String last_updated_text;
	double custom_arrow_step = 0.0;

	void _line_edit_input(const Ref<InputEvent> &p_event);

	struct Drag {
		double base_val = 0.0;
		bool allowed = false;
		bool enabled = false;
		Vector2 capture_pos;
		double diff_y = 0.0;
	} drag;

	void _line_edit_focus_enter();
	void _line_edit_focus_exit();

	inline void _adjust_width_for_icon(const Ref<Texture2D> &icon);

	struct ThemeCache {
		Ref<Texture2D> updown_icon;
	} theme_cache;

protected:
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	void _notification(int p_what);
	static void _bind_methods();

public:
	LineEdit *get_line_edit();

	virtual Size2 get_minimum_size() const override;

	void set_horizontal_alignment(HorizontalAlignment p_alignment);
	HorizontalAlignment get_horizontal_alignment() const;

	void set_editable(bool p_enabled);
	bool is_editable() const;

	void set_suffix(const String &p_suffix);
	String get_suffix() const;

	void set_prefix(const String &p_prefix);
	String get_prefix() const;

	void set_update_on_text_changed(bool p_enabled);
	bool get_update_on_text_changed() const;

	void set_select_all_on_focus(bool p_enabled);
	bool is_select_all_on_focus() const;

	void apply();
	void set_custom_arrow_step(const double p_custom_arrow_step);
	double get_custom_arrow_step() const;

	SpinBox();
};

#endif // SPIN_BOX_H
