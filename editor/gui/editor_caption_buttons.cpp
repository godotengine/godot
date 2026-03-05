/**************************************************************************/
/*  editor_caption_buttons.cpp                                            */
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

#include "editor_caption_buttons.h"

#include "core/math/math_funcs.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"

constexpr float CAPTION_BUTTON_WIDTH = 24.0f;
constexpr float CAPTION_BUTTON_HEIGHT = 20.0f;
constexpr float CAPTION_BUTTON_SIDE_PADDING = 8.0f;

constexpr float CAPTION_GLYPH_WIDTH = 9.0f;
constexpr float CAPTION_GLYPH_HEIGHT = 8.0f;
constexpr float CAPTION_RESTORE_OFFSET = 2.0f;
constexpr float CAPTION_CLOSE_DIAGONAL = 4.0f;

class EditorCaptionButton : public Button {
	GDCLASS(EditorCaptionButton, Button);

public:
	enum CaptionGlyph {
		GLYPH_MINIMIZE,
		GLYPH_MAXIMIZE,
		GLYPH_RESTORE,
		GLYPH_CLOSE,
	};

private:
	CaptionGlyph glyph = GLYPH_CLOSE;

protected:
	void _notification(int p_what) {
		if (p_what != NOTIFICATION_DRAW) {
			return;
		}

		const StringName type = get_theme_type_variation().is_empty() ? SNAME("Button") : get_theme_type_variation();
		Color color;
		if (is_disabled()) {
			color = get_theme_color(SNAME("font_disabled_color"), type);
		} else if (is_pressed()) {
			color = get_theme_color(SNAME("font_pressed_color"), type);
		} else if (is_hovered()) {
			color = get_theme_color(SNAME("font_hover_color"), type);
		} else {
			color = get_theme_color(SNAME("font_color"), type);
		}

		const Size2 size = get_size();
		const Vector2 center = size * 0.5f;
		const float stroke = MAX(1.0f, Math::round(EDSCALE));
		const float w = CAPTION_GLYPH_WIDTH * EDSCALE;
		const float h = CAPTION_GLYPH_HEIGHT * EDSCALE;

		switch (glyph) {
			case GLYPH_MINIMIZE: {
				const float y = center.y;
				draw_line(Vector2(center.x - w * 0.5f, y), Vector2(center.x + w * 0.5f, y), color, stroke, true);
			} break;
			case GLYPH_MAXIMIZE: {
				draw_rect(Rect2(center.x - w * 0.5f, center.y - h * 0.5f, w, h), color, false, stroke);
			} break;
			case GLYPH_RESTORE: {
				const float dx = CAPTION_RESTORE_OFFSET * EDSCALE;
				const float dy = CAPTION_RESTORE_OFFSET * EDSCALE;
				draw_rect(Rect2(center.x - w * 0.5f + dx, center.y - h * 0.5f - dy, w, h), color, false, stroke);
				draw_rect(Rect2(center.x - w * 0.5f - dx, center.y - h * 0.5f + dy, w, h), color, false, stroke);
			} break;
			case GLYPH_CLOSE: {
				const float d = CAPTION_CLOSE_DIAGONAL * EDSCALE;
				draw_line(Vector2(center.x - d, center.y - d), Vector2(center.x + d, center.y + d), color, stroke, true);
				draw_line(Vector2(center.x + d, center.y - d), Vector2(center.x - d, center.y + d), color, stroke, true);
			} break;
		}
	}

	static void _bind_methods() {}

public:
	void set_caption_glyph(CaptionGlyph p_glyph) {
		if (glyph == p_glyph) {
			return;
		}
		glyph = p_glyph;
		queue_redraw();
	}
};

void EditorCaptionButtons::_minimize_pressed() {
	emit_signal(SNAME("minimize_requested"));
}

void EditorCaptionButtons::_toggle_maximize_pressed() {
	emit_signal(SNAME("toggle_maximize_requested"));
}

void EditorCaptionButtons::_close_pressed() {
	emit_signal(SNAME("close_requested"));
}

void EditorCaptionButtons::update_for_window(Window *p_window) {
	if (!p_window) {
		return;
	}

	if (minimize_button) {
		minimize_button->set_disabled(p_window->get_flag(Window::FLAG_MINIMIZE_DISABLED));
	}
	if (maximize_button) {
		const bool maximized = p_window->get_mode() == Window::MODE_MAXIMIZED;
		if (EditorCaptionButton *max_btn = Object::cast_to<EditorCaptionButton>(maximize_button)) {
			max_btn->set_caption_glyph(maximized ? EditorCaptionButton::GLYPH_RESTORE : EditorCaptionButton::GLYPH_MAXIMIZE);
		}
		maximize_button->set_tooltip_text(maximized ? TTR("Restore") : TTR("Maximize"));
		maximize_button->set_disabled(p_window->get_flag(Window::FLAG_MAXIMIZE_DISABLED));
	}
}

EditorCaptionButtons::EditorCaptionButtons() {
	set_name("WindowButtons");
	set_mouse_filter(Control::MOUSE_FILTER_STOP);
	set_alignment(BoxContainer::ALIGNMENT_END);

	Control *caption_left_padding = memnew(Control);
	caption_left_padding->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	caption_left_padding->set_custom_minimum_size(Size2(CAPTION_BUTTON_SIDE_PADDING, 0) * EDSCALE);
	add_child(caption_left_padding);

	EditorCaptionButton *min_btn = memnew(EditorCaptionButton);
	min_btn->set_caption_glyph(EditorCaptionButton::GLYPH_MINIMIZE);
	minimize_button = min_btn;
	minimize_button->set_theme_type_variation(SceneStringName(FlatButton));
	minimize_button->set_flat(true);
	minimize_button->set_focus_mode(Control::FOCUS_NONE);
	minimize_button->set_tooltip_text(TTR("Minimize"));
	minimize_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	minimize_button->connect(SceneStringName(pressed), callable_mp(this, &EditorCaptionButtons::_minimize_pressed));
	add_child(minimize_button);

	EditorCaptionButton *max_btn = memnew(EditorCaptionButton);
	max_btn->set_caption_glyph(EditorCaptionButton::GLYPH_MAXIMIZE);
	maximize_button = max_btn;
	maximize_button->set_theme_type_variation(SceneStringName(FlatButton));
	maximize_button->set_flat(true);
	maximize_button->set_focus_mode(Control::FOCUS_NONE);
	maximize_button->set_tooltip_text(TTR("Maximize"));
	maximize_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	maximize_button->connect(SceneStringName(pressed), callable_mp(this, &EditorCaptionButtons::_toggle_maximize_pressed));
	add_child(maximize_button);

	EditorCaptionButton *cls_btn = memnew(EditorCaptionButton);
	cls_btn->set_caption_glyph(EditorCaptionButton::GLYPH_CLOSE);
	Button *close_button = cls_btn;
	close_button->set_theme_type_variation(SceneStringName(FlatButton));
	close_button->set_flat(true);
	close_button->set_focus_mode(Control::FOCUS_NONE);
	close_button->set_tooltip_text(TTR("Close"));
	close_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	close_button->connect(SceneStringName(pressed), callable_mp(this, &EditorCaptionButtons::_close_pressed));
	add_child(close_button);

	Control *caption_right_padding = memnew(Control);
	caption_right_padding->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	caption_right_padding->set_custom_minimum_size(Size2(CAPTION_BUTTON_SIDE_PADDING, 0) * EDSCALE);
	add_child(caption_right_padding);
}

void EditorCaptionButtons::_bind_methods() {
	ADD_SIGNAL(MethodInfo("minimize_requested"));
	ADD_SIGNAL(MethodInfo("toggle_maximize_requested"));
	ADD_SIGNAL(MethodInfo("close_requested"));
}

