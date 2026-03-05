/**************************************************************************/
/*  editor_title_bar.cpp                                                  */
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

#include "editor_title_bar.h"

#include "core/math/math_funcs.h"
#include "core/object/callable_mp.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "servers/display/display_server.h"

#ifdef WINDOWS_ENABLED
#include <windows.h>

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
#endif

void EditorTitleBar::_ensure_window_buttons() {
#ifdef WINDOWS_ENABLED
	if (window_buttons || !can_move || !DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_EXTEND_TO_TITLE)) {
		return;
	}

	Window *win = get_window();
	if (!win || !win->get_flag(Window::FLAG_EXTEND_TO_TITLE)) {
		return;
	}

	window_buttons = memnew(HBoxContainer);
	window_buttons->set_name("WindowButtons");
	window_buttons->set_mouse_filter(Control::MOUSE_FILTER_STOP);
	window_buttons->set_alignment(BoxContainer::ALIGNMENT_END);

	Control *caption_left_padding = memnew(Control);
	caption_left_padding->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	caption_left_padding->set_custom_minimum_size(Size2(CAPTION_BUTTON_SIDE_PADDING, 0) * EDSCALE);
	window_buttons->add_child(caption_left_padding);

	EditorCaptionButton *min_btn = memnew(EditorCaptionButton);
	min_btn->set_caption_glyph(EditorCaptionButton::GLYPH_MINIMIZE);
	minimize_button = min_btn;
	minimize_button->set_theme_type_variation("FlatButton");
	minimize_button->set_flat(true);
	minimize_button->set_focus_mode(Control::FOCUS_NONE);
	minimize_button->set_tooltip_text(TTR("Minimize"));
	minimize_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	minimize_button->connect(SceneStringName(pressed), callable_mp(this, &EditorTitleBar::_minimize_pressed));
	window_buttons->add_child(minimize_button);

	EditorCaptionButton *max_btn = memnew(EditorCaptionButton);
	max_btn->set_caption_glyph(EditorCaptionButton::GLYPH_MAXIMIZE);
	maximize_button = max_btn;
	maximize_button->set_theme_type_variation("FlatButton");
	maximize_button->set_flat(true);
	maximize_button->set_focus_mode(Control::FOCUS_NONE);
	maximize_button->set_tooltip_text(TTR("Maximize"));
	maximize_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	maximize_button->connect(SceneStringName(pressed), callable_mp(this, &EditorTitleBar::_maximize_pressed));
	window_buttons->add_child(maximize_button);

	EditorCaptionButton *cls_btn = memnew(EditorCaptionButton);
	cls_btn->set_caption_glyph(EditorCaptionButton::GLYPH_CLOSE);
	close_button = cls_btn;
	close_button->set_theme_type_variation("FlatButton");
	close_button->set_flat(true);
	close_button->set_focus_mode(Control::FOCUS_NONE);
	close_button->set_tooltip_text(TTR("Close"));
	close_button->set_custom_minimum_size(Size2(CAPTION_BUTTON_WIDTH, CAPTION_BUTTON_HEIGHT) * EDSCALE);
	close_button->connect(SceneStringName(pressed), callable_mp(this, &EditorTitleBar::_close_pressed));
	window_buttons->add_child(close_button);

	Control *caption_right_padding = memnew(Control);
	caption_right_padding->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	caption_right_padding->set_custom_minimum_size(Size2(CAPTION_BUTTON_SIDE_PADDING, 0) * EDSCALE);
	window_buttons->add_child(caption_right_padding);

	add_child(window_buttons);
	move_child(window_buttons, get_child_count() - 1);
	_update_window_buttons();
#endif
}

void EditorTitleBar::_update_window_buttons() {
#ifdef WINDOWS_ENABLED
	if (!window_buttons) {
		return;
	}

	Window *win = get_window();
	if (!win) {
		return;
	}

	if (minimize_button) {
		minimize_button->set_disabled(win->get_flag(Window::FLAG_MINIMIZE_DISABLED));
	}
	if (maximize_button) {
		const bool maximized = win->get_mode() == Window::MODE_MAXIMIZED;
		if (EditorCaptionButton *max_btn = Object::cast_to<EditorCaptionButton>(maximize_button)) {
			max_btn->set_caption_glyph(maximized ? EditorCaptionButton::GLYPH_RESTORE : EditorCaptionButton::GLYPH_MAXIMIZE);
		}
		maximize_button->set_tooltip_text(maximized ? TTR("Restore") : TTR("Maximize"));
		maximize_button->set_disabled(win->get_flag(Window::FLAG_MAXIMIZE_DISABLED));
	}
#endif
}

void EditorTitleBar::_minimize_pressed() {
#ifdef WINDOWS_ENABLED
	Window *win = get_window();
	if (win) {
		win->set_mode(Window::MODE_MINIMIZED);
	}
#endif
}

void EditorTitleBar::_maximize_pressed() {
#ifdef WINDOWS_ENABLED
	Window *win = get_window();
	if (!win) {
		return;
	}

	if (win->get_mode() == Window::MODE_MAXIMIZED) {
		win->set_mode(Window::MODE_WINDOWED);
	} else {
		win->set_mode(Window::MODE_MAXIMIZED);
	}
	_update_window_buttons();
#endif
}

void EditorTitleBar::_close_pressed() {
#ifdef WINDOWS_ENABLED
	if (EditorNode::get_singleton()) {
		callable_mp(EditorNode::get_singleton(), &EditorNode::trigger_menu_option).call_deferred((int)EditorNode::SCENE_QUIT, false);
		return;
	}

	Window *win = get_window();
	if (!win) {
		return;
	}

	const int64_t native_handle = DisplayServer::get_singleton()->window_get_native_handle(DisplayServerEnums::WINDOW_HANDLE, win->get_window_id());
	if (native_handle != 0) {
		PostMessageW(reinterpret_cast<HWND>(native_handle), WM_CLOSE, 0, 0);
	}
#endif
}

void EditorTitleBar::gui_input(const Ref<InputEvent> &p_event) {
	if (!can_move) {
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && moving) {
		if (mm->get_button_mask().has_flag(MouseButtonMask::LEFT)) {
			Window *w = Object::cast_to<Window>(get_viewport());
			if (w) {
				Point2 mouse = DisplayServer::get_singleton()->mouse_get_position();
				w->set_position(mouse - click_pos);
			}
		} else {
			moving = false;
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && has_point(mb->get_position())) {
		Window *w = Object::cast_to<Window>(get_viewport());
		if (w) {
			if (mb->get_button_index() == MouseButton::LEFT && mb->is_double_click() && mb->is_pressed()) {
				if (DisplayServer::get_singleton()->window_maximize_on_title_dbl_click()) {
					if (w->get_mode() == Window::MODE_WINDOWED) {
						w->set_mode(Window::MODE_MAXIMIZED);
					} else if (w->get_mode() == Window::MODE_MAXIMIZED) {
						w->set_mode(Window::MODE_WINDOWED);
					}
				} else if (DisplayServer::get_singleton()->window_minimize_on_title_dbl_click()) {
					w->set_mode(Window::MODE_MINIMIZED);
				}
				moving = false;
				return;
			}

			if (mb->get_button_index() == MouseButton::LEFT) {
				if (mb->is_pressed()) {
					if (DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_WINDOW_DRAG)) {
						DisplayServer::get_singleton()->window_start_drag(w->get_window_id());
					} else {
						click_pos = DisplayServer::get_singleton()->mouse_get_position() - w->get_position();
						moving = true;
					}
				} else {
					moving = false;
				}
			}
		}
	}
}

void EditorTitleBar::set_center_control(Control *p_center_control) {
	center_control = p_center_control;
}

Control *EditorTitleBar::get_center_control() const {
	return center_control;
}

void EditorTitleBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_EXIT_TREE: {
			SceneTree::get_singleton()->get_root()->disconnect(SceneStringName(nonclient_window_input), callable_mp(this, &EditorTitleBar::gui_input));
			get_window()->set_nonclient_area(Rect2i());
		} break;
		case NOTIFICATION_ENTER_TREE: {
			SceneTree::get_singleton()->get_root()->connect(SceneStringName(nonclient_window_input), callable_mp(this, &EditorTitleBar::gui_input));
			_ensure_window_buttons();
			[[fallthrough]];
		}
		case NOTIFICATION_RESIZED: {
			get_window()->set_nonclient_area(get_global_transform().xform(Rect2i(get_position(), get_size())));
			Window *win = get_window();
			if (win) {
				const Size2i min_size = win->get_min_size();
				const int titlebar_min_width = get_combined_minimum_size().x;
				if (min_size.x < titlebar_min_width) {
					win->set_min_size(Size2i(titlebar_min_width, min_size.y));
				}
			}
			_update_window_buttons();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_update_window_buttons();
		} break;
		case NOTIFICATION_SORT_CHILDREN: {
			if (!center_control) {
				break;
			}
			Control *prev = nullptr;
			Control *base = nullptr;
			Control *next = nullptr;

			bool rtl = is_layout_rtl();

			int start;
			int end;
			int delta;
			if (rtl) {
				start = get_child_count() - 1;
				end = -1;
				delta = -1;
			} else {
				start = 0;
				end = get_child_count();
				delta = +1;
			}

			for (int i = start; i != end; i += delta) {
				Control *c = as_sortable_control(get_child(i));
				if (!c) {
					continue;
				}
				if (base) {
					next = c;
					break;
				}
				if (c != center_control) {
					prev = c;
					continue;
				}
				base = c;
			}
			if (base && prev && next) {
				Size2i title_size = get_size();
				Size2i c_size = base->get_combined_minimum_size();

				int min_offset = prev->get_position().x + prev->get_combined_minimum_size().x;
				int max_offset = next->get_position().x + next->get_size().x - next->get_combined_minimum_size().x - c_size.x;

				int offset = (title_size.width - c_size.width) / 2;
				offset = CLAMP(offset, min_offset, max_offset);

				fit_child_in_rect(prev, Rect2i(prev->get_position().x, 0, offset - prev->get_position().x, title_size.height));
				fit_child_in_rect(base, Rect2i(offset, 0, c_size.width, title_size.height));
				fit_child_in_rect(next, Rect2i(offset + c_size.width, 0, next->get_position().x + next->get_size().x - (offset + c_size.width), title_size.height));
			}
		} break;
	}
}

void EditorTitleBar::set_can_move_window(bool p_enabled) {
	can_move = p_enabled;
	set_process_input(can_move);
	_ensure_window_buttons();
	_update_window_buttons();
}

bool EditorTitleBar::get_can_move_window() const {
	return can_move;
}
