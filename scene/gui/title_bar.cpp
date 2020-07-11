/*************************************************************************/
/*  title_bar.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "title_bar.h"

void TitleBar::_update_button_rects() {
	Size2 size = get_size();

	float margin = get_theme_constant("margin", "TitleBar");
	float btn_margin = get_theme_constant("button_margin", "TitleBar");
	Size2 btn_pos{ size.x - margin, margin };
	if (close_btn->is_visible()) {
		Size2 close_minsize = close_btn->get_combined_minimum_size();
		btn_pos.x -= close_minsize.x;
		close_btn->set_position(btn_pos);
		btn_pos.x -= btn_margin;
	}
	if (maximize_btn->is_visible()) {
		Size2 maximize_minsize = maximize_btn->get_combined_minimum_size();
		btn_pos.x -= maximize_minsize.x;
		maximize_btn->set_position(btn_pos);
		btn_pos.x -= btn_margin;
	}
	if (minimize_btn->is_visible()) {
		Size2 minimize_minsize = minimize_btn->get_combined_minimum_size();
		btn_pos.x -= minimize_minsize.x;
		minimize_btn->set_position(btn_pos);
		btn_pos.x -= btn_margin;
	}
}

void TitleBar::_update_button_textures() {
	if (window->get_mode() == Window::MODE_MAXIMIZED) {
		auto restore = get_theme_icon("restore", "TitleBar");
		auto restore_highlight = get_theme_icon("restore_highlight", "TitleBar");
		maximize_btn->set_normal_texture(restore);
		maximize_btn->set_hover_texture(restore_highlight);
		maximize_btn->set_pressed_texture(restore);
	} else {
		auto maximize = get_theme_icon("maximize", "TitleBar");
		auto maximize_highlight = get_theme_icon("maximize_highlight", "TitleBar");
		maximize_btn->set_normal_texture(maximize);
		maximize_btn->set_hover_texture(maximize_highlight);
		maximize_btn->set_pressed_texture(maximize);
	}
}

void TitleBar::_close_pressed() {
	close_window();
}

void TitleBar::_maximize_pressed() {
	switch (window->get_mode()) {
		case Window::MODE_WINDOWED: {
			maximize_window();
		} break;
		case Window::MODE_MINIMIZED:
		case Window::MODE_MAXIMIZED: {
			restore_window();
		} break;
		default:
			break; // Do nothing
	}
}

void TitleBar::_minimize_pressed() {
	minimize_window();
}

void TitleBar::_gui_input(Ref<InputEvent> p_event) {
	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid() && initial_drag_pos.x != -1) {
		Point2 mouse = DisplayServer::get_singleton()->mouse_get_absolute_position();
		window->set_position(mouse - initial_drag_pos);
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_LEFT) {
			if (mb->is_pressed() && this->has_point(mb->get_position())) {
				initial_drag_pos = DisplayServer::get_singleton()->mouse_get_absolute_position() - window->get_position();
			} else {
				initial_drag_pos = Point2{ -1, -1 };
			}
		}
	}
}

void TitleBar::_bind_methods() {
	ClassDB::bind_method("_gui_input", &TitleBar::_gui_input);
	ClassDB::bind_method("is_forcing_custom_buttons", &TitleBar::is_forcing_custom_buttons);
	ClassDB::bind_method("set_force_custom_buttons", &TitleBar::set_force_custom_buttons);
	ClassDB::bind_method("is_button_enabled", &TitleBar::is_button_enabled);
	ClassDB::bind_method("set_buttons_enabled", &TitleBar::set_buttons_enabled);
	ClassDB::bind_method("close_window", &TitleBar::close_window);
	ClassDB::bind_method("maximize_window", &TitleBar::maximize_window);
	ClassDB::bind_method("minimize_window", &TitleBar::minimize_window);
	ClassDB::bind_method("restore_window", &TitleBar::restore_window);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_custom_buttons"), "set_force_custom_buttons", "is_forcing_custom_buttons");

	BIND_ENUM_CONSTANT(BUTTON_CLOSE);
	BIND_ENUM_CONSTANT(BUTTON_MAXIMIZE);
	BIND_ENUM_CONSTANT(BUTTON_MINIMIZE);

	ADD_SIGNAL(MethodInfo("window_closing"));
	ADD_SIGNAL(MethodInfo("window_maximizing"));
	ADD_SIGNAL(MethodInfo("window_minimizing"));
	ADD_SIGNAL(MethodInfo("window_restoring"));
}

void TitleBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			auto canvas_item = get_canvas_item();
			float margin = get_theme_constant("margin", "TitleBar");
			auto abs_pos = get_global_position();
			auto size = get_size();
			auto title = window->get_title();
			auto title_font = get_theme_font("title_font", "TitleBar");
			auto title_color = get_theme_color("title_color", "TitleBar");
			float font_height = title_font->get_height() - title_font->get_descent() * 2;
			float y = abs_pos.y + (size.y + font_height) / 2;
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_TITLE_BUTTONS) && !force_custom_buttons) {
				float x = DisplayServer::get_singleton()->window_get_suggested_title_x(window->get_window_id());
				title_font->draw(canvas_item, Point2{ x, y }, title, title_color);
			} else {
				float x = abs_pos.x + (size.x - title_font->get_string_size(title).x) / 2;
				title_font->draw(canvas_item, Point2{ x, y }, title, title_color);
			}
		} break;

		case NOTIFICATION_THEME_CHANGED:
		case NOTIFICATION_ENTER_TREE: {
			auto close = get_theme_icon("close", "TitleBar");
			auto close_highlight = get_theme_icon("close_highlight", "TitleBar");
			close_btn->set_normal_texture(close);
			close_btn->set_hover_texture(close_highlight);
			close_btn->set_pressed_texture(close);
			auto maximize = get_theme_icon("maximize", "TitleBar");
			auto maximize_highlight = get_theme_icon("maximize_highlight", "TitleBar");
			maximize_btn->set_normal_texture(maximize);
			maximize_btn->set_hover_texture(maximize_highlight);
			maximize_btn->set_pressed_texture(maximize);
			auto minimize = get_theme_icon("minimize", "TitleBar");
			auto minimize_highlight = get_theme_icon("minimize_highlight", "TitleBar");
			minimize_btn->set_normal_texture(minimize);
			minimize_btn->set_hover_texture(minimize_highlight);
			minimize_btn->set_pressed_texture(minimize);

			// Reuse the logic for toggling between native and custom buttons
			set_force_custom_buttons(force_custom_buttons);
		} break;

		case NOTIFICATION_READY:
		case NOTIFICATION_RESIZED: {
			if (is_visible()) {
				_update_button_rects();
				update();
			}
		} break;
	}
}

bool TitleBar::is_forcing_custom_buttons() const {
	return force_custom_buttons;
}

void TitleBar::set_force_custom_buttons(bool p_force) {
	force_custom_buttons = p_force;

	// If the current DisplayServer doens't support native title buttons at all, the custom button states don't need to be updated
	auto ds = DisplayServer::get_singleton();
	if (ds->has_feature(DisplayServer::FEATURE_NATIVE_TITLE_BUTTONS)) {
		auto id = window->get_window_id();
		if (p_force) {
			close_btn->set_visible(ds->window_get_decoration(DisplayServer::DECORATION_CLOSE_BUTTON, id));
			maximize_btn->set_visible(ds->window_get_decoration(DisplayServer::DECORATION_MAXIMIZE_BUTTON, id));
			minimize_btn->set_visible(ds->window_get_decoration(DisplayServer::DECORATION_MINIMIZE_BUTTON, id));
			ds->window_set_decoration(DisplayServer::DECORATION_CLOSE_BUTTON, false, id);
			ds->window_set_decoration(DisplayServer::DECORATION_MAXIMIZE_BUTTON, false, id);
			ds->window_set_decoration(DisplayServer::DECORATION_MINIMIZE_BUTTON, false, id);
		} else {
			ds->window_set_decoration(DisplayServer::DECORATION_CLOSE_BUTTON, close_btn->is_visible(), id);
			ds->window_set_decoration(DisplayServer::DECORATION_MAXIMIZE_BUTTON, maximize_btn->is_visible(), id);
			ds->window_set_decoration(DisplayServer::DECORATION_MINIMIZE_BUTTON, minimize_btn->is_visible(), id);
			close_btn->set_visible(false);
			maximize_btn->set_visible(false);
			minimize_btn->set_visible(false);
		}

		_update_button_rects();
		// Update the insynchronization between window modes and buttons states, because native title buttons don't notify this title bar control to update the button states
		_update_button_textures();
		update();
	}
}

bool TitleBar::is_button_enabled(TitleButton p_button) {
	bool native = DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_TITLE_BUTTONS) && !force_custom_buttons;
	switch (p_button) {
		case BUTTON_CLOSE: {
			if (native) {
				return DisplayServer::get_singleton()->window_get_decoration(DisplayServer::DECORATION_CLOSE_BUTTON, window->get_window_id());
			} else {
				return close_btn->is_visible();
			}
		} break;
		case BUTTON_MAXIMIZE: {
			if (native) {
				return DisplayServer::get_singleton()->window_get_decoration(DisplayServer::DECORATION_MAXIMIZE_BUTTON, window->get_window_id());
			} else {
				return maximize_btn->is_visible();
			}
		} break;
		case BUTTON_MINIMIZE: {
			if (native) {
				return DisplayServer::get_singleton()->window_get_decoration(DisplayServer::DECORATION_MINIMIZE_BUTTON, window->get_window_id());
			} else {
				return minimize_btn->is_visible();
			}
		} break;
	}

	ERR_FAIL_V_MSG(false, "Invalid TitleButton enum parameter.");
}

void TitleBar::set_buttons_enabled(int p_flags, bool p_enabled) {
	bool native = DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_TITLE_BUTTONS) && !force_custom_buttons;
	bool update_rects = false;
	if ((p_flags & BUTTON_CLOSE) == BUTTON_CLOSE) {
		if (native) {
			DisplayServer::get_singleton()->window_set_decoration(DisplayServer::DECORATION_CLOSE_BUTTON, p_enabled, window->get_window_id());
		} else {
			close_btn->set_visible(p_enabled);
			update_rects = true;
		}
		update();
	}
	if ((p_flags & BUTTON_MAXIMIZE) == BUTTON_MAXIMIZE) {
		if (native) {
			DisplayServer::get_singleton()->window_set_decoration(DisplayServer::DECORATION_MAXIMIZE_BUTTON, p_enabled, window->get_window_id());
		} else {
			maximize_btn->set_visible(p_enabled);
			update_rects = true;
		}
		update();
	}
	if ((p_flags & BUTTON_MINIMIZE) == BUTTON_MINIMIZE) {
		if (native) {
			DisplayServer::get_singleton()->window_set_decoration(DisplayServer::DECORATION_MINIMIZE_BUTTON, p_enabled, window->get_window_id());
		} else {
			minimize_btn->set_visible(p_enabled);
			update_rects = true;
		}
		update();
	}

	if (update_rects) {
		_update_button_rects();
	}
}

void TitleBar::close_window() {
	emit_signal("window_closing");
	window->hide();
}

void TitleBar::maximize_window() {
	if (window->get_mode() != Window::MODE_MAXIMIZED) {
		emit_signal("window_maximizing");
		window->set_mode(Window::MODE_MAXIMIZED);
		_update_button_textures();
	}
}

void TitleBar::minimize_window() {
	if (window->get_mode() != Window::MODE_MINIMIZED) {
		emit_signal("window_minimizing");
		window->set_mode(Window::MODE_MINIMIZED);
	}
}

void TitleBar::restore_window() {
	auto mode = window->get_mode();
	if (mode != Window::MODE_WINDOWED) {
		emit_signal("window_restoring");
		window->set_mode(Window::MODE_WINDOWED);
		_update_button_textures();
	}
}

void TitleBar::bind_window(Window *p_window) {
	if (window == nullptr) {
		window = p_window;
		window->connect("title_changed", callable_mp(static_cast<CanvasItem *>(this), &CanvasItem::update));
	} else {
		ERR_PRINT("TitleBar can only be bound to one window.");
	}
}

Size2 TitleBar::get_minimum_size() const {
	Size2 btn_sizes{};
	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_TITLE_BUTTONS) && !force_custom_buttons) {
		auto rect = DisplayServer::get_singleton()->window_get_native_title_buttons_rect(window->get_window_id());
		btn_sizes = rect.get_size();
	} else {
		int btn_margin = get_theme_constant("button_margin", "TitleBar");
		if (close_btn->is_visible()) {
			Size2 close_minsize = close_btn->get_combined_minimum_size();
			btn_sizes.x += close_minsize.x + btn_margin;
			btn_sizes.y = MAX(close_minsize.y, btn_sizes.y);
		}
		if (maximize_btn->is_visible()) {
			Size2 maximize_minsize = maximize_btn->get_combined_minimum_size();
			btn_sizes.x += maximize_minsize.x + btn_margin;
			btn_sizes.y = MAX(maximize_minsize.y, btn_sizes.y);
		}
		if (minimize_btn->is_visible()) {
			Size2 minimize_minsize = minimize_btn->get_combined_minimum_size();
			btn_sizes.x += minimize_minsize.x + btn_margin;
			btn_sizes.y = MAX(minimize_minsize.y, btn_sizes.y);
		}
	}

	auto title_font = get_theme_font("title_font");
	auto title_size = title_font->get_string_size(window->get_title());

	int margin = get_theme_constant("margin", "TitleBar");
	return Size2{
		title_size.x + margin + btn_sizes.x,
		MAX(title_size.y, margin + btn_sizes.y + margin)
	};
}

TitleBar::TitleBar() {
	close_btn = memnew(TextureButton);
	close_btn->connect("pressed", callable_mp(this, &TitleBar::_close_pressed));
	add_child(close_btn);
	maximize_btn = memnew(TextureButton);
	maximize_btn->connect("pressed", callable_mp(this, &TitleBar::_maximize_pressed));
	add_child(maximize_btn);
	minimize_btn = memnew(TextureButton);
	minimize_btn->connect("pressed", callable_mp(this, &TitleBar::_minimize_pressed));
	add_child(minimize_btn);

	set_process_input(true);
}

TitleBar::~TitleBar() {
}
