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

void TitleBar::_close_pressed() {
	close_window();
}

void TitleBar::_maximize_pressed() {
	switch (window->get_mode()) {
		case Window::MODE_WINDOWED: {
			maximize_window();
		} break;
		case Window::MODE_MAXIMIZED: {
			restore_window();
		} break;
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
	ClassDB::bind_method("get_close_button", &TitleBar::get_close_button);
	ClassDB::bind_method("get_maximize_button", &TitleBar::get_maximize_button);
	ClassDB::bind_method("get_minimize_button", &TitleBar::get_minimize_button);
	ClassDB::bind_method("close_window", &TitleBar::close_window);
	ClassDB::bind_method("maximize_window", &TitleBar::maximize_window);
	ClassDB::bind_method("minimize_window", &TitleBar::minimize_window);
	ClassDB::bind_method("restore_window", &TitleBar::restore_window);

	BIND_ENUM_CONSTANT(ALIGN_LEFT);
	BIND_ENUM_CONSTANT(ALIGN_CENTER);
	BIND_ENUM_CONSTANT(ALIGN_RIGHT);
	BIND_ENUM_CONSTANT(BUTTON_CLOSE);
	BIND_ENUM_CONSTANT(BUTTON_MAXIMIZE);
	BIND_ENUM_CONSTANT(BUTTON_MINIMIZE);

	ADD_SIGNAL(MethodInfo("close_window"));
	ADD_SIGNAL(MethodInfo("maximize_window"));
	ADD_SIGNAL(MethodInfo("minimize_window"));
	ADD_SIGNAL(MethodInfo("restore_window"));
}

void TitleBar::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_DRAW: {
			auto canvas_item = get_canvas_item();
			float margin = get_theme_constant("margin", "TitleBar");
			auto abs_pos = get_global_position();
			auto size = get_size();
			auto title = get_title();
			auto title_font = get_theme_font("title_font", "TitleBar");
			auto title_color = get_theme_color("title_color", "TitleBar");
			float font_height = title_font->get_height() - title_font->get_descent() * 2;
			float y = abs_pos.y + (size.y + font_height) / 2;

			switch (title_align) {
				case ALIGN_LEFT: {
					float x = abs_pos.x + margin;
					title_font->draw(canvas_item, Point2{ x, y }, title, title_color);
				} break;
				case ALIGN_CENTER: {
					float x = abs_pos.x + (size.x - title_font->get_string_size(title).x) / 2;
					title_font->draw(canvas_item, Point2{ x, y }, title, title_color);
				} break;
				case ALIGN_RIGHT: {
					int btn_margin = get_theme_constant("button_margin", "TitleBar");
					float btn_width = 0.0f;
					if (close_btn->is_visible()) {
						btn_width += close_btn->get_combined_minimum_size().x + btn_margin;
					}
					if (maximize_btn->is_visible()) {
						btn_width += maximize_btn->get_combined_minimum_size().x + btn_margin;
					}
					if (minimize_btn->is_visible()) {
						btn_width += minimize_btn->get_combined_minimum_size().x + btn_margin;
					}

					float x = abs_pos.x + size.x - margin - btn_width - margin - title_font->get_string_size(title).x;
					title_font->draw(canvas_item, Point2{ x, y }, title, title_color);
				} break;
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

			_update_button_rects();
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

String TitleBar::get_title() const {
	return window->get_title();
}

void TitleBar::set_title(const String &p_title) {
	window->set_title(p_title);
	update();
}

TitleBar::TitleAlign TitleBar::get_title_align() const {
	return title_align;
}

void TitleBar::set_title_align(TitleAlign p_align) {
	title_align = p_align;
	update();
}

void TitleBar::bind_window(Window *p_window) {
	if (window == nullptr) {
		window = p_window;
	} else {
		ERR_PRINT("TitleBar can only be bound to one window.");
	}
}

void TitleBar::bind_default_behaviors(int p_flags) {
	if (!has_default_behaviors) {
		if ((p_flags & BUTTON_CLOSE) == BUTTON_CLOSE) {
			close_btn->connect("pressed", callable_mp(this, &TitleBar::_close_pressed));
		}
		if ((p_flags & BUTTON_MAXIMIZE) == BUTTON_MAXIMIZE) {
			maximize_btn->connect("pressed", callable_mp(this, &TitleBar::_maximize_pressed));
		}
		if ((p_flags & BUTTON_MINIMIZE) == BUTTON_MINIMIZE) {
			minimize_btn->connect("pressed", callable_mp(this, &TitleBar::_minimize_pressed));
		}
		has_default_behaviors = true;
	}
}

void TitleBar::close_window() {
	window->hide();
	emit_signal("close_window");
}

void TitleBar::maximize_window() {
	if (window->get_mode() != Window::MODE_MAXIMIZED) {
		auto restore = get_theme_icon("restore", "TitleBar");
		auto restore_highlight = get_theme_icon("restore_highlight", "TitleBar");
		maximize_btn->set_normal_texture(restore);
		maximize_btn->set_hover_texture(restore_highlight);
		maximize_btn->set_pressed_texture(restore);

		window->set_mode(Window::MODE_MAXIMIZED);
		emit_signal("maximize_window");
	}
}

void TitleBar::minimize_window() {
	if (window->get_mode() != Window::MODE_MINIMIZED) {
		window->set_mode(Window::MODE_MINIMIZED);
		emit_signal("minimize_window");
	}
}

void TitleBar::restore_window() {
	auto mode = window->get_mode();
	if (mode == Window::MODE_MAXIMIZED) {
		auto maximize = get_theme_icon("maximize", "TitleBar");
		auto maximize_highlight = get_theme_icon("maximize_highlight", "TitleBar");
		maximize_btn->set_normal_texture(maximize);
		maximize_btn->set_hover_texture(maximize_highlight);
		maximize_btn->set_pressed_texture(maximize);

		window->set_mode(Window::MODE_WINDOWED);
		emit_signal("restore_window");
	} else if (mode == Window::MODE_MINIMIZED) {
		window->set_mode(Window::MODE_WINDOWED);
		emit_signal("restore_window");
	}
}

Size2 TitleBar::get_minimum_size() const {
	int btn_margin = get_theme_constant("button_margin", "TitleBar");
	Size2 btn_sizes{};
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

	auto title = get_title();
	auto title_font = get_theme_font("title_font");
	auto title_size = title_font->get_string_size(title);

	int margin = get_theme_constant("margin", "TitleBar");
	return Size2{
		title_size.x + margin + btn_sizes.x,
		MAX(title_size.y, margin + btn_sizes.y + margin)
	};
}

TitleBar::TitleBar() {
	set_h_size_flags(SIZE_EXPAND);

	close_btn = memnew(TextureButton);
	close_btn->connect("visibility_changed", callable_mp(this, &TitleBar::_update_button_rects));
	add_child(close_btn);
	maximize_btn = memnew(TextureButton);
	maximize_btn->connect("visibility_changed", callable_mp(this, &TitleBar::_update_button_rects));
	add_child(maximize_btn);
	minimize_btn = memnew(TextureButton);
	minimize_btn->connect("visibility_changed", callable_mp(this, &TitleBar::_update_button_rects));
	add_child(minimize_btn);

	set_process_input(true);
}

TitleBar::~TitleBar() {
}
