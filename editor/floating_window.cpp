#include "floating_window.h"

void FloatingWindow::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
		this->connect("window_input", callable_mp(this, &FloatingWindow::_on_window_event));
		} break;
	}
};

void FloatingWindow::_on_window_event(const Ref<InputEvent> &p_event) {
	bool single_window_mode = EditorSettings::get_singleton()->get_setting(
			"interface/editor/single_window_mode");
	if (single_window_mode) {
		Size2i editor_window_size = get_parent_rect().size;
		Vector2 window_position = this->get_position();
		print_line("Window position:");
		print_line(window_position);
		print_line("Size");
		print_line(editor_window_size);
		Rect2 window_size = this->get_visible_rect();
		if (editor_window_size.x < (window_position.x + window_size.size.x)) {
			this->set_position(Vector2((editor_window_size.x - window_size.size.x), window_position.y));
		} else if (window_position.x < 0) {
			this->set_position(Vector2(0, window_position.y));
		}
		if (window_position.y < 0) {
			this->set_position(Vector2(window_position.x, 20));
		} else if (editor_window_size.y < window_position.y) {
			this->set_position(Vector2(window_position.x, editor_window_size.y));
		}
	}
};
