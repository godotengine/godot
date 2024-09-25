/**************************************************************************/
/*  embedded_process.cpp                                                  */
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

#include "embedded_process.h"
#include "editor/editor_string_names.h"
#include "scene/main/window.h"
#include "scene/resources/style_box_flat.h"
#include "scene/theme/theme_db.h"

void EmbeddedProcess::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_window = get_window();
		} break;
		case NOTIFICATION_PROCESS: {
			_check_focused_process_id();
			_check_mouse_over();
		} break;
		case NOTIFICATION_DRAW: {
			_draw();
		} break;
		case NOTIFICATION_RESIZED:
		case NOTIFICATION_VISIBILITY_CHANGED:
		case NOTIFICATION_WM_POSITION_CHANGED: {
			_queue_update_embedded_process();
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_focus_style_box = get_theme_stylebox(SNAME("FocusViewport"), EditorStringName(EditorStyles));
			Ref<StyleBoxFlat> focus_style_box_flat = _focus_style_box;
			if (focus_style_box_flat.is_valid()) {
				_margin_top_left = Point2i(focus_style_box_flat->get_corner_radius(CORNER_TOP_LEFT), focus_style_box_flat->get_corner_radius(CORNER_TOP_LEFT));
				_margin_bottom_right = Point2i(focus_style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT), focus_style_box_flat->get_corner_radius(CORNER_BOTTOM_RIGHT));
			} else if (_focus_style_box.is_valid()) {
				_margin_top_left = Point2i(_focus_style_box->get_margin(SIDE_LEFT), _focus_style_box->get_margin(SIDE_TOP));
				_margin_bottom_right = Point2i(_focus_style_box->get_margin(SIDE_RIGHT), _focus_style_box->get_margin(SIDE_BOTTOM));
			} else {
				_margin_top_left = Point2i();
				_margin_bottom_right = Point2i();
			}
		} break;
		case NOTIFICATION_FOCUS_ENTER: {
			_queue_update_embedded_process();
		} break;
		case NOTIFICATION_APPLICATION_FOCUS_IN: {
			_application_has_focus = true;
			if (_embedded_process_was_focused) {
				_embedded_process_was_focused = false;
				// Refocus the embedded process if it was focused when the application lost focus,
				// but do not refocus if the embedded process is currently focused (indicating it just lost focus)
				// or if the current window is a different popup or secondary window.
				if (_embedding_completed && _current_process_id != _focused_process_id && _window && _window->has_focus()) {
					grab_focus();
					_queue_update_embedded_process();
				}
			}
		} break;
		case NOTIFICATION_APPLICATION_FOCUS_OUT: {
			_application_has_focus = false;
			_embedded_process_was_focused = _embedding_completed && _current_process_id == _focused_process_id;
		} break;
	}
}

void EmbeddedProcess::set_embedding_timeout(int p_timeout) {
	_embedding_timeout = p_timeout;
}

int EmbeddedProcess::get_embedding_timeout() {
	return _embedding_timeout;
}

void EmbeddedProcess::set_window_size(Size2i p_window_size) {
	if (_window_size != p_window_size) {
		_window_size = p_window_size;
		_queue_update_embedded_process();
	}
}

Size2i EmbeddedProcess::get_window_size() {
	return _window_size;
}

void EmbeddedProcess::set_keep_aspect(bool p_keep_aspect) {
	if (_keep_aspect != p_keep_aspect) {
		_keep_aspect = p_keep_aspect;
		_queue_update_embedded_process();
	}
}

bool EmbeddedProcess::get_keep_aspect() {
	return _keep_aspect;
}

Rect2i EmbeddedProcess::get_global_embedded_window_rect() {
	Rect2i control_rect = this->get_global_rect();
	control_rect = Rect2i(control_rect.position, Size2i(MAX(control_rect.size.x, 1), MAX(control_rect.size.y, 1)));
	if (_keep_aspect) {
		Rect2i desired_rect = control_rect;
		float ratio = MIN((float)control_rect.size.x / _window_size.x, (float)control_rect.size.y / _window_size.y);
		desired_rect.size = Size2i(MAX(_window_size.x * ratio, 1), MAX(_window_size.y * ratio, 1));
		desired_rect.position = Size2i(control_rect.position.x + ((control_rect.size.x - desired_rect.size.x) / 2), control_rect.position.y + ((control_rect.size.y - desired_rect.size.y) / 2));
		return desired_rect;
	} else {
		return control_rect;
	}
}

Rect2i EmbeddedProcess::get_screen_embedded_window_rect() {
	Rect2i rect = get_global_embedded_window_rect();
	if (_window) {
		rect.position += _window->get_position();
	}

	// Removing margins to make space for the focus border style.
	return Rect2i(rect.position.x + _margin_top_left.x, rect.position.y + _margin_top_left.y, MAX(rect.size.x - (_margin_top_left.x + _margin_bottom_right.x), 1), MAX(rect.size.y - (_margin_top_left.y + _margin_bottom_right.y), 1));
}

bool EmbeddedProcess::is_embedding_in_progress() {
	return !_timer_embedding->is_stopped();
}

bool EmbeddedProcess::is_embedding_completed() {
	return _embedding_completed;
}

void EmbeddedProcess::embed_process(OS::ProcessID p_pid) {
	if (!_window) {
		return;
	}

	ERR_FAIL_COND_MSG(!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_WINDOW_EMBEDDING), "Embedded process not supported by this display server.");

	if (_current_process_id != 0) {
		// Stop embedding the last process.
		OS::get_singleton()->kill(_current_process_id);
	}

	reset();

	_current_process_id = p_pid;
	_start_embedding_time = OS::get_singleton()->get_ticks_msec();
	_embedding_grab_focus = has_focus();
	set_process(true);

	// Attempt to embed the process, but if it has just started and the window is not ready yet,
	// we will retry in this case.
	_try_embed_process();
}

void EmbeddedProcess::reset() {
	if (_current_process_id != 0 && _embedding_completed) {
		DisplayServer::get_singleton()->remove_embedded_process(_current_process_id);
	}
	_current_process_id = 0;
	_embedding_completed = false;
	_start_embedding_time = 0;
	_embedding_grab_focus = false;
	_timer_embedding->stop();
	set_process(false);
	queue_redraw();
}

void EmbeddedProcess::_try_embed_process() {
	Error err = DisplayServer::get_singleton()->embed_process(_window->get_window_id(), _current_process_id, get_screen_embedded_window_rect(), is_visible_in_tree(), _embedding_grab_focus);
	if (err == OK) {
		_embedding_completed = true;
		queue_redraw();
		emit_signal(SNAME("embedding_completed"));
	} else if (err == ERR_DOES_NOT_EXIST) {
		if (OS::get_singleton()->get_ticks_msec() - _start_embedding_time >= (uint64_t)_embedding_timeout) {
			// Embedding process timed out.
			reset();
			emit_signal(SNAME("embedding_failed"));
		} else {
			// Tries another shot.
			_timer_embedding->start();
		}
	} else {
		// Another unknown error.
		reset();
		emit_signal(SNAME("embedding_failed"));
	}
}

bool EmbeddedProcess::_is_embedded_process_updatable() {
	return _window && _current_process_id != 0 && _embedding_completed;
}

void EmbeddedProcess::_queue_update_embedded_process() {
	if (_updated_embedded_process_queued || !_is_embedded_process_updatable()) {
		return;
	}

	_updated_embedded_process_queued = true;

	callable_mp(this, &EmbeddedProcess::_update_embedded_process).call_deferred();
}

void EmbeddedProcess::_update_embedded_process() {
	_updated_embedded_process_queued = false;

	if (!_is_embedded_process_updatable()) {
		return;
	}

	bool must_grab_focus = false;
	bool focus = has_focus();
	if (_last_updated_embedded_process_focused != focus) {
		if (focus) {
			must_grab_focus = true;
		}
		_last_updated_embedded_process_focused = focus;
	}

	DisplayServer::get_singleton()->embed_process(_window->get_window_id(), _current_process_id, get_screen_embedded_window_rect(), is_visible_in_tree(), must_grab_focus);
}

void EmbeddedProcess::_timer_embedding_timeout() {
	_try_embed_process();
}

void EmbeddedProcess::_draw() {
	if (_focused_process_id == _current_process_id && has_focus() && _focus_style_box.is_valid()) {
		Size2 size = get_size();
		Rect2 r = Rect2(Point2(), size);
		_focus_style_box->draw(get_canvas_item(), r);
	}
}

void EmbeddedProcess::_check_mouse_over() {
	// This method checks if the mouse is over the embedded process while the current application is focused.
	// The goal is to give focus to the embedded process as soon as the mouse hovers over it,
	// allowing the user to interact with it immediately without needing to click first.
	if (!is_visible_in_tree() || !_embedding_completed || !_application_has_focus || !_window || !_window->has_focus()) {
		return;
	}

	bool focused = has_focus();

	// Not stealing focus from a textfield.
	if (!focused && get_viewport()->gui_get_focus_owner() && get_viewport()->gui_get_focus_owner()->is_text_field()) {
		return;
	}

	Vector2 mouse_position = DisplayServer::get_singleton()->mouse_get_position();
	Rect2i window_rect = get_screen_embedded_window_rect();
	if (!window_rect.has_point(mouse_position)) {
		return;
	}

	// Don't grab the focus if mouse over another window.
	DisplayServer::WindowID window_id_over = DisplayServer::get_singleton()->get_window_at_screen_position(mouse_position);
	if (window_id_over > 0 && window_id_over != _window->get_window_id()) {
		return;
	}

	// When we already have the focus and the user moves the mouse over the embedded process,
	// we just need to refocus the process.
	if (focused) {
		_queue_update_embedded_process();
	} else {
		grab_focus();
		queue_redraw();
	}
}

void EmbeddedProcess::_check_focused_process_id() {
	OS::ProcessID process_id = DisplayServer::get_singleton()->get_focused_process_id();
	if (process_id != _focused_process_id) {
		_focused_process_id = process_id;
		if (_focused_process_id == _current_process_id) {
			// The embedded process got the focus.
			if (has_focus()) {
				// Redraw to updated the focus style.
				queue_redraw();
			} else {
				grab_focus();
			}
		} else if (has_focus()) {
			release_focus();
		}
	}
}

void EmbeddedProcess::_bind_methods() {
	ClassDB::bind_method(D_METHOD("embed_process", "process_id"), &EmbeddedProcess::embed_process);
	ClassDB::bind_method(D_METHOD("reset"), &EmbeddedProcess::reset);
	ClassDB::bind_method(D_METHOD("set_embedding_timeout", "timeout"), &EmbeddedProcess::set_embedding_timeout);
	ClassDB::bind_method(D_METHOD("get_embedding_timeout"), &EmbeddedProcess::get_embedding_timeout);
	ClassDB::bind_method(D_METHOD("is_embedding_completed"), &EmbeddedProcess::is_embedding_completed);
	ClassDB::bind_method(D_METHOD("is_embedding_in_progress"), &EmbeddedProcess::is_embedding_in_progress);

	ADD_SIGNAL(MethodInfo("embedding_completed"));
	ADD_SIGNAL(MethodInfo("embedding_failed"));
}

EmbeddedProcess::EmbeddedProcess() {
	_timer_embedding = memnew(Timer);
	_timer_embedding->set_wait_time(0.1);
	_timer_embedding->set_one_shot(true);
	add_child(_timer_embedding);
	_timer_embedding->connect("timeout", callable_mp(this, &EmbeddedProcess::_timer_embedding_timeout));
	set_focus_mode(FOCUS_ALL);
}

EmbeddedProcess::~EmbeddedProcess() {
	if (_current_process_id != 0) {
		// Stop embedding the last process.
		OS::get_singleton()->kill(_current_process_id);
		reset();
	}
}
