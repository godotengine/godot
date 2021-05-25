/*************************************************************************/
/*  window.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "window.h"

#include "core/debugger/engine_debugger.h"
#include "core/os/keyboard.h"
#include "core/string/translation.h"
#include "scene/gui/control.h"
#include "scene/resources/font.h"
#include "scene/scene_string_names.h"

void Window::set_title(const String &p_title) {
	title = p_title;

	if (embedder) {
		embedder->_sub_window_update(this);

	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_title(p_title, window_id);
	}
}

String Window::get_title() const {
	return title;
}

void Window::set_current_screen(int p_screen) {
	current_screen = p_screen;
	if (window_id == DisplayServer::INVALID_WINDOW_ID) {
		return;
	}
	DisplayServer::get_singleton()->window_set_current_screen(p_screen, window_id);
}

int Window::get_current_screen() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		current_screen = DisplayServer::get_singleton()->window_get_current_screen(window_id);
	}
	return current_screen;
}

void Window::set_position(const Point2i &p_position) {
	position = p_position;

	if (embedder) {
		embedder->_sub_window_update(this);

	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_position(p_position, window_id);
	}
}

Point2i Window::get_position() const {
	return position;
}

void Window::set_size(const Size2i &p_size) {
	size = p_size;
	_update_window_size();
}

Size2i Window::get_size() const {
	return size;
}

Size2i Window::get_real_size() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		return DisplayServer::get_singleton()->window_get_real_size(window_id);
	}
	return size;
}

void Window::set_max_size(const Size2i &p_max_size) {
	max_size = p_max_size;
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_max_size(max_size, window_id);
	}
	_update_window_size();
}

Size2i Window::get_max_size() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		max_size = DisplayServer::get_singleton()->window_get_max_size(window_id);
	}
	return max_size;
}

void Window::set_min_size(const Size2i &p_min_size) {
	min_size = p_min_size;
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_min_size(min_size, window_id);
	}
	_update_window_size();
}

Size2i Window::get_min_size() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		min_size = DisplayServer::get_singleton()->window_get_min_size(window_id);
	}
	return min_size;
}

void Window::set_mode(Mode p_mode) {
	mode = p_mode;

	if (embedder) {
		embedder->_sub_window_update(this);

	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_mode(DisplayServer::WindowMode(p_mode), window_id);
	}
}

Window::Mode Window::get_mode() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		mode = (Mode)DisplayServer::get_singleton()->window_get_mode(window_id);
	}
	return mode;
}

void Window::set_flag(Flags p_flag, bool p_enabled) {
	ERR_FAIL_INDEX(p_flag, FLAG_MAX);
	flags[p_flag] = p_enabled;

	if (embedder) {
		embedder->_sub_window_update(this);

	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_flag(DisplayServer::WindowFlags(p_flag), p_enabled, window_id);
	}
}

bool Window::get_flag(Flags p_flag) const {
	ERR_FAIL_INDEX_V(p_flag, FLAG_MAX, false);
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		flags[p_flag] = DisplayServer::get_singleton()->window_get_flag(DisplayServer::WindowFlags(p_flag), window_id);
	}
	return flags[p_flag];
}

bool Window::is_maximize_allowed() const {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		return DisplayServer::get_singleton()->window_is_maximize_allowed(window_id);
	}
	return true;
}

void Window::request_attention() {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_request_attention(window_id);
	}
}

void Window::move_to_foreground() {
	if (embedder) {
		embedder->_sub_window_grab_focus(this);

	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_move_to_foreground(window_id);
	}
}

bool Window::can_draw() const {
	if (!is_inside_tree()) {
		return false;
	}
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		return DisplayServer::get_singleton()->window_can_draw(window_id);
	}

	return visible;
}

void Window::set_ime_active(bool p_active) {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_ime_active(p_active, window_id);
	}
}

void Window::set_ime_position(const Point2i &p_pos) {
	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_ime_position(p_pos, window_id);
	}
}

bool Window::is_embedded() const {
	ERR_FAIL_COND_V(!is_inside_tree(), false);

	return _get_embedder() != nullptr;
}

void Window::_make_window() {
	ERR_FAIL_COND(window_id != DisplayServer::INVALID_WINDOW_ID);

	uint32_t f = 0;
	for (int i = 0; i < FLAG_MAX; i++) {
		if (flags[i]) {
			f |= (1 << i);
		}
	}

	window_id = DisplayServer::get_singleton()->create_sub_window(DisplayServer::WindowMode(mode), f, Rect2i(position, size));
	ERR_FAIL_COND(window_id == DisplayServer::INVALID_WINDOW_ID);
	DisplayServer::get_singleton()->window_set_current_screen(current_screen, window_id);
	DisplayServer::get_singleton()->window_set_max_size(max_size, window_id);
	DisplayServer::get_singleton()->window_set_min_size(min_size, window_id);
	DisplayServer::get_singleton()->window_set_title(tr(title), window_id);
	DisplayServer::get_singleton()->window_attach_instance_id(get_instance_id(), window_id);

	_update_window_size();

	if (transient_parent && transient_parent->window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_transient(window_id, transient_parent->window_id);
	}

	for (Set<Window *>::Element *E = transient_children.front(); E; E = E->next()) {
		if (E->get()->window_id != DisplayServer::INVALID_WINDOW_ID) {
			DisplayServer::get_singleton()->window_set_transient(E->get()->window_id, transient_parent->window_id);
		}
	}

	_update_window_callbacks();

	RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_WHEN_VISIBLE);
	DisplayServer::get_singleton()->show_window(window_id);
}

void Window::_update_from_window() {
	ERR_FAIL_COND(window_id == DisplayServer::INVALID_WINDOW_ID);
	mode = (Mode)DisplayServer::get_singleton()->window_get_mode(window_id);
	for (int i = 0; i < FLAG_MAX; i++) {
		flags[i] = DisplayServer::get_singleton()->window_get_flag(DisplayServer::WindowFlags(i), window_id);
	}
}

void Window::_clear_window() {
	ERR_FAIL_COND(window_id == DisplayServer::INVALID_WINDOW_ID);

	if (transient_parent && transient_parent->window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_transient(window_id, DisplayServer::INVALID_WINDOW_ID);
	}

	for (Set<Window *>::Element *E = transient_children.front(); E; E = E->next()) {
		if (E->get()->window_id != DisplayServer::INVALID_WINDOW_ID) {
			DisplayServer::get_singleton()->window_set_transient(E->get()->window_id, DisplayServer::INVALID_WINDOW_ID);
		}
	}

	_update_from_window();

	DisplayServer::get_singleton()->delete_sub_window(window_id);
	window_id = DisplayServer::INVALID_WINDOW_ID;

	_update_viewport_size();
	RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_DISABLED);
}

void Window::_rect_changed_callback(const Rect2i &p_callback) {
	//we must always accept this as the truth
	if (size == p_callback.size && position == p_callback.position) {
		return;
	}
	position = p_callback.position;

	if (size != p_callback.size) {
		size = p_callback.size;
		_update_viewport_size();
	}
}

void Window::_propagate_window_notification(Node *p_node, int p_notification) {
	p_node->notification(p_notification);
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		Window *window = Object::cast_to<Window>(child);
		if (window) {
			break;
		}
		_propagate_window_notification(child, p_notification);
	}
}

void Window::_event_callback(DisplayServer::WindowEvent p_event) {
	switch (p_event) {
		case DisplayServer::WINDOW_EVENT_MOUSE_ENTER: {
			_propagate_window_notification(this, NOTIFICATION_WM_MOUSE_ENTER);
			emit_signal("mouse_entered");
			DisplayServer::get_singleton()->cursor_set_shape(DisplayServer::CURSOR_ARROW); //restore cursor shape
		} break;
		case DisplayServer::WINDOW_EVENT_MOUSE_EXIT: {
			_propagate_window_notification(this, NOTIFICATION_WM_MOUSE_EXIT);
			emit_signal("mouse_exited");
		} break;
		case DisplayServer::WINDOW_EVENT_FOCUS_IN: {
			focused = true;
			_propagate_window_notification(this, NOTIFICATION_WM_WINDOW_FOCUS_IN);
			emit_signal("focus_entered");

		} break;
		case DisplayServer::WINDOW_EVENT_FOCUS_OUT: {
			focused = false;
			_propagate_window_notification(this, NOTIFICATION_WM_WINDOW_FOCUS_OUT);
			emit_signal("focus_exited");
		} break;
		case DisplayServer::WINDOW_EVENT_CLOSE_REQUEST: {
			if (exclusive_child != nullptr) {
				break; //has an exclusive child, can't get events until child is closed
			}
			_propagate_window_notification(this, NOTIFICATION_WM_CLOSE_REQUEST);
			emit_signal("close_requested");
		} break;
		case DisplayServer::WINDOW_EVENT_GO_BACK_REQUEST: {
			_propagate_window_notification(this, NOTIFICATION_WM_GO_BACK_REQUEST);
			emit_signal("go_back_requested");
		} break;
		case DisplayServer::WINDOW_EVENT_DPI_CHANGE: {
			_update_viewport_size();
			_propagate_window_notification(this, NOTIFICATION_WM_DPI_CHANGE);
			emit_signal("dpi_changed");
		} break;
	}
}

void Window::show() {
	set_visible(true);
}

void Window::hide() {
	set_visible(false);
}

void Window::set_visible(bool p_visible) {
	if (visible == p_visible) {
		return;
	}

	visible = p_visible;

	if (!is_inside_tree()) {
		return;
	}

	if (updating_child_controls) {
		_update_child_controls();
	}

	ERR_FAIL_COND_MSG(get_parent() == nullptr, "Can't change visibility of main window.");

	Viewport *embedder_vp = _get_embedder();

	if (!embedder_vp) {
		if (!p_visible && window_id != DisplayServer::INVALID_WINDOW_ID) {
			_clear_window();
		}
		if (p_visible && window_id == DisplayServer::INVALID_WINDOW_ID) {
			_make_window();
		}
	} else {
		if (visible) {
			embedder = embedder_vp;
			embedder->_sub_window_register(this);
			RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_WHEN_PARENT_VISIBLE);
		} else {
			embedder->_sub_window_remove(this);
			embedder = nullptr;
			RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_DISABLED);
		}
		_update_window_size();
	}

	if (!visible) {
		focused = false;
	}
	notification(NOTIFICATION_VISIBILITY_CHANGED);
	emit_signal(SceneStringNames::get_singleton()->visibility_changed);

	RS::get_singleton()->viewport_set_active(get_viewport_rid(), visible);

	//update transient exclusive
	if (transient_parent) {
		if (exclusive && visible) {
			ERR_FAIL_COND_MSG(transient_parent->exclusive_child && transient_parent->exclusive_child != this, "Transient parent has another exclusive child.");
			transient_parent->exclusive_child = this;
		} else {
			if (transient_parent->exclusive_child == this) {
				transient_parent->exclusive_child = nullptr;
			}
		}
	}
}

void Window::_clear_transient() {
	if (transient_parent) {
		if (transient_parent->window_id != DisplayServer::INVALID_WINDOW_ID && window_id != DisplayServer::INVALID_WINDOW_ID) {
			DisplayServer::get_singleton()->window_set_transient(window_id, DisplayServer::INVALID_WINDOW_ID);
		}
		transient_parent->transient_children.erase(this);
		if (transient_parent->exclusive_child == this) {
			transient_parent->exclusive_child = nullptr;
		}
		transient_parent = nullptr;
	}
}

void Window::_make_transient() {
	if (!get_parent()) {
		//main window, can't be transient
		return;
	}
	//find transient parent
	Viewport *vp = get_parent()->get_viewport();
	Window *window = nullptr;
	while (vp) {
		window = Object::cast_to<Window>(vp);
		if (window) {
			break;
		}
		if (!vp->get_parent()) {
			break;
		}

		vp = vp->get_parent()->get_viewport();
	}

	if (window) {
		transient_parent = window;
		window->transient_children.insert(this);
		if (is_inside_tree() && is_visible() && exclusive) {
			if (transient_parent->exclusive_child == nullptr) {
				transient_parent->exclusive_child = this;
			} else if (transient_parent->exclusive_child != this) {
				ERR_PRINT("Making child transient exclusive, but parent has another exclusive child");
			}
		}
	}

	//see if we can make transient
	if (transient_parent->window_id != DisplayServer::INVALID_WINDOW_ID && window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_transient(window_id, transient_parent->window_id);
	}
}

void Window::set_transient(bool p_transient) {
	if (transient == p_transient) {
		return;
	}

	transient = p_transient;

	if (!is_inside_tree()) {
		return;
	}

	if (transient) {
		_make_transient();
	} else {
		_clear_transient();
	}
}

bool Window::is_transient() const {
	return transient;
}

void Window::set_exclusive(bool p_exclusive) {
	if (exclusive == p_exclusive) {
		return;
	}

	exclusive = p_exclusive;

	if (transient_parent) {
		if (p_exclusive && is_inside_tree() && is_visible()) {
			ERR_FAIL_COND_MSG(transient_parent->exclusive_child && transient_parent->exclusive_child != this, "Transient parent has another exclusive child.");
			transient_parent->exclusive_child = this;
		} else {
			if (transient_parent->exclusive_child == this) {
				transient_parent->exclusive_child = nullptr;
			}
		}
	}
}

bool Window::is_exclusive() const {
	return exclusive;
}

bool Window::is_visible() const {
	return visible;
}

void Window::_update_window_size() {
	Size2i size_limit;
	if (wrap_controls) {
		size_limit = get_contents_minimum_size();
	}

	size_limit.x = MAX(size_limit.x, min_size.x);
	size_limit.y = MAX(size_limit.y, min_size.y);

	size.x = MAX(size_limit.x, size.x);
	size.y = MAX(size_limit.y, size.y);

	if (max_size.x > 0 && max_size.x > min_size.x && size.x > max_size.x) {
		size.x = max_size.x;
	}

	if (max_size.y > 0 && max_size.y > min_size.y && size.y > max_size.y) {
		size.y = max_size.y;
	}

	if (embedder) {
		embedder->_sub_window_update(this);
	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_set_size(size, window_id);
	}

	//update the viewport
	_update_viewport_size();
}

void Window::_update_viewport_size() {
	//update the viewport part

	Size2i final_size;
	Size2i final_size_override;
	Rect2i attach_to_screen_rect(Point2i(), size);
	Transform2D stretch_transform;
	float font_oversampling = 1.0;

	if (content_scale_mode == CONTENT_SCALE_MODE_DISABLED || content_scale_size.x == 0 || content_scale_size.y == 0) {
		stretch_transform = Transform2D();
		final_size = size;

	} else {
		//actual screen video mode
		Size2 video_mode = size;
		Size2 desired_res = content_scale_size;

		Size2 viewport_size;
		Size2 screen_size;

		float viewport_aspect = desired_res.aspect();
		float video_mode_aspect = video_mode.aspect();

		if (content_scale_aspect == CONTENT_SCALE_ASPECT_IGNORE || Math::is_equal_approx(viewport_aspect, video_mode_aspect)) {
			//same aspect or ignore aspect
			viewport_size = desired_res;
			screen_size = video_mode;
		} else if (viewport_aspect < video_mode_aspect) {
			// screen ratio is smaller vertically

			if (content_scale_aspect == CONTENT_SCALE_ASPECT_KEEP_HEIGHT || content_scale_aspect == CONTENT_SCALE_ASPECT_EXPAND) {
				//will stretch horizontally
				viewport_size.x = desired_res.y * video_mode_aspect;
				viewport_size.y = desired_res.y;
				screen_size = video_mode;

			} else {
				//will need black bars
				viewport_size = desired_res;
				screen_size.x = video_mode.y * viewport_aspect;
				screen_size.y = video_mode.y;
			}
		} else {
			//screen ratio is smaller horizontally
			if (content_scale_aspect == CONTENT_SCALE_ASPECT_KEEP_WIDTH || content_scale_aspect == CONTENT_SCALE_ASPECT_EXPAND) {
				//will stretch horizontally
				viewport_size.x = desired_res.x;
				viewport_size.y = desired_res.x / video_mode_aspect;
				screen_size = video_mode;

			} else {
				//will need black bars
				viewport_size = desired_res;
				screen_size.x = video_mode.x;
				screen_size.y = video_mode.x / viewport_aspect;
			}
		}

		screen_size = screen_size.floor();
		viewport_size = viewport_size.floor();

		Size2 margin;
		Size2 offset;
		//black bars and margin
		if (content_scale_aspect != CONTENT_SCALE_ASPECT_EXPAND && screen_size.x < video_mode.x) {
			margin.x = Math::round((video_mode.x - screen_size.x) / 2.0);
			//RenderingServer::get_singleton()->black_bars_set_margins(margin.x, 0, margin.x, 0);
			offset.x = Math::round(margin.x * viewport_size.y / screen_size.y);
		} else if (content_scale_aspect != CONTENT_SCALE_ASPECT_EXPAND && screen_size.y < video_mode.y) {
			margin.y = Math::round((video_mode.y - screen_size.y) / 2.0);
			//RenderingServer::get_singleton()->black_bars_set_margins(0, margin.y, 0, margin.y);
			offset.y = Math::round(margin.y * viewport_size.x / screen_size.x);
		} else {
			//RenderingServer::get_singleton()->black_bars_set_margins(0, 0, 0, 0);
		}

		switch (content_scale_mode) {
			case CONTENT_SCALE_MODE_DISABLED: {
				// Already handled above
				//_update_font_oversampling(1.0);
			} break;
			case CONTENT_SCALE_MODE_CANVAS_ITEMS: {
				final_size = screen_size;
				final_size_override = viewport_size;
				attach_to_screen_rect = Rect2(margin, screen_size);
				font_oversampling = screen_size.x / viewport_size.x;

				Size2 scale = Vector2(screen_size) / Vector2(final_size_override);
				stretch_transform.scale(scale);

			} break;
			case CONTENT_SCALE_MODE_VIEWPORT: {
				final_size = viewport_size;
				attach_to_screen_rect = Rect2(margin, screen_size);

			} break;
		}
	}

	bool allocate = is_inside_tree() && visible && (window_id != DisplayServer::INVALID_WINDOW_ID || embedder != nullptr);
	_set_size(final_size, final_size_override, attach_to_screen_rect, stretch_transform, allocate);

	if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		RenderingServer::get_singleton()->viewport_attach_to_screen(get_viewport_rid(), attach_to_screen_rect, window_id);
	} else {
		RenderingServer::get_singleton()->viewport_attach_to_screen(get_viewport_rid(), Rect2i(), DisplayServer::INVALID_WINDOW_ID);
	}

	if (window_id == DisplayServer::MAIN_WINDOW_ID) {
		if (!use_font_oversampling) {
			font_oversampling = 1.0;
		}
		if (TS->font_get_oversampling() != font_oversampling) {
			TS->font_set_oversampling(font_oversampling);
		}
	}

	notification(NOTIFICATION_WM_SIZE_CHANGED);

	if (embedder) {
		embedder->_sub_window_update(this);
	}
}

void Window::_update_window_callbacks() {
	DisplayServer::get_singleton()->window_set_rect_changed_callback(callable_mp(this, &Window::_rect_changed_callback), window_id);
	DisplayServer::get_singleton()->window_set_window_event_callback(callable_mp(this, &Window::_event_callback), window_id);
	DisplayServer::get_singleton()->window_set_input_event_callback(callable_mp(this, &Window::_window_input), window_id);
	DisplayServer::get_singleton()->window_set_input_text_callback(callable_mp(this, &Window::_window_input_text), window_id);
	DisplayServer::get_singleton()->window_set_drop_files_callback(callable_mp(this, &Window::_window_drop_files), window_id);
}

Viewport *Window::_get_embedder() const {
	Viewport *vp = get_parent_viewport();

	while (vp) {
		if (vp->is_embedding_subwindows()) {
			return vp;
		}

		if (vp->get_parent()) {
			vp = vp->get_parent()->get_viewport();
		} else {
			vp = nullptr;
		}
	}
	return nullptr;
}

void Window::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		bool embedded = false;
		{
			embedder = _get_embedder();

			if (embedder) {
				embedded = true;

				if (!visible) {
					embedder = nullptr; //not yet since not visible
				}
			}
		}

		if (embedded) {
			//create as embedded
			if (embedder) {
				embedder->_sub_window_register(this);
				RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_WHEN_PARENT_VISIBLE);
				_update_window_size();
			}

		} else {
			if (get_parent() == nullptr) {
				//it's the root window!
				visible = true; //always visible
				window_id = DisplayServer::MAIN_WINDOW_ID;
				DisplayServer::get_singleton()->window_attach_instance_id(get_instance_id(), window_id);
				_update_from_window();
				//since this window already exists (created on start), we must update pos and size from it
				{
					position = DisplayServer::get_singleton()->window_get_position(window_id);
					size = DisplayServer::get_singleton()->window_get_size(window_id);
				}
				_update_viewport_size(); //then feed back to the viewport
				_update_window_callbacks();
				RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_WHEN_VISIBLE);
			} else {
				//create
				if (visible) {
					_make_window();
				}
			}
		}

		if (transient) {
			_make_transient();
		}
		if (visible) {
			notification(NOTIFICATION_VISIBILITY_CHANGED);
			emit_signal(SceneStringNames::get_singleton()->visibility_changed);
			RS::get_singleton()->viewport_set_active(get_viewport_rid(), true);
		}
	}

	if (p_what == NOTIFICATION_READY) {
		if (wrap_controls) {
			_update_child_controls();
		}
	}

	if (p_what == NOTIFICATION_TRANSLATION_CHANGED) {
		child_controls_changed();
	}

	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (transient) {
			_clear_transient();
		}

		if (!is_embedded() && window_id != DisplayServer::INVALID_WINDOW_ID) {
			if (window_id == DisplayServer::MAIN_WINDOW_ID) {
				RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_DISABLED);
				_update_window_callbacks();
			} else {
				_clear_window();
			}
		} else {
			if (embedder) {
				embedder->_sub_window_remove(this);
				embedder = nullptr;
				RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_DISABLED);
			}
			_update_viewport_size(); //called by clear and make, which does not happen here
		}

		RS::get_singleton()->viewport_set_active(get_viewport_rid(), false);
	}
}

void Window::set_content_scale_size(const Size2i &p_size) {
	ERR_FAIL_COND(p_size.x < 0);
	ERR_FAIL_COND(p_size.y < 0);
	content_scale_size = p_size;
	_update_viewport_size();
}

Size2i Window::get_content_scale_size() const {
	return content_scale_size;
}

void Window::set_content_scale_mode(ContentScaleMode p_mode) {
	content_scale_mode = p_mode;
	_update_viewport_size();
}

Window::ContentScaleMode Window::get_content_scale_mode() const {
	return content_scale_mode;
}

void Window::set_content_scale_aspect(ContentScaleAspect p_aspect) {
	content_scale_aspect = p_aspect;
	_update_viewport_size();
}

Window::ContentScaleAspect Window::get_content_scale_aspect() const {
	return content_scale_aspect;
}

void Window::set_use_font_oversampling(bool p_oversampling) {
	if (is_inside_tree() && window_id != DisplayServer::MAIN_WINDOW_ID) {
		ERR_FAIL_MSG("Only the root window can set and use font oversampling.");
	}
	use_font_oversampling = p_oversampling;
	_update_viewport_size();
}

bool Window::is_using_font_oversampling() const {
	return use_font_oversampling;
}

DisplayServer::WindowID Window::get_window_id() const {
	if (embedder) {
		return parent->get_window_id();
	}
	return window_id;
}

void Window::set_wrap_controls(bool p_enable) {
	wrap_controls = p_enable;
	if (wrap_controls) {
		child_controls_changed();
	}
}

bool Window::is_wrapping_controls() const {
	return wrap_controls;
}

Size2 Window::_get_contents_minimum_size() const {
	Size2 max;

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (c) {
			Point2i pos = c->get_position();
			Size2i min = c->get_combined_minimum_size();

			max.x = MAX(pos.x + min.x, max.x);
			max.y = MAX(pos.y + min.y, max.y);
		}
	}

	return max;
}

void Window::_update_child_controls() {
	if (!updating_child_controls) {
		return;
	}

	_update_window_size();

	updating_child_controls = false;
}

void Window::child_controls_changed() {
	if (!is_inside_tree() || !visible || updating_child_controls) {
		return;
	}

	updating_child_controls = true;
	call_deferred("_update_child_controls");
}

bool Window::_can_consume_input_events() const {
	return exclusive_child == nullptr;
}

void Window::_window_input(const Ref<InputEvent> &p_ev) {
	if (EngineDebugger::is_active()) {
		//quit from game window using F8
		Ref<InputEventKey> k = p_ev;
		if (k.is_valid() && k->is_pressed() && !k->is_echo() && k->get_keycode() == KEY_F8) {
			EngineDebugger::get_singleton()->send_message("request_quit", Array());
		}
	}

	if (exclusive_child != nullptr) {
		/*
		Window *focus_target = exclusive_child;
		focus_target->grab_focus();
		while (focus_target->exclusive_child != nullptr) {
			focus_target = focus_target->exclusive_child;
			focus_target->grab_focus();
		}*/

		if (!is_embedding_subwindows()) { //not embedding, no need for event
			return;
		}
	}

	emit_signal(SceneStringNames::get_singleton()->window_input, p_ev);

	input(p_ev);
	if (!is_input_handled()) {
		unhandled_input(p_ev);
	}
}

void Window::_window_input_text(const String &p_text) {
	input_text(p_text);
}

void Window::_window_drop_files(const Vector<String> &p_files) {
	emit_signal("files_dropped", p_files, current_screen);
}

Viewport *Window::get_parent_viewport() const {
	if (get_parent()) {
		return get_parent()->get_viewport();
	} else {
		return nullptr;
	}
}

Window *Window::get_parent_visible_window() const {
	Viewport *vp = get_parent_viewport();
	Window *window = nullptr;
	while (vp) {
		window = Object::cast_to<Window>(vp);
		if (window && window->visible) {
			break;
		}
		if (!vp->get_parent()) {
			break;
		}

		vp = vp->get_parent()->get_viewport();
	}
	return window;
}

void Window::popup_on_parent(const Rect2i &p_parent_rect) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND_MSG(window_id == DisplayServer::MAIN_WINDOW_ID, "Can't popup the main window.");

	if (!is_embedded()) {
		Window *window = get_parent_visible_window();

		if (!window) {
			popup(p_parent_rect);
		} else {
			popup(Rect2i(window->get_position() + p_parent_rect.position, p_parent_rect.size));
		}
	} else {
		popup(p_parent_rect);
	}
}

void Window::popup_centered_clamped(const Size2i &p_size, float p_fallback_ratio) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND_MSG(window_id == DisplayServer::MAIN_WINDOW_ID, "Can't popup the main window.");

	Rect2 parent_rect;

	if (is_embedded()) {
		parent_rect = get_parent_viewport()->get_visible_rect();
	} else {
		DisplayServer::WindowID parent_id = get_parent_visible_window()->get_window_id();
		int parent_screen = DisplayServer::get_singleton()->window_get_current_screen(parent_id);
		parent_rect.position = DisplayServer::get_singleton()->screen_get_position(parent_screen);
		parent_rect.size = DisplayServer::get_singleton()->screen_get_size(parent_screen);
	}

	Vector2i size_ratio = parent_rect.size * p_fallback_ratio;

	Rect2i popup_rect;
	popup_rect.size = Vector2i(MIN(size_ratio.x, p_size.x), MIN(size_ratio.y, p_size.y));
	popup_rect.position = parent_rect.position + (parent_rect.size - popup_rect.size) / 2;

	popup(popup_rect);
}

void Window::popup_centered(const Size2i &p_minsize) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND_MSG(window_id == DisplayServer::MAIN_WINDOW_ID, "Can't popup the main window.");

	Rect2 parent_rect;

	if (is_embedded()) {
		parent_rect = get_parent_viewport()->get_visible_rect();
	} else {
		DisplayServer::WindowID parent_id = get_parent_visible_window()->get_window_id();
		int parent_screen = DisplayServer::get_singleton()->window_get_current_screen(parent_id);
		parent_rect.position = DisplayServer::get_singleton()->screen_get_position(parent_screen);
		parent_rect.size = DisplayServer::get_singleton()->screen_get_size(parent_screen);
	}

	Rect2i popup_rect;
	if (p_minsize == Size2i()) {
		popup_rect.size = _get_contents_minimum_size();
	} else {
		popup_rect.size = p_minsize;
	}
	popup_rect.position = parent_rect.position + (parent_rect.size - popup_rect.size) / 2;

	popup(popup_rect);
}

void Window::popup_centered_ratio(float p_ratio) {
	ERR_FAIL_COND(!is_inside_tree());
	ERR_FAIL_COND_MSG(window_id == DisplayServer::MAIN_WINDOW_ID, "Can't popup the main window.");

	Rect2 parent_rect;

	if (is_embedded()) {
		parent_rect = get_parent_viewport()->get_visible_rect();
	} else {
		DisplayServer::WindowID parent_id = get_parent_visible_window()->get_window_id();
		int parent_screen = DisplayServer::get_singleton()->window_get_current_screen(parent_id);
		parent_rect.position = DisplayServer::get_singleton()->screen_get_position(parent_screen);
		parent_rect.size = DisplayServer::get_singleton()->screen_get_size(parent_screen);
	}

	Rect2i popup_rect;
	popup_rect.size = parent_rect.size * p_ratio;
	popup_rect.position = parent_rect.position + (parent_rect.size - popup_rect.size) / 2;

	popup(popup_rect);
}

void Window::popup(const Rect2i &p_screen_rect) {
	emit_signal("about_to_popup");

	// Update window size to calculate the actual window size based on contents minimum size and minimum size.
	_update_window_size();

	if (p_screen_rect != Rect2i()) {
		set_position(p_screen_rect.position);
		set_size(p_screen_rect.size);
	}

	Rect2i adjust = _popup_adjust_rect();
	if (adjust != Rect2i()) {
		set_position(adjust.position);
		set_size(adjust.size);
	}

	int scr = DisplayServer::get_singleton()->get_screen_count();
	for (int i = 0; i < scr; i++) {
		Rect2i r = DisplayServer::get_singleton()->screen_get_usable_rect(i);
		if (r.has_point(position)) {
			current_screen = i;
			break;
		}
	}

	set_transient(true);
	set_visible(true);
	_post_popup();
	notification(NOTIFICATION_POST_POPUP);
}

Size2 Window::get_contents_minimum_size() const {
	return _get_contents_minimum_size();
}

void Window::grab_focus() {
	if (embedder) {
		embedder->_sub_window_grab_focus(this);
	} else if (window_id != DisplayServer::INVALID_WINDOW_ID) {
		DisplayServer::get_singleton()->window_move_to_foreground(window_id);
	}
}

bool Window::has_focus() const {
	return focused;
}

Rect2i Window::get_usable_parent_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2());
	Rect2i parent;
	if (is_embedded()) {
		parent = _get_embedder()->get_visible_rect();
	} else {
		const Window *w = is_visible() ? this : get_parent_visible_window();
		//find a parent that can contain us
		ERR_FAIL_COND_V(!w, Rect2());

		parent = DisplayServer::get_singleton()->screen_get_usable_rect(DisplayServer::get_singleton()->window_get_current_screen(w->get_window_id()));
	}
	return parent;
}

void Window::add_child_notify(Node *p_child) {
	Control *child_c = Object::cast_to<Control>(p_child);

	if (child_c && child_c->data.theme.is_null() && (theme_owner || theme_owner_window)) {
		Control::_propagate_theme_changed(child_c, theme_owner, theme_owner_window); //need to propagate here, since many controls may require setting up stuff
	}

	Window *child_w = Object::cast_to<Window>(p_child);

	if (child_w && child_w->theme.is_null() && (theme_owner || theme_owner_window)) {
		Control::_propagate_theme_changed(child_w, theme_owner, theme_owner_window); //need to propagate here, since many controls may require setting up stuff
	}

	if (is_inside_tree() && wrap_controls) {
		child_controls_changed();
	}
}

void Window::remove_child_notify(Node *p_child) {
	Control *child_c = Object::cast_to<Control>(p_child);

	if (child_c && (child_c->data.theme_owner || child_c->data.theme_owner_window) && child_c->data.theme.is_null()) {
		Control::_propagate_theme_changed(child_c, nullptr, nullptr);
	}

	Window *child_w = Object::cast_to<Window>(p_child);

	if (child_w && (child_w->theme_owner || child_w->theme_owner_window) && child_w->theme.is_null()) {
		Control::_propagate_theme_changed(child_w, nullptr, nullptr);
	}

	if (is_inside_tree() && wrap_controls) {
		child_controls_changed();
	}
}

void Window::set_theme(const Ref<Theme> &p_theme) {
	if (theme == p_theme) {
		return;
	}

	theme = p_theme;

	if (!p_theme.is_null()) {
		theme_owner = nullptr;
		theme_owner_window = this;
		Control::_propagate_theme_changed(this, nullptr, this);
	} else {
		Control *parent_c = cast_to<Control>(get_parent());
		if (parent_c && (parent_c->data.theme_owner || parent_c->data.theme_owner_window)) {
			Control::_propagate_theme_changed(this, parent_c->data.theme_owner, parent_c->data.theme_owner_window);
		} else {
			Window *parent_w = cast_to<Window>(get_parent());
			if (parent_w && (parent_w->theme_owner || parent_w->theme_owner_window)) {
				Control::_propagate_theme_changed(this, parent_w->theme_owner, parent_w->theme_owner_window);
			} else {
				Control::_propagate_theme_changed(this, nullptr, nullptr);
			}
		}
	}
}

Ref<Theme> Window::get_theme() const {
	return theme;
}

void Window::set_theme_custom_type(const StringName &p_theme_type) {
	theme_custom_type = p_theme_type;
	Control::_propagate_theme_changed(this, theme_owner, theme_owner_window);
}

StringName Window::get_theme_custom_type() const {
	return theme_custom_type;
}

void Window::_get_theme_type_dependencies(const StringName &p_theme_type, List<StringName> *p_list) const {
	if (p_theme_type == StringName() || p_theme_type == get_class_name() || p_theme_type == theme_custom_type) {
		if (theme_custom_type != StringName()) {
			p_list->push_back(theme_custom_type);
		}
		Theme::get_type_dependencies(get_class_name(), p_list);
	} else {
		Theme::get_type_dependencies(p_theme_type, p_list);
	}
}

Ref<Texture2D> Window::get_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<Ref<Texture2D>>(theme_owner, theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
}

Ref<StyleBox> Window::get_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<Ref<StyleBox>>(theme_owner, theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
}

Ref<Font> Window::get_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<Ref<Font>>(theme_owner, theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
}

int Window::get_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<int>(theme_owner, theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
}

Color Window::get_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<Color>(theme_owner, theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
}

int Window::get_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::get_theme_item_in_types<int>(theme_owner, theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
}

bool Window::has_theme_icon(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_ICON, p_name, theme_types);
}

bool Window::has_theme_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_STYLEBOX, p_name, theme_types);
}

bool Window::has_theme_font(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_FONT, p_name, theme_types);
}

bool Window::has_theme_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_FONT_SIZE, p_name, theme_types);
}

bool Window::has_theme_color(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_COLOR, p_name, theme_types);
}

bool Window::has_theme_constant(const StringName &p_name, const StringName &p_theme_type) const {
	List<StringName> theme_types;
	_get_theme_type_dependencies(p_theme_type, &theme_types);
	return Control::has_theme_item_in_types(theme_owner, theme_owner_window, Theme::DATA_TYPE_CONSTANT, p_name, theme_types);
}

Rect2i Window::get_parent_rect() const {
	ERR_FAIL_COND_V(!is_inside_tree(), Rect2i());
	if (is_embedded()) {
		//viewport
		Node *n = get_parent();
		ERR_FAIL_COND_V(!n, Rect2i());
		Viewport *p = n->get_viewport();
		ERR_FAIL_COND_V(!p, Rect2i());

		return p->get_visible_rect();
	} else {
		int x = get_position().x;
		int closest_dist = 0x7FFFFFFF;
		Rect2i closest_rect;
		for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
			Rect2i s(DisplayServer::get_singleton()->screen_get_position(i), DisplayServer::get_singleton()->screen_get_size(i));
			int d;
			if (x >= s.position.x && x < s.size.x) {
				//contained
				closest_rect = s;
				break;
			} else if (x < s.position.x) {
				d = s.position.x - x;
			} else {
				d = x - (s.position.x + s.size.x);
			}

			if (d < closest_dist) {
				closest_dist = d;
				closest_rect = s;
			}
		}
		return closest_rect;
	}
}

void Window::set_clamp_to_embedder(bool p_enable) {
	clamp_to_embedder = p_enable;
}

bool Window::is_clamped_to_embedder() const {
	return clamp_to_embedder;
}

void Window::set_layout_direction(Window::LayoutDirection p_direction) {
	ERR_FAIL_INDEX((int)p_direction, 4);

	layout_dir = p_direction;
	propagate_notification(Control::NOTIFICATION_LAYOUT_DIRECTION_CHANGED);
}

Window::LayoutDirection Window::get_layout_direction() const {
	return layout_dir;
}

bool Window::is_layout_rtl() const {
	if (layout_dir == LAYOUT_DIRECTION_INHERITED) {
		Window *parent = Object::cast_to<Window>(get_parent());
		if (parent) {
			return parent->is_layout_rtl();
		} else {
			if (GLOBAL_GET("internationalization/rendering/force_right_to_left_layout_direction")) {
				return true;
			}
			String locale = TranslationServer::get_singleton()->get_tool_locale();
			return TS->is_locale_right_to_left(locale);
		}
	} else if (layout_dir == LAYOUT_DIRECTION_LOCALE) {
		if (GLOBAL_GET("internationalization/rendering/force_right_to_left_layout_direction")) {
			return true;
		}
		String locale = TranslationServer::get_singleton()->get_tool_locale();
		return TS->is_locale_right_to_left(locale);
	} else {
		return (layout_dir == LAYOUT_DIRECTION_RTL);
	}
}

void Window::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_title", "title"), &Window::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &Window::get_title);

	ClassDB::bind_method(D_METHOD("set_current_screen", "index"), &Window::set_current_screen);
	ClassDB::bind_method(D_METHOD("get_current_screen"), &Window::get_current_screen);

	ClassDB::bind_method(D_METHOD("set_position", "position"), &Window::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &Window::get_position);

	ClassDB::bind_method(D_METHOD("set_size", "size"), &Window::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &Window::get_size);

	ClassDB::bind_method(D_METHOD("get_real_size"), &Window::get_real_size);

	ClassDB::bind_method(D_METHOD("set_max_size", "max_size"), &Window::set_max_size);
	ClassDB::bind_method(D_METHOD("get_max_size"), &Window::get_max_size);

	ClassDB::bind_method(D_METHOD("set_min_size", "min_size"), &Window::set_min_size);
	ClassDB::bind_method(D_METHOD("get_min_size"), &Window::get_min_size);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &Window::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &Window::get_mode);

	ClassDB::bind_method(D_METHOD("set_flag", "flag", "enabled"), &Window::set_flag);
	ClassDB::bind_method(D_METHOD("get_flag", "flag"), &Window::get_flag);

	ClassDB::bind_method(D_METHOD("is_maximize_allowed"), &Window::is_maximize_allowed);

	ClassDB::bind_method(D_METHOD("request_attention"), &Window::request_attention);

	ClassDB::bind_method(D_METHOD("move_to_foreground"), &Window::move_to_foreground);

	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &Window::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &Window::is_visible);

	ClassDB::bind_method(D_METHOD("hide"), &Window::hide);
	ClassDB::bind_method(D_METHOD("show"), &Window::show);

	ClassDB::bind_method(D_METHOD("set_transient", "transient"), &Window::set_transient);
	ClassDB::bind_method(D_METHOD("is_transient"), &Window::is_transient);

	ClassDB::bind_method(D_METHOD("set_exclusive", "exclusive"), &Window::set_exclusive);
	ClassDB::bind_method(D_METHOD("is_exclusive"), &Window::is_exclusive);

	ClassDB::bind_method(D_METHOD("can_draw"), &Window::can_draw);
	ClassDB::bind_method(D_METHOD("has_focus"), &Window::has_focus);
	ClassDB::bind_method(D_METHOD("grab_focus"), &Window::grab_focus);

	ClassDB::bind_method(D_METHOD("set_ime_active", "active"), &Window::set_ime_active);
	ClassDB::bind_method(D_METHOD("set_ime_position", "position"), &Window::set_ime_position);

	ClassDB::bind_method(D_METHOD("is_embedded"), &Window::is_embedded);

	ClassDB::bind_method(D_METHOD("set_content_scale_size", "size"), &Window::set_content_scale_size);
	ClassDB::bind_method(D_METHOD("get_content_scale_size"), &Window::get_content_scale_size);

	ClassDB::bind_method(D_METHOD("set_content_scale_mode", "mode"), &Window::set_content_scale_mode);
	ClassDB::bind_method(D_METHOD("get_content_scale_mode"), &Window::get_content_scale_mode);

	ClassDB::bind_method(D_METHOD("set_content_scale_aspect", "aspect"), &Window::set_content_scale_aspect);
	ClassDB::bind_method(D_METHOD("get_content_scale_aspect"), &Window::get_content_scale_aspect);

	ClassDB::bind_method(D_METHOD("set_use_font_oversampling", "enable"), &Window::set_use_font_oversampling);
	ClassDB::bind_method(D_METHOD("is_using_font_oversampling"), &Window::is_using_font_oversampling);

	ClassDB::bind_method(D_METHOD("set_wrap_controls", "enable"), &Window::set_wrap_controls);
	ClassDB::bind_method(D_METHOD("is_wrapping_controls"), &Window::is_wrapping_controls);
	ClassDB::bind_method(D_METHOD("child_controls_changed"), &Window::child_controls_changed);

	ClassDB::bind_method(D_METHOD("_update_child_controls"), &Window::_update_child_controls);

	ClassDB::bind_method(D_METHOD("set_theme", "theme"), &Window::set_theme);
	ClassDB::bind_method(D_METHOD("get_theme"), &Window::get_theme);

	ClassDB::bind_method(D_METHOD("set_theme_custom_type", "theme_type"), &Window::set_theme_custom_type);
	ClassDB::bind_method(D_METHOD("get_theme_custom_type"), &Window::get_theme_custom_type);

	ClassDB::bind_method(D_METHOD("get_theme_icon", "name", "theme_type"), &Window::get_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_stylebox", "name", "theme_type"), &Window::get_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font", "name", "theme_type"), &Window::get_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_font_size", "name", "theme_type"), &Window::get_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_color", "name", "theme_type"), &Window::get_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_theme_constant", "name", "theme_type"), &Window::get_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("has_theme_icon", "name", "theme_type"), &Window::has_theme_icon, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_stylebox", "name", "theme_type"), &Window::has_theme_stylebox, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font", "name", "theme_type"), &Window::has_theme_font, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_font_size", "name", "theme_type"), &Window::has_theme_font_size, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_color", "name", "theme_type"), &Window::has_theme_color, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("has_theme_constant", "name", "theme_type"), &Window::has_theme_constant, DEFVAL(""));

	ClassDB::bind_method(D_METHOD("set_layout_direction", "direction"), &Window::set_layout_direction);
	ClassDB::bind_method(D_METHOD("get_layout_direction"), &Window::get_layout_direction);
	ClassDB::bind_method(D_METHOD("is_layout_rtl"), &Window::is_layout_rtl);

	ClassDB::bind_method(D_METHOD("popup", "rect"), &Window::popup, DEFVAL(Rect2i()));
	ClassDB::bind_method(D_METHOD("popup_on_parent", "parent_rect"), &Window::popup_on_parent);
	ClassDB::bind_method(D_METHOD("popup_centered_ratio", "ratio"), &Window::popup_centered_ratio, DEFVAL(0.8));
	ClassDB::bind_method(D_METHOD("popup_centered", "minsize"), &Window::popup_centered, DEFVAL(Size2i()));
	ClassDB::bind_method(D_METHOD("popup_centered_clamped", "minsize", "fallback_ratio"), &Window::popup_centered_clamped, DEFVAL(Size2i()), DEFVAL(0.75));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Windowed,Minimized,Maximized,Fullscreen"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "current_screen"), "set_current_screen", "get_current_screen");
	ADD_GROUP("Flags", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "wrap_controls"), "set_wrap_controls", "is_wrapping_controls");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "transient"), "set_transient", "is_transient");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclusive"), "set_exclusive", "is_exclusive");
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "unresizable"), "set_flag", "get_flag", FLAG_RESIZE_DISABLED);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "borderless"), "set_flag", "get_flag", FLAG_BORDERLESS);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "always_on_top"), "set_flag", "get_flag", FLAG_ALWAYS_ON_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "transparent"), "set_flag", "get_flag", FLAG_TRANSPARENT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "unfocusable"), "set_flag", "get_flag", FLAG_NO_FOCUS);
	ADD_GROUP("Limits", "");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "min_size"), "set_min_size", "get_min_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "max_size"), "set_max_size", "get_max_size");
	ADD_GROUP("Content Scale", "content_scale_");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2I, "content_scale_size"), "set_content_scale_size", "get_content_scale_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "content_scale_mode", PROPERTY_HINT_ENUM, "Disabled,Canvas Items,Viewport"), "set_content_scale_mode", "get_content_scale_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "content_scale_aspect", PROPERTY_HINT_ENUM, "Ignore,Keep,Keep Width,Keep Height,Expand"), "set_content_scale_aspect", "get_content_scale_aspect");
	ADD_GROUP("Theme", "theme_");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "theme", PROPERTY_HINT_RESOURCE_TYPE, "Theme"), "set_theme", "get_theme");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "theme_custom_type"), "set_theme_custom_type", "get_theme_custom_type");

	ADD_SIGNAL(MethodInfo("window_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
	ADD_SIGNAL(MethodInfo("files_dropped", PropertyInfo(Variant::PACKED_STRING_ARRAY, "files")));
	ADD_SIGNAL(MethodInfo("mouse_entered"));
	ADD_SIGNAL(MethodInfo("mouse_exited"));
	ADD_SIGNAL(MethodInfo("focus_entered"));
	ADD_SIGNAL(MethodInfo("focus_exited"));
	ADD_SIGNAL(MethodInfo("close_requested"));
	ADD_SIGNAL(MethodInfo("go_back_requested"));
	ADD_SIGNAL(MethodInfo("visibility_changed"));
	ADD_SIGNAL(MethodInfo("about_to_popup"));

	BIND_CONSTANT(NOTIFICATION_VISIBILITY_CHANGED);

	BIND_ENUM_CONSTANT(MODE_WINDOWED);
	BIND_ENUM_CONSTANT(MODE_MINIMIZED);
	BIND_ENUM_CONSTANT(MODE_MAXIMIZED);
	BIND_ENUM_CONSTANT(MODE_FULLSCREEN);

	BIND_ENUM_CONSTANT(FLAG_RESIZE_DISABLED);
	BIND_ENUM_CONSTANT(FLAG_BORDERLESS);
	BIND_ENUM_CONSTANT(FLAG_ALWAYS_ON_TOP);
	BIND_ENUM_CONSTANT(FLAG_TRANSPARENT);
	BIND_ENUM_CONSTANT(FLAG_NO_FOCUS);
	BIND_ENUM_CONSTANT(FLAG_MAX);

	BIND_ENUM_CONSTANT(CONTENT_SCALE_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_MODE_CANVAS_ITEMS);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_MODE_VIEWPORT);

	BIND_ENUM_CONSTANT(CONTENT_SCALE_ASPECT_IGNORE);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_ASPECT_KEEP);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_ASPECT_KEEP_WIDTH);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_ASPECT_KEEP_HEIGHT);
	BIND_ENUM_CONSTANT(CONTENT_SCALE_ASPECT_EXPAND);

	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_INHERITED);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LOCALE);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_LTR);
	BIND_ENUM_CONSTANT(LAYOUT_DIRECTION_RTL);
}

Window::Window() {
	RS::get_singleton()->viewport_set_update_mode(get_viewport_rid(), RS::VIEWPORT_UPDATE_DISABLED);
}

Window::~Window() {
}
