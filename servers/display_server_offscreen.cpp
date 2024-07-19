/**************************************************************************/
/*  display_server_offscreen.cpp                                          */
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

#include "display_server_offscreen.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "core/version.h"
#include "main/main.h"

#if defined(VULKAN_ENABLED)
#include "rendering_context_driver_vulkan_headless.h"
#endif

bool DisplayServerOffscreen::has_feature(Feature p_feature) const {
	switch (p_feature) {
		default:
			return false;
	}
}

String DisplayServerOffscreen::get_name() const {
	return "Offscreen";
}

int DisplayServerOffscreen::get_screen_count() const {
	return 1;
}

int DisplayServerOffscreen::get_primary_screen() const {
	return 0;
}

int DisplayServerOffscreen::get_keyboard_focus_screen() const {
	return get_primary_screen();
}

Point2i DisplayServerOffscreen::screen_get_position(int p_screen) const {
	return Point2i(0, 0);
}

Size2i DisplayServerOffscreen::screen_get_size(int p_screen) const {
	return Size2i(1920, 1080);
}

Rect2i DisplayServerOffscreen::screen_get_usable_rect(int p_screen) const {
	return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
}

int DisplayServerOffscreen::screen_get_dpi(int p_screen) const {
	return 96;
}

float DisplayServerOffscreen::screen_get_refresh_rate(int p_screen) const {
	return SCREEN_REFRESH_RATE_FALLBACK;
}

Vector<DisplayServer::WindowID> DisplayServerOffscreen::get_window_list() const {
	_THREAD_SAFE_METHOD_

	Vector<DisplayServer::WindowID> ret;
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		ret.push_back(E.key);
	}
	return ret;
}

DisplayServer::WindowID DisplayServerOffscreen::get_window_at_screen_position(const Point2i &p_position) const {
	if (windows.is_empty()) {
		return INVALID_WINDOW_ID;
	}

	return windows.begin()->key;
}

DisplayServer::WindowID DisplayServerOffscreen::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	WindowID window_id = _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
	ERR_FAIL_COND_V_MSG(window_id == INVALID_WINDOW_ID, INVALID_WINDOW_ID, "Failed to create sub window.");

	WindowData &wd = windows[window_id];

	if (p_flags & WINDOW_FLAG_RESIZE_DISABLED_BIT) {
		wd.resizable = false;
	}
	if (p_flags & WINDOW_FLAG_BORDERLESS_BIT) {
		wd.borderless = true;
	}
	if (p_flags & WINDOW_FLAG_ALWAYS_ON_TOP_BIT && p_mode != WINDOW_MODE_FULLSCREEN && p_mode != WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		wd.always_on_top = true;
	}
	if (p_flags & WINDOW_FLAG_NO_FOCUS_BIT) {
		wd.no_focus = true;
	}
	if (p_flags & WINDOW_FLAG_MOUSE_PASSTHROUGH_BIT) {
		wd.mpass = true;
	}
	if (p_flags & WINDOW_FLAG_POPUP_BIT) {
		wd.is_popup = true;
	}
	if (p_flags & WINDOW_FLAG_TRANSPARENT_BIT) {
		wd.layered_window = true;
	}

#ifdef RD_ENABLED
	if (rendering_device) {
		//rendering_device->screen_create(window_id);
	}
#endif

	return window_id;
}

void DisplayServerOffscreen::show_window(WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));

	popup_open(p_id);
}

void DisplayServerOffscreen::delete_sub_window(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	ERR_FAIL_COND_MSG(p_window == MAIN_WINDOW_ID, "Main window cannot be deleted.");

	popup_close(p_window);

	WindowData &wd = windows[p_window];

	while (wd.transient_children.size()) {
		window_set_transient(*wd.transient_children.begin(), INVALID_WINDOW_ID);
	}

	if (wd.transient_parent != INVALID_WINDOW_ID) {
		window_set_transient(p_window, INVALID_WINDOW_ID);
	}

#ifdef RD_ENABLED
	if (rendering_device) {
		//rendering_device->screen_free(p_window);
	}

	if (rendering_context) {
		//rendering_context->window_destroy(p_window);
	}
#endif

	windows.erase(p_window);

	if (last_focused_window == p_window) {
		last_focused_window = INVALID_WINDOW_ID;
	}
}

int64_t DisplayServerOffscreen::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	return 0;
}

void DisplayServerOffscreen::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].instance_id = p_instance;
}

ObjectID DisplayServerOffscreen::window_get_attached_instance_id(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), ObjectID());
	return windows[p_window].instance_id;
}

void DisplayServerOffscreen::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].rect_changed_callback = p_callable;
}

void DisplayServerOffscreen::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].event_callback = p_callable;
}

void DisplayServerOffscreen::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].input_event_callback = p_callable;
}

void DisplayServerOffscreen::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].input_text_callback = p_callable;
}

void DisplayServerOffscreen::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].drop_files_callback = p_callable;
}

void DisplayServerOffscreen::window_set_title(const String &p_title, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
}

Size2i DisplayServerOffscreen::window_get_title_size(const String &p_title, WindowID p_window) const {
	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());

	return Size2i();
}

int DisplayServerOffscreen::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), -1);

	return 0;
}

void DisplayServerOffscreen::window_set_current_screen(int p_screen, WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));
	ERR_FAIL_INDEX(p_screen, get_screen_count());
}

Point2i DisplayServerOffscreen::window_get_position(WindowID p_window) const {
	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());

	return Point2i();
}

Point2i DisplayServerOffscreen::window_get_position_with_decorations(WindowID p_window) const {
	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());

	return Point2i();
}

void DisplayServerOffscreen::window_set_position(const Point2i &p_position, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (wd.fullscreen || wd.maximized) {
		return;
	}

	wd.position = p_position;
}

void DisplayServerOffscreen::window_set_exclusive(WindowID p_window, bool p_exclusive) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	if (wd.exclusive != p_exclusive) {
		wd.exclusive = p_exclusive;
	}
}

void DisplayServerOffscreen::window_set_transient(WindowID p_window, WindowID p_parent) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(p_window == p_parent);
	ERR_FAIL_COND(!windows.has(p_window));

	WindowData &wd_window = windows[p_window];

	ERR_FAIL_COND(wd_window.transient_parent == p_parent);
	ERR_FAIL_COND_MSG(wd_window.always_on_top, "Offscreen with the 'on top' can't become transient.");

	if (p_parent == INVALID_WINDOW_ID) {
		// Remove transient.

		ERR_FAIL_COND(wd_window.transient_parent == INVALID_WINDOW_ID);
		ERR_FAIL_COND(!windows.has(wd_window.transient_parent));

		WindowData &wd_parent = windows[wd_window.transient_parent];

		wd_window.transient_parent = INVALID_WINDOW_ID;
		wd_parent.transient_children.erase(p_window);
	} else {
		ERR_FAIL_COND(!windows.has(p_parent));
		ERR_FAIL_COND_MSG(wd_window.transient_parent != INVALID_WINDOW_ID, "Window already has a transient parent");
		WindowData &wd_parent = windows[p_parent];

		wd_window.transient_parent = p_parent;
		wd_parent.transient_children.insert(p_window);
	}
}

void DisplayServerOffscreen::window_set_max_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2()) && ((p_size.x < wd.min_size.x) || (p_size.y < wd.min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	wd.max_size = p_size;
}

Size2i DisplayServerOffscreen::window_get_max_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.max_size;
}

void DisplayServerOffscreen::window_set_min_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2()) && (wd.max_size != Size2()) && ((p_size.x > wd.max_size.x) || (p_size.y > wd.max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	wd.min_size = p_size;
}

Size2i DisplayServerOffscreen::window_get_min_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.min_size;
}

void DisplayServerOffscreen::window_set_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (wd.fullscreen || wd.maximized) {
		return;
	}

	wd.size = p_size;

#if defined(RD_ENABLED)
	if (rendering_context) {
		//rendering_context->window_set_size(p_window, p_size.width, p_size.height);
	}
#endif
}

Size2i DisplayServerOffscreen::window_get_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	return wd.size;
}

Size2i DisplayServerOffscreen::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerOffscreen::window_set_mode(WindowMode p_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (wd.fullscreen && p_mode != WINDOW_MODE_FULLSCREEN && p_mode != WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		wd.fullscreen = false;
		wd.multiwindow_fs = false;
		wd.maximized = wd.was_maximized;

		if (!wd.pre_fs_valid) {
			wd.pre_fs_valid = true;
		}
	}

	if (p_mode == WINDOW_MODE_WINDOWED) {
		wd.maximized = false;
		wd.minimized = false;
	}

	if (p_mode == WINDOW_MODE_MAXIMIZED) {
		wd.maximized = true;
		wd.minimized = false;
	}

	if (p_mode == WINDOW_MODE_MINIMIZED) {
		wd.maximized = false;
		wd.minimized = true;
	}

	if (p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		wd.multiwindow_fs = false;
	} else {
		wd.multiwindow_fs = true;
	}

	if ((p_mode == WINDOW_MODE_FULLSCREEN || p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) && !wd.fullscreen) {
		wd.was_maximized = wd.maximized;

		if (wd.pre_fs_valid) {
			wd.pre_fs_rect = Rect2(wd.position, wd.size);
		}

		int cs = window_get_current_screen(p_window);
		wd.position = screen_get_position(cs);
		wd.size = screen_get_size(cs);

		wd.fullscreen = true;
		wd.maximized = false;
		wd.minimized = false;
	}
}

DisplayServer::WindowMode DisplayServerOffscreen::window_get_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), WINDOW_MODE_WINDOWED);
	const WindowData &wd = windows[p_window];

	if (wd.fullscreen) {
		if (wd.multiwindow_fs) {
			return WINDOW_MODE_FULLSCREEN;
		} else {
			return WINDOW_MODE_EXCLUSIVE_FULLSCREEN;
		}
	} else if (wd.minimized) {
		return WINDOW_MODE_MINIMIZED;
	} else if (wd.maximized) {
		return WINDOW_MODE_MAXIMIZED;
	} else {
		return WINDOW_MODE_WINDOWED;
	}
}

bool DisplayServerOffscreen::window_is_maximize_allowed(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);

	// FIXME: Implement this, or confirm that it should always be true.

	return true;
}

void DisplayServerOffscreen::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			wd.resizable = !p_enabled;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			wd.borderless = p_enabled;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			ERR_FAIL_COND_MSG(wd.transient_parent != INVALID_WINDOW_ID && p_enabled, "Transient windows can't become on top");
			wd.always_on_top = p_enabled;
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			wd.layered_window = p_enabled;
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			wd.no_focus = p_enabled;
		} break;
		case WINDOW_FLAG_MOUSE_PASSTHROUGH: {
			wd.mpass = p_enabled;
		} break;
		case WINDOW_FLAG_POPUP: {
			ERR_FAIL_COND_MSG(p_window == MAIN_WINDOW_ID, "Main window can't be popup.");
			wd.is_popup = p_enabled;
		} break;
		default:
			break;
	}
}

bool DisplayServerOffscreen::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];
	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			return !wd.resizable;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			return wd.borderless;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			return wd.always_on_top;
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			return wd.layered_window;
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			return wd.no_focus;
		} break;
		case WINDOW_FLAG_MOUSE_PASSTHROUGH: {
			return wd.mpass;
		} break;
		case WINDOW_FLAG_POPUP: {
			return wd.is_popup;
		} break;
		default:
			break;
	}

	return false;
}

void DisplayServerOffscreen::window_request_attention(WindowID p_window) {
	// Do nothing.
}

void DisplayServerOffscreen::window_move_to_foreground(WindowID p_window) {
	// Do nothing.
}

bool DisplayServerOffscreen::window_is_focused(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	return wd.window_focused;
}

DisplayServerOffscreen::WindowID DisplayServerOffscreen::get_focused_window() const {
	return last_focused_window;
}

bool DisplayServerOffscreen::window_can_draw(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);

	return true;
}

bool DisplayServerOffscreen::can_any_window_draw() const {
	_THREAD_SAFE_METHOD_

	return true;
}

void DisplayServerOffscreen::process_events() {
	_THREAD_SAFE_METHOD_

	if (!drop_events) {
		Input::get_singleton()->flush_buffered_events();
	}
}

void DisplayServerOffscreen::force_process_and_drop_events() {
	_THREAD_SAFE_METHOD_

	drop_events = true;
	process_events();
	drop_events = false;
}

void DisplayServerOffscreen::release_rendering_thread() {
}

void DisplayServerOffscreen::swap_buffers() {
}

void DisplayServerOffscreen::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		//rendering_context->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerOffscreen::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		// return rendering_context->window_get_vsync_mode(p_window);
	}
#endif

	return DisplayServer::VSYNC_ENABLED;
}

void DisplayServerOffscreen::set_context(Context p_context) {
}

DisplayServer::WindowID DisplayServerOffscreen::window_get_active_popup() const {
	const List<WindowID>::Element *E = popup_list.back();
	if (E) {
		return E->get();
	} else {
		return INVALID_WINDOW_ID;
	}
}

void DisplayServerOffscreen::window_set_popup_safe_rect(WindowID p_window, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.parent_safe_rect = p_rect;
}

Rect2i DisplayServerOffscreen::window_get_popup_safe_rect(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Rect2i());
	const WindowData &wd = windows[p_window];
	return wd.parent_safe_rect;
}

void DisplayServerOffscreen::popup_open(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	bool has_popup_ancestor = false;
	WindowID transient_root = p_window;
	while (true) {
		WindowID parent = windows[transient_root].transient_parent;
		if (parent == INVALID_WINDOW_ID) {
			break;
		} else {
			transient_root = parent;
			if (windows[parent].is_popup) {
				has_popup_ancestor = true;
				break;
			}
		}
	}

	WindowData &wd = windows[p_window];
	if (wd.is_popup || has_popup_ancestor) {
		// Find current popup parent, or root popup if new window is not transient.
		List<WindowID>::Element *C = nullptr;
		List<WindowID>::Element *E = popup_list.back();
		while (E) {
			if (wd.transient_parent != E->get() || wd.transient_parent == INVALID_WINDOW_ID) {
				C = E;
				E = E->prev();
			} else {
				break;
			}
		}
		if (C) {
			// TODO: Send close request to C.
		}

		time_since_popup = OS::get_singleton()->get_ticks_msec();
		popup_list.push_back(p_window);
	}
}

void DisplayServerOffscreen::popup_close(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	List<WindowID>::Element *E = popup_list.find(p_window);
	while (E) {
		List<WindowID>::Element *F = E->next();
		WindowID win_id = E->get();
		popup_list.erase(E);

		if (win_id != p_window) {
			// Only request close on related windows, not this window.  We are already processing it.
			// TODO: Send close request to win_id.
		}
		E = F;
	}
}

DisplayServer::WindowID DisplayServerOffscreen::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	Rect2i window_rect;

	if (p_mode == WINDOW_MODE_FULLSCREEN || p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		window_rect = screen_get_usable_rect(get_primary_screen());
	} else {
		window_rect = screen_get_usable_rect(get_primary_screen());

		if (window_rect != Rect2i()) {
			window_rect.position.x = CLAMP(window_rect.position.x, window_rect.position.x, window_rect.position.x + window_rect.size.width - p_rect.size.width / 3);
			window_rect.position.y = CLAMP(window_rect.position.y, window_rect.position.y, window_rect.position.y + window_rect.size.height - p_rect.size.height / 3);
		}

		window_rect.size = p_rect.size;
	}

	WindowID id = window_id_counter;
	{
		WindowData &wd = windows[id];

		if (p_mode == WINDOW_MODE_FULLSCREEN || p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
			wd.fullscreen = true;
			if (p_mode == WINDOW_MODE_FULLSCREEN) {
				wd.multiwindow_fs = true;
			}
		}
		if (p_mode != WINDOW_MODE_FULLSCREEN && p_mode != WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
			wd.pre_fs_valid = true;
		}

#ifdef RD_ENABLED
		if (rendering_context) {
			union {
#ifdef VULKAN_ENABLED
				RenderingContextDriverVulkanHeadless::WindowPlatformData vulkan;
#endif
			} wpd;
#ifdef VULKAN_ENABLED
			if (rendering_driver == "vulkan") {
				// Do nothing (wpd is empty).
			}
#endif
			// if (rendering_context->window_create(id, &wpd) != OK) {
			// 	ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			// 	memdelete(rendering_context);
			// 	rendering_context = nullptr;
			// 	windows.erase(id);
			// 	return INVALID_WINDOW_ID;
			// }

			// rendering_context->window_set_size(id, window_rect.size.width, window_rect.size.height);
			// rendering_context->window_set_vsync_mode(id, p_vsync_mode);
			wd.context_created = true;
		}
#endif

		if (p_mode == WINDOW_MODE_MAXIMIZED) {
			wd.maximized = true;
			wd.minimized = false;
		}

		if (p_mode == WINDOW_MODE_MINIMIZED) {
			wd.maximized = false;
			wd.minimized = true;
		}

		wd.position = window_rect.position;
		wd.size = window_rect.size;

		window_id_counter++;
	}

	return id;
}

DisplayServerOffscreen::DisplayServerOffscreen(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, Error &r_error) {
	native_menu = memnew(NativeMenu);

	drop_events = false;

	rendering_driver = p_rendering_driver;

#if defined(RD_ENABLED)
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		rendering_context = memnew(RenderingContextDriverVulkanHeadless);
	}
#endif

	if (rendering_context) {
		if (rendering_context->initialize() != OK) {
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}
	}
#endif

	WindowID main_window = _create_window(p_mode, p_vsync_mode, p_flags, Rect2i(Point2i(), p_resolution));
	ERR_FAIL_COND_MSG(main_window == INVALID_WINDOW_ID, "Failed to create main window.");

	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, main_window);
		}
	}

	show_window(MAIN_WINDOW_ID);

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_device = memnew(RenderingDevice);
		rendering_device->initialize(rendering_context, INVALID_WINDOW_ID);
		//rendering_device->screen_create(MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
	}
#endif

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

Vector<String> DisplayServerOffscreen::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

	return drivers;
}

DisplayServer *DisplayServerOffscreen::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerOffscreen(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, r_error));
	if (r_error != OK) {
		if (p_rendering_driver == "vulkan") {
			String executable_name = OS::get_singleton()->get_executable_path().get_file();
			OS::get_singleton()->alert(
					vformat("Your video card drivers seem not to support the required Vulkan version.\n\n"
							"If possible, consider updating your video card drivers or using the OpenGL 3 driver.\n\n"
							"You can enable the OpenGL 3 driver by starting the engine from the\n"
							"command line with the command:\n\n    \"%s\" --rendering-driver opengl3\n\n"
							"If you have recently updated your video card drivers, try rebooting.",
							executable_name),
					"Unable to initialize Vulkan video driver");
		} else if (p_rendering_driver == "d3d12") {
			String executable_name = OS::get_singleton()->get_executable_path().get_file();
			OS::get_singleton()->alert(
					vformat("Your video card drivers seem not to support the required DirectX 12 version.\n\n"
							"If possible, consider updating your video card drivers or using the OpenGL 3 driver.\n\n"
							"You can enable the OpenGL 3 driver by starting the engine from the\n"
							"command line with the command:\n\n    \"%s\" --rendering-driver opengl3\n\n"
							"If you have recently updated your video card drivers, try rebooting.",
							executable_name),
					"Unable to initialize DirectX 12 video driver");
		} else {
			OS::get_singleton()->alert(
					"Your video card drivers seem not to support the required OpenGL 3.3 version.\n\n"
					"If possible, consider updating your video card drivers.\n\n"
					"If you have recently updated your video card drivers, try rebooting.",
					"Unable to initialize OpenGL video driver");
		}
	}
	return ds;
}

DisplayServerOffscreen::~DisplayServerOffscreen() {
	if (windows.has(MAIN_WINDOW_ID)) {
#ifdef RD_ENABLED
		if (rendering_device) {
			//rendering_device->screen_free(MAIN_WINDOW_ID);
		}

		if (rendering_context) {
			//rendering_context->window_destroy(MAIN_WINDOW_ID);
		}
#endif
	}

#ifdef RD_ENABLED
	if (rendering_device) {
		memdelete(rendering_device);
		rendering_device = nullptr;
	}

	if (rendering_context) {
		memdelete(rendering_context);
		rendering_context = nullptr;
	}
#endif

	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}
}

void DisplayServerOffscreen::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	static_cast<DisplayServerOffscreen *>(get_singleton())->_dispatch_input_event(p_event);
}

void DisplayServerOffscreen::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	if (in_dispatch_input_event) {
		return;
	}
	in_dispatch_input_event = true;

	{
		List<WindowID>::Element *E = popup_list.back();
		if (E && Object::cast_to<InputEventKey>(*p_event)) {
			// Redirect keyboard input to active popup.
			if (windows.has(E->get())) {
				Callable callable = windows[E->get()].input_event_callback;
				if (callable.is_valid()) {
					callable.call(p_event);
				}
			}
			in_dispatch_input_event = false;
			return;
		}
	}

	Ref<InputEventFromWindow> event_from_window = p_event;
	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		// Send to a single window.
		if (windows.has(event_from_window->get_window_id())) {
			Callable callable = windows[event_from_window->get_window_id()].input_event_callback;
			if (callable.is_valid()) {
				callable.call(p_event);
			}
		}
	} else {
		// Send to all windows.
		for (const KeyValue<WindowID, WindowData> &E : windows) {
			const Callable callable = E.value.input_event_callback;
			if (callable.is_valid()) {
				callable.call(p_event);
			}
		}
	}

	in_dispatch_input_event = false;
}
