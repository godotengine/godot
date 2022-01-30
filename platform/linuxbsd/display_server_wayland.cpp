#include "display_server_wayland.h"

#ifdef WAYLAND_ENABLED

#ifdef VULKAN_ENABLED
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#endif

// Implementation specific methods.

// Took from DisplayServerX11.
void DisplayServerWayland::dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerWayland *)(get_singleton()))->_dispatch_input_event(p_event);
}

// Took from DisplayServerX11.
void DisplayServerWayland::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	Variant ev = p_event;
	Variant *evp = &ev;
	Variant ret;
	Callable::CallError ce;

	Ref<InputEventFromWindow> event_from_window = p_event;
	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		//send to a window
		ERR_FAIL_COND(!wls.windows.has(event_from_window->get_window_id()));
		Callable callable = wls.windows[event_from_window->get_window_id()].input_event_callback;
		if (callable.is_null()) {
			return;
		}
		callable.call((const Variant **)&evp, 1, ret, ce);
	} else {
		//send to all windows
		for (KeyValue<WindowID, WindowData> &E : wls.windows) {
			Callable callable = E.value.input_event_callback;
			if (callable.is_null()) {
				continue;
			}
			callable.call((const Variant **)&evp, 1, ret, ce);
		}
	}
}

DisplayServerWayland::WindowID DisplayServerWayland::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	WindowID id = wls.window_id_counter++;

	WindowData &wd = wls.windows[id];

	wd.vsync_mode = p_vsync_mode;
	wd.rect = p_rect;

#ifdef VULKAN_ENABLED
	wd.context_vulkan = context_vulkan;
#endif
	wd.id = id;

	wd.wl_surface = wl_compositor_create_surface(wls.globals.wl_compositor);
	wd.xdg_surface = xdg_wm_base_get_xdg_surface(wls.globals.xdg_wm_base, wd.wl_surface);
	wd.xdg_toplevel = xdg_surface_get_toplevel(wd.xdg_surface);

	xdg_surface_add_listener(wd.xdg_surface, &xdg_surface_listener, &wd);
	xdg_toplevel_add_listener(wd.xdg_toplevel, &xdg_toplevel_listener, &wd);

	wl_surface_commit(wd.wl_surface);

	xdg_toplevel_set_title(wd.xdg_toplevel, "Godot");

	// Wait for a wl_surface.configure event.
	wl_display_roundtrip(wls.display);

	// TODO: positioners and whatnot.

	return id;
}

void DisplayServerWayland::_wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version) {
	// `data` is expected to be a `WaylandState`.
	WaylandState *wls = (WaylandState*) data;

	// `wl_compositor_interface` is defined in `thirdparty/wayland/wayland.c`
	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		// This will select the latest version supported by the server.
		// I'm not sure whether this is the best thing to do.
		wls->globals.wl_compositor = (struct wl_compositor*) wl_registry_bind(wl_registry, name, &wl_compositor_interface, version);
		wls->globals.wl_compositor_name = name;
		return;
	}

	// `wl_seat_interface` is defined in `thirdparty/wayland/wayland.c`
	if (strcmp(interface, wl_seat_interface.name) == 0) {
		wls->globals.wl_seat = (struct wl_seat*) wl_registry_bind(wl_registry, name, &wl_seat_interface, version);
		wls->globals.wl_seat_name = name;
		wl_seat_add_listener(wls->globals.wl_seat, &wl_seat_listener, &wls->seat_state);
		return;
	}

	// `xdg_wm_base_interface` is defined in `thirdparty/xdg-shell/xdg-shell.c`
	if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		wls->globals.xdg_wm_base = (struct xdg_wm_base*) wl_registry_bind(wl_registry, name, &xdg_wm_base_interface, version);
		wls->globals.xdg_wm_base_name = name;
		return;
	}
}

void DisplayServerWayland::_wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name) {
	// `data` is expected to be a `WaylandGlobals`.
	WaylandGlobals *globals = (WaylandGlobals*) data;

	if (name == globals->wl_compositor_name) {
		wl_compositor_destroy(globals->wl_compositor);
		return;
	}

	if (name == globals->wl_seat_name) {
		wl_seat_destroy(globals->wl_seat);
		return;
	}

	if (name == globals->xdg_wm_base_name) {
		xdg_wm_base_destroy(globals->xdg_wm_base);
		return;
	}
}

// `data` is expected to be a pointer to a `SeatState` struct.
void DisplayServerWayland::_wl_seat_on_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities) {
	SeatState *seat_state = (SeatState*) data;

	// TODO: Handle touch.

	if (capabilities & WL_SEAT_CAPABILITY_POINTER) {
		seat_state->wl_pointer = wl_seat_get_pointer(wl_seat);
		wl_pointer_add_listener(seat_state->wl_pointer, &wl_pointer_listener, &seat_state->pointer_state);
	} else if (seat_state->wl_pointer) {
		wl_pointer_destroy(seat_state->wl_pointer);
	}

	if (capabilities & WL_SEAT_CAPABILITY_KEYBOARD) {
		seat_state->wl_keyboard = wl_seat_get_keyboard(wl_seat);
		ERR_FAIL_COND(!seat_state->wl_keyboard);

		seat_state->keyboard_state.xkb_context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
		ERR_FAIL_COND(!seat_state->keyboard_state.xkb_context);

		wl_keyboard_add_listener(seat_state->wl_keyboard, &wl_keyboard_listener, &seat_state->keyboard_state);
	} else if (seat_state->wl_keyboard) {
		wl_keyboard_destroy(seat_state->wl_keyboard);
	}
}

void DisplayServerWayland::_wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name) {
}

// `PointerState` struct.
void DisplayServerWayland::_wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	PointerState* pointer_state = (PointerState*) data;
	pointer_state->data_buffer.focused_wl_surface = surface;
}

void DisplayServerWayland::_wl_pointer_on_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface) {
	PointerState* pointer_state = (PointerState*) data;
	pointer_state->data_buffer.focused_wl_surface = nullptr;
}

void DisplayServerWayland::_wl_pointer_on_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y) {
	PointerState* pointer_state = (PointerState*) data;

	pointer_state->data_buffer.position.x = wl_fixed_to_int(surface_x);
	pointer_state->data_buffer.position.y = wl_fixed_to_int(surface_y);

	pointer_state->data_buffer.time = time;
}

void DisplayServerWayland::_wl_pointer_on_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
	PointerState* pointer_state = (PointerState*) data;
	MouseButton &mouse_button_mask = pointer_state->data_buffer.pressed_button_mask;

	MouseButton button_pressed;

	// TODO: Handle more bttons
	switch (button) {
		case BTN_LEFT:
			button_pressed = MouseButton::LEFT;
			break;

		case BTN_MIDDLE:
			button_pressed = MouseButton::MIDDLE;
			break;

		case BTN_RIGHT:
			button_pressed = MouseButton::RIGHT;
			break;

		default:
			button_pressed = MouseButton::NONE;
			break;
	}

	if (state & WL_POINTER_BUTTON_STATE_PRESSED) {
		mouse_button_mask |= mouse_button_to_mask(button_pressed);
		pointer_state->data_buffer.last_button_pressed = button_pressed;
	} else {
		mouse_button_mask &= ~mouse_button_to_mask(button_pressed);
	}

	pointer_state->data_buffer.button_time = time;
	pointer_state->data_buffer.time = time;
}

void DisplayServerWayland::_wl_pointer_on_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {
}

void DisplayServerWayland::_wl_pointer_on_frame(void *data, struct wl_pointer *wl_pointer) {
	PointerState* pointer_state = (PointerState*) data;

	pointer_state->data = pointer_state->data_buffer;
}

void DisplayServerWayland::_wl_pointer_on_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source) {
}

void DisplayServerWayland::_wl_pointer_on_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis) {
}

void DisplayServerWayland::_wl_pointer_on_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete) {
}


// `wl_keyboard` event callbacks. These expect `data` to be a pointer to a
// `KeyboardState` struct.
void DisplayServerWayland::_wl_keyboard_on_keymap(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size) {
	ERR_FAIL_COND_MSG(format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1, "Unsupported keymap format announced from the Wayland compositor.");

	KeyboardState *keyboard_state = (KeyboardState*) data;

	// TODO: Unmap on destruction.
	keyboard_state->keymap_buffer = (const char*) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	keyboard_state->keymap_buffer_size = size;

	// FIXME: This relies on the buffer being null-terminated. Is this the correct
	// behaviour? If yes, then why does wayland offer a buffer size? Apparently
	// `xkb_keymap_new_from_buffer` gets confused about the ending null byte.
	// This behaviour should be investigated.
	keyboard_state->xkb_keymap = xkb_keymap_new_from_string(keyboard_state->xkb_context, keyboard_state->keymap_buffer,
								XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);

	// TODO: Handle layout changes.
	keyboard_state->xkb_state = xkb_state_new(keyboard_state->xkb_keymap);
}

void DisplayServerWayland::_wl_keyboard_on_enter(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys) {
}

void DisplayServerWayland::_wl_keyboard_on_leave(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface) {
}

void DisplayServerWayland::_wl_keyboard_on_key(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state) {
	KeyboardState *keyboard_state = (KeyboardState*) data;

	// We have to add 8 to the scancode to get an XKB-compatible keycode.
	xkb_keycode_t xkb_keycode = key + 8;

	bool pressed = state & WL_KEYBOARD_KEY_STATE_PRESSED;

	// FIXME: This is far from ideal and inconsistent with the pointer handling,
	// but this is easier and less issue-prone in this phase of development.
	// IMO ideally we should use these events to update the appropriate state
	// which `data` points and then process everything in `process_events`.
	// The reason we're doing this for now is that, opposed to events which
	// accumulate in a single state by design (eg. `wl_pointer` events), this would
	// require us to write some sort of message queue which is both way too
	// complicated considering the state of all other features and the possibility
	// of it being implemented with an (IMO) more ideal multithreaded approach.
	// (see: https://gitlab.freedesktop.org/wayland/wayland/issues/159).

	// TODO: Handle keys that release multiple symbols?
	Key keycode = KeyMappingXKB::get_keycode(xkb_state_key_get_one_sym(keyboard_state->xkb_state, xkb_keycode));
	Key physical_keycode = KeyMappingXKB::get_scancode(xkb_keycode);

	Ref<InputEventKey> k;
	k.instantiate();
	k->set_keycode(keycode);
	k->set_physical_keycode(physical_keycode);
	k->set_unicode(xkb_state_key_get_utf32(keyboard_state->xkb_state, xkb_keycode));
	k->set_pressed(pressed);
	k->set_echo(false);

	if (k->get_keycode() == Key::BACKTAB) {
		//make it consistent across platforms.
		k->set_keycode(Key::TAB);
		k->set_physical_keycode(Key::TAB);
		k->set_shift_pressed(true);
	}

	Input::get_singleton()->parse_input_event(k);
}

void DisplayServerWayland::_wl_keyboard_on_modifiers(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group) {
	KeyboardState *keyboard_state = (KeyboardState*) data;
	xkb_state_update_mask(keyboard_state->xkb_state, mods_depressed, mods_latched, mods_locked, 0, 0, group);
}

void DisplayServerWayland::_wl_keyboard_on_repeat_info(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay) {
}


void DisplayServerWayland::_xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial) {
	xdg_wm_base_pong(xdg_wm_base, serial);
}


void DisplayServerWayland::_xdg_surface_on_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial) {
	xdg_surface_ack_configure(xdg_surface, serial);

	WindowData *window_data = (WindowData*) data;

	if (window_data->buffer_created) {
		xdg_surface_set_window_geometry(window_data->xdg_surface,
						window_data->rect.position.x, window_data->rect.position.y,
						window_data->rect.size.width, window_data->rect.size.height);

		window_data->context_vulkan->window_resize(window_data->id, window_data->rect.size.width, window_data->rect.size.height);

		wl_surface_commit(window_data->wl_surface);

		if (!window_data->rect_changed_callback.is_null()) {
			// FIXME: I'm not sure if this is needed to avoid that the callback
			// function has access the actual window's rect reference.
			Rect2i callback_rect = Rect2i(window_data->rect);
			Variant variant_rect = callback_rect;
			Variant *rectp = &variant_rect;
			Variant ret;
			Callable::CallError ce;

			window_data->rect_changed_callback.call((const Variant **)&rectp, 1, ret, ce);
		}
	}
}

void DisplayServerWayland::_xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states) {
	WindowData *window_data = (WindowData*) data;

	if (width != 0 && height != 0) {
		window_data->rect.size.width = width;
		window_data->rect.size.height = height;
	}
}

// Interface mthods

bool DisplayServerWayland::has_feature(Feature p_feature) const {
	// TODO

	/*
	switch (p_feature) {
		default: {
		}
	}
	*/

	return false;
}

String DisplayServerWayland::get_name() const {
	return "Wayland";
}

void DisplayServerWayland::mouse_set_mode(MouseMode p_mode) {
	// TODO
	print_verbose("wayland stub mouse_set_mode");
}


DisplayServerWayland::MouseMode DisplayServerWayland::mouse_get_mode() const {
	// TODO
	print_verbose("wayland stub mouse_get_mode");
	return MOUSE_MODE_VISIBLE;
}

void DisplayServerWayland::mouse_warp_to_position(const Point2i &p_to) {
	// TODO
	print_verbose("wayland stub mouse_warp_to_position");
}

Point2i DisplayServerWayland::mouse_get_position() const {
	return wls.seat_state.pointer_state.data.position;
}

MouseButton DisplayServerWayland::mouse_get_button_state() const {
	return wls.seat_state.pointer_state.data.pressed_button_mask;
}

void DisplayServerWayland::clipboard_set(const String &p_text) {
	// TODO
	print_verbose("wayland stub clipboard_set");
}

String DisplayServerWayland::clipboard_get() const {
	// TODO
	print_verbose("wayland stub clibpoard_get");
	return "";
}

void DisplayServerWayland::clipboard_set_primary(const String &p_text) {
	// TODO
	print_verbose("wayland stub clibpoard_set_primary");
}

String DisplayServerWayland::clipboard_get_primary() const {
	// TODO
	print_verbose("wayland stub clibpoard_get_primary");
	return "";
}

int DisplayServerWayland::get_screen_count() const {
	// TODO
	print_verbose("wayland stub get_screen_count");
	return 1;
}

Point2i DisplayServerWayland::screen_get_position(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_get_position");
	return Point2i(0, 0);
}

Size2i DisplayServerWayland::screen_get_size(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_get_size");
	return Point2i(1920, 1080);
}

Rect2i DisplayServerWayland::screen_get_usable_rect(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_get_usable_rect");
	return Rect2i(0, 0, 1920, 1080);
}

int DisplayServerWayland::screen_get_dpi(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_get_dpi");
	return 0;
}

bool DisplayServerWayland::screen_is_touchscreen(int p_screen) const {
	// TODO
	print_verbose("wayland stub screen_is_touchscreen");
	return false;
}

#if defined(DBUS_ENABLED)

void DisplayServerWayland::screen_set_keep_on(bool p_enable) {
	// TODO
	print_verbose("wayland stub screen_set_keep_on");
}

bool screen_is_kept_on() {
	// TODO
	print_verbose("wayland stub screen_is_kept_on");
}

#endif

Vector<DisplayServer::WindowID> DisplayServerWayland::get_window_list() const {
	// TODO
	print_verbose("wayland stub get_window_list");

	return Vector<DisplayServer::WindowID>();
}


DisplayServer::WindowID DisplayServerWayland::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	// TODO: Actually create subwindows instead of some broken toplevel window.
	return _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
}

void DisplayServerWayland::show_window(DisplayServer::WindowID p_id) {
	WindowData &wd = wls.windows[p_id];

	ERR_FAIL_COND(!wls.windows.has(p_id));

	if (!wd.buffer_created) {
		// Since `VulkanContextWayland::window_create` automatically assigns a buffer
		// to the `wl_surface` and doing so instantly maps it, moving this method here
		// is the only solution I can think of to implement this method properly.
		context_vulkan->window_create(p_id, wd.vsync_mode, wls.display, wd.wl_surface, wd.rect.size.width, wd.rect.size.height);
		wd.buffer_created = true;
	}
}

void DisplayServerWayland::delete_sub_window(DisplayServer::WindowID p_id) {
	// TODO
	print_verbose("wayland stub delete_sub_window");
}


DisplayServer::WindowID DisplayServerWayland::get_window_at_screen_position(const Point2i &p_position) const {
	// TODO
	print_verbose("wayland stub get_window_at_screen_position");
	return WindowID(0);
}

void DisplayServerWayland::window_attach_instance_id(ObjectID p_instance, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_attach_instance_id");
}


ObjectID DisplayServerWayland::window_get_attached_instance_id(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_attached_instance_id");
	return ObjectID();
}


void DisplayServerWayland::window_set_title(const String &p_title, DisplayServer::WindowID p_window) {
	WindowData &wd = wls.windows[p_window];

	ERR_FAIL_COND(!wls.windows.has(p_window));

	wd.title = p_title;

	xdg_toplevel_set_title(wd.xdg_toplevel, p_title.utf8().get_data());
}

void DisplayServerWayland::window_set_mouse_passthrough(const Vector<Vector2> &p_region, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_mouse_passthrough");
}


void DisplayServerWayland::window_set_rect_changed_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	WindowData &wd = wls.windows[p_window];
	wd.rect_changed_callback = p_callable;
}

void DisplayServerWayland::window_set_window_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_window_event_callback");
}

void DisplayServerWayland::window_set_input_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	WindowData &wd = wls.windows[p_window];
	wd.input_event_callback = p_callable;
}

void DisplayServerWayland::window_set_input_text_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_input_text_callback");
}


void DisplayServerWayland::window_set_drop_files_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_drop_files_callback");
}

int DisplayServerWayland::window_get_current_screen(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_current_screen");
	return 0;
}

void DisplayServerWayland::window_set_current_screen(int p_screen, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_current_screen");
}

Point2i DisplayServerWayland::window_get_position(DisplayServer::WindowID p_window) const {
	return wls.windows[p_window].rect.position;
}

void DisplayServerWayland::window_set_position(const Point2i &p_position, DisplayServer::WindowID p_window) {
	WindowData &wd = wls.windows[p_window];

	wd.rect.position = p_position;

	// FIXME: The size may be changed after a reposition, I believe.
	xdg_surface_set_window_geometry(wd.xdg_surface, wd.rect.position.x, wd.rect.position.y, wd.rect.size.width, wd.rect.size.height);
	wl_surface_commit(wd.wl_surface);
}

void DisplayServerWayland::window_set_max_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_max_size");
}

Size2i DisplayServerWayland::window_get_max_size(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_max_size");
	return Size2i(1920, 1080);
}

void DisplayServerWayland::gl_window_make_current(DisplayServer::WindowID p_window_id) {
	// TODO
	print_verbose("wayland stub gl_window_make_current");
}


void DisplayServerWayland::window_set_transient(DisplayServer::WindowID p_window, DisplayServer::WindowID p_parent) {
	// TODO
	print_verbose("wayland stub window_set_transient");
}

void DisplayServerWayland::window_set_min_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_min_size");
}

Size2i DisplayServerWayland::window_get_min_size(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_min_size");
	return Size2i(0, 0);
}

void DisplayServerWayland::window_set_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	WindowData &wd = wls.windows[p_window];

	wd.rect.size = p_size;

	// FIXME: The position may be changed after a resize, I believe.
	xdg_surface_set_window_geometry(wd.xdg_surface, wd.rect.position.x, wd.rect.position.y, wd.rect.size.width, wd.rect.size.height);

	context_vulkan->window_resize(p_window, wd.rect.size.width, wd.rect.size.height);
	wl_surface_commit(wd.wl_surface);
}

Size2i DisplayServerWayland::window_get_size(DisplayServer::WindowID p_window) const {
	return wls.windows[p_window].rect.size;
}

Size2i DisplayServerWayland::window_get_real_size(DisplayServer::WindowID p_window) const {
	// I don't think there's a way of actually knowing the window size in wayland,
	// other than the one requested by the compositor, which happens to be
	// the one the windows always uses
	return wls.windows[p_window].rect.size;
}

void DisplayServerWayland::window_set_mode(WindowMode p_mode, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_mode");
}

DisplayServerWayland::WindowMode DisplayServerWayland::window_get_mode(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_mode");
	return WINDOW_MODE_WINDOWED;
}

bool DisplayServerWayland::window_is_maximize_allowed(DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_is_maximize_allowed");
	return false;
}

void DisplayServerWayland::window_set_flag(WindowFlags p_flag, bool p_enabled, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_flag");
}

bool DisplayServerWayland::window_get_flag(WindowFlags p_flag, DisplayServer::WindowID p_window) const {
	// TODO
	print_verbose("wayland stub window_get_flag");
	return false;
}

void DisplayServerWayland::window_request_attention(DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_request_attention");
}

void DisplayServerWayland::window_move_to_foreground(DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_move_to_foreground");
}

bool DisplayServerWayland::window_can_draw(DisplayServer::WindowID p_window) const {
	// TODO: Implement this. For now a simple return true will work tough
	return true;
}


bool DisplayServerWayland::can_any_window_draw() const {
	// TODO: Implement this. For now a simple return true will work tough
	return true;
}

void DisplayServerWayland::window_set_ime_active(const bool p_active, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_ime_active");
}

void DisplayServerWayland::window_set_ime_position(const Point2i &p_pos, DisplayServer::WindowID p_window) {
	// TODO
	print_verbose("wayland stub window_set_ime_position");
}


void DisplayServerWayland::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, DisplayServer::WindowID p_window) {
	// TODO: Figure out whether it is possible to disable VSync with Wayland
	// (doubt it) or handle any other mode.
	print_verbose("wayland stub window_set_vsync_mode");
}

DisplayServer::VSyncMode DisplayServerWayland::window_get_vsync_mode(DisplayServer::WindowID p_vsync_mode) const {
	// TODO: Figure out whether it is possible to disable VSync with Wayland
	// (doubt it) or handle any other mode.
	return VSYNC_ENABLED;
}

void DisplayServerWayland::cursor_set_shape(CursorShape p_shape) {
	// TODO
	print_verbose("wayland stub cursor_set_shape");
}

DisplayServerWayland::CursorShape DisplayServerWayland::cursor_get_shape() const {
	// TODO
	print_verbose("wayland stub cursot_get_shape");
	return CURSOR_ARROW;
}

void DisplayServerWayland::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	// TODO
	print_verbose("wayland stub cursor_set_custom_image");
}


int DisplayServerWayland::keyboard_get_layout_count() const {
	// TODO
	print_verbose("wayland stub keyboard_get_layout_count");
	return 0;
}

int DisplayServerWayland::keyboard_get_current_layout() const {
	// TODO
	print_verbose("wayland stub keyboard_get_current_layout");
	return 0;
}

void DisplayServerWayland::keyboard_set_current_layout(int p_index) {
	// TODO
	print_verbose("wayland stub keyboard_set_current_layout");
}

String DisplayServerWayland::keyboard_get_layout_language(int p_index) const {
	// TODO
	print_verbose("wayland stub keyboard_get_layout_language");
	return "";
}

String DisplayServerWayland::keyboard_get_layout_name(int p_index) const {
	// TODO
	print_verbose("wayland stub keyboard_get_layout_name");
	return "";
}

Key DisplayServerWayland::keyboard_get_keycode_from_physical(Key p_keycode) const {
	// TODO
	print_verbose("wayland stub keyboard_get_keycode_from_physical");
	return Key::NONE;
}


void DisplayServerWayland::process_events() {
	// Wait for all events.
	wl_display_dispatch(wls.display);

	PointerState &old_pointer_state = wls.seat_state.old_pointer_state;
	PointerState &pointer_state = wls.seat_state.pointer_state;

	if (old_pointer_state.data.time != pointer_state.data.time && pointer_state.data.focused_wl_surface) {
		WindowID focused_window_id;
		bool id_found = false;
		for (KeyValue<WindowID, WindowData> &E : wls.windows) {
			WindowData &wd = E.value;

			if (wd.wl_surface == pointer_state.data.focused_wl_surface) {
				focused_window_id = E.key;
				id_found = true;
				break;
			}
		}

		ERR_FAIL_COND_MSG(!id_found, "Cursor focused to an invalid window ID.");

		if (old_pointer_state.data.position != pointer_state.data.position) {
			Ref<InputEventMouseMotion> mouse_motion;
			mouse_motion.instantiate();
			mouse_motion->set_window_id(focused_window_id);
			mouse_motion->set_button_mask(pointer_state.data.pressed_button_mask);
			mouse_motion->set_position(pointer_state.data.position);
			// FIXME: We're lying!
			mouse_motion->set_global_position(pointer_state.data.position);
			Input::get_singleton()->set_mouse_position(pointer_state.data.position);

			mouse_motion->set_velocity(Input::get_singleton()->get_last_mouse_velocity());
			mouse_motion->set_relative(pointer_state.data.position - old_pointer_state.data.position);

			Input::get_singleton()->parse_input_event(mouse_motion);
		}

		if (old_pointer_state.data.pressed_button_mask != pointer_state.data.pressed_button_mask) {
			MouseButton pressed_mask_delta = old_pointer_state.data.pressed_button_mask ^ pointer_state.data.pressed_button_mask;

			// TODO: Simplify with a function or something.
			if ((pressed_mask_delta & MouseButton::MASK_LEFT) != MouseButton::NONE) {
				Ref<InputEventMouseButton> mouse_button;
				mouse_button.instantiate();
				mouse_button->set_window_id(focused_window_id);
				mouse_button->set_position(pointer_state.data.position);
				// FIXME: We're lying!
				mouse_button->set_global_position(pointer_state.data.position);
				mouse_button->set_button_mask(mouse_get_button_state());

				mouse_button->set_button_index(MouseButton::LEFT);
				mouse_button->set_pressed((pointer_state.data.pressed_button_mask & MouseButton::MASK_LEFT) != MouseButton::NONE);

				Input::get_singleton()->parse_input_event(mouse_button);
			}
			if ((pressed_mask_delta & MouseButton::MASK_MIDDLE) != MouseButton::NONE) {
				Ref<InputEventMouseButton> mouse_button;
				mouse_button.instantiate();
				mouse_button->set_window_id(focused_window_id);
				mouse_button->set_position(pointer_state.data.position);
				// FIXME: We're lying!
				mouse_button->set_global_position(pointer_state.data.position);
				mouse_button->set_button_mask(mouse_get_button_state());

				mouse_button->set_button_index(MouseButton::MIDDLE);
				mouse_button->set_pressed((pointer_state.data.pressed_button_mask & MouseButton::MASK_MIDDLE) != MouseButton::NONE);

				Input::get_singleton()->parse_input_event(mouse_button);
			}
			if ((pressed_mask_delta & MouseButton::MASK_RIGHT) != MouseButton::NONE) {
				Ref<InputEventMouseButton> mouse_button;
				mouse_button.instantiate();
				mouse_button->set_window_id(focused_window_id);
				mouse_button->set_position(pointer_state.data.position);
				// FIXME: We're lying!
				mouse_button->set_global_position(pointer_state.data.position);
				mouse_button->set_button_mask(mouse_get_button_state());

				mouse_button->set_button_index(MouseButton::RIGHT);
				mouse_button->set_pressed((pointer_state.data.pressed_button_mask & MouseButton::MASK_RIGHT) != MouseButton::NONE);

				Input::get_singleton()->parse_input_event(mouse_button);
			}

		}

		wls.seat_state.old_pointer_state = pointer_state;
	}

	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerWayland::release_rendering_thread() {
	// TODO
	print_verbose("wayland stub release_rendering_thread");
}

void DisplayServerWayland::make_rendering_thread() {
	// TODO
	print_verbose("wayland stub make_rendering_thread");
}

void DisplayServerWayland::swap_buffers() {
	// TODO
	print_verbose("wayland stub swap_buffers");
}


void DisplayServerWayland::set_context(Context p_context) {
	// TODO
	print_verbose("wayland stub set_context");
}


void DisplayServerWayland::set_native_icon(const String &p_filename) {
	// TODO
	print_verbose("wayland stub set_native_icon");
}

void DisplayServerWayland::set_icon(const Ref<Image> &p_icon) {
	// TODO
	print_verbose("wayland stub set_icon");
}

Vector<String> DisplayServerWayland::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif

// TODO
/*
 * #ifdef GLES3_ENABLED
 * 	drivers.push_back("opengl3");
 * #endif
 */

	return drivers;
}

DisplayServer *DisplayServerWayland::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerWayland(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported Vulkan or OpenGL versions.\n"
			"Please update your drivers or if you have a very old or integrated GPU, upgrade it.\n"
			"If you have updated your graphics drivers recently, try rebooting.",
			"Unable to initialize Video driver");
	}
	return ds;
}

DisplayServerWayland::DisplayServerWayland(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	wls.display = wl_display_connect(nullptr);

	// TODO: Better error handling.
	ERR_FAIL_COND(!wls.display);

	wls.registry = wl_display_get_registry(wls.display);

	// TODO: Better error handling.
	ERR_FAIL_COND(!wls.display);

	wl_registry_add_listener(wls.registry, &registry_listener, &wls);

	// Wait for globals to get notified from the compositor.
	wl_display_roundtrip(wls.display);

	ERR_FAIL_COND(!wls.globals.wl_compositor || !wls.globals.wl_seat || !wls.globals.xdg_wm_base);

	// Input.
	Input::get_singleton()->set_event_dispatch_function(dispatch_input_events);

	// Wait for seat capabilities.
	wl_display_roundtrip(wls.display);

	xdg_wm_base_add_listener(wls.globals.xdg_wm_base, &xdg_wm_base_listener, nullptr);

#if defined(VULKAN_ENABLED)
	if (p_rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextWayland);

		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan");
		}
	}
#endif

	WindowID main_window_id =_create_window(p_mode, p_vsync_mode, p_flags, screen_get_usable_rect());
	show_window(main_window_id);

#ifdef VULKAN_ENABLED
	if (p_rendering_driver == "vulkan") {
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}

	r_error = OK;
#endif
}

DisplayServerWayland::~DisplayServerWayland() {
	// Destroy all windows.
	for (KeyValue<WindowID, WindowData> &E : wls.windows) {
#ifdef VULKAN_ENABLED
		if (context_vulkan) {
			context_vulkan->window_destroy(E.key);
		}
#endif
		WindowData &wd = E.value;

		if (wd.xdg_toplevel) {
			xdg_toplevel_destroy(wd.xdg_toplevel);
		}

		if (wd.xdg_surface) {
			xdg_surface_destroy(wd.xdg_surface);
		}

		if (wd.wl_surface) {
			wl_surface_destroy(wd.wl_surface);
		}
	}


	// Destroy all drivers.
#ifdef VULKAN_ENABLED
	if (rendering_device_vulkan) {
		rendering_device_vulkan->finalize();
		memdelete(rendering_device_vulkan);
		rendering_device_vulkan = nullptr;
	}

	if (context_vulkan) {
		memdelete(context_vulkan);
		context_vulkan = nullptr;
	}
#endif
}

void DisplayServerWayland::register_wayland_driver() {
	register_create_function("wayland", create_func, get_rendering_drivers_func);
}

#endif //WAYLAND_ENABLED
