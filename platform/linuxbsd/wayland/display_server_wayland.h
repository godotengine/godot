/*************************************************************************/
/*  display_server_wayland.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DISPLAY_SERVER_WAYLAND_H
#define DISPLAY_SERVER_WAYLAND_H

#ifdef WAYLAND_ENABLED

// FIXME: Linux only?
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include "key_mapping_xkb.h"
#include "servers/display_server.h"

#include "wayland-cursor.h"

#include "protocol/idle_inhibit.gen.h"
#include "protocol/pointer_constraints.gen.h"
#include "protocol/primary_selection.gen.h"
#include "protocol/relative_pointer.gen.h"
#include "protocol/wayland.gen.h"
#include "protocol/xdg_decoration.gen.h"
#include "protocol/xdg_shell.gen.h"

// FIXME: Since this platform is called linuxbsd, can we avoid this include?
#include "linux/input-event-codes.h"

#ifdef VULKAN_ENABLED
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "vulkan_context_wayland.h"
#endif

#ifdef GLES3_ENABLED
#include "gl_manager_wayland.h"
#endif

#if defined(SPEECHD_ENABLED)
#include "../tts_linux.h"
#endif

#ifdef DBUS_ENABLED
#include "../freedesktop_portal_desktop.h"
#include "../freedesktop_screensaver.h"
#endif

#include "core/input/input.h"
#include "core/os/thread.h"

#include "scene/resources/texture.h"

#undef CursorShape

// Fix the wl_array_for_each macro to work with C++. This is based on the
// original from `wayland-util.h` in the Wayland client library.
#undef wl_array_for_each
#define wl_array_for_each(pos, array) \
	for (pos = (decltype(pos))(array)->data; (const char *)pos < ((const char *)(array)->data + (array)->size); (pos)++)

class DisplayServerWayland : public DisplayServer {
	// Wayland stuff.

	// Messages used for sending stuff to the main thread to be dispatched there.
	class WaylandMessage : public RefCounted {
	public:
		WaylandMessage() {}
		virtual ~WaylandMessage() = default;
	};

	// WaylandMessage data for window rect changes.
	class WaylandWindowRectMessage : public WaylandMessage {
	public:
		WindowID id;
		Rect2i rect;
	};

	class WaylandWindowEventMessage : public WaylandMessage {
	public:
		WindowID id;
		WindowEvent event;
	};

	class WaylandInputEventMessage : public WaylandMessage {
	public:
		Ref<InputEvent> event;
	};

	class WaylandDropFilesEventMessage : public WaylandMessage {
	public:
		WindowID id;
		Vector<String> files;
	};

	struct WaylandGlobals {
		struct wl_shm *wl_shm = nullptr;
		uint32_t wl_shm_name = 0;

		struct wl_compositor *wl_compositor = nullptr;
		uint32_t wl_compositor_name = 0;

		struct wl_subcompositor *wl_subcompositor = nullptr;
		uint32_t wl_subcompositor_name = 0;

		struct wl_data_device_manager *wl_data_device_manager = nullptr;
		uint32_t wl_data_device_manager_name = 0;

		struct xdg_wm_base *xdg_wm_base = nullptr;
		uint32_t xdg_wm_base_name = 0;

		struct zxdg_decoration_manager_v1 *xdg_decoration_manager = nullptr;
		uint32_t xdg_decoration_manager_name = 0;

		struct zwp_primary_selection_device_manager_v1 *wp_primary_selection_device_manager = nullptr;
		uint32_t wp_primary_selection_device_manager_name = 0;

		struct zwp_relative_pointer_manager_v1 *wp_relative_pointer_manager = nullptr;
		uint32_t wp_relative_pointer_manager_name = 0;

		struct zwp_pointer_constraints_v1 *wp_pointer_constraints = nullptr;
		uint32_t wp_pointer_constraints_name = 0;

		struct zwp_idle_inhibit_manager_v1 *wp_idle_inhibit_manager = nullptr;
		uint32_t wp_idle_inhibit_manager_name = 0;
	};

	// This forward declaration is needed due to some circular dependencies.
	// The actual struct members are declared later in this header.
	struct WaylandState;

	struct WindowData {
		// For use in event sending and whatnot.
		WindowID id;
		WaylandState *wls;

		struct wl_surface *wl_surface = nullptr;
		struct xdg_surface *xdg_surface = nullptr;
		struct xdg_toplevel *xdg_toplevel = nullptr;

		struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration = nullptr;

		struct wl_subsurface *wl_subsurface = nullptr;

		HashSet<WindowID> children;
		WindowID parent = INVALID_WINDOW_ID;

		bool visible = false;

		WindowMode mode;
		uint32_t flags;
		VSyncMode vsync_mode;
		Rect2i rect;
		Rect2i safe_rect;
		Size2i max_size;
		Size2i min_size;

		Callable rect_changed_callback;
		Callable window_event_callback;
		Callable input_event_callback;
		Callable drop_files_callback;

		// Metadata.
		String title;

		ObjectID instance_id;
	};

	struct ScreenData {
		struct wl_output *wl_output = nullptr;
		uint32_t wl_output_name = 0;

		// Geometry data.
		Point2i position;

		String make;
		String model;

		Size2i size;
		Size2i physical_size;

		float refresh_rate = 0;
		int scale = 0;
	};

	struct PointerData {
		Point2i position;
		uint32_t motion_time = 0;

		// Relative motion has its own optional event and so needs its own time.
		Vector2 relative_motion;
		uint32_t relative_motion_time = 0;

		WindowID pointed_window_id = INVALID_WINDOW_ID;

		MouseButton pressed_button_mask = MouseButton::NONE;

		MouseButton last_button_pressed = MouseButton::NONE;
		uint32_t button_time = 0;
		uint32_t button_serial = 0;

		Vector2 scroll_vector;
	};

	struct SeatState {
		WaylandState *wls = nullptr;

		struct wl_seat *wl_seat = nullptr;
		uint32_t wl_seat_name = 0;

		// Pointer.
		struct wl_pointer *wl_pointer = nullptr;

		struct zwp_relative_pointer_v1 *wp_relative_pointer = nullptr;
		struct zwp_locked_pointer_v1 *wp_locked_pointer = nullptr;
		struct zwp_confined_pointer_v1 *wp_confined_pointer = nullptr;

		struct wl_surface *cursor_surface = nullptr;

		// This variable is needed to buffer all pointer changes until a
		// wl_pointer.frame event, as per Wayland's specification. Everything is
		// first set in `data_buffer` and then `data` is set with its contents on
		// an input frame event. All methods should generally read from
		// `pointer_data` and write to `data_buffer`.
		PointerData pointer_data_buffer;
		PointerData pointer_data;

		// Keyboard.
		struct wl_keyboard *wl_keyboard = nullptr;

		struct xkb_context *xkb_context = nullptr;
		struct xkb_keymap *xkb_keymap = nullptr;
		struct xkb_state *xkb_state = nullptr;

		const char *keymap_buffer = nullptr;
		uint32_t keymap_buffer_size = 0;

		xkb_layout_index_t current_layout_index = 0;

		WindowID keyboard_focused_window_id = INVALID_WINDOW_ID;

		int32_t repeat_key_delay_msec = 0;
		int32_t repeat_start_delay_msec = 0;

		xkb_keycode_t repeating_keycode = XKB_KEYCODE_INVALID;
		uint64_t last_repeat_start_msec = 0;
		uint64_t last_repeat_msec = 0;

		bool shift_pressed = false;
		bool ctrl_pressed = false;
		bool alt_pressed = false;
		bool meta_pressed = false;

		uint32_t last_key_pressed_serial = 0;

		struct wl_data_device *wl_data_device = nullptr;

		// Drag and drop.
		struct wl_data_offer *wl_data_offer_dnd = nullptr;
		uint32_t dnd_enter_serial = 0;

		WindowID dnd_current_window_id = INVALID_WINDOW_ID;

		// Clipboard.
		struct wl_data_source *wl_data_source_selection = nullptr;
		struct wl_data_offer *wl_data_offer_selection = nullptr;

		Vector<uint8_t> selection_data;

		// Primary selection.
		struct zwp_primary_selection_device_v1 *wp_primary_selection_device = nullptr;

		struct zwp_primary_selection_source_v1 *wp_primary_selection_source = nullptr;
		struct zwp_primary_selection_offer_v1 *wp_primary_selection_offer = nullptr;

		Vector<uint8_t> primary_data;
	};

	struct CustomWaylandCursor {
		struct wl_buffer *wl_buffer = nullptr;
		uint32_t *buffer_data = nullptr;
		uint32_t buffer_data_size = 0;

		RID cursor_rid;
		Point2i hotspot;
	};

	struct WaylandState {
		Mutex mutex;

		struct wl_display *wl_display = nullptr;
		struct wl_registry *wl_registry = nullptr;

		WaylandGlobals globals;

		WindowID window_id_counter = MAIN_WINDOW_ID;
		HashMap<WindowID, WindowData> windows;

		SeatState *current_seat = nullptr;

		struct wl_cursor_theme *wl_cursor_theme = nullptr;
		struct wl_cursor_image *cursor_images[CURSOR_MAX] = {};
		struct wl_buffer *cursor_bufs[CURSOR_MAX] = {};

		HashMap<CursorShape, CustomWaylandCursor> custom_cursors;

		CursorShape cursor_shape = CURSOR_ARROW;
		MouseMode mouse_mode = MOUSE_MODE_VISIBLE;

		List<ScreenData> screens;
		List<SeatState> seats;

		List<WindowID> popup_list;

		SafeFlag events_thread_done;

		List<Ref<WaylandMessage>> message_queue;

		struct zwp_idle_inhibitor_v1 *wp_idle_inhibitor = nullptr;

#ifdef VULKAN_ENABLED
		VulkanContextWayland *context_vulkan = nullptr;
		RenderingDeviceVulkan *rendering_device_vulkan = nullptr;
#endif

#ifdef GLES3_ENABLED
		GLManagerWayland *gl_manager = nullptr;
#endif
	};

	WaylandState wls;

	Thread events_thread;

#ifdef SPEECHD_ENABLED
	TTS_Linux *tts = nullptr;
#endif

#if DBUS_ENABLED
	FreeDesktopPortalDesktop *portal_desktop = nullptr;

	FreeDesktopScreenSaver *screensaver = nullptr;
	bool screensaver_inhibited = false;
#endif

	static String _string_read_fd(int fd);

	String _wl_data_offer_read(wl_data_offer *wl_data_offer) const;
	String _wp_primary_selection_offer_read(zwp_primary_selection_offer_v1 *wp_primary_selection_offer) const;

	static void _seat_state_override_cursor_shape(SeatState &p_ss, CursorShape p_shape);
	static void _seat_state_set_current(SeatState &p_ss);
	static void _wayland_state_update_cursor(WaylandState &p_wls);
	static Point2i _wayland_state_point_window_to_global(const WaylandState &wls, WindowID p_window, Point2i p_position);

	static void _get_key_modifier_state(SeatState &p_seat, Ref<InputEventWithModifiers> p_event);

	static bool _seat_state_configure_key_event(SeatState &p_seat, Ref<InputEventKey> p_event, xkb_keycode_t p_keycode, bool p_pressed);

	WindowID _create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect);
	void _window_data_set_mode(WindowData &p_wd, WindowMode p_mode);
	void _send_window_event(WindowID p_window, WindowEvent p_event);

	static int _allocate_shm_file(size_t size);

	static void _poll_events_thread(void *p_wls);

	static void dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);

	// Wayland event handlers.
	static void _wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version);
	static void _wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name);

	static void _wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output);
	static void _wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output);

	static void _wl_output_on_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform);
	static void _wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh);
	static void _wl_output_on_done(void *data, struct wl_output *wl_output);
	static void _wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor);

	static void _wl_seat_on_capabilities(void *data, struct wl_seat *wl_seat, uint32_t capabilities);
	static void _wl_seat_on_name(void *data, struct wl_seat *wl_seat, const char *name);

	static void _wl_pointer_on_enter(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface, wl_fixed_t surface_x, wl_fixed_t surface_y);
	static void _wl_pointer_on_leave(void *data, struct wl_pointer *wl_pointer, uint32_t serial, struct wl_surface *surface);
	static void _wl_pointer_on_motion(void *data, struct wl_pointer *wl_pointer, uint32_t time, wl_fixed_t surface_x, wl_fixed_t surface_y);
	static void _wl_pointer_on_button(void *data, struct wl_pointer *wl_pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state);
	static void _wl_pointer_on_axis(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis, wl_fixed_t value);
	static void _wl_pointer_on_frame(void *data, struct wl_pointer *wl_pointer);
	static void _wl_pointer_on_axis_source(void *data, struct wl_pointer *wl_pointer, uint32_t axis_source);
	static void _wl_pointer_on_axis_stop(void *data, struct wl_pointer *wl_pointer, uint32_t time, uint32_t axis);
	static void _wl_pointer_on_axis_discrete(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t discrete);

	static void _wl_keyboard_on_keymap(void *data, struct wl_keyboard *wl_keyboard, uint32_t format, int32_t fd, uint32_t size);
	static void _wl_keyboard_on_enter(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface, struct wl_array *keys);
	static void _wl_keyboard_on_leave(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, struct wl_surface *surface);
	static void _wl_keyboard_on_key(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t time, uint32_t key, uint32_t state);
	static void _wl_keyboard_on_modifiers(void *data, struct wl_keyboard *wl_keyboard, uint32_t serial, uint32_t mods_depressed, uint32_t mods_latched, uint32_t mods_locked, uint32_t group);
	static void _wl_keyboard_on_repeat_info(void *data, struct wl_keyboard *wl_keyboard, int32_t rate, int32_t delay);

	static void _wl_data_device_on_data_offer(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id);
	static void _wl_data_device_on_enter(void *data, struct wl_data_device *wl_data_device, uint32_t serial, struct wl_surface *surface, wl_fixed_t x, wl_fixed_t y, struct wl_data_offer *id);
	static void _wl_data_device_on_leave(void *data, struct wl_data_device *wl_data_device);
	static void _wl_data_device_on_motion(void *data, struct wl_data_device *wl_data_device, uint32_t time, wl_fixed_t x, wl_fixed_t y);
	static void _wl_data_device_on_drop(void *data, struct wl_data_device *wl_data_device);
	static void _wl_data_device_on_selection(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id);

	static void _wl_data_offer_on_offer(void *data, struct wl_data_offer *wl_data_offer, const char *mime_type);
	static void _wl_data_offer_on_source_actions(void *data, struct wl_data_offer *wl_data_offer, uint32_t source_actions);
	static void _wl_data_offer_on_action(void *data, struct wl_data_offer *wl_data_offer, uint32_t dnd_action);

	static void _wl_data_source_on_target(void *data, struct wl_data_source *wl_data_source, const char *mime_type);
	static void _wl_data_source_on_send(void *data, struct wl_data_source *wl_data_source, const char *mime_type, int32_t fd);
	static void _wl_data_source_on_cancelled(void *data, struct wl_data_source *wl_data_source);
	static void _wl_data_source_on_dnd_drop_performed(void *data, struct wl_data_source *wl_data_source);
	static void _wl_data_source_on_dnd_finished(void *data, struct wl_data_source *wl_data_source);
	static void _wl_data_source_on_action(void *data, struct wl_data_source *wl_data_source, uint32_t dnd_action);

	// xdg-shell event handlers.
	static void _xdg_wm_base_on_ping(void *data, struct xdg_wm_base *xdg_wm_base, uint32_t serial);
	static void _xdg_surface_on_configure(void *data, struct xdg_surface *xdg_surface, uint32_t serial);

	static void _xdg_toplevel_on_configure(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height, struct wl_array *states);
	static void _xdg_toplevel_on_close(void *data, struct xdg_toplevel *xdg_toplevel);

	static void _xdg_popup_on_configure(void *data, struct xdg_popup *xdg_popup, int32_t x, int32_t y, int32_t width, int32_t height);
	static void _xdg_popup_on_popup_done(void *data, struct xdg_popup *xdg_popup);
	static void _xdg_popup_on_repositioned(void *data, struct xdg_popup *xdg_popup, uint32_t token);

	// wayland-protocols event handlers.
	static void _xdg_toplevel_decoration_on_configure(void *data, struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration, uint32_t mode);

	static void _wp_relative_pointer_on_relative_motion(void *data, struct zwp_relative_pointer_v1 *wp_relative_pointer_v1, uint32_t uptime_hi, uint32_t uptime_lo, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel);

	static void _wp_primary_selection_device_on_data_offer(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *offer);
	static void _wp_primary_selection_device_on_selection(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *id);

	static void _wp_primary_selection_source_on_send(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1, const char *mime_type, int32_t fd);
	static void _wp_primary_selection_source_on_cancelled(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1);

	// Wayland event listeners.
	static constexpr struct wl_registry_listener wl_registry_listener = {
		.global = _wl_registry_on_global,
		.global_remove = _wl_registry_on_global_remove,
	};

	static constexpr struct wl_surface_listener wl_surface_listener = {
		.enter = _wl_surface_on_enter,
		.leave = _wl_surface_on_leave,
	};

	static constexpr struct wl_output_listener wl_output_listener = {
		.geometry = _wl_output_on_geometry,
		.mode = _wl_output_on_mode,
		.done = _wl_output_on_done,
		.scale = _wl_output_on_scale,
	};

	static constexpr struct wl_seat_listener wl_seat_listener = {
		.capabilities = _wl_seat_on_capabilities,
		.name = _wl_seat_on_name,
	};

	static constexpr struct wl_pointer_listener wl_pointer_listener = {
		.enter = _wl_pointer_on_enter,
		.leave = _wl_pointer_on_leave,
		.motion = _wl_pointer_on_motion,
		.button = _wl_pointer_on_button,
		.axis = _wl_pointer_on_axis,
		.frame = _wl_pointer_on_frame,
		.axis_source = _wl_pointer_on_axis_source,
		.axis_stop = _wl_pointer_on_axis_stop,
		.axis_discrete = _wl_pointer_on_axis_discrete,
	};

	static constexpr struct wl_keyboard_listener wl_keyboard_listener = {
		.keymap = _wl_keyboard_on_keymap,
		.enter = _wl_keyboard_on_enter,
		.leave = _wl_keyboard_on_leave,
		.key = _wl_keyboard_on_key,
		.modifiers = _wl_keyboard_on_modifiers,
		.repeat_info = _wl_keyboard_on_repeat_info,
	};

	static constexpr struct wl_data_device_listener wl_data_device_listener = {
		.data_offer = _wl_data_device_on_data_offer,
		.enter = _wl_data_device_on_enter,
		.leave = _wl_data_device_on_leave,
		.motion = _wl_data_device_on_motion,
		.drop = _wl_data_device_on_drop,
		.selection = _wl_data_device_on_selection,
	};

	static constexpr struct wl_data_offer_listener wl_data_offer_listener = {
		.offer = _wl_data_offer_on_offer,
		.source_actions = _wl_data_offer_on_source_actions,
		.action = _wl_data_offer_on_action,
	};

	static constexpr struct wl_data_source_listener wl_data_source_listener = {
		.target = _wl_data_source_on_target,
		.send = _wl_data_source_on_send,
		.cancelled = _wl_data_source_on_cancelled,
		.dnd_drop_performed = _wl_data_source_on_dnd_drop_performed,
		.dnd_finished = _wl_data_source_on_dnd_finished,
		.action = _wl_data_source_on_action,
	};

	// xdg-shell event listeners.
	static constexpr struct xdg_wm_base_listener xdg_wm_base_listener = {
		.ping = _xdg_wm_base_on_ping,
	};

	static constexpr struct xdg_surface_listener xdg_surface_listener = {
		.configure = _xdg_surface_on_configure,
	};

	static constexpr struct xdg_toplevel_listener xdg_toplevel_listener = {
		.configure = _xdg_toplevel_on_configure,
		.close = _xdg_toplevel_on_close,
	};

	static constexpr struct xdg_popup_listener xdg_popup_listener = {
		.configure = _xdg_popup_on_configure,
		.popup_done = _xdg_popup_on_popup_done,
		.repositioned = _xdg_popup_on_repositioned,
	};

	// wayland-protocols event listeners.
	static constexpr struct zxdg_toplevel_decoration_v1_listener xdg_toplevel_decoration_listener = {
		.configure = _xdg_toplevel_decoration_on_configure,
	};

	static constexpr struct zwp_relative_pointer_v1_listener wp_relative_pointer_listener = {
		.relative_motion = _wp_relative_pointer_on_relative_motion,
	};

	static constexpr struct zwp_primary_selection_device_v1_listener wp_primary_selection_device_listener = {
		.data_offer = _wp_primary_selection_device_on_data_offer,
		.selection = _wp_primary_selection_device_on_selection,
	};

	static constexpr struct zwp_primary_selection_source_v1_listener wp_primary_selection_source_listener = {
		.send = _wp_primary_selection_source_on_send,
		.cancelled = _wp_primary_selection_source_on_cancelled,
	};

public:
	virtual bool has_feature(Feature p_feature) const override;

	virtual String get_name() const override;

#ifdef SPEECHD_ENABLED
	virtual bool tts_is_speaking() const override;
	virtual bool tts_is_paused() const override;
	virtual TypedArray<Dictionary> tts_get_voices() const override;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int p_utterance_id = 0, bool p_interrupt = false) override;
	virtual void tts_pause() override;
	virtual void tts_resume() override;
	virtual void tts_stop() override;
#endif

#ifdef DBUS_ENABLED
	virtual bool is_dark_mode_supported() const override;
	virtual bool is_dark_mode() const override;
#endif

	virtual void mouse_set_mode(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode() const override;

	virtual void warp_mouse(const Point2i &p_to) override;
	virtual Point2i mouse_get_position() const override;
	virtual MouseButton mouse_get_button_state() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;
	virtual void clipboard_set_primary(const String &p_text) override;
	virtual String clipboard_get_primary() const override;

	virtual int get_screen_count() const override;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_refresh_rate(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	virtual void screen_set_keep_on(bool p_enable) override;
	virtual bool screen_is_kept_on() const override;

	virtual Vector<DisplayServer::WindowID> get_window_list() const override;

	virtual WindowID create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i()) override;
	virtual void show_window(WindowID p_id) override;
	virtual void delete_sub_window(WindowID p_id) override;

	virtual WindowID window_get_active_popup() const override;
	virtual void window_set_popup_safe_rect(WindowID p_window, const Rect2i &p_rect) override;
	virtual Rect2i window_get_popup_safe_rect(WindowID p_window) const override;

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void gl_window_make_current(DisplayServer::WindowID p_window_id) override;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent) override;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool can_any_window_draw() const override;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual DisplayServer::VSyncMode window_get_vsync_mode(WindowID p_vsync_mode) const override;

	virtual void cursor_set_shape(CursorShape p_shape) override;
	virtual CursorShape cursor_get_shape() const override;
	virtual void cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) override;

	virtual int keyboard_get_layout_count() const override;
	virtual int keyboard_get_current_layout() const override;
	virtual void keyboard_set_current_layout(int p_index) override;
	virtual String keyboard_get_layout_language(int p_index) const override;
	virtual String keyboard_get_layout_name(int p_index) const override;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const override;

	virtual void process_events() override;

	virtual void release_rendering_thread() override;
	virtual void make_rendering_thread() override;
	virtual void swap_buffers() override;

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	static void register_wayland_driver();

	DisplayServerWayland(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerWayland();
};

#endif // WAYLAND_ENABLED

#endif // DISPLAY_SERVER_WAYLAND_H
