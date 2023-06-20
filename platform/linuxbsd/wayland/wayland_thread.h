/**************************************************************************/
/*  wayland_thread.h                                                      */
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

#ifndef WAYLAND_THREAD_H
#define WAYLAND_THREAD_H

#ifdef SOWRAP_ENABLED
#include "dynwrappers/wayland-client-core-so_wrap.h"
#include "dynwrappers/wayland-cursor-so_wrap.h"
#include "dynwrappers/wayland-egl-core-so_wrap.h"

#include "../xkbcommon-so_wrap.h"
#else
#include <wayland-client-core.h>
#include <wayland-cursor.h>
#include <xkbcommon/xkbcommon.h>
#endif // SOWRAP_ENABLED

#ifdef LIBDECOR_ENABLED
#ifdef SOWRAP_ENABLED
#include "dynwrappers/libdecor-so_wrap.h"
#else
#include <libdecor-0/libdecor.h>
#endif // SOWRAP_ENABLED
#endif // LIBDECOR_ENABLED

#include "protocol/idle_inhibit.gen.h"
#include "protocol/pointer_constraints.gen.h"
#include "protocol/pointer_gestures.gen.h"
#include "protocol/primary_selection.gen.h"
#include "protocol/relative_pointer.gen.h"
#include "protocol/tablet.gen.h"
#include "protocol/wayland.gen.h"
#include "protocol/xdg_activation.gen.h"
#include "protocol/xdg_decoration.gen.h"
#include "protocol/xdg_shell.gen.h"

#include "servers/display_server.h"

#include "core/os/thread.h"

#ifdef WAYLAND_ENABLED

class WaylandThread {
public:
	// Messages used for exchanging information between Godot's and Wayland's thread.
	class Message : public RefCounted {
	public:
		Message() {}
		virtual ~Message() = default;
	};

	// Message data for window rect changes.
	class WindowRectMessage : public Message {
	public:
		// NOTE: This is in "scaled" terms. For example, if there's a 1920x1080 rect
		// with a scale factor of 2, the actual value of `rect` will be 3840x2160.
		Rect2i rect;
	};

	class WindowEventMessage : public Message {
	public:
		DisplayServer::WindowEvent event;
	};

	class InputEventMessage : public Message {
	public:
		Ref<InputEvent> event;
	};

	class DropFilesEventMessage : public Message {
	public:
		Vector<String> files;
	};

	struct RegistryState {
		WaylandThread *wayland_thread;

		// Core Wayland globals.
		struct wl_shm *wl_shm = nullptr;
		uint32_t wl_shm_name = 0;

		struct wl_compositor *wl_compositor = nullptr;
		uint32_t wl_compositor_name = 0;

		struct wl_subcompositor *wl_subcompositor = nullptr;
		uint32_t wl_subcompositor_name = 0;

		struct wl_data_device_manager *wl_data_device_manager = nullptr;
		uint32_t wl_data_device_manager_name = 0;

		List<struct wl_output *> wl_outputs;
		List<struct wl_seat *> wl_seats;

		// xdg-shell globals.

		struct xdg_wm_base *xdg_wm_base = nullptr;
		uint32_t xdg_wm_base_name = 0;

		// wayland-protocols globals.

		struct zxdg_decoration_manager_v1 *xdg_decoration_manager = nullptr;
		uint32_t xdg_decoration_manager_name = 0;

		struct xdg_activation_v1 *xdg_activation = nullptr;
		uint32_t xdg_activation_name = 0;

		struct zwp_primary_selection_device_manager_v1 *wp_primary_selection_device_manager = nullptr;
		uint32_t wp_primary_selection_device_manager_name = 0;

		struct zwp_relative_pointer_manager_v1 *wp_relative_pointer_manager = nullptr;
		uint32_t wp_relative_pointer_manager_name = 0;

		struct zwp_pointer_constraints_v1 *wp_pointer_constraints = nullptr;
		uint32_t wp_pointer_constraints_name = 0;

		struct zwp_pointer_gestures_v1 *wp_pointer_gestures = nullptr;
		uint32_t wp_pointer_gestures_name = 0;

		struct zwp_idle_inhibit_manager_v1 *wp_idle_inhibit_manager = nullptr;
		uint32_t wp_idle_inhibit_manager_name = 0;

		struct zwp_tablet_manager_v2 *wp_tablet_manager = nullptr;
		uint32_t wp_tablet_manager_name = 0;
	};

	// General Wayland-specific states. Shouldn't be accessed directly.
	// TODO: Make private?

	struct WindowState {
		DisplayServer::WindowID id;

		Rect2i rect;
		DisplayServer::WindowMode mode = DisplayServer::WINDOW_MODE_WINDOWED;

		// These are true by default as it isn't guaranteed that we'll find an
		// xdg-shell implementation with wm_capabilities available. If and once we
		// receive a wm_capabilities event these will get reset and updated with
		// whatever the compositor says.
		bool can_minimize = false;
		bool can_maximize = false;
		bool can_fullscreen = false;

		struct wl_output *wl_output = nullptr;

		struct wl_surface *wl_surface = nullptr;
		struct xdg_surface *xdg_surface = nullptr;
		struct xdg_toplevel *xdg_toplevel = nullptr;

		struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration = nullptr;

		struct zwp_idle_inhibitor_v1 *wp_idle_inhibitor = nullptr;

#ifdef LIBDECOR_ENABLED
		// If this is null the xdg_* variables must be set and vice-versa. This way we
		// can handle this mess gracefully enough to hopefully being able of getting
		// rid of this cleanly once we have our own CSDs.
		struct libdecor_frame *libdecor_frame = nullptr;
#endif

		RegistryState *registry;
		WaylandThread *wayland_thread;
	};

	// "High level" Godot-side screen data.
	struct ScreenData {
		// Geometry data.
		Point2i position;

		String make;
		String model;

		Size2i size;
		Size2i physical_size;

		float refresh_rate = -1;
		int scale = 1;
	};

	struct ScreenState {
		uint32_t wl_output_name = 0;

		ScreenData pending_data;
		ScreenData data;
	};

	// BEGIN DISPLAYSERVERWAYLAND STUFF DUMP

	// Because circular dependencies.
	struct WaylandState;

	enum class Gesture {
		NONE,
		MAGNIFY,
	};

	enum class PointerConstraint {
		NONE,
		LOCKED,
		CONFINED,
	};

	struct PointerData {
		Point2i position;
		uint32_t motion_time = 0;

		// Relative motion has its own optional event and so needs its own time.
		Vector2 relative_motion;
		uint32_t relative_motion_time = 0;

		BitField<MouseButtonMask> pressed_button_mask;

		MouseButton last_button_pressed = MouseButton::NONE;
		Point2i last_pressed_position;

		// This is needed to check for a new double click every time.
		bool double_click_begun = false;

		uint32_t button_time = 0;
		uint32_t button_serial = 0;

		uint32_t scroll_type = WL_POINTER_AXIS_SOURCE_WHEEL;

		// The amount "scrolled" in pixels, in each direction.
		Vector2 scroll_vector;

		// The amount of scroll "clicks" in each direction.
		Vector2i discrete_scroll_vector;

		uint32_t pinch_scale = 1;
	};

	struct TabletToolData {
		Point2i position;
		Vector2i tilt;
		uint32_t pressure = 0;

		BitField<MouseButtonMask> pressed_button_mask;

		MouseButton last_button_pressed = MouseButton::NONE;
		Point2i last_pressed_position;

		bool double_click_begun = false;

		// Note: the protocol doesn't have it (I guess that this isn't really meant to
		// be used as a mouse...), but we'll hack one in with the current ticks.
		uint64_t button_time = 0;

		bool is_eraser = false;

		bool in_proximity = false;
		bool touching = false;
	};

	struct SeatState {
		WaylandThread::WaylandState *wls = nullptr;

		RegistryState *registry = nullptr;

		WaylandThread *wayland_thread = nullptr;

		struct wl_seat *wl_seat = nullptr;
		uint32_t wl_seat_name = 0;

		// Pointer.
		struct wl_pointer *wl_pointer = nullptr;

		uint32_t pointer_enter_serial = 0;

		struct wl_surface *pointed_surface = nullptr;
		struct wl_surface *last_pointed_surface = nullptr;

		struct zwp_relative_pointer_v1 *wp_relative_pointer = nullptr;
		struct zwp_locked_pointer_v1 *wp_locked_pointer = nullptr;
		struct zwp_confined_pointer_v1 *wp_confined_pointer = nullptr;

		struct zwp_pointer_gesture_pinch_v1 *wp_pointer_gesture_pinch = nullptr;

		// NOTE: According to the wp_pointer_gestures protocol specification, there
		// can be only one active gesture at a time.
		Gesture active_gesture = Gesture::NONE;

		// Used for delta calculations.
		// NOTE: The wp_pointer_gestures protocol keeps track of the total scale of
		// the pinch gesture, while godot instead wants its delta.
		wl_fixed_t old_pinch_scale = 0;

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

		// Clipboard.
		struct wl_data_source *wl_data_source_selection = nullptr;
		struct wl_data_offer *wl_data_offer_selection = nullptr;

		Vector<uint8_t> selection_data;

		// Primary selection.
		struct zwp_primary_selection_device_v1 *wp_primary_selection_device = nullptr;

		struct zwp_primary_selection_source_v1 *wp_primary_selection_source = nullptr;
		struct zwp_primary_selection_offer_v1 *wp_primary_selection_offer = nullptr;

		Vector<uint8_t> primary_data;

		// Tablet.
		struct zwp_tablet_seat_v2 *wp_tablet_seat = nullptr;

		List<struct zwp_tablet_tool_v2 *> tablet_tools;

		TabletToolData tablet_tool_data_buffer;
		TabletToolData tablet_tool_data;
	};

	struct CustomCursor {
		struct wl_buffer *wl_buffer = nullptr;
		uint32_t *buffer_data = nullptr;
		uint32_t buffer_data_size = 0;

		RID cursor_rid;
		Point2i hotspot;
	};

	// Jack of all trades. Currently used only by `DisplayServerWayland` code.
	// TODO: Get rid of this thing.
	struct WaylandState {
		struct wl_display *wl_display = nullptr;
		struct wl_registry *wl_registry = nullptr;

		SeatState *current_seat = nullptr;

		WaylandThread *wayland_thread = nullptr;
	};

	// END DISPLAYSERVERWAYLAND STUFF DUMP

private:
	struct ThreadData {
		SafeFlag thread_done;
		Mutex mutex;

		struct wl_display *wl_display = nullptr;
	};

	// TODO: Get rid of this.
	WaylandState *wls;

	// FIXME: Is this the right thing to do?
	inline static const char *proxy_tag = "godot";

	Thread events_thread;
	ThreadData thread_data;

	WindowState main_window;

	List<Ref<Message>> messages;

	struct wl_cursor_theme *wl_cursor_theme = nullptr;
	struct wl_cursor_image *cursor_images[DisplayServer::CURSOR_MAX] = {};
	struct wl_buffer *cursor_bufs[DisplayServer::CURSOR_MAX] = {};

	HashMap<DisplayServer::CursorShape, CustomCursor> custom_cursors;

	struct wl_buffer *cursor_buffer = nullptr;
	Point2i cursor_hotspot;

	PointerConstraint pointer_constraint = PointerConstraint::NONE;

	struct wl_display *wl_display = nullptr;
	struct wl_registry *wl_registry = nullptr;

	struct wl_seat *wl_seat_current = nullptr;

	RegistryState registry;

	bool initialized = false;

#ifdef LIBDECOR_ENABLED
	struct libdecor *libdecor_context = nullptr;
#endif // LIBDECOR_ENABLED

	// Main polling method.
	static void _poll_events_thread(void *p_data);

	// Core Wayland event handlers.
	static void _wl_registry_on_global(void *data, struct wl_registry *wl_registry, uint32_t name, const char *interface, uint32_t version);
	static void _wl_registry_on_global_remove(void *data, struct wl_registry *wl_registry, uint32_t name);

	static void _wl_surface_on_enter(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output);
	static void _wl_surface_on_leave(void *data, struct wl_surface *wl_surface, struct wl_output *wl_output);

	static void _wl_output_on_geometry(void *data, struct wl_output *wl_output, int32_t x, int32_t y, int32_t physical_width, int32_t physical_height, int32_t subpixel, const char *make, const char *model, int32_t transform);
	static void _wl_output_on_mode(void *data, struct wl_output *wl_output, uint32_t flags, int32_t width, int32_t height, int32_t refresh);
	static void _wl_output_on_done(void *data, struct wl_output *wl_output);
	static void _wl_output_on_scale(void *data, struct wl_output *wl_output, int32_t factor);
	static void _wl_output_on_name(void *data, struct wl_output *wl_output, const char *name);
	static void _wl_output_on_description(void *data, struct wl_output *wl_output, const char *description);

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
	static void _wl_pointer_on_axis_value120(void *data, struct wl_pointer *wl_pointer, uint32_t axis, int32_t value120);

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
	static void _xdg_toplevel_on_configure_bounds(void *data, struct xdg_toplevel *xdg_toplevel, int32_t width, int32_t height);
	static void _xdg_toplevel_on_wm_capabilities(void *data, struct xdg_toplevel *xdg_toplevel, struct wl_array *capabilities);

	// wayland-protocols event handlers.
	static void _wp_relative_pointer_on_relative_motion(void *data, struct zwp_relative_pointer_v1 *wp_relative_pointer_v1, uint32_t uptime_hi, uint32_t uptime_lo, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t dx_unaccel, wl_fixed_t dy_unaccel);

	static void _wp_pointer_gesture_pinch_on_begin(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, struct wl_surface *surface, uint32_t fingers);
	static void _wp_pointer_gesture_pinch_on_update(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t time, wl_fixed_t dx, wl_fixed_t dy, wl_fixed_t scale, wl_fixed_t rotation);
	static void _wp_pointer_gesture_pinch_on_end(void *data, struct zwp_pointer_gesture_pinch_v1 *zwp_pointer_gesture_pinch_v1, uint32_t serial, uint32_t time, int32_t cancelled);

	static void _wp_primary_selection_device_on_data_offer(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *offer);
	static void _wp_primary_selection_device_on_selection(void *data, struct zwp_primary_selection_device_v1 *wp_primary_selection_device_v1, struct zwp_primary_selection_offer_v1 *id);

	static void _wp_primary_selection_source_on_send(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1, const char *mime_type, int32_t fd);
	static void _wp_primary_selection_source_on_cancelled(void *data, struct zwp_primary_selection_source_v1 *wp_primary_selection_source_v1);

	static void _wp_tablet_seat_on_tablet_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_v2 *id);
	static void _wp_tablet_seat_on_tool_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_tool_v2 *id);
	static void _wp_tablet_seat_on_pad_added(void *data, struct zwp_tablet_seat_v2 *zwp_tablet_seat_v2, struct zwp_tablet_pad_v2 *id);

	static void _wp_tablet_tool_on_type(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t tool_type);
	static void _wp_tablet_tool_on_hardware_serial(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t hardware_serial_hi, uint32_t hardware_serial_lo);
	static void _wp_tablet_tool_on_hardware_id_wacom(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t hardware_id_hi, uint32_t hardware_id_lo);
	static void _wp_tablet_tool_on_capability(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t capability);
	static void _wp_tablet_tool_on_done(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2);
	static void _wp_tablet_tool_on_removed(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2);
	static void _wp_tablet_tool_on_proximity_in(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial, struct zwp_tablet_v2 *tablet, struct wl_surface *surface);
	static void _wp_tablet_tool_on_proximity_out(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2);
	static void _wp_tablet_tool_on_down(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial);
	static void _wp_tablet_tool_on_up(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2);
	static void _wp_tablet_tool_on_motion(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t x, wl_fixed_t y);
	static void _wp_tablet_tool_on_pressure(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t pressure);
	static void _wp_tablet_tool_on_distance(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t distance);
	static void _wp_tablet_tool_on_tilt(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t tilt_x, wl_fixed_t tilt_y);
	static void _wp_tablet_tool_on_rotation(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t degrees);
	static void _wp_tablet_tool_on_slider(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, int32_t position);
	static void _wp_tablet_tool_on_wheel(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, wl_fixed_t degrees, int32_t clicks);
	static void _wp_tablet_tool_on_button(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t serial, uint32_t button, uint32_t state);
	static void _wp_tablet_tool_on_frame(void *data, struct zwp_tablet_tool_v2 *zwp_tablet_tool_v2, uint32_t time);

	static void _xdg_toplevel_decoration_on_configure(void *data, struct zxdg_toplevel_decoration_v1 *xdg_toplevel_decoration, uint32_t mode);

	static void _xdg_activation_token_on_done(void *data, struct xdg_activation_token_v1 *xdg_activation_token, const char *token);

	// Core Wayland event listeners.
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
		.name = _wl_output_on_name,
		.description = _wl_output_on_description,
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
		.axis_value120 = _wl_pointer_on_axis_value120,
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
		.configure_bounds = _xdg_toplevel_on_configure_bounds,
		.wm_capabilities = _xdg_toplevel_on_wm_capabilities,
	};

	// wayland-protocols event listeners.
	static constexpr struct zwp_relative_pointer_v1_listener wp_relative_pointer_listener = {
		.relative_motion = _wp_relative_pointer_on_relative_motion,
	};

	static constexpr struct zwp_pointer_gesture_pinch_v1_listener wp_pointer_gesture_pinch_listener = {
		.begin = _wp_pointer_gesture_pinch_on_begin,
		.update = _wp_pointer_gesture_pinch_on_update,
		.end = _wp_pointer_gesture_pinch_on_end,
	};

	static constexpr struct zwp_primary_selection_device_v1_listener wp_primary_selection_device_listener = {
		.data_offer = _wp_primary_selection_device_on_data_offer,
		.selection = _wp_primary_selection_device_on_selection,
	};

	static constexpr struct zwp_primary_selection_source_v1_listener wp_primary_selection_source_listener = {
		.send = _wp_primary_selection_source_on_send,
		.cancelled = _wp_primary_selection_source_on_cancelled,
	};

	static constexpr struct zwp_tablet_seat_v2_listener wp_tablet_seat_listener = {
		.tablet_added = _wp_tablet_seat_on_tablet_added,
		.tool_added = _wp_tablet_seat_on_tool_added,
		.pad_added = _wp_tablet_seat_on_pad_added,
	};

	static constexpr struct zwp_tablet_tool_v2_listener wp_tablet_tool_listener = {
		.type = _wp_tablet_tool_on_type,
		.hardware_serial = _wp_tablet_tool_on_hardware_serial,
		.hardware_id_wacom = _wp_tablet_tool_on_hardware_id_wacom,
		.capability = _wp_tablet_tool_on_capability,
		.done = _wp_tablet_tool_on_done,
		.removed = _wp_tablet_tool_on_removed,
		.proximity_in = _wp_tablet_tool_on_proximity_in,
		.proximity_out = _wp_tablet_tool_on_proximity_out,
		.down = _wp_tablet_tool_on_down,
		.up = _wp_tablet_tool_on_up,
		.motion = _wp_tablet_tool_on_motion,
		.pressure = _wp_tablet_tool_on_pressure,
		.distance = _wp_tablet_tool_on_distance,
		.tilt = _wp_tablet_tool_on_tilt,
		.rotation = _wp_tablet_tool_on_rotation,
		.slider = _wp_tablet_tool_on_slider,
		.wheel = _wp_tablet_tool_on_wheel,
		.button = _wp_tablet_tool_on_button,
		.frame = _wp_tablet_tool_on_frame,
	};

	static constexpr struct zxdg_toplevel_decoration_v1_listener xdg_toplevel_decoration_listener = {
		.configure = _xdg_toplevel_decoration_on_configure,
	};

	static constexpr struct xdg_activation_token_v1_listener xdg_activation_token_listener = {
		.done = _xdg_activation_token_on_done,
	};

#ifdef LIBDECOR_ENABLED
	// libdecor event handlers.
	static void libdecor_on_error(struct libdecor *context, enum libdecor_error error, const char *message);

	static void libdecor_frame_on_configure(struct libdecor_frame *frame, struct libdecor_configuration *configuration, void *user_data);

	static void libdecor_frame_on_close(struct libdecor_frame *frame, void *user_data);

	static void libdecor_frame_on_commit(struct libdecor_frame *frame, void *user_data);

	static void libdecor_frame_on_dismiss_popup(struct libdecor_frame *frame, const char *seat_name, void *user_data);

	// libdecor event listeners.
	static constexpr struct libdecor_interface libdecor_interface = {
		.error = libdecor_on_error,
		.reserved0 = nullptr,
		.reserved1 = nullptr,
		.reserved2 = nullptr,
		.reserved3 = nullptr,
		.reserved4 = nullptr,
		.reserved5 = nullptr,
		.reserved6 = nullptr,
		.reserved7 = nullptr,
		.reserved8 = nullptr,
		.reserved9 = nullptr,
	};

	static constexpr struct libdecor_frame_interface libdecor_frame_interface = {
		.configure = libdecor_frame_on_configure,
		.close = libdecor_frame_on_close,
		.commit = libdecor_frame_on_commit,
		.dismiss_popup = libdecor_frame_on_dismiss_popup,
		.reserved0 = nullptr,
		.reserved1 = nullptr,
		.reserved2 = nullptr,
		.reserved3 = nullptr,
		.reserved4 = nullptr,
		.reserved5 = nullptr,
		.reserved6 = nullptr,
		.reserved7 = nullptr,
		.reserved8 = nullptr,
		.reserved9 = nullptr,
	};
#endif // LIBDECOR_ENABLED

public:
	Mutex &mutex = thread_data.mutex;

	static String _string_read_fd(int fd);
	static int _allocate_shm_file(size_t size);

	static String _wl_data_offer_read(struct wl_display *wl_display, struct wl_data_offer *wl_data_offer);
	static String _wp_primary_selection_offer_read(struct wl_display *wl_display, struct zwp_primary_selection_offer_v1 *wp_primary_selection_offer);

	static void _seat_state_set_current(WaylandThread::SeatState &p_ss);
	static bool _seat_state_configure_key_event(WaylandThread::SeatState &p_seat, Ref<InputEventKey> p_event, xkb_keycode_t p_keycode, bool p_pressed);

	static void _wayland_state_update_cursor(WaylandThread::WaylandState &p_wls);

	void _set_current_seat(struct wl_seat *p_seat);

	// Core Wayland utilities for integrating with our own data structures.
	static bool wl_proxy_is_godot(struct wl_proxy *p_proxy);
	static void wl_proxy_tag_godot(struct wl_proxy *p_proxy);

	static WindowState *wl_surface_get_window_state(struct wl_surface *p_surface);
	static ScreenState *wl_output_get_screen_state(struct wl_output *p_output);
	static SeatState *wl_seat_get_seat_state(struct wl_seat *p_seat);

	void seat_state_unlock_pointer(SeatState *p_ss);
	void seat_state_lock_pointer(SeatState *p_ss);
	void seat_state_set_hint(SeatState *p_ss, int p_x, int p_y);
	void seat_state_confine_pointer(SeatState *p_ss);

	void seat_state_update_cursor(SeatState *p_ss);

	void seat_state_echo_keys(SeatState *p_ss);

	static int window_state_calculate_scale(WindowState *p_ws);

	void push_message(Ref<Message> message);
	bool has_message();
	Ref<Message> pop_message();

	void window_create(DisplayServer::WindowID p_window_id_id, int p_width, int p_height);

	struct wl_surface *window_get_wl_surface(DisplayServer::WindowID p_window_id) const;

	void window_resize(DisplayServer::WindowID p_window_id_id, Size2i p_size);
	void window_set_max_size(DisplayServer::WindowID p_window_id, Size2i p_size);
	void window_set_min_size(DisplayServer::WindowID p_window_id, Size2i p_size);

	bool window_can_set_mode(DisplayServer::WindowID p_window_id, DisplayServer::WindowMode p_mode) const;

	void window_set_borderless(DisplayServer::WindowID p_window_id, bool p_borderless);
	void window_set_title(DisplayServer::WindowID p_window_id, String p_title);
	void window_set_app_id(DisplayServer::WindowID p_window_id, String p_app_id);

	bool window_is_focused(DisplayServer::WindowID p_window_id);

	// Implemented by xdg_activation_v1
	void window_request_attention(DisplayServer::WindowID p_window_id);

	// Implemented by wp_idle_inhibit_manager_v1
	void window_set_idle_inhibition(DisplayServer::WindowID p_window_id, bool p_enable);
	bool window_get_idle_inhibition(DisplayServer::WindowID p_window_id) const;

	ScreenData screen_get_data(int p_screen) const;
	int get_screen_count() const;

	void pointer_set_constraint(PointerConstraint p_constraint);
	void pointer_set_hint(int p_x, int p_y);
	PointerConstraint pointer_get_constraint() const;
	DisplayServer::WindowID pointer_get_pointed_window_id();
	BitField<MouseButtonMask> pointer_get_button_mask() const;

	void cursor_hide();
	void cursor_set_shape(DisplayServer::CursorShape p_cursor_shape);
	void cursor_cache_custom_shape(DisplayServer::CursorShape p_cursor_shape, Ref<Image> p_image);

	void echo_keys();

	Error init(WaylandState &p_wls);
	void destroy();
};

#endif // WAYLAND_ENABLED

#endif // WAYLAND_THREAD_H
