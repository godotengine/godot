/*************************************************************************/
/*  display_server_x11.cpp                                               */
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

#include "display_server_x11.h"

#ifdef X11_ENABLED

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "detect_prime_x11.h"
#include "key_mapping_x11.h"
#include "main/main.h"
#include "scene/resources/texture.h"

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xinerama.h>
#include <X11/extensions/shape.h>

// ICCCM
#define WM_NormalState 1L // window normal state
#define WM_IconicState 3L // window minimized
// EWMH
#define _NET_WM_STATE_REMOVE 0L // remove/unset property
#define _NET_WM_STATE_ADD 1L // add/set property
#define _NET_WM_STATE_TOGGLE 2L // toggle property

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//stupid linux.h
#ifdef KEY_TAB
#undef KEY_TAB
#endif

#undef CursorShape
#include <X11/XKBlib.h>

// 2.2 is the first release with multitouch
#define XINPUT_CLIENT_VERSION_MAJOR 2
#define XINPUT_CLIENT_VERSION_MINOR 2

#define VALUATOR_ABSX 0
#define VALUATOR_ABSY 1
#define VALUATOR_PRESSURE 2
#define VALUATOR_TILTX 3
#define VALUATOR_TILTY 4

//#define DISPLAY_SERVER_X11_DEBUG_LOGS_ENABLED
#ifdef DISPLAY_SERVER_X11_DEBUG_LOGS_ENABLED
#define DEBUG_LOG_X11(...) printf(__VA_ARGS__)
#else
#define DEBUG_LOG_X11(...)
#endif

static const double abs_resolution_mult = 10000.0;
static const double abs_resolution_range_mult = 10.0;

// Hints for X11 fullscreen
struct Hints {
	unsigned long flags = 0;
	unsigned long functions = 0;
	unsigned long decorations = 0;
	long inputMode = 0;
	unsigned long status = 0;
};

bool DisplayServerX11::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_SUBWINDOWS:
#ifdef TOUCH_ENABLED
		case FEATURE_TOUCHSCREEN:
#endif
		case FEATURE_MOUSE:
		case FEATURE_MOUSE_WARP:
		case FEATURE_CLIPBOARD:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_CUSTOM_CURSOR_SHAPE:
		case FEATURE_IME:
		case FEATURE_WINDOW_TRANSPARENCY:
		//case FEATURE_HIDPI:
		case FEATURE_ICON:
		case FEATURE_NATIVE_ICON:
		case FEATURE_SWAP_BUFFERS:
#ifdef DBUS_ENABLED
		case FEATURE_KEEP_SCREEN_ON:
#endif
		case FEATURE_CLIPBOARD_PRIMARY:
			return true;
		default: {
		}
	}

	return false;
}

String DisplayServerX11::get_name() const {
	return "X11";
}

void DisplayServerX11::_update_real_mouse_position(const WindowData &wd) {
	Window root_return, child_return;
	int root_x, root_y, win_x, win_y;
	unsigned int mask_return;

	Bool xquerypointer_result = XQueryPointer(x11_display, wd.x11_window, &root_return, &child_return, &root_x, &root_y,
			&win_x, &win_y, &mask_return);

	if (xquerypointer_result) {
		if (win_x > 0 && win_y > 0 && win_x <= wd.size.width && win_y <= wd.size.height) {
			last_mouse_pos.x = win_x;
			last_mouse_pos.y = win_y;
			last_mouse_pos_valid = true;
			Input::get_singleton()->set_mouse_position(last_mouse_pos);
		}
	}
}

bool DisplayServerX11::_refresh_device_info() {
	int event_base, error_base;

	print_verbose("XInput: Refreshing devices.");

	if (!XQueryExtension(x11_display, "XInputExtension", &xi.opcode, &event_base, &error_base)) {
		print_verbose("XInput extension not available. Please upgrade your distribution.");
		return false;
	}

	int xi_major_query = XINPUT_CLIENT_VERSION_MAJOR;
	int xi_minor_query = XINPUT_CLIENT_VERSION_MINOR;

	if (XIQueryVersion(x11_display, &xi_major_query, &xi_minor_query) != Success) {
		print_verbose(vformat("XInput 2 not available (server supports %d.%d).", xi_major_query, xi_minor_query));
		xi.opcode = 0;
		return false;
	}

	if (xi_major_query < XINPUT_CLIENT_VERSION_MAJOR || (xi_major_query == XINPUT_CLIENT_VERSION_MAJOR && xi_minor_query < XINPUT_CLIENT_VERSION_MINOR)) {
		print_verbose(vformat("XInput %d.%d not available (server supports %d.%d). Touch input unavailable.",
				XINPUT_CLIENT_VERSION_MAJOR, XINPUT_CLIENT_VERSION_MINOR, xi_major_query, xi_minor_query));
	}

	xi.absolute_devices.clear();
	xi.touch_devices.clear();

	int dev_count;
	XIDeviceInfo *info = XIQueryDevice(x11_display, XIAllDevices, &dev_count);

	for (int i = 0; i < dev_count; i++) {
		XIDeviceInfo *dev = &info[i];
		if (!dev->enabled) {
			continue;
		}
		if (!(dev->use == XIMasterPointer || dev->use == XIFloatingSlave)) {
			continue;
		}

		bool direct_touch = false;
		bool absolute_mode = false;
		int resolution_x = 0;
		int resolution_y = 0;
		double abs_x_min = 0;
		double abs_x_max = 0;
		double abs_y_min = 0;
		double abs_y_max = 0;
		double pressure_min = 0;
		double pressure_max = 0;
		double tilt_x_min = 0;
		double tilt_x_max = 0;
		double tilt_y_min = 0;
		double tilt_y_max = 0;
		for (int j = 0; j < dev->num_classes; j++) {
#ifdef TOUCH_ENABLED
			if (dev->classes[j]->type == XITouchClass && ((XITouchClassInfo *)dev->classes[j])->mode == XIDirectTouch) {
				direct_touch = true;
			}
#endif
			if (dev->classes[j]->type == XIValuatorClass) {
				XIValuatorClassInfo *class_info = (XIValuatorClassInfo *)dev->classes[j];

				if (class_info->number == VALUATOR_ABSX && class_info->mode == XIModeAbsolute) {
					resolution_x = class_info->resolution;
					abs_x_min = class_info->min;
					abs_y_max = class_info->max;
					absolute_mode = true;
				} else if (class_info->number == VALUATOR_ABSY && class_info->mode == XIModeAbsolute) {
					resolution_y = class_info->resolution;
					abs_y_min = class_info->min;
					abs_y_max = class_info->max;
					absolute_mode = true;
				} else if (class_info->number == VALUATOR_PRESSURE && class_info->mode == XIModeAbsolute) {
					pressure_min = class_info->min;
					pressure_max = class_info->max;
				} else if (class_info->number == VALUATOR_TILTX && class_info->mode == XIModeAbsolute) {
					tilt_x_min = class_info->min;
					tilt_x_max = class_info->max;
				} else if (class_info->number == VALUATOR_TILTY && class_info->mode == XIModeAbsolute) {
					tilt_x_min = class_info->min;
					tilt_x_max = class_info->max;
				}
			}
		}
		if (direct_touch) {
			xi.touch_devices.push_back(dev->deviceid);
			print_verbose("XInput: Using touch device: " + String(dev->name));
		}
		if (absolute_mode) {
			// If no resolution was reported, use the min/max ranges.
			if (resolution_x <= 0) {
				resolution_x = (abs_x_max - abs_x_min) * abs_resolution_range_mult;
			}
			if (resolution_y <= 0) {
				resolution_y = (abs_y_max - abs_y_min) * abs_resolution_range_mult;
			}
			xi.absolute_devices[dev->deviceid] = Vector2(abs_resolution_mult / resolution_x, abs_resolution_mult / resolution_y);
			print_verbose("XInput: Absolute pointing device: " + String(dev->name));
		}

		xi.pressure = 0;
		xi.pen_pressure_range[dev->deviceid] = Vector2(pressure_min, pressure_max);
		xi.pen_tilt_x_range[dev->deviceid] = Vector2(tilt_x_min, tilt_x_max);
		xi.pen_tilt_y_range[dev->deviceid] = Vector2(tilt_y_min, tilt_y_max);
	}

	XIFreeDeviceInfo(info);
#ifdef TOUCH_ENABLED
	if (!xi.touch_devices.size()) {
		print_verbose("XInput: No touch devices found.");
	}
#endif

	return true;
}

void DisplayServerX11::_flush_mouse_motion() {
	// Block events polling while flushing motion events.
	MutexLock mutex_lock(events_mutex);

	for (uint32_t event_index = 0; event_index < polled_events.size(); ++event_index) {
		XEvent &event = polled_events[event_index];
		if (XGetEventData(x11_display, &event.xcookie) && event.xcookie.type == GenericEvent && event.xcookie.extension == xi.opcode) {
			XIDeviceEvent *event_data = (XIDeviceEvent *)event.xcookie.data;
			if (event_data->evtype == XI_RawMotion) {
				XFreeEventData(x11_display, &event.xcookie);
				polled_events.remove(event_index--);
				continue;
			}
			XFreeEventData(x11_display, &event.xcookie);
			break;
		}
	}

	xi.relative_motion.x = 0;
	xi.relative_motion.y = 0;
}

void DisplayServerX11::mouse_set_mode(MouseMode p_mode) {
	_THREAD_SAFE_METHOD_

	if (p_mode == mouse_mode) {
		return;
	}

	if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		XUngrabPointer(x11_display, CurrentTime);
	}

	// The only modes that show a cursor are VISIBLE and CONFINED
	bool showCursor = (p_mode == MOUSE_MODE_VISIBLE || p_mode == MOUSE_MODE_CONFINED);

	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (showCursor) {
			XDefineCursor(x11_display, E.value.x11_window, cursors[current_cursor]); // show cursor
		} else {
			XDefineCursor(x11_display, E.value.x11_window, null_cursor); // hide cursor
		}
	}
	mouse_mode = p_mode;

	if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		//flush pending motion events
		_flush_mouse_motion();
		WindowData &main_window = windows[MAIN_WINDOW_ID];

		if (XGrabPointer(
					x11_display, main_window.x11_window, True,
					ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
					GrabModeAsync, GrabModeAsync, windows[MAIN_WINDOW_ID].x11_window, None, CurrentTime) != GrabSuccess) {
			ERR_PRINT("NO GRAB");
		}

		if (mouse_mode == MOUSE_MODE_CAPTURED) {
			center.x = main_window.size.width / 2;
			center.y = main_window.size.height / 2;

			XWarpPointer(x11_display, None, main_window.x11_window,
					0, 0, 0, 0, (int)center.x, (int)center.y);

			Input::get_singleton()->set_mouse_position(center);
		}
	} else {
		do_mouse_warp = false;
	}

	XFlush(x11_display);
}

DisplayServerX11::MouseMode DisplayServerX11::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerX11::mouse_warp_to_position(const Point2i &p_to) {
	_THREAD_SAFE_METHOD_

	if (mouse_mode == MOUSE_MODE_CAPTURED) {
		last_mouse_pos = p_to;
	} else {
		XWarpPointer(x11_display, None, windows[MAIN_WINDOW_ID].x11_window,
				0, 0, 0, 0, (int)p_to.x, (int)p_to.y);
	}
}

Point2i DisplayServerX11::mouse_get_position() const {
	int root_x, root_y;
	int win_x, win_y;
	unsigned int mask_return;
	Window window_returned;

	Bool result = XQueryPointer(x11_display, RootWindow(x11_display, DefaultScreen(x11_display)), &window_returned,
			&window_returned, &root_x, &root_y, &win_x, &win_y,
			&mask_return);
	if (result == True) {
		return Point2i(root_x, root_y);
	}
	return Point2i();
}

Point2i DisplayServerX11::mouse_get_absolute_position() const {
	int number_of_screens = XScreenCount(x11_display);
	for (int i = 0; i < number_of_screens; i++) {
		Window root, child;
		int root_x, root_y, win_x, win_y;
		unsigned int mask;
		if (XQueryPointer(x11_display, XRootWindow(x11_display, i), &root, &child, &root_x, &root_y, &win_x, &win_y, &mask)) {
			XWindowAttributes root_attrs;
			XGetWindowAttributes(x11_display, root, &root_attrs);

			return Vector2i(root_attrs.x + root_x, root_attrs.y + root_y);
		}
	}
	return Vector2i();
}

MouseButton DisplayServerX11::mouse_get_button_state() const {
	return last_button_state;
}

void DisplayServerX11::clipboard_set(const String &p_text) {
	_THREAD_SAFE_METHOD_

	{
		// The clipboard content can be accessed while polling for events.
		MutexLock mutex_lock(events_mutex);
		internal_clipboard = p_text;
	}

	XSetSelectionOwner(x11_display, XA_PRIMARY, windows[MAIN_WINDOW_ID].x11_window, CurrentTime);
	XSetSelectionOwner(x11_display, XInternAtom(x11_display, "CLIPBOARD", 0), windows[MAIN_WINDOW_ID].x11_window, CurrentTime);
}

void DisplayServerX11::clipboard_set_primary(const String &p_text) {
	_THREAD_SAFE_METHOD_
	if (!p_text.is_empty()) {
		{
			// The clipboard content can be accessed while polling for events.
			MutexLock mutex_lock(events_mutex);
			internal_clipboard_primary = p_text;
		}

		XSetSelectionOwner(x11_display, XA_PRIMARY, windows[MAIN_WINDOW_ID].x11_window, CurrentTime);
		XSetSelectionOwner(x11_display, XInternAtom(x11_display, "PRIMARY", 0), windows[MAIN_WINDOW_ID].x11_window, CurrentTime);
	}
}

Bool DisplayServerX11::_predicate_clipboard_selection(Display *display, XEvent *event, XPointer arg) {
	if (event->type == SelectionNotify && event->xselection.requestor == *(Window *)arg) {
		return True;
	} else {
		return False;
	}
}

Bool DisplayServerX11::_predicate_clipboard_incr(Display *display, XEvent *event, XPointer arg) {
	if (event->type == PropertyNotify && event->xproperty.state == PropertyNewValue) {
		return True;
	} else {
		return False;
	}
}

String DisplayServerX11::_clipboard_get_impl(Atom p_source, Window x11_window, Atom target) const {
	String ret;

	Window selection_owner = XGetSelectionOwner(x11_display, p_source);
	if (selection_owner == x11_window) {
		static const char *target_type = "PRIMARY";
		if (p_source != None && String(XGetAtomName(x11_display, p_source)) == target_type) {
			return internal_clipboard_primary;
		} else {
			return internal_clipboard;
		}
	}

	if (selection_owner != None) {
		// Block events polling while processing selection events.
		MutexLock mutex_lock(events_mutex);

		Atom selection = XA_PRIMARY;
		XConvertSelection(x11_display, p_source, target, selection,
				x11_window, CurrentTime);

		XFlush(x11_display);

		// Blocking wait for predicate to be True and remove the event from the queue.
		XEvent event;
		XIfEvent(x11_display, &event, _predicate_clipboard_selection, (XPointer)&x11_window);

		// Do not get any data, see how much data is there.
		Atom type;
		int format, result;
		unsigned long len, bytes_left, dummy;
		unsigned char *data;
		XGetWindowProperty(x11_display, x11_window,
				selection, // Tricky..
				0, 0, // offset - len
				0, // Delete 0==FALSE
				AnyPropertyType, // flag
				&type, // return type
				&format, // return format
				&len, &bytes_left, // data length
				&data);

		if (data) {
			XFree(data);
		}

		if (type == XInternAtom(x11_display, "INCR", 0)) {
			// Data is going to be received incrementally.
			DEBUG_LOG_X11("INCR selection started.\n");

			LocalVector<uint8_t> incr_data;
			uint32_t data_size = 0;
			bool success = false;

			// Delete INCR property to notify the owner.
			XDeleteProperty(x11_display, x11_window, type);

			// Process events from the queue.
			bool done = false;
			while (!done) {
				if (!_wait_for_events()) {
					// Error or timeout, abort.
					break;
				}

				// Non-blocking wait for next event and remove it from the queue.
				XEvent ev;
				while (XCheckIfEvent(x11_display, &ev, _predicate_clipboard_incr, nullptr)) {
					result = XGetWindowProperty(x11_display, x11_window,
							selection, // selection type
							0, LONG_MAX, // offset - len
							True, // delete property to notify the owner
							AnyPropertyType, // flag
							&type, // return type
							&format, // return format
							&len, &bytes_left, // data length
							&data);

					DEBUG_LOG_X11("PropertyNotify: len=%lu, format=%i\n", len, format);

					if (result == Success) {
						if (data && (len > 0)) {
							uint32_t prev_size = incr_data.size();
							if (prev_size == 0) {
								// First property contains initial data size.
								unsigned long initial_size = *(unsigned long *)data;
								incr_data.resize(initial_size);
							} else {
								// New chunk, resize to be safe and append data.
								incr_data.resize(MAX(data_size + len, prev_size));
								memcpy(incr_data.ptr() + data_size, data, len);
								data_size += len;
							}
						} else {
							// Last chunk, process finished.
							done = true;
							success = true;
						}
					} else {
						printf("Failed to get selection data chunk.\n");
						done = true;
					}

					if (data) {
						XFree(data);
					}

					if (done) {
						break;
					}
				}
			}

			if (success && (data_size > 0)) {
				ret.parse_utf8((const char *)incr_data.ptr(), data_size);
			}
		} else if (bytes_left > 0) {
			// Data is ready and can be processed all at once.
			result = XGetWindowProperty(x11_display, x11_window,
					selection, 0, bytes_left, 0,
					AnyPropertyType, &type, &format,
					&len, &dummy, &data);

			if (result == Success) {
				ret.parse_utf8((const char *)data);
			} else {
				printf("Failed to get selection data.\n");
			}

			if (data) {
				XFree(data);
			}
		}
	}

	return ret;
}

String DisplayServerX11::_clipboard_get(Atom p_source, Window x11_window) const {
	String ret;
	Atom utf8_atom = XInternAtom(x11_display, "UTF8_STRING", True);
	if (utf8_atom != None) {
		ret = _clipboard_get_impl(p_source, x11_window, utf8_atom);
	}
	if (ret.is_empty()) {
		ret = _clipboard_get_impl(p_source, x11_window, XA_STRING);
	}
	return ret;
}

String DisplayServerX11::clipboard_get() const {
	_THREAD_SAFE_METHOD_

	String ret;
	ret = _clipboard_get(XInternAtom(x11_display, "CLIPBOARD", 0), windows[MAIN_WINDOW_ID].x11_window);

	if (ret.is_empty()) {
		ret = _clipboard_get(XA_PRIMARY, windows[MAIN_WINDOW_ID].x11_window);
	}

	return ret;
}

String DisplayServerX11::clipboard_get_primary() const {
	_THREAD_SAFE_METHOD_

	String ret;
	ret = _clipboard_get(XInternAtom(x11_display, "PRIMARY", 0), windows[MAIN_WINDOW_ID].x11_window);

	if (ret.is_empty()) {
		ret = _clipboard_get(XA_PRIMARY, windows[MAIN_WINDOW_ID].x11_window);
	}

	return ret;
}

Bool DisplayServerX11::_predicate_clipboard_save_targets(Display *display, XEvent *event, XPointer arg) {
	if (event->xany.window == *(Window *)arg) {
		return (event->type == SelectionRequest) ||
				(event->type == SelectionNotify);
	} else {
		return False;
	}
}

void DisplayServerX11::_clipboard_transfer_ownership(Atom p_source, Window x11_window) const {
	_THREAD_SAFE_METHOD_

	Window selection_owner = XGetSelectionOwner(x11_display, p_source);

	if (selection_owner != x11_window) {
		return;
	}

	// Block events polling while processing selection events.
	MutexLock mutex_lock(events_mutex);

	Atom clipboard_manager = XInternAtom(x11_display, "CLIPBOARD_MANAGER", False);
	Atom save_targets = XInternAtom(x11_display, "SAVE_TARGETS", False);
	XConvertSelection(x11_display, clipboard_manager, save_targets, None,
			x11_window, CurrentTime);

	// Process events from the queue.
	while (true) {
		if (!_wait_for_events()) {
			// Error or timeout, abort.
			break;
		}

		// Non-blocking wait for next event and remove it from the queue.
		XEvent ev;
		while (XCheckIfEvent(x11_display, &ev, _predicate_clipboard_save_targets, (XPointer)&x11_window)) {
			switch (ev.type) {
				case SelectionRequest:
					_handle_selection_request_event(&(ev.xselectionrequest));
					break;

				case SelectionNotify: {
					if (ev.xselection.target == save_targets) {
						// Once SelectionNotify is received, we're done whether it succeeded or not.
						return;
					}

					break;
				}
			}
		}
	}
}

int DisplayServerX11::get_screen_count() const {
	_THREAD_SAFE_METHOD_

	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) {
		return 0;
	}

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);
	XFree(xsi);
	return count;
}

Point2i DisplayServerX11::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) {
		return Point2i(0, 0);
	}

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);

	// Check if screen is valid
	ERR_FAIL_INDEX_V(p_screen, count, Point2i(0, 0));

	Point2i position = Point2i(xsi[p_screen].x_org, xsi[p_screen].y_org);

	XFree(xsi);

	return position;
}

Size2i DisplayServerX11::screen_get_size(int p_screen) const {
	return screen_get_usable_rect(p_screen).size;
}

Rect2i DisplayServerX11::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) {
		return Rect2i(0, 0, 0, 0);
	}

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);

	// Check if screen is valid
	ERR_FAIL_INDEX_V(p_screen, count, Rect2i(0, 0, 0, 0));

	Rect2i rect = Rect2i(xsi[p_screen].x_org, xsi[p_screen].y_org, xsi[p_screen].width, xsi[p_screen].height);
	XFree(xsi);
	return rect;
}

int DisplayServerX11::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	//invalid screen?
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), 0);

	//Get physical monitor Dimensions through XRandR and calculate dpi
	Size2i sc = screen_get_size(p_screen);
	if (xrandr_ext_ok) {
		int count = 0;
		if (xrr_get_monitors) {
			xrr_monitor_info *monitors = xrr_get_monitors(x11_display, windows[MAIN_WINDOW_ID].x11_window, true, &count);
			if (p_screen < count) {
				double xdpi = sc.width / (double)monitors[p_screen].mwidth * 25.4;
				double ydpi = sc.height / (double)monitors[p_screen].mheight * 25.4;
				xrr_free_monitors(monitors);
				return (xdpi + ydpi) / 2;
			}
			xrr_free_monitors(monitors);
		} else if (p_screen == 0) {
			XRRScreenSize *sizes = XRRSizes(x11_display, 0, &count);
			if (sizes) {
				double xdpi = sc.width / (double)sizes[0].mwidth * 25.4;
				double ydpi = sc.height / (double)sizes[0].mheight * 25.4;
				return (xdpi + ydpi) / 2;
			}
		}
	}

	int width_mm = DisplayWidthMM(x11_display, p_screen);
	int height_mm = DisplayHeightMM(x11_display, p_screen);
	double xdpi = (width_mm ? sc.width / (double)width_mm * 25.4 : 0);
	double ydpi = (height_mm ? sc.height / (double)height_mm * 25.4 : 0);
	if (xdpi || ydpi) {
		return (xdpi + ydpi) / (xdpi && ydpi ? 2 : 1);
	}

	//could not get dpi
	return 96;
}

bool DisplayServerX11::screen_is_touchscreen(int p_screen) const {
	_THREAD_SAFE_METHOD_

#ifndef _MSC_VER
#warning Need to get from proper window
#endif

	return DisplayServer::screen_is_touchscreen(p_screen);
}

#ifdef DBUS_ENABLED
void DisplayServerX11::screen_set_keep_on(bool p_enable) {
	if (screen_is_kept_on() == p_enable) {
		return;
	}

	if (p_enable) {
		screensaver->inhibit();
	} else {
		screensaver->uninhibit();
	}

	keep_screen_on = p_enable;
}

bool DisplayServerX11::screen_is_kept_on() const {
	return keep_screen_on;
}
#endif

Vector<DisplayServer::WindowID> DisplayServerX11::get_window_list() const {
	_THREAD_SAFE_METHOD_

	Vector<int> ret;
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		ret.push_back(E.key);
	}
	return ret;
}

DisplayServer::WindowID DisplayServerX11::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	WindowID id = _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, id);
		}
	}

	return id;
}

void DisplayServerX11::show_window(WindowID p_id) {
	_THREAD_SAFE_METHOD_

	const WindowData &wd = windows[p_id];

	DEBUG_LOG_X11("show_window: %lu (%u) \n", wd.x11_window, p_id);

	XMapWindow(x11_display, wd.x11_window);
}

void DisplayServerX11::delete_sub_window(WindowID p_id) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_id));
	ERR_FAIL_COND_MSG(p_id == MAIN_WINDOW_ID, "Main window can't be deleted"); //ma

	WindowData &wd = windows[p_id];

	DEBUG_LOG_X11("delete_sub_window: %lu (%u) \n", wd.x11_window, p_id);

	while (wd.transient_children.size()) {
		window_set_transient(wd.transient_children.front()->get(), INVALID_WINDOW_ID);
	}

	if (wd.transient_parent != INVALID_WINDOW_ID) {
		window_set_transient(p_id, INVALID_WINDOW_ID);
	}

#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		context_vulkan->window_destroy(p_id);
	}
#endif
	XUnmapWindow(x11_display, wd.x11_window);
	XDestroyWindow(x11_display, wd.x11_window);
	if (wd.xic) {
		XDestroyIC(wd.xic);
		wd.xic = nullptr;
	}

	windows.erase(p_id);
}

void DisplayServerX11::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.instance_id = p_instance;
}

ObjectID DisplayServerX11::window_get_attached_instance_id(WindowID p_window) const {
	ERR_FAIL_COND_V(!windows.has(p_window), ObjectID());
	const WindowData &wd = windows[p_window];
	return wd.instance_id;
}

DisplayServerX11::WindowID DisplayServerX11::get_window_at_screen_position(const Point2i &p_position) const {
	WindowID found_window = INVALID_WINDOW_ID;
	WindowID parent_window = INVALID_WINDOW_ID;
	unsigned int focus_order = 0;
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		const WindowData &wd = E.value;

		// Discard windows with no focus.
		if (wd.focus_order == 0) {
			continue;
		}

		// Find topmost window which contains the given position.
		WindowID window_id = E.key;
		Rect2i win_rect = Rect2i(window_get_position(window_id), window_get_size(window_id));
		if (win_rect.has_point(p_position)) {
			// For siblings, pick the window which was focused last.
			if ((parent_window != wd.transient_parent) || (wd.focus_order > focus_order)) {
				found_window = window_id;
				parent_window = wd.transient_parent;
				focus_order = wd.focus_order;
			}
		}
	}

	return found_window;
}

void DisplayServerX11::window_set_title(const String &p_title, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	XStoreName(x11_display, wd.x11_window, p_title.utf8().get_data());

	Atom _net_wm_name = XInternAtom(x11_display, "_NET_WM_NAME", false);
	Atom utf8_string = XInternAtom(x11_display, "UTF8_STRING", false);
	if (_net_wm_name != None && utf8_string != None) {
		XChangeProperty(x11_display, wd.x11_window, _net_wm_name, utf8_string, 8, PropModeReplace, (unsigned char *)p_title.utf8().get_data(), p_title.utf8().length());
	}
}

void DisplayServerX11::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	const WindowData &wd = windows[p_window];

	int event_base, error_base;
	const Bool ext_okay = XShapeQueryExtension(x11_display, &event_base, &error_base);
	if (ext_okay) {
		Region region;
		if (p_region.size() == 0) {
			region = XCreateRegion();
			XRectangle rect;
			rect.x = 0;
			rect.y = 0;
			rect.width = window_get_real_size(p_window).x;
			rect.height = window_get_real_size(p_window).y;
			XUnionRectWithRegion(&rect, region, region);
		} else {
			XPoint *points = (XPoint *)memalloc(sizeof(XPoint) * p_region.size());
			for (int i = 0; i < p_region.size(); i++) {
				points[i].x = p_region[i].x;
				points[i].y = p_region[i].y;
			}
			region = XPolygonRegion(points, p_region.size(), EvenOddRule);
			memfree(points);
		}
		XShapeCombineRegion(x11_display, wd.x11_window, ShapeInput, 0, 0, region, ShapeSet);
		XDestroyRegion(region);
	}
}

void DisplayServerX11::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.rect_changed_callback = p_callable;
}

void DisplayServerX11::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.event_callback = p_callable;
}

void DisplayServerX11::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_event_callback = p_callable;
}

void DisplayServerX11::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_text_callback = p_callable;
}

void DisplayServerX11::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.drop_files_callback = p_callable;
}

int DisplayServerX11::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	const WindowData &wd = windows[p_window];

	int x, y;
	Window child;
	XTranslateCoordinates(x11_display, wd.x11_window, DefaultRootWindow(x11_display), 0, 0, &x, &y, &child);

	int count = get_screen_count();
	for (int i = 0; i < count; i++) {
		Point2i pos = screen_get_position(i);
		Size2i size = screen_get_size(i);
		if ((x >= pos.x && x < pos.x + size.width) && (y >= pos.y && y < pos.y + size.height)) {
			return i;
		}
	}
	return 0;
}

void DisplayServerX11::window_set_current_screen(int p_screen, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	// Check if screen is valid
	ERR_FAIL_INDEX(p_screen, get_screen_count());

	if (window_get_mode(p_window) == WINDOW_MODE_FULLSCREEN) {
		Point2i position = screen_get_position(p_screen);
		Size2i size = screen_get_size(p_screen);

		XMoveResizeWindow(x11_display, wd.x11_window, position.x, position.y, size.x, size.y);
	} else {
		if (p_screen != window_get_current_screen(p_window)) {
			Point2i position = screen_get_position(p_screen);
			XMoveWindow(x11_display, wd.x11_window, position.x, position.y);
		}
	}
}

void DisplayServerX11::window_set_transient(WindowID p_window, WindowID p_parent) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(p_window == p_parent);

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd_window = windows[p_window];

	WindowID prev_parent = wd_window.transient_parent;
	ERR_FAIL_COND(prev_parent == p_parent);

	DEBUG_LOG_X11("window_set_transient: %lu (%u), prev_parent=%u, parent=%u\n", wd_window.x11_window, p_window, prev_parent, p_parent);

	ERR_FAIL_COND_MSG(wd_window.on_top, "Windows with the 'on top' can't become transient.");
	if (p_parent == INVALID_WINDOW_ID) {
		//remove transient

		ERR_FAIL_COND(prev_parent == INVALID_WINDOW_ID);
		ERR_FAIL_COND(!windows.has(prev_parent));

		WindowData &wd_parent = windows[prev_parent];

		wd_window.transient_parent = INVALID_WINDOW_ID;
		wd_parent.transient_children.erase(p_window);

		XSetTransientForHint(x11_display, wd_window.x11_window, None);

		// Set focus to parent sub window to avoid losing all focus when closing a nested sub-menu.
		// RevertToPointerRoot is used to make sure we don't lose all focus in case
		// a subwindow and its parent are both destroyed.
		if (wd_window.menu_type && !wd_window.no_focus && wd_window.focused) {
			if (!wd_parent.no_focus) {
				XSetInputFocus(x11_display, wd_parent.x11_window, RevertToPointerRoot, CurrentTime);
			}
		}
	} else {
		ERR_FAIL_COND(!windows.has(p_parent));
		ERR_FAIL_COND_MSG(prev_parent != INVALID_WINDOW_ID, "Window already has a transient parent");
		WindowData &wd_parent = windows[p_parent];

		wd_window.transient_parent = p_parent;
		wd_parent.transient_children.insert(p_window);

		XSetTransientForHint(x11_display, wd_window.x11_window, wd_parent.x11_window);
	}
}

// Helper method. Assumes that the window id has already been checked and exists.
void DisplayServerX11::_update_size_hints(WindowID p_window) {
	WindowData &wd = windows[p_window];
	WindowMode window_mode = window_get_mode(p_window);
	XSizeHints *xsh = XAllocSizeHints();

	// Always set the position and size hints - they should be synchronized with the actual values after the window is mapped anyway
	xsh->flags |= PPosition | PSize;
	xsh->x = wd.position.x;
	xsh->y = wd.position.y;
	xsh->width = wd.size.width;
	xsh->height = wd.size.height;

	if (window_mode == WINDOW_MODE_FULLSCREEN) {
		// Do not set any other hints to prevent the window manager from ignoring the fullscreen flags
	} else if (window_get_flag(WINDOW_FLAG_RESIZE_DISABLED, p_window)) {
		// If resizing is disabled, use the forced size
		xsh->flags |= PMinSize | PMaxSize;
		xsh->min_width = wd.size.x;
		xsh->max_width = wd.size.x;
		xsh->min_height = wd.size.y;
		xsh->max_height = wd.size.y;
	} else {
		// Otherwise, just respect min_size and max_size
		if (wd.min_size != Size2i()) {
			xsh->flags |= PMinSize;
			xsh->min_width = wd.min_size.x;
			xsh->min_height = wd.min_size.y;
		}
		if (wd.max_size != Size2i()) {
			xsh->flags |= PMaxSize;
			xsh->max_width = wd.max_size.x;
			xsh->max_height = wd.max_size.y;
		}
	}

	XSetWMNormalHints(x11_display, wd.x11_window, xsh);
	XFree(xsh);
}

Point2i DisplayServerX11::window_get_position(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());
	const WindowData &wd = windows[p_window];

	return wd.position;
}

void DisplayServerX11::window_set_position(const Point2i &p_position, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	int x = 0;
	int y = 0;
	if (!window_get_flag(WINDOW_FLAG_BORDERLESS, p_window)) {
		//exclude window decorations
		XSync(x11_display, False);
		Atom prop = XInternAtom(x11_display, "_NET_FRAME_EXTENTS", True);
		if (prop != None) {
			Atom type;
			int format;
			unsigned long len;
			unsigned long remaining;
			unsigned char *data = nullptr;
			if (XGetWindowProperty(x11_display, wd.x11_window, prop, 0, 4, False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
				if (format == 32 && len == 4 && data) {
					long *extents = (long *)data;
					x = extents[0];
					y = extents[2];
				}
				XFree(data);
			}
		}
	}
	XMoveWindow(x11_display, wd.x11_window, p_position.x - x, p_position.y - y);
	_update_real_mouse_position(wd);
}

void DisplayServerX11::window_set_max_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2i()) && ((p_size.x < wd.min_size.x) || (p_size.y < wd.min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	wd.max_size = p_size;

	_update_size_hints(p_window);
	XFlush(x11_display);
}

Size2i DisplayServerX11::window_get_max_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	return wd.max_size;
}

void DisplayServerX11::window_set_min_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2i()) && (wd.max_size != Size2i()) && ((p_size.x > wd.max_size.x) || (p_size.y > wd.max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	wd.min_size = p_size;

	_update_size_hints(p_window);
	XFlush(x11_display);
}

Size2i DisplayServerX11::window_get_min_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	return wd.min_size;
}

void DisplayServerX11::window_set_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));

	Size2i size = p_size;
	size.x = MAX(1, size.x);
	size.y = MAX(1, size.y);

	WindowData &wd = windows[p_window];

	if (wd.size.width == size.width && wd.size.height == size.height) {
		return;
	}

	XWindowAttributes xwa;
	XSync(x11_display, False);
	XGetWindowAttributes(x11_display, wd.x11_window, &xwa);
	int old_w = xwa.width;
	int old_h = xwa.height;

	// Update our videomode width and height
	wd.size = size;

	// Update the size hints first to make sure the window size can be set
	_update_size_hints(p_window);

	// Resize the window
	XResizeWindow(x11_display, wd.x11_window, size.x, size.y);

	for (int timeout = 0; timeout < 50; ++timeout) {
		XSync(x11_display, False);
		XGetWindowAttributes(x11_display, wd.x11_window, &xwa);

		if (old_w != xwa.width || old_h != xwa.height) {
			break;
		}

		usleep(10000);
	}
}

Size2i DisplayServerX11::window_get_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.size;
}

Size2i DisplayServerX11::window_get_real_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	XWindowAttributes xwa;
	XSync(x11_display, False);
	XGetWindowAttributes(x11_display, wd.x11_window, &xwa);
	int w = xwa.width;
	int h = xwa.height;
	Atom prop = XInternAtom(x11_display, "_NET_FRAME_EXTENTS", True);
	if (prop != None) {
		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = nullptr;
		if (XGetWindowProperty(x11_display, wd.x11_window, prop, 0, 4, False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
			if (format == 32 && len == 4 && data) {
				long *extents = (long *)data;
				w += extents[0] + extents[1]; // left, right
				h += extents[2] + extents[3]; // top, bottom
			}
			XFree(data);
		}
	}
	return Size2i(w, h);
}

// Just a helper to reduce code duplication in `window_is_maximize_allowed`
// and `_set_wm_maximized`.
bool DisplayServerX11::_window_maximize_check(WindowID p_window, const char *p_atom_name) const {
	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	Atom property = XInternAtom(x11_display, p_atom_name, False);
	Atom type;
	int format;
	unsigned long len;
	unsigned long remaining;
	unsigned char *data = nullptr;
	bool retval = false;

	if (property == None) {
		return false;
	}

	int result = XGetWindowProperty(
			x11_display,
			wd.x11_window,
			property,
			0,
			1024,
			False,
			XA_ATOM,
			&type,
			&format,
			&len,
			&remaining,
			&data);

	if (result == Success && data) {
		Atom *atoms = (Atom *)data;
		Atom wm_act_max_horz = XInternAtom(x11_display, "_NET_WM_ACTION_MAXIMIZE_HORZ", False);
		Atom wm_act_max_vert = XInternAtom(x11_display, "_NET_WM_ACTION_MAXIMIZE_VERT", False);
		bool found_wm_act_max_horz = false;
		bool found_wm_act_max_vert = false;

		for (uint64_t i = 0; i < len; i++) {
			if (atoms[i] == wm_act_max_horz) {
				found_wm_act_max_horz = true;
			}
			if (atoms[i] == wm_act_max_vert) {
				found_wm_act_max_vert = true;
			}

			if (found_wm_act_max_horz || found_wm_act_max_vert) {
				retval = true;
				break;
			}
		}

		XFree(data);
	}

	return retval;
}

bool DisplayServerX11::window_is_maximize_allowed(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
	return _window_maximize_check(p_window, "_NET_WM_ALLOWED_ACTIONS");
}

void DisplayServerX11::_set_wm_maximized(WindowID p_window, bool p_enabled) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	// Using EWMH -- Extended Window Manager Hints
	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_max_horz = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	Atom wm_max_vert = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_VERT", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = wd.x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = p_enabled ? _NET_WM_STATE_ADD : _NET_WM_STATE_REMOVE;
	xev.xclient.data.l[1] = wm_max_horz;
	xev.xclient.data.l[2] = wm_max_vert;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

	if (p_enabled && window_is_maximize_allowed(p_window)) {
		// Wait for effective resizing (so the GLX context is too).
		// Give up after 0.5s, it's not going to happen on this WM.
		// https://github.com/godotengine/godot/issues/19978
		for (int attempt = 0; window_get_mode(p_window) != WINDOW_MODE_MAXIMIZED && attempt < 50; attempt++) {
			usleep(10000);
		}
	}
}

void DisplayServerX11::_set_wm_fullscreen(WindowID p_window, bool p_enabled) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (p_enabled && !window_get_flag(WINDOW_FLAG_BORDERLESS, p_window)) {
		// remove decorations if the window is not already borderless
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = 0;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		if (property != None) {
			XChangeProperty(x11_display, wd.x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		}
	}

	if (p_enabled) {
		// Set the window as resizable to prevent window managers to ignore the fullscreen state flag.
		_update_size_hints(p_window);
	}

	// Using EWMH -- Extended Window Manager Hints
	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_fullscreen = XInternAtom(x11_display, "_NET_WM_STATE_FULLSCREEN", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = wd.x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = p_enabled ? _NET_WM_STATE_ADD : _NET_WM_STATE_REMOVE;
	xev.xclient.data.l[1] = wm_fullscreen;
	xev.xclient.data.l[2] = 0;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

	// set bypass compositor hint
	Atom bypass_compositor = XInternAtom(x11_display, "_NET_WM_BYPASS_COMPOSITOR", False);
	unsigned long compositing_disable_on = p_enabled ? 1 : 0;
	if (bypass_compositor != None) {
		XChangeProperty(x11_display, wd.x11_window, bypass_compositor, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)&compositing_disable_on, 1);
	}

	XFlush(x11_display);

	if (!p_enabled) {
		// Reset the non-resizable flags if we un-set these before.
		_update_size_hints(p_window);

		// put back or remove decorations according to the last set borderless state
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = window_get_flag(WINDOW_FLAG_BORDERLESS, p_window) ? 0 : 1;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		if (property != None) {
			XChangeProperty(x11_display, wd.x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		}
	}
}

void DisplayServerX11::window_set_mode(WindowMode p_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	WindowMode old_mode = window_get_mode(p_window);
	if (old_mode == p_mode) {
		return; // do nothing
	}
	//remove all "extra" modes

	switch (old_mode) {
		case WINDOW_MODE_WINDOWED: {
			//do nothing
		} break;
		case WINDOW_MODE_MINIMIZED: {
			//Un-Minimize
			// Using ICCCM -- Inter-Client Communication Conventions Manual
			XEvent xev;
			Atom wm_change = XInternAtom(x11_display, "WM_CHANGE_STATE", False);

			memset(&xev, 0, sizeof(xev));
			xev.type = ClientMessage;
			xev.xclient.window = wd.x11_window;
			xev.xclient.message_type = wm_change;
			xev.xclient.format = 32;
			xev.xclient.data.l[0] = WM_NormalState;

			XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

			Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
			Atom wm_hidden = XInternAtom(x11_display, "_NET_WM_STATE_HIDDEN", False);

			memset(&xev, 0, sizeof(xev));
			xev.type = ClientMessage;
			xev.xclient.window = wd.x11_window;
			xev.xclient.message_type = wm_state;
			xev.xclient.format = 32;
			xev.xclient.data.l[0] = _NET_WM_STATE_ADD;
			xev.xclient.data.l[1] = wm_hidden;

			XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
		} break;
		case WINDOW_MODE_FULLSCREEN: {
			//Remove full-screen
			wd.fullscreen = false;

			_set_wm_fullscreen(p_window, false);

			//un-maximize required for always on top
			bool on_top = window_get_flag(WINDOW_FLAG_ALWAYS_ON_TOP, p_window);

			window_set_position(wd.last_position_before_fs, p_window);

			if (on_top) {
				_set_wm_maximized(p_window, false);
			}

		} break;
		case WINDOW_MODE_MAXIMIZED: {
			_set_wm_maximized(p_window, false);
		} break;
	}

	switch (p_mode) {
		case WINDOW_MODE_WINDOWED: {
			//do nothing
		} break;
		case WINDOW_MODE_MINIMIZED: {
			// Using ICCCM -- Inter-Client Communication Conventions Manual
			XEvent xev;
			Atom wm_change = XInternAtom(x11_display, "WM_CHANGE_STATE", False);

			memset(&xev, 0, sizeof(xev));
			xev.type = ClientMessage;
			xev.xclient.window = wd.x11_window;
			xev.xclient.message_type = wm_change;
			xev.xclient.format = 32;
			xev.xclient.data.l[0] = WM_IconicState;

			XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

			Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
			Atom wm_hidden = XInternAtom(x11_display, "_NET_WM_STATE_HIDDEN", False);

			memset(&xev, 0, sizeof(xev));
			xev.type = ClientMessage;
			xev.xclient.window = wd.x11_window;
			xev.xclient.message_type = wm_state;
			xev.xclient.format = 32;
			xev.xclient.data.l[0] = _NET_WM_STATE_ADD;
			xev.xclient.data.l[1] = wm_hidden;

			XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
		} break;
		case WINDOW_MODE_FULLSCREEN: {
			wd.last_position_before_fs = wd.position;

			if (window_get_flag(WINDOW_FLAG_ALWAYS_ON_TOP, p_window)) {
				_set_wm_maximized(p_window, true);
			}

			wd.fullscreen = true;
			_set_wm_fullscreen(p_window, true);
		} break;
		case WINDOW_MODE_MAXIMIZED: {
			_set_wm_maximized(p_window, true);
		} break;
	}
}

DisplayServer::WindowMode DisplayServerX11::window_get_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), WINDOW_MODE_WINDOWED);
	const WindowData &wd = windows[p_window];

	if (wd.fullscreen) { //if fullscreen, it's not in another mode
		return WINDOW_MODE_FULLSCREEN;
	}

	// Test maximized.
	// Using EWMH -- Extended Window Manager Hints
	if (_window_maximize_check(p_window, "_NET_WM_STATE")) {
		return WINDOW_MODE_MAXIMIZED;
	}

	{ // Test minimized.
		// Using ICCCM -- Inter-Client Communication Conventions Manual
		Atom property = XInternAtom(x11_display, "WM_STATE", True);
		if (property == None) {
			return WINDOW_MODE_WINDOWED;
		}

		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = nullptr;

		int result = XGetWindowProperty(
				x11_display,
				wd.x11_window,
				property,
				0,
				32,
				False,
				AnyPropertyType,
				&type,
				&format,
				&len,
				&remaining,
				&data);

		if (result == Success && data) {
			long *state = (long *)data;
			if (state[0] == WM_IconicState) {
				XFree(data);
				return WINDOW_MODE_MINIMIZED;
			}
			XFree(data);
		}
	}

	// All other discarded, return windowed.

	return WINDOW_MODE_WINDOWED;
}

void DisplayServerX11::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			wd.resize_disabled = p_enabled;

			_update_size_hints(p_window);

			XFlush(x11_display);
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			Hints hints;
			Atom property;
			hints.flags = 2;
			hints.decorations = p_enabled ? 0 : 1;
			property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
			if (property != None) {
				XChangeProperty(x11_display, wd.x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
			}

			// Preserve window size
			window_set_size(window_get_size(p_window), p_window);

			wd.borderless = p_enabled;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			ERR_FAIL_COND_MSG(wd.transient_parent != INVALID_WINDOW_ID, "Can't make a window transient if the 'on top' flag is active.");
			if (p_enabled && wd.fullscreen) {
				_set_wm_maximized(p_window, true);
			}

			Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
			Atom wm_above = XInternAtom(x11_display, "_NET_WM_STATE_ABOVE", False);

			XClientMessageEvent xev;
			memset(&xev, 0, sizeof(xev));
			xev.type = ClientMessage;
			xev.window = wd.x11_window;
			xev.message_type = wm_state;
			xev.format = 32;
			xev.data.l[0] = p_enabled ? _NET_WM_STATE_ADD : _NET_WM_STATE_REMOVE;
			xev.data.l[1] = wm_above;
			xev.data.l[3] = 1;
			XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, (XEvent *)&xev);

			if (!p_enabled && !wd.fullscreen) {
				_set_wm_maximized(p_window, false);
			}
			wd.on_top = p_enabled;

		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			//todo reimplement
		} break;
		default: {
		}
	}
}

bool DisplayServerX11::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			return wd.resize_disabled;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			bool borderless = wd.borderless;
			Atom prop = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
			if (prop != None) {
				Atom type;
				int format;
				unsigned long len;
				unsigned long remaining;
				unsigned char *data = nullptr;
				if (XGetWindowProperty(x11_display, wd.x11_window, prop, 0, sizeof(Hints), False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
					if (data && (format == 32) && (len >= 5)) {
						borderless = !((Hints *)data)->decorations;
					}
					if (data) {
						XFree(data);
					}
				}
			}
			return borderless;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			return wd.on_top;
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			//todo reimplement
		} break;
		default: {
		}
	}

	return false;
}

void DisplayServerX11::window_request_attention(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	// Using EWMH -- Extended Window Manager Hints
	//
	// Sets the _NET_WM_STATE_DEMANDS_ATTENTION atom for WM_STATE
	// Will be unset by the window manager after user react on the request for attention

	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_attention = XInternAtom(x11_display, "_NET_WM_STATE_DEMANDS_ATTENTION", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = wd.x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = _NET_WM_STATE_ADD;
	xev.xclient.data.l[1] = wm_attention;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
	XFlush(x11_display);
}

void DisplayServerX11::window_move_to_foreground(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	XEvent xev;
	Atom net_active_window = XInternAtom(x11_display, "_NET_ACTIVE_WINDOW", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = wd.x11_window;
	xev.xclient.message_type = net_active_window;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 1;
	xev.xclient.data.l[1] = CurrentTime;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
	XFlush(x11_display);
}

bool DisplayServerX11::window_can_draw(WindowID p_window) const {
	//this seems to be all that is provided by X11
	return window_get_mode(p_window) != WINDOW_MODE_MINIMIZED;
}

bool DisplayServerX11::can_any_window_draw() const {
	_THREAD_SAFE_METHOD_

	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (window_get_mode(E.key) != WINDOW_MODE_MINIMIZED) {
			return true;
		}
	}

	return false;
}

void DisplayServerX11::window_set_ime_active(const bool p_active, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_active = p_active;

	if (!wd.xic) {
		return;
	}

	// Block events polling while changing input focus
	// because it triggers some event polling internally.
	if (p_active) {
		{
			MutexLock mutex_lock(events_mutex);
			XSetICFocus(wd.xic);
		}
		window_set_ime_position(wd.im_position, p_window);
	} else {
		MutexLock mutex_lock(events_mutex);
		XUnsetICFocus(wd.xic);
	}
}

void DisplayServerX11::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_position = p_pos;

	if (!wd.xic) {
		return;
	}

	::XPoint spot;
	spot.x = short(p_pos.x);
	spot.y = short(p_pos.y);
	XVaNestedList preedit_attr = XVaCreateNestedList(0, XNSpotLocation, &spot, nullptr);

	{
		// Block events polling during this call
		// because it triggers some event polling internally.
		MutexLock mutex_lock(events_mutex);
		XSetICValues(wd.xic, XNPreeditAttributes, preedit_attr, nullptr);
	}

	XFree(preedit_attr);
}

void DisplayServerX11::cursor_set_shape(CursorShape p_shape) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (p_shape == current_cursor) {
		return;
	}

	if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
		if (cursors[p_shape] != None) {
			for (const KeyValue<WindowID, WindowData> &E : windows) {
				XDefineCursor(x11_display, E.value.x11_window, cursors[p_shape]);
			}
		} else if (cursors[CURSOR_ARROW] != None) {
			for (const KeyValue<WindowID, WindowData> &E : windows) {
				XDefineCursor(x11_display, E.value.x11_window, cursors[CURSOR_ARROW]);
			}
		}
	}

	current_cursor = p_shape;
}

DisplayServerX11::CursorShape DisplayServerX11::cursor_get_shape() const {
	return current_cursor;
}

void DisplayServerX11::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	_THREAD_SAFE_METHOD_

	if (p_cursor.is_valid()) {
		Map<CursorShape, Vector<Variant>>::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}

			cursors_cache.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2i texture_size;
		Rect2i atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_image();
		}

		if (!image.is_valid() && atlas_texture.is_valid()) {
			texture = atlas_texture->get_atlas();

			atlas_rect.size.width = texture->get_width();
			atlas_rect.size.height = texture->get_height();
			atlas_rect.position.x = atlas_texture->get_region().position.x;
			atlas_rect.position.y = atlas_texture->get_region().position.y;

			texture_size.width = atlas_texture->get_region().size.x;
			texture_size.height = atlas_texture->get_region().size.y;
		} else if (image.is_valid()) {
			texture_size.width = texture->get_width();
			texture_size.height = texture->get_height();
		}

		ERR_FAIL_COND(!texture.is_valid());
		ERR_FAIL_COND(p_hotspot.x < 0 || p_hotspot.y < 0);
		ERR_FAIL_COND(texture_size.width > 256 || texture_size.height > 256);
		ERR_FAIL_COND(p_hotspot.x > texture_size.width || p_hotspot.y > texture_size.height);

		image = texture->get_image();

		ERR_FAIL_COND(!image.is_valid());

		// Create the cursor structure
		XcursorImage *cursor_image = XcursorImageCreate(texture_size.width, texture_size.height);
		XcursorUInt image_size = texture_size.width * texture_size.height;
		XcursorDim size = sizeof(XcursorPixel) * image_size;

		cursor_image->version = 1;
		cursor_image->size = size;
		cursor_image->xhot = p_hotspot.x;
		cursor_image->yhot = p_hotspot.y;

		// allocate memory to contain the whole file
		cursor_image->pixels = (XcursorPixel *)memalloc(size);

		for (XcursorPixel index = 0; index < image_size; index++) {
			int row_index = floor(index / texture_size.width) + atlas_rect.position.y;
			int column_index = (index % int(texture_size.width)) + atlas_rect.position.x;

			if (atlas_texture.is_valid()) {
				column_index = MIN(column_index, atlas_rect.size.width - 1);
				row_index = MIN(row_index, atlas_rect.size.height - 1);
			}

			*(cursor_image->pixels + index) = image->get_pixel(column_index, row_index).to_argb32();
		}

		ERR_FAIL_COND(cursor_image->pixels == nullptr);

		// Save it for a further usage
		cursors[p_shape] = XcursorImageLoadCursor(x11_display, cursor_image);

		Vector<Variant> params;
		params.push_back(p_cursor);
		params.push_back(p_hotspot);
		cursors_cache.insert(p_shape, params);

		if (p_shape == current_cursor) {
			if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
				for (const KeyValue<WindowID, WindowData> &E : windows) {
					XDefineCursor(x11_display, E.value.x11_window, cursors[p_shape]);
				}
			}
		}

		memfree(cursor_image->pixels);
		XcursorImageDestroy(cursor_image);
	} else {
		// Reset to default system cursor
		if (img[p_shape]) {
			cursors[p_shape] = XcursorImageLoadCursor(x11_display, img[p_shape]);
		}

		CursorShape c = current_cursor;
		current_cursor = CURSOR_MAX;
		cursor_set_shape(c);

		cursors_cache.erase(p_shape);
	}
}

int DisplayServerX11::keyboard_get_layout_count() const {
	int _group_count = 0;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);

		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != nullptr) {
			_group_count = kbd->ctrls->num_groups;
		} else {
			while (_group_count < XkbNumKbdGroups && groups[_group_count] != None) {
				_group_count++;
			}
		}
		XkbFreeKeyboard(kbd, 0, true);
	}
	return _group_count;
}

int DisplayServerX11::keyboard_get_current_layout() const {
	XkbStateRec state;
	XkbGetState(x11_display, XkbUseCoreKbd, &state);
	return state.group;
}

void DisplayServerX11::keyboard_set_current_layout(int p_index) {
	ERR_FAIL_INDEX(p_index, keyboard_get_layout_count());
	XkbLockGroup(x11_display, XkbUseCoreKbd, p_index);
}

String DisplayServerX11::keyboard_get_layout_language(int p_index) const {
	String ret;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);
		XkbGetNames(x11_display, XkbGroupNamesMask, kbd);

		int _group_count = 0;
		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != nullptr) {
			_group_count = kbd->ctrls->num_groups;
		} else {
			while (_group_count < XkbNumKbdGroups && groups[_group_count] != None) {
				_group_count++;
			}
		}

		Atom names = kbd->names->symbols;
		if (names != None) {
			char *name = XGetAtomName(x11_display, names);
			Vector<String> info = String(name).split("+");
			if (p_index >= 0 && p_index < _group_count) {
				if (p_index + 1 < info.size()) {
					ret = info[p_index + 1]; // Skip "pc" at the start and "inet"/"group" at the end of symbols.
				} else {
					ret = "en"; // No symbol for layout fallback to "en".
				}
			} else {
				ERR_PRINT("Index " + itos(p_index) + "is out of bounds (" + itos(_group_count) + ").");
			}
			XFree(name);
		}
		XkbFreeKeyboard(kbd, 0, true);
	}
	return ret.substr(0, 2);
}

String DisplayServerX11::keyboard_get_layout_name(int p_index) const {
	String ret;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);
		XkbGetNames(x11_display, XkbGroupNamesMask, kbd);

		int _group_count = 0;
		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != nullptr) {
			_group_count = kbd->ctrls->num_groups;
		} else {
			while (_group_count < XkbNumKbdGroups && groups[_group_count] != None) {
				_group_count++;
			}
		}

		if (p_index >= 0 && p_index < _group_count) {
			char *full_name = XGetAtomName(x11_display, groups[p_index]);
			ret.parse_utf8(full_name);
			XFree(full_name);
		} else {
			ERR_PRINT("Index " + itos(p_index) + "is out of bounds (" + itos(_group_count) + ").");
		}
		XkbFreeKeyboard(kbd, 0, true);
	}
	return ret;
}

Key DisplayServerX11::keyboard_get_keycode_from_physical(Key p_keycode) const {
	unsigned int modifiers = p_keycode & KEY_MODIFIER_MASK;
	unsigned int keycode_no_mod = p_keycode & KEY_CODE_MASK;
	unsigned int xkeycode = KeyMappingX11::get_xlibcode((Key)keycode_no_mod);
	KeySym xkeysym = XkbKeycodeToKeysym(x11_display, xkeycode, 0, 0);
	if (xkeysym >= 'a' && xkeysym <= 'z') {
		xkeysym -= ('a' - 'A');
	}

	Key key = KeyMappingX11::get_keycode(xkeysym);
	// If not found, fallback to QWERTY.
	// This should match the behavior of the event pump
	if (key == KEY_NONE) {
		return p_keycode;
	}
	return (Key)(key | modifiers);
}

DisplayServerX11::Property DisplayServerX11::_read_property(Display *p_display, Window p_window, Atom p_property) {
	Atom actual_type = None;
	int actual_format = 0;
	unsigned long nitems = 0;
	unsigned long bytes_after = 0;
	unsigned char *ret = nullptr;

	int read_bytes = 1024;

	// Keep trying to read the property until there are no bytes unread.
	if (p_property != None) {
		do {
			if (ret != nullptr) {
				XFree(ret);
			}

			XGetWindowProperty(p_display, p_window, p_property, 0, read_bytes, False, AnyPropertyType,
					&actual_type, &actual_format, &nitems, &bytes_after,
					&ret);

			read_bytes *= 2;

		} while (bytes_after != 0);
	}

	Property p = { ret, actual_format, (int)nitems, actual_type };

	return p;
}

static Atom pick_target_from_list(Display *p_display, Atom *p_list, int p_count) {
	static const char *target_type = "text/uri-list";

	for (int i = 0; i < p_count; i++) {
		Atom atom = p_list[i];

		if (atom != None && String(XGetAtomName(p_display, atom)) == target_type) {
			return atom;
		}
	}
	return None;
}

static Atom pick_target_from_atoms(Display *p_disp, Atom p_t1, Atom p_t2, Atom p_t3) {
	static const char *target_type = "text/uri-list";
	if (p_t1 != None && String(XGetAtomName(p_disp, p_t1)) == target_type) {
		return p_t1;
	}

	if (p_t2 != None && String(XGetAtomName(p_disp, p_t2)) == target_type) {
		return p_t2;
	}

	if (p_t3 != None && String(XGetAtomName(p_disp, p_t3)) == target_type) {
		return p_t3;
	}

	return None;
}

void DisplayServerX11::_get_key_modifier_state(unsigned int p_x11_state, Ref<InputEventWithModifiers> state) {
	state->set_shift_pressed((p_x11_state & ShiftMask));
	state->set_ctrl_pressed((p_x11_state & ControlMask));
	state->set_alt_pressed((p_x11_state & Mod1Mask /*|| p_x11_state&Mod5Mask*/)); //altgr should not count as alt
	state->set_meta_pressed((p_x11_state & Mod4Mask));
}

MouseButton DisplayServerX11::_get_mouse_button_state(MouseButton p_x11_button, int p_x11_type) {
	MouseButton mask = MouseButton(1 << (p_x11_button - 1));

	if (p_x11_type == ButtonPress) {
		last_button_state |= mask;
	} else {
		last_button_state &= MouseButton(~mask);
	}

	return last_button_state;
}

void DisplayServerX11::_handle_key_event(WindowID p_window, XKeyEvent *p_event, LocalVector<XEvent> &p_events, uint32_t &p_event_index, bool p_echo) {
	WindowData wd = windows[p_window];
	// X11 functions don't know what const is
	XKeyEvent *xkeyevent = p_event;

	// This code was pretty difficult to write.
	// The docs stink and every toolkit seems to
	// do it in a different way.

	/* Phase 1, obtain a proper keysym */

	// This was also very difficult to figure out.
	// You'd expect you could just use Keysym provided by
	// XKeycodeToKeysym to obtain internationalized
	// input.. WRONG!!
	// you must use XLookupString (???) which not only wastes
	// cycles generating an unnecessary string, but also
	// still works in half the cases. (won't handle deadkeys)
	// For more complex input methods (deadkeys and more advanced)
	// you have to use XmbLookupString (??).
	// So then you have to choose which of both results
	// you want to keep.
	// This is a real bizarreness and cpu waster.

	KeySym keysym_keycode = 0; // keysym used to find a keycode
	KeySym keysym_unicode = 0; // keysym used to find unicode

	// XLookupString returns keysyms usable as nice keycodes.
	char str[256 + 1];
	XKeyEvent xkeyevent_no_mod = *xkeyevent;
	xkeyevent_no_mod.state &= ~ShiftMask;
	xkeyevent_no_mod.state &= ~ControlMask;
	XLookupString(xkeyevent, str, 256, &keysym_unicode, nullptr);
	XLookupString(&xkeyevent_no_mod, nullptr, 0, &keysym_keycode, nullptr);

	// Meanwhile, XLookupString returns keysyms useful for unicode.

	if (!xmbstring) {
		// keep a temporary buffer for the string
		xmbstring = (char *)memalloc(sizeof(char) * 8);
		xmblen = 8;
	}

	if (xkeyevent->type == KeyPress && wd.xic) {
		Status status;
#ifdef X_HAVE_UTF8_STRING
		int utf8len = 8;
		char *utf8string = (char *)memalloc(sizeof(char) * utf8len);
		int utf8bytes = Xutf8LookupString(wd.xic, xkeyevent, utf8string,
				utf8len - 1, &keysym_unicode, &status);
		if (status == XBufferOverflow) {
			utf8len = utf8bytes + 1;
			utf8string = (char *)memrealloc(utf8string, utf8len);
			utf8bytes = Xutf8LookupString(wd.xic, xkeyevent, utf8string,
					utf8len - 1, &keysym_unicode, &status);
		}
		utf8string[utf8bytes] = '\0';

		if (status == XLookupChars) {
			bool keypress = xkeyevent->type == KeyPress;
			Key keycode = KeyMappingX11::get_keycode(keysym_keycode);
			unsigned int physical_keycode = KeyMappingX11::get_scancode(xkeyevent->keycode);

			if (keycode >= 'a' && keycode <= 'z') {
				keycode -= 'a' - 'A';
			}

			String tmp;
			tmp.parse_utf8(utf8string, utf8bytes);
			for (int i = 0; i < tmp.length(); i++) {
				Ref<InputEventKey> k;
				k.instantiate();
				if (physical_keycode == 0 && keycode == 0 && tmp[i] == 0) {
					continue;
				}

				if (keycode == 0) {
					keycode = (Key)physical_keycode;
				}

				_get_key_modifier_state(xkeyevent->state, k);

				k->set_window_id(p_window);
				k->set_unicode(tmp[i]);

				k->set_pressed(keypress);

				k->set_keycode(keycode);

				k->set_physical_keycode((Key)physical_keycode);

				k->set_echo(false);

				if (k->get_keycode() == KEY_BACKTAB) {
					//make it consistent across platforms.
					k->set_keycode(KEY_TAB);
					k->set_physical_keycode(KEY_TAB);
					k->set_shift_pressed(true);
				}

				Input::get_singleton()->parse_input_event(k);
			}
			memfree(utf8string);
			return;
		}
		memfree(utf8string);
#else
		do {
			int mnbytes = XmbLookupString(xic, xkeyevent, xmbstring, xmblen - 1, &keysym_unicode, &status);
			xmbstring[mnbytes] = '\0';

			if (status == XBufferOverflow) {
				xmblen = mnbytes + 1;
				xmbstring = (char *)memrealloc(xmbstring, xmblen);
			}
		} while (status == XBufferOverflow);
#endif
	}

	/* Phase 2, obtain a Godot keycode from the keysym */

	// KeyMappingX11 just translated the X11 keysym to a PIGUI
	// keysym, so it works in all platforms the same.

	Key keycode = KeyMappingX11::get_keycode(keysym_keycode);
	unsigned int physical_keycode = KeyMappingX11::get_scancode(xkeyevent->keycode);

	/* Phase 3, obtain a unicode character from the keysym */

	// KeyMappingX11 also translates keysym to unicode.
	// It does a binary search on a table to translate
	// most properly.
	unsigned int unicode = keysym_unicode > 0 ? KeyMappingX11::get_unicode_from_keysym(keysym_unicode) : 0;

	/* Phase 4, determine if event must be filtered */

	// This seems to be a side-effect of using XIM.
	// XFilterEvent looks like a core X11 function,
	// but it's actually just used to see if we must
	// ignore a deadkey, or events XIM determines
	// must not reach the actual gui.
	// Guess it was a design problem of the extension

	bool keypress = xkeyevent->type == KeyPress;

	if (physical_keycode == 0 && keycode == KEY_NONE && unicode == 0) {
		return;
	}

	if (keycode == KEY_NONE) {
		keycode = (Key)physical_keycode;
	}

	/* Phase 5, determine modifier mask */

	// No problems here, except I had no way to
	// know Mod1 was ALT and Mod4 was META (applekey/winkey)
	// just tried Mods until i found them.

	//print_verbose("mod1: "+itos(xkeyevent->state&Mod1Mask)+" mod 5: "+itos(xkeyevent->state&Mod5Mask));

	Ref<InputEventKey> k;
	k.instantiate();
	k->set_window_id(p_window);

	_get_key_modifier_state(xkeyevent->state, k);

	/* Phase 6, determine echo character */

	// Echo characters in X11 are a keyrelease and a keypress
	// one after the other with the (almot) same timestamp.
	// To detect them, i compare to the next event in list and
	// check that their difference in time is below a threshold.

	if (xkeyevent->type != KeyPress) {
		p_echo = false;

		// make sure there are events pending,
		// so this call won't block.
		if (p_event_index + 1 < p_events.size()) {
			XEvent &peek_event = p_events[p_event_index + 1];

			// I'm using a threshold of 5 msecs,
			// since sometimes there seems to be a little
			// jitter. I'm still not convinced that all this approach
			// is correct, but the xorg developers are
			// not very helpful today.

#define ABSDIFF(x, y) (((x) < (y)) ? ((y) - (x)) : ((x) - (y)))
			::Time threshold = ABSDIFF(peek_event.xkey.time, xkeyevent->time);
#undef ABSDIFF
			if (peek_event.type == KeyPress && threshold < 5) {
				KeySym rk;
				XLookupString((XKeyEvent *)&peek_event, str, 256, &rk, nullptr);
				if (rk == keysym_keycode) {
					// Consume to next event.
					++p_event_index;
					_handle_key_event(p_window, (XKeyEvent *)&peek_event, p_events, p_event_index, true);
					return; //ignore current, echo next
				}
			}

			// use the time from peek_event so it always works
		}

		// save the time to check for echo when keypress happens
	}

	/* Phase 7, send event to Window */

	k->set_pressed(keypress);

	if (keycode >= 'a' && keycode <= 'z') {
		keycode -= int('a' - 'A');
	}

	k->set_keycode(keycode);
	k->set_physical_keycode((Key)physical_keycode);
	k->set_unicode(unicode);
	k->set_echo(p_echo);

	if (k->get_keycode() == KEY_BACKTAB) {
		//make it consistent across platforms.
		k->set_keycode(KEY_TAB);
		k->set_physical_keycode(KEY_TAB);
		k->set_shift_pressed(true);
	}

	//don't set mod state if modifier keys are released by themselves
	//else event.is_action() will not work correctly here
	if (!k->is_pressed()) {
		if (k->get_keycode() == KEY_SHIFT) {
			k->set_shift_pressed(false);
		} else if (k->get_keycode() == KEY_CTRL) {
			k->set_ctrl_pressed(false);
		} else if (k->get_keycode() == KEY_ALT) {
			k->set_alt_pressed(false);
		} else if (k->get_keycode() == KEY_META) {
			k->set_meta_pressed(false);
		}
	}

	bool last_is_pressed = Input::get_singleton()->is_key_pressed(k->get_keycode());
	if (k->is_pressed()) {
		if (last_is_pressed) {
			k->set_echo(true);
		}
	}

	Input::get_singleton()->parse_input_event(k);
}

Atom DisplayServerX11::_process_selection_request_target(Atom p_target, Window p_requestor, Atom p_property, Atom p_selection) const {
	if (p_target == XInternAtom(x11_display, "TARGETS", 0)) {
		// Request to list all supported targets.
		Atom data[9];
		data[0] = XInternAtom(x11_display, "TARGETS", 0);
		data[1] = XInternAtom(x11_display, "SAVE_TARGETS", 0);
		data[2] = XInternAtom(x11_display, "MULTIPLE", 0);
		data[3] = XInternAtom(x11_display, "UTF8_STRING", 0);
		data[4] = XInternAtom(x11_display, "COMPOUND_TEXT", 0);
		data[5] = XInternAtom(x11_display, "TEXT", 0);
		data[6] = XA_STRING;
		data[7] = XInternAtom(x11_display, "text/plain;charset=utf-8", 0);
		data[8] = XInternAtom(x11_display, "text/plain", 0);

		XChangeProperty(x11_display,
				p_requestor,
				p_property,
				XA_ATOM,
				32,
				PropModeReplace,
				(unsigned char *)&data,
				sizeof(data) / sizeof(data[0]));
		return p_property;
	} else if (p_target == XInternAtom(x11_display, "SAVE_TARGETS", 0)) {
		// Request to check if SAVE_TARGETS is supported, nothing special to do.
		XChangeProperty(x11_display,
				p_requestor,
				p_property,
				XInternAtom(x11_display, "NULL", False),
				32,
				PropModeReplace,
				nullptr,
				0);
		return p_property;
	} else if (p_target == XInternAtom(x11_display, "UTF8_STRING", 0) ||
			p_target == XInternAtom(x11_display, "COMPOUND_TEXT", 0) ||
			p_target == XInternAtom(x11_display, "TEXT", 0) ||
			p_target == XA_STRING ||
			p_target == XInternAtom(x11_display, "text/plain;charset=utf-8", 0) ||
			p_target == XInternAtom(x11_display, "text/plain", 0)) {
		// Directly using internal clipboard because we know our window
		// is the owner during a selection request.
		CharString clip;
		static const char *target_type = "PRIMARY";
		if (p_selection != None && String(XGetAtomName(x11_display, p_selection)) == target_type) {
			clip = internal_clipboard_primary.utf8();
		} else {
			clip = internal_clipboard.utf8();
		}
		XChangeProperty(x11_display,
				p_requestor,
				p_property,
				p_target,
				8,
				PropModeReplace,
				(unsigned char *)clip.get_data(),
				clip.length());
		return p_property;
	} else {
		char *target_name = XGetAtomName(x11_display, p_target);
		printf("Target '%s' not supported.\n", target_name);
		if (target_name) {
			XFree(target_name);
		}
		return None;
	}
}

void DisplayServerX11::_handle_selection_request_event(XSelectionRequestEvent *p_event) const {
	XEvent respond;
	if (p_event->target == XInternAtom(x11_display, "MULTIPLE", 0)) {
		// Request for multiple target conversions at once.
		Atom atom_pair = XInternAtom(x11_display, "ATOM_PAIR", False);
		respond.xselection.property = None;

		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = nullptr;
		if (XGetWindowProperty(x11_display, p_event->requestor, p_event->property, 0, LONG_MAX, False, atom_pair, &type, &format, &len, &remaining, &data) == Success) {
			if ((len >= 2) && data) {
				Atom *targets = (Atom *)data;
				for (uint64_t i = 0; i < len; i += 2) {
					Atom target = targets[i];
					Atom &property = targets[i + 1];
					property = _process_selection_request_target(target, p_event->requestor, property, p_event->selection);
				}

				XChangeProperty(x11_display,
						p_event->requestor,
						p_event->property,
						atom_pair,
						32,
						PropModeReplace,
						(unsigned char *)targets,
						len);

				respond.xselection.property = p_event->property;
			}
			XFree(data);
		}
	} else {
		// Request for target conversion.
		respond.xselection.property = _process_selection_request_target(p_event->target, p_event->requestor, p_event->property, p_event->selection);
	}

	respond.xselection.type = SelectionNotify;
	respond.xselection.display = p_event->display;
	respond.xselection.requestor = p_event->requestor;
	respond.xselection.selection = p_event->selection;
	respond.xselection.target = p_event->target;
	respond.xselection.time = p_event->time;

	XSendEvent(x11_display, p_event->requestor, True, NoEventMask, &respond);
	XFlush(x11_display);
}

void DisplayServerX11::_xim_destroy_callback(::XIM im, ::XPointer client_data,
		::XPointer call_data) {
	WARN_PRINT("Input method stopped");
	DisplayServerX11 *ds = reinterpret_cast<DisplayServerX11 *>(client_data);
	ds->xim = nullptr;

	for (KeyValue<WindowID, WindowData> &E : ds->windows) {
		E.value.xic = nullptr;
	}
}

void DisplayServerX11::_window_changed(XEvent *event) {
	WindowID window_id = MAIN_WINDOW_ID;

	// Assign the event to the relevant window
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (event->xany.window == E.value.x11_window) {
			window_id = E.key;
			break;
		}
	}

	Rect2i new_rect;

	WindowData &wd = windows[window_id];
	if (wd.x11_window != event->xany.window) { // Check if the correct window, in case it was not main window or anything else
		return;
	}

	{
		//the position in xconfigure is not useful here, obtain it manually
		int x, y;
		Window child;
		XTranslateCoordinates(x11_display, wd.x11_window, DefaultRootWindow(x11_display), 0, 0, &x, &y, &child);
		new_rect.position.x = x;
		new_rect.position.y = y;

		new_rect.size.width = event->xconfigure.width;
		new_rect.size.height = event->xconfigure.height;
	}

	if (new_rect == Rect2i(wd.position, wd.size)) {
		return;
	}
	if (wd.xic) {
		//  Not portable.
		window_set_ime_position(Point2(0, 1));
	}

	wd.position = new_rect.position;
	wd.size = new_rect.size;

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		context_vulkan->window_resize(window_id, wd.size.width, wd.size.height);
	}
#endif

	if (!wd.rect_changed_callback.is_null()) {
		Rect2i r = new_rect;

		Variant rect = r;

		Variant *rectp = &rect;
		Variant ret;
		Callable::CallError ce;
		wd.rect_changed_callback.call((const Variant **)&rectp, 1, ret, ce);
	}
}

void DisplayServerX11::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerX11 *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerX11::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	Variant ev = p_event;
	Variant *evp = &ev;
	Variant ret;
	Callable::CallError ce;

	Ref<InputEventFromWindow> event_from_window = p_event;
	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		//send to a window
		ERR_FAIL_COND(!windows.has(event_from_window->get_window_id()));
		Callable callable = windows[event_from_window->get_window_id()].input_event_callback;
		if (callable.is_null()) {
			return;
		}
		callable.call((const Variant **)&evp, 1, ret, ce);
	} else {
		//send to all windows
		for (KeyValue<WindowID, WindowData> &E : windows) {
			Callable callable = E.value.input_event_callback;
			if (callable.is_null()) {
				continue;
			}
			callable.call((const Variant **)&evp, 1, ret, ce);
		}
	}
}

void DisplayServerX11::_send_window_event(const WindowData &wd, WindowEvent p_event) {
	if (!wd.event_callback.is_null()) {
		Variant event = int(p_event);
		Variant *eventp = &event;
		Variant ret;
		Callable::CallError ce;
		wd.event_callback.call((const Variant **)&eventp, 1, ret, ce);
	}
}

void DisplayServerX11::_poll_events_thread(void *ud) {
	DisplayServerX11 *display_server = (DisplayServerX11 *)ud;
	display_server->_poll_events();
}

Bool DisplayServerX11::_predicate_all_events(Display *display, XEvent *event, XPointer arg) {
	// Just accept all events.
	return True;
}

bool DisplayServerX11::_wait_for_events() const {
	int x11_fd = ConnectionNumber(x11_display);
	fd_set in_fds;

	XFlush(x11_display);

	FD_ZERO(&in_fds);
	FD_SET(x11_fd, &in_fds);

	struct timeval tv;
	tv.tv_usec = 0;
	tv.tv_sec = 1;

	// Wait for next event or timeout.
	int num_ready_fds = select(x11_fd + 1, &in_fds, nullptr, nullptr, &tv);

	if (num_ready_fds > 0) {
		// Event received.
		return true;
	} else {
		// Error or timeout.
		if (num_ready_fds < 0) {
			ERR_PRINT("_wait_for_events: select error: " + itos(errno));
		}
		return false;
	}
}

void DisplayServerX11::_poll_events() {
	while (!events_thread_done.is_set()) {
		_wait_for_events();

		// Process events from the queue.
		{
			MutexLock mutex_lock(events_mutex);

			_check_pending_events(polled_events);
		}
	}
}

void DisplayServerX11::_check_pending_events(LocalVector<XEvent> &r_events) {
	// Flush to make sure to gather all pending events.
	XFlush(x11_display);

	// Non-blocking wait for next event and remove it from the queue.
	XEvent ev;
	while (XCheckIfEvent(x11_display, &ev, _predicate_all_events, nullptr)) {
		// Check if the input manager wants to process the event.
		if (XFilterEvent(&ev, None)) {
			// Event has been filtered by the Input Manager,
			// it has to be ignored and a new one will be received.
			continue;
		}

		// Handle selection request events directly in the event thread, because
		// communication through the x server takes several events sent back and forth
		// and we don't want to block other programs while processing only one each frame.
		if (ev.type == SelectionRequest) {
			_handle_selection_request_event(&(ev.xselectionrequest));
			continue;
		}

		r_events.push_back(ev);
	}
}

void DisplayServerX11::process_events() {
	_THREAD_SAFE_METHOD_

#ifdef DISPLAY_SERVER_X11_DEBUG_LOGS_ENABLED
	static int frame = 0;
	++frame;
#endif

	if (app_focused) {
		//verify that one of the windows has focus, else send focus out notification
		bool focus_found = false;
		for (const KeyValue<WindowID, WindowData> &E : windows) {
			if (E.value.focused) {
				focus_found = true;
				break;
			}
		}

		if (!focus_found) {
			uint64_t delta = OS::get_singleton()->get_ticks_msec() - time_since_no_focus;

			if (delta > 250) {
				//X11 can go between windows and have no focus for a while, when creating them or something else. Use this as safety to avoid unnecessary focus in/outs.
				if (OS::get_singleton()->get_main_loop()) {
					DEBUG_LOG_X11("All focus lost, triggering NOTIFICATION_APPLICATION_FOCUS_OUT\n");
					OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
				}
				app_focused = false;
			}
		} else {
			time_since_no_focus = OS::get_singleton()->get_ticks_msec();
		}
	}

	do_mouse_warp = false;

	// Is the current mouse mode one where it needs to be grabbed.
	bool mouse_mode_grab = mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN;

	xi.pressure = 0;
	xi.tilt = Vector2();
	xi.pressure_supported = false;

	LocalVector<XEvent> events;
	{
		// Block events polling while flushing events.
		MutexLock mutex_lock(events_mutex);
		events = polled_events;
		polled_events.clear();

		// Check for more pending events to avoid an extra frame delay.
		_check_pending_events(events);
	}

	for (uint32_t event_index = 0; event_index < events.size(); ++event_index) {
		XEvent &event = events[event_index];

		WindowID window_id = MAIN_WINDOW_ID;

		// Assign the event to the relevant window
		for (const KeyValue<WindowID, WindowData> &E : windows) {
			if (event.xany.window == E.value.x11_window) {
				window_id = E.key;
				break;
			}
		}

		if (XGetEventData(x11_display, &event.xcookie)) {
			if (event.xcookie.type == GenericEvent && event.xcookie.extension == xi.opcode) {
				XIDeviceEvent *event_data = (XIDeviceEvent *)event.xcookie.data;
				int index = event_data->detail;
				Vector2 pos = Vector2(event_data->event_x, event_data->event_y);

				switch (event_data->evtype) {
					case XI_HierarchyChanged:
					case XI_DeviceChanged: {
						_refresh_device_info();
					} break;
					case XI_RawMotion: {
						XIRawEvent *raw_event = (XIRawEvent *)event_data;
						int device_id = raw_event->deviceid;

						// Determine the axis used (called valuators in XInput for some forsaken reason)
						//  Mask is a bitmask indicating which axes are involved.
						//  We are interested in the values of axes 0 and 1.
						if (raw_event->valuators.mask_len <= 0) {
							break;
						}

						const double *values = raw_event->raw_values;

						double rel_x = 0.0;
						double rel_y = 0.0;

						if (XIMaskIsSet(raw_event->valuators.mask, VALUATOR_ABSX)) {
							rel_x = *values;
							values++;
						}

						if (XIMaskIsSet(raw_event->valuators.mask, VALUATOR_ABSY)) {
							rel_y = *values;
							values++;
						}

						if (XIMaskIsSet(raw_event->valuators.mask, VALUATOR_PRESSURE)) {
							Map<int, Vector2>::Element *pen_pressure = xi.pen_pressure_range.find(device_id);
							if (pen_pressure) {
								Vector2 pen_pressure_range = pen_pressure->value();
								if (pen_pressure_range != Vector2()) {
									xi.pressure_supported = true;
									xi.pressure = (*values - pen_pressure_range[0]) /
											(pen_pressure_range[1] - pen_pressure_range[0]);
								}
							}

							values++;
						}

						if (XIMaskIsSet(raw_event->valuators.mask, VALUATOR_TILTX)) {
							Map<int, Vector2>::Element *pen_tilt_x = xi.pen_tilt_x_range.find(device_id);
							if (pen_tilt_x) {
								Vector2 pen_tilt_x_range = pen_tilt_x->value();
								if (pen_tilt_x_range != Vector2()) {
									xi.tilt.x = ((*values - pen_tilt_x_range[0]) / (pen_tilt_x_range[1] - pen_tilt_x_range[0])) * 2 - 1;
								}
							}

							values++;
						}

						if (XIMaskIsSet(raw_event->valuators.mask, VALUATOR_TILTY)) {
							Map<int, Vector2>::Element *pen_tilt_y = xi.pen_tilt_y_range.find(device_id);
							if (pen_tilt_y) {
								Vector2 pen_tilt_y_range = pen_tilt_y->value();
								if (pen_tilt_y_range != Vector2()) {
									xi.tilt.y = ((*values - pen_tilt_y_range[0]) / (pen_tilt_y_range[1] - pen_tilt_y_range[0])) * 2 - 1;
								}
							}

							values++;
						}

						// https://bugs.freedesktop.org/show_bug.cgi?id=71609
						// http://lists.libsdl.org/pipermail/commits-libsdl.org/2015-June/000282.html
						if (raw_event->time == xi.last_relative_time && rel_x == xi.relative_motion.x && rel_y == xi.relative_motion.y) {
							break; // Flush duplicate to avoid overly fast motion
						}

						xi.old_raw_pos.x = xi.raw_pos.x;
						xi.old_raw_pos.y = xi.raw_pos.y;
						xi.raw_pos.x = rel_x;
						xi.raw_pos.y = rel_y;

						Map<int, Vector2>::Element *abs_info = xi.absolute_devices.find(device_id);

						if (abs_info) {
							// Absolute mode device
							Vector2 mult = abs_info->value();

							xi.relative_motion.x += (xi.raw_pos.x - xi.old_raw_pos.x) * mult.x;
							xi.relative_motion.y += (xi.raw_pos.y - xi.old_raw_pos.y) * mult.y;
						} else {
							// Relative mode device
							xi.relative_motion.x = xi.raw_pos.x;
							xi.relative_motion.y = xi.raw_pos.y;
						}

						xi.last_relative_time = raw_event->time;
					} break;
#ifdef TOUCH_ENABLED
					case XI_TouchBegin:
					case XI_TouchEnd: {
						bool is_begin = event_data->evtype == XI_TouchBegin;

						Ref<InputEventScreenTouch> st;
						st.instantiate();
						st->set_window_id(window_id);
						st->set_index(index);
						st->set_position(pos);
						st->set_pressed(is_begin);

						if (is_begin) {
							if (xi.state.has(index)) { // Defensive
								break;
							}
							xi.state[index] = pos;
							if (xi.state.size() == 1) {
								// X11 may send a motion event when a touch gesture begins, that would result
								// in a spurious mouse motion event being sent to Godot; remember it to be able to filter it out
								xi.mouse_pos_to_filter = pos;
							}
							Input::get_singleton()->parse_input_event(st);
						} else {
							if (!xi.state.has(index)) { // Defensive
								break;
							}
							xi.state.erase(index);
							Input::get_singleton()->parse_input_event(st);
						}
					} break;

					case XI_TouchUpdate: {
						Map<int, Vector2>::Element *curr_pos_elem = xi.state.find(index);
						if (!curr_pos_elem) { // Defensive
							break;
						}

						if (curr_pos_elem->value() != pos) {
							Ref<InputEventScreenDrag> sd;
							sd.instantiate();
							sd->set_window_id(window_id);
							sd->set_index(index);
							sd->set_position(pos);
							sd->set_relative(pos - curr_pos_elem->value());
							Input::get_singleton()->parse_input_event(sd);

							curr_pos_elem->value() = pos;
						}
					} break;
#endif
				}
			}
		}
		XFreeEventData(x11_display, &event.xcookie);

		switch (event.type) {
			case MapNotify: {
				DEBUG_LOG_X11("[%u] MapNotify window=%lu (%u) \n", frame, event.xmap.window, window_id);

				const WindowData &wd = windows[window_id];

				// Set focus when menu window is started.
				// RevertToPointerRoot is used to make sure we don't lose all focus in case
				// a subwindow and its parent are both destroyed.
				if (wd.menu_type && !wd.no_focus) {
					XSetInputFocus(x11_display, wd.x11_window, RevertToPointerRoot, CurrentTime);
				}
			} break;

			case Expose: {
				DEBUG_LOG_X11("[%u] Expose window=%lu (%u), count='%u' \n", frame, event.xexpose.window, window_id, event.xexpose.count);

				Main::force_redraw();
			} break;

			case NoExpose: {
				DEBUG_LOG_X11("[%u] NoExpose drawable=%lu (%u) \n", frame, event.xnoexpose.drawable, window_id);

				windows[window_id].minimized = true;
			} break;

			case VisibilityNotify: {
				DEBUG_LOG_X11("[%u] VisibilityNotify window=%lu (%u), state=%u \n", frame, event.xvisibility.window, window_id, event.xvisibility.state);

				XVisibilityEvent *visibility = (XVisibilityEvent *)&event;
				windows[window_id].minimized = (visibility->state == VisibilityFullyObscured);
			} break;

			case LeaveNotify: {
				DEBUG_LOG_X11("[%u] LeaveNotify window=%lu (%u), mode='%u' \n", frame, event.xcrossing.window, window_id, event.xcrossing.mode);

				if (!mouse_mode_grab) {
					_send_window_event(windows[window_id], WINDOW_EVENT_MOUSE_EXIT);
				}

			} break;

			case EnterNotify: {
				DEBUG_LOG_X11("[%u] EnterNotify window=%lu (%u), mode='%u' \n", frame, event.xcrossing.window, window_id, event.xcrossing.mode);

				if (!mouse_mode_grab) {
					_send_window_event(windows[window_id], WINDOW_EVENT_MOUSE_ENTER);
				}
			} break;

			case FocusIn: {
				DEBUG_LOG_X11("[%u] FocusIn window=%lu (%u), mode='%u' \n", frame, event.xfocus.window, window_id, event.xfocus.mode);

				WindowData &wd = windows[window_id];

				wd.focused = true;

				if (wd.xic) {
					// Block events polling while changing input focus
					// because it triggers some event polling internally.
					MutexLock mutex_lock(events_mutex);
					XSetICFocus(wd.xic);
				}

				// Keep track of focus order for overlapping windows.
				static unsigned int focus_order = 0;
				wd.focus_order = ++focus_order;

				_send_window_event(wd, WINDOW_EVENT_FOCUS_IN);

				if (mouse_mode_grab) {
					// Show and update the cursor if confined and the window regained focus.

					for (const KeyValue<WindowID, WindowData> &E : windows) {
						if (mouse_mode == MOUSE_MODE_CONFINED) {
							XUndefineCursor(x11_display, E.value.x11_window);
						} else if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN) { // Or re-hide it.
							XDefineCursor(x11_display, E.value.x11_window, null_cursor);
						}

						XGrabPointer(
								x11_display, E.value.x11_window, True,
								ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
								GrabModeAsync, GrabModeAsync, E.value.x11_window, None, CurrentTime);
					}
				}
#ifdef TOUCH_ENABLED
				// Grab touch devices to avoid OS gesture interference
				/*for (int i = 0; i < xi.touch_devices.size(); ++i) {
					XIGrabDevice(x11_display, xi.touch_devices[i], x11_window, CurrentTime, None, XIGrabModeAsync, XIGrabModeAsync, False, &xi.touch_event_mask);
				}*/
#endif

				if (!app_focused) {
					if (OS::get_singleton()->get_main_loop()) {
						OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
					}
					app_focused = true;
				}
			} break;

			case FocusOut: {
				DEBUG_LOG_X11("[%u] FocusOut window=%lu (%u), mode='%u' \n", frame, event.xfocus.window, window_id, event.xfocus.mode);

				WindowData &wd = windows[window_id];

				wd.focused = false;

				if (wd.xic) {
					// Block events polling while changing input focus
					// because it triggers some event polling internally.
					MutexLock mutex_lock(events_mutex);
					XUnsetICFocus(wd.xic);
				}

				Input::get_singleton()->release_pressed_events();
				_send_window_event(wd, WINDOW_EVENT_FOCUS_OUT);

				if (mouse_mode_grab) {
					for (const KeyValue<WindowID, WindowData> &E : windows) {
						//dear X11, I try, I really try, but you never work, you do whathever you want.
						if (mouse_mode == MOUSE_MODE_CAPTURED) {
							// Show the cursor if we're in captured mode so it doesn't look weird.
							XUndefineCursor(x11_display, E.value.x11_window);
						}
					}
					XUngrabPointer(x11_display, CurrentTime);
				}
#ifdef TOUCH_ENABLED
				// Ungrab touch devices so input works as usual while we are unfocused
				/*for (int i = 0; i < xi.touch_devices.size(); ++i) {
					XIUngrabDevice(x11_display, xi.touch_devices[i], CurrentTime);
				}*/

				// Release every pointer to avoid sticky points
				for (const KeyValue<int, Vector2> &E : xi.state) {
					Ref<InputEventScreenTouch> st;
					st.instantiate();
					st->set_index(E.key);
					st->set_window_id(window_id);
					st->set_position(E.value);
					Input::get_singleton()->parse_input_event(st);
				}
				xi.state.clear();
#endif
			} break;

			case ConfigureNotify: {
				DEBUG_LOG_X11("[%u] ConfigureNotify window=%lu (%u), event=%lu, above=%lu, override_redirect=%u \n", frame, event.xconfigure.window, window_id, event.xconfigure.event, event.xconfigure.above, event.xconfigure.override_redirect);

				const WindowData &wd = windows[window_id];

				// Set focus when menu window is re-used.
				// RevertToPointerRoot is used to make sure we don't lose all focus in case
				// a subwindow and its parent are both destroyed.
				if (wd.menu_type && !wd.no_focus) {
					XSetInputFocus(x11_display, wd.x11_window, RevertToPointerRoot, CurrentTime);
				}

				_window_changed(&event);
			} break;

			case ButtonPress:
			case ButtonRelease: {
				/* exit in case of a mouse button press */
				last_timestamp = event.xbutton.time;
				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					event.xbutton.x = last_mouse_pos.x;
					event.xbutton.y = last_mouse_pos.y;
				}

				Ref<InputEventMouseButton> mb;
				mb.instantiate();

				mb->set_window_id(window_id);
				_get_key_modifier_state(event.xbutton.state, mb);
				mb->set_button_index((MouseButton)event.xbutton.button);
				if (mb->get_button_index() == MOUSE_BUTTON_RIGHT) {
					mb->set_button_index(MOUSE_BUTTON_MIDDLE);
				} else if (mb->get_button_index() == MOUSE_BUTTON_MIDDLE) {
					mb->set_button_index(MOUSE_BUTTON_RIGHT);
				}
				mb->set_button_mask(_get_mouse_button_state(mb->get_button_index(), event.xbutton.type));
				mb->set_position(Vector2(event.xbutton.x, event.xbutton.y));
				mb->set_global_position(mb->get_position());

				mb->set_pressed((event.type == ButtonPress));

				const WindowData &wd = windows[window_id];

				if (event.type == ButtonPress) {
					DEBUG_LOG_X11("[%u] ButtonPress window=%lu (%u), button_index=%u \n", frame, event.xbutton.window, window_id, mb->get_button_index());

					// Ensure window focus on click.
					// RevertToPointerRoot is used to make sure we don't lose all focus in case
					// a subwindow and its parent are both destroyed.
					if (!wd.no_focus) {
						XSetInputFocus(x11_display, wd.x11_window, RevertToPointerRoot, CurrentTime);
					}

					uint64_t diff = OS::get_singleton()->get_ticks_usec() / 1000 - last_click_ms;

					if (mb->get_button_index() == last_click_button_index) {
						if (diff < 400 && Vector2(last_click_pos).distance_to(Vector2(event.xbutton.x, event.xbutton.y)) < 5) {
							last_click_ms = 0;
							last_click_pos = Point2i(-100, -100);
							last_click_button_index = -1;
							mb->set_double_click(true);
						}

					} else if (mb->get_button_index() < 4 || mb->get_button_index() > 7) {
						last_click_button_index = mb->get_button_index();
					}

					if (!mb->is_double_click()) {
						last_click_ms += diff;
						last_click_pos = Point2i(event.xbutton.x, event.xbutton.y);
					}
				} else {
					DEBUG_LOG_X11("[%u] ButtonRelease window=%lu (%u), button_index=%u \n", frame, event.xbutton.window, window_id, mb->get_button_index());

					if (!wd.focused) {
						// Propagate the event to the focused window,
						// because it's received only on the topmost window.
						// Note: This is needed for drag & drop to work between windows,
						// because the engine expects events to keep being processed
						// on the same window dragging started.
						for (const KeyValue<WindowID, WindowData> &E : windows) {
							const WindowData &wd_other = E.value;
							WindowID window_id_other = E.key;
							if (wd_other.focused) {
								if (window_id_other != window_id) {
									int x, y;
									Window child;
									XTranslateCoordinates(x11_display, wd.x11_window, wd_other.x11_window, event.xbutton.x, event.xbutton.y, &x, &y, &child);

									mb->set_window_id(window_id_other);
									mb->set_position(Vector2(x, y));
									mb->set_global_position(mb->get_position());
									Input::get_singleton()->parse_input_event(mb);
								}
								break;
							}
						}
					}
				}

				Input::get_singleton()->parse_input_event(mb);

			} break;
			case MotionNotify: {
				// The X11 API requires filtering one-by-one through the motion
				// notify events, in order to figure out which event is the one
				// generated by warping the mouse pointer.

				while (true) {
					if (mouse_mode == MOUSE_MODE_CAPTURED && event.xmotion.x == windows[MAIN_WINDOW_ID].size.width / 2 && event.xmotion.y == windows[MAIN_WINDOW_ID].size.height / 2) {
						//this is likely the warp event since it was warped here
						center = Vector2(event.xmotion.x, event.xmotion.y);
						break;
					}

					if (event_index + 1 < events.size()) {
						const XEvent &next_event = events[event_index + 1];
						if (next_event.type == MotionNotify) {
							++event_index;
							event = next_event;
						} else {
							break;
						}
					} else {
						break;
					}
				}

				last_timestamp = event.xmotion.time;

				// Motion is also simple.
				// A little hack is in order
				// to be able to send relative motion events.
				Point2i pos(event.xmotion.x, event.xmotion.y);

				// Avoidance of spurious mouse motion (see handling of touch)
				bool filter = false;
				// Adding some tolerance to match better Point2i to Vector2
				if (xi.state.size() && Vector2(pos).distance_squared_to(xi.mouse_pos_to_filter) < 2) {
					filter = true;
				}
				// Invalidate to avoid filtering a possible legitimate similar event coming later
				xi.mouse_pos_to_filter = Vector2(1e10, 1e10);
				if (filter) {
					break;
				}

				const WindowData &wd = windows[window_id];
				bool focused = wd.focused;

				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					if (xi.relative_motion.x == 0 && xi.relative_motion.y == 0) {
						break;
					}

					Point2i new_center = pos;
					pos = last_mouse_pos + xi.relative_motion;
					center = new_center;
					do_mouse_warp = focused; // warp the cursor if we're focused in
				}

				if (!last_mouse_pos_valid) {
					last_mouse_pos = pos;
					last_mouse_pos_valid = true;
				}

				// Hackish but relative mouse motion is already handled in the RawMotion event.
				//  RawMotion does not provide the absolute mouse position (whereas MotionNotify does).
				//  Therefore, RawMotion cannot be the authority on absolute mouse position.
				//  RawMotion provides more precision than MotionNotify, which doesn't sense subpixel motion.
				//  Therefore, MotionNotify cannot be the authority on relative mouse motion.
				//  This means we need to take a combined approach...
				Point2i rel;

				// Only use raw input if in capture mode. Otherwise use the classic behavior.
				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					rel = xi.relative_motion;
				} else {
					rel = pos - last_mouse_pos;
				}

				// Reset to prevent lingering motion
				xi.relative_motion.x = 0;
				xi.relative_motion.y = 0;

				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					pos = Point2i(windows[MAIN_WINDOW_ID].size.width / 2, windows[MAIN_WINDOW_ID].size.height / 2);
				}

				Ref<InputEventMouseMotion> mm;
				mm.instantiate();

				mm->set_window_id(window_id);
				if (xi.pressure_supported) {
					mm->set_pressure(xi.pressure);
				} else {
					mm->set_pressure((mouse_get_button_state() & MOUSE_BUTTON_MASK_LEFT) ? 1.0f : 0.0f);
				}
				mm->set_tilt(xi.tilt);

				_get_key_modifier_state(event.xmotion.state, mm);
				mm->set_button_mask(mouse_get_button_state());
				mm->set_position(pos);
				mm->set_global_position(pos);
				Input::get_singleton()->set_mouse_position(pos);
				mm->set_speed(Input::get_singleton()->get_last_mouse_speed());

				mm->set_relative(rel);

				last_mouse_pos = pos;

				// printf("rel: %d,%d\n", rel.x, rel.y );
				// Don't propagate the motion event unless we have focus
				// this is so that the relative motion doesn't get messed up
				// after we regain focus.
				if (focused) {
					Input::get_singleton()->parse_input_event(mm);
				} else {
					// Propagate the event to the focused window,
					// because it's received only on the topmost window.
					// Note: This is needed for drag & drop to work between windows,
					// because the engine expects events to keep being processed
					// on the same window dragging started.
					for (const KeyValue<WindowID, WindowData> &E : windows) {
						const WindowData &wd_other = E.value;
						if (wd_other.focused) {
							int x, y;
							Window child;
							XTranslateCoordinates(x11_display, wd.x11_window, wd_other.x11_window, event.xmotion.x, event.xmotion.y, &x, &y, &child);

							Point2i pos_focused(x, y);

							mm->set_window_id(E.key);
							mm->set_position(pos_focused);
							mm->set_global_position(pos_focused);
							mm->set_speed(Input::get_singleton()->get_last_mouse_speed());
							Input::get_singleton()->parse_input_event(mm);

							break;
						}
					}
				}

			} break;
			case KeyPress:
			case KeyRelease: {
				last_timestamp = event.xkey.time;

				// key event is a little complex, so
				// it will be handled in its own function.
				_handle_key_event(window_id, (XKeyEvent *)&event, events, event_index);
			} break;

			case SelectionNotify:

				if (event.xselection.target == requested) {
					Property p = _read_property(x11_display, windows[window_id].x11_window, XInternAtom(x11_display, "PRIMARY", 0));

					Vector<String> files = String((char *)p.data).split("\n", false);
					for (int i = 0; i < files.size(); i++) {
						files.write[i] = files[i].replace("file://", "").uri_decode().strip_edges();
					}

					if (!windows[window_id].drop_files_callback.is_null()) {
						Variant v = files;
						Variant *vp = &v;
						Variant ret;
						Callable::CallError ce;
						windows[window_id].drop_files_callback.call((const Variant **)&vp, 1, ret, ce);
					}

					//Reply that all is well.
					XClientMessageEvent m;
					memset(&m, 0, sizeof(m));
					m.type = ClientMessage;
					m.display = x11_display;
					m.window = xdnd_source_window;
					m.message_type = xdnd_finished;
					m.format = 32;
					m.data.l[0] = windows[window_id].x11_window;
					m.data.l[1] = 1;
					m.data.l[2] = xdnd_action_copy; //We only ever copy.

					XSendEvent(x11_display, xdnd_source_window, False, NoEventMask, (XEvent *)&m);
				}
				break;

			case ClientMessage:

				if ((unsigned int)event.xclient.data.l[0] == (unsigned int)wm_delete) {
					_send_window_event(windows[window_id], WINDOW_EVENT_CLOSE_REQUEST);
				}

				else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_enter) {
					//File(s) have been dragged over the window, check for supported target (text/uri-list)
					xdnd_version = (event.xclient.data.l[1] >> 24);
					Window source = event.xclient.data.l[0];
					bool more_than_3 = event.xclient.data.l[1] & 1;
					if (more_than_3) {
						Property p = _read_property(x11_display, source, XInternAtom(x11_display, "XdndTypeList", False));
						requested = pick_target_from_list(x11_display, (Atom *)p.data, p.nitems);
					} else {
						requested = pick_target_from_atoms(x11_display, event.xclient.data.l[2], event.xclient.data.l[3], event.xclient.data.l[4]);
					}
				} else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_position) {
					//xdnd position event, reply with an XDND status message
					//just depending on type of data for now
					XClientMessageEvent m;
					memset(&m, 0, sizeof(m));
					m.type = ClientMessage;
					m.display = event.xclient.display;
					m.window = event.xclient.data.l[0];
					m.message_type = xdnd_status;
					m.format = 32;
					m.data.l[0] = windows[window_id].x11_window;
					m.data.l[1] = (requested != None);
					m.data.l[2] = 0; //empty rectangle
					m.data.l[3] = 0;
					m.data.l[4] = xdnd_action_copy;

					XSendEvent(x11_display, event.xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
					XFlush(x11_display);
				} else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_drop) {
					if (requested != None) {
						xdnd_source_window = event.xclient.data.l[0];
						if (xdnd_version >= 1) {
							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), windows[window_id].x11_window, event.xclient.data.l[2]);
						} else {
							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), windows[window_id].x11_window, CurrentTime);
						}
					} else {
						//Reply that we're not interested.
						XClientMessageEvent m;
						memset(&m, 0, sizeof(m));
						m.type = ClientMessage;
						m.display = event.xclient.display;
						m.window = event.xclient.data.l[0];
						m.message_type = xdnd_finished;
						m.format = 32;
						m.data.l[0] = windows[window_id].x11_window;
						m.data.l[1] = 0;
						m.data.l[2] = None; //Failed.
						XSendEvent(x11_display, event.xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
					}
				}
				break;
			default:
				break;
		}
	}

	XFlush(x11_display);

	if (do_mouse_warp) {
		XWarpPointer(x11_display, None, windows[MAIN_WINDOW_ID].x11_window,
				0, 0, 0, 0, (int)windows[MAIN_WINDOW_ID].size.width / 2, (int)windows[MAIN_WINDOW_ID].size.height / 2);

		/*
		Window root, child;
		int root_x, root_y;
		int win_x, win_y;
		unsigned int mask;
		XQueryPointer( x11_display, x11_window, &root, &child, &root_x, &root_y, &win_x, &win_y, &mask );

		printf("Root: %d,%d\n", root_x, root_y);
		printf("Win: %d,%d\n", win_x, win_y);
		*/
	}

	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerX11::release_rendering_thread() {
}

void DisplayServerX11::make_rendering_thread() {
}

void DisplayServerX11::swap_buffers() {
}

void DisplayServerX11::_update_context(WindowData &wd) {
	XClassHint *classHint = XAllocClassHint();

	if (classHint) {
		CharString name_str;
		switch (context) {
			case CONTEXT_EDITOR:
				name_str = "Godot_Editor";
				break;
			case CONTEXT_PROJECTMAN:
				name_str = "Godot_ProjectList";
				break;
			case CONTEXT_ENGINE:
				name_str = "Godot_Engine";
				break;
		}

		CharString class_str;
		if (context == CONTEXT_ENGINE) {
			String config_name = GLOBAL_GET("application/config/name");
			if (config_name.length() == 0) {
				class_str = "Godot_Engine";
			} else {
				class_str = config_name.utf8();
			}
		} else {
			class_str = "Godot";
		}

		classHint->res_class = class_str.ptrw();
		classHint->res_name = name_str.ptrw();

		XSetClassHint(x11_display, wd.x11_window, classHint);
		XFree(classHint);
	}
}

void DisplayServerX11::set_context(Context p_context) {
	_THREAD_SAFE_METHOD_

	context = p_context;

	for (KeyValue<WindowID, WindowData> &E : windows) {
		_update_context(E.value);
	}
}

void DisplayServerX11::set_native_icon(const String &p_filename) {
	WARN_PRINT("Native icon not supported by this display server.");
}

bool g_set_icon_error = false;
int set_icon_errorhandler(Display *dpy, XErrorEvent *ev) {
	g_set_icon_error = true;
	return 0;
}

void DisplayServerX11::set_icon(const Ref<Image> &p_icon) {
	_THREAD_SAFE_METHOD_

	WindowData &wd = windows[MAIN_WINDOW_ID];

	int (*oldHandler)(Display *, XErrorEvent *) = XSetErrorHandler(&set_icon_errorhandler);

	Atom net_wm_icon = XInternAtom(x11_display, "_NET_WM_ICON", False);

	if (p_icon.is_valid()) {
		Ref<Image> img = p_icon->duplicate();
		img->convert(Image::FORMAT_RGBA8);

		while (true) {
			int w = img->get_width();
			int h = img->get_height();

			if (g_set_icon_error) {
				g_set_icon_error = false;

				WARN_PRINT("Icon too large, attempting to resize icon.");

				int new_width, new_height;
				if (w > h) {
					new_width = w / 2;
					new_height = h * new_width / w;
				} else {
					new_height = h / 2;
					new_width = w * new_height / h;
				}

				w = new_width;
				h = new_height;

				if (!w || !h) {
					WARN_PRINT("Unable to set icon.");
					break;
				}

				img->resize(w, h, Image::INTERPOLATE_CUBIC);
			}

			// We're using long to have wordsize (32Bit build -> 32 Bits, 64 Bit build -> 64 Bits
			Vector<long> pd;

			pd.resize(2 + w * h);

			pd.write[0] = w;
			pd.write[1] = h;

			const uint8_t *r = img->get_data().ptr();

			long *wr = &pd.write[2];
			uint8_t const *pr = r;

			for (int i = 0; i < w * h; i++) {
				long v = 0;
				//    A             R             G            B
				v |= pr[3] << 24 | pr[0] << 16 | pr[1] << 8 | pr[2];
				*wr++ = v;
				pr += 4;
			}

			if (net_wm_icon != None) {
				XChangeProperty(x11_display, wd.x11_window, net_wm_icon, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)pd.ptr(), pd.size());
			}

			if (!g_set_icon_error) {
				break;
			}
		}
	} else {
		XDeleteProperty(x11_display, wd.x11_window, net_wm_icon);
	}

	XFlush(x11_display);
	XSetErrorHandler(oldHandler);
}

void DisplayServerX11::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
#endif
}

DisplayServer::VSyncMode DisplayServerX11::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	return context_vulkan->get_vsync_mode(p_window);
#else
	return DisplayServer::VSYNC_ENABLED;
#endif
}

Vector<String> DisplayServerX11::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif
#ifdef OPENGL_ENABLED
	drivers.push_back("opengl");
#endif

	return drivers;
}

DisplayServer *DisplayServerX11::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerX11(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported Vulkan versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU, upgrade it.\n"
								   "If you have updated your graphics drivers recently, try rebooting.",
				"Unable to initialize Video driver");
	}
	return ds;
}

DisplayServerX11::WindowID DisplayServerX11::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	//Create window

	long visualMask = VisualScreenMask;
	int numberOfVisuals;
	XVisualInfo vInfoTemplate = {};
	vInfoTemplate.screen = DefaultScreen(x11_display);
	XVisualInfo *visualInfo = XGetVisualInfo(x11_display, visualMask, &vInfoTemplate, &numberOfVisuals);

	Colormap colormap = XCreateColormap(x11_display, RootWindow(x11_display, vInfoTemplate.screen), visualInfo->visual, AllocNone);

	XSetWindowAttributes windowAttributes = {};
	windowAttributes.colormap = colormap;
	windowAttributes.background_pixel = 0xFFFFFFFF;
	windowAttributes.border_pixel = 0;
	windowAttributes.event_mask = KeyPressMask | KeyReleaseMask | StructureNotifyMask | ExposureMask;

	unsigned long valuemask = CWBorderPixel | CWColormap | CWEventMask;

	WindowID id = window_id_counter++;
	WindowData &wd = windows[id];

	if ((id != MAIN_WINDOW_ID) && (p_flags & WINDOW_FLAG_BORDERLESS_BIT)) {
		wd.menu_type = true;
	}

	if (p_flags & WINDOW_FLAG_NO_FOCUS_BIT) {
		wd.menu_type = true;
		wd.no_focus = true;
	}

	// Setup for menu subwindows:
	// - override_redirect forces the WM not to interfere with the window, to avoid delays due to
	//   handling decorations and placement.
	//   On the other hand, focus changes need to be handled manually when this is set.
	// - save_under is a hint for the WM to keep the content of windows behind to avoid repaint.
	if (wd.menu_type) {
		windowAttributes.override_redirect = True;
		windowAttributes.save_under = True;
		valuemask |= CWOverrideRedirect | CWSaveUnder;
	}

	{
		wd.x11_window = XCreateWindow(x11_display, RootWindow(x11_display, visualInfo->screen), p_rect.position.x, p_rect.position.y, p_rect.size.width > 0 ? p_rect.size.width : 1, p_rect.size.height > 0 ? p_rect.size.height : 1, 0, visualInfo->depth, InputOutput, visualInfo->visual, valuemask, &windowAttributes);

		// Enable receiving notification when the window is initialized (MapNotify)
		// so the focus can be set at the right time.
		if (wd.menu_type && !wd.no_focus) {
			XSelectInput(x11_display, wd.x11_window, StructureNotifyMask);
		}

		//associate PID
		// make PID known to X11
		{
			const long pid = OS::get_singleton()->get_process_id();
			Atom net_wm_pid = XInternAtom(x11_display, "_NET_WM_PID", False);
			if (net_wm_pid != None) {
				XChangeProperty(x11_display, wd.x11_window, net_wm_pid, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)&pid, 1);
			}
		}

		long im_event_mask = 0;

		{
			XIEventMask all_event_mask;
			XSetWindowAttributes new_attr;

			new_attr.event_mask = KeyPressMask | KeyReleaseMask | ButtonPressMask |
					ButtonReleaseMask | EnterWindowMask |
					LeaveWindowMask | PointerMotionMask |
					Button1MotionMask |
					Button2MotionMask | Button3MotionMask |
					Button4MotionMask | Button5MotionMask |
					ButtonMotionMask | KeymapStateMask |
					ExposureMask | VisibilityChangeMask |
					StructureNotifyMask |
					SubstructureNotifyMask | SubstructureRedirectMask |
					FocusChangeMask | PropertyChangeMask |
					ColormapChangeMask | OwnerGrabButtonMask |
					im_event_mask;

			XChangeWindowAttributes(x11_display, wd.x11_window, CWEventMask, &new_attr);

			static unsigned char all_mask_data[XIMaskLen(XI_LASTEVENT)] = {};

			all_event_mask.deviceid = XIAllDevices;
			all_event_mask.mask_len = sizeof(all_mask_data);
			all_event_mask.mask = all_mask_data;

			XISetMask(all_event_mask.mask, XI_HierarchyChanged);

#ifdef TOUCH_ENABLED
			if (xi.touch_devices.size()) {
				XISetMask(all_event_mask.mask, XI_TouchBegin);
				XISetMask(all_event_mask.mask, XI_TouchUpdate);
				XISetMask(all_event_mask.mask, XI_TouchEnd);
				XISetMask(all_event_mask.mask, XI_TouchOwnership);
			}
#endif

			XISelectEvents(x11_display, wd.x11_window, &all_event_mask, 1);
		}

		/* set the titlebar name */
		XStoreName(x11_display, wd.x11_window, "Godot");
		XSetWMProtocols(x11_display, wd.x11_window, &wm_delete, 1);
		if (xdnd_aware != None) {
			XChangeProperty(x11_display, wd.x11_window, xdnd_aware, XA_ATOM, 32, PropModeReplace, (unsigned char *)&xdnd_version, 1);
		}

		if (xim && xim_style) {
			// Block events polling while changing input focus
			// because it triggers some event polling internally.
			MutexLock mutex_lock(events_mutex);

			wd.xic = XCreateIC(xim, XNInputStyle, xim_style, XNClientWindow, wd.x11_window, XNFocusWindow, wd.x11_window, (char *)nullptr);
			if (XGetICValues(wd.xic, XNFilterEvents, &im_event_mask, nullptr) != nullptr) {
				WARN_PRINT("XGetICValues couldn't obtain XNFilterEvents value");
				XDestroyIC(wd.xic);
				wd.xic = nullptr;
			}
			if (wd.xic) {
				XUnsetICFocus(wd.xic);
			} else {
				WARN_PRINT("XCreateIC couldn't create wd.xic");
			}
		} else {
			wd.xic = nullptr;
			WARN_PRINT("XCreateIC couldn't create wd.xic");
		}

		_update_context(wd);

		if (p_flags & WINDOW_FLAG_BORDERLESS_BIT) {
			Hints hints;
			Atom property;
			hints.flags = 2;
			hints.decorations = 0;
			property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
			if (property != None) {
				XChangeProperty(x11_display, wd.x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
			}
		}

		if (wd.menu_type) {
			// Set Utility type to disable fade animations.
			Atom type_atom = XInternAtom(x11_display, "_NET_WM_WINDOW_TYPE_UTILITY", False);
			Atom wt_atom = XInternAtom(x11_display, "_NET_WM_WINDOW_TYPE", False);
			if (wt_atom != None && type_atom != None) {
				XChangeProperty(x11_display, wd.x11_window, wt_atom, XA_ATOM, 32, PropModeReplace, (unsigned char *)&type_atom, 1);
			}
		} else {
			Atom type_atom = XInternAtom(x11_display, "_NET_WM_WINDOW_TYPE_NORMAL", False);
			Atom wt_atom = XInternAtom(x11_display, "_NET_WM_WINDOW_TYPE", False);

			if (wt_atom != None && type_atom != None) {
				XChangeProperty(x11_display, wd.x11_window, wt_atom, XA_ATOM, 32, PropModeReplace, (unsigned char *)&type_atom, 1);
			}
		}

		_update_size_hints(id);

#if defined(VULKAN_ENABLED)
		if (context_vulkan) {
			Error err = context_vulkan->window_create(id, p_vsync_mode, wd.x11_window, x11_display, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create a Vulkan window");
		}
#endif

		//set_class_hint(x11_display, wd.x11_window);
		XFlush(x11_display);

		XSync(x11_display, False);
		//XSetErrorHandler(oldHandler);

		XFree(visualInfo);
	}

	window_set_mode(p_mode, id);

	//sync size
	{
		XWindowAttributes xwa;

		XSync(x11_display, False);
		XGetWindowAttributes(x11_display, wd.x11_window, &xwa);

		wd.position.x = xwa.x;
		wd.position.y = xwa.y;
		wd.size.width = xwa.width;
		wd.size.height = xwa.height;
	}

	//set cursor
	if (cursors[current_cursor] != None) {
		XDefineCursor(x11_display, wd.x11_window, cursors[current_cursor]);
	}

	return id;
}

DisplayServerX11::DisplayServerX11(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;

	current_cursor = CURSOR_ARROW;
	mouse_mode = MOUSE_MODE_VISIBLE;

	for (int i = 0; i < CURSOR_MAX; i++) {
		cursors[i] = None;
		img[i] = nullptr;
	}

	xmbstring = nullptr;

	last_click_ms = 0;
	last_click_button_index = -1;
	last_click_pos = Point2i(-100, -100);

	last_timestamp = 0;
	last_mouse_pos_valid = false;
	last_keyrelease_time = 0;

	XInitThreads(); //always use threads

	/** XLIB INITIALIZATION **/
	x11_display = XOpenDisplay(nullptr);

	if (!x11_display) {
		ERR_PRINT("X11 Display is not available");
		r_error = ERR_UNAVAILABLE;
		return;
	}

	char *modifiers = nullptr;
	Bool xkb_dar = False;
	XAutoRepeatOn(x11_display);
	xkb_dar = XkbSetDetectableAutoRepeat(x11_display, True, nullptr);

	// Try to support IME if detectable auto-repeat is supported
	if (xkb_dar == True) {
#ifdef X_HAVE_UTF8_STRING
		// Xutf8LookupString will be used later instead of XmbLookupString before
		// the multibyte sequences can be converted to unicode string.
		modifiers = XSetLocaleModifiers("");
#endif
	}

	if (modifiers == nullptr) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			WARN_PRINT("IME is disabled");
		}
		XSetLocaleModifiers("@im=none");
		WARN_PRINT("Error setting locale modifiers");
	}

	const char *err;
	xrr_get_monitors = nullptr;
	xrr_free_monitors = nullptr;
	int xrandr_major = 0;
	int xrandr_minor = 0;
	int event_base, error_base;
	xrandr_ext_ok = XRRQueryExtension(x11_display, &event_base, &error_base);
	xrandr_handle = dlopen("libXrandr.so.2", RTLD_LAZY);
	if (!xrandr_handle) {
		err = dlerror();
		// For some arcane reason, NetBSD now ships libXrandr.so.3 while the rest of the world has libXrandr.so.2...
		// In case this happens for other X11 platforms in the future, let's give it a try too before failing.
		xrandr_handle = dlopen("libXrandr.so.3", RTLD_LAZY);
		if (!xrandr_handle) {
			fprintf(stderr, "could not load libXrandr.so.2, Error: %s\n", err);
		}
	} else {
		XRRQueryVersion(x11_display, &xrandr_major, &xrandr_minor);
		if (((xrandr_major << 8) | xrandr_minor) >= 0x0105) {
			xrr_get_monitors = (xrr_get_monitors_t)dlsym(xrandr_handle, "XRRGetMonitors");
			if (!xrr_get_monitors) {
				err = dlerror();
				fprintf(stderr, "could not find symbol XRRGetMonitors\nError: %s\n", err);
			} else {
				xrr_free_monitors = (xrr_free_monitors_t)dlsym(xrandr_handle, "XRRFreeMonitors");
				if (!xrr_free_monitors) {
					err = dlerror();
					fprintf(stderr, "could not find XRRFreeMonitors\nError: %s\n", err);
					xrr_get_monitors = nullptr;
				}
			}
		}
	}

	if (!_refresh_device_info()) {
		OS::get_singleton()->alert("Your system does not support XInput 2.\n"
								   "Please upgrade your distribution.",
				"Unable to initialize XInput");
		r_error = ERR_UNAVAILABLE;
		return;
	}

	xim = XOpenIM(x11_display, nullptr, nullptr, nullptr);

	if (xim == nullptr) {
		WARN_PRINT("XOpenIM failed");
		xim_style = 0L;
	} else {
		::XIMCallback im_destroy_callback;
		im_destroy_callback.client_data = (::XPointer)(this);
		im_destroy_callback.callback = (::XIMProc)(_xim_destroy_callback);
		if (XSetIMValues(xim, XNDestroyCallback, &im_destroy_callback,
					nullptr) != nullptr) {
			WARN_PRINT("Error setting XIM destroy callback");
		}

		::XIMStyles *xim_styles = nullptr;
		xim_style = 0L;
		char *imvalret = XGetIMValues(xim, XNQueryInputStyle, &xim_styles, nullptr);
		if (imvalret != nullptr || xim_styles == nullptr) {
			fprintf(stderr, "Input method doesn't support any styles\n");
		}

		if (xim_styles) {
			xim_style = 0L;
			for (int i = 0; i < xim_styles->count_styles; i++) {
				if (xim_styles->supported_styles[i] ==
						(XIMPreeditNothing | XIMStatusNothing)) {
					xim_style = xim_styles->supported_styles[i];
					break;
				}
			}

			XFree(xim_styles);
		}
		XFree(imvalret);
	}

	/* Atorm internment */
	wm_delete = XInternAtom(x11_display, "WM_DELETE_WINDOW", true);
	//Set Xdnd (drag & drop) support
	xdnd_aware = XInternAtom(x11_display, "XdndAware", False);
	xdnd_version = 5;
	xdnd_enter = XInternAtom(x11_display, "XdndEnter", False);
	xdnd_position = XInternAtom(x11_display, "XdndPosition", False);
	xdnd_status = XInternAtom(x11_display, "XdndStatus", False);
	xdnd_action_copy = XInternAtom(x11_display, "XdndActionCopy", False);
	xdnd_drop = XInternAtom(x11_display, "XdndDrop", False);
	xdnd_finished = XInternAtom(x11_display, "XdndFinished", False);
	xdnd_selection = XInternAtom(x11_display, "XdndSelection", False);

	//!!!!!!!!!!!!!!!!!!!!!!!!!!
	//TODO - do Vulkan and GLES2 support checks, driver selection and fallback
	rendering_driver = p_rendering_driver;

#ifndef _MSC_VER
#warning Forcing vulkan rendering driver because OpenGL not implemented yet
#endif
	rendering_driver = "vulkan";

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextX11);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan");
		}
	}
#endif
	// Init context and rendering device
#if defined(OPENGL_ENABLED)
	if (rendering_driver == "opengl_es") {
		if (getenv("DRI_PRIME") == nullptr) {
			int use_prime = -1;

			if (getenv("PRIMUS_DISPLAY") ||
					getenv("PRIMUS_libGLd") ||
					getenv("PRIMUS_libGLa") ||
					getenv("PRIMUS_libGL") ||
					getenv("PRIMUS_LOAD_GLOBAL") ||
					getenv("BUMBLEBEE_SOCKET")) {
				print_verbose("Optirun/primusrun detected. Skipping GPU detection");
				use_prime = 0;
			}

			// Some tools use fake libGL libraries and have them override the real one using
			// LD_LIBRARY_PATH, so we skip them. *But* Steam also sets LD_LIBRARY_PATH for its
			// runtime and includes system `/lib` and `/lib64`... so ignore Steam.
			if (use_prime == -1 && getenv("LD_LIBRARY_PATH") && !getenv("STEAM_RUNTIME_LIBRARY_PATH")) {
				String ld_library_path(getenv("LD_LIBRARY_PATH"));
				Vector<String> libraries = ld_library_path.split(":");

				for (int i = 0; i < libraries.size(); ++i) {
					if (FileAccess::exists(libraries[i] + "/libGL.so.1") ||
							FileAccess::exists(libraries[i] + "/libGL.so")) {
						print_verbose("Custom libGL override detected. Skipping GPU detection");
						use_prime = 0;
					}
				}
			}

			if (use_prime == -1) {
				print_verbose("Detecting GPUs, set DRI_PRIME in the environment to override GPU detection logic.");
				use_prime = detect_prime();
			}

			if (use_prime) {
				print_line("Found discrete GPU, setting DRI_PRIME=1 to use it.");
				print_line("Note: Set DRI_PRIME=0 in the environment to disable Godot from using the discrete GPU.");
				setenv("DRI_PRIME", "1", 1);
			}
		}

		ContextGL_X11::ContextType opengl_api_type = ContextGL_X11::GLES_2_0_COMPATIBLE;

		context_gles2 = memnew(ContextGL_X11(x11_display, x11_window, current_videomode, opengl_api_type));

		if (context_gles2->initialize() != OK) {
			memdelete(context_gles2);
			context_gles2 = nullptr;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}

		context_gles2->set_use_vsync(current_videomode.use_vsync);

		if (RasterizerGLES2::is_viable() == OK) {
			RasterizerGLES2::register_config();
			RasterizerGLES2::make_current();
		} else {
			memdelete(context_gles2);
			context_gles2 = nullptr;
			ERR_FAIL_V(ERR_UNAVAILABLE);
		}
	}
#endif
	Point2i window_position(
			(screen_get_size(0).width - p_resolution.width) / 2,
			(screen_get_size(0).height - p_resolution.height) / 2);
	WindowID main_window = _create_window(p_mode, p_vsync_mode, p_flags, Rect2i(window_position, p_resolution));
	if (main_window == INVALID_WINDOW_ID) {
		r_error = ERR_CANT_CREATE;
		return;
	}
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, main_window);
		}
	}
	show_window(main_window);

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		//temporary
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

	{
		//set all event master mask
		XIEventMask all_master_event_mask;
		static unsigned char all_master_mask_data[XIMaskLen(XI_LASTEVENT)] = {};
		all_master_event_mask.deviceid = XIAllMasterDevices;
		all_master_event_mask.mask_len = sizeof(all_master_mask_data);
		all_master_event_mask.mask = all_master_mask_data;
		XISetMask(all_master_event_mask.mask, XI_DeviceChanged);
		XISetMask(all_master_event_mask.mask, XI_RawMotion);
		XISelectEvents(x11_display, DefaultRootWindow(x11_display), &all_master_event_mask, 1);
	}

	cursor_size = XcursorGetDefaultSize(x11_display);
	cursor_theme = XcursorGetTheme(x11_display);

	if (!cursor_theme) {
		print_verbose("XcursorGetTheme could not get cursor theme");
		cursor_theme = "default";
	}

	for (int i = 0; i < CURSOR_MAX; i++) {
		static const char *cursor_file[] = {
			"left_ptr",
			"xterm",
			"hand2",
			"cross",
			"watch",
			"left_ptr_watch",
			"fleur",
			"dnd-move",
			"crossed_circle",
			"v_double_arrow",
			"h_double_arrow",
			"size_bdiag",
			"size_fdiag",
			"move",
			"row_resize",
			"col_resize",
			"question_arrow"
		};

		img[i] = XcursorLibraryLoadImage(cursor_file[i], cursor_theme, cursor_size);
		if (!img[i]) {
			const char *fallback = nullptr;

			switch (i) {
				case CURSOR_POINTING_HAND:
					fallback = "pointer";
					break;
				case CURSOR_CROSS:
					fallback = "crosshair";
					break;
				case CURSOR_WAIT:
					fallback = "wait";
					break;
				case CURSOR_BUSY:
					fallback = "progress";
					break;
				case CURSOR_DRAG:
					fallback = "grabbing";
					break;
				case CURSOR_CAN_DROP:
					fallback = "hand1";
					break;
				case CURSOR_FORBIDDEN:
					fallback = "forbidden";
					break;
				case CURSOR_VSIZE:
					fallback = "ns-resize";
					break;
				case CURSOR_HSIZE:
					fallback = "ew-resize";
					break;
				case CURSOR_BDIAGSIZE:
					fallback = "fd_double_arrow";
					break;
				case CURSOR_FDIAGSIZE:
					fallback = "bd_double_arrow";
					break;
				case CURSOR_MOVE:
					img[i] = img[CURSOR_DRAG];
					break;
				case CURSOR_VSPLIT:
					fallback = "sb_v_double_arrow";
					break;
				case CURSOR_HSPLIT:
					fallback = "sb_h_double_arrow";
					break;
				case CURSOR_HELP:
					fallback = "help";
					break;
			}
			if (fallback != nullptr) {
				img[i] = XcursorLibraryLoadImage(fallback, cursor_theme, cursor_size);
			}
		}
		if (img[i]) {
			cursors[i] = XcursorImageLoadCursor(x11_display, img[i]);
		} else {
			print_verbose("Failed loading custom cursor: " + String(cursor_file[i]));
		}
	}

	{
		// Creating an empty/transparent cursor

		// Create 1x1 bitmap
		Pixmap cursormask = XCreatePixmap(x11_display,
				RootWindow(x11_display, DefaultScreen(x11_display)), 1, 1, 1);

		// Fill with zero
		XGCValues xgc;
		xgc.function = GXclear;
		GC gc = XCreateGC(x11_display, cursormask, GCFunction, &xgc);
		XFillRectangle(x11_display, cursormask, gc, 0, 0, 1, 1);

		// Color value doesn't matter. Mask zero means no foreground or background will be drawn
		XColor col = {};

		Cursor cursor = XCreatePixmapCursor(x11_display,
				cursormask, // source (using cursor mask as placeholder, since it'll all be ignored)
				cursormask, // mask
				&col, &col, 0, 0);

		XFreePixmap(x11_display, cursormask);
		XFreeGC(x11_display, gc);

		if (cursor == None) {
			ERR_PRINT("FAILED CREATING CURSOR");
		}

		null_cursor = cursor;
	}
	cursor_set_shape(CURSOR_BUSY);

	requested = None;

	/*if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}*/

	XEvent xevent;
	while (XPending(x11_display) > 0) {
		XNextEvent(x11_display, &xevent);
		if (xevent.type == ConfigureNotify) {
			_window_changed(&xevent);
		}
	}

	events_thread.start(_poll_events_thread, this);

	_update_real_mouse_position(windows[MAIN_WINDOW_ID]);

#ifdef DBUS_ENABLED
	screensaver = memnew(FreeDesktopScreenSaver);
	screen_set_keep_on(GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true));
#endif

	r_error = OK;
}

DisplayServerX11::~DisplayServerX11() {
	// Send owned clipboard data to clipboard manager before exit.
	Window x11_main_window = windows[MAIN_WINDOW_ID].x11_window;
	_clipboard_transfer_ownership(XA_PRIMARY, x11_main_window);
	_clipboard_transfer_ownership(XInternAtom(x11_display, "CLIPBOARD", 0), x11_main_window);

	events_thread_done.set();
	events_thread.wait_to_finish();

	//destroy all windows
	for (KeyValue<WindowID, WindowData> &E : windows) {
#ifdef VULKAN_ENABLED
		if (rendering_driver == "vulkan") {
			context_vulkan->window_destroy(E.key);
		}
#endif

		WindowData &wd = E.value;
		if (wd.xic) {
			XDestroyIC(wd.xic);
			wd.xic = nullptr;
		}
		XUnmapWindow(x11_display, wd.x11_window);
		XDestroyWindow(x11_display, wd.x11_window);
	}

	//destroy drivers
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		if (rendering_device_vulkan) {
			rendering_device_vulkan->finalize();
			memdelete(rendering_device_vulkan);
		}

		if (context_vulkan) {
			memdelete(context_vulkan);
		}
	}
#endif

	if (xrandr_handle) {
		dlclose(xrandr_handle);
	}

	for (int i = 0; i < CURSOR_MAX; i++) {
		if (cursors[i] != None) {
			XFreeCursor(x11_display, cursors[i]);
		}
		if (img[i] != nullptr) {
			XcursorImageDestroy(img[i]);
		}
	};

	if (xim) {
		XCloseIM(xim);
	}

	XCloseDisplay(x11_display);
	if (xmbstring) {
		memfree(xmbstring);
	}

#ifdef DBUS_ENABLED
	memdelete(screensaver);
#endif
}

void DisplayServerX11::register_x11_driver() {
	register_create_function("x11", create_func, get_rendering_drivers_func);
}

#endif // X11 enabled
