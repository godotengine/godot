/*************************************************************************/
/*  os_x11.cpp                                                           */
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

#include "os_x11.h"

#include "core/os/dir_access.h"
#include "core/print_string.h"
#include "detect_prime.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "key_mapping_x11.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"

#ifdef HAVE_MNTENT
#include <mntent.h>
#endif

#include <errno.h>
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

static const double abs_resolution_mult = 10000.0;
static const double abs_resolution_range_mult = 10.0;

void OS_X11::initialize_core() {

	crash_handler.initialize();

	OS_Unix::initialize_core();
}

int OS_X11::get_current_video_driver() const {
	return video_driver_index;
}

Error OS_X11::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	long im_event_mask = 0;
	last_button_state = 0;

	xmbstring = NULL;
	x11_window = 0;
	last_click_ms = 0;
	last_click_button_index = -1;
	last_click_pos = Point2(-100, -100);
	args = OS::get_singleton()->get_cmdline_args();
	current_videomode = p_desired;
	main_loop = NULL;
	last_timestamp = 0;
	last_mouse_pos_valid = false;
	last_keyrelease_time = 0;
	xdnd_version = 0;

	XInitThreads();

	/** XLIB INITIALIZATION **/
	x11_display = XOpenDisplay(NULL);

	if (!x11_display) {
		ERR_PRINT("X11 Display is not available");
		return ERR_UNAVAILABLE;
	}

	char *modifiers = NULL;
	Bool xkb_dar = False;
	XAutoRepeatOn(x11_display);
	xkb_dar = XkbSetDetectableAutoRepeat(x11_display, True, NULL);

	// Try to support IME if detectable auto-repeat is supported
	if (xkb_dar == True) {

#ifdef X_HAVE_UTF8_STRING
		// Xutf8LookupString will be used later instead of XmbLookupString before
		// the multibyte sequences can be converted to unicode string.
		modifiers = XSetLocaleModifiers("");
#endif
	}

	if (modifiers == NULL) {
		if (is_stdout_verbose()) {
			WARN_PRINT("IME is disabled");
		}
		XSetLocaleModifiers("@im=none");
		WARN_PRINT("Error setting locale modifiers");
	}

	const char *err;
	xrr_get_monitors = NULL;
	xrr_free_monitors = NULL;
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
					xrr_get_monitors = NULL;
				}
			}
		}
	}

	if (!refresh_device_info()) {
		OS::get_singleton()->alert("Your system does not support XInput 2.\n"
								   "Please upgrade your distribution.",
				"Unable to initialize XInput");
		return ERR_UNAVAILABLE;
	}

	xim = XOpenIM(x11_display, NULL, NULL, NULL);

	if (xim == NULL) {
		WARN_PRINT("XOpenIM failed");
		xim_style = 0L;
	} else {
		::XIMCallback im_destroy_callback;
		im_destroy_callback.client_data = (::XPointer)(this);
		im_destroy_callback.callback = (::XIMProc)(xim_destroy_callback);
		if (XSetIMValues(xim, XNDestroyCallback, &im_destroy_callback,
					NULL) != NULL) {
			WARN_PRINT("Error setting XIM destroy callback");
		}

		::XIMStyles *xim_styles = NULL;
		xim_style = 0L;
		char *imvalret = XGetIMValues(xim, XNQueryInputStyle, &xim_styles, NULL);
		if (imvalret != NULL || xim_styles == NULL) {
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

/*
	char* windowid = getenv("GODOT_WINDOWID");
	if (windowid) {

		//freopen("/home/punto/stdout", "w", stdout);
		//reopen("/home/punto/stderr", "w", stderr);
		x11_window = atol(windowid);

		XWindowAttributes xwa;
		XGetWindowAttributes(x11_display,x11_window,&xwa);

		current_videomode.width = xwa.width;
		current_videomode.height = xwa.height;
	};
	*/

// maybe contextgl wants to be in charge of creating the window
#if defined(OPENGL_ENABLED)
	if (getenv("DRI_PRIME") == NULL) {
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

		if (getenv("LD_LIBRARY_PATH")) {
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

	ContextGL_X11::ContextType opengl_api_type = ContextGL_X11::GLES_3_0_COMPATIBLE;

	if (p_video_driver == VIDEO_DRIVER_GLES2) {
		opengl_api_type = ContextGL_X11::GLES_2_0_COMPATIBLE;
	}

	bool editor = Engine::get_singleton()->is_editor_hint();
	bool gl_initialization_error = false;

	context_gl = NULL;
	while (!context_gl) {
		context_gl = memnew(ContextGL_X11(x11_display, x11_window, current_videomode, opengl_api_type));

		if (context_gl->initialize() != OK) {
			memdelete(context_gl);
			context_gl = NULL;

			if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2") || editor) {
				if (p_video_driver == VIDEO_DRIVER_GLES2) {
					gl_initialization_error = true;
					break;
				}

				p_video_driver = VIDEO_DRIVER_GLES2;
				opengl_api_type = ContextGL_X11::GLES_2_0_COMPATIBLE;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	while (true) {
		if (opengl_api_type == ContextGL_X11::GLES_3_0_COMPATIBLE) {
			if (RasterizerGLES3::is_viable() == OK) {
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2") || editor) {
					p_video_driver = VIDEO_DRIVER_GLES2;
					opengl_api_type = ContextGL_X11::GLES_2_0_COMPATIBLE;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		}

		if (opengl_api_type == ContextGL_X11::GLES_2_0_COMPATIBLE) {
			if (RasterizerGLES2::is_viable() == OK) {
				RasterizerGLES2::register_config();
				RasterizerGLES2::make_current();
				break;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported OpenGL versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU upgrade it.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;

	context_gl->set_use_vsync(current_videomode.use_vsync);

#endif

	visual_server = memnew(VisualServerRaster);
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	if (current_videomode.maximized) {
		current_videomode.maximized = false;
		set_window_maximized(true);
		// borderless fullscreen window mode
	} else if (current_videomode.fullscreen) {
		current_videomode.fullscreen = false;
		set_window_fullscreen(true);
	} else if (current_videomode.borderless_window) {
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = 0;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		if (property != None) {
			XChangeProperty(x11_display, x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		}
	}

	// make PID known to X11
	{
		const long pid = this->get_process_id();
		Atom net_wm_pid = XInternAtom(x11_display, "_NET_WM_PID", False);
		if (net_wm_pid != None) {
			XChangeProperty(x11_display, x11_window, net_wm_pid, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)&pid, 1);
		}
	}

	// disable resizable window
	if (!current_videomode.resizable && !current_videomode.fullscreen) {
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		xsh->flags = PMinSize | PMaxSize;
		XWindowAttributes xwa;
		if (current_videomode.fullscreen) {
			XGetWindowAttributes(x11_display, DefaultRootWindow(x11_display), &xwa);
		} else {
			XGetWindowAttributes(x11_display, x11_window, &xwa);
		}
		xsh->min_width = xwa.width;
		xsh->max_width = xwa.width;
		xsh->min_height = xwa.height;
		xsh->max_height = xwa.height;
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);
	}

	if (current_videomode.always_on_top) {
		current_videomode.always_on_top = false;
		set_window_always_on_top(true);
	}

	ERR_FAIL_COND_V(!visual_server, ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(x11_window == 0, ERR_UNAVAILABLE);

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

	XChangeWindowAttributes(x11_display, x11_window, CWEventMask, &new_attr);

	static unsigned char all_mask_data[XIMaskLen(XI_LASTEVENT)] = {};
	static unsigned char all_master_mask_data[XIMaskLen(XI_LASTEVENT)] = {};

	xi.all_event_mask.deviceid = XIAllDevices;
	xi.all_event_mask.mask_len = sizeof(all_mask_data);
	xi.all_event_mask.mask = all_mask_data;

	xi.all_master_event_mask.deviceid = XIAllMasterDevices;
	xi.all_master_event_mask.mask_len = sizeof(all_master_mask_data);
	xi.all_master_event_mask.mask = all_master_mask_data;

	XISetMask(xi.all_event_mask.mask, XI_HierarchyChanged);
	XISetMask(xi.all_master_event_mask.mask, XI_DeviceChanged);
	XISetMask(xi.all_master_event_mask.mask, XI_RawMotion);

#ifdef TOUCH_ENABLED
	if (xi.touch_devices.size()) {
		XISetMask(xi.all_event_mask.mask, XI_TouchBegin);
		XISetMask(xi.all_event_mask.mask, XI_TouchUpdate);
		XISetMask(xi.all_event_mask.mask, XI_TouchEnd);
		XISetMask(xi.all_event_mask.mask, XI_TouchOwnership);
	}
#endif

	XISelectEvents(x11_display, x11_window, &xi.all_event_mask, 1);
	XISelectEvents(x11_display, DefaultRootWindow(x11_display), &xi.all_master_event_mask, 1);

	// Disabled by now since grabbing also blocks mouse events
	// (they are received as extended events instead of standard events)
	/*XIClearMask(xi.touch_event_mask.mask, XI_TouchOwnership);

	// Grab touch devices to avoid OS gesture interference
	for (int i = 0; i < xi.touch_devices.size(); ++i) {
		XIGrabDevice(x11_display, xi.touch_devices[i], x11_window, CurrentTime, None, XIGrabModeAsync, XIGrabModeAsync, False, &xi.touch_event_mask);
	}*/

	/* set the titlebar name */
	XStoreName(x11_display, x11_window, "Godot");

	wm_delete = XInternAtom(x11_display, "WM_DELETE_WINDOW", true);
	XSetWMProtocols(x11_display, x11_window, &wm_delete, 1);

	im_active = false;
	im_position = Vector2();

	if (xim && xim_style) {
		xic = XCreateIC(xim, XNInputStyle, xim_style, XNClientWindow, x11_window, XNFocusWindow, x11_window, (char *)NULL);
		if (XGetICValues(xic, XNFilterEvents, &im_event_mask, NULL) != NULL) {
			WARN_PRINT("XGetICValues couldn't obtain XNFilterEvents value");
			XDestroyIC(xic);
			xic = NULL;
		}
		if (xic) {
			XUnsetICFocus(xic);
		} else {
			WARN_PRINT("XCreateIC couldn't create xic");
		}
	} else {

		xic = NULL;
		WARN_PRINT("XCreateIC couldn't create xic");
	}

	cursor_size = XcursorGetDefaultSize(x11_display);
	cursor_theme = XcursorGetTheme(x11_display);

	if (!cursor_theme) {
		print_verbose("XcursorGetTheme could not get cursor theme");
		cursor_theme = "default";
	}

	for (int i = 0; i < CURSOR_MAX; i++) {

		cursors[i] = None;
		img[i] = NULL;
	}

	current_cursor = CURSOR_ARROW;

	for (int i = 0; i < CURSOR_MAX; i++) {

		static const char *cursor_file[] = {
			"left_ptr",
			"xterm",
			"hand2",
			"cross",
			"watch",
			"left_ptr_watch",
			"fleur",
			"hand1",
			"X_cursor",
			"sb_v_double_arrow",
			"sb_h_double_arrow",
			"size_bdiag",
			"size_fdiag",
			"hand1",
			"sb_v_double_arrow",
			"sb_h_double_arrow",
			"question_arrow"
		};

		img[i] = XcursorLibraryLoadImage(cursor_file[i], cursor_theme, cursor_size);
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
	set_cursor_shape(CURSOR_BUSY);

	//Set Xdnd (drag & drop) support
	Atom XdndAware = XInternAtom(x11_display, "XdndAware", False);
	Atom version = 5;
	if (XdndAware != None) {
		XChangeProperty(x11_display, x11_window, XdndAware, XA_ATOM, 32, PropModeReplace, (unsigned char *)&version, 1);
	}

	xdnd_enter = XInternAtom(x11_display, "XdndEnter", False);
	xdnd_position = XInternAtom(x11_display, "XdndPosition", False);
	xdnd_status = XInternAtom(x11_display, "XdndStatus", False);
	xdnd_action_copy = XInternAtom(x11_display, "XdndActionCopy", False);
	xdnd_drop = XInternAtom(x11_display, "XdndDrop", False);
	xdnd_finished = XInternAtom(x11_display, "XdndFinished", False);
	xdnd_selection = XInternAtom(x11_display, "XdndSelection", False);
	requested = None;

	visual_server->init();

	AudioDriverManager::initialize(p_audio_driver);

	input = memnew(InputDefault);

	window_has_focus = true; // Set focus to true at init
#ifdef JOYDEV_ENABLED
	joypad = memnew(JoypadLinux(input));
#endif

	power_manager = memnew(PowerX11);

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

	XEvent xevent;
	while (XPending(x11_display) > 0) {
		XNextEvent(x11_display, &xevent);
		if (xevent.type == ConfigureNotify) {
			_window_changed(&xevent);
		}
	}

	events_thread.start(_poll_events_thread, this);

	update_real_mouse_position();

	return OK;
}

bool OS_X11::refresh_device_info() {
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
		if (!dev->enabled)
			continue;
		if (!(dev->use == XIMasterPointer || dev->use == XIFloatingSlave))
			continue;

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

void OS_X11::xim_destroy_callback(::XIM im, ::XPointer client_data,
		::XPointer call_data) {

	WARN_PRINT("Input method stopped");
	OS_X11 *os = reinterpret_cast<OS_X11 *>(client_data);
	os->xim = NULL;
	os->xic = NULL;
}

void OS_X11::set_ime_active(const bool p_active) {

	im_active = p_active;

	if (!xic) {
		return;
	}

	// Block events polling while changing input focus
	// because it triggers some event polling internally.
	if (p_active) {
		{
			MutexLock mutex_lock(events_mutex);
			XSetICFocus(xic);
		}
		set_ime_position(im_position);
	} else {
		MutexLock mutex_lock(events_mutex);
		XUnsetICFocus(xic);
	}
}

void OS_X11::set_ime_position(const Point2 &p_pos) {

	im_position = p_pos;

	if (!xic)
		return;

	::XPoint spot;
	spot.x = short(p_pos.x);
	spot.y = short(p_pos.y);
	XVaNestedList preedit_attr = XVaCreateNestedList(0, XNSpotLocation, &spot, NULL);

	{
		// Block events polling during this call
		// because it triggers some event polling internally.
		MutexLock mutex_lock(events_mutex);
		XSetICValues(xic, XNPreeditAttributes, preedit_attr, NULL);
	}

	XFree(preedit_attr);
}

String OS_X11::get_unique_id() const {

	static String machine_id;
	if (machine_id.empty()) {
		if (FileAccess *f = FileAccess::open("/etc/machine-id", FileAccess::READ)) {
			while (machine_id.empty() && !f->eof_reached()) {
				machine_id = f->get_line().strip_edges();
			}
			f->close();
			memdelete(f);
		}
	}
	return machine_id;
}

void OS_X11::finalize() {
	events_thread_done = true;
	events_thread.wait_to_finish();

	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;

	/*
	if (debugger_connection_console) {
		memdelete(debugger_connection_console);
	}
	*/
#ifdef ALSAMIDI_ENABLED
	driver_alsamidi.close();
#endif

#ifdef JOYDEV_ENABLED
	memdelete(joypad);
#endif

	xi.touch_devices.clear();
	xi.state.clear();

	memdelete(input);

	cursors_cache.clear();
	visual_server->finish();
	memdelete(visual_server);
	//memdelete(rasterizer);

	memdelete(power_manager);

	if (xrandr_handle)
		dlclose(xrandr_handle);

	if (!OS::get_singleton()->is_no_window_mode_enabled()) {
		XUnmapWindow(x11_display, x11_window);
	}
	XDestroyWindow(x11_display, x11_window);

#if defined(OPENGL_ENABLED)
	memdelete(context_gl);
#endif
	for (int i = 0; i < CURSOR_MAX; i++) {
		if (cursors[i] != None)
			XFreeCursor(x11_display, cursors[i]);
		if (img[i] != NULL)
			XcursorImageDestroy(img[i]);
	};

	if (xic) {
		XDestroyIC(xic);
	}
	if (xim) {
		XCloseIM(xim);
	}

	XCloseDisplay(x11_display);
	if (xmbstring)
		memfree(xmbstring);

	args.clear();
}

void OS_X11::set_mouse_mode(MouseMode p_mode) {

	if (p_mode == mouse_mode)
		return;

	if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED)
		XUngrabPointer(x11_display, CurrentTime);

	// The only modes that show a cursor are VISIBLE and CONFINED
	bool showCursor = (p_mode == MOUSE_MODE_VISIBLE || p_mode == MOUSE_MODE_CONFINED);

	if (showCursor) {
		XDefineCursor(x11_display, x11_window, cursors[current_cursor]); // show cursor
	} else {
		XDefineCursor(x11_display, x11_window, null_cursor); // hide cursor
	}

	mouse_mode = p_mode;

	if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED) {

		//flush pending motion events
		flush_mouse_motion();

		if (XGrabPointer(
					x11_display, x11_window, True,
					ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
					GrabModeAsync, GrabModeAsync, x11_window, None, CurrentTime) != GrabSuccess) {
			ERR_PRINT("NO GRAB");
		}

		if (mouse_mode == MOUSE_MODE_CAPTURED) {
			center.x = current_videomode.width / 2;
			center.y = current_videomode.height / 2;

			XWarpPointer(x11_display, None, x11_window,
					0, 0, 0, 0, (int)center.x, (int)center.y);

			input->set_mouse_position(center);
		}
	} else {
		do_mouse_warp = false;
	}

	XFlush(x11_display);
}

void OS_X11::warp_mouse_position(const Point2 &p_to) {

	if (mouse_mode == MOUSE_MODE_CAPTURED) {

		last_mouse_pos = p_to;
	} else {

		/*XWindowAttributes xwa;
		XGetWindowAttributes(x11_display, x11_window, &xwa);
		printf("%d %d\n", xwa.x, xwa.y); needed? */

		XWarpPointer(x11_display, None, x11_window,
				0, 0, 0, 0, (int)p_to.x, (int)p_to.y);
	}
}

void OS_X11::flush_mouse_motion() {
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

OS::MouseMode OS_X11::get_mouse_mode() const {
	return mouse_mode;
}

int OS_X11::get_mouse_button_state() const {
	return last_button_state;
}

Point2 OS_X11::get_mouse_position() const {
	return last_mouse_pos;
}

bool OS_X11::get_window_per_pixel_transparency_enabled() const {

	if (!is_layered_allowed()) return false;
	return layered_window;
}

void OS_X11::set_window_per_pixel_transparency_enabled(bool p_enabled) {

	if (!is_layered_allowed()) return;
	if (layered_window != p_enabled) {
		if (p_enabled) {
			layered_window = true;
		} else {
			layered_window = false;
		}
	}
}

void OS_X11::set_window_title(const String &p_title) {
	XStoreName(x11_display, x11_window, p_title.utf8().get_data());

	Atom _net_wm_name = XInternAtom(x11_display, "_NET_WM_NAME", false);
	Atom utf8_string = XInternAtom(x11_display, "UTF8_STRING", false);
	if (_net_wm_name != None && utf8_string != None) {
		XChangeProperty(x11_display, x11_window, _net_wm_name, utf8_string, 8, PropModeReplace, (unsigned char *)p_title.utf8().get_data(), p_title.utf8().length());
	}
}

void OS_X11::set_window_mouse_passthrough(const PoolVector2Array &p_region) {
	int event_base, error_base;
	const Bool ext_okay = XShapeQueryExtension(x11_display, &event_base, &error_base);
	if (ext_okay) {
		Region region;
		if (p_region.size() == 0) {
			region = XCreateRegion();
			XRectangle rect;
			rect.x = 0;
			rect.y = 0;
			rect.width = get_real_window_size().x;
			rect.height = get_real_window_size().y;
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
		XShapeCombineRegion(x11_display, x11_window, ShapeInput, 0, 0, region, ShapeSet);
		XDestroyRegion(region);
	}
}

void OS_X11::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_X11::get_video_mode(int p_screen) const {
	return current_videomode;
}

void OS_X11::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
}

void OS_X11::set_wm_fullscreen(bool p_enabled) {
	if (p_enabled && !get_borderless_window()) {
		// remove decorations if the window is not already borderless
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = 0;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		if (property != None) {
			XChangeProperty(x11_display, x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		}
	}

	if (p_enabled && !is_window_resizable()) {
		// Set the window as resizable to prevent window managers to ignore the fullscreen state flag.
		XSizeHints *xsh;

		xsh = XAllocSizeHints();
		xsh->flags = 0L;
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);
	}

	// Using EWMH -- Extended Window Manager Hints
	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_fullscreen = XInternAtom(x11_display, "_NET_WM_STATE_FULLSCREEN", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
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
		XChangeProperty(x11_display, x11_window, bypass_compositor, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)&compositing_disable_on, 1);
	}

	XFlush(x11_display);

	if (!p_enabled) {
		// Reset the non-resizable flags if we un-set these before.
		Size2 size = get_window_size();
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		if (!is_window_resizable()) {
			xsh->flags = PMinSize | PMaxSize;
			xsh->min_width = size.x;
			xsh->max_width = size.x;
			xsh->min_height = size.y;
			xsh->max_height = size.y;
		} else {
			xsh->flags = 0L;
			if (min_size != Size2()) {
				xsh->flags |= PMinSize;
				xsh->min_width = min_size.x;
				xsh->min_height = min_size.y;
			}
			if (max_size != Size2()) {
				xsh->flags |= PMaxSize;
				xsh->max_width = max_size.x;
				xsh->max_height = max_size.y;
			}
		}
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);

		// put back or remove decorations according to the last set borderless state
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = current_videomode.borderless_window ? 0 : 1;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		if (property != None) {
			XChangeProperty(x11_display, x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		}
	}
}

void OS_X11::set_wm_above(bool p_enabled) {
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_above = XInternAtom(x11_display, "_NET_WM_STATE_ABOVE", False);

	XClientMessageEvent xev;
	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.window = x11_window;
	xev.message_type = wm_state;
	xev.format = 32;
	xev.data.l[0] = p_enabled ? _NET_WM_STATE_ADD : _NET_WM_STATE_REMOVE;
	xev.data.l[1] = wm_above;
	xev.data.l[3] = 1;
	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, (XEvent *)&xev);
}

int OS_X11::get_screen_count() const {
	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) return 0;

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);
	XFree(xsi);
	return count;
}

int OS_X11::get_current_screen() const {
	int x, y;
	Window child;
	XTranslateCoordinates(x11_display, x11_window, DefaultRootWindow(x11_display), 0, 0, &x, &y, &child);

	int count = get_screen_count();
	for (int i = 0; i < count; i++) {
		Point2i pos = get_screen_position(i);
		Size2i size = get_screen_size(i);
		if ((x >= pos.x && x < pos.x + size.width) && (y >= pos.y && y < pos.y + size.height))
			return i;
	}
	return 0;
}

void OS_X11::set_current_screen(int p_screen) {
	int count = get_screen_count();
	if (p_screen >= count) return;

	if (current_videomode.fullscreen) {
		Point2i position = get_screen_position(p_screen);
		Size2i size = get_screen_size(p_screen);

		XMoveResizeWindow(x11_display, x11_window, position.x, position.y, size.x, size.y);
	} else {
		if (p_screen != get_current_screen()) {
			Point2i position = get_screen_position(p_screen);
			XMoveWindow(x11_display, x11_window, position.x, position.y);
		}
	}
}

Point2 OS_X11::get_screen_position(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}

	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) {
		return Point2i(0, 0);
	}

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);
	if (p_screen >= count) {
		return Point2i(0, 0);
	}

	Point2i position = Point2i(xsi[p_screen].x_org, xsi[p_screen].y_org);

	XFree(xsi);

	return position;
}

Size2 OS_X11::get_screen_size(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}

	// Using Xinerama Extension
	int event_base, error_base;
	const Bool ext_okay = XineramaQueryExtension(x11_display, &event_base, &error_base);
	if (!ext_okay) return Size2i(0, 0);

	int count;
	XineramaScreenInfo *xsi = XineramaQueryScreens(x11_display, &count);
	if (p_screen >= count) return Size2i(0, 0);

	Size2i size = Point2i(xsi[p_screen].width, xsi[p_screen].height);
	XFree(xsi);
	return size;
}

int OS_X11::get_screen_dpi(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}

	//invalid screen?
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), 0);

	//Get physical monitor Dimensions through XRandR and calculate dpi
	Size2 sc = get_screen_size(p_screen);
	if (xrandr_ext_ok) {
		int count = 0;
		if (xrr_get_monitors) {
			xrr_monitor_info *monitors = xrr_get_monitors(x11_display, x11_window, true, &count);
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
	if (xdpi || ydpi)
		return (xdpi + ydpi) / (xdpi && ydpi ? 2 : 1);

	//could not get dpi
	return 96;
}

Point2 OS_X11::get_window_position() const {
	int x, y;
	Window child;
	XTranslateCoordinates(x11_display, x11_window, DefaultRootWindow(x11_display), 0, 0, &x, &y, &child);
	return Point2i(x, y);
}

void OS_X11::set_window_position(const Point2 &p_position) {
	int x = 0;
	int y = 0;
	if (!get_borderless_window()) {
		//exclude window decorations
		XSync(x11_display, False);
		Atom prop = XInternAtom(x11_display, "_NET_FRAME_EXTENTS", True);
		if (prop != None) {
			Atom type;
			int format;
			unsigned long len;
			unsigned long remaining;
			unsigned char *data = NULL;
			if (XGetWindowProperty(x11_display, x11_window, prop, 0, 4, False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
				if (format == 32 && len == 4) {
					long *extents = (long *)data;
					x = extents[0];
					y = extents[2];
				}
				XFree(data);
			}
		}
	}
	XMoveWindow(x11_display, x11_window, p_position.x - x, p_position.y - y);
	update_real_mouse_position();
}

Size2 OS_X11::get_window_size() const {
	// Use current_videomode width and height instead of XGetWindowAttributes
	// since right after a XResizeWindow the attributes may not be updated yet
	return Size2i(current_videomode.width, current_videomode.height);
}

Size2 OS_X11::get_real_window_size() const {
	XWindowAttributes xwa;
	XSync(x11_display, False);
	XGetWindowAttributes(x11_display, x11_window, &xwa);
	int w = xwa.width;
	int h = xwa.height;
	Atom prop = XInternAtom(x11_display, "_NET_FRAME_EXTENTS", True);
	if (prop != None) {
		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = NULL;
		if (XGetWindowProperty(x11_display, x11_window, prop, 0, 4, False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
			if (format == 32 && len == 4) {
				long *extents = (long *)data;
				w += extents[0] + extents[1]; // left, right
				h += extents[2] + extents[3]; // top, bottom
			}
			XFree(data);
		}
	}
	return Size2(w, h);
}

Size2 OS_X11::get_max_window_size() const {
	return max_size;
}

Size2 OS_X11::get_min_window_size() const {
	return min_size;
}

void OS_X11::set_min_window_size(const Size2 p_size) {

	if ((p_size != Size2()) && (max_size != Size2()) && ((p_size.x > max_size.x) || (p_size.y > max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	min_size = p_size;

	if (is_window_resizable()) {
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		xsh->flags = 0L;
		if (min_size != Size2()) {
			xsh->flags |= PMinSize;
			xsh->min_width = min_size.x;
			xsh->min_height = min_size.y;
		}
		if (max_size != Size2()) {
			xsh->flags |= PMaxSize;
			xsh->max_width = max_size.x;
			xsh->max_height = max_size.y;
		}
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);

		XFlush(x11_display);
	}
}

void OS_X11::set_max_window_size(const Size2 p_size) {

	if ((p_size != Size2()) && ((p_size.x < min_size.x) || (p_size.y < min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	max_size = p_size;

	if (is_window_resizable()) {
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		xsh->flags = 0L;
		if (min_size != Size2()) {
			xsh->flags |= PMinSize;
			xsh->min_width = min_size.x;
			xsh->min_height = min_size.y;
		}
		if (max_size != Size2()) {
			xsh->flags |= PMaxSize;
			xsh->max_width = max_size.x;
			xsh->max_height = max_size.y;
		}
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);

		XFlush(x11_display);
	}
}

void OS_X11::set_window_size(const Size2 p_size) {

	if (current_videomode.width == p_size.width && current_videomode.height == p_size.height)
		return;

	XWindowAttributes xwa;
	XSync(x11_display, False);
	XGetWindowAttributes(x11_display, x11_window, &xwa);
	int old_w = xwa.width;
	int old_h = xwa.height;

	// If window resizable is disabled we need to update the attributes first
	XSizeHints *xsh;
	xsh = XAllocSizeHints();
	if (!is_window_resizable()) {
		xsh->flags = PMinSize | PMaxSize;
		xsh->min_width = p_size.x;
		xsh->max_width = p_size.x;
		xsh->min_height = p_size.y;
		xsh->max_height = p_size.y;
	} else {
		xsh->flags = 0L;
		if (min_size != Size2()) {
			xsh->flags |= PMinSize;
			xsh->min_width = min_size.x;
			xsh->min_height = min_size.y;
		}
		if (max_size != Size2()) {
			xsh->flags |= PMaxSize;
			xsh->max_width = max_size.x;
			xsh->max_height = max_size.y;
		}
	}
	XSetWMNormalHints(x11_display, x11_window, xsh);
	XFree(xsh);

	// Resize the window
	XResizeWindow(x11_display, x11_window, p_size.x, p_size.y);

	// Update our videomode width and height
	current_videomode.width = p_size.x;
	current_videomode.height = p_size.y;

	for (int timeout = 0; timeout < 50; ++timeout) {
		XSync(x11_display, False);
		XGetWindowAttributes(x11_display, x11_window, &xwa);

		if (old_w != xwa.width || old_h != xwa.height)
			break;

		usleep(10000);
	}
}

void OS_X11::set_window_fullscreen(bool p_enabled) {

	if (current_videomode.fullscreen == p_enabled)
		return;

	if (layered_window)
		set_window_per_pixel_transparency_enabled(false);

	if (p_enabled && current_videomode.always_on_top) {
		// Fullscreen + Always-on-top requires a maximized window on some window managers (Metacity)
		set_window_maximized(true);
	}
	set_wm_fullscreen(p_enabled);
	if (!p_enabled && current_videomode.always_on_top) {
		// Restore
		set_window_maximized(false);
	}
	if (!p_enabled) {
		set_window_position(last_position_before_fs);
	} else {
		last_position_before_fs = get_window_position();
	}
	current_videomode.fullscreen = p_enabled;
}

bool OS_X11::is_window_fullscreen() const {
	return current_videomode.fullscreen;
}

void OS_X11::set_window_resizable(bool p_enabled) {

	XSizeHints *xsh;
	xsh = XAllocSizeHints();
	if (!p_enabled) {
		Size2 size = get_window_size();

		xsh->flags = PMinSize | PMaxSize;
		xsh->min_width = size.x;
		xsh->max_width = size.x;
		xsh->min_height = size.y;
		xsh->max_height = size.y;
	} else {
		xsh->flags = 0L;
		if (min_size != Size2()) {
			xsh->flags |= PMinSize;
			xsh->min_width = min_size.x;
			xsh->min_height = min_size.y;
		}
		if (max_size != Size2()) {
			xsh->flags |= PMaxSize;
			xsh->max_width = max_size.x;
			xsh->max_height = max_size.y;
		}
	}

	XSetWMNormalHints(x11_display, x11_window, xsh);
	XFree(xsh);

	current_videomode.resizable = p_enabled;

	XFlush(x11_display);
}

bool OS_X11::is_window_resizable() const {
	return current_videomode.resizable;
}

void OS_X11::set_window_minimized(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}
	// Using ICCCM -- Inter-Client Communication Conventions Manual
	XEvent xev;
	Atom wm_change = XInternAtom(x11_display, "WM_CHANGE_STATE", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
	xev.xclient.message_type = wm_change;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = p_enabled ? WM_IconicState : WM_NormalState;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_hidden = XInternAtom(x11_display, "_NET_WM_STATE_HIDDEN", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = _NET_WM_STATE_ADD;
	xev.xclient.data.l[1] = wm_hidden;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
}

bool OS_X11::is_window_minimized() const {
	// Using ICCCM -- Inter-Client Communication Conventions Manual
	Atom property = XInternAtom(x11_display, "WM_STATE", True);
	if (property == None) {
		return false;
	}
	Atom type;
	int format;
	unsigned long len;
	unsigned long remaining;
	unsigned char *data = NULL;
	bool retval = false;

	int result = XGetWindowProperty(
			x11_display,
			x11_window,
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

	if (result == Success) {
		long *state = (long *)data;
		if (state[0] == WM_IconicState) {
			retval = true;
		}
		XFree(data);
	}

	return retval;
}

void OS_X11::set_window_maximized(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}
	if (is_window_maximized() == p_enabled)
		return;

	// Using EWMH -- Extended Window Manager Hints
	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_max_horz = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
	Atom wm_max_vert = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_VERT", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = p_enabled ? _NET_WM_STATE_ADD : _NET_WM_STATE_REMOVE;
	xev.xclient.data.l[1] = wm_max_horz;
	xev.xclient.data.l[2] = wm_max_vert;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);

	if (p_enabled && is_window_maximize_allowed()) {
		// Wait for effective resizing (so the GLX context is too).
		// Give up after 0.5s, it's not going to happen on this WM.
		// https://github.com/godotengine/godot/issues/19978
		for (int attempt = 0; !is_window_maximized() && attempt < 50; attempt++) {
			usleep(10000);
		}
	}

	maximized = p_enabled;
}

// Just a helper to reduce code duplication in `is_window_maximize_allowed`
// and `is_window_maximized`.
bool OS_X11::window_maximize_check(const char *p_atom_name) const {
	Atom property = XInternAtom(x11_display, p_atom_name, False);
	Atom type;
	int format;
	unsigned long len;
	unsigned long remaining;
	unsigned char *data = NULL;
	bool retval = false;

	if (property == None) {
		return false;
	}

	int result = XGetWindowProperty(
			x11_display,
			x11_window,
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

	if (result == Success) {
		Atom *atoms = (Atom *)data;
		Atom wm_act_max_horz = XInternAtom(x11_display, "_NET_WM_ACTION_MAXIMIZE_HORZ", False);
		Atom wm_act_max_vert = XInternAtom(x11_display, "_NET_WM_ACTION_MAXIMIZE_VERT", False);
		bool found_wm_act_max_horz = false;
		bool found_wm_act_max_vert = false;

		for (uint64_t i = 0; i < len; i++) {
			if (atoms[i] == wm_act_max_horz)
				found_wm_act_max_horz = true;
			if (atoms[i] == wm_act_max_vert)
				found_wm_act_max_vert = true;

			if (found_wm_act_max_horz || found_wm_act_max_vert) {
				retval = true;
				break;
			}
		}

		XFree(data);
	}

	return retval;
}

bool OS_X11::is_window_maximize_allowed() const {
	return window_maximize_check("_NET_WM_ALLOWED_ACTIONS");
}

bool OS_X11::is_window_maximized() const {
	// Using EWMH -- Extended Window Manager Hints
	return window_maximize_check("_NET_WM_STATE");
}

void OS_X11::set_window_always_on_top(bool p_enabled) {
	if (is_window_always_on_top() == p_enabled)
		return;

	if (p_enabled && current_videomode.fullscreen) {
		// Fullscreen + Always-on-top requires a maximized window on some window managers (Metacity)
		set_window_maximized(true);
	}
	set_wm_above(p_enabled);
	if (!p_enabled && !current_videomode.fullscreen) {
		// Restore
		set_window_maximized(false);
	}

	current_videomode.always_on_top = p_enabled;
}

bool OS_X11::is_window_always_on_top() const {
	return current_videomode.always_on_top;
}

bool OS_X11::is_window_focused() const {
	return window_focused;
}

void OS_X11::set_borderless_window(bool p_borderless) {

	if (get_borderless_window() == p_borderless)
		return;

	current_videomode.borderless_window = p_borderless;

	Hints hints;
	Atom property;
	hints.flags = 2;
	hints.decorations = current_videomode.borderless_window ? 0 : 1;
	property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
	if (property != None) {
		XChangeProperty(x11_display, x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
	}

	// Preserve window size
	set_window_size(Size2(current_videomode.width, current_videomode.height));
}

bool OS_X11::get_borderless_window() {

	bool borderless = current_videomode.borderless_window;
	Atom prop = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
	if (prop != None) {

		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = NULL;
		if (XGetWindowProperty(x11_display, x11_window, prop, 0, sizeof(Hints), False, AnyPropertyType, &type, &format, &len, &remaining, &data) == Success) {
			if (data && (format == 32) && (len >= 5)) {
				borderless = !((Hints *)data)->decorations;
			}
			XFree(data);
		}
	}
	return borderless;
}

void OS_X11::request_attention() {
	// Using EWMH -- Extended Window Manager Hints
	//
	// Sets the _NET_WM_STATE_DEMANDS_ATTENTION atom for WM_STATE
	// Will be unset by the window manager after user react on the request for attention

	XEvent xev;
	Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom wm_attention = XInternAtom(x11_display, "_NET_WM_STATE_DEMANDS_ATTENTION", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
	xev.xclient.message_type = wm_state;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = _NET_WM_STATE_ADD;
	xev.xclient.data.l[1] = wm_attention;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
	XFlush(x11_display);
}

void *OS_X11::get_native_handle(int p_handle_type) {
	switch (p_handle_type) {
		case APPLICATION_HANDLE: return NULL; // Do we have a value to return here?
		case DISPLAY_HANDLE: return (void *)x11_display;
		case WINDOW_HANDLE: return (void *)x11_window;
		case WINDOW_VIEW: return NULL; // Do we have a value to return here?
		case OPENGL_CONTEXT: return context_gl->get_glx_context();
		default: return NULL;
	}
}

void OS_X11::get_key_modifier_state(unsigned int p_x11_state, Ref<InputEventWithModifiers> state) {

	state->set_shift((p_x11_state & ShiftMask));
	state->set_control((p_x11_state & ControlMask));
	state->set_alt((p_x11_state & Mod1Mask /*|| p_x11_state&Mod5Mask*/)); //altgr should not count as alt
	state->set_metakey((p_x11_state & Mod4Mask));
}

unsigned int OS_X11::get_mouse_button_state(unsigned int p_x11_button, int p_x11_type) {

	unsigned int mask = 1 << (p_x11_button - 1);

	if (p_x11_type == ButtonPress) {
		last_button_state |= mask;
	} else {
		last_button_state &= ~mask;
	}

	return last_button_state;
}

void OS_X11::_handle_key_event(XKeyEvent *p_event, LocalVector<XEvent> &p_events, uint32_t &p_event_index, bool p_echo) {

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
	// So.. then you have to chosse which of both results
	// you want to keep.
	// This is a real bizarreness and cpu waster.

	KeySym keysym_keycode = 0; // keysym used to find a keycode
	KeySym keysym_unicode = 0; // keysym used to find unicode

	// XLookupString returns keysyms usable as nice scancodes/
	char str[256 + 1];
	XKeyEvent xkeyevent_no_mod = *xkeyevent;
	xkeyevent_no_mod.state &= ~ShiftMask;
	xkeyevent_no_mod.state &= ~ControlMask;
	XLookupString(xkeyevent, str, 256, &keysym_unicode, NULL);
	XLookupString(&xkeyevent_no_mod, NULL, 0, &keysym_keycode, NULL);

	// Meanwhile, XLookupString returns keysyms useful for unicode.

	if (!xmbstring) {
		// keep a temporary buffer for the string
		xmbstring = (char *)memalloc(sizeof(char) * 8);
		xmblen = 8;
	}

	if (xkeyevent->type == KeyPress && xic) {

		Status status;
#ifdef X_HAVE_UTF8_STRING
		int utf8len = 8;
		char *utf8string = (char *)memalloc(sizeof(char) * utf8len);
		int utf8bytes = Xutf8LookupString(xic, xkeyevent, utf8string,
				utf8len - 1, &keysym_unicode, &status);
		if (status == XBufferOverflow) {
			utf8len = utf8bytes + 1;
			utf8string = (char *)memrealloc(utf8string, utf8len);
			utf8bytes = Xutf8LookupString(xic, xkeyevent, utf8string,
					utf8len - 1, &keysym_unicode, &status);
		}
		utf8string[utf8bytes] = '\0';

		if (status == XLookupChars) {
			bool keypress = xkeyevent->type == KeyPress;
			unsigned int keycode = KeyMappingX11::get_keycode(keysym_keycode);
			if (keycode >= 'a' && keycode <= 'z')
				keycode -= 'a' - 'A';

			String tmp;
			tmp.parse_utf8(utf8string, utf8bytes);
			for (int i = 0; i < tmp.length(); i++) {
				Ref<InputEventKey> k;
				k.instance();
				if (keycode == 0 && tmp[i] == 0) {
					continue;
				}

				get_key_modifier_state(xkeyevent->state, k);

				k->set_unicode(tmp[i]);

				k->set_pressed(keypress);

				k->set_scancode(keycode);

				k->set_echo(false);

				if (k->get_scancode() == KEY_BACKTAB) {
					//make it consistent across platforms.
					k->set_scancode(KEY_TAB);
					k->set_shift(true);
				}

				input->accumulate_input_event(k);
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

	/* Phase 2, obtain a pigui keycode from the keysym */

	// KeyMappingX11 just translated the X11 keysym to a PIGUI
	// keysym, so it works in all platforms the same.

	unsigned int keycode = KeyMappingX11::get_keycode(keysym_keycode);

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

	if (keycode == 0 && unicode == 0)
		return;

	/* Phase 5, determine modifier mask */

	// No problems here, except I had no way to
	// know Mod1 was ALT and Mod4 was META (applekey/winkey)
	// just tried Mods until i found them.

	//print_verbose("mod1: "+itos(xkeyevent->state&Mod1Mask)+" mod 5: "+itos(xkeyevent->state&Mod5Mask));

	Ref<InputEventKey> k;
	k.instance();

	get_key_modifier_state(xkeyevent->state, k);

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

			::Time tresh = ABSDIFF(peek_event.xkey.time, xkeyevent->time);
			if (peek_event.type == KeyPress && tresh < 5) {
				KeySym rk;
				XLookupString((XKeyEvent *)&peek_event, str, 256, &rk, NULL);
				if (rk == keysym_keycode) {
					// Consume to next event.
					++p_event_index;
					_handle_key_event((XKeyEvent *)&peek_event, p_events, p_event_index, true);
					return; //ignore current, echo next
				}
			}

			// use the time from peek_event so it always works
		}

		// save the time to check for echo when keypress happens
	}

	/* Phase 7, send event to Window */

	k->set_pressed(keypress);

	if (keycode >= 'a' && keycode <= 'z')
		keycode -= 'a' - 'A';

	k->set_scancode(keycode);
	k->set_unicode(unicode);
	k->set_echo(p_echo);

	if (k->get_scancode() == KEY_BACKTAB) {
		//make it consistent across platforms.
		k->set_scancode(KEY_TAB);
		k->set_shift(true);
	}

	//don't set mod state if modifier keys are released by themselves
	//else event.is_action() will not work correctly here
	if (!k->is_pressed()) {
		if (k->get_scancode() == KEY_SHIFT)
			k->set_shift(false);
		else if (k->get_scancode() == KEY_CONTROL)
			k->set_control(false);
		else if (k->get_scancode() == KEY_ALT)
			k->set_alt(false);
		else if (k->get_scancode() == KEY_META)
			k->set_metakey(false);
	}

	bool last_is_pressed = Input::get_singleton()->is_key_pressed(k->get_scancode());
	if (k->is_pressed()) {
		if (last_is_pressed) {
			k->set_echo(true);
		}
	}

	//printf("key: %x\n",k->get_scancode());
	input->accumulate_input_event(k);
}

Atom OS_X11::_process_selection_request_target(Atom p_target, Window p_requestor, Atom p_property) const {
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
				NULL,
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
		CharString clip = OS::get_clipboard().utf8();
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

void OS_X11::_handle_selection_request_event(XSelectionRequestEvent *p_event) const {
	XEvent respond;
	if (p_event->target == XInternAtom(x11_display, "MULTIPLE", 0)) {
		// Request for multiple target conversions at once.
		Atom atom_pair = XInternAtom(x11_display, "ATOM_PAIR", False);
		respond.xselection.property = None;

		Atom type;
		int format;
		unsigned long len;
		unsigned long remaining;
		unsigned char *data = NULL;
		if (XGetWindowProperty(x11_display, p_event->requestor, p_event->property, 0, LONG_MAX, False, atom_pair, &type, &format, &len, &remaining, &data) == Success) {
			if ((len >= 2) && data) {
				Atom *targets = (Atom *)data;
				for (uint64_t i = 0; i < len; i += 2) {
					Atom target = targets[i];
					Atom &property = targets[i + 1];
					property = _process_selection_request_target(target, p_event->requestor, property);
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
		respond.xselection.property = _process_selection_request_target(p_event->target, p_event->requestor, p_event->property);
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

struct Property {
	unsigned char *data;
	int format, nitems;
	Atom type;
};

static Property read_property(Display *p_display, Window p_window, Atom p_property) {

	Atom actual_type = None;
	int actual_format = 0;
	unsigned long nitems = 0;
	unsigned long bytes_after = 0;
	unsigned char *ret = 0;

	int read_bytes = 1024;

	//Keep trying to read the property until there are no
	//bytes unread.
	if (p_property != None) {
		do {
			if (ret != 0)
				XFree(ret);

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

		if (atom != None && String(XGetAtomName(p_display, atom)) == target_type)
			return atom;
	}
	return None;
}

static Atom pick_target_from_atoms(Display *p_disp, Atom p_t1, Atom p_t2, Atom p_t3) {

	static const char *target_type = "text/uri-list";
	if (p_t1 != None && String(XGetAtomName(p_disp, p_t1)) == target_type)
		return p_t1;

	if (p_t2 != None && String(XGetAtomName(p_disp, p_t2)) == target_type)
		return p_t2;

	if (p_t3 != None && String(XGetAtomName(p_disp, p_t3)) == target_type)
		return p_t3;

	return None;
}

void OS_X11::_window_changed(XEvent *event) {

	if (xic) {
		//  Not portable.
		set_ime_position(Point2(0, 1));
	}
	if ((event->xconfigure.width == current_videomode.width) &&
			(event->xconfigure.height == current_videomode.height))
		return;

	current_videomode.width = event->xconfigure.width;
	current_videomode.height = event->xconfigure.height;
}

void OS_X11::_poll_events_thread(void *ud) {
	OS_X11 *os = (OS_X11 *)ud;
	os->_poll_events();
}

Bool OS_X11::_predicate_all_events(Display *display, XEvent *event, XPointer arg) {
	// Just accept all events.
	return True;
}

bool OS_X11::_wait_for_events() const {
	int x11_fd = ConnectionNumber(x11_display);
	fd_set in_fds;

	XFlush(x11_display);

	FD_ZERO(&in_fds);
	FD_SET(x11_fd, &in_fds);

	struct timeval tv;
	tv.tv_usec = 0;
	tv.tv_sec = 1;

	// Wait for next event or timeout.
	int num_ready_fds = select(x11_fd + 1, &in_fds, NULL, NULL, &tv);

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

void OS_X11::_poll_events() {
	while (!events_thread_done) {
		_wait_for_events();

		// Process events from the queue.
		{
			MutexLock mutex_lock(events_mutex);

			// Non-blocking wait for next event and remove it from the queue.
			XEvent ev;
			while (XCheckIfEvent(x11_display, &ev, _predicate_all_events, NULL)) {
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

				polled_events.push_back(ev);
			}
		}
	}
}

void OS_X11::process_xevents() {

	//printf("checking events %i\n", XPending(x11_display));

	do_mouse_warp = false;

	// Is the current mouse mode one where it needs to be grabbed.
	bool mouse_mode_grab = mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED;

	xi.pressure = 0;
	xi.tilt = Vector2();
	xi.pressure_supported = false;

	LocalVector<XEvent> events;
	{
		// Block events polling while flushing events.
		MutexLock mutex_lock(events_mutex);
		events = polled_events;
		polled_events.clear();
	}

	for (uint32_t event_index = 0; event_index < events.size(); ++event_index) {
		XEvent &event = events[event_index];

		if (XGetEventData(x11_display, &event.xcookie)) {

			if (event.xcookie.type == GenericEvent && event.xcookie.extension == xi.opcode) {

				XIDeviceEvent *event_data = (XIDeviceEvent *)event.xcookie.data;
				int index = event_data->detail;
				Vector2 pos = Vector2(event_data->event_x, event_data->event_y);

				switch (event_data->evtype) {
					case XI_HierarchyChanged:
					case XI_DeviceChanged: {
						refresh_device_info();
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
					case XI_TouchBegin: // Fall-through
							// Disabled hand-in-hand with the grabbing
							//XIAllowTouchEvents(x11_display, event_data->deviceid, event_data->detail, x11_window, XIAcceptTouch);

					case XI_TouchEnd: {

						bool is_begin = event_data->evtype == XI_TouchBegin;

						Ref<InputEventScreenTouch> st;
						st.instance();
						st->set_index(index);
						st->set_position(pos);
						st->set_pressed(is_begin);

						if (is_begin) {
							if (xi.state.has(index)) // Defensive
								break;
							xi.state[index] = pos;
							if (xi.state.size() == 1) {
								// X11 may send a motion event when a touch gesture begins, that would result
								// in a spurious mouse motion event being sent to Godot; remember it to be able to filter it out
								xi.mouse_pos_to_filter = pos;
							}
							input->accumulate_input_event(st);
						} else {
							if (!xi.state.has(index)) // Defensive
								break;
							xi.state.erase(index);
							input->accumulate_input_event(st);
						}
					} break;

					case XI_TouchUpdate: {

						Map<int, Vector2>::Element *curr_pos_elem = xi.state.find(index);
						if (!curr_pos_elem) { // Defensive
							break;
						}

						if (curr_pos_elem->value() != pos) {

							Ref<InputEventScreenDrag> sd;
							sd.instance();
							sd->set_index(index);
							sd->set_position(pos);
							sd->set_relative(pos - curr_pos_elem->value());
							input->accumulate_input_event(sd);

							curr_pos_elem->value() = pos;
						}
					} break;
#endif
				}
			}
		}
		XFreeEventData(x11_display, &event.xcookie);

		switch (event.type) {
			case Expose:
				Main::force_redraw();
				break;

			case NoExpose:
				minimized = true;
				break;

			case VisibilityNotify: {
				XVisibilityEvent *visibility = (XVisibilityEvent *)&event;
				minimized = (visibility->state == VisibilityFullyObscured);
			} break;
			case LeaveNotify: {
				if (main_loop && !mouse_mode_grab)
					main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_EXIT);

			} break;
			case EnterNotify: {
				if (main_loop && !mouse_mode_grab)
					main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_ENTER);
			} break;
			case FocusIn:
				minimized = false;
				window_has_focus = true;
				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
				window_focused = true;

				if (mouse_mode_grab) {
					// Show and update the cursor if confined and the window regained focus.
					if (mouse_mode == MOUSE_MODE_CONFINED)
						XUndefineCursor(x11_display, x11_window);
					else if (mouse_mode == MOUSE_MODE_CAPTURED) // or re-hide it in captured mode
						XDefineCursor(x11_display, x11_window, null_cursor);

					XGrabPointer(
							x11_display, x11_window, True,
							ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
							GrabModeAsync, GrabModeAsync, x11_window, None, CurrentTime);
				}
#ifdef TOUCH_ENABLED
				// Grab touch devices to avoid OS gesture interference
				/*for (int i = 0; i < xi.touch_devices.size(); ++i) {
					XIGrabDevice(x11_display, xi.touch_devices[i], x11_window, CurrentTime, None, XIGrabModeAsync, XIGrabModeAsync, False, &xi.touch_event_mask);
				}*/
#endif
				if (xic) {
					// Block events polling while changing input focus
					// because it triggers some event polling internally.
					MutexLock mutex_lock(events_mutex);
					XSetICFocus(xic);
				}
				break;

			case FocusOut:
				window_has_focus = false;
				input->release_pressed_events();
				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
				window_focused = false;

				if (mouse_mode_grab) {
					//dear X11, I try, I really try, but you never work, you do whathever you want.
					if (mouse_mode == MOUSE_MODE_CAPTURED) {
						// Show the cursor if we're in captured mode so it doesn't look weird.
						XUndefineCursor(x11_display, x11_window);
					}
					XUngrabPointer(x11_display, CurrentTime);
				}
#ifdef TOUCH_ENABLED
				// Ungrab touch devices so input works as usual while we are unfocused
				/*for (int i = 0; i < xi.touch_devices.size(); ++i) {
					XIUngrabDevice(x11_display, xi.touch_devices[i], CurrentTime);
				}*/

				// Release every pointer to avoid sticky points
				for (Map<int, Vector2>::Element *E = xi.state.front(); E; E = E->next()) {

					Ref<InputEventScreenTouch> st;
					st.instance();
					st->set_index(E->key());
					st->set_position(E->get());
					input->accumulate_input_event(st);
				}
				xi.state.clear();
#endif
				if (xic) {
					// Block events polling while changing input focus
					// because it triggers some event polling internally.
					MutexLock mutex_lock(events_mutex);
					XUnsetICFocus(xic);
				}
				break;

			case ConfigureNotify:
				_window_changed(&event);
				break;
			case ButtonPress:
			case ButtonRelease: {

				/* exit in case of a mouse button press */
				last_timestamp = event.xbutton.time;
				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					event.xbutton.x = last_mouse_pos.x;
					event.xbutton.y = last_mouse_pos.y;
				}

				Ref<InputEventMouseButton> mb;
				mb.instance();

				get_key_modifier_state(event.xbutton.state, mb);
				mb->set_button_index(event.xbutton.button);
				if (mb->get_button_index() == 2)
					mb->set_button_index(3);
				else if (mb->get_button_index() == 3)
					mb->set_button_index(2);
				mb->set_button_mask(get_mouse_button_state(mb->get_button_index(), event.xbutton.type));
				mb->set_position(Vector2(event.xbutton.x, event.xbutton.y));
				mb->set_global_position(mb->get_position());

				mb->set_pressed((event.type == ButtonPress));

				if (event.type == ButtonPress) {

					uint64_t diff = get_ticks_usec() / 1000 - last_click_ms;

					if (mb->get_button_index() == last_click_button_index) {

						if (diff < 400 && Point2(last_click_pos).distance_to(Point2(event.xbutton.x, event.xbutton.y)) < 5) {

							last_click_ms = 0;
							last_click_pos = Point2(-100, -100);
							last_click_button_index = -1;
							mb->set_doubleclick(true);
						}

					} else if (mb->get_button_index() < 4 || mb->get_button_index() > 7) {
						last_click_button_index = mb->get_button_index();
					}

					if (!mb->is_doubleclick()) {
						last_click_ms += diff;
						last_click_pos = Point2(event.xbutton.x, event.xbutton.y);
					}
				}

				input->accumulate_input_event(mb);

			} break;
			case MotionNotify: {

				// The X11 API requires filtering one-by-one through the motion
				// notify events, in order to figure out which event is the one
				// generated by warping the mouse pointer.

				while (true) {
					if (mouse_mode == MOUSE_MODE_CAPTURED && event.xmotion.x == current_videomode.width / 2 && event.xmotion.y == current_videomode.height / 2) {
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
				Point2 pos(event.xmotion.x, event.xmotion.y);

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

				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					if (xi.relative_motion.x == 0 && xi.relative_motion.y == 0) {
						break;
					}

					Point2i new_center = pos;
					pos = last_mouse_pos + xi.relative_motion;
					center = new_center;
					do_mouse_warp = window_has_focus; // warp the cursor if we're focused in
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
				Point2 rel;

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
					pos = Point2i(current_videomode.width / 2, current_videomode.height / 2);
				}

				Ref<InputEventMouseMotion> mm;
				mm.instance();

				if (xi.pressure_supported) {
					mm->set_pressure(xi.pressure);
				} else {
					mm->set_pressure((get_mouse_button_state() & (1 << (BUTTON_LEFT - 1))) ? 1.0f : 0.0f);
				}
				mm->set_tilt(xi.tilt);

				// Make the absolute position integral so it doesn't look _too_ weird :)
				Point2i posi(pos);

				get_key_modifier_state(event.xmotion.state, mm);
				mm->set_button_mask(get_mouse_button_state());
				mm->set_position(posi);
				mm->set_global_position(posi);
				input->set_mouse_position(posi);
				mm->set_speed(input->get_last_mouse_speed());

				mm->set_relative(rel);

				last_mouse_pos = pos;

				// printf("rel: %d,%d\n", rel.x, rel.y );
				// Don't propagate the motion event unless we have focus
				// this is so that the relative motion doesn't get messed up
				// after we regain focus.
				if (window_has_focus || !mouse_mode_grab)
					input->accumulate_input_event(mm);

			} break;
			case KeyPress:
			case KeyRelease: {

				last_timestamp = event.xkey.time;

				// key event is a little complex, so
				// it will be handled in its own function.
				_handle_key_event((XKeyEvent *)&event, events, event_index);
			} break;

			case SelectionNotify:

				if (event.xselection.target == requested) {

					Property p = read_property(x11_display, x11_window, XInternAtom(x11_display, "PRIMARY", 0));

					Vector<String> files = String((char *)p.data).split("\n", false);
					for (int i = 0; i < files.size(); i++) {
						files.write[i] = files[i].replace("file://", "").http_unescape().strip_edges();
					}
					main_loop->drop_files(files);

					//Reply that all is well.
					XClientMessageEvent m;
					memset(&m, 0, sizeof(m));
					m.type = ClientMessage;
					m.display = x11_display;
					m.window = xdnd_source_window;
					m.message_type = xdnd_finished;
					m.format = 32;
					m.data.l[0] = x11_window;
					m.data.l[1] = 1;
					m.data.l[2] = xdnd_action_copy; //We only ever copy.

					XSendEvent(x11_display, xdnd_source_window, False, NoEventMask, (XEvent *)&m);
				}
				break;

			case ClientMessage:

				if ((unsigned int)event.xclient.data.l[0] == (unsigned int)wm_delete)
					main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);

				else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_enter) {

					//File(s) have been dragged over the window, check for supported target (text/uri-list)
					xdnd_version = (event.xclient.data.l[1] >> 24);
					Window source = event.xclient.data.l[0];
					bool more_than_3 = event.xclient.data.l[1] & 1;
					if (more_than_3) {
						Property p = read_property(x11_display, source, XInternAtom(x11_display, "XdndTypeList", False));
						requested = pick_target_from_list(x11_display, (Atom *)p.data, p.nitems);
					} else
						requested = pick_target_from_atoms(x11_display, event.xclient.data.l[2], event.xclient.data.l[3], event.xclient.data.l[4]);
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
					m.data.l[0] = x11_window;
					m.data.l[1] = (requested != None);
					m.data.l[2] = 0; //empty rectangle
					m.data.l[3] = 0;
					m.data.l[4] = xdnd_action_copy;

					XSendEvent(x11_display, event.xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
					XFlush(x11_display);
				} else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_drop) {

					if (requested != None) {
						xdnd_source_window = event.xclient.data.l[0];
						if (xdnd_version >= 1)
							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), x11_window, event.xclient.data.l[2]);
						else
							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), x11_window, CurrentTime);
					} else {
						//Reply that we're not interested.
						XClientMessageEvent m;
						memset(&m, 0, sizeof(m));
						m.type = ClientMessage;
						m.display = event.xclient.display;
						m.window = event.xclient.data.l[0];
						m.message_type = xdnd_finished;
						m.format = 32;
						m.data.l[0] = x11_window;
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

		XWarpPointer(x11_display, None, x11_window,
				0, 0, 0, 0, (int)current_videomode.width / 2, (int)current_videomode.height / 2);

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

	input->flush_accumulated_events();
}

MainLoop *OS_X11::get_main_loop() const {

	return main_loop;
}

void OS_X11::delete_main_loop() {
	// Send owned clipboard data to clipboard manager before exit.
	// This has to be done here because the clipboard data is cleared before finalize().
	_clipboard_transfer_ownership(XA_PRIMARY, x11_window);
	_clipboard_transfer_ownership(XInternAtom(x11_display, "CLIPBOARD", 0), x11_window);

	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;
}

void OS_X11::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

bool OS_X11::can_draw() const {

	return !minimized;
};

void OS_X11::set_clipboard(const String &p_text) {
	{
		// The clipboard content can be accessed while polling for events.
		MutexLock mutex_lock(events_mutex);
		OS::set_clipboard(p_text);
	}

	XSetSelectionOwner(x11_display, XA_PRIMARY, x11_window, CurrentTime);
	XSetSelectionOwner(x11_display, XInternAtom(x11_display, "CLIPBOARD", 0), x11_window, CurrentTime);
};

Bool OS_X11::_predicate_clipboard_selection(Display *display, XEvent *event, XPointer arg) {
	if (event->type == SelectionNotify && event->xselection.requestor == *(Window *)arg) {
		return True;
	} else {
		return False;
	}
}

Bool OS_X11::_predicate_clipboard_incr(Display *display, XEvent *event, XPointer arg) {
	if (event->type == PropertyNotify && event->xproperty.state == PropertyNewValue) {
		return True;
	} else {
		return False;
	}
}

String OS_X11::_get_clipboard_impl(Atom p_source, Window x11_window, Atom target) const {
	String ret;

	Window selection_owner = XGetSelectionOwner(x11_display, p_source);
	if (selection_owner == x11_window) {
		return OS::get_clipboard();
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
				while (XCheckIfEvent(x11_display, &ev, _predicate_clipboard_incr, NULL)) {
					result = XGetWindowProperty(x11_display, x11_window,
							selection, // selection type
							0, LONG_MAX, // offset - len
							True, // delete property to notify the owner
							AnyPropertyType, // flag
							&type, // return type
							&format, // return format
							&len, &bytes_left, // data length
							&data);

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

String OS_X11::_get_clipboard(Atom p_source, Window x11_window) const {
	String ret;
	Atom utf8_atom = XInternAtom(x11_display, "UTF8_STRING", True);
	if (utf8_atom != None) {
		ret = _get_clipboard_impl(p_source, x11_window, utf8_atom);
	}
	if (ret.empty()) {
		ret = _get_clipboard_impl(p_source, x11_window, XA_STRING);
	}
	return ret;
}

String OS_X11::get_clipboard() const {

	String ret;
	ret = _get_clipboard(XInternAtom(x11_display, "CLIPBOARD", 0), x11_window);

	if (ret.empty()) {
		ret = _get_clipboard(XA_PRIMARY, x11_window);
	};

	return ret;
}

Bool OS_X11::_predicate_clipboard_save_targets(Display *display, XEvent *event, XPointer arg) {
	if (event->xany.window == *(Window *)arg) {
		return (event->type == SelectionRequest) ||
			   (event->type == SelectionNotify);
	} else {
		return False;
	}
}

void OS_X11::_clipboard_transfer_ownership(Atom p_source, Window x11_window) const {
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

String OS_X11::get_name() const {

	return "X11";
}

Error OS_X11::shell_open(String p_uri) {
	Error ok;
	int err_code;
	List<String> args;
	args.push_back(p_uri);

	// Agnostic
	ok = execute("xdg-open", args, true, NULL, NULL, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	// GNOME
	args.push_front("open"); // The command is `gio open`, so we need to add it to args
	ok = execute("gio", args, true, NULL, NULL, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	args.pop_front();
	ok = execute("gvfs-open", args, true, NULL, NULL, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	} else if (err_code == 2) {
		return ERR_FILE_NOT_FOUND;
	}
	// KDE
	ok = execute("kde-open5", args, true, NULL, NULL, &err_code);
	if (ok == OK && !err_code) {
		return OK;
	}
	ok = execute("kde-open", args, true, NULL, NULL, &err_code);
	return !err_code ? ok : FAILED;
}

bool OS_X11::_check_internal_feature_support(const String &p_feature) {

	return p_feature == "pc";
}

String OS_X11::get_config_path() const {

	if (has_environment("XDG_CONFIG_HOME")) {
		return get_environment("XDG_CONFIG_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".config");
	} else {
		return ".";
	}
}

String OS_X11::get_data_path() const {

	if (has_environment("XDG_DATA_HOME")) {
		return get_environment("XDG_DATA_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".local/share");
	} else {
		return get_config_path();
	}
}

String OS_X11::get_cache_path() const {

	if (has_environment("XDG_CACHE_HOME")) {
		return get_environment("XDG_CACHE_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".cache");
	} else {
		return get_config_path();
	}
}

String OS_X11::get_system_dir(SystemDir p_dir) const {

	String xdgparam;

	switch (p_dir) {
		case SYSTEM_DIR_DESKTOP: {

			xdgparam = "DESKTOP";
		} break;
		case SYSTEM_DIR_DCIM: {

			xdgparam = "PICTURES";

		} break;
		case SYSTEM_DIR_DOCUMENTS: {

			xdgparam = "DOCUMENTS";

		} break;
		case SYSTEM_DIR_DOWNLOADS: {

			xdgparam = "DOWNLOAD";

		} break;
		case SYSTEM_DIR_MOVIES: {

			xdgparam = "VIDEOS";

		} break;
		case SYSTEM_DIR_MUSIC: {

			xdgparam = "MUSIC";

		} break;
		case SYSTEM_DIR_PICTURES: {

			xdgparam = "PICTURES";

		} break;
		case SYSTEM_DIR_RINGTONES: {

			xdgparam = "MUSIC";

		} break;
	}

	String pipe;
	List<String> arg;
	arg.push_back(xdgparam);
	Error err = const_cast<OS_X11 *>(this)->execute("xdg-user-dir", arg, true, NULL, &pipe);
	if (err != OK)
		return ".";
	return pipe.strip_edges();
}

void OS_X11::move_window_to_foreground() {

	XEvent xev;
	Atom net_active_window = XInternAtom(x11_display, "_NET_ACTIVE_WINDOW", False);

	memset(&xev, 0, sizeof(xev));
	xev.type = ClientMessage;
	xev.xclient.window = x11_window;
	xev.xclient.message_type = net_active_window;
	xev.xclient.format = 32;
	xev.xclient.data.l[0] = 1;
	xev.xclient.data.l[1] = CurrentTime;

	XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureRedirectMask | SubstructureNotifyMask, &xev);
	XFlush(x11_display);
}

void OS_X11::set_cursor_shape(CursorShape p_shape) {

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (p_shape == current_cursor) {
		return;
	}

	if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
		if (cursors[p_shape] != None) {
			XDefineCursor(x11_display, x11_window, cursors[p_shape]);
		} else if (cursors[CURSOR_ARROW] != None) {
			XDefineCursor(x11_display, x11_window, cursors[CURSOR_ARROW]);
		}
	}

	current_cursor = p_shape;
}

OS::CursorShape OS_X11::get_cursor_shape() const {

	return current_cursor;
}

void OS_X11::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {

	if (p_cursor.is_valid()) {

		Map<CursorShape, Vector<Variant> >::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				set_cursor_shape(p_shape);
				return;
			}

			cursors_cache.erase(p_shape);
		}

		Ref<Texture> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_data();
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

		image = texture->get_data();

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

		image->lock();

		for (XcursorPixel index = 0; index < image_size; index++) {
			int row_index = floor(index / texture_size.width) + atlas_rect.position.y;
			int column_index = (index % int(texture_size.width)) + atlas_rect.position.x;

			if (atlas_texture.is_valid()) {
				column_index = MIN(column_index, atlas_rect.size.width - 1);
				row_index = MIN(row_index, atlas_rect.size.height - 1);
			}

			*(cursor_image->pixels + index) = image->get_pixel(column_index, row_index).to_argb32();
		}

		image->unlock();

		ERR_FAIL_COND(cursor_image->pixels == NULL);

		// Save it for a further usage
		cursors[p_shape] = XcursorImageLoadCursor(x11_display, cursor_image);

		Vector<Variant> params;
		params.push_back(p_cursor);
		params.push_back(p_hotspot);
		cursors_cache.insert(p_shape, params);

		if (p_shape == current_cursor) {
			if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
				XDefineCursor(x11_display, x11_window, cursors[p_shape]);
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
		set_cursor_shape(c);

		cursors_cache.erase(p_shape);
	}
}

void OS_X11::release_rendering_thread() {

#if defined(OPENGL_ENABLED)
	context_gl->release_current();
#endif
}

void OS_X11::make_rendering_thread() {

#if defined(OPENGL_ENABLED)
	context_gl->make_current();
#endif
}

void OS_X11::swap_buffers() {

#if defined(OPENGL_ENABLED)
	context_gl->swap_buffers();
#endif
}

void OS_X11::alert(const String &p_alert, const String &p_title) {

	if (is_no_window_mode_enabled()) {
		print_line("ALERT: " + p_title + ": " + p_alert);
		return;
	}

	const char *message_programs[] = { "zenity", "kdialog", "Xdialog", "xmessage" };

	String path = get_environment("PATH");
	Vector<String> path_elems = path.split(":", false);
	String program;

	for (int i = 0; i < path_elems.size(); i++) {
		for (uint64_t k = 0; k < sizeof(message_programs) / sizeof(char *); k++) {
			String tested_path = path_elems[i].plus_file(message_programs[k]);

			if (FileAccess::exists(tested_path)) {
				program = tested_path;
				break;
			}
		}

		if (program.length())
			break;
	}

	List<String> args;

	if (program.ends_with("zenity")) {
		args.push_back("--error");
		args.push_back("--width");
		args.push_back("500");
		args.push_back("--title");
		args.push_back(p_title);
		args.push_back("--text");
		args.push_back(p_alert);
	}

	if (program.ends_with("kdialog")) {
		args.push_back("--error");
		args.push_back(p_alert);
		args.push_back("--title");
		args.push_back(p_title);
	}

	if (program.ends_with("Xdialog")) {
		args.push_back("--title");
		args.push_back(p_title);
		args.push_back("--msgbox");
		args.push_back(p_alert);
		args.push_back("0");
		args.push_back("0");
	}

	if (program.ends_with("xmessage")) {
		args.push_back("-center");
		args.push_back("-title");
		args.push_back(p_title);
		args.push_back(p_alert);
	}

	if (program.length()) {
		execute(program, args, true);
	} else {
		print_line(p_alert);
	}
}

bool g_set_icon_error = false;
int set_icon_errorhandler(Display *dpy, XErrorEvent *ev) {
	g_set_icon_error = true;
	return 0;
}

void OS_X11::set_icon(const Ref<Image> &p_icon) {
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

			PoolVector<uint8_t>::Read r = img->get_data().read();

			long *wr = &pd.write[2];
			uint8_t const *pr = r.ptr();

			for (int i = 0; i < w * h; i++) {
				long v = 0;
				//    A             R             G            B
				v |= pr[3] << 24 | pr[0] << 16 | pr[1] << 8 | pr[2];
				*wr++ = v;
				pr += 4;
			}

			if (net_wm_icon != None) {
				XChangeProperty(x11_display, x11_window, net_wm_icon, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)pd.ptr(), pd.size());
			}

			if (!g_set_icon_error)
				break;
		}
	} else {
		XDeleteProperty(x11_display, x11_window, net_wm_icon);
	}

	XFlush(x11_display);
	XSetErrorHandler(oldHandler);
}

void OS_X11::force_process_input() {
	process_xevents(); // get rid of pending events
#ifdef JOYDEV_ENABLED
	joypad->process_joypads();
#endif
}

void OS_X11::run() {

	force_quit = false;

	if (!main_loop)
		return;

	main_loop->init();

	//uint64_t last_ticks=get_ticks_usec();

	//int frames=0;
	//uint64_t frame=0;

	while (!force_quit) {

		process_xevents(); // get rid of pending events
#ifdef JOYDEV_ENABLED
		joypad->process_joypads();
#endif
		if (Main::iteration())
			break;
	};

	main_loop->finish();
}

bool OS_X11::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_X11::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

void OS_X11::_set_use_vsync(bool p_enable) {
#if defined(OPENGL_ENABLED)
	if (context_gl)
		context_gl->set_use_vsync(p_enable);
#endif
}
/*
bool OS_X11::is_vsync_enabled() const {

	if (context_gl)
		return context_gl->is_using_vsync();

	return true;
}
*/
void OS_X11::set_context(int p_context) {

	XClassHint *classHint = XAllocClassHint();

	if (classHint) {

		CharString name_str;
		switch (p_context) {
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
		if (p_context == CONTEXT_ENGINE) {
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

		XSetClassHint(x11_display, x11_window, classHint);
		XFree(classHint);
	}
}

OS::PowerState OS_X11::get_power_state() {
	return power_manager->get_power_state();
}

int OS_X11::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_X11::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

void OS_X11::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_X11::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}

static String get_mountpoint(const String &p_path) {
	struct stat s;
	if (stat(p_path.utf8().get_data(), &s)) {
		return "";
	}

#ifdef HAVE_MNTENT
	dev_t dev = s.st_dev;
	FILE *fd = setmntent("/proc/mounts", "r");
	if (!fd) {
		return "";
	}

	struct mntent mnt;
	char buf[1024];
	size_t buflen = 1024;
	while (getmntent_r(fd, &mnt, buf, buflen)) {
		if (!stat(mnt.mnt_dir, &s) && s.st_dev == dev) {
			endmntent(fd);
			return String(mnt.mnt_dir);
		}
	}

	endmntent(fd);
#endif
	return "";
}

Error OS_X11::move_to_trash(const String &p_path) {
	String trash_can = "";
	String mnt = get_mountpoint(p_path);

	// If there is a directory "[Mountpoint]/.Trash-[UID]/files", use it as the trash can.
	if (mnt != "") {
		String path(mnt + "/.Trash-" + itos(getuid()) + "/files");
		struct stat s;
		if (!stat(path.utf8().get_data(), &s)) {
			trash_can = path;
		}
	}

	// Otherwise, if ${XDG_DATA_HOME} is defined, use "${XDG_DATA_HOME}/Trash/files" as the trash can.
	if (trash_can == "") {
		char *dhome = getenv("XDG_DATA_HOME");
		if (dhome) {
			trash_can = String(dhome) + "/Trash/files";
		}
	}

	// Otherwise, if ${HOME} is defined, use "${HOME}/.local/share/Trash/files" as the trash can.
	if (trash_can == "") {
		char *home = getenv("HOME");
		if (home) {
			trash_can = String(home) + "/.local/share/Trash/files";
		}
	}

	// Issue an error if none of the previous locations is appropriate for the trash can.
	if (trash_can == "") {
		ERR_PRINTS("move_to_trash: Could not determine the trash can location");
		return FAILED;
	}

	// Create needed directories for decided trash can location.
	DirAccess *dir_access = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = dir_access->make_dir_recursive(trash_can);
	memdelete(dir_access);

	// Issue an error if trash can is not created proprely.
	if (err != OK) {
		ERR_PRINTS("move_to_trash: Could not create the trash can \"" + trash_can + "\"");
		return err;
	}

	// The trash can is successfully created, now move the given resource to it.
	// Do not use DirAccess:rename() because it can't move files across multiple mountpoints.
	List<String> mv_args;
	mv_args.push_back(p_path);
	mv_args.push_back(trash_can);
	int retval;
	err = execute("mv", mv_args, true, NULL, NULL, &retval);

	// Issue an error if "mv" failed to move the given resource to the trash can.
	if (err != OK || retval != 0) {
		ERR_PRINTS("move_to_trash: Could not move the resource \"" + p_path + "\" to the trash can \"" + trash_can + "\"");
		return FAILED;
	}

	return OK;
}

OS::LatinKeyboardVariant OS_X11::get_latin_keyboard_variant() const {

	XkbDescRec *xkbdesc = XkbAllocKeyboard();
	ERR_FAIL_COND_V(!xkbdesc, LATIN_KEYBOARD_QWERTY);

	XkbGetNames(x11_display, XkbSymbolsNameMask, xkbdesc);
	ERR_FAIL_COND_V(!xkbdesc->names, LATIN_KEYBOARD_QWERTY);
	ERR_FAIL_COND_V(!xkbdesc->names->symbols, LATIN_KEYBOARD_QWERTY);

	char *layout = XGetAtomName(x11_display, xkbdesc->names->symbols);
	ERR_FAIL_COND_V(!layout, LATIN_KEYBOARD_QWERTY);

	Vector<String> info = String(layout).split("+");
	ERR_FAIL_INDEX_V(1, info.size(), LATIN_KEYBOARD_QWERTY);

	if (info[1].find("colemak") != -1) {
		return LATIN_KEYBOARD_COLEMAK;
	} else if (info[1].find("qwertz") != -1) {
		return LATIN_KEYBOARD_QWERTZ;
	} else if (info[1].find("azerty") != -1) {
		return LATIN_KEYBOARD_AZERTY;
	} else if (info[1].find("qzerty") != -1) {
		return LATIN_KEYBOARD_QZERTY;
	} else if (info[1].find("dvorak") != -1) {
		return LATIN_KEYBOARD_DVORAK;
	} else if (info[1].find("neo") != -1) {
		return LATIN_KEYBOARD_NEO;
	}

	return LATIN_KEYBOARD_QWERTY;
}

int OS_X11::keyboard_get_layout_count() const {
	int _group_count = 0;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);

		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != NULL) {
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

int OS_X11::keyboard_get_current_layout() const {
	XkbStateRec state;
	XkbGetState(x11_display, XkbUseCoreKbd, &state);
	return state.group;
}

void OS_X11::keyboard_set_current_layout(int p_index) {
	ERR_FAIL_INDEX(p_index, keyboard_get_layout_count());
	XkbLockGroup(x11_display, XkbUseCoreKbd, p_index);
}

String OS_X11::keyboard_get_layout_language(int p_index) const {
	String ret;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);
		XkbGetNames(x11_display, XkbGroupNamesMask, kbd);

		int _group_count = 0;
		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != NULL) {
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

String OS_X11::keyboard_get_layout_name(int p_index) const {
	String ret;
	XkbDescRec *kbd = XkbAllocKeyboard();
	if (kbd) {
		kbd->dpy = x11_display;
		XkbGetControls(x11_display, XkbAllControlsMask, kbd);
		XkbGetNames(x11_display, XkbSymbolsNameMask, kbd);
		XkbGetNames(x11_display, XkbGroupNamesMask, kbd);

		int _group_count = 0;
		const Atom *groups = kbd->names->groups;
		if (kbd->ctrls != NULL) {
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

void OS_X11::update_real_mouse_position() {
	Window root_return, child_return;
	int root_x, root_y, win_x, win_y;
	unsigned int mask_return;

	Bool xquerypointer_result = XQueryPointer(x11_display, x11_window, &root_return, &child_return, &root_x, &root_y,
			&win_x, &win_y, &mask_return);

	if (xquerypointer_result) {
		if (win_x > 0 && win_y > 0 && win_x <= current_videomode.width && win_y <= current_videomode.height) {

			last_mouse_pos.x = win_x;
			last_mouse_pos.y = win_y;
			last_mouse_pos_valid = true;
			input->set_mouse_position(last_mouse_pos);
		}
	}
}

OS_X11::OS_X11() {

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

#ifdef ALSA_ENABLED
	AudioDriverManager::add_driver(&driver_alsa);
#endif

	xi.opcode = 0;
	xi.last_relative_time = 0;
	layered_window = false;
	minimized = false;
	window_focused = true;
	xim_style = 0L;
	mouse_mode = MOUSE_MODE_VISIBLE;
	last_position_before_fs = Vector2();
}
