/*************************************************************************/
/*  os_x11.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "drivers/gles3/rasterizer_gles3.h"
#include "errno.h"
#include "key_mapping_x11.h"
#include "print_string.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/visual/visual_server_raster.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "X11/Xutil.h"

#include "X11/Xatom.h"
#include "X11/extensions/Xinerama.h"
// ICCCM
#define WM_NormalState 1L // window normal state
#define WM_IconicState 3L // window minimized
// EWMH
#define _NET_WM_STATE_REMOVE 0L // remove/unset property
#define _NET_WM_STATE_ADD 1L // add/set property
#define _NET_WM_STATE_TOGGLE 2L // toggle property

#include "main/main.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//stupid linux.h
#ifdef KEY_TAB
#undef KEY_TAB
#endif

#include <X11/Xatom.h>

#undef CursorShape

int OS_X11::get_video_driver_count() const {
	return 1;
}

const char *OS_X11::get_video_driver_name(int p_driver) const {
	return "GLES3";
}

OS::VideoMode OS_X11::get_default_video_mode() const {
	return OS::VideoMode(1024, 600, false);
}

int OS_X11::get_audio_driver_count() const {
	return AudioDriverManager::get_driver_count();
}

const char *OS_X11::get_audio_driver_name(int p_driver) const {

	AudioDriver *driver = AudioDriverManager::get_driver(p_driver);
	ERR_FAIL_COND_V(!driver, "");
	return AudioDriverManager::get_driver(p_driver)->get_name();
}

void OS_X11::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	last_button_state = 0;

	xmbstring = NULL;
	x11_window = 0;
	last_click_ms = 0;
	args = OS::get_singleton()->get_cmdline_args();
	current_videomode = p_desired;
	main_loop = NULL;
	last_timestamp = 0;
	last_mouse_pos_valid = false;
	last_keyrelease_time = 0;
	xdnd_version = 0;

	if (get_render_thread_mode() == RENDER_SEPARATE_THREAD) {
		XInitThreads();
	}

	/** XLIB INITIALIZATION **/
	x11_display = XOpenDisplay(NULL);

	char *modifiers = XSetLocaleModifiers("@im=none");
	if (modifiers == NULL) {
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
		fprintf(stderr, "could not load libXrandr.so.2, Error: %s\n", err);
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

	xim = XOpenIM(x11_display, NULL, NULL, NULL);

	if (xim == NULL) {
		WARN_PRINT("XOpenIM failed");
		xim_style = 0L;
	} else {
		::XIMStyles *xim_styles = NULL;
		xim_style = 0L;
		char *imvalret = NULL;
		imvalret = XGetIMValues(xim, XNQueryInputStyle, &xim_styles, NULL);
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
//print_line("def videomode "+itos(current_videomode.width)+","+itos(current_videomode.height));
#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)

	context_gl = memnew(ContextGL_X11(x11_display, x11_window, current_videomode, true));
	context_gl->initialize();

	RasterizerGLES3::register_config();

	RasterizerGLES3::make_current();

#endif
	visual_server = memnew(VisualServerRaster);
#if 0
	if (get_render_thread_mode()!=RENDER_THREAD_UNSAFE) {

		visual_server =memnew(VisualServerWrapMT(visual_server,get_render_thread_mode()==RENDER_SEPARATE_THREAD));
	}
#endif
	// borderless fullscreen window mode
	if (current_videomode.fullscreen) {
		// needed for lxde/openbox, possibly others
		Hints hints;
		Atom property;
		hints.flags = 2;
		hints.decorations = 0;
		property = XInternAtom(x11_display, "_MOTIF_WM_HINTS", True);
		XChangeProperty(x11_display, x11_window, property, property, 32, PropModeReplace, (unsigned char *)&hints, 5);
		XMapRaised(x11_display, x11_window);
		XWindowAttributes xwa;
		XGetWindowAttributes(x11_display, DefaultRootWindow(x11_display), &xwa);
		XMoveResizeWindow(x11_display, x11_window, 0, 0, xwa.width, xwa.height);

		// code for netwm-compliants
		XEvent xev;
		Atom wm_state = XInternAtom(x11_display, "_NET_WM_STATE", False);
		Atom fullscreen = XInternAtom(x11_display, "_NET_WM_STATE_FULLSCREEN", False);

		memset(&xev, 0, sizeof(xev));
		xev.type = ClientMessage;
		xev.xclient.window = x11_window;
		xev.xclient.message_type = wm_state;
		xev.xclient.format = 32;
		xev.xclient.data.l[0] = 1;
		xev.xclient.data.l[1] = fullscreen;
		xev.xclient.data.l[2] = 0;

		XSendEvent(x11_display, DefaultRootWindow(x11_display), False, SubstructureNotifyMask, &xev);
	}

	// disable resizable window
	if (!current_videomode.resizable) {
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

	AudioDriverManager::get_driver(p_audio_driver)->set_singleton();

	audio_driver_index = p_audio_driver;
	if (AudioDriverManager::get_driver(p_audio_driver)->init() != OK) {

		bool success = false;
		audio_driver_index = -1;
		for (int i = 0; i < AudioDriverManager::get_driver_count(); i++) {
			if (i == p_audio_driver)
				continue;
			AudioDriverManager::get_driver(i)->set_singleton();
			if (AudioDriverManager::get_driver(i)->init() == OK) {
				success = true;
				print_line("Audio Driver Failed: " + String(AudioDriverManager::get_driver(p_audio_driver)->get_name()));
				print_line("Using alternate audio driver: " + String(AudioDriverManager::get_driver(i)->get_name()));
				audio_driver_index = i;
				break;
			}
		}
		if (!success) {
			ERR_PRINT("Initializing audio failed.");
		}
	}

	ERR_FAIL_COND(!visual_server);
	ERR_FAIL_COND(x11_window == 0);

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
						  ColormapChangeMask | OwnerGrabButtonMask;

	XChangeWindowAttributes(x11_display, x11_window, CWEventMask, &new_attr);

	XClassHint *classHint;

	/* set the titlebar name */
	XStoreName(x11_display, x11_window, "Godot");

	/* set the name and class hints for the window manager to use */
	classHint = XAllocClassHint();
	if (classHint) {
		classHint->res_name = (char *)"Godot_Engine";
		classHint->res_class = (char *)"Godot";
	}
	XSetClassHint(x11_display, x11_window, classHint);
	XFree(classHint);

	wm_delete = XInternAtom(x11_display, "WM_DELETE_WINDOW", true);
	XSetWMProtocols(x11_display, x11_window, &wm_delete, 1);

	if (xim && xim_style) {

		xic = XCreateIC(xim, XNInputStyle, xim_style, XNClientWindow, x11_window, XNFocusWindow, x11_window, (char *)NULL);
	} else {

		xic = NULL;
		WARN_PRINT("XCreateIC couldn't create xic");
	}

	cursor_size = XcursorGetDefaultSize(x11_display);
	cursor_theme = XcursorGetTheme(x11_display);

	if (!cursor_theme) {
		WARN_PRINT("Could not find cursor theme");
		cursor_theme = "default";
	}

	for (int i = 0; i < CURSOR_MAX; i++) {

		cursors[i] = None;
		img[i] = NULL;
	}

	current_cursor = CURSOR_ARROW;

	if (cursor_theme) {
		//print_line("cursor theme: "+String(cursor_theme));
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
				//print_line("found cursor: "+String(cursor_file[i])+" id "+itos(cursors[i]));
			} else {
				if (OS::is_stdout_verbose())
					print_line("failed cursor: " + String(cursor_file[i]));
			}
		}
	}

	{
		Pixmap cursormask;
		XGCValues xgc;
		GC gc;
		XColor col;
		Cursor cursor;

		cursormask = XCreatePixmap(x11_display, RootWindow(x11_display, DefaultScreen(x11_display)), 1, 1, 1);
		xgc.function = GXclear;
		gc = XCreateGC(x11_display, cursormask, GCFunction, &xgc);
		XFillRectangle(x11_display, cursormask, gc, 0, 0, 1, 1);
		col.pixel = 0;
		col.red = 0;
		col.flags = 4;
		cursor = XCreatePixmapCursor(x11_display,
				cursormask, cursormask,
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
	XChangeProperty(x11_display, x11_window, XdndAware, XA_ATOM, 32, PropModeReplace, (unsigned char *)&version, 1);

	xdnd_enter = XInternAtom(x11_display, "XdndEnter", False);
	xdnd_position = XInternAtom(x11_display, "XdndPosition", False);
	xdnd_status = XInternAtom(x11_display, "XdndStatus", False);
	xdnd_action_copy = XInternAtom(x11_display, "XdndActionCopy", False);
	xdnd_drop = XInternAtom(x11_display, "XdndDrop", False);
	xdnd_finished = XInternAtom(x11_display, "XdndFinished", False);
	xdnd_selection = XInternAtom(x11_display, "XdndSelection", False);
	requested = None;

	visual_server->init();
	//
	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	//physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server = Physics2DServerWrapMT::init_server<Physics2DServerSW>();
	physics_2d_server->init();

	input = memnew(InputDefault);

	window_has_focus = true; // Set focus to true at init
#ifdef JOYDEV_ENABLED
	joypad = memnew(JoypadLinux(input));
#endif
	_ensure_data_dir();
}

void OS_X11::finalize() {

	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;

	for (int i = 0; i < get_audio_driver_count(); i++) {
		AudioDriverManager::get_driver(i)->finish();
	}

/*
	if (debugger_connection_console) {
		memdelete(debugger_connection_console);
	}
	*/

#ifdef JOYDEV_ENABLED
	memdelete(joypad);
#endif
	memdelete(input);

	visual_server->finish();
	memdelete(visual_server);
	//memdelete(rasterizer);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);

	if (xrandr_handle)
		dlclose(xrandr_handle);

	XUnmapWindow(x11_display, x11_window);
	XDestroyWindow(x11_display, x11_window);

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	memdelete(context_gl);
#endif
	for (int i = 0; i < CURSOR_MAX; i++) {
		if (cursors[i] != None)
			XFreeCursor(x11_display, cursors[i]);
		if (img[i] != NULL)
			XcursorImageDestroy(img[i]);
	};

	XDestroyIC(xic);
	XCloseIM(xim);

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
		XUndefineCursor(x11_display, x11_window); // show cursor
	} else {
		XDefineCursor(x11_display, x11_window, null_cursor); // hide cursor
	}

	mouse_mode = p_mode;

	if (mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED) {

		while (true) {
			//flush pending motion events

			if (XPending(x11_display) > 0) {
				XEvent event;
				XPeekEvent(x11_display, &event);
				if (event.type == MotionNotify) {
					XNextEvent(x11_display, &event);
				} else {
					break;
				}
			} else {
				break;
			}
		}

		if (XGrabPointer(
					x11_display, x11_window, True,
					ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
					GrabModeAsync, GrabModeAsync, x11_window, None, CurrentTime) != GrabSuccess) {
			ERR_PRINT("NO GRAB");
		}

		center.x = current_videomode.width / 2;
		center.y = current_videomode.height / 2;
		XWarpPointer(x11_display, None, x11_window,
				0, 0, 0, 0, (int)center.x, (int)center.y);

		input->set_mouse_pos(center);
	} else {
		do_mouse_warp = false;
	}

	XFlush(x11_display);
}

void OS_X11::warp_mouse_pos(const Point2 &p_to) {

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

OS::MouseMode OS_X11::get_mouse_mode() const {
	return mouse_mode;
}

int OS_X11::get_mouse_button_state() const {
	return last_button_state;
}

Point2 OS_X11::get_mouse_pos() const {
	return last_mouse_pos;
}

void OS_X11::set_window_title(const String &p_title) {
	XStoreName(x11_display, x11_window, p_title.utf8().get_data());

	Atom _net_wm_name = XInternAtom(x11_display, "_NET_WM_NAME", false);
	Atom utf8_string = XInternAtom(x11_display, "UTF8_STRING", false);
	XChangeProperty(x11_display, x11_window, _net_wm_name, utf8_string, 8, PropModeReplace, (unsigned char *)p_title.utf8().get_data(), p_title.utf8().length());
}

void OS_X11::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_X11::get_video_mode(int p_screen) const {
	return current_videomode;
}

void OS_X11::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
}

void OS_X11::set_wm_fullscreen(bool p_enabled) {
	// Using EWMH -- Extened Window Manager Hints
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
	if (xdpi || xdpi)
		return (xdpi + ydpi) / (xdpi && ydpi ? 2 : 1);

	//could not get dpi
	return 96;
}

Point2 OS_X11::get_window_position() const {
	int x, y;
	Window child;
	XTranslateCoordinates(x11_display, x11_window, DefaultRootWindow(x11_display), 0, 0, &x, &y, &child);

	int screen = get_current_screen();
	Point2i screen_position = get_screen_position(screen);

	return Point2i(x - screen_position.x, y - screen_position.y);
}

void OS_X11::set_window_position(const Point2 &p_position) {
	XMoveWindow(x11_display, x11_window, p_position.x, p_position.y);
}

Size2 OS_X11::get_window_size() const {
	// Use current_videomode width and height instead of XGetWindowAttributes
	// since right after a XResizeWindow the attributes may not be updated yet
	return Size2i(current_videomode.width, current_videomode.height);
}

void OS_X11::set_window_size(const Size2 p_size) {
	// If window resizable is disabled we need to update the attributes first
	if (is_window_resizable() == false) {
		XSizeHints *xsh;
		xsh = XAllocSizeHints();
		xsh->flags = PMinSize | PMaxSize;
		xsh->min_width = p_size.x;
		xsh->max_width = p_size.x;
		xsh->min_height = p_size.y;
		xsh->max_height = p_size.y;
		XSetWMNormalHints(x11_display, x11_window, xsh);
		XFree(xsh);
	}

	// Resize the window
	XResizeWindow(x11_display, x11_window, p_size.x, p_size.y);

	// Update our videomode width and height
	current_videomode.width = p_size.x;
	current_videomode.height = p_size.y;
}

void OS_X11::set_window_fullscreen(bool p_enabled) {
	set_wm_fullscreen(p_enabled);
	current_videomode.fullscreen = p_enabled;
}

bool OS_X11::is_window_fullscreen() const {
	return current_videomode.fullscreen;
}

void OS_X11::set_window_resizable(bool p_enabled) {
	XSizeHints *xsh;
	Size2 size = get_window_size();

	xsh = XAllocSizeHints();
	xsh->flags = p_enabled ? 0L : PMinSize | PMaxSize;
	if (!p_enabled) {
		xsh->min_width = size.x;
		xsh->max_width = size.x;
		xsh->min_height = size.y;
		xsh->max_height = size.y;
	}
	XSetWMNormalHints(x11_display, x11_window, xsh);
	XFree(xsh);
	current_videomode.resizable = p_enabled;
}

bool OS_X11::is_window_resizable() const {
	return current_videomode.resizable;
}

void OS_X11::set_window_minimized(bool p_enabled) {
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
	Atom type;
	int format;
	unsigned long len;
	unsigned long remaining;
	unsigned char *data = NULL;

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
		if (state[0] == WM_IconicState)
			return true;
	}
	return false;
}

void OS_X11::set_window_maximized(bool p_enabled) {
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

	maximized = p_enabled;
}

bool OS_X11::is_window_maximized() const {
	// Using EWMH -- Extended Window Manager Hints
	Atom property = XInternAtom(x11_display, "_NET_WM_STATE", False);
	Atom type;
	int format;
	unsigned long len;
	unsigned long remaining;
	unsigned char *data = NULL;

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
		Atom wm_max_horz = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
		Atom wm_max_vert = XInternAtom(x11_display, "_NET_WM_STATE_MAXIMIZED_VERT", False);
		bool found_wm_max_horz = false;
		bool found_wm_max_vert = false;

		for (unsigned int i = 0; i < len; i++) {
			if (atoms[i] == wm_max_horz)
				found_wm_max_horz = true;
			if (atoms[i] == wm_max_vert)
				found_wm_max_vert = true;

			if (found_wm_max_horz && found_wm_max_vert)
				return true;
		}
		XFree(atoms);
	}

	return false;
}

void OS_X11::request_attention() {
	// Using EWMH -- Extended Window Manager Hints
	//
	// Sets the _NET_WM_STATE_DEMANDS_ATTENTION atom for WM_STATE
	// Will be unset by the window manager after user react on the request for attention
	//
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
}

InputModifierState OS_X11::get_key_modifier_state(unsigned int p_x11_state) {

	InputModifierState state;

	state.shift = (p_x11_state & ShiftMask);
	state.control = (p_x11_state & ControlMask);
	state.alt = (p_x11_state & Mod1Mask /*|| p_x11_state&Mod5Mask*/); //altgr should not count as alt
	state.meta = (p_x11_state & Mod4Mask);

	return state;
}

unsigned int OS_X11::get_mouse_button_state(unsigned int p_x11_state) {

	unsigned int state = 0;

	if (p_x11_state & Button1Mask) {

		state |= 1 << 0;
	}

	if (p_x11_state & Button3Mask) {

		state |= 1 << 1;
	}

	if (p_x11_state & Button2Mask) {

		state |= 1 << 2;
	}

	if (p_x11_state & Button4Mask) {

		state |= 1 << 3;
	}

	if (p_x11_state & Button5Mask) {

		state |= 1 << 4;
	}

	last_button_state = state;
	return state;
}

void OS_X11::handle_key_event(XKeyEvent *p_event, bool p_echo) {

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
	XLookupString(xkeyevent, str, 256, &keysym_keycode, NULL);

	// Meanwhile, XLookupString returns keysyms useful for unicode.

	if (!xmbstring) {
		// keep a temporary buffer for the string
		xmbstring = (char *)memalloc(sizeof(char) * 8);
		xmblen = 8;
	}

	if (xkeyevent->type == KeyPress && xic) {

		Status status;
		do {

			int mnbytes = XmbLookupString(xic, xkeyevent, xmbstring, xmblen - 1, &keysym_unicode, &status);
			xmbstring[mnbytes] = '\0';

			if (status == XBufferOverflow) {
				xmblen = mnbytes + 1;
				xmbstring = (char *)memrealloc(xmbstring, xmblen);
			}
		} while (status == XBufferOverflow);
	}

	/* Phase 2, obtain a pigui keycode from the keysym */

	// KeyMappingX11 just translated the X11 keysym to a PIGUI
	// keysym, so it works in all platforms the same.

	unsigned int keycode = KeyMappingX11::get_keycode(keysym_keycode);

	/* Phase 3, obtain an unicode character from the keysym */

	// KeyMappingX11 also translates keysym to unicode.
	// It does a binary search on a table to translate
	// most properly.
	//print_line("keysym_unicode: "+rtos(keysym_unicode));
	unsigned int unicode = keysym_unicode > 0 ? KeyMappingX11::get_unicode_from_keysym(keysym_unicode) : 0;

	/* Phase 4, determine if event must be filtered */

	// This seems to be a side-effect of using XIM.
	// XEventFilter looks like a core X11 function,
	// but it's actually just used to see if we must
	// ignore a deadkey, or events XIM determines
	// must not reach the actual gui.
	// Guess it was a design problem of the extension

	bool keypress = xkeyevent->type == KeyPress;

	if (xkeyevent->type == KeyPress && xic) {
		if (XFilterEvent((XEvent *)xkeyevent, x11_window))
			return;
	}

	if (keycode == 0 && unicode == 0)
		return;

	/* Phase 5, determine modifier mask */

	// No problems here, except I had no way to
	// know Mod1 was ALT and Mod4 was META (applekey/winkey)
	// just tried Mods until i found them.

	//print_line("mod1: "+itos(xkeyevent->state&Mod1Mask)+" mod 5: "+itos(xkeyevent->state&Mod5Mask));

	InputModifierState state = get_key_modifier_state(xkeyevent->state);

	/* Phase 6, determine echo character */

	// Echo characters in X11 are a keyrelease and a keypress
	// one after the other with the (almot) same timestamp.
	// To detect them, i use XPeekEvent and check that their
	// difference in time is below a treshold.

	if (xkeyevent->type != KeyPress) {

		// make sure there are events pending,
		// so this call won't block.
		if (XPending(x11_display) > 0) {
			XEvent peek_event;
			XPeekEvent(x11_display, &peek_event);

			// I'm using a treshold of 5 msecs,
			// since sometimes there seems to be a little
			// jitter. I'm still not convinced that all this approach
			// is correct, but the xorg developers are
			// not very helpful today.

			::Time tresh = ABS(peek_event.xkey.time - xkeyevent->time);
			if (peek_event.type == KeyPress && tresh < 5) {
				KeySym rk;
				XLookupString((XKeyEvent *)&peek_event, str, 256, &rk, NULL);
				if (rk == keysym_keycode) {
					XEvent event;
					XNextEvent(x11_display, &event); //erase next event
					handle_key_event((XKeyEvent *)&event, true);
					return; //ignore current, echo next
				}
			}

			// use the time from peek_event so it always works
		}

		// save the time to check for echo when keypress happens
	}

	/* Phase 7, send event to Window */

	InputEvent event;
	event.type = InputEvent::KEY;
	event.device = 0;
	event.key.mod = state;
	event.key.pressed = keypress;

	if (keycode >= 'a' && keycode <= 'z')
		keycode -= 'a' - 'A';

	event.key.scancode = keycode;
	event.key.unicode = unicode;
	event.key.echo = p_echo;

	if (event.key.scancode == KEY_BACKTAB) {
		//make it consistent across platforms.
		event.key.scancode = KEY_TAB;
		event.key.mod.shift = true;
	}

	//don't set mod state if modifier keys are released by themselves
	//else event.is_action() will not work correctly here
	if (!event.key.pressed) {
		if (event.key.scancode == KEY_SHIFT)
			event.key.mod.shift = false;
		else if (event.key.scancode == KEY_CONTROL)
			event.key.mod.control = false;
		else if (event.key.scancode == KEY_ALT)
			event.key.mod.alt = false;
		else if (event.key.scancode == KEY_META)
			event.key.mod.meta = false;
	}

	//printf("key: %x\n",event.key.scancode);
	input->parse_input_event(event);
}

struct Property {
	unsigned char *data;
	int format, nitems;
	Atom type;
};

static Property read_property(Display *p_display, Window p_window, Atom p_property) {

	Atom actual_type;
	int actual_format;
	unsigned long nitems;
	unsigned long bytes_after;
	unsigned char *ret = 0;

	int read_bytes = 1024;

	//Keep trying to read the property until there are no
	//bytes unread.
	do {
		if (ret != 0)
			XFree(ret);

		XGetWindowProperty(p_display, p_window, p_property, 0, read_bytes, False, AnyPropertyType,
				&actual_type, &actual_format, &nitems, &bytes_after,
				&ret);

		read_bytes *= 2;

	} while (bytes_after != 0);

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

void OS_X11::process_xevents() {

	//printf("checking events %i\n", XPending(x11_display));

	do_mouse_warp = false;

	// Is the current mouse mode one where it needs to be grabbed.
	bool mouse_mode_grab = mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED;

	while (XPending(x11_display) > 0) {
		XEvent event;
		XNextEvent(x11_display, &event);

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
				if (input)
					input->set_mouse_in_window(false);

			} break;
			case EnterNotify: {
				if (main_loop && !mouse_mode_grab)
					main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_ENTER);
				if (input)
					input->set_mouse_in_window(true);
			} break;
			case FocusIn:
				minimized = false;
				window_has_focus = true;
				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
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
				break;

			case FocusOut:
				window_has_focus = false;
				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
				if (mouse_mode_grab) {
					//dear X11, I try, I really try, but you never work, you do whathever you want.
					if (mouse_mode == MOUSE_MODE_CAPTURED) {
						// Show the cursor if we're in captured mode so it doesn't look weird.
						XUndefineCursor(x11_display, x11_window);
					}
					XUngrabPointer(x11_display, CurrentTime);
				}
				break;

			case ConfigureNotify:
				/* call resizeGLScene only if our window-size changed */

				if ((event.xconfigure.width == current_videomode.width) &&
						(event.xconfigure.height == current_videomode.height))
					break;

				current_videomode.width = event.xconfigure.width;
				current_videomode.height = event.xconfigure.height;
				break;
			case ButtonPress:
			case ButtonRelease: {

				/* exit in case of a mouse button press */
				last_timestamp = event.xbutton.time;
				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					event.xbutton.x = last_mouse_pos.x;
					event.xbutton.y = last_mouse_pos.y;
				}

				InputEvent mouse_event;
				mouse_event.type = InputEvent::MOUSE_BUTTON;
				mouse_event.device = 0;
				mouse_event.mouse_button.mod = get_key_modifier_state(event.xbutton.state);
				mouse_event.mouse_button.button_mask = get_mouse_button_state(event.xbutton.state);
				mouse_event.mouse_button.x = event.xbutton.x;
				mouse_event.mouse_button.y = event.xbutton.y;
				mouse_event.mouse_button.global_x = event.xbutton.x;
				mouse_event.mouse_button.global_y = event.xbutton.y;
				mouse_event.mouse_button.button_index = event.xbutton.button;
				if (mouse_event.mouse_button.button_index == 2)
					mouse_event.mouse_button.button_index = 3;
				else if (mouse_event.mouse_button.button_index == 3)
					mouse_event.mouse_button.button_index = 2;

				mouse_event.mouse_button.pressed = (event.type == ButtonPress);

				if (event.type == ButtonPress && event.xbutton.button == 1) {

					uint64_t diff = get_ticks_usec() / 1000 - last_click_ms;

					if (diff < 400 && Point2(last_click_pos).distance_to(Point2(event.xbutton.x, event.xbutton.y)) < 5) {

						last_click_ms = 0;
						last_click_pos = Point2(-100, -100);
						mouse_event.mouse_button.doubleclick = true;

					} else {
						last_click_ms += diff;
						last_click_pos = Point2(event.xbutton.x, event.xbutton.y);
					}
				}

				input->parse_input_event(mouse_event);

			} break;
			case MotionNotify: {

				// FUCK YOU X11 API YOU SERIOUSLY GROSS ME OUT
				// YOU ARE AS GROSS AS LOOKING AT A PUTRID PILE
				// OF POOP STICKING OUT OF A CLOGGED TOILET
				// HOW THE FUCK I AM SUPPOSED TO KNOW WHICH ONE
				// OF THE MOTION NOTIFY EVENTS IS THE ONE GENERATED
				// BY WARPING THE MOUSE POINTER?
				// YOU ARE FORCING ME TO FILTER ONE BY ONE TO FIND IT
				// PLEASE DO ME A FAVOR AND DIE DROWNED IN A FECAL
				// MOUNTAIN BECAUSE THAT'S WHERE YOU BELONG.

				while (true) {
					if (mouse_mode == MOUSE_MODE_CAPTURED && event.xmotion.x == current_videomode.width / 2 && event.xmotion.y == current_videomode.height / 2) {
						//this is likely the warp event since it was warped here
						center = Vector2(event.xmotion.x, event.xmotion.y);
						break;
					}

					if (XPending(x11_display) > 0) {
						XEvent tevent;
						XPeekEvent(x11_display, &tevent);
						if (tevent.type == MotionNotify) {
							XNextEvent(x11_display, &event);
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

				if (mouse_mode == MOUSE_MODE_CAPTURED) {
#if 1
					//Vector2 c = Point2i(current_videomode.width/2,current_videomode.height/2);
					if (pos == Point2i(current_videomode.width / 2, current_videomode.height / 2)) {
						//this sucks, it's a hack, etc and is a little inaccurate, etc.
						//but nothing I can do, X11 sucks.

						center = pos;
						break;
					}

					Point2i new_center = pos;
					pos = last_mouse_pos + (pos - center);
					center = new_center;
					do_mouse_warp = window_has_focus; // warp the cursor if we're focused in
#else
					//Dear X11, thanks for making my life miserable

					center.x = current_videomode.width / 2;
					center.y = current_videomode.height / 2;
					pos = last_mouse_pos + (pos - center);
					if (pos == last_mouse_pos)
						break;
					XWarpPointer(x11_display, None, x11_window,
							0, 0, 0, 0, (int)center.x, (int)center.y);
#endif
				}

				if (!last_mouse_pos_valid) {

					last_mouse_pos = pos;
					last_mouse_pos_valid = true;
				}

				Point2i rel = pos - last_mouse_pos;

				InputEvent motion_event;
				motion_event.type = InputEvent::MOUSE_MOTION;
				motion_event.device = 0;

				motion_event.mouse_motion.mod = get_key_modifier_state(event.xmotion.state);
				motion_event.mouse_motion.button_mask = get_mouse_button_state(event.xmotion.state);
				motion_event.mouse_motion.x = pos.x;
				motion_event.mouse_motion.y = pos.y;
				input->set_mouse_pos(pos);
				motion_event.mouse_motion.global_x = pos.x;
				motion_event.mouse_motion.global_y = pos.y;
				motion_event.mouse_motion.speed_x = input->get_last_mouse_speed().x;
				motion_event.mouse_motion.speed_y = input->get_last_mouse_speed().y;

				motion_event.mouse_motion.relative_x = rel.x;
				motion_event.mouse_motion.relative_y = rel.y;

				last_mouse_pos = pos;

				// printf("rel: %d,%d\n", rel.x, rel.y );
				// Don't propagate the motion event unless we have focus
				// this is so that the relative motion doesn't get messed up
				// after we regain focus.
				if (window_has_focus || !mouse_mode_grab)
					input->parse_input_event(motion_event);

			} break;
			case KeyPress:
			case KeyRelease: {

				last_timestamp = event.xkey.time;

				// key event is a little complex, so
				// it will be handled in it's own function.
				handle_key_event((XKeyEvent *)&event);
			} break;
			case SelectionRequest: {

				XSelectionRequestEvent *req;
				XEvent e, respond;
				e = event;

				req = &(e.xselectionrequest);
				if (req->target == XA_STRING || req->target == XInternAtom(x11_display, "COMPOUND_TEXT", 0) ||
						req->target == XInternAtom(x11_display, "UTF8_STRING", 0)) {
					CharString clip = OS::get_clipboard().utf8();
					XChangeProperty(x11_display,
							req->requestor,
							req->property,
							req->target,
							8,
							PropModeReplace,
							(unsigned char *)clip.get_data(),
							clip.length());
					respond.xselection.property = req->property;
				} else if (req->target == XInternAtom(x11_display, "TARGETS", 0)) {

					Atom data[2];
					data[0] = XInternAtom(x11_display, "UTF8_STRING", 0);
					data[1] = XA_STRING;
					XChangeProperty(x11_display, req->requestor, req->property, req->target,
							8, PropModeReplace, (unsigned char *)&data,
							sizeof(data));
					respond.xselection.property = req->property;

				} else {
					printf("No String %x\n",
							(int)req->target);
					respond.xselection.property = None;
				}
				respond.xselection.type = SelectionNotify;
				respond.xselection.display = req->display;
				respond.xselection.requestor = req->requestor;
				respond.xselection.selection = req->selection;
				respond.xselection.target = req->target;
				respond.xselection.time = req->time;
				XSendEvent(x11_display, req->requestor, 0, 0, &respond);
				XFlush(x11_display);
			} break;

			case SelectionNotify:

				if (event.xselection.target == requested) {

					Property p = read_property(x11_display, x11_window, XInternAtom(x11_display, "PRIMARY", 0));

					Vector<String> files = String((char *)p.data).split("\n", false);
					for (int i = 0; i < files.size(); i++) {
						files[i] = files[i].replace("file://", "").replace("%20", " ").strip_escapes();
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
}

MainLoop *OS_X11::get_main_loop() const {

	return main_loop;
}

void OS_X11::delete_main_loop() {

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

	OS::set_clipboard(p_text);

	XSetSelectionOwner(x11_display, XA_PRIMARY, x11_window, CurrentTime);
	XSetSelectionOwner(x11_display, XInternAtom(x11_display, "CLIPBOARD", 0), x11_window, CurrentTime);
};

static String _get_clipboard(Atom p_source, Window x11_window, ::Display *x11_display, String p_internal_clipboard) {

	String ret;

	Atom type;
	Atom selection = XA_PRIMARY;
	int format, result;
	unsigned long len, bytes_left, dummy;
	unsigned char *data;
	Window Sown = XGetSelectionOwner(x11_display, p_source);

	if (Sown == x11_window) {

		return p_internal_clipboard;
	};

	if (Sown != None) {
		XConvertSelection(x11_display, p_source, XA_STRING, selection,
				x11_window, CurrentTime);
		XFlush(x11_display);
		while (true) {
			XEvent event;
			XNextEvent(x11_display, &event);
			if (event.type == SelectionNotify && event.xselection.requestor == x11_window) {
				break;
			};
		};

		//
		// Do not get any data, see how much data is there
		//
		XGetWindowProperty(x11_display, x11_window,
				selection, // Tricky..
				0, 0, // offset - len
				0, // Delete 0==FALSE
				AnyPropertyType, //flag
				&type, // return type
				&format, // return format
				&len, &bytes_left, //that
				&data);
		// DATA is There
		if (bytes_left > 0) {
			result = XGetWindowProperty(x11_display, x11_window,
					selection, 0, bytes_left, 0,
					AnyPropertyType, &type, &format,
					&len, &dummy, &data);
			if (result == Success) {
				ret.parse_utf8((const char *)data);
			} else
				printf("FAIL\n");
			XFree(data);
		}
	}

	return ret;
}

String OS_X11::get_clipboard() const {

	String ret;
	ret = _get_clipboard(XInternAtom(x11_display, "CLIPBOARD", 0), x11_window, x11_display, OS::get_clipboard());

	if (ret == "") {
		ret = _get_clipboard(XA_PRIMARY, x11_window, x11_display, OS::get_clipboard());
	};

	return ret;
}

String OS_X11::get_name() {

	return "X11";
}

Error OS_X11::shell_open(String p_uri) {

	Error ok;
	List<String> args;
	args.push_back(p_uri);
	ok = execute("/usr/bin/xdg-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("gnome-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("kde-open", args, false);
	return ok;
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
	Error err = const_cast<OS_X11 *>(this)->execute("/usr/bin/xdg-user-dir", arg, true, NULL, &pipe);
	if (err != OK)
		return ".";
	return pipe.strip_edges();
}

void OS_X11::move_window_to_foreground() {

	XRaiseWindow(x11_display, x11_window);
}

void OS_X11::set_cursor_shape(CursorShape p_shape) {

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (p_shape == current_cursor)
		return;
	if (mouse_mode == MOUSE_MODE_VISIBLE) {
		if (cursors[p_shape] != None)
			XDefineCursor(x11_display, x11_window, cursors[p_shape]);
		else if (cursors[CURSOR_ARROW] != None)
			XDefineCursor(x11_display, x11_window, cursors[CURSOR_ARROW]);
	}

	current_cursor = p_shape;
}

void OS_X11::release_rendering_thread() {

	context_gl->release_current();
}

void OS_X11::make_rendering_thread() {

	context_gl->make_current();
}

void OS_X11::swap_buffers() {

	context_gl->swap_buffers();
}

void OS_X11::alert(const String &p_alert, const String &p_title) {

	List<String> args;
	args.push_back("-center");
	args.push_back("-title");
	args.push_back(p_title);
	args.push_back(p_alert);

	execute("/usr/bin/xmessage", args, true);
}

void OS_X11::set_icon(const Image &p_icon) {
	Atom net_wm_icon = XInternAtom(x11_display, "_NET_WM_ICON", False);

	if (!p_icon.empty()) {
		Image img = p_icon;
		img.convert(Image::FORMAT_RGBA8);

		int w = img.get_width();
		int h = img.get_height();

		// We're using long to have wordsize (32Bit build -> 32 Bits, 64 Bit build -> 64 Bits
		Vector<long> pd;

		pd.resize(2 + w * h);

		pd[0] = w;
		pd[1] = h;

		PoolVector<uint8_t>::Read r = img.get_data().read();

		long *wr = &pd[2];
		uint8_t const *pr = r.ptr();

		for (int i = 0; i < w * h; i++) {
			long v = 0;
			//    A             R             G            B
			v |= pr[3] << 24 | pr[0] << 16 | pr[1] << 8 | pr[2];
			*wr++ = v;
			pr += 4;
		}
		XChangeProperty(x11_display, x11_window, net_wm_icon, XA_CARDINAL, 32, PropModeReplace, (unsigned char *)pd.ptr(), pd.size());
	} else {
		XDeleteProperty(x11_display, x11_window, net_wm_icon);
	}
	XFlush(x11_display);
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
		if (Main::iteration() == true)
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

void OS_X11::set_use_vsync(bool p_enable) {
	if (context_gl)
		return context_gl->set_use_vsync(p_enable);
}

bool OS_X11::is_vsync_enabled() const {

	if (context_gl)
		return context_gl->is_using_vsync();

	return true;
}

void OS_X11::set_context(int p_context) {

	XClassHint *classHint = NULL;
	classHint = XAllocClassHint();
	if (classHint) {

		if (p_context == CONTEXT_EDITOR)
			classHint->res_name = (char *)"Godot_Editor";
		if (p_context == CONTEXT_PROJECTMAN)
			classHint->res_name = (char *)"Godot_ProjectList";
		classHint->res_class = (char *)"Godot";
		XSetClassHint(x11_display, x11_window, classHint);
		XFree(classHint);
	}
}

PowerState OS_X11::get_power_state() {
	return power_manager->get_power_state();
}

int OS_X11::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_X11::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

OS_X11::OS_X11() {

#ifdef RTAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_rtaudio);
#endif

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

#ifdef ALSA_ENABLED
	AudioDriverManager::add_driver(&driver_alsa);
#endif

	if (AudioDriverManager::get_driver_count() == 0) {
		WARN_PRINT("No sound driver found... Defaulting to dummy driver");
		AudioDriverManager::add_driver(&driver_dummy);
	}

	minimized = false;
	xim_style = 0L;
	mouse_mode = MOUSE_MODE_VISIBLE;
}
