/*************************************************************************/
/*  os_sdl.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os_sdl.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "errno.h"
#include "key_mapping_sdl.h"
#include "print_string.h"
#include "core/string_builder.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"

#ifdef HAVE_MNTENT
#include <mntent.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL.h>
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

#undef CursorShape

int OS_SDL::get_video_driver_count() const {
	return 1;
}

const char *OS_SDL::get_video_driver_name(int p_driver) const {
	return "GLES3";
}

int OS_SDL::get_audio_driver_count() const {
	return AudioDriverManager::get_driver_count();
}

const char *OS_SDL::get_audio_driver_name(int p_driver) const {

	AudioDriver *driver = AudioDriverManager::get_driver(p_driver);
	ERR_FAIL_COND_V(!driver, "");
	return AudioDriverManager::get_driver(p_driver)->get_name();
}

void OS_SDL::initialize_core() {

	crash_handler.initialize();

	OS_Unix::initialize_core();
}

Error OS_SDL::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	long im_event_mask = 0;
	last_button_state = 0;

	sdl_window = NULL;
	last_click_ms = 0;
	args = OS::get_singleton()->get_cmdline_args();
	current_videomode = p_desired;
	main_loop = NULL;
	last_timestamp = 0;
	last_mouse_pos_valid = false;
	last_keyrelease_time = 0;
	const SDL_RendererInfo* info = 0;

	// ** SDL INIT ** //

	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		ERR_PRINT("SDL Initialization Failed!");
		return ERR_UNAVAILABLE;
	}

	const char *err;
	int sdl_displays = 0;
	int event_base, error_base;

	sdl_displays = SDL_GetNumVideoDisplays();
	if (sdl_displays < 0) {
		fprintf(stderr, "Could not list displays from SDL, Error: %s\n", SDL_GetError());
	}


#ifdef TOUCH_ENABLED
	// if (!XQueryExtension(x11_display, "XInputExtension", &touch.opcode, &event_base, &error_base)) {
	// 	fprintf(stderr, "XInput extension not available");
	// } else {
		// 2.2 is the first release with multitouch
	// 	int xi_major = 2;
	// 	int xi_minor = 2;
	// 	if (XIQueryVersion(x11_display, &xi_major, &xi_minor) != Success) {
	// 		fprintf(stderr, "XInput 2.2 not available (server supports %d.%d)\n", xi_major, xi_minor);
	// 		touch.opcode = 0;
	// 	} else {
	// 		int dev_count;
	// 		XIDeviceInfo *info = XIQueryDevice(x11_display, XIAllDevices, &dev_count);

	// 		for (int i = 0; i < dev_count; i++) {
	// 			XIDeviceInfo *dev = &info[i];
	// 			if (!dev->enabled)
	// 				continue;
	// 			if (!(dev->use == XIMasterPointer || dev->use == XIFloatingSlave))
	// 				continue;

	// 			bool direct_touch = false;
	// 			for (int j = 0; j < dev->num_classes; j++) {
	// 				if (dev->classes[j]->type == XITouchClass && ((XITouchClassInfo *)dev->classes[j])->mode == XIDirectTouch) {
	// 					direct_touch = true;
	// 					break;
	// 				}
	// 			}
	// 			if (direct_touch) {
	// 				touch.devices.push_back(dev->deviceid);
	// 				fprintf(stderr, "Using touch device: %s\n", dev->name);
	// 			}
	// 		}

	// 		XIFreeDeviceInfo(info);

	// 		if (is_stdout_verbose() && !touch.devices.size()) {
	// 			fprintf(stderr, "No touch devices found\n");
	// 		}
	// 	}
	// }
#endif

// maybe contextgl wants to be in charge of creating the window
//print_line("def videomode "+itos(current_videomode.width)+","+itos(current_videomode.height));
#if defined(OPENGL_ENABLED)

	context_gl = memnew(ContextGL_SDL(sdl_display_mode, current_videomode, true));
	context_gl->initialize();
	sdl_window = context_gl->get_window_pointer();

	RasterizerGLES3::register_config();

	RasterizerGLES3::make_current();

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
		set_borderless_window(current_videomode.borderless_window);
	}

	// enable / disable resizable window
	if (current_videomode.resizable) {
		set_window_resizable(current_videomode.resizable);
	}

	for (int i = 0; i < CURSOR_MAX; i++) {
		cursors[i] = NULL;
	}

	AudioDriverManager::initialize(p_audio_driver);

	ERR_FAIL_COND_V(!visual_server, ERR_UNAVAILABLE);
	ERR_FAIL_COND_V(sdl_window < 0, ERR_UNAVAILABLE);

#ifdef TOUCH_ENABLED
	// if (touch.devices.size()) {

		// Must be alive after this block
	// 	static unsigned char mask_data[XIMaskLen(XI_LASTEVENT)] = {};

	// 	touch.event_mask.deviceid = XIAllDevices;
	// 	touch.event_mask.mask_len = sizeof(mask_data);
	// 	touch.event_mask.mask = mask_data;

	// 	XISetMask(touch.event_mask.mask, XI_TouchBegin);
	// 	XISetMask(touch.event_mask.mask, XI_TouchUpdate);
	// 	XISetMask(touch.event_mask.mask, XI_TouchEnd);
	// 	XISetMask(touch.event_mask.mask, XI_TouchOwnership);

	// 	XISelectEvents(x11_display, x11_window, &touch.event_mask, 1);

		// Disabled by now since grabbing also blocks mouse events
		// (they are received as extended events instead of standard events)
	// 	/*XIClearMask(touch.event_mask.mask, XI_TouchOwnership);

		// Grab touch devices to avoid OS gesture interference
	// 	for (int i = 0; i < touch.devices.size(); ++i) {
	// 		XIGrabDevice(x11_display, touch.devices[i], x11_window, CurrentTime, None, XIGrabModeAsync, XIGrabModeAsync, False, &touch.event_mask);
	// 	}*/
	// }
#endif

	visual_server->init();

	input = memnew(InputDefault);

	window_has_focus = true; // Set focus to true at init
#ifdef JOYDEV_ENABLED
	joypad = memnew(JoypadLinux(input));
#endif
	_ensure_user_data_dir();

	power_manager = memnew(PowerSDL);

	return OK;
}

void OS_SDL::set_ime_position(const Point2 &p_pos) {
	SDL_Rect ime_position;
	ime_position.x = p_pos.x / 2;
	ime_position.y = p_pos.y / 2;
	// FIXME: What should these values be?
	ime_position.w = 1;
	ime_position.h = 1;

	SDL_SetTextInputRect(&ime_position);
}

void OS_SDL::finalize() {

 	if (main_loop)
 		memdelete(main_loop);
 	main_loop = NULL;

	/*
	if (debugger_connection_console) {
		memdelete(debugger_connection_console);
	}
	*/

#ifdef JOYDEV_ENABLED
	memdelete(joypad);
#endif
#ifdef TOUCH_ENABLED
	touch.devices.clear();
	touch.state.clear();
#endif
	memdelete(input);

	visual_server->finish();
	memdelete(visual_server);
	//memdelete(rasterizer);

 	memdelete(power_manager);

#if defined(OPENGL_ENABLED)
	memdelete(context_gl);
#endif
	for (int i = 0; i < CURSOR_MAX; i++) {
		if (cursors[i] != NULL)
			SDL_FreeCursor(cursors[i]);
	}

	args.clear();
}

void OS_SDL::set_mouse_mode(MouseMode p_mode) {
	if (p_mode == mouse_mode) return;
	mouse_mode = p_mode;

	if (mouse_mode == MOUSE_MODE_VISIBLE)	SDL_ShowCursor(SDL_ENABLE);
	else if (mouse_mode == MOUSE_MODE_HIDDEN) SDL_ShowCursor(SDL_DISABLE);

	// Allow for undoing of previously made changes.
	SDL_SetRelativeMouseMode(mouse_mode == MOUSE_MODE_CAPTURED ? SDL_TRUE : SDL_FALSE);
	SDL_SetWindowGrab(sdl_window, mouse_mode == MOUSE_MODE_CONFINED ? SDL_TRUE : SDL_FALSE);
}

void OS_SDL::warp_mouse_position(const Point2 &p_to) {

	if (mouse_mode == MOUSE_MODE_CAPTURED) {

		last_mouse_pos = p_to;
	} else {
		SDL_WarpMouseInWindow(sdl_window, (int)p_to.x, (int)p_to.y);
	}
}

OS::MouseMode OS_SDL::get_mouse_mode() const {
	return mouse_mode;
}

int OS_SDL::get_mouse_button_state() const {
	return last_button_state;
}

Point2 OS_SDL::get_mouse_position() const {
	return last_mouse_pos;
}

void OS_SDL::set_window_title(const String &p_title) {
	SDL_SetWindowTitle(sdl_window, (const char *)p_title.utf8().get_data());
}

void OS_SDL::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_SDL::get_video_mode(int p_screen) const {
	return current_videomode;
}

void OS_SDL::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
}

void OS_SDL::set_wm_fullscreen(bool p_enabled) {
	if (current_videomode.fullscreen == p_enabled)
		return;

	SDL_SetWindowFullscreen(sdl_window, p_enabled ? SDL_WINDOW_FULLSCREEN : 0);
}

int OS_SDL::get_screen_count() const {
	int num_displays = SDL_GetNumVideoDisplays();

	if (num_displays < 0) {
		fprintf(stderr, "Could not get display count from SDL, Error: %s\n", SDL_GetError());
	}

	return num_displays;
}

int OS_SDL::get_current_screen() const {
	int current_display = SDL_GetWindowDisplayIndex(sdl_window);

	if (current_display < 0) {
		fprintf(stderr, "Could not get the current display from SDL, Error: %s\n", SDL_GetError());
	}

	return current_display;
}

void OS_SDL::set_current_screen(int p_screen) {
	int screen_count = get_screen_count();
	if (p_screen >= screen_count) return;

	if (current_videomode.fullscreen) {
		Size2i size = get_screen_size(p_screen);
		Point2i position = get_screen_position(p_screen);

		SDL_SetWindowPosition(sdl_window, position.x, position.y);
	} else {
		if (p_screen != get_current_screen()) {
			Point2i position = get_screen_position(p_screen);
			SDL_SetWindowPosition(sdl_window, position.x, position.y);
		}
	}
}

Point2 OS_SDL::get_screen_position(int p_screen) const {
	SDL_Rect display_bounds;

	if (SDL_GetDisplayBounds(p_screen, &display_bounds) != 0) {
		fprintf(stderr, "Could not get the screen position for display %i from SDL, Error: %s\n", p_screen, SDL_GetError());
		return Point2i(0, 0);
	}

	return Point2i(display_bounds.x, display_bounds.y);
}

Size2 OS_SDL::get_screen_size(int p_screen) const {
	SDL_Rect display_bounds;

	if (SDL_GetDisplayBounds(p_screen, &display_bounds) != 0) {
		fprintf(stderr, "Could not get the screen position for display %i from SDL, Error: %s\n", p_screen, SDL_GetError());
		return Size2i(0, 0);
	}

	return Size2i(display_bounds.w, display_bounds.h);
}

int OS_SDL::get_screen_dpi(int p_screen) const {
	if (p_screen == -1) {
		p_screen = get_current_screen();
	}

	// Invalid screen?
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), 0);

	float diagonal_dpi = 96.0f;

	if (SDL_GetDisplayDPI(p_screen, &diagonal_dpi, NULL, NULL) != 0) {
		fprintf(stderr, "Could not get the screen DPI for display %i from SDL, Error: %s\n", p_screen, SDL_GetError());
	}

	return static_cast<int>(diagonal_dpi);
}

Point2 OS_SDL::get_window_position() const {
	int x, y;
	
	SDL_GetWindowPosition(sdl_window, &x, &y);

	return Point2i(x, y);
}

void OS_SDL::set_window_position(const Point2 &p_position) {
	SDL_SetWindowPosition(sdl_window, p_position.x, p_position.y);
}

Size2 OS_SDL::get_window_size() const {
	int w, h;
	SDL_GetWindowSize(sdl_window, &w, &h);

	// FIXME: Should we use current_videomode here instead?
	// FIXME: How should HIDPI be handled?
	return Size2i(w, h);
}

void OS_SDL::set_window_size(const Size2 p_size) {
	SDL_SetWindowSize(sdl_window, p_size.x, p_size.y);

	// Update our videomode width and height
	current_videomode.width = p_size.x;
	current_videomode.height = p_size.y;
}

void OS_SDL::set_window_fullscreen(bool p_enabled) {
	set_wm_fullscreen(p_enabled);
	current_videomode.fullscreen = p_enabled;
}

bool OS_SDL::is_window_fullscreen() const {
	return current_videomode.fullscreen;
}

void OS_SDL::set_window_resizable(bool p_enabled) {
	SDL_SetWindowResizable(sdl_window, p_enabled ? SDL_TRUE : SDL_FALSE);
	current_videomode.resizable = p_enabled;
}

bool OS_SDL::is_window_resizable() const {
	return current_videomode.resizable;
}

void OS_SDL::set_window_minimized(bool p_enabled) {
	if (is_window_minimized() == p_enabled) return;

	if (p_enabled) {
		SDL_MinimizeWindow(sdl_window);
	} else {
		SDL_RestoreWindow(sdl_window);
	}
}

bool OS_SDL::is_window_minimized() const {
	uint32_t flags = SDL_GetWindowFlags(sdl_window);

	return flags & SDL_WINDOW_MINIMIZED;
}

void OS_SDL::set_window_maximized(bool p_enabled) {
	if (is_window_maximized() == p_enabled) return;

	if (p_enabled) {
		SDL_MaximizeWindow(sdl_window);
	} else {
		SDL_RestoreWindow(sdl_window);
	}

	if (is_window_maximize_allowed()) {
	 	while (p_enabled && !is_window_maximized()) {
			// Wait for effective resizing (so the GLX context is too).
	 	}
	}

	maximized = p_enabled;
}

bool OS_SDL::is_window_maximize_allowed() {
	uint32_t flags = SDL_GetWindowFlags(sdl_window);

	// FIXME: Not quite exact, but hopefully close enough.
	return flags & SDL_WINDOW_RESIZABLE;
}

bool OS_SDL::is_window_maximized() const {
	uint32_t flags = SDL_GetWindowFlags(sdl_window);

	// FIXME: Not quite exact, but hopefully close enough.
	return flags & SDL_WINDOW_MAXIMIZED;
}

void OS_SDL::set_borderless_window(bool p_borderless) {

	if (current_videomode.borderless_window == p_borderless)
		return;

	current_videomode.borderless_window = p_borderless;

	SDL_SetWindowBordered(sdl_window, p_borderless ? SDL_FALSE : SDL_TRUE);
}

bool OS_SDL::get_borderless_window() {
	return current_videomode.borderless_window;
}

// FIXME: Unimplemented
void OS_SDL::request_attention() {
}

void OS_SDL::get_key_modifier_state(Ref<InputEventWithModifiers> state) {
	SDL_Keymod mod_state = SDL_GetModState();
	state->set_shift(mod_state & KMOD_SHIFT);
	state->set_control(mod_state & KMOD_CTRL);
	state->set_alt(mod_state & KMOD_ALT);
	state->set_metakey(mod_state & KMOD_GUI);
}

unsigned int OS_SDL::get_mouse_button_state(uint32_t button_mask, bool refresh) {
	if(refresh) button_mask = SDL_GetMouseState(NULL, NULL);

	unsigned int state = 0;

	if (button_mask & SDL_BUTTON(SDL_BUTTON_LEFT)) state |= 1 << 0;
	if (button_mask & SDL_BUTTON(SDL_BUTTON_RIGHT)) state |= 1 << 1;
	if (button_mask & SDL_BUTTON(SDL_BUTTON_MIDDLE)) state |= 1 << 2;
	if (button_mask & SDL_BUTTON(SDL_BUTTON_X1)) state |= 1 << 3;
	if (button_mask & SDL_BUTTON(SDL_BUTTON_X2)) state |= 1 << 4;

	last_button_state = state;
	return state;
}

void OS_SDL::process_events() {
	SDL_Event event;

	do_mouse_warp = false;
 	bool mouse_mode_grab = mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED;
	Size2i window_size;

	while (SDL_PollEvent(&event)) {

		if(event.type == SDL_WINDOWEVENT) {

			switch(event.window.event) {
				case SDL_WINDOWEVENT_EXPOSED:
					Main::force_redraw();
					break;
				case SDL_WINDOWEVENT_MINIMIZED:
					minimized = true;
					break;
				case SDL_WINDOWEVENT_LEAVE:
					if (main_loop && !mouse_mode_grab)
						main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_EXIT);
					if (input)
						input->set_mouse_in_window(false);
					break;
				case SDL_WINDOWEVENT_ENTER:
					if (main_loop && !mouse_mode_grab)
						main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_ENTER);
					if (input)
						input->set_mouse_in_window(true);
					break;
				case SDL_WINDOWEVENT_FOCUS_GAINED:
					minimized = false;
					window_has_focus = true;
					main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
					// FIXME: Mot sure if we should handle the mouse grabbing manually or if SDL will handle it. Test.
					break;
				case SDL_WINDOWEVENT_SIZE_CHANGED:
					window_size = get_window_size();
					current_videomode.width = window_size.x;
					current_videomode.height = window_size.y;
					break;
				case SDL_WINDOWEVENT_CLOSE:
					main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
					break;
			}

			continue; // Probably not a good pattern, but I'm not a fan of else-if in this case.
		}

		if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) {
				/* exit in case of a mouse button press */
				last_timestamp = event.button.timestamp;
				if (mouse_mode == MOUSE_MODE_CAPTURED) {
					event.button.x = last_mouse_pos.x;
					event.button.y = last_mouse_pos.y;
				}

				Ref<InputEventMouseButton> mb;
				mb.instance();

				get_key_modifier_state(mb);
				mb->set_button_mask(get_mouse_button_state(0, true));
				mb->set_position(Vector2(event.button.x, event.button.y));
				mb->set_global_position(mb->get_position());
				mb->set_button_index(event.button.button);

				// Swapping buttons around?
				if (mb->get_button_index() == 2)
					mb->set_button_index(3);
				else if (mb->get_button_index() == 3)
					mb->set_button_index(2);

				mb->set_pressed(event.button.state == SDL_PRESSED);

				mb->set_doubleclick(event.button.clicks > 1);

				input->parse_input_event(mb);
		}

		// Ahh, good ol' abstractions. :3
		if(event.type == SDL_MOUSEMOTION) {
				last_timestamp = event.motion.timestamp;

				// Motion is also simple.
				// A little hack is in order
				// to be able to send relative motion events.
				Point2i pos(event.motion.x, event.motion.y);

				// TODO: See if this can be replaced.
				if (mouse_mode == MOUSE_MODE_CAPTURED) {

					if (pos == Point2i(current_videomode.width / 2, current_videomode.height / 2)) {
						//this sucks, it's a hack, etc and is a little inaccurate, etc.

						center = pos;
						break;
					}

					Point2i new_center = pos;
					pos = last_mouse_pos + (pos - center);
					center = new_center;
					do_mouse_warp = window_has_focus; // warp the cursor if we're focused in
				}

				if (!last_mouse_pos_valid) {
					last_mouse_pos = pos;
					last_mouse_pos_valid = true;
				}

				Point2i rel(event.motion.xrel, event.motion.yrel);

				Ref<InputEventMouseMotion> mm;
				mm.instance();

				get_key_modifier_state(mm);
				mm->set_button_mask(get_mouse_button_state(event.motion.state, false));
				mm->set_position(pos);
				mm->set_global_position(pos);
				input->set_mouse_position(pos);
				mm->set_speed(input->get_last_mouse_speed());
				mm->set_relative(rel);

				last_mouse_pos = pos;

				// Don't propagate the motion event unless we have focus
				// this is so that the relative motion doesn't get messed up
				// after we regain focus.
				// FIXME: Not sure if we need this or not.
				if (window_has_focus || !mouse_mode_grab)
					input->parse_input_event(mm);

				continue;
		}

		if(event.type == SDL_KEYDOWN) {
			last_timestamp = event.key.timestamp;
			SDL_Keysym keysym = event.key.keysym;
			SDL_Scancode scancode = keysym.scancode;
			SDL_Keycode keycode = keysym.sym;

			Status status;
			Ref<InputEventKey> k;

			k.instance();
			if (keycode == 0) {
				continue;
			}

			get_key_modifier_state(k);

			k->set_pressed(event.key.state == SDL_PRESSED);
			k->set_scancode(scancode);
			k->set_echo(event.key.repeat > 0);

			input->parse_input_event(k);
			continue;
		}

		if(event.type == SDL_KEYUP) {
			last_timestamp = event.key.timestamp;
			SDL_Keysym keysym = event.key.keysym;
			SDL_Scancode scancode = keysym.scancode;
			SDL_Keycode keycode = keysym.sym;

			Status status;
			Ref<InputEventKey> k;
			k.instance();
			if (keycode == 0) {
				continue;
			}

			get_key_modifier_state(k);

			k->set_unicode(keycode);
			k->set_pressed(event.key.state == SDL_PRESSED);
			k->set_scancode(scancode);
			k->set_echo(event.key.repeat > 0);

			input->parse_input_event(k);
			continue;
		}
	}

	// if(pending_key_event != None) {
	// 	printf("TRYING TO PARSE PENDING\n");
	// 	input->parse_input_event(pending_key_event);
	// }
		
	//printf("checking events %i\n", XPending(x11_display));

// 	do_mouse_warp = false;

// 	// Is the current mouse mode one where it needs to be grabbed.
// 	bool mouse_mode_grab = mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED;

// 	while (XPending(x11_display) > 0) {
// 		XEvent event;
// 		XNextEvent(x11_display, &event);

// 		if (XFilterEvent(&event, None)) {
// 			continue;
// 		}

// #ifdef TOUCH_ENABLED
// 		if (XGetEventData(x11_display, &event.xcookie)) {

// 			if (event.xcookie.type == GenericEvent && event.xcookie.extension == touch.opcode) {

// 				XIDeviceEvent *event_data = (XIDeviceEvent *)event.xcookie.data;
// 				int index = event_data->detail;
// 				Vector2 pos = Vector2(event_data->event_x, event_data->event_y);

// 				switch (event_data->evtype) {

// 					case XI_TouchBegin: // Fall-through
// 							// Disabled hand-in-hand with the grabbing
// 							//XIAllowTouchEvents(x11_display, event_data->deviceid, event_data->detail, x11_window, XIAcceptTouch);

// 					case XI_TouchEnd: {

// 						bool is_begin = event_data->evtype == XI_TouchBegin;

// 						Ref<InputEventScreenTouch> st;
// 						st.instance();
// 						st->set_index(index);
// 						st->set_position(pos);
// 						st->set_pressed(is_begin);

// 						if (is_begin) {
// 							if (touch.state.has(index)) // Defensive
// 								break;
// 							touch.state[index] = pos;
// 							input->parse_input_event(st);
// 						} else {
// 							if (!touch.state.has(index)) // Defensive
// 								break;
// 							touch.state.erase(index);
// 							input->parse_input_event(st);
// 						}
// 					} break;

// 					case XI_TouchUpdate: {

// 						Map<int, Vector2>::Element *curr_pos_elem = touch.state.find(index);
// 						if (!curr_pos_elem) { // Defensive
// 							break;
// 						}

// 						if (curr_pos_elem->value() != pos) {

// 							Ref<InputEventScreenDrag> sd;
// 							sd.instance();
// 							sd->set_index(index);
// 							sd->set_position(pos);
// 							sd->set_relative(pos - curr_pos_elem->value());
// 							input->parse_input_event(sd);

// 							curr_pos_elem->value() = pos;
// 						}
// 					} break;
// 				}
// 			}
// 		}
// 		XFreeEventData(x11_display, &event.xcookie);
// #endif

// 		switch (event.type) {
// 			case Expose:
// 				Main::force_redraw();
// 				break;

// 			case NoExpose:
// 				minimized = true;
// 				break;

// 			case VisibilityNotify: {
// 				XVisibilityEvent *visibility = (XVisibilityEvent *)&event;
// 				minimized = (visibility->state == VisibilityFullyObscured);
// 			} break;
// 			case LeaveNotify: {
// 				if (main_loop && !mouse_mode_grab)
// 					main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_EXIT);
// 				if (input)
// 					input->set_mouse_in_window(false);

// 			} break;
// 			case EnterNotify: {
// 				if (main_loop && !mouse_mode_grab)
// 					main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_ENTER);
// 				if (input)
// 					input->set_mouse_in_window(true);
// 			} break;
// 			case FocusIn:
// 				minimized = false;
// 				window_has_focus = true;
// 				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
// 				if (mouse_mode_grab) {
// 					// Show and update the cursor if confined and the window regained focus.
// 					if (mouse_mode == MOUSE_MODE_CONFINED)
// 						XUndefineCursor(x11_display, x11_window);
// 					else if (mouse_mode == MOUSE_MODE_CAPTURED) // or re-hide it in captured mode
// 						XDefineCursor(x11_display, x11_window, null_cursor);

// 					XGrabPointer(
// 							x11_display, x11_window, True,
// 							ButtonPressMask | ButtonReleaseMask | PointerMotionMask,
// 							GrabModeAsync, GrabModeAsync, x11_window, None, CurrentTime);
// 				}
// #ifdef TOUCH_ENABLED
// 					// Grab touch devices to avoid OS gesture interference
// 					/*for (int i = 0; i < touch.devices.size(); ++i) {
// 					XIGrabDevice(x11_display, touch.devices[i], x11_window, CurrentTime, None, XIGrabModeAsync, XIGrabModeAsync, False, &touch.event_mask);
// 				}*/
// #endif
// 				if (xic) {
// 					XSetICFocus(xic);
// 				}
// 				break;

// 			case FocusOut:
// 				window_has_focus = false;
// 				main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
// 				if (mouse_mode_grab) {
// 					//dear X11, I try, I really try, but you never work, you do whathever you want.
// 					if (mouse_mode == MOUSE_MODE_CAPTURED) {
// 						// Show the cursor if we're in captured mode so it doesn't look weird.
// 						XUndefineCursor(x11_display, x11_window);
// 					}
// 					XUngrabPointer(x11_display, CurrentTime);
// 				}
// #ifdef TOUCH_ENABLED
// 				// Ungrab touch devices so input works as usual while we are unfocused
// 				/*for (int i = 0; i < touch.devices.size(); ++i) {
// 					XIUngrabDevice(x11_display, touch.devices[i], CurrentTime);
// 				}*/

// 				// Release every pointer to avoid sticky points
// 				for (Map<int, Vector2>::Element *E = touch.state.front(); E; E = E->next()) {

// 					Ref<InputEventScreenTouch> st;
// 					st.instance();
// 					st->set_index(E->key());
// 					st->set_position(E->get());
// 					input->parse_input_event(st);
// 				}
// 				touch.state.clear();
// #endif
// 				if (xic) {
// 					XUnsetICFocus(xic);
// 				}
// 				break;

// 			case ConfigureNotify:
// 				_window_changed(&event);
// 				break;
// 			case KeyPress:
// 			case KeyRelease: {

// 				last_timestamp = event.xkey.time;

// 				// key event is a little complex, so
// 				// it will be handled in it's own function.
// 				handle_key_event((XKeyEvent *)&event);
// 			} break;
// 			case SelectionRequest: {

// 				XSelectionRequestEvent *req;
// 				XEvent e, respond;
// 				e = event;

// 				req = &(e.xselectionrequest);
// 				if (req->target == XInternAtom(x11_display, "UTF8_STRING", 0) ||
// 						req->target == XInternAtom(x11_display, "COMPOUND_TEXT", 0) ||
// 						req->target == XInternAtom(x11_display, "TEXT", 0) ||
// 						req->target == XA_STRING ||
// 						req->target == XInternAtom(x11_display, "text/plain;charset=utf-8", 0) ||
// 						req->target == XInternAtom(x11_display, "text/plain", 0)) {
// 					CharString clip = OS::get_clipboard().utf8();
// 					XChangeProperty(x11_display,
// 							req->requestor,
// 							req->property,
// 							req->target,
// 							8,
// 							PropModeReplace,
// 							(unsigned char *)clip.get_data(),
// 							clip.length());
// 					respond.xselection.property = req->property;
// 				} else if (req->target == XInternAtom(x11_display, "TARGETS", 0)) {

// 					Atom data[7];
// 					data[0] = XInternAtom(x11_display, "TARGETS", 0);
// 					data[1] = XInternAtom(x11_display, "UTF8_STRING", 0);
// 					data[2] = XInternAtom(x11_display, "COMPOUND_TEXT", 0);
// 					data[3] = XInternAtom(x11_display, "TEXT", 0);
// 					data[4] = XA_STRING;
// 					data[5] = XInternAtom(x11_display, "text/plain;charset=utf-8", 0);
// 					data[6] = XInternAtom(x11_display, "text/plain", 0);

// 					XChangeProperty(x11_display,
// 							req->requestor,
// 							req->property,
// 							XA_ATOM,
// 							32,
// 							PropModeReplace,
// 							(unsigned char *)&data,
// 							sizeof(data) / sizeof(data[0]));
// 					respond.xselection.property = req->property;

// 				} else {
// 					char *targetname = XGetAtomName(x11_display, req->target);
// 					printf("No Target '%s'\n", targetname);
// 					if (targetname)
// 						XFree(targetname);
// 					respond.xselection.property = None;
// 				}

// 				respond.xselection.type = SelectionNotify;
// 				respond.xselection.display = req->display;
// 				respond.xselection.requestor = req->requestor;
// 				respond.xselection.selection = req->selection;
// 				respond.xselection.target = req->target;
// 				respond.xselection.time = req->time;
// 				XSendEvent(x11_display, req->requestor, True, NoEventMask, &respond);
// 				XFlush(x11_display);
// 			} break;

// 			case SelectionNotify:

// 				if (event.xselection.target == requested) {

// 					Property p = read_property(x11_display, x11_window, XInternAtom(x11_display, "PRIMARY", 0));

// 					Vector<String> files = String((char *)p.data).split("\n", false);
// 					for (int i = 0; i < files.size(); i++) {
// 						files[i] = files[i].replace("file://", "").replace("%20", " ").strip_escapes();
// 					}
// 					main_loop->drop_files(files);

// 					//Reply that all is well.
// 					XClientMessageEvent m;
// 					memset(&m, 0, sizeof(m));
// 					m.type = ClientMessage;
// 					m.display = x11_display;
// 					m.window = xdnd_source_window;
// 					m.message_type = xdnd_finished;
// 					m.format = 32;
// 					m.data.l[0] = x11_window;
// 					m.data.l[1] = 1;
// 					m.data.l[2] = xdnd_action_copy; //We only ever copy.

// 					XSendEvent(x11_display, xdnd_source_window, False, NoEventMask, (XEvent *)&m);
// 				}
// 				break;

// 			case ClientMessage:

// 				if ((unsigned int)event.xclient.data.l[0] == (unsigned int)wm_delete)
// 					main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);

// 				else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_enter) {

// 					//File(s) have been dragged over the window, check for supported target (text/uri-list)
// 					xdnd_version = (event.xclient.data.l[1] >> 24);
// 					Window source = event.xclient.data.l[0];
// 					bool more_than_3 = event.xclient.data.l[1] & 1;
// 					if (more_than_3) {
// 						Property p = read_property(x11_display, source, XInternAtom(x11_display, "XdndTypeList", False));
// 						requested = pick_target_from_list(x11_display, (Atom *)p.data, p.nitems);
// 					} else
// 						requested = pick_target_from_atoms(x11_display, event.xclient.data.l[2], event.xclient.data.l[3], event.xclient.data.l[4]);
// 				} else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_position) {

// 					//xdnd position event, reply with an XDND status message
// 					//just depending on type of data for now
// 					XClientMessageEvent m;
// 					memset(&m, 0, sizeof(m));
// 					m.type = ClientMessage;
// 					m.display = event.xclient.display;
// 					m.window = event.xclient.data.l[0];
// 					m.message_type = xdnd_status;
// 					m.format = 32;
// 					m.data.l[0] = x11_window;
// 					m.data.l[1] = (requested != None);
// 					m.data.l[2] = 0; //empty rectangle
// 					m.data.l[3] = 0;
// 					m.data.l[4] = xdnd_action_copy;

// 					XSendEvent(x11_display, event.xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
// 					XFlush(x11_display);
// 				} else if ((unsigned int)event.xclient.message_type == (unsigned int)xdnd_drop) {

// 					if (requested != None) {
// 						xdnd_source_window = event.xclient.data.l[0];
// 						if (xdnd_version >= 1)
// 							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), x11_window, event.xclient.data.l[2]);
// 						else
// 							XConvertSelection(x11_display, xdnd_selection, requested, XInternAtom(x11_display, "PRIMARY", 0), x11_window, CurrentTime);
// 					} else {
// 						//Reply that we're not interested.
// 						XClientMessageEvent m;
// 						memset(&m, 0, sizeof(m));
// 						m.type = ClientMessage;
// 						m.display = event.xclient.display;
// 						m.window = event.xclient.data.l[0];
// 						m.message_type = xdnd_finished;
// 						m.format = 32;
// 						m.data.l[0] = x11_window;
// 						m.data.l[1] = 0;
// 						m.data.l[2] = None; //Failed.
// 						XSendEvent(x11_display, event.xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
// 					}
// 				}
// 				break;
// 			default:
// 				break;
// 		}
// 	}

	// XFlush(x11_display);

	if (do_mouse_warp) {

		// XWarpPointer(x11_display, None, x11_window,
		// 		0, 0, 0, 0, (int)current_videomode.width / 2, (int)current_videomode.height / 2);

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

MainLoop *OS_SDL::get_main_loop() const {
	return main_loop;
}

void OS_SDL::delete_main_loop() {
	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;
}

void OS_SDL::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

bool OS_SDL::can_draw() const {
	return !minimized;
};

void OS_SDL::set_clipboard(const String &p_text) {

	OS::set_clipboard(p_text);

	SDL_SetClipboardText((const char *)p_text.utf8().get_data());
};

String OS_SDL::get_clipboard() const {

	StringBuilder *sb = new StringBuilder();	
	sb->append(SDL_GetClipboardText());

	return sb->as_string();
}

String OS_SDL::get_name() {
	return "SDL";
}

Error OS_SDL::shell_open(String p_uri) {

	Error ok;
	List<String> args;
	args.push_back(p_uri);
	ok = execute("xdg-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("gnome-open", args, false);
	if (ok == OK)
		return OK;
	ok = execute("kde-open", args, false);
	return ok;
}

bool OS_SDL::_check_internal_feature_support(const String &p_feature) {

	return p_feature == "pc" || p_feature == "s3tc";
}

String OS_SDL::get_config_path() const {

	if (has_environment("XDG_CONFIG_HOME")) {
		return get_environment("XDG_CONFIG_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".config");
	} else {
		return ".";
	}
}

String OS_SDL::get_data_path() const {

	if (has_environment("XDG_DATA_HOME")) {
		return get_environment("XDG_DATA_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".local/share");
	} else {
		return get_config_path();
	}
}

String OS_SDL::get_cache_path() const {

	if (has_environment("XDG_CACHE_HOME")) {
		return get_environment("XDG_CACHE_HOME");
	} else if (has_environment("HOME")) {
		return get_environment("HOME").plus_file(".cache");
	} else {
		return get_config_path();
	}
}

String OS_SDL::get_system_dir(SystemDir p_dir) const {

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
	Error err = const_cast<OS_SDL *>(this)->execute("xdg-user-dir", arg, true, NULL, &pipe);
	if (err != OK)
		return ".";
	return pipe.strip_edges();
}

void OS_SDL::move_window_to_foreground() {
	SDL_RaiseWindow(sdl_window);
}

void OS_SDL::set_cursor_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);
	if (p_shape == current_cursor) return;

	SDL_Cursor* sdl_cursor;

	if (cursors[p_shape] == NULL) {
		SDL_SystemCursor sdl_cursor_id;
		// FIXME: SDL only supports a subset of system cursors, perhaps we should figure out how to load them ourselves?
		switch (p_shape) {
			case CURSOR_ARROW:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_ARROW;
				break;
			case CURSOR_IBEAM:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_IBEAM;
				break;
			case CURSOR_POINTING_HAND:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_HAND;
				break;
			case CURSOR_CROSS:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_CROSSHAIR;
				break;
			case CURSOR_WAIT:
			case CURSOR_BUSY:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_WAIT;
				break;
			case CURSOR_DRAG:
			case CURSOR_CAN_DROP:
			case CURSOR_MOVE:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_HAND;
				break;
			case CURSOR_FORBIDDEN:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_NO;
				break;
			case CURSOR_VSIZE:
			case CURSOR_VSPLIT:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_SIZENS;
				break;
			case CURSOR_HSIZE:
			case CURSOR_HSPLIT:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_SIZEWE;
				break;
			case CURSOR_BDIAGSIZE:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_SIZENWSE;
				break;
			case CURSOR_FDIAGSIZE:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_SIZENESW;
				break;
			default:
				sdl_cursor_id = SDL_SYSTEM_CURSOR_ARROW;
		}

		cursors[p_shape] = SDL_CreateSystemCursor(sdl_cursor_id);
	}
	
	sdl_cursor = cursors[p_shape];
	current_cursor = p_shape;

	SDL_SetCursor(sdl_cursor);
}

// TODO: Not sure if this works..
void OS_SDL::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	if (p_cursor.is_valid()) {
	 	Ref<Texture> texture = p_cursor;
	 	Ref<Image> img = texture->get_data()->duplicate();
		img->convert(Image::FORMAT_RGBA8);

		int w = img->get_width();
		int h = img->get_height();

	 	// FIXME: Should this fail with SDL?
	 	ERR_FAIL_COND(w != 32 || h != 32);

		PoolVector<uint8_t>::Read r = img->get_data().read();

		uint8_t *pr = const_cast<uint8_t*>(r.ptr());

		SDL_Surface* cursor_surface = SDL_CreateRGBSurfaceFrom(pr, w, h, 32, w*4, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
		if (cursor_surface == NULL) {
		  fprintf(stderr, "Creating window icon surface failed: %s", SDL_GetError());
		}

		SDL_Cursor* prev_cursor = cursors[p_shape];
		cursors[p_shape] = SDL_CreateColorCursor(cursor_surface, p_hotspot.x, p_hotspot.y);
		SDL_SetCursor(cursors[p_shape]);
		// Don't free the previous cursor set for this shape until it is no longer in use.
		if(prev_cursor) SDL_FreeCursor(cursors[p_shape]);

 		// FIXME: I can't tell from the SDL docs whether or not the surface should be freed after setting the cursor or not.
 		// I assume yes based on the implementation of SetWindowIcon, but this should be tested.
		SDL_FreeSurface(cursor_surface);
	}
}

void OS_SDL::release_rendering_thread() {

	context_gl->release_current();
}

void OS_SDL::make_rendering_thread() {

	context_gl->make_current();
}

void OS_SDL::swap_buffers() {

	context_gl->swap_buffers();
}

void OS_SDL::alert(const String &p_alert, const String &p_title) {

	const char* alert = p_alert.utf8().get_data();
	const char* title = p_title.utf8().get_data();

	SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, title, alert, sdl_window);
}

void OS_SDL::set_icon(const Ref<Image> &p_icon) {

	if (p_icon.is_valid()) {
		Ref<Image> img = p_icon->duplicate();
		img->convert(Image::FORMAT_RGBA8);

		int w = img->get_width();
		int h = img->get_height();

		PoolVector<uint8_t>::Read r = img->get_data().read();

		uint8_t *pr = const_cast<uint8_t*>(r.ptr());

		SDL_Surface* icon_surface = SDL_CreateRGBSurfaceFrom(pr, w, h, 32, w*4, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
		if (icon_surface == NULL) {
		  fprintf(stderr, "Creating window icon surface failed: %s", SDL_GetError());
		}

		SDL_SetWindowIcon(sdl_window, icon_surface);
		SDL_FreeSurface(icon_surface);

	} else {
		// FIXME: Does this actually work? Test.
		SDL_SetWindowIcon(sdl_window, NULL);
	}
}

void OS_SDL::force_process_input() {
	process_events(); // get rid of pending events
#ifdef JOYDEV_ENABLED
	joypad->process_joypads();
#endif
}

void OS_SDL::run() {

	force_quit = false;

	if (!main_loop)
		return;

	main_loop->init();

	//uint64_t last_ticks=get_ticks_usec();

	//int frames=0;
	//uint64_t frame=0;

	while (!force_quit) {

		process_events(); // get rid of pending events
#ifdef JOYDEV_ENABLED
		joypad->process_joypads();
#endif
		if (Main::iteration() == true)
			break;
	};

	main_loop->finish();
}

bool OS_SDL::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_SDL::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

void OS_SDL::_set_use_vsync(bool p_enable) {
	if (context_gl)
		return context_gl->set_use_vsync(p_enable);
}
/*
bool OS_SDL::is_vsync_enabled() const {

	if (context_gl)
		return context_gl->is_using_vsync();

	return true;
}
*/

OS::PowerState OS_SDL::get_power_state() {
	return power_manager->get_power_state();
}

int OS_SDL::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_SDL::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

void OS_SDL::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_SDL::is_disable_crash_handler() const {
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

Error OS_SDL::move_to_trash(const String &p_path) {
	String trashcan = "";
	String mnt = get_mountpoint(p_path);

	if (mnt != "") {
		String path(mnt + "/.Trash-" + itos(getuid()) + "/files");
		struct stat s;
		if (!stat(path.utf8().get_data(), &s)) {
			trashcan = path;
		}
	}

	if (trashcan == "") {
		char *dhome = getenv("XDG_DATA_HOME");
		if (dhome) {
			trashcan = String(dhome) + "/Trash/files";
		}
	}

	if (trashcan == "") {
		char *home = getenv("HOME");
		if (home) {
			trashcan = String(home) + "/.local/share/Trash/files";
		}
	}

	if (trashcan == "") {
		ERR_PRINTS("move_to_trash: Could not determine trashcan location");
		return FAILED;
	}

	List<String> args;
	args.push_back("-p");
	args.push_back(trashcan);
	Error err = execute("mkdir", args, true);
	if (err == OK) {
		List<String> args2;
		args2.push_back(p_path);
		args2.push_back(trashcan);
		err = execute("mv", args2, true);
	}

	return err;
}

// This could probably be done in a less eye-offending fashion.
OS::LatinKeyboardVariant OS_SDL::get_latin_keyboard_variant() const {
	SDL_Keycode keysFromScanCodes[6] = {
		SDL_GetKeyFromScancode(SDL_SCANCODE_Q),
		SDL_GetKeyFromScancode(SDL_SCANCODE_W),
		SDL_GetKeyFromScancode(SDL_SCANCODE_E),
		SDL_GetKeyFromScancode(SDL_SCANCODE_R),
		SDL_GetKeyFromScancode(SDL_SCANCODE_T),
		SDL_GetKeyFromScancode(SDL_SCANCODE_Y)
	};

	if (
		keysFromScanCodes[0] == SDLK_q &&
		keysFromScanCodes[1] == SDLK_w &&
		keysFromScanCodes[2] == SDLK_e &&
		keysFromScanCodes[3] == SDLK_r &&
		keysFromScanCodes[4] == SDLK_t &&
		keysFromScanCodes[5] == SDLK_z
	) return LATIN_KEYBOARD_QWERTZ;

	if (
		keysFromScanCodes[0] == SDLK_a &&
		keysFromScanCodes[1] == SDLK_z &&
		keysFromScanCodes[2] == SDLK_e &&
		keysFromScanCodes[3] == SDLK_r &&
		keysFromScanCodes[4] == SDLK_t &&
		keysFromScanCodes[5] == SDLK_y
	) return LATIN_KEYBOARD_AZERTY;

	if (
		keysFromScanCodes[0] == SDLK_QUOTE &&
		keysFromScanCodes[1] == SDLK_COMMA &&
		keysFromScanCodes[2] == SDLK_PERIOD &&
		keysFromScanCodes[3] == SDLK_p &&
		keysFromScanCodes[4] == SDLK_y &&
		keysFromScanCodes[5] == SDLK_f
	) return LATIN_KEYBOARD_DVORAK;

	if (
		keysFromScanCodes[0] == SDLK_x &&
		keysFromScanCodes[1] == SDLK_v &&
		keysFromScanCodes[2] == SDLK_l &&
		keysFromScanCodes[3] == SDLK_c &&
		keysFromScanCodes[4] == SDLK_w &&
		keysFromScanCodes[5] == SDLK_k
	) return LATIN_KEYBOARD_NEO;

	if (
		keysFromScanCodes[0] == SDLK_q &&
		keysFromScanCodes[1] == SDLK_w &&
		keysFromScanCodes[2] == SDLK_f &&
		keysFromScanCodes[3] == SDLK_p &&
		keysFromScanCodes[4] == SDLK_g &&
		keysFromScanCodes[5] == SDLK_j
	) return LATIN_KEYBOARD_COLEMAK;

	return LATIN_KEYBOARD_QWERTY;
}

OS_SDL::OS_SDL() {

#ifdef PULSEAUDIO_ENABLED
	AudioDriverManager::add_driver(&driver_pulseaudio);
#endif

#ifdef ALSA_ENABLED
	AudioDriverManager::add_driver(&driver_alsa);
#endif

	minimized = false;
	// xim_style = 0L;
	mouse_mode = MOUSE_MODE_VISIBLE;
}
