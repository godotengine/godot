/**************************************************************************/
/*  bridge_openharmony.cpp                                                */
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

#include "bridge_openharmony.h"

#include "dir_access_openharmony.h"
#include "file_access_openharmony.h"
#include "os_openharmony.h"

#include "core/config/project_settings.h"
#include "display_server_openharmony.h"
#include "main/main.h"

#include <native_vsync/native_vsync.h>

OS_OpenHarmony *os_openharmony = nullptr;
OH_NativeVSync *native_vsync = nullptr;
uint32_t step = 0;

Mutex godot_step_mutex;
uint32_t latest_window_width = 0;
uint32_t latest_window_height = 0;
int32_t latest_window_event = 0;

enum GodotStartupStep {
	STEP_TERMINATED = -1,
	STEP_SETUP,
	STEP_SHOW_LOGO,
	STEP_STARTED,
};

void godot_finalize() {
	if (step == STEP_TERMINATED) {
		return;
	}
	step = STEP_TERMINATED;

	if (os_openharmony) {
		os_openharmony->main_loop_end();
		Main::cleanup();
		memdelete(os_openharmony);
		os_openharmony = nullptr;
	}

	OH_NativeVSync_Destroy(native_vsync);
	native_vsync = nullptr;
}

void godot_step(long long timestamp, void *data) {
	if (step == STEP_TERMINATED) {
		return;
	}

	switch (step) {
		case STEP_SETUP:
			// Since Godot is initialized on the UI thread, main_thread_id was set to that thread's id,
			// but for Godot purposes, the main thread is the one running the game loop
			Main::setup2(false); // The logo is shown in the next frame otherwise we run into rendering issues
			step++;
			break;
		case STEP_SHOW_LOGO:
			Main::setup_boot_logo();
			step++;

			break;
		case STEP_STARTED:
			if (Main::start() != EXIT_SUCCESS) {
				return; // should exit instead and print the error
			}
			os_openharmony->main_loop_begin();
			step++;
			break;
		default:

			godot_step_mutex.lock();
			uint32_t current_window_width = latest_window_width;
			uint32_t current_window_height = latest_window_height;
			int32_t current_window_event = latest_window_event;
			latest_window_width = 0;
			latest_window_height = 0;
			latest_window_event = 0;
			godot_step_mutex.unlock();

			if (current_window_width != 0 && current_window_height != 0) {
				DisplayServerOpenHarmony::get_singleton()->resize_window(current_window_width, current_window_height);
			}
			if (latest_window_event != 0) {
				switch (current_window_event) {
					case 1: // SHOWN
					case 2: // ACTIVE
						OS_OpenHarmony::get_singleton()->on_focus_in();
						break;
					case 3: // INACTIVE
					case 4: // HIDDEN
						OS_OpenHarmony::get_singleton()->on_focus_out();
						break;
					case 5: // RESUMED
						OS_OpenHarmony::get_singleton()->on_exit_background();
						break;
					case 6: // PAUSED
						OS_OpenHarmony::get_singleton()->on_enter_background();
						break;
				}
			}

			if (os_openharmony->main_loop_iterate()) {
				// If the main loop iteration returns true, it means we should exit.
				// In this case, we do not request another frame.
				godot_finalize();
				return;
			}
			break;
	}

	// Request the next frame
	OH_NativeVSync_RequestFrame(native_vsync, godot_step, nullptr);
}

int64_t godot_init(NativeResourceManager *p_resource_manager, void *p_native_window, int32_t window_id, int64_t window_width, int64_t window_height, const char *p_allowed_permissions) {
	OHNativeWindow *window = static_cast<OHNativeWindow *>(p_native_window);

	FileAccessOpenHarmony::setup(p_resource_manager);
	DirAccessOpenHarmony::setup(p_resource_manager);
	os_openharmony = memnew(OS_OpenHarmony);
	os_openharmony->set_window_id(window_id);
	os_openharmony->set_native_window(window);
	os_openharmony->set_display_size(Size2i(window_width, window_height));
	os_openharmony->set_allowed_permissions(p_allowed_permissions);

	Vector<String> args;
	String content;
	FileAccessOpenHarmony::get_rawfile_content("_cl_", content);

	if (!content.is_empty()) {
		Vector<String> lines = content.split("\n", false);
		for (const String &line : lines) {
			String arg = line.strip_edges();
			if (!arg.is_empty()) {
				args.push_back(arg);
			}
		}
	}

	const char **cmdline = nullptr;

	if (args.size() > 0) {
		cmdline = (const char **)memalloc(args.size() * sizeof(const char *));
		for (int i = 0; i < args.size(); i++) {
			CharString cs = args[i].utf8();
			char *flag = (char *)memalloc(cs.length() + 1);
			memcpy((void *)flag, cs.get_data(), cs.length() + 1);
			flag[cs.length()] = '\0';
			cmdline[i] = flag;
		}
	}

	Error err = Main::setup(OS_OpenHarmony::EXEC_PATH, args.size(), (char **)cmdline, false);

	if (cmdline) {
		for (int i = 0; i < args.size(); i++) {
			memfree((void *)cmdline[i]);
		}
		memfree(cmdline);
	}

	if (err != OK) {
		return err;
	}

	const char *connection_name = "godot";
	native_vsync = OH_NativeVSync_Create(connection_name, strlen(connection_name));
	return OH_NativeVSync_RequestFrame(native_vsync, godot_step, nullptr);
}

void godot_touch(GodotTouchEvent *p_event, int count) {
	static Vector<GodotTouchEvent> last_touch_events;
	for (int i = 0; i < count; i++) {
		GodotTouchEvent &event = p_event[i];
		if (event.id >= last_touch_events.size()) {
			last_touch_events.resize(event.id + 1);
		}
		switch (event.type) {
			case 0: { // Touch begin
				Ref<InputEventScreenTouch> ev;
				ev.instantiate();
				ev->set_index(event.id);
				ev->set_pressed(true);
				ev->set_position(Vector2(event.x, event.y));
				Input::get_singleton()->parse_input_event(ev);
			} break;
			case 1: { // Touch up
				Ref<InputEventScreenTouch> ev;
				ev.instantiate();
				ev->set_index(event.id);
				ev->set_pressed(false);
				ev->set_position(Vector2(event.x, event.y));
				Input::get_singleton()->parse_input_event(ev);
			} break;
			case 2: { // Touch move
				Ref<InputEventScreenDrag> ev;
				ev.instantiate();
				ev->set_index(event.id);
				ev->set_position(Vector2(event.x, event.y));
				ev->set_relative(Vector2(event.x - last_touch_events[event.id].x, event.y - last_touch_events[event.id].y));
				ev->set_relative_screen_position(ev->get_relative());
				Input::get_singleton()->parse_input_event(ev);
			} break;
			case 3: { // Touch cancel
				Ref<InputEventScreenTouch> ev;
				ev.instantiate();
				ev->set_index(event.id);
				ev->set_canceled(true);
				ev->set_position(Vector2(event.x, event.y));
				Input::get_singleton()->parse_input_event(ev);
			} break;
		}
		last_touch_events.set(event.id, event);
	}
}

void godot_mouse(GodotMouseEvent *p_event) {
	static GodotMouseEvent last_mouse_event;
	GodotMouseEvent &event = *p_event;
	switch (event.type) {
		case 0: { // Mouse down
			Ref<InputEventMouseButton> ev;
			ev.instantiate();
			ev->set_pressed(true);
			ev->set_position(Vector2(event.x, event.y));
			ev->set_global_position(ev->get_position());
			ev->set_button_index(MouseButton(event.button));
			ev->set_button_mask(BitField<MouseButtonMask>(event.mask));
			Input::get_singleton()->parse_input_event(ev);
		} break;
		case 1: { // Mouse up
			Ref<InputEventMouseButton> ev;
			ev.instantiate();
			ev->set_pressed(false);
			ev->set_position(Vector2(event.x, event.y));
			ev->set_global_position(ev->get_position());
			ev->set_button_index(MouseButton(event.button));
			Input::get_singleton()->parse_input_event(ev);
		} break;
		case 2: { // Mouse move
			Ref<InputEventMouseMotion> ev;
			ev.instantiate();
			ev->set_position(Vector2(event.x, event.y));
			ev->set_global_position(ev->get_position());
			ev->set_relative(Vector2(event.x - last_mouse_event.x, event.y - last_mouse_event.y));
			ev->set_relative_screen_position(ev->get_relative());
			Input::get_singleton()->parse_input_event(ev);
		} break;
	}
	last_mouse_event = event;
}

void godot_key(GodotKeyEvent *p_event) {
	GodotKeyEvent &event = *p_event;
	Ref<InputEventKey> ev;
	ev.instantiate();
	ev->set_pressed(event.pressed);
	ev->set_echo(false);
	ev->set_keycode(Key(event.code));
	ev->set_physical_keycode(Key(event.code));
	ev->set_key_label(Key(event.code));
	ev->set_unicode(event.unicode);
	ev->set_location(KeyLocation::UNSPECIFIED);
	ev->set_alt_pressed(event.alt);
	ev->set_ctrl_pressed(event.ctrl);
	ev->set_shift_pressed(event.shift);
	ev->set_meta_pressed(event.meta);
	Input::get_singleton()->parse_input_event(ev);
}

void godot_resize(uint32_t width, uint32_t height) {
	godot_step_mutex.lock();
	latest_window_width = width;
	latest_window_height = height;
	godot_step_mutex.unlock();
}

void godot_window_event(int32_t event) {
	godot_step_mutex.lock();
	latest_window_event = event;
	godot_step_mutex.unlock();
}
