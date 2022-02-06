/*************************************************************************/
/*  uikit_os.mm                                                          */
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

#include "uikit_os.h"

#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"

#include "main/main.h"

#include "core/io/file_access_pack.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"
#include "drivers/unix/syslog_logger.h"

#import <UIKit/UIKit.h>
#include <dlfcn.h>
#import <sys/utsname.h>

extern int gl_view_base_fb; // from gl_view.mm
extern bool gles3_available; // from gl_view.mm

int OS_UIKit::get_video_driver_count() const {
	return 2;
}

const char *OS_UIKit::get_video_driver_name(int p_driver) const {
	switch (p_driver) {
		case VIDEO_DRIVER_GLES3:
			return "GLES3";
		case VIDEO_DRIVER_GLES2:
			return "GLES2";
	}
	ERR_FAIL_V_MSG(NULL, "Invalid video driver index: " + itos(p_driver) + ".");
}

OS_UIKit *OS_UIKit::get_singleton() {
	return (OS_UIKit *)OS::get_singleton();
}

void OS_UIKit::set_data_dir(String p_dir) {
	DirAccess *da = DirAccess::open(p_dir);

	data_dir = da->get_current_dir();
	printf("setting data dir to %ls from %ls\n", data_dir.c_str(), p_dir.c_str());
	memdelete(da);
}

String OS_UIKit::get_unique_id() const {
	NSString *uuid = [UIDevice currentDevice].identifierForVendor.UUIDString;
	return String::utf8([uuid UTF8String]);
}

void OS_UIKit::initialize_core() {
	OS_Unix::initialize_core();

	set_data_dir(data_dir);
}

int OS_UIKit::get_current_video_driver() const {
	return video_driver_index;
}

void OS_UIKit::start() {
	Main::start();

	if (uikit_joypad) {
		uikit_joypad->start_processing();
	}
}

Error OS_UIKit::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	bool use_gl3 = GLOBAL_GET("rendering/quality/driver/driver_name") == "GLES3";
	bool gl_initialization_error = false;

	while (true) {
		if (use_gl3) {
			if (RasterizerGLES3::is_viable() == OK && gles3_available) {
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2")) {
					p_video_driver = VIDEO_DRIVER_GLES2;
					use_gl3 = false;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		} else {
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
		OS::get_singleton()->alert("Your device does not support any of the supported OpenGL versions.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;
	visual_server = memnew(VisualServerRaster);
	// FIXME: Reimplement threaded rendering
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, false));
	}

	visual_server->init();
	//visual_server->cursor_set_visible(false, 0);

	// reset this to what it should be, it will have been set to 0 after visual_server->init() is called
	if (use_gl3) {
		RasterizerStorageGLES3::system_fbo = gl_view_base_fb;
	} else {
		RasterizerStorageGLES2::system_fbo = gl_view_base_fb;
	}

	AudioDriverManager::initialize(p_audio_driver);

	input = memnew(InputDefault);

	uikit_joypad = memnew(UIKitJoypad);

	return OK;
};

MainLoop *OS_UIKit::get_main_loop() const {
	return main_loop;
};

void OS_UIKit::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;

	if (main_loop) {
		input->set_main_loop(p_main_loop);
		main_loop->init();
	}
};

bool OS_UIKit::iterate() {
	if (!main_loop) {
		return true;
	}

	return Main::iteration();
};

void OS_UIKit::key(uint32_t p_key, bool p_pressed) {
	Ref<InputEventKey> ev;
	ev.instance();
	ev->set_echo(false);
	ev->set_pressed(p_pressed);
	ev->set_scancode(p_key);
	ev->set_physical_scancode(p_key);
	ev->set_unicode(p_key);
	perform_event(ev);
};

void OS_UIKit::pencil_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick) {
	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_button_index(1);
	ev->set_pressed(p_pressed);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_global_position(Vector2(p_x, p_y));
	ev->set_doubleclick(p_doubleclick);
	perform_event(ev);
};

void OS_UIKit::pencil_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, float p_force) {
	Ref<InputEventMouseMotion> ev;
	ev.instance();
	ev->set_pressure(p_force);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_global_position(Vector2(p_x, p_y));
	ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
	perform_event(ev);
};

void OS_UIKit::pencil_cancelled(int p_idx) {
	pencil_press(p_idx, -1, -1, false, false);
}

void OS_UIKit::touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick) {
	if (GLOBAL_DEF("debug/disable_touch", false)) {
		return;
	}

	Ref<InputEventScreenTouch> ev;
	ev.instance();

	ev->set_index(p_idx);
	ev->set_pressed(p_pressed);
	ev->set_position(Vector2(p_x, p_y));
	perform_event(ev);
};

void OS_UIKit::touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y) {
	if (GLOBAL_DEF("debug/disable_touch", false)) {
		return;
	}

	Ref<InputEventScreenDrag> ev;
	ev.instance();
	ev->set_index(p_idx);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
	perform_event(ev);
}

void OS_UIKit::perform_event(const Ref<InputEvent> &p_event) {
	input->parse_input_event(p_event);
}

void OS_UIKit::touches_cancelled(int p_idx) {
	touch_press(p_idx, -1, -1, false, false);
}

static const float ACCEL_RANGE = 1;

void OS_UIKit::update_gravity(float p_x, float p_y, float p_z) {
	input->set_gravity(Vector3(p_x, p_y, p_z));
};

void OS_UIKit::update_accelerometer(float p_x, float p_y, float p_z) {
	// Found out the Z should not be negated! Pass as is!
	input->set_accelerometer(Vector3(p_x / (float)ACCEL_RANGE, p_y / (float)ACCEL_RANGE, p_z / (float)ACCEL_RANGE));
};

void OS_UIKit::update_magnetometer(float p_x, float p_y, float p_z) {
	input->set_magnetometer(Vector3(p_x, p_y, p_z));
};

void OS_UIKit::update_gyroscope(float p_x, float p_y, float p_z) {
	input->set_gyroscope(Vector3(p_x, p_y, p_z));
};

int OS_UIKit::get_unused_joy_id() {
	return input->get_unused_joy_id();
}

int OS_UIKit::joy_id_for_name(const String &p_name) {
	return uikit_joypad->joy_id_for_name(p_name);
}

void OS_UIKit::joy_connection_changed(int p_idx, bool p_connected, String p_name) {
	input->joy_connection_changed(p_idx, p_connected, p_name);
}

void OS_UIKit::joy_button(int p_device, int p_button, bool p_pressed) {
	input->joy_button(p_device, p_button, p_pressed);
}

void OS_UIKit::joy_axis(int p_device, int p_axis, float p_value) {
	input->joy_axis(p_device, p_axis, p_value);
}

void OS_UIKit::delete_main_loop() {
	if (main_loop) {
		main_loop->finish();
		memdelete(main_loop);
	};

	main_loop = NULL;
}

void OS_UIKit::finalize() {
	delete_main_loop();

	if (uikit_joypad) {
		memdelete(uikit_joypad);
	}

	if (input) {
		memdelete(input);
	}

	visual_server->finish();
	memdelete(visual_server);
	//    memdelete(rasterizer);
}

void OS_UIKit::set_mouse_show(bool p_show) {
	// Not supported for iOS or tvOS
}

void OS_UIKit::set_mouse_grab(bool p_grab) {
	// Not supported for iOS or tvOS
}

bool OS_UIKit::is_mouse_grab_enabled() const {
	// Not supported for iOS or tvOS
	return true;
}

Point2 OS_UIKit::get_mouse_position() const {
	// Not supported for iOS or tvOS
	return Point2();
}

int OS_UIKit::get_mouse_button_state() const {
	// Not supported for iOS or tvOS
	return 0;
}

void OS_UIKit::set_window_title(const String &p_title) {
	// Not supported for iOS or tvOS
}

// MARK: Dynamic Libraries

Error OS_UIKit::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	if (p_path.length() == 0) {
		p_library_handle = RTLD_SELF;
		return OK;
	}
	return OS_Unix::open_dynamic_library(p_path, p_library_handle, p_also_set_library_path);
}

Error OS_UIKit::close_dynamic_library(void *p_library_handle) {
	if (p_library_handle == RTLD_SELF) {
		return OK;
	}
	return OS_Unix::close_dynamic_library(p_library_handle);
}

void OS_UIKit::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
	video_mode = p_video_mode;
}

OS::VideoMode OS_UIKit::get_video_mode(int p_screen) const {
	return video_mode;
}

void OS_UIKit::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
	p_list->push_back(video_mode);
}

void OS_UIKit::set_offscreen_gl_context(EAGLContext *p_context) {
	offscreen_gl_context = p_context;
}

bool OS_UIKit::is_offscreen_gl_available() const {
	return offscreen_gl_context;
}

void OS_UIKit::set_offscreen_gl_current(bool p_current) {
	if (p_current) {
		[EAGLContext setCurrentContext:offscreen_gl_context];
	} else {
		[EAGLContext setCurrentContext:nil];
	}
}

bool OS_UIKit::can_draw() const {
	if (native_video_is_playing()) {
		return false;
	}

	return true;
}

int OS_UIKit::set_base_framebuffer(int p_fb) {
	// gl_view_base_fb has not been updated yet
	RasterizerStorageGLES3::system_fbo = p_fb;

	return 0;
}

Error OS_UIKit::shell_open(String p_uri) {
	NSString *urlPath = [[NSString alloc] initWithUTF8String:p_uri.utf8().get_data()];
	NSURL *url = [NSURL URLWithString:urlPath];

	if (![[UIApplication sharedApplication] canOpenURL:url]) {
		return ERR_CANT_OPEN;
	}

	printf("opening url %s\n", p_uri.utf8().get_data());

	[[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];

	return OK;
}

void OS_UIKit::set_keep_screen_on(bool p_enabled) {
	OS::set_keep_screen_on(p_enabled);
	[UIApplication sharedApplication].idleTimerDisabled = p_enabled;
};

String OS_UIKit::get_user_data_dir() const {
	return data_dir;
}

String OS_UIKit::get_name() const {
	return "UIKit";
}

String OS_UIKit::get_cache_path() const {
	return cache_dir;
}

Size2 OS_UIKit::get_window_size() const {
	return Vector2(video_mode.width, video_mode.height);
}

String OS_UIKit::get_locale() const {
	NSString *preferedLanguage = [NSLocale preferredLanguages].firstObject;

	if (preferedLanguage) {
		return String::utf8([preferedLanguage UTF8String]).replace("-", "_");
	}

	NSString *localeIdentifier = [[NSLocale currentLocale] localeIdentifier];
	return String::utf8([localeIdentifier UTF8String]).replace("-", "_");
}

OS_UIKit::OS_UIKit(String p_data_dir, String p_cache_dir) {
	main_loop = NULL;
	visual_server = NULL;
	offscreen_gl_context = NULL;

	// can't call set_data_dir from here, since it requires DirAccess
	// which is initialized in initialize_core
	data_dir = p_data_dir;
	cache_dir = p_cache_dir;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(SyslogLogger));
#ifdef DEBUG_ENABLED
	// it seems iOS app's stdout/stderr is only obtainable if you launch it from Xcode
	loggers.push_back(memnew(StdLogger));
#endif
	_set_logger(memnew(CompositeLogger(loggers)));

	AudioDriverManager::add_driver(&audio_driver);
};

OS_UIKit::~OS_UIKit() {
}
