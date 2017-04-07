/*************************************************************************/
/*  os_bb10.cpp                                                          */
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
#include "os_bb10.h"

#include "bbutil.h"
#include "core/global_config.h"
#include "core/os/dir_access.h"
#include "core/os/keyboard.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"

#include <assert.h>
#include <bps/accelerometer.h>
#include <bps/audiodevice.h>
#include <bps/bps.h>
#include <bps/navigator.h>
#include <bps/orientation.h>
#include <bps/screen.h>
#include <bps/virtualkeyboard.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef BB10_SCORELOOP_ENABLED
#include "modules/scoreloop/scoreloop_bb10.h"
#endif

static char launch_dir[512];
char *launch_dir_ptr;

int OSBB10::get_video_driver_count() const {

	return 1;
}
const char *OSBB10::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OSBB10::get_default_video_mode() const {

	return OS::VideoMode();
}

int OSBB10::get_audio_driver_count() const {

	return 1;
}
const char *OSBB10::get_audio_driver_name(int p_driver) const {

	return "BB10";
}

void OSBB10::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	data_dir = getenv("HOME");

	//Create a screen context that will be used to create an EGL surface to to receive libscreen events
	screen_create_context(&screen_cxt, 0);

	//Initialize BPS library
	bps_initialize();

	//Use utility code to initialize EGL for 2D rendering with GL ES 1.1
	enum RENDERING_API api = GL_ES_2;
#ifdef BB10_LGLES_OVERRIDE
	api = GL_ES_1;
#endif
	if (EXIT_SUCCESS != bbutil_init(screen_cxt, api)) {
		bbutil_terminate();
		screen_destroy_context(screen_cxt);
		return;
	};

	EGLint surface_width, surface_height;

	eglQuerySurface(egl_disp, egl_surf, EGL_WIDTH, &surface_width);
	eglQuerySurface(egl_disp, egl_surf, EGL_HEIGHT, &surface_height);
	printf("screen size: %ix%i\n", surface_width, surface_height);
	VideoMode mode;
	mode.width = surface_width;
	mode.height = surface_height;
	mode.fullscreen = true;
	mode.resizable = false;
	set_video_mode(mode);

	//Signal BPS library that navigator and screen events will be requested
	screen_request_events(screen_cxt);
	navigator_request_events(0);
	virtualkeyboard_request_events(0);
	audiodevice_request_events(0);

#ifdef DEBUG_ENABLED
	bps_set_verbosity(3);
#endif

	accel_supported = accelerometer_is_supported();
	if (accel_supported)
		accelerometer_set_update_frequency(FREQ_40_HZ);
	pitch = 0;
	roll = 0;

#ifdef BB10_LGLES_OVERRIDE
	rasterizer = memnew(RasterizerGLES1(false));
#else
	rasterizer = memnew(RasterizerGLES2(false, false));
#endif

	visual_server = memnew(VisualServerRaster(rasterizer));
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	audio_driver = memnew(AudioDriverBB10);
	audio_driver->set_singleton();
	audio_driver->init(NULL);

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = memnew(Physics2DServerSW);
	physics_2d_server->init();

	input = memnew(InputDefault);

	power_manager = memnew(PowerBB10);

#ifdef PAYMENT_SERVICE_ENABLED
	payment_service = memnew(PaymentService);
	Globals::get_singleton()->add_singleton(Globals::Singleton("InAppStore", payment_service));
#endif
}

void OSBB10::set_main_loop(MainLoop *p_main_loop) {

	input->set_main_loop(p_main_loop);
	main_loop = p_main_loop;
}

void OSBB10::delete_main_loop() {

	memdelete(main_loop);
	main_loop = NULL;
}

void OSBB10::finalize() {

	if (main_loop)
		memdelete(main_loop);
	main_loop = NULL;

	/*
	if (debugger_connection_console) {
		memdelete(debugger_connection_console);
	}
	*/

	visual_server->finish();
	memdelete(visual_server);
	memdelete(rasterizer);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);

#ifdef PAYMENT_SERVICE_ENABLED
	memdelete(payment_service);
#endif

	memdelete(input);

	bbutil_terminate();
	screen_destroy_context(screen_cxt);

	bps_shutdown();
}

void OSBB10::set_mouse_show(bool p_show) {

	//android has no mouse...
}

void OSBB10::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OSBB10::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}
Point2 OSBB10::get_mouse_pos() const {

	return Point2();
}
int OSBB10::get_mouse_button_state() const {

	return 0;
}
void OSBB10::set_window_title(const String &p_title) {
}

//interesting byt not yet
//void set_clipboard(const String& p_text);
//String get_clipboard() const;

void OSBB10::set_video_mode(const VideoMode &p_video_mode, int p_screen) {

	default_videomode = p_video_mode;
}

OS::VideoMode OSBB10::get_video_mode(int p_screen) const {

	return default_videomode;
}
void OSBB10::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	p_list->push_back(default_videomode);
}

String OSBB10::get_name() {

	return "BlackBerry 10";
}

MainLoop *OSBB10::get_main_loop() const {

	return main_loop;
}

bool OSBB10::can_draw() const {

	return !minimized;
}

void OSBB10::set_cursor_shape(CursorShape p_shape) {

	//android really really really has no mouse.. how amazing..
}

void OSBB10::handle_screen_event(bps_event_t *event) {

	screen_event_t screen_event = screen_event_get_event(event);

	int screen_val;
	screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_TYPE, &screen_val);

	int pos[2];

	switch (screen_val) {
		case SCREEN_EVENT_MTOUCH_TOUCH:
		case SCREEN_EVENT_MTOUCH_RELEASE: {

			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_POSITION, pos);

			InputEvent ievent;
			ievent.type = InputEvent::SCREEN_TOUCH;
			ievent.device = 0;
			ievent.screen_touch.pressed = (screen_val == SCREEN_EVENT_MTOUCH_TOUCH);
			ievent.screen_touch.x = pos[0];
			ievent.screen_touch.y = pos[1];
			Point2 mpos(ievent.screen_touch.x, ievent.screen_touch.y);

			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_TOUCH_ID, &pos[0]);
			ievent.screen_touch.index = pos[0];

			last_touch_x[pos[0]] = ievent.screen_touch.x;
			last_touch_y[pos[0]] = ievent.screen_touch.y;

			input->parse_input_event(ievent);

			if (ievent.screen_touch.index == 0) {

				InputEvent ievent;
				ievent.type = InputEvent::MOUSE_BUTTON;
				ievent.device = 0;
				ievent.mouse_button.pressed = (screen_val == SCREEN_EVENT_MTOUCH_TOUCH);
				ievent.mouse_button.button_index = BUTTON_LEFT;
				ievent.mouse_button.doubleclick = 0;
				ievent.mouse_button.x = ievent.mouse_button.global_x = mpos.x;
				ievent.mouse_button.y = ievent.mouse_button.global_y = mpos.y;
				input->parse_input_event(ievent);
			};

		} break;
		case SCREEN_EVENT_MTOUCH_MOVE: {

			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_POSITION, pos);

			InputEvent ievent;
			ievent.type = InputEvent::SCREEN_DRAG;
			ievent.device = 0;
			ievent.screen_drag.x = pos[0];
			ievent.screen_drag.y = pos[1];

			/*
		screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_SOURCE_POSITION, pos);
		ievent.screen_drag.relative_x = ievent.screen_drag.x - pos[0];
		ievent.screen_drag.relative_y = ievent.screen_drag.y - pos[1];
		*/

			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_TOUCH_ID, &pos[0]);
			ievent.screen_drag.index = pos[0];

			ievent.screen_drag.relative_x = ievent.screen_drag.x - last_touch_x[ievent.screen_drag.index];
			ievent.screen_drag.relative_y = ievent.screen_drag.y - last_touch_y[ievent.screen_drag.index];

			last_touch_x[ievent.screen_drag.index] = ievent.screen_drag.x;
			last_touch_y[ievent.screen_drag.index] = ievent.screen_drag.y;

			Point2 mpos(ievent.screen_drag.x, ievent.screen_drag.y);
			Point2 mrel(ievent.screen_drag.relative_x, ievent.screen_drag.relative_y);

			input->parse_input_event(ievent);

			if (ievent.screen_touch.index == 0) {

				InputEvent ievent;
				ievent.type = InputEvent::MOUSE_MOTION;
				ievent.device = 0;
				ievent.mouse_motion.x = ievent.mouse_motion.global_x = mpos.x;
				ievent.mouse_motion.y = ievent.mouse_motion.global_y = mpos.y;
				input->set_mouse_pos(Point2(ievent.mouse_motion.x, ievent.mouse_motion.y));
				ievent.mouse_motion.speed_x = input->get_last_mouse_speed().x;
				ievent.mouse_motion.speed_y = input->get_last_mouse_speed().y;
				ievent.mouse_motion.relative_x = mrel.x;
				ievent.mouse_motion.relative_y = mrel.y;
				ievent.mouse_motion.button_mask = 1; // pressed

				input->parse_input_event(ievent);
			};
		} break;

		case SCREEN_EVENT_KEYBOARD: {

			InputEvent ievent;
			ievent.type = InputEvent::KEY;
			ievent.device = 0;
			int val = 0;
			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_KEY_SCAN, &val);
			ievent.key.scancode = val;
			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_KEY_SYM, &val);
			ievent.key.unicode = val;
			if (val == 61448) {
				ievent.key.scancode = KEY_BACKSPACE;
				ievent.key.unicode = KEY_BACKSPACE;
			};
			if (val == 61453) {
				ievent.key.scancode = KEY_ENTER;
				ievent.key.unicode = KEY_ENTER;
			};

			int flags;
			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_KEY_FLAGS, &flags);
			ievent.key.pressed = flags & 1; // bit 1 is pressed apparently

			int mod;
			screen_get_event_property_iv(screen_event, SCREEN_PROPERTY_KEY_MODIFIERS, &mod);

			input->parse_input_event(ievent);
		} break;

		default:
			break;
	}
};

void OSBB10::handle_accelerometer() {

	if (!accel_supported)
		return;

	if (!fullscreen)
		return;

	double force_x, force_y, force_z;
	accelerometer_read_forces(&force_x, &force_y, &force_z);
	Vector3 accel = Vector3(force_x, flip_accelerometer ? force_y : -force_y, force_z);
	input->set_accelerometer(accel);
	// rotate 90 degrees
	//input->set_accelerometer(Vector3(force_y, flip_accelerometer?force_x:(-force_x), force_z));
};

void OSBB10::_resize(bps_event_t *event) {

	int angle = navigator_event_get_orientation_angle(event);
	bbutil_rotate_screen_surface(angle);

	EGLint surface_width, surface_height;
	eglQuerySurface(egl_disp, egl_surf, EGL_WIDTH, &surface_width);
	eglQuerySurface(egl_disp, egl_surf, EGL_HEIGHT, &surface_height);

	VideoMode mode;
	mode.width = surface_width;
	mode.height = surface_height;
	mode.fullscreen = true;
	mode.resizable = false;
	set_video_mode(mode);
};

void OSBB10::process_events() {

	handle_accelerometer();

	bps_event_t *event = NULL;

	do {
		int rc = bps_get_event(&event, 0);
		assert(rc == BPS_SUCCESS);

		if (!event) break;

#ifdef BB10_SCORELOOP_ENABLED
		ScoreloopBB10 *sc = Globals::get_singleton()->get_singleton_object("Scoreloop")->cast_to<ScoreloopBB10>();
		if (sc->handle_event(event))
			continue;
#endif

#ifdef PAYMENT_SERVICE_ENABLED
		if (payment_service->handle_event(event))
			continue;
#endif

		int domain = bps_event_get_domain(event);
		if (domain == screen_get_domain()) {

			handle_screen_event(event);

		} else if (domain == navigator_get_domain()) {

			if (NAVIGATOR_EXIT == bps_event_get_code(event)) {
				if (main_loop)
					main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
				bps_event_destroy(event);
				exit(0);
				return;
				/*
			} else if (bps_event_get_code(event) == NAVIGATOR_ORIENTATION_CHECK) {

				int angle = navigator_event_get_orientation_angle(event);
				navigator_orientation_check_response(event, false);

			} else if (bps_event_get_code(event) == NAVIGATOR_ORIENTATION) {

				_resize(event);
			*/
			} else if (bps_event_get_code(event) == NAVIGATOR_WINDOW_STATE) {

				int state = navigator_event_get_window_state(event);
				bool was_fullscreen = fullscreen;
				minimized = state == NAVIGATOR_WINDOW_INVISIBLE;
				fullscreen = state == NAVIGATOR_WINDOW_FULLSCREEN;
				set_low_processor_usage_mode(!fullscreen);
				if (fullscreen != was_fullscreen) {
					if (fullscreen) {
						audio_server->set_fx_global_volume_scale(fullscreen_mixer_volume);
						audio_server->set_stream_global_volume_scale(fullscreen_stream_volume);
					} else {
						fullscreen_mixer_volume = audio_server->get_fx_global_volume_scale();
						fullscreen_stream_volume = audio_server->get_stream_global_volume_scale();
						audio_server->set_fx_global_volume_scale(0);
						audio_server->set_stream_global_volume_scale(0);
					};
				};
			};
		} else if (domain == audiodevice_get_domain()) {

			const char *audiodevice_path = audiodevice_event_get_path(event);
			printf("************* got audiodevice event, path %s\n", audiodevice_path);
			audio_driver->finish();
			audio_driver->init(audiodevice_path);
			audio_driver->start();
		};

		//bps_event_destroy(event);
	} while (event);
};

bool OSBB10::has_virtual_keyboard() const {

	return true;
};

void OSBB10::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect) {

	virtualkeyboard_show();
};

void OSBB10::hide_virtual_keyboard() {

	virtualkeyboard_hide();
};

void OSBB10::run() {

	if (!main_loop)
		return;

	main_loop->init();

	int flip = bbutil_is_flipped();
	int rot = bbutil_get_rotation();
	flip_accelerometer = rot == 90;
	printf("**************** rot is %i, flip %i\n", rot, (int)flip_accelerometer);
	/*
	orientation_direction_t orientation;
	int angle;
	orientation_get(&orientation, &angle);
	printf("******************** orientation %i, %i, %i\n", orientation, ORIENTATION_BOTTOM_UP, ORIENTATION_TOP_UP);
	if (orientation == ORIENTATION_BOTTOM_UP) {
		flip_accelerometer = true;
	};
	*/

	while (true) {

		process_events(); // get rid of pending events
		if (Main::iteration() == true)
			break;
		bbutil_swap();
		//#ifdef DEBUG_ENABLED
		fflush(stdout);
		//#endif
	};

	main_loop->finish();
};

bool OSBB10::has_touchscreen_ui_hint() const {

	return true;
}

Error OSBB10::shell_open(String p_uri) {

	char *msg = NULL;
	int ret = navigator_invoke(p_uri.utf8().get_data(), &msg);

	return ret == BPS_SUCCESS ? OK : FAILED;
};

String OSBB10::get_data_dir() const {

	return data_dir;
};

Size2 OSBB10::get_window_size() const {
	return Vector2(default_videomode.width, default_videomode.height);
}

PowerState OSBB10::get_power_state() {
	return power_manager->get_power_state();
}

int OSBB10::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OSBB10::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

OSBB10::OSBB10() {

	main_loop = NULL;
	minimized = false;
	fullscreen = true;
	flip_accelerometer = true;
	fullscreen_mixer_volume = 1;
	fullscreen_stream_volume = 1;

	printf("godot bb10!\n");
	getcwd(launch_dir, sizeof(launch_dir));
	printf("launch dir %s\n", launch_dir);
	chdir("app/native");
	launch_dir_ptr = launch_dir;
}

OSBB10::~OSBB10() {
}
