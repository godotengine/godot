/*************************************************************************/
/*  os_iphone.cpp                                                        */
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
#ifdef IPHONE_ENABLED

#include "os_iphone.h"

#include "drivers/gles3/rasterizer_gles3.h"
#include "servers/visual/visual_server_raster.h"
//#include "servers/visual/visual_server_wrap_mt.h"

#include "audio_driver_iphone.h"
#include "main/main.h"

#include "core/global_config.h"
#include "core/io/file_access_pack.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"

#include "sem_iphone.h"

#include "ios.h"

int OSIPhone::get_video_driver_count() const {

	return 1;
};

const char *OSIPhone::get_video_driver_name(int p_driver) const {

	return "GLES2";
};

OSIPhone *OSIPhone::get_singleton() {

	return (OSIPhone *)OS::get_singleton();
};

OS::VideoMode OSIPhone::get_default_video_mode() const {

	return video_mode;
};

uint8_t OSIPhone::get_orientations() const {

	return supported_orientations;
};

extern int gl_view_base_fb; // from gl_view.mm

void OSIPhone::set_data_dir(String p_dir) {

	DirAccess *da = DirAccess::open(p_dir);

	data_dir = da->get_current_dir();
	printf("setting data dir to %ls from %ls\n", data_dir.c_str(), p_dir.c_str());
	memdelete(da);
};

void OSIPhone::set_unique_ID(String p_ID) {

	unique_ID = p_ID;
};

String OSIPhone::get_unique_ID() const {

	return unique_ID;
};

void OSIPhone::initialize_core() {

	OS_Unix::initialize_core();
	SemaphoreIphone::make_default();
};

void OSIPhone::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	supported_orientations = 0;
	supported_orientations |= ((GLOBAL_DEF("video_mode/allow_horizontal", true) ? 1 : 0) << LandscapeLeft);
	supported_orientations |= ((GLOBAL_DEF("video_mode/allow_horizontal_flipped", false) ? 1 : 0) << LandscapeRight);
	supported_orientations |= ((GLOBAL_DEF("video_mode/allow_vertical", false) ? 1 : 0) << PortraitDown);
	supported_orientations |= ((GLOBAL_DEF("video_mode/allow_vertical_flipped", false) ? 1 : 0) << PortraitUp);

	RasterizerGLES3::register_config();
	RasterizerGLES3::make_current();
	RasterizerStorageGLES3::system_fbo = gl_view_base_fb;

	visual_server = memnew(VisualServerRaster());
	/*
		FIXME: Reimplement threaded rendering? Or remove?
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, false));
	};
	*/

	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	audio_driver = memnew(AudioDriverIphone);
	audio_driver->set_singleton();
	audio_driver->init();

	// init physics servers
	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	//physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server = Physics2DServerWrapMT::init_server<Physics2DServerSW>();
	physics_2d_server->init();

	input = memnew(InputDefault);

/*
#ifdef IOS_SCORELOOP_ENABLED
	scoreloop = memnew(ScoreloopIOS);
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("Scoreloop", scoreloop));
	scoreloop->connect();
#endif
	*/

#ifdef GAME_CENTER_ENABLED
	game_center = memnew(GameCenter);
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("GameCenter", game_center));
	game_center->connect();
#endif

#ifdef STOREKIT_ENABLED
	store_kit = memnew(InAppStore);
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("InAppStore", store_kit));
#endif

#ifdef ICLOUD_ENABLED
	icloud = memnew(ICloud);
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("ICloud", icloud));
//icloud->connect();
#endif
	GlobalConfig::get_singleton()->add_singleton(GlobalConfig::Singleton("iOS", memnew(iOS)));
};

MainLoop *OSIPhone::get_main_loop() const {

	return main_loop;
};

void OSIPhone::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;

	if (main_loop) {
		input->set_main_loop(p_main_loop);
		main_loop->init();
	}
};

bool OSIPhone::iterate() {

	if (!main_loop)
		return true;

	if (main_loop) {
		for (int i = 0; i < event_count; i++) {

			input->parse_input_event(event_queue[i]);
		};
	};
	event_count = 0;

	return Main::iteration();
};

void OSIPhone::key(uint32_t p_key, bool p_pressed) {

	InputEvent ev;
	ev.type = InputEvent::KEY;
	ev.key.echo = false;
	ev.key.pressed = p_pressed;
	ev.key.scancode = p_key;
	ev.key.unicode = p_key;
	queue_event(ev);
};

void OSIPhone::mouse_button(int p_idx, int p_x, int p_y, bool p_pressed, bool p_doubleclick, bool p_use_as_mouse) {

	if (!GLOBAL_DEF("debug/disable_touch", false)) {
		InputEvent ev;
		ev.type = InputEvent::SCREEN_TOUCH;
		ev.screen_touch.index = p_idx;
		ev.screen_touch.pressed = p_pressed;
		ev.screen_touch.x = p_x;
		ev.screen_touch.y = p_y;
		queue_event(ev);
	};

	mouse_list.pressed[p_idx] = p_pressed;

	if (p_use_as_mouse) {

		InputEvent ev;
		ev.type = InputEvent::MOUSE_BUTTON;
		ev.device = 0;
		ev.mouse_button.pointer_index = p_idx;

		// swaped it for tilted screen
		//ev.mouse_button.x = ev.mouse_button.global_x = video_mode.height - p_y;
		//ev.mouse_button.y = ev.mouse_button.global_y = p_x;
		ev.mouse_button.x = ev.mouse_button.global_x = p_x;
		ev.mouse_button.y = ev.mouse_button.global_y = p_y;

		//mouse_list.pressed[p_idx] = p_pressed;

		input->set_mouse_pos(Point2(ev.mouse_motion.x, ev.mouse_motion.y));
		ev.mouse_button.button_index = BUTTON_LEFT;
		ev.mouse_button.doubleclick = p_doubleclick;
		ev.mouse_button.pressed = p_pressed;

		queue_event(ev);
	};
};

void OSIPhone::mouse_move(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, bool p_use_as_mouse) {

	if (!GLOBAL_DEF("debug/disable_touch", false)) {

		InputEvent ev;
		ev.type = InputEvent::SCREEN_DRAG;
		ev.screen_drag.index = p_idx;
		ev.screen_drag.x = p_x;
		ev.screen_drag.y = p_y;
		ev.screen_drag.relative_x = p_x - p_prev_x;
		ev.screen_drag.relative_y = p_y - p_prev_y;
		queue_event(ev);
	};

	if (p_use_as_mouse) {
		InputEvent ev;
		ev.type = InputEvent::MOUSE_MOTION;
		ev.device = 0;
		ev.mouse_motion.pointer_index = p_idx;

		if (true) { // vertical

			ev.mouse_motion.x = ev.mouse_button.global_x = p_x;
			ev.mouse_motion.y = ev.mouse_button.global_y = p_y;
			ev.mouse_motion.relative_x = ev.mouse_motion.x - p_prev_x;
			ev.mouse_motion.relative_y = ev.mouse_motion.y - p_prev_y;

		} else { // horizontal?
			ev.mouse_motion.x = ev.mouse_button.global_x = video_mode.height - p_y;
			ev.mouse_motion.y = ev.mouse_button.global_y = p_x;
			ev.mouse_motion.relative_x = ev.mouse_motion.x - (video_mode.height - p_prev_x);
			ev.mouse_motion.relative_y = ev.mouse_motion.y - p_prev_x;
		};

		input->set_mouse_pos(Point2(ev.mouse_motion.x, ev.mouse_motion.y));
		ev.mouse_motion.speed_x = input->get_last_mouse_speed().x;
		ev.mouse_motion.speed_y = input->get_last_mouse_speed().y;
		ev.mouse_motion.button_mask = 1; // pressed

		queue_event(ev);
	};
};

void OSIPhone::queue_event(const InputEvent &p_event) {

	ERR_FAIL_INDEX(event_count, MAX_EVENTS);

	event_queue[event_count++] = p_event;
};

void OSIPhone::touches_cancelled() {

	for (int i = 0; i < MAX_MOUSE_COUNT; i++) {

		if (mouse_list.pressed[i]) {

			// send a mouse_up outside the screen
			mouse_button(i, -1, -1, false, false, false);
		};
	};
};

static const float ACCEL_RANGE = 1;

void OSIPhone::update_gravity(float p_x, float p_y, float p_z) {
	input->set_gravity(Vector3(p_x, p_y, p_z));
};

void OSIPhone::update_accelerometer(float p_x, float p_y, float p_z) {

	// Found out the Z should not be negated! Pass as is!
	input->set_accelerometer(Vector3(p_x / (float)ACCEL_RANGE, p_y / (float)ACCEL_RANGE, p_z / (float)ACCEL_RANGE));

	/*
	if (p_x != last_accel.x) {
		//printf("updating accel x %f\n", p_x);
		InputEvent ev;
		ev.type = InputEvent::JOYPAD_MOTION;
		ev.device = 0;
		ev.joy_motion.axis = JOY_ANALOG_0;
		ev.joy_motion.axis_value = (p_x / (float)ACCEL_RANGE);
		last_accel.x = p_x;
		queue_event(ev);
	};
	if (p_y != last_accel.y) {
		//printf("updating accel y %f\n", p_y);
		InputEvent ev;
		ev.type = InputEvent::JOYPAD_MOTION;
		ev.device = 0;
		ev.joy_motion.axis = JOY_ANALOG_1;
		ev.joy_motion.axis_value = (p_y / (float)ACCEL_RANGE);
		last_accel.y = p_y;
		queue_event(ev);
	};
	if (p_z != last_accel.z) {
		//printf("updating accel z %f\n", p_z);
		InputEvent ev;
		ev.type = InputEvent::JOYPAD_MOTION;
		ev.device = 0;
		ev.joy_motion.axis = JOY_ANALOG_2;
		ev.joy_motion.axis_value = ( (1.0 - p_z) / (float)ACCEL_RANGE);
		last_accel.z = p_z;
		queue_event(ev);
	};
	*/
};

void OSIPhone::update_magnetometer(float p_x, float p_y, float p_z) {
	input->set_magnetometer(Vector3(p_x, p_y, p_z));
};

void OSIPhone::update_gyroscope(float p_x, float p_y, float p_z) {
	input->set_gyroscope(Vector3(p_x, p_y, p_z));
};

void OSIPhone::delete_main_loop() {

	if (main_loop) {
		main_loop->finish();
		memdelete(main_loop);
	};

	main_loop = NULL;
};

void OSIPhone::finalize() {

	if (main_loop) // should not happen?
		memdelete(main_loop);

	visual_server->finish();
	memdelete(visual_server);
	//	memdelete(rasterizer);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);

	memdelete(input);
};

void OSIPhone::set_mouse_show(bool p_show){};
void OSIPhone::set_mouse_grab(bool p_grab){};

bool OSIPhone::is_mouse_grab_enabled() const {

	return true;
};

Point2 OSIPhone::get_mouse_pos() const {

	return Point2();
};

int OSIPhone::get_mouse_button_state() const {

	return mouse_list.pressed[0];
};

void OSIPhone::set_window_title(const String &p_title){};

void OSIPhone::set_video_mode(const VideoMode &p_video_mode, int p_screen) {

	video_mode = p_video_mode;
};

OS::VideoMode OSIPhone::get_video_mode(int p_screen) const {

	return video_mode;
};

void OSIPhone::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	p_list->push_back(video_mode);
};

bool OSIPhone::can_draw() const {

	if (native_video_is_playing())
		return false;
	return true;
};

int OSIPhone::set_base_framebuffer(int p_fb) {

	RasterizerStorageGLES3::system_fbo = gl_view_base_fb;

	return 0;
};

bool OSIPhone::has_virtual_keyboard() const {
	return true;
};

extern void _show_keyboard(String p_existing);
extern void _hide_keyboard();
extern Error _shell_open(String p_uri);
extern void _set_keep_screen_on(bool p_enabled);

void OSIPhone::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect) {
	_show_keyboard(p_existing_text);
};

void OSIPhone::hide_virtual_keyboard() {
	_hide_keyboard();
};

Error OSIPhone::shell_open(String p_uri) {
	return _shell_open(p_uri);
};

void OSIPhone::set_keep_screen_on(bool p_enabled) {
	OS::set_keep_screen_on(p_enabled);
	_set_keep_screen_on(p_enabled);
};

void OSIPhone::set_cursor_shape(CursorShape p_shape){

};

String OSIPhone::get_data_dir() const {

	return data_dir;
};

String OSIPhone::get_name() {

	return "iOS";
};

Size2 OSIPhone::get_window_size() const {

	return Vector2(video_mode.width, video_mode.height);
}

bool OSIPhone::has_touchscreen_ui_hint() const {

	return true;
}

void OSIPhone::set_locale(String p_locale) {
	locale_code = p_locale;
}

String OSIPhone::get_locale() const {
	return locale_code;
}

extern bool _play_video(String p_path, float p_volume, String p_audio_track, String p_subtitle_track);
extern bool _is_video_playing();
extern void _pause_video();
extern void _unpause_video();
extern void _stop_video();
extern void _focus_out_video();

Error OSIPhone::native_video_play(String p_path, float p_volume, String p_audio_track, String p_subtitle_track) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	bool exists = f && f->is_open();

	String tempFile = get_data_dir();
	if (!exists)
		return FAILED;

	if (p_path.begins_with("res://")) {
		if (PackedData::get_singleton()->has_path(p_path)) {
			print("Unable to play %S using the native player as it resides in a .pck file\n", p_path.c_str());
			return ERR_INVALID_PARAMETER;
		} else {
			p_path = p_path.replace("res:/", GlobalConfig::get_singleton()->get_resource_path());
		}
	} else if (p_path.begins_with("user://"))
		p_path = p_path.replace("user:/", get_data_dir());

	memdelete(f);

	print("Playing video: %S\n", p_path.c_str());
	if (_play_video(p_path, p_volume, p_audio_track, p_subtitle_track))
		return OK;
	return FAILED;
}

bool OSIPhone::native_video_is_playing() const {
	return _is_video_playing();
}

void OSIPhone::native_video_pause() {
	if (native_video_is_playing())
		_pause_video();
}

void OSIPhone::native_video_unpause() {
	_unpause_video();
};

void OSIPhone::native_video_focus_out() {
	_focus_out_video();
};

void OSIPhone::native_video_stop() {
	if (native_video_is_playing())
		_stop_video();
}

OSIPhone::OSIPhone(int width, int height) {

	main_loop = NULL;
	visual_server = NULL;

	VideoMode vm;
	vm.fullscreen = true;
	vm.width = width;
	vm.height = height;
	vm.resizable = false;
	set_video_mode(vm);
	event_count = 0;
};

OSIPhone::~OSIPhone() {
}

#endif
