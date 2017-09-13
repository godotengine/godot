/*************************************************************************/
/*  os_android.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "os_android.h"

#include "core/io/file_access_buffered_fa.h"
#include "core/project_settings.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "file_access_android.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"
//#include "servers/visual/visual_server_wrap_mt.h"

#ifdef ANDROID_NATIVE_ACTIVITY
#include "dir_access_android.h"
#include "file_access_android.h"
#else
#include "dir_access_jandroid.h"
#include "file_access_jandroid.h"
#endif

int OS_Android::get_video_driver_count() const {

	return 1;
}

const char *OS_Android::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OS_Android::get_default_video_mode() const {

	return OS::VideoMode();
}

int OS_Android::get_audio_driver_count() const {

	return 1;
}

const char *OS_Android::get_audio_driver_name(int p_driver) const {

	return "Android";
}

void OS_Android::initialize_core() {

	OS_Unix::initialize_core();

#ifdef ANDROID_NATIVE_ACTIVITY

	FileAccess::make_default<FileAccessAndroid>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	//FileAccessBufferedFA<FileAccessUnix>::make_default();
	DirAccess::make_default<DirAccessAndroid>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

#else

	if (use_apk_expansion)
		FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	else {
#ifdef USE_JAVA_FILE_ACCESS
		FileAccess::make_default<FileAccessBufferedFA<FileAccessJAndroid> >(FileAccess::ACCESS_RESOURCES);
#else
		//FileAccess::make_default<FileAccessBufferedFA<FileAccessAndroid> >(FileAccess::ACCESS_RESOURCES);
		FileAccess::make_default<FileAccessAndroid>(FileAccess::ACCESS_RESOURCES);
#endif
	}
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	//FileAccessBufferedFA<FileAccessUnix>::make_default();
	if (use_apk_expansion)
		DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	else
		DirAccess::make_default<DirAccessJAndroid>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

#endif
}

void OS_Android::set_opengl_extensions(const char *p_gl_extensions) {

	ERR_FAIL_COND(!p_gl_extensions);
	gl_extensions = p_gl_extensions;
}

void OS_Android::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	use_gl2 = p_video_driver != 1;

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud, use_gl2);

	AudioDriverManager::add_driver(&audio_driver_android);

	RasterizerGLES3::register_config();
	RasterizerGLES3::make_current();

	visual_server = memnew(VisualServerRaster);
	/*	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {

		visual_server = memnew(VisualServerWrapMT(visual_server, false));
	};*/
	visual_server->init();
	//	visual_server->cursor_set_visible(false, 0);

	AudioDriverManager::initialize(p_audio_driver);

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = Physics2DServerWrapMT::init_server<Physics2DServerSW>();
	physics_2d_server->init();

	input = memnew(InputDefault);
	input->set_fallback_mapping("Default Android Gamepad");

	//power_manager = memnew(power_android);
}

void OS_Android::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

void OS_Android::delete_main_loop() {

	memdelete(main_loop);
}

void OS_Android::finalize() {

	memdelete(input);
}

void OS_Android::vprint(const char *p_format, va_list p_list, bool p_stderr) {

	__android_log_vprint(p_stderr ? ANDROID_LOG_ERROR : ANDROID_LOG_INFO, "godot", p_format, p_list);
}

void OS_Android::print(const char *p_format, ...) {

	va_list argp;
	va_start(argp, p_format);
	__android_log_vprint(ANDROID_LOG_INFO, "godot", p_format, argp);
	va_end(argp);
}

void OS_Android::alert(const String &p_alert, const String &p_title) {

	//print("ALERT: %s\n", p_alert.utf8().get_data());
	if (alert_func)
		alert_func(p_alert, p_title);
}

void OS_Android::set_mouse_show(bool p_show) {

	//android has no mouse...
}

void OS_Android::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OS_Android::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}

Point2 OS_Android::get_mouse_position() const {

	return Point2();
}

int OS_Android::get_mouse_button_state() const {

	return 0;
}
void OS_Android::set_window_title(const String &p_title) {
}

//interesting byt not yet
//void set_clipboard(const String& p_text);
//String get_clipboard() const;

void OS_Android::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_Android::get_video_mode(int p_screen) const {

	return default_videomode;
}

void OS_Android::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	p_list->push_back(default_videomode);
}

void OS_Android::set_keep_screen_on(bool p_enabled) {
	OS::set_keep_screen_on(p_enabled);

	if (set_keep_screen_on_func) {
		set_keep_screen_on_func(p_enabled);
	}
}

Size2 OS_Android::get_window_size() const {

	return Vector2(default_videomode.width, default_videomode.height);
}

String OS_Android::get_name() {

	return "Android";
}

MainLoop *OS_Android::get_main_loop() const {

	return main_loop;
}

bool OS_Android::can_draw() const {

	return true; //always?
}

void OS_Android::set_cursor_shape(CursorShape p_shape) {

	//android really really really has no mouse.. how amazing..
}

void OS_Android::main_loop_begin() {

	if (main_loop)
		main_loop->init();
}

bool OS_Android::main_loop_iterate() {

	if (!main_loop)
		return false;
	return Main::iteration();
}

void OS_Android::main_loop_end() {

	if (main_loop)
		main_loop->finish();
}

void OS_Android::main_loop_focusout() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	audio_driver_android.set_pause(true);
}

void OS_Android::main_loop_focusin() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	audio_driver_android.set_pause(false);
}

void OS_Android::process_joy_event(OS_Android::JoypadEvent p_event) {

	switch (p_event.type) {
		case JOY_EVENT_BUTTON:
			input->joy_button(p_event.device, p_event.index, p_event.pressed);
			break;
		case JOY_EVENT_AXIS:
			InputDefault::JoyAxis value;
			value.min = -1;
			value.value = p_event.value;
			input->joy_axis(p_event.device, p_event.index, value);
			break;
		case JOY_EVENT_HAT:
			input->joy_hat(p_event.device, p_event.hat);
			break;
		default:
			return;
	}
}

void OS_Android::process_event(Ref<InputEvent> p_event) {

	input->parse_input_event(p_event);
}

void OS_Android::process_touch(int p_what, int p_pointer, const Vector<TouchPos> &p_points) {

	//print_line("ev: "+itos(p_what)+" pnt: "+itos(p_pointer)+" pointc: "+itos(p_points.size()));

	switch (p_what) {
		case 0: { //gesture begin

			if (touch.size()) {
				//end all if exist
				{
					Ref<InputEventMouseButton> ev;
					ev.instance();
					ev->set_button_index(BUTTON_LEFT);
					ev->set_button_mask(BUTTON_MASK_LEFT);
					ev->set_pressed(false);
					ev->set_position(touch[0].pos);
					ev->set_global_position(touch[0].pos);
					input->parse_input_event(ev);
				}

				for (int i = 0; i < touch.size(); i++) {

					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
				}
			}

			touch.resize(p_points.size());
			for (int i = 0; i < p_points.size(); i++) {
				touch[i].id = p_points[i].id;
				touch[i].pos = p_points[i].pos;
			}

			{
				//send mouse
				Ref<InputEventMouseButton> ev;
				ev.instance();
				// ev.type = Ref<InputEvent>::MOUSE_BUTTON;
				ev->set_button_index(BUTTON_LEFT);
				ev->set_button_mask(BUTTON_MASK_LEFT);
				ev->set_pressed(true);
				ev->set_position(touch[0].pos);
				ev->set_global_position(touch[0].pos);
				input->set_mouse_position(Point2(touch[0].pos.x, touch[0].pos.y));
				last_mouse = touch[0].pos;
				input->parse_input_event(ev);
			}

			//send touch
			for (int i = 0; i < touch.size(); i++) {

				Ref<InputEventScreenTouch> ev;
				ev.instance();
				ev->set_index(touch[i].id);
				ev->set_pressed(true);
				ev->set_position(touch[i].pos);
				input->parse_input_event(ev);
			}

		} break;
		case 1: { //motion

			if (p_points.size()) {
				//send mouse, should look for point 0?
				Ref<InputEventMouseMotion> ev;
				ev.instance();
				ev->set_button_mask(BUTTON_MASK_LEFT);
				ev->set_position(p_points[0].pos);
				input->set_mouse_position(Point2(ev->get_position().x, ev->get_position().y));
				ev->set_speed(input->get_last_mouse_speed());
				ev->set_relative(p_points[0].pos - last_mouse);
				last_mouse = p_points[0].pos;
				input->parse_input_event(ev);
			}

			ERR_FAIL_COND(touch.size() != p_points.size());

			for (int i = 0; i < touch.size(); i++) {

				int idx = -1;
				for (int j = 0; j < p_points.size(); j++) {

					if (touch[i].id == p_points[j].id) {
						idx = j;
						break;
					}
				}

				ERR_CONTINUE(idx == -1);

				if (touch[i].pos == p_points[idx].pos)
					continue; //no move unncesearily

				Ref<InputEventScreenDrag> ev;
				ev.instance();
				ev->set_index(touch[i].id);
				ev->set_position(p_points[idx].pos);
				ev->set_relative(p_points[idx].pos - touch[i].pos);
				input->parse_input_event(ev);
				touch[i].pos = p_points[idx].pos;
			}

		} break;
		case 2: { //release

			if (touch.size()) {
				//end all if exist
				Ref<InputEventMouseButton> ev;
				ev.instance();
				ev->set_button_index(BUTTON_LEFT);
				ev->set_button_mask(BUTTON_MASK_LEFT);
				ev->set_pressed(false);
				ev->set_position(touch[0].pos);
				ev->set_global_position(touch[0].pos);
				input->set_mouse_position(Point2(touch[0].pos.x, touch[0].pos.y));
				input->parse_input_event(ev);

				for (int i = 0; i < touch.size(); i++) {

					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
				}
				touch.clear();
			}

		} break;
		case 3: { // add tuchi

			ERR_FAIL_INDEX(p_pointer, p_points.size());

			TouchPos tp = p_points[p_pointer];
			touch.push_back(tp);

			Ref<InputEventScreenTouch> ev;
			ev.instance();

			ev->set_index(tp.id);
			ev->set_pressed(true);
			ev->set_position(tp.pos);
			input->parse_input_event(ev);

		} break;
		case 4: {

			for (int i = 0; i < touch.size(); i++) {
				if (touch[i].id == p_pointer) {

					Ref<InputEventScreenTouch> ev;
					ev.instance();
					ev->set_index(touch[i].id);
					ev->set_pressed(false);
					ev->set_position(touch[i].pos);
					input->parse_input_event(ev);
					touch.remove(i);
					i--;
				}
			}

		} break;
	}
}

void OS_Android::process_accelerometer(const Vector3 &p_accelerometer) {

	input->set_accelerometer(p_accelerometer);
}

void OS_Android::process_magnetometer(const Vector3 &p_magnetometer) {

	input->set_magnetometer(p_magnetometer);
}

void OS_Android::process_gyroscope(const Vector3 &p_gyroscope) {

	input->set_gyroscope(p_gyroscope);
}

bool OS_Android::has_touchscreen_ui_hint() const {

	return true;
}

bool OS_Android::has_virtual_keyboard() const {

	return true;
}

void OS_Android::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect) {

	if (show_virtual_keyboard_func) {
		show_virtual_keyboard_func(p_existing_text);
	} else {

		ERR_PRINT("Virtual keyboard not available");
	};
}

void OS_Android::hide_virtual_keyboard() {

	if (hide_virtual_keyboard_func) {

		hide_virtual_keyboard_func();
	} else {

		ERR_PRINT("Virtual keyboard not available");
	};
}

void OS_Android::init_video_mode(int p_video_width, int p_video_height) {

	default_videomode.width = p_video_width;
	default_videomode.height = p_video_height;
	default_videomode.fullscreen = true;
	default_videomode.resizable = false;
}

void OS_Android::main_loop_request_go_back() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_GO_BACK_REQUEST);
}

void OS_Android::set_display_size(Size2 p_size) {

	default_videomode.width = p_size.x;
	default_videomode.height = p_size.y;
}

void OS_Android::reload_gfx() {

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud, use_gl2);
	//if (rasterizer)
	//	rasterizer->reload_vram();
}

Error OS_Android::shell_open(String p_uri) {

	if (open_uri_func)
		return open_uri_func(p_uri) ? ERR_CANT_OPEN : OK;
	return ERR_UNAVAILABLE;
}

String OS_Android::get_resource_dir() const {

	return "/"; //android has it's own filesystem for resources inside the APK
}

String OS_Android::get_locale() const {

	if (get_locale_func)
		return get_locale_func();
	return OS_Unix::get_locale();
}

String OS_Android::get_model_name() const {

	if (get_model_func)
		return get_model_func();
	return OS_Unix::get_model_name();
}

int OS_Android::get_screen_dpi(int p_screen) const {

	if (get_screen_dpi_func) {
		return get_screen_dpi_func();
	}
	return 160;
}

void OS_Android::set_need_reload_hooks(bool p_needs_them) {

	use_reload_hooks = p_needs_them;
}

String OS_Android::get_data_dir() const {

	if (data_dir_cache != String())
		return data_dir_cache;

	if (get_data_dir_func) {
		String data_dir = get_data_dir_func();

		//store current dir
		char real_current_dir_name[2048];
		getcwd(real_current_dir_name, 2048);

		//go to data dir
		chdir(data_dir.utf8().get_data());

		//get actual data dir, so we resolve potential symlink (Android 6.0+ seems to use symlink)
		char data_current_dir_name[2048];
		getcwd(data_current_dir_name, 2048);

		//cache by parsing utf8
		data_dir_cache.parse_utf8(data_current_dir_name);

		//restore original dir so we don't mess things up
		chdir(real_current_dir_name);

		return data_dir_cache;
	}

	return ".";
	//return ProjectSettings::get_singleton()->get_singleton_object("GodotOS")->call("get_data_dir");
}

void OS_Android::set_screen_orientation(ScreenOrientation p_orientation) {

	if (set_screen_orientation_func)
		set_screen_orientation_func(p_orientation);
}

String OS_Android::get_unique_id() const {

	if (get_unique_id_func)
		return get_unique_id_func();
	return OS::get_unique_id();
}

Error OS_Android::native_video_play(String p_path, float p_volume) {
	if (video_play_func)
		video_play_func(p_path);
	return OK;
}

bool OS_Android::native_video_is_playing() {
	if (video_is_playing_func)
		return video_is_playing_func();
	return false;
}

void OS_Android::native_video_pause() {
	if (video_pause_func)
		video_pause_func();
}

String OS_Android::get_system_dir(SystemDir p_dir) const {

	if (get_system_dir_func)
		return get_system_dir_func(p_dir);
	return String(".");
}

void OS_Android::native_video_stop() {
	if (video_stop_func)
		video_stop_func();
}

void OS_Android::set_context_is_16_bits(bool p_is_16) {

	//use_16bits_fbo = p_is_16;
	//if (rasterizer)
	//	rasterizer->set_force_16_bits_fbo(p_is_16);
}

void OS_Android::joy_connection_changed(int p_device, bool p_connected, String p_name) {
	return input->joy_connection_changed(p_device, p_connected, p_name, "");
}

bool OS_Android::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_Android::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

bool OS_Android::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "mobile" || p_feature == "etc" || p_feature == "etc2"; //TODO support etc2 only if GLES3 driver is selected
}

OS_Android::OS_Android(GFXInitFunc p_gfx_init_func, void *p_gfx_init_ud, OpenURIFunc p_open_uri_func, GetDataDirFunc p_get_data_dir_func, GetLocaleFunc p_get_locale_func, GetModelFunc p_get_model_func, GetScreenDPIFunc p_get_screen_dpi_func, ShowVirtualKeyboardFunc p_show_vk, HideVirtualKeyboardFunc p_hide_vk, SetScreenOrientationFunc p_screen_orient, GetUniqueIDFunc p_get_unique_id, GetSystemDirFunc p_get_sdir_func, VideoPlayFunc p_video_play_func, VideoIsPlayingFunc p_video_is_playing_func, VideoPauseFunc p_video_pause_func, VideoStopFunc p_video_stop_func, SetKeepScreenOnFunc p_set_keep_screen_on_func, AlertFunc p_alert_func, bool p_use_apk_expansion) {

	use_apk_expansion = p_use_apk_expansion;
	default_videomode.width = 800;
	default_videomode.height = 600;
	default_videomode.fullscreen = true;
	default_videomode.resizable = false;

	gfx_init_func = p_gfx_init_func;
	gfx_init_ud = p_gfx_init_ud;
	main_loop = NULL;
	gl_extensions = NULL;
	//rasterizer = NULL;
	use_gl2 = false;

	open_uri_func = p_open_uri_func;
	get_data_dir_func = p_get_data_dir_func;
	get_locale_func = p_get_locale_func;
	get_model_func = p_get_model_func;
	get_screen_dpi_func = p_get_screen_dpi_func;
	get_unique_id_func = p_get_unique_id;
	get_system_dir_func = p_get_sdir_func;

	video_play_func = p_video_play_func;
	video_is_playing_func = p_video_is_playing_func;
	video_pause_func = p_video_pause_func;
	video_stop_func = p_video_stop_func;

	show_virtual_keyboard_func = p_show_vk;
	hide_virtual_keyboard_func = p_hide_vk;

	set_screen_orientation_func = p_screen_orient;
	set_keep_screen_on_func = p_set_keep_screen_on_func;
	alert_func = p_alert_func;
	use_reload_hooks = false;
}

OS_Android::~OS_Android() {
}
