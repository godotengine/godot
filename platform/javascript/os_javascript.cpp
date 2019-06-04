/*************************************************************************/
/*  os_javascript.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "os_javascript.h"
#include "core/io/file_access_buffered_fa.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"

#include "servers/visual/visual_server_raster.h"

#include "main/main.h"

#include "core/globals.h"
#include "dom_keys.h"
#include "emscripten.h"
#include "stdlib.h"

#define DOM_BUTTON_LEFT 0
#define DOM_BUTTON_MIDDLE 1
#define DOM_BUTTON_RIGHT 2

template <typename T>
static InputModifierState dom2godot_mod(T emscripten_event_ptr) {

	InputModifierState mod;
	mod.shift = emscripten_event_ptr->shiftKey;
	mod.alt = emscripten_event_ptr->altKey;
	mod.control = emscripten_event_ptr->ctrlKey;
	mod.meta = emscripten_event_ptr->metaKey;
	return mod;
}

int OS_JavaScript::get_video_driver_count() const {

	return 1;
}
const char *OS_JavaScript::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OS_JavaScript::get_default_video_mode() const {

	return OS::VideoMode();
}

int OS_JavaScript::get_audio_driver_count() const {

	return 1;
}

const char *OS_JavaScript::get_audio_driver_name(int p_driver) const {

	return "JavaScript";
}

void OS_JavaScript::initialize_core() {

	OS_Unix::initialize_core();
	FileAccess::make_default<FileAccessBufferedFA<FileAccessUnix> >(FileAccess::ACCESS_RESOURCES);
}

void OS_JavaScript::set_opengl_extensions(const char *p_gl_extensions) {

	ERR_FAIL_COND(!p_gl_extensions);
	gl_extensions = p_gl_extensions;
}

static EM_BOOL _browser_resize_callback(int event_type, const EmscriptenUiEvent *ui_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_RESIZE, false);

	OS_JavaScript *os = static_cast<OS_JavaScript *>(user_data);

	// the order in which _browser_resize_callback and
	// _fullscreen_change_callback are called is browser-dependent,
	// so try adjusting for fullscreen in both
	if (os->is_window_fullscreen() || os->is_window_maximized()) {

		OS::VideoMode vm = os->get_video_mode();
		vm.width = ui_event->windowInnerWidth;
		vm.height = ui_event->windowInnerHeight;
		os->set_video_mode(vm);
		emscripten_set_canvas_size(ui_event->windowInnerWidth, ui_event->windowInnerHeight);
	}
	return false;
}

static Size2 _windowed_size;

static EM_BOOL _fullscreen_change_callback(int event_type, const EmscriptenFullscreenChangeEvent *event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_FULLSCREENCHANGE, false);

	OS_JavaScript *os = static_cast<OS_JavaScript *>(user_data);
	String id = String::utf8(event->id);

	// empty id is canvas
	if (id.empty() || id == "canvas") {

		OS::VideoMode vm = os->get_video_mode();
		// this event property is the only reliable information on
		// browser fullscreen state
		vm.fullscreen = event->isFullscreen;

		if (event->isFullscreen) {
			vm.width = event->screenWidth;
			vm.height = event->screenHeight;
			os->set_video_mode(vm);
			emscripten_set_canvas_size(vm.width, vm.height);
		} else {
			os->set_video_mode(vm);
			if (!os->is_window_maximized()) {
				os->set_window_size(_windowed_size);
			}
		}
	}
	return false;
}

static InputDefault *_input;
static int last_button_mask;

static EM_BOOL _mousebutton_callback(int event_type, const EmscriptenMouseEvent *mouse_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_MOUSEDOWN && event_type != EMSCRIPTEN_EVENT_MOUSEUP, false);

	InputEvent ev;
	ev.type = InputEvent::MOUSE_BUTTON;
	ev.mouse_button.pressed = event_type == EMSCRIPTEN_EVENT_MOUSEDOWN;
	ev.mouse_button.global_x = ev.mouse_button.x = mouse_event->canvasX;
	ev.mouse_button.global_y = ev.mouse_button.y = mouse_event->canvasY;
	ev.mouse_button.mod = dom2godot_mod(mouse_event);

	switch (mouse_event->button) {
		case DOM_BUTTON_LEFT: ev.mouse_button.button_index = BUTTON_LEFT; break;
		case DOM_BUTTON_MIDDLE: ev.mouse_button.button_index = BUTTON_MIDDLE; break;
		case DOM_BUTTON_RIGHT: ev.mouse_button.button_index = BUTTON_RIGHT; break;
		default: return false;
	}

	ev.mouse_button.button_mask = last_button_mask;
	if (ev.mouse_button.pressed)
		ev.mouse_button.button_mask |= 1 << (ev.mouse_button.button_index - 1);
	else
		ev.mouse_button.button_mask &= ~(1 << (ev.mouse_button.button_index - 1));
	last_button_mask = ev.mouse_button.button_mask;

	_input->parse_input_event(ev);
	return true;
}

static EM_BOOL _mousemove_callback(int event_type, const EmscriptenMouseEvent *mouse_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_MOUSEMOVE, false);

	InputEvent ev;
	ev.type = InputEvent::MOUSE_MOTION;
	ev.mouse_motion.mod = dom2godot_mod(mouse_event);
	ev.mouse_motion.button_mask = last_button_mask;

	ev.mouse_motion.global_x = ev.mouse_motion.x = mouse_event->canvasX;
	ev.mouse_motion.global_y = ev.mouse_motion.y = mouse_event->canvasY;

	ev.mouse_motion.relative_x = ev.mouse_motion.x - _input->get_mouse_pos().x;
	ev.mouse_motion.relative_y = ev.mouse_motion.y - _input->get_mouse_pos().y;

	_input->set_mouse_pos(Point2(ev.mouse_motion.x, ev.mouse_motion.y));
	ev.mouse_motion.speed_x = _input->get_mouse_speed().x;
	ev.mouse_motion.speed_y = _input->get_mouse_speed().y;

	_input->parse_input_event(ev);
	return true;
}

static EM_BOOL _wheel_callback(int event_type, const EmscriptenWheelEvent *wheel_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_WHEEL, false);

	InputEvent ev;
	ev.type = InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_mask = last_button_mask;
	ev.mouse_button.global_x = ev.mouse_button.x = _input->get_mouse_pos().x;
	ev.mouse_button.global_y = ev.mouse_button.y = _input->get_mouse_pos().y;
	ev.mouse_button.mod.shift = _input->is_key_pressed(KEY_SHIFT);
	ev.mouse_button.mod.alt = _input->is_key_pressed(KEY_ALT);
	ev.mouse_button.mod.control = _input->is_key_pressed(KEY_CONTROL);
	ev.mouse_button.mod.meta = _input->is_key_pressed(KEY_META);

	if (wheel_event->deltaY < 0)
		ev.mouse_button.button_index = BUTTON_WHEEL_UP;
	else if (wheel_event->deltaY > 0)
		ev.mouse_button.button_index = BUTTON_WHEEL_DOWN;
	else if (wheel_event->deltaX > 0)
		ev.mouse_button.button_index = BUTTON_WHEEL_LEFT;
	else if (wheel_event->deltaX < 0)
		ev.mouse_button.button_index = BUTTON_WHEEL_RIGHT;
	else
		return false;

	ev.mouse_button.pressed = true;
	_input->parse_input_event(ev);

	ev.mouse_button.pressed = false;
	_input->parse_input_event(ev);

	return true;
}

static Point2 _prev_touches[32];

static EM_BOOL _touchpress_callback(int event_type, const EmscriptenTouchEvent *touch_event, void *user_data) {

	ERR_FAIL_COND_V(
			event_type != EMSCRIPTEN_EVENT_TOUCHSTART &&
					event_type != EMSCRIPTEN_EVENT_TOUCHEND &&
					event_type != EMSCRIPTEN_EVENT_TOUCHCANCEL,
			false);

	InputEvent ev;
	ev.type = InputEvent::SCREEN_TOUCH;
	int lowest_id_index = -1;
	for (int i = 0; i < touch_event->numTouches; ++i) {

		const EmscriptenTouchPoint &touch = touch_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < touch_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev.screen_touch.index = touch.identifier;
		_prev_touches[i].x = ev.screen_touch.x = touch.canvasX;
		_prev_touches[i].y = ev.screen_touch.y = touch.canvasY;
		ev.screen_touch.pressed = event_type == EMSCRIPTEN_EVENT_TOUCHSTART;

		_input->parse_input_event(ev);
	}

	if (touch_event->touches[lowest_id_index].isChanged) {

		ev.type = InputEvent::MOUSE_BUTTON;
		ev.mouse_button.mod = dom2godot_mod(touch_event);
		ev.mouse_button.button_mask = _input->get_mouse_button_mask() >> 1;
		ev.mouse_button.global_x = ev.mouse_button.x = touch_event->touches[lowest_id_index].canvasX;
		ev.mouse_button.global_y = ev.mouse_button.y = touch_event->touches[lowest_id_index].canvasY;
		ev.mouse_button.button_index = BUTTON_LEFT;
		ev.mouse_button.pressed = event_type == EMSCRIPTEN_EVENT_TOUCHSTART;

		_input->parse_input_event(ev);
	}
	return true;
}

static EM_BOOL _touchmove_callback(int event_type, const EmscriptenTouchEvent *touch_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_TOUCHMOVE, false);

	InputEvent ev;
	ev.type = InputEvent::SCREEN_DRAG;
	int lowest_id_index = -1;
	for (int i = 0; i < touch_event->numTouches; ++i) {

		const EmscriptenTouchPoint &touch = touch_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < touch_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev.screen_drag.index = touch.identifier;
		ev.screen_drag.x = touch.canvasX;
		ev.screen_drag.y = touch.canvasY;
		Point2 &prev = _prev_touches[i];
		ev.screen_drag.relative_x = touch.canvasX - prev.x;
		ev.screen_drag.relative_y = touch.canvasY - prev.y;
		prev.x = ev.screen_drag.x;
		prev.y = ev.screen_drag.y;

		_input->parse_input_event(ev);
	}

	if (touch_event->touches[lowest_id_index].isChanged) {

		ev.type = InputEvent::MOUSE_MOTION;
		ev.mouse_motion.mod = dom2godot_mod(touch_event);
		ev.mouse_motion.button_mask = _input->get_mouse_button_mask() >> 1;
		ev.mouse_motion.global_x = ev.mouse_motion.x = touch_event->touches[lowest_id_index].canvasX;
		ev.mouse_motion.global_y = ev.mouse_motion.y = touch_event->touches[lowest_id_index].canvasY;
		ev.mouse_motion.relative_x = ev.mouse_motion.x - _input->get_mouse_pos().x;
		ev.mouse_motion.relative_y = ev.mouse_motion.y - _input->get_mouse_pos().y;

		_input->set_mouse_pos(Point2(ev.mouse_motion.x, ev.mouse_motion.y));
		ev.mouse_motion.speed_x = _input->get_mouse_speed().x;
		ev.mouse_motion.speed_y = _input->get_mouse_speed().y;

		_input->parse_input_event(ev);
	}
	return true;
}

static InputEvent _setup_key_event(const EmscriptenKeyboardEvent *emscripten_event) {

	InputEvent ev;
	ev.type = InputEvent::KEY;
	ev.key.echo = emscripten_event->repeat;
	ev.key.mod = dom2godot_mod(emscripten_event);
	ev.key.scancode = dom2godot_scancode(emscripten_event->keyCode);

	String unicode = String::utf8(emscripten_event->key);
	// check if empty or multi-character (e.g. `CapsLock`)
	if (unicode.length() != 1) {
		// might be empty as well, but better than nonsense
		unicode = String::utf8(emscripten_event->charValue);
	}
	if (unicode.length() == 1) {
		ev.key.unicode = unicode[0];
	}

	return ev;
}

static InputEvent deferred_key_event;

static EM_BOOL _keydown_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYDOWN, false);

	InputEvent ev = _setup_key_event(key_event);
	ev.key.pressed = true;
	if (ev.key.unicode == 0 && keycode_has_unicode(ev.key.scancode)) {
		// defer to keypress event for legacy unicode retrieval
		deferred_key_event = ev;
		return false; // do not suppress keypress event
	}
	_input->parse_input_event(ev);
	return true;
}

static EM_BOOL _keypress_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYPRESS, false);

	deferred_key_event.key.unicode = key_event->charCode;
	_input->parse_input_event(deferred_key_event);
	return true;
}

static EM_BOOL _keyup_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYUP, false);

	InputEvent ev = _setup_key_event(key_event);
	ev.key.pressed = false;
	_input->parse_input_event(ev);
	return ev.key.scancode != KEY_UNKNOWN && ev.key.scancode != 0;
}

static EM_BOOL joy_callback_func(int p_type, const EmscriptenGamepadEvent *p_event, void *p_user) {
	OS_JavaScript *os = (OS_JavaScript *)OS::get_singleton();
	if (os) {
		return os->joy_connection_changed(p_type, p_event);
	}
	return false;
}

void OS_JavaScript::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	print_line("Init OS");

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud, use_gl2, p_desired.width, p_desired.height, p_desired.fullscreen);

	// nothing to do here, can't fulfil fullscreen request due to
	// browser security, window size is already set from HTML
	video_mode = p_desired;
	video_mode.fullscreen = false;
	_windowed_size = get_window_size();

	// find locale, emscripten only sets "C"
	char locale_ptr[16];
	/* clang-format off */
	EM_ASM_({
		var locale = "";
		if (Module.locale) {
			// best case: server-side script reads Accept-Language early and
			// defines the locale to be read here
			locale = Module.locale;
		} else {
			// no luck, use what the JS engine can tell us
			// if this turns out not compatible enough, add tests for
			// browserLanguage, systemLanguage and userLanguage
			locale = navigator.languages ? navigator.languages[0] : navigator.language;
		}
		locale = locale.split('.')[0];
		stringToUTF8(locale, $0, 16);
	}, locale_ptr);
	/* clang-format on */
	setenv("LANG", locale_ptr, true);

	print_line("Init Audio");

	AudioDriverManagerSW::add_driver(&audio_driver_javascript);

	if (true) {
		RasterizerGLES2 *rasterizer_gles22 = memnew(RasterizerGLES2(false, false, false, false));
		rasterizer_gles22->set_use_framebuffers(false); //not supported by emscripten
		if (gl_extensions)
			rasterizer_gles22->set_extensions(gl_extensions);
		rasterizer = rasterizer_gles22;
	} else {
		//		rasterizer = memnew( RasterizerGLES1(true, false) );
	}

	print_line("Init VS");

	visual_server = memnew(VisualServerRaster(rasterizer));
	visual_server->init();

	/*AudioDriverManagerSW::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManagerSW::get_driver(p_audio_driver)->init()!=OK) {

		ERR_PRINT("Initializing audio failed.");
	}*/

	print_line("Init SM");

	//sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew(AudioServerJavascript);

	print_line("Init Mixer");

	//audio_server->set_mixer_params(AudioMixerSW::INTERPOLATION_LINEAR,false);
	audio_server->init();

	print_line("Init SoundServer");

	spatial_sound_server = memnew(SpatialSoundServerSW);
	spatial_sound_server->init();

	print_line("Init SpatialSoundServer");

	spatial_sound_2d_server = memnew(SpatialSound2DServerSW);
	spatial_sound_2d_server->init();

	//
	print_line("Init Physicsserver");

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = memnew(Physics2DServerSW);
	physics_2d_server->init();

	input = memnew(InputDefault);
	_input = input;

#define EM_CHECK(ev)                         \
	if (result != EMSCRIPTEN_RESULT_SUCCESS) \
	ERR_PRINTS("Error while setting " #ev " callback: Code " + itos(result))
#define SET_EM_CALLBACK(target, ev, cb)                               \
	result = emscripten_set_##ev##_callback(target, this, true, &cb); \
	EM_CHECK(ev)
#define SET_EM_CALLBACK_NODATA(ev, cb)                        \
	result = emscripten_set_##ev##_callback(NULL, true, &cb); \
	EM_CHECK(ev)

	EMSCRIPTEN_RESULT result;
	SET_EM_CALLBACK("#canvas", mousemove, _mousemove_callback)
	SET_EM_CALLBACK("#canvas", mousedown, _mousebutton_callback)
	SET_EM_CALLBACK("#canvas", mouseup, _mousebutton_callback)
	SET_EM_CALLBACK("#canvas", wheel, _wheel_callback)
	SET_EM_CALLBACK("#canvas", touchstart, _touchpress_callback)
	SET_EM_CALLBACK("#canvas", touchmove, _touchmove_callback)
	SET_EM_CALLBACK("#canvas", touchend, _touchpress_callback)
	SET_EM_CALLBACK("#canvas", touchcancel, _touchpress_callback)
	SET_EM_CALLBACK("#window", keydown, _keydown_callback)
	SET_EM_CALLBACK("#window", keypress, _keypress_callback)
	SET_EM_CALLBACK("#window", keyup, _keyup_callback)
	SET_EM_CALLBACK(NULL, resize, _browser_resize_callback)
	SET_EM_CALLBACK(NULL, fullscreenchange, _fullscreen_change_callback)
	SET_EM_CALLBACK_NODATA(gamepadconnected, joy_callback_func)
	SET_EM_CALLBACK_NODATA(gamepaddisconnected, joy_callback_func)

#undef SET_EM_CALLBACK_NODATA
#undef SET_EM_CALLBACK
#undef EM_CHECK
}

void OS_JavaScript::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

void OS_JavaScript::delete_main_loop() {

	memdelete(main_loop);
}

void OS_JavaScript::finalize() {

	memdelete(input);
}

void OS_JavaScript::alert(const String &p_alert, const String &p_title) {

	/* clang-format off */
	EM_ASM_({
		window.alert(UTF8ToString($0));
	}, p_alert.utf8().get_data());
	/* clang-format on */
}

void OS_JavaScript::set_mouse_show(bool p_show) {

	//javascript has no mouse...
}

void OS_JavaScript::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OS_JavaScript::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}

Point2 OS_JavaScript::get_mouse_pos() const {

	return input->get_mouse_pos();
}

int OS_JavaScript::get_mouse_button_state() const {

	return last_button_mask;
}

void OS_JavaScript::set_window_title(const String &p_title) {

	/* clang-format off */
	EM_ASM_({
		document.title = UTF8ToString($0);
	}, p_title.utf8().get_data());
	/* clang-format on */
}

//interesting byt not yet
//void set_clipboard(const String& p_text);
//String get_clipboard() const;

void OS_JavaScript::set_video_mode(const VideoMode &p_video_mode, int p_screen) {

	video_mode = p_video_mode;
}

OS::VideoMode OS_JavaScript::get_video_mode(int p_screen) const {

	return video_mode;
}

Size2 OS_JavaScript::get_screen_size(int p_screen) const {

	ERR_FAIL_COND_V(p_screen != 0, Size2());

	EmscriptenFullscreenChangeEvent ev;
	EMSCRIPTEN_RESULT result = emscripten_get_fullscreen_status(&ev);
	ERR_FAIL_COND_V(result != EMSCRIPTEN_RESULT_SUCCESS, Size2());
	return Size2(ev.screenWidth, ev.screenHeight);
}

void OS_JavaScript::set_window_size(const Size2 p_size) {

	window_maximized = false;
	if (is_window_fullscreen()) {
		set_window_fullscreen(false);
	}
	_windowed_size = p_size;
	video_mode.width = p_size.x;
	video_mode.height = p_size.y;
	emscripten_set_canvas_size(p_size.x, p_size.y);
}

Size2 OS_JavaScript::get_window_size() const {

	int canvas[3];
	emscripten_get_canvas_size(canvas, canvas + 1, canvas + 2);
	return Size2(canvas[0], canvas[1]);
}

void OS_JavaScript::set_window_maximized(bool p_enabled) {

	window_maximized = p_enabled;
	if (p_enabled) {

		if (is_window_fullscreen()) {
			// _browser_resize callback will set canvas size
			set_window_fullscreen(false);
		} else {
			/* clang-format off */
			video_mode.width = EM_ASM_INT_V(return window.innerWidth);
			video_mode.height = EM_ASM_INT_V(return window.innerHeight);
			/* clang-format on */
			emscripten_set_canvas_size(video_mode.width, video_mode.height);
		}
	} else {
		set_window_size(_windowed_size);
	}
}

void OS_JavaScript::set_window_fullscreen(bool p_enable) {

	if (p_enable == is_window_fullscreen()) {
		return;
	}

	// only requesting changes here, if successful, canvas is resized in
	// _browser_resize_callback or _fullscreen_change_callback
	EMSCRIPTEN_RESULT result;
	if (p_enable) {
		/* clang-format off */
		EM_ASM(Module.requestFullscreen(false, false););
		/* clang-format on */
	} else {
		result = emscripten_exit_fullscreen();
		if (result != EMSCRIPTEN_RESULT_SUCCESS) {
			ERR_PRINTS("Failed to exit fullscreen: Code " + itos(result));
		}
	}
}

bool OS_JavaScript::is_window_fullscreen() const {

	return video_mode.fullscreen;
}

void OS_JavaScript::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	Size2 screen = get_screen_size();
	p_list->push_back(OS::VideoMode(screen.width, screen.height, true));
}

String OS_JavaScript::get_name() {

	return "HTML5";
}

MainLoop *OS_JavaScript::get_main_loop() const {

	return main_loop;
}

bool OS_JavaScript::can_draw() const {

	return true; //always?
}

void OS_JavaScript::set_cursor_shape(CursorShape p_shape) {

	//javascript really really really has no mouse.. how amazing..
}

void OS_JavaScript::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
}

void OS_JavaScript::main_loop_begin() {

	if (main_loop)
		main_loop->init();
}
bool OS_JavaScript::main_loop_iterate() {

	if (!main_loop)
		return false;

	if (time_to_save_sync >= 0) {
		int64_t newtime = get_ticks_msec();
		int64_t elapsed = newtime - last_sync_time;
		last_sync_time = newtime;

		time_to_save_sync -= elapsed;

		print_line("elapsed " + itos(elapsed) + " tts " + itos(time_to_save_sync));

		if (time_to_save_sync < 0) {
			//time to sync, for real
			/* clang-format off */
			EM_ASM(
			  FS.syncfs(function (err) {
			    assert(!err);
				console.log("Synched!");
			    //ccall('success', 'v');
			  });
			);
			/* clang-format on */
		}
	}
	process_joysticks();
	return Main::iteration();
}

void OS_JavaScript::main_loop_end() {

	if (main_loop)
		main_loop->finish();
}

void OS_JavaScript::main_loop_focusout() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	//audio_driver_javascript.set_pause(true);
}

void OS_JavaScript::main_loop_focusin() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	//audio_driver_javascript.set_pause(false);
}

void OS_JavaScript::process_accelerometer(const Vector3 &p_accelerometer) {

	input->set_accelerometer(p_accelerometer);
}

bool OS_JavaScript::has_touchscreen_ui_hint() const {

	return false; //???
}

void OS_JavaScript::main_loop_request_quit() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
}

void OS_JavaScript::reload_gfx() {

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud, use_gl2, video_mode.width, video_mode.height, video_mode.fullscreen);
	if (rasterizer)
		rasterizer->reload_vram();
}

Error OS_JavaScript::shell_open(String p_uri) {
	/* clang-format off */
	EM_ASM_({
		window.open(UTF8ToString($0), '_blank');
	}, p_uri.utf8().get_data());
	/* clang-format on */
	return OK;
}

String OS_JavaScript::get_resource_dir() const {

	return "/"; //javascript has it's own filesystem for resources inside the APK
}

String OS_JavaScript::get_data_dir() const {

	//if (get_data_dir_func)
	//	return get_data_dir_func();
	return "/userfs";
	//return Globals::get_singleton()->get_singleton_object("GodotOS")->call("get_data_dir");
};

String OS_JavaScript::get_executable_path() const {

	return String();
}

void OS_JavaScript::_close_notification_funcs(const String &p_file, int p_flags) {

	print_line("close " + p_file + " flags " + itos(p_flags));
	if (p_file.begins_with("/userfs") && p_flags & FileAccess::WRITE) {
		static_cast<OS_JavaScript *>(get_singleton())->last_sync_time = OS::get_singleton()->get_ticks_msec();
		static_cast<OS_JavaScript *>(get_singleton())->time_to_save_sync = 5000; //five seconds since last save
	}
}

void OS_JavaScript::process_joysticks() {

	int joy_count = emscripten_get_num_gamepads();
	for (int i = 0; i < joy_count; i++) {
		EmscriptenGamepadEvent state;
		emscripten_get_gamepad_status(i, &state);
		if (state.connected) {

			int num_buttons = MIN(state.numButtons, 18);
			int num_axes = MIN(state.numAxes, 8);
			for (int j = 0; j < num_buttons; j++) {

				float value = state.analogButton[j];
				if (String(state.mapping) == "standard" && (j == 6 || j == 7)) {
					InputDefault::JoyAxis jx;
					jx.min = 0;
					jx.value = value;
					last_id = input->joy_axis(last_id, i, j, jx);
				} else {
					last_id = input->joy_button(last_id, i, j, value);
				}
			}
			for (int j = 0; j < num_axes; j++) {

				InputDefault::JoyAxis jx;
				jx.min = -1;
				jx.value = state.axis[j];
				last_id = input->joy_axis(last_id, i, j, jx);
			}
		}
	}
}

bool OS_JavaScript::joy_connection_changed(int p_type, const EmscriptenGamepadEvent *p_event) {
	if (p_type == EMSCRIPTEN_EVENT_GAMEPADCONNECTED) {

		String guid = "";
		if (String(p_event->mapping) == "standard")
			guid = "Default HTML5 Gamepad";
		input->joy_connection_changed(p_event->index, true, String(p_event->id), guid);
	} else {
		input->joy_connection_changed(p_event->index, false, "");
	}
	return true;
}

bool OS_JavaScript::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_JavaScript::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

OS_JavaScript::OS_JavaScript(GFXInitFunc p_gfx_init_func, void *p_gfx_init_ud, GetDataDirFunc p_get_data_dir_func) {

	gfx_init_func = p_gfx_init_func;
	gfx_init_ud = p_gfx_init_ud;
	last_button_mask = 0;
	main_loop = NULL;
	last_id = 1;
	gl_extensions = NULL;
	rasterizer = NULL;
	window_maximized = false;

	get_data_dir_func = p_get_data_dir_func;
	FileAccessUnix::close_notification_func = _close_notification_funcs;

	time_to_save_sync = -1;
}

OS_JavaScript::~OS_JavaScript() {
}
