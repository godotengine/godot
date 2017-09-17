/*************************************************************************/
/*  os_javascript.cpp                                                    */
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
#include "os_javascript.h"

#include "core/io/file_access_buffered_fa.h"
#include "core/project_settings.h"
#include "dom_keys.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"

#include <emscripten.h>
#include <stdlib.h>

#define DOM_BUTTON_LEFT 0
#define DOM_BUTTON_MIDDLE 1
#define DOM_BUTTON_RIGHT 2

template <typename T>
static void dom2godot_mod(T emscripten_event_ptr, Ref<InputEventWithModifiers> godot_event) {

	godot_event->set_shift(emscripten_event_ptr->shiftKey);
	godot_event->set_alt(emscripten_event_ptr->altKey);
	godot_event->set_control(emscripten_event_ptr->ctrlKey);
	godot_event->set_metakey(emscripten_event_ptr->metaKey);
}

int OS_JavaScript::get_video_driver_count() const {

	return 1;
}

const char *OS_JavaScript::get_video_driver_name(int p_driver) const {

	return "GLES3";
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
	// The order of the fullscreen change event and the window size change
	// event varies, even within just one browser, so defer handling
	os->request_canvas_size_adjustment();
	return false;
}

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
		os->set_video_mode(vm);
		os->request_canvas_size_adjustment();
	}
	return false;
}

static InputDefault *_input;

static bool is_canvas_focused() {

	/* clang-format off */
	return EM_ASM_INT_V(
		return document.activeElement == Module.canvas;
	);
	/* clang-format on */
}

static void focus_canvas() {

	/* clang-format off */
	EM_ASM(
		Module.canvas.focus();
	);
	/* clang-format on */
}

static bool _cursor_inside_canvas = true;

static bool is_cursor_inside_canvas() {

	return _cursor_inside_canvas;
}

static EM_BOOL _mousebutton_callback(int event_type, const EmscriptenMouseEvent *mouse_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_MOUSEDOWN && event_type != EMSCRIPTEN_EVENT_MOUSEUP, false);

	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_pressed(event_type == EMSCRIPTEN_EVENT_MOUSEDOWN);
	ev->set_position(Point2(mouse_event->canvasX, mouse_event->canvasY));
	ev->set_global_position(ev->get_position());
	dom2godot_mod(mouse_event, ev);

	switch (mouse_event->button) {
		case DOM_BUTTON_LEFT: ev->set_button_index(BUTTON_LEFT); break;
		case DOM_BUTTON_MIDDLE: ev->set_button_index(BUTTON_MIDDLE); break;
		case DOM_BUTTON_RIGHT: ev->set_button_index(BUTTON_RIGHT); break;
		default: return false;
	}

	int mask = _input->get_mouse_button_mask();
	if (ev->is_pressed()) {
		// since the event is consumed, focus manually
		if (!is_canvas_focused()) {
			focus_canvas();
		}
		mask |= 1 << ev->get_button_index();
	} else if (mask & (1 << ev->get_button_index())) {
		mask &= ~(1 << ev->get_button_index());
	} else {
		// release event, but press was outside the canvas, so ignore
		return false;
	}
	ev->set_button_mask(mask >> 1);

	_input->parse_input_event(ev);
	// prevent selection dragging
	return true;
}

static EM_BOOL _mousemove_callback(int event_type, const EmscriptenMouseEvent *mouse_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_MOUSEMOVE, false);
	OS_JavaScript *os = static_cast<OS_JavaScript *>(user_data);
	int input_mask = _input->get_mouse_button_mask();
	Point2 pos = Point2(mouse_event->canvasX, mouse_event->canvasY);
	// outside the canvas, only read mouse movement if dragging started inside
	// the canvas; imitating desktop app behaviour
	if (!is_cursor_inside_canvas() && !input_mask)
		return false;

	Ref<InputEventMouseMotion> ev;
	ev.instance();
	dom2godot_mod(mouse_event, ev);
	ev->set_button_mask(input_mask >> 1);

	ev->set_position(pos);
	ev->set_global_position(ev->get_position());

	ev->set_relative(_input->get_mouse_position() - ev->get_position());
	_input->set_mouse_position(ev->get_position());
	ev->set_speed(_input->get_last_mouse_speed());

	_input->parse_input_event(ev);
	// don't suppress mouseover/leave events
	return false;
}

static EM_BOOL _wheel_callback(int event_type, const EmscriptenWheelEvent *wheel_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_WHEEL, false);
	if (!is_canvas_focused()) {
		if (is_cursor_inside_canvas()) {
			focus_canvas();
		} else {
			return false;
		}
	}

	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_button_mask(_input->get_mouse_button_mask() >> 1);
	ev->set_position(_input->get_mouse_position());
	ev->set_global_position(ev->get_position());

	ev->set_shift(_input->is_key_pressed(KEY_SHIFT));
	ev->set_alt(_input->is_key_pressed(KEY_ALT));
	ev->set_control(_input->is_key_pressed(KEY_CONTROL));
	ev->set_metakey(_input->is_key_pressed(KEY_META));

	if (wheel_event->deltaY < 0)
		ev->set_button_index(BUTTON_WHEEL_UP);
	else if (wheel_event->deltaY > 0)
		ev->set_button_index(BUTTON_WHEEL_DOWN);
	else if (wheel_event->deltaX > 0)
		ev->set_button_index(BUTTON_WHEEL_LEFT);
	else if (wheel_event->deltaX < 0)
		ev->set_button_index(BUTTON_WHEEL_RIGHT);
	else
		return false;

	// Different browsers give wildly different delta values, and we can't
	// interpret deltaMode, so use default value for wheel events' factor

	ev->set_pressed(true);
	_input->parse_input_event(ev);

	ev->set_pressed(false);
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

	Ref<InputEventScreenTouch> ev;
	ev.instance();
	int lowest_id_index = -1;
	for (int i = 0; i < touch_event->numTouches; ++i) {

		const EmscriptenTouchPoint &touch = touch_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < touch_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev->set_index(touch.identifier);
		ev->set_position(Point2(touch.canvasX, touch.canvasY));
		_prev_touches[i] = ev->get_position();
		ev->set_pressed(event_type == EMSCRIPTEN_EVENT_TOUCHSTART);

		_input->parse_input_event(ev);
	}

	if (touch_event->touches[lowest_id_index].isChanged) {

		Ref<InputEventMouseButton> ev_mouse;
		ev_mouse.instance();
		ev_mouse->set_button_mask(_input->get_mouse_button_mask() >> 1);
		dom2godot_mod(touch_event, ev_mouse);

		const EmscriptenTouchPoint &first_touch = touch_event->touches[lowest_id_index];
		ev_mouse->set_position(Point2(first_touch.canvasX, first_touch.canvasY));
		ev_mouse->set_global_position(ev_mouse->get_position());

		ev_mouse->set_button_index(BUTTON_LEFT);
		ev_mouse->set_pressed(event_type == EMSCRIPTEN_EVENT_TOUCHSTART);

		_input->parse_input_event(ev_mouse);
	}
	return true;
}

static EM_BOOL _touchmove_callback(int event_type, const EmscriptenTouchEvent *touch_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_TOUCHMOVE, false);

	Ref<InputEventScreenDrag> ev;
	ev.instance();
	int lowest_id_index = -1;
	for (int i = 0; i < touch_event->numTouches; ++i) {

		const EmscriptenTouchPoint &touch = touch_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < touch_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev->set_index(touch.identifier);
		ev->set_position(Point2(touch.canvasX, touch.canvasY));
		Point2 &prev = _prev_touches[i];
		ev->set_relative(ev->get_position() - prev);
		prev = ev->get_position();

		_input->parse_input_event(ev);
	}

	if (touch_event->touches[lowest_id_index].isChanged) {

		Ref<InputEventMouseMotion> ev_mouse;
		ev_mouse.instance();
		dom2godot_mod(touch_event, ev_mouse);
		ev_mouse->set_button_mask(_input->get_mouse_button_mask() >> 1);

		const EmscriptenTouchPoint &first_touch = touch_event->touches[lowest_id_index];
		ev_mouse->set_position(Point2(first_touch.canvasX, first_touch.canvasY));
		ev_mouse->set_global_position(ev_mouse->get_position());

		ev_mouse->set_relative(_input->get_mouse_position() - ev_mouse->get_position());
		_input->set_mouse_position(ev_mouse->get_position());
		ev_mouse->set_speed(_input->get_last_mouse_speed());

		_input->parse_input_event(ev_mouse);
	}
	return true;
}

static Ref<InputEventKey> _setup_key_event(const EmscriptenKeyboardEvent *emscripten_event) {

	Ref<InputEventKey> ev;
	ev.instance();
	ev->set_echo(emscripten_event->repeat);
	dom2godot_mod(emscripten_event, ev);
	ev->set_scancode(dom2godot_scancode(emscripten_event->keyCode));

	String unicode = String::utf8(emscripten_event->key);
	// check if empty or multi-character (e.g. `CapsLock`)
	if (unicode.length() != 1) {
		// might be empty as well, but better than nonsense
		unicode = String::utf8(emscripten_event->charValue);
	}
	if (unicode.length() == 1) {
		ev->set_unicode(unicode[0]);
	}

	return ev;
}

static Ref<InputEventKey> deferred_key_event;

static EM_BOOL _keydown_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYDOWN, false);

	Ref<InputEventKey> ev = _setup_key_event(key_event);
	ev->set_pressed(true);
	if (ev->get_unicode() == 0 && keycode_has_unicode(ev->get_scancode())) {
		// defer to keypress event for legacy unicode retrieval
		deferred_key_event = ev;
		return false; // do not suppress keypress event
	}
	_input->parse_input_event(ev);
	return true;
}

static EM_BOOL _keypress_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYPRESS, false);

	deferred_key_event->set_unicode(key_event->charCode);
	_input->parse_input_event(deferred_key_event);
	return true;
}

static EM_BOOL _keyup_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type != EMSCRIPTEN_EVENT_KEYUP, false);

	Ref<InputEventKey> ev = _setup_key_event(key_event);
	ev->set_pressed(false);
	_input->parse_input_event(ev);
	return ev->get_scancode() != KEY_UNKNOWN && ev->get_scancode() != 0;
}

static EM_BOOL joy_callback_func(int p_type, const EmscriptenGamepadEvent *p_event, void *p_user) {
	OS_JavaScript *os = (OS_JavaScript *)OS::get_singleton();
	if (os) {
		return os->joy_connection_changed(p_type, p_event);
	}
	return false;
}

extern "C" {
void send_notification(int notif) {
	if (notif == MainLoop::NOTIFICATION_WM_MOUSE_ENTER || notif == MainLoop::NOTIFICATION_WM_MOUSE_EXIT) {
		_cursor_inside_canvas = notif == MainLoop::NOTIFICATION_WM_MOUSE_ENTER;
	}
	OS_JavaScript::get_singleton()->get_main_loop()->notification(notif);
}
}

void OS_JavaScript::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	print_line("Init OS");

	EmscriptenWebGLContextAttributes attributes;
	emscripten_webgl_init_context_attributes(&attributes);
	attributes.alpha = false;
	attributes.antialias = false;
	attributes.majorVersion = 2;
	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context(NULL, &attributes);
	ERR_FAIL_COND(emscripten_webgl_make_context_current(ctx) != EMSCRIPTEN_RESULT_SUCCESS);

	video_mode = p_desired;
	// can't fulfil fullscreen request due to browser security
	video_mode.fullscreen = false;
	set_window_size(Size2(p_desired.width, p_desired.height));

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

	AudioDriverManager::add_driver(&audio_driver_javascript);
	AudioDriverManager::initialize(p_audio_driver);

	RasterizerGLES3::register_config();
	RasterizerGLES3::make_current();

	print_line("Init VS");

	visual_server = memnew(VisualServerRaster());
	//	visual_server->cursor_set_visible(false, 0);

	print_line("Init Physicsserver");

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = memnew(Physics2DServerSW);
	physics_2d_server->init();

	input = memnew(InputDefault);
	_input = input;

	power_manager = memnew(PowerJavascript);

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
	SET_EM_CALLBACK("#window", mousemove, _mousemove_callback)
	SET_EM_CALLBACK("#canvas", mousedown, _mousebutton_callback)
	SET_EM_CALLBACK("#window", mouseup, _mousebutton_callback)
	SET_EM_CALLBACK("#window", wheel, _wheel_callback)
	SET_EM_CALLBACK("#window", touchstart, _touchpress_callback)
	SET_EM_CALLBACK("#window", touchmove, _touchmove_callback)
	SET_EM_CALLBACK("#window", touchend, _touchpress_callback)
	SET_EM_CALLBACK("#window", touchcancel, _touchpress_callback)
	SET_EM_CALLBACK("#canvas", keydown, _keydown_callback)
	SET_EM_CALLBACK("#canvas", keypress, _keypress_callback)
	SET_EM_CALLBACK("#canvas", keyup, _keyup_callback)
	SET_EM_CALLBACK(NULL, resize, _browser_resize_callback)
	SET_EM_CALLBACK(NULL, fullscreenchange, _fullscreen_change_callback)
	SET_EM_CALLBACK_NODATA(gamepadconnected, joy_callback_func)
	SET_EM_CALLBACK_NODATA(gamepaddisconnected, joy_callback_func)

#undef SET_EM_CALLBACK_NODATA
#undef SET_EM_CALLBACK
#undef EM_CHECK

#ifdef JAVASCRIPT_EVAL_ENABLED
	javascript_eval = memnew(JavaScript);
	ProjectSettings::get_singleton()->add_singleton(ProjectSettings::Singleton("JavaScript", javascript_eval));
#endif

	visual_server->init();
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

static const char *godot2dom_cursor(OS::CursorShape p_shape) {

	switch (p_shape) {
		case OS::CURSOR_ARROW:
		default:
			return "auto";
		case OS::CURSOR_IBEAM: return "text";
		case OS::CURSOR_POINTING_HAND: return "pointer";
		case OS::CURSOR_CROSS: return "crosshair";
		case OS::CURSOR_WAIT: return "progress";
		case OS::CURSOR_BUSY: return "wait";
		case OS::CURSOR_DRAG: return "grab";
		case OS::CURSOR_CAN_DROP: return "grabbing";
		case OS::CURSOR_FORBIDDEN: return "no-drop";
		case OS::CURSOR_VSIZE: return "ns-resize";
		case OS::CURSOR_HSIZE: return "ew-resize";
		case OS::CURSOR_BDIAGSIZE: return "nesw-resize";
		case OS::CURSOR_FDIAGSIZE: return "nwse-resize";
		case OS::CURSOR_MOVE: return "move";
		case OS::CURSOR_VSPLIT: return "row-resize";
		case OS::CURSOR_HSPLIT: return "col-resize";
		case OS::CURSOR_HELP: return "help";
	}
}

void OS_JavaScript::set_css_cursor(const char *p_cursor) {

	/* clang-format off */
	EM_ASM_({
		Module.canvas.style.cursor = Module.UTF8ToString($0);
	}, p_cursor);
	/* clang-format on */
}

const char *OS_JavaScript::get_css_cursor() const {

	char cursor[16];
	/* clang-format off */
	EM_ASM_INT({
		Module.stringToUTF8(Module.canvas.style.cursor ? Module.canvas.style.cursor : 'auto', $0, 16);
	}, cursor);
	/* clang-format on */
	return cursor;
}

void OS_JavaScript::set_mouse_mode(OS::MouseMode p_mode) {

	ERR_FAIL_INDEX(p_mode, MOUSE_MODE_CONFINED + 1);
	ERR_EXPLAIN("MOUSE_MODE_CONFINED is not supported for the HTML5 platform");
	ERR_FAIL_COND(p_mode == MOUSE_MODE_CONFINED);
	if (p_mode == get_mouse_mode())
		return;

	if (p_mode == MOUSE_MODE_VISIBLE) {

		set_css_cursor(godot2dom_cursor(cursor_shape));
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_HIDDEN) {

		set_css_cursor("none");
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_CAPTURED) {

		EMSCRIPTEN_RESULT result = emscripten_request_pointerlock("canvas", false);
		ERR_EXPLAIN("MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback");
		ERR_FAIL_COND(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED);
		ERR_FAIL_COND(result != EMSCRIPTEN_RESULT_SUCCESS);
		set_css_cursor(godot2dom_cursor(cursor_shape));
	}
}

OS::MouseMode OS_JavaScript::get_mouse_mode() const {

	if (!strcmp(get_css_cursor(), "none"))
		return MOUSE_MODE_HIDDEN;

	EmscriptenPointerlockChangeEvent ev;
	emscripten_get_pointerlock_status(&ev);
	return ev.isActive && (strcmp(ev.id, "canvas") == 0) ? MOUSE_MODE_CAPTURED : MOUSE_MODE_VISIBLE;
}

Point2 OS_JavaScript::get_mouse_position() const {

	return input->get_mouse_position();
}

int OS_JavaScript::get_mouse_button_state() const {

	return input->get_mouse_button_mask();
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

	EmscriptenFullscreenChangeEvent ev;
	EMSCRIPTEN_RESULT result = emscripten_get_fullscreen_status(&ev);
	ERR_FAIL_COND_V(result != EMSCRIPTEN_RESULT_SUCCESS, Size2());
	return Size2(ev.screenWidth, ev.screenHeight);
}

void OS_JavaScript::set_window_size(const Size2 p_size) {

	windowed_size = p_size;
	if (is_window_fullscreen()) {
		window_maximized = false;
		set_window_fullscreen(false);
	} else if (is_window_maximized()) {
		set_window_maximized(false);
	} else {
		video_mode.width = p_size.x;
		video_mode.height = p_size.y;
		emscripten_set_canvas_size(p_size.x, p_size.y);
	}
}

Size2 OS_JavaScript::get_window_size() const {

	int canvas[3];
	emscripten_get_canvas_size(canvas, canvas + 1, canvas + 2);
	return Size2(canvas[0], canvas[1]);
}

void OS_JavaScript::set_window_maximized(bool p_enabled) {

	window_maximized = p_enabled;
	if (is_window_fullscreen()) {
		set_window_fullscreen(false);
		return;
	}
	// Calling emscripten_enter_soft_fullscreen mutltiple times hides all
	// page elements except the canvas permanently, so track state
	if (p_enabled && !soft_fs_enabled) {

		EmscriptenFullscreenStrategy strategy;
		strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
		strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
		strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
		strategy.canvasResizedCallback = NULL;
		emscripten_enter_soft_fullscreen(NULL, &strategy);
		soft_fs_enabled = true;
		video_mode.width = get_window_size().width;
		video_mode.height = get_window_size().height;
	} else if (!p_enabled) {

		emscripten_exit_soft_fullscreen();
		soft_fs_enabled = false;
		video_mode.width = windowed_size.width;
		video_mode.height = windowed_size.height;
		emscripten_set_canvas_size(video_mode.width, video_mode.height);
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
		if (window_maximized) {
			// soft fs during real fs can cause issues
			set_window_maximized(false);
			window_maximized = true;
		}
		EmscriptenFullscreenStrategy strategy;
		strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
		strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
		strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
		strategy.canvasResizedCallback = NULL;
		emscripten_request_fullscreen_strategy(NULL, false, &strategy);
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

void OS_JavaScript::request_canvas_size_adjustment() {

	canvas_size_adjustment_requested = true;
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

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	cursor_shape = p_shape;
	if (get_mouse_mode() != MOUSE_MODE_HIDDEN)
		set_css_cursor(godot2dom_cursor(cursor_shape));
}

void OS_JavaScript::main_loop_begin() {

	if (main_loop)
		main_loop->init();

	/* clang-format off */
	EM_ASM_ARGS({
		const send_notification = Module.cwrap('send_notification', null, ['number']);
		const notifs = arguments;
		(['mouseover', 'mouseleave', 'focus', 'blur']).forEach(function(event, i) {
			Module.canvas.addEventListener(event, send_notification.bind(null, notifs[i]));
		});
	},
		MainLoop::NOTIFICATION_WM_MOUSE_ENTER,
		MainLoop::NOTIFICATION_WM_MOUSE_EXIT,
		MainLoop::NOTIFICATION_WM_FOCUS_IN,
		MainLoop::NOTIFICATION_WM_FOCUS_OUT
	);
	/* clang-format on */
}

bool OS_JavaScript::main_loop_iterate() {

	if (!main_loop)
		return false;

	if (time_to_save_sync >= 0) {
		int64_t newtime = get_ticks_msec();
		int64_t elapsed = newtime - last_sync_time;
		last_sync_time = newtime;

		time_to_save_sync -= elapsed;

		if (time_to_save_sync < 0) {
			//time to sync, for real
			/* clang-format off */
			EM_ASM(
				FS.syncfs(function(err) {
					if (err) { Module.printErr('Failed to save IDB file system: ' + err.message); }
				});
			);
			/* clang-format on */
		}
	}
	process_joypads();
	if (canvas_size_adjustment_requested) {

		if (video_mode.fullscreen || window_maximized) {
			video_mode.width = get_window_size().width;
			video_mode.height = get_window_size().height;
		}
		if (!video_mode.fullscreen) {
			set_window_maximized(window_maximized);
		}
		canvas_size_adjustment_requested = false;
	}
	return Main::iteration();
}

void OS_JavaScript::main_loop_end() {

	if (main_loop)
		main_loop->finish();
}

void OS_JavaScript::process_accelerometer(const Vector3 &p_accelerometer) {

	input->set_accelerometer(p_accelerometer);
}

bool OS_JavaScript::has_touchscreen_ui_hint() const {

	/* clang-format off */
	return EM_ASM_INT_V(
		return 'ontouchstart' in window;
	);
	/* clang-format on */
}

void OS_JavaScript::main_loop_request_quit() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
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

	/*
	if (get_data_dir_func)
		return get_data_dir_func();
	*/
	return "/userfs";
	//return ProjectSettings::get_singleton()->get_singleton_object("GodotOS")->call("get_data_dir");
};

String OS_JavaScript::get_executable_path() const {

	return OS::get_executable_path();
}

void OS_JavaScript::_close_notification_funcs(const String &p_file, int p_flags) {

	print_line("close " + p_file + " flags " + itos(p_flags));
	if (p_file.begins_with("/userfs") && p_flags & FileAccess::WRITE) {
		static_cast<OS_JavaScript *>(get_singleton())->last_sync_time = OS::get_singleton()->get_ticks_msec();
		static_cast<OS_JavaScript *>(get_singleton())->time_to_save_sync = 5000; //five seconds since last save
	}
}

void OS_JavaScript::process_joypads() {

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
					input->joy_axis(i, j, jx);
				} else {
					input->joy_button(i, j, value);
				}
			}
			for (int j = 0; j < num_axes; j++) {

				InputDefault::JoyAxis jx;
				jx.min = -1;
				jx.value = state.axis[j];
				input->joy_axis(i, j, jx);
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

OS::PowerState OS_JavaScript::get_power_state() {
	return power_manager->get_power_state();
}

int OS_JavaScript::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_JavaScript::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

bool OS_JavaScript::_check_internal_feature_support(const String &p_feature) {

	return p_feature == "web" || p_feature == "s3tc"; // TODO check for these features really being available
}

OS_JavaScript::OS_JavaScript(const char *p_execpath, GetDataDirFunc p_get_data_dir_func) {
	set_cmdline(p_execpath, get_cmdline_args());
	main_loop = NULL;
	gl_extensions = NULL;
	window_maximized = false;
	soft_fs_enabled = false;
	canvas_size_adjustment_requested = false;

	get_data_dir_func = p_get_data_dir_func;
	FileAccessUnix::close_notification_func = _close_notification_funcs;

	time_to_save_sync = -1;
}

OS_JavaScript::~OS_JavaScript() {
}
