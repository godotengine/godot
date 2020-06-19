/*************************************************************************/
/*  os_javascript.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/debugger/engine_debugger.h"
#include "core/io/file_access_buffered_fa.h"
#include "core/io/json.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "modules/modules_enabled.gen.h"
#include "platform/javascript/display_server_javascript.h"

#ifdef MODULE_WEBSOCKET_ENABLED
#include "modules/websocket/remote_debugger_peer_websocket.h"
#endif

#include <emscripten.h>
#include <stdlib.h>

<<<<<<< HEAD
=======
#include "dom_keys.inc"

#define DOM_BUTTON_LEFT 0
#define DOM_BUTTON_MIDDLE 1
#define DOM_BUTTON_RIGHT 2
#define DOM_BUTTON_XBUTTON1 3
#define DOM_BUTTON_XBUTTON2 4
#define GODOT_CANVAS_SELECTOR "#canvas"

// Window (canvas)

static void focus_canvas() {

	/* clang-format off */
	EM_ASM(
		Module.canvas.focus();
	);
	/* clang-format on */
}

static bool is_canvas_focused() {

	/* clang-format off */
	return EM_ASM_INT_V(
		return document.activeElement == Module.canvas;
	);
	/* clang-format on */
}

static Point2 compute_position_in_canvas(int x, int y) {
	int canvas_x = EM_ASM_INT({
		return document.getElementById('canvas').getBoundingClientRect().x;
	});
	int canvas_y = EM_ASM_INT({
		return document.getElementById('canvas').getBoundingClientRect().y;
	});
	int canvas_width;
	int canvas_height;
	emscripten_get_canvas_element_size(GODOT_CANVAS_SELECTOR, &canvas_width, &canvas_height);

	double element_width;
	double element_height;
	emscripten_get_element_css_size(GODOT_CANVAS_SELECTOR, &element_width, &element_height);

	return Point2((int)(canvas_width / element_width * (x - canvas_x)),
			(int)(canvas_height / element_height * (y - canvas_y)));
}

static bool cursor_inside_canvas = true;

EM_BOOL OS_JavaScript::fullscreen_change_callback(int p_event_type, const EmscriptenFullscreenChangeEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
	// Empty ID is canvas.
	String target_id = String::utf8(p_event->id);
	if (target_id.empty() || target_id == "canvas") {
		// This event property is the only reliable data on
		// browser fullscreen state.
		os->video_mode.fullscreen = p_event->isFullscreen;
		if (os->video_mode.fullscreen) {
			os->entering_fullscreen = false;
		} else {
			// Restoring maximized window now will cause issues,
			// so delay until main_loop_iterate.
			os->just_exited_fullscreen = true;
		}
	}
	return false;
}

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
	if (video_mode.fullscreen) {
		window_maximized = false;
		set_window_fullscreen(false);
	} else {
		if (window_maximized) {
			emscripten_exit_soft_fullscreen();
			window_maximized = false;
		}
		emscripten_set_canvas_element_size(GODOT_CANVAS_SELECTOR, p_size.x, p_size.y);
	}
}

Size2 OS_JavaScript::get_window_size() const {

	int canvas[2];
	emscripten_get_canvas_element_size(GODOT_CANVAS_SELECTOR, canvas, canvas + 1);
	return Size2(canvas[0], canvas[1]);
}

void OS_JavaScript::set_window_maximized(bool p_enabled) {

	if (video_mode.fullscreen) {
		window_maximized = p_enabled;
		set_window_fullscreen(false);
	} else if (!p_enabled) {
		emscripten_exit_soft_fullscreen();
		window_maximized = false;
	} else if (!window_maximized) {
		// Prevent calling emscripten_enter_soft_fullscreen mutltiple times,
		// this would hide page elements permanently.
		EmscriptenFullscreenStrategy strategy;
		strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
		strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
		strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
		strategy.canvasResizedCallback = NULL;
		emscripten_enter_soft_fullscreen(GODOT_CANVAS_SELECTOR, &strategy);
		window_maximized = p_enabled;
	}
}

bool OS_JavaScript::is_window_maximized() const {

	return window_maximized;
}

void OS_JavaScript::set_window_fullscreen(bool p_enabled) {

	if (p_enabled == video_mode.fullscreen) {
		return;
	}

	// Just request changes here, if successful, logic continues in
	// fullscreen_change_callback.
	if (p_enabled) {
		if (window_maximized) {
			// Soft fullsreen during real fullscreen can cause issues, so exit.
			// This must be called before requesting full screen.
			emscripten_exit_soft_fullscreen();
		}
		EmscriptenFullscreenStrategy strategy;
		strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
		strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
		strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
		strategy.canvasResizedCallback = NULL;
		EMSCRIPTEN_RESULT result = emscripten_request_fullscreen_strategy(GODOT_CANVAS_SELECTOR, false, &strategy);
		ERR_FAIL_COND_MSG(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED, "Enabling fullscreen is only possible from an input callback for the HTML5 platform.");
		ERR_FAIL_COND_MSG(result != EMSCRIPTEN_RESULT_SUCCESS, "Enabling fullscreen is only possible from an input callback for the HTML5 platform.");
		// Not fullscreen yet, so prevent "windowed" canvas dimensions from
		// being overwritten.
		entering_fullscreen = true;
	} else {
		// No logic allowed here, since exiting w/ ESC key won't use this function.
		ERR_FAIL_COND(emscripten_exit_fullscreen() != EMSCRIPTEN_RESULT_SUCCESS);
	}
}

bool OS_JavaScript::is_window_fullscreen() const {

	return video_mode.fullscreen;
}

void OS_JavaScript::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {

	Size2 screen = get_screen_size();
	p_list->push_back(OS::VideoMode(screen.width, screen.height, true));
}

bool OS_JavaScript::get_window_per_pixel_transparency_enabled() const {
	if (!is_layered_allowed()) {
		return false;
	}
	return transparency_enabled;
}

void OS_JavaScript::set_window_per_pixel_transparency_enabled(bool p_enabled) {
	if (!is_layered_allowed()) {
		return;
	}
	transparency_enabled = p_enabled;
}

// Keys

template <typename T>
static void dom2godot_mod(T *emscripten_event_ptr, Ref<InputEventWithModifiers> godot_event) {

	godot_event->set_shift(emscripten_event_ptr->shiftKey);
	godot_event->set_alt(emscripten_event_ptr->altKey);
	godot_event->set_control(emscripten_event_ptr->ctrlKey);
	godot_event->set_metakey(emscripten_event_ptr->metaKey);
}

static Ref<InputEventKey> setup_key_event(const EmscriptenKeyboardEvent *emscripten_event) {

	Ref<InputEventKey> ev;
	ev.instance();
	ev->set_echo(emscripten_event->repeat);
	dom2godot_mod(emscripten_event, ev);
	ev->set_scancode(dom_code2godot_scancode(emscripten_event->code, emscripten_event->key));

	String unicode = String::utf8(emscripten_event->key);
	// Check if empty or multi-character (e.g. `CapsLock`).
	if (unicode.length() != 1) {
		// Might be empty as well, but better than nonsense.
		unicode = String::utf8(emscripten_event->charValue);
	}
	if (unicode.length() == 1) {
		ev->set_unicode(unicode[0]);
	}

	return ev;
}

EM_BOOL OS_JavaScript::keydown_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
	Ref<InputEventKey> ev = setup_key_event(p_event);
	ev->set_pressed(true);
	if (ev->get_unicode() == 0 && keycode_has_unicode(ev->get_scancode())) {
		// Defer to keypress event for legacy unicode retrieval.
		os->deferred_key_event = ev;
		// Do not suppress keypress event.
		return false;
	}
	os->input->parse_input_event(ev);
	// Resume audio context after input in case autoplay was denied.
	os->audio_driver_javascript.resume();
	return true;
}

EM_BOOL OS_JavaScript::keypress_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
	os->deferred_key_event->set_unicode(p_event->charCode);
	os->input->parse_input_event(os->deferred_key_event);
	return true;
}

EM_BOOL OS_JavaScript::keyup_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {

	Ref<InputEventKey> ev = setup_key_event(p_event);
	ev->set_pressed(false);
	get_singleton()->input->parse_input_event(ev);
	return ev->get_scancode() != KEY_UNKNOWN && ev->get_scancode() != 0;
}

// Mouse

Point2 OS_JavaScript::get_mouse_position() const {

	return input->get_mouse_position();
}

int OS_JavaScript::get_mouse_button_state() const {

	return input->get_mouse_button_mask();
}

EM_BOOL OS_JavaScript::mouse_button_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();

	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_pressed(p_event_type == EMSCRIPTEN_EVENT_MOUSEDOWN);
	ev->set_position(compute_position_in_canvas(p_event->clientX, p_event->clientY));
	ev->set_global_position(ev->get_position());
	dom2godot_mod(p_event, ev);

	switch (p_event->button) {
		case DOM_BUTTON_LEFT: ev->set_button_index(BUTTON_LEFT); break;
		case DOM_BUTTON_MIDDLE: ev->set_button_index(BUTTON_MIDDLE); break;
		case DOM_BUTTON_RIGHT: ev->set_button_index(BUTTON_RIGHT); break;
		case DOM_BUTTON_XBUTTON1: ev->set_button_index(BUTTON_XBUTTON1); break;
		case DOM_BUTTON_XBUTTON2: ev->set_button_index(BUTTON_XBUTTON2); break;
		default: return false;
	}

	if (ev->is_pressed()) {

		double diff = emscripten_get_now() - os->last_click_ms;

		if (ev->get_button_index() == os->last_click_button_index) {

			if (diff < 400 && Point2(os->last_click_pos).distance_to(ev->get_position()) < 5) {

				os->last_click_ms = 0;
				os->last_click_pos = Point2(-100, -100);
				os->last_click_button_index = -1;
				ev->set_doubleclick(true);
			}

		} else {
			os->last_click_button_index = ev->get_button_index();
		}

		if (!ev->is_doubleclick()) {
			os->last_click_ms += diff;
			os->last_click_pos = ev->get_position();
		}
	}

	int mask = os->input->get_mouse_button_mask();
	int button_flag = 1 << (ev->get_button_index() - 1);
	if (ev->is_pressed()) {
		// Since the event is consumed, focus manually. The containing iframe,
		// if exists, may not have focus yet, so focus even if already focused.
		focus_canvas();
		mask |= button_flag;
	} else if (mask & button_flag) {
		mask &= ~button_flag;
	} else {
		// Received release event, but press was outside the canvas, so ignore.
		return false;
	}
	ev->set_button_mask(mask);

	os->input->parse_input_event(ev);
	// Resume audio context after input in case autoplay was denied.
	os->audio_driver_javascript.resume();
	// Prevent multi-click text selection and wheel-click scrolling anchor.
	// Context menu is prevented through contextmenu event.
	return true;
}

EM_BOOL OS_JavaScript::mousemove_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();

	int input_mask = os->input->get_mouse_button_mask();
	Point2 pos = compute_position_in_canvas(p_event->clientX, p_event->clientY);
	// For motion outside the canvas, only read mouse movement if dragging
	// started inside the canvas; imitating desktop app behaviour.
	if (!cursor_inside_canvas && !input_mask)
		return false;

	Ref<InputEventMouseMotion> ev;
	ev.instance();
	dom2godot_mod(p_event, ev);
	ev->set_button_mask(input_mask);

	ev->set_position(pos);
	ev->set_global_position(ev->get_position());

	ev->set_relative(Vector2(p_event->movementX, p_event->movementY));
	os->input->set_mouse_position(ev->get_position());
	ev->set_speed(os->input->get_last_mouse_speed());

	os->input->parse_input_event(ev);
	// Don't suppress mouseover/-leave events.
	return false;
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

static void set_css_cursor(const char *p_cursor) {

	/* clang-format off */
	EM_ASM_({
		Module.canvas.style.cursor = UTF8ToString($0);
	}, p_cursor);
	/* clang-format on */
}

static bool is_css_cursor_hidden() {

	/* clang-format off */
	return EM_ASM_INT({
		return Module.canvas.style.cursor === 'none';
	});
	/* clang-format on */
}

void OS_JavaScript::set_cursor_shape(CursorShape p_shape) {

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (get_mouse_mode() == MOUSE_MODE_VISIBLE) {
		if (cursors[p_shape] != "") {
			Vector<String> url = cursors[p_shape].split("?");
			set_css_cursor(("url(\"" + url[0] + "\") " + url[1] + ", auto").utf8());
		} else {
			set_css_cursor(godot2dom_cursor(p_shape));
		}
	}

	cursor_shape = p_shape;
}

void OS_JavaScript::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {

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
			if (image.is_valid()) {
				image->duplicate();
			}
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

		image = image->duplicate();

		if (atlas_texture.is_valid())
			image->crop_from_point(
					atlas_rect.position.x,
					atlas_rect.position.y,
					texture_size.width,
					texture_size.height);

		if (image->get_format() != Image::FORMAT_RGBA8) {
			image->convert(Image::FORMAT_RGBA8);
		}

		png_image png_meta;
		memset(&png_meta, 0, sizeof png_meta);
		png_meta.version = PNG_IMAGE_VERSION;
		png_meta.width = texture_size.width;
		png_meta.height = texture_size.height;
		png_meta.format = PNG_FORMAT_RGBA;

		PoolByteArray png;
		size_t len;
		PoolByteArray::Read r = image->get_data().read();
		ERR_FAIL_COND(!png_image_write_get_memory_size(png_meta, len, 0, r.ptr(), 0, NULL));

		png.resize(len);
		PoolByteArray::Write w = png.write();
		ERR_FAIL_COND(!png_image_write_to_memory(&png_meta, w.ptr(), &len, 0, r.ptr(), 0, NULL));
		w = PoolByteArray::Write();

		r = png.read();

		char *object_url;
		/* clang-format off */
		EM_ASM({
			var PNG_PTR = $0;
			var PNG_LEN = $1;
			var PTR = $2;

			var png = new Blob([HEAPU8.slice(PNG_PTR, PNG_PTR + PNG_LEN)], { type: 'image/png' });
			var url = URL.createObjectURL(png);
			var length_bytes = lengthBytesUTF8(url) + 1;
			var string_on_wasm_heap = _malloc(length_bytes);
			setValue(PTR, string_on_wasm_heap, '*');
			stringToUTF8(url, string_on_wasm_heap, length_bytes);
		}, r.ptr(), len, &object_url);
		/* clang-format on */
		r = PoolByteArray::Read();

		String url = String::utf8(object_url) + "?" + itos(p_hotspot.x) + " " + itos(p_hotspot.y);

		/* clang-format off */
		EM_ASM({ _free($0); }, object_url);
		/* clang-format on */

		if (cursors[p_shape] != "") {
			/* clang-format off */
			EM_ASM({
				URL.revokeObjectURL(UTF8ToString($0).split('?')[0]);
			}, cursors[p_shape].utf8().get_data());
			/* clang-format on */
			cursors[p_shape] = "";
		}

		cursors[p_shape] = url;

		Vector<Variant> params;
		params.push_back(p_cursor);
		params.push_back(p_hotspot);
		cursors_cache.insert(p_shape, params);

	} else if (cursors[p_shape] != "") {
		/* clang-format off */
		EM_ASM({
			URL.revokeObjectURL(UTF8ToString($0).split('?')[0]);
		}, cursors[p_shape].utf8().get_data());
		/* clang-format on */
		cursors[p_shape] = "";

		cursors_cache.erase(p_shape);
	}

	set_cursor_shape(cursor_shape);
}

void OS_JavaScript::set_mouse_mode(OS::MouseMode p_mode) {

	ERR_FAIL_COND_MSG(p_mode == MOUSE_MODE_CONFINED, "MOUSE_MODE_CONFINED is not supported for the HTML5 platform.");
	if (p_mode == get_mouse_mode())
		return;

	if (p_mode == MOUSE_MODE_VISIBLE) {

		// set_css_cursor must be called before set_cursor_shape to make the cursor visible
		set_css_cursor(godot2dom_cursor(cursor_shape));
		set_cursor_shape(cursor_shape);
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_HIDDEN) {

		set_css_cursor("none");
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_CAPTURED) {

		EMSCRIPTEN_RESULT result = emscripten_request_pointerlock("canvas", false);
		ERR_FAIL_COND_MSG(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
		ERR_FAIL_COND_MSG(result != EMSCRIPTEN_RESULT_SUCCESS, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
		// set_css_cursor must be called before set_cursor_shape to make the cursor visible
		set_css_cursor(godot2dom_cursor(cursor_shape));
		set_cursor_shape(cursor_shape);
	}
}

OS::MouseMode OS_JavaScript::get_mouse_mode() const {

	if (is_css_cursor_hidden())
		return MOUSE_MODE_HIDDEN;

	EmscriptenPointerlockChangeEvent ev;
	emscripten_get_pointerlock_status(&ev);
	return (ev.isActive && String::utf8(ev.id) == "canvas") ? MOUSE_MODE_CAPTURED : MOUSE_MODE_VISIBLE;
}

// Wheel

EM_BOOL OS_JavaScript::wheel_callback(int p_event_type, const EmscriptenWheelEvent *p_event, void *p_user_data) {

	ERR_FAIL_COND_V(p_event_type != EMSCRIPTEN_EVENT_WHEEL, false);
	if (!is_canvas_focused()) {
		if (cursor_inside_canvas) {
			focus_canvas();
		} else {
			return false;
		}
	}

	InputDefault *input = get_singleton()->input;
	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_position(input->get_mouse_position());
	ev->set_global_position(ev->get_position());

	ev->set_shift(input->is_key_pressed(KEY_SHIFT));
	ev->set_alt(input->is_key_pressed(KEY_ALT));
	ev->set_control(input->is_key_pressed(KEY_CONTROL));
	ev->set_metakey(input->is_key_pressed(KEY_META));

	if (p_event->deltaY < 0)
		ev->set_button_index(BUTTON_WHEEL_UP);
	else if (p_event->deltaY > 0)
		ev->set_button_index(BUTTON_WHEEL_DOWN);
	else if (p_event->deltaX > 0)
		ev->set_button_index(BUTTON_WHEEL_LEFT);
	else if (p_event->deltaX < 0)
		ev->set_button_index(BUTTON_WHEEL_RIGHT);
	else
		return false;

	// Different browsers give wildly different delta values, and we can't
	// interpret deltaMode, so use default value for wheel events' factor.

	int button_flag = 1 << (ev->get_button_index() - 1);

	ev->set_pressed(true);
	ev->set_button_mask(input->get_mouse_button_mask() | button_flag);
	input->parse_input_event(ev);

	ev->set_pressed(false);
	ev->set_button_mask(input->get_mouse_button_mask() & ~button_flag);
	input->parse_input_event(ev);

	return true;
}

// Touch

>>>>>>> master
bool OS_JavaScript::has_touchscreen_ui_hint() const {
	/* clang-format off */
	return EM_ASM_INT_V(
		return 'ontouchstart' in window;
	);
	/* clang-format on */
}

// Audio

int OS_JavaScript::get_audio_driver_count() const {
	return 1;
}

const char *OS_JavaScript::get_audio_driver_name(int p_driver) const {
	return "JavaScript";
}

// Lifecycle
void OS_JavaScript::initialize() {
	OS_Unix::initialize_core();
<<<<<<< HEAD
	FileAccess::make_default<FileAccessBufferedFA<FileAccessUnix>>(FileAccess::ACCESS_RESOURCES);
	DisplayServerJavaScript::register_javascript_driver();
=======
	FileAccess::make_default<FileAccessBufferedFA<FileAccessUnix> >(FileAccess::ACCESS_RESOURCES);
}

Error OS_JavaScript::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	EmscriptenWebGLContextAttributes attributes;
	emscripten_webgl_init_context_attributes(&attributes);
	attributes.alpha = GLOBAL_GET("display/window/per_pixel_transparency/allowed");
	attributes.antialias = false;
	ERR_FAIL_INDEX_V(p_video_driver, VIDEO_DRIVER_MAX, ERR_INVALID_PARAMETER);

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

	bool gles3 = true;
	if (p_video_driver == VIDEO_DRIVER_GLES2) {
		gles3 = false;
	}

	bool gl_initialization_error = false;

	while (true) {
		if (gles3) {
			if (RasterizerGLES3::is_viable() == OK) {
				attributes.majorVersion = 2;
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2")) {
					p_video_driver = VIDEO_DRIVER_GLES2;
					gles3 = false;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		} else {
			if (RasterizerGLES2::is_viable() == OK) {
				attributes.majorVersion = 1;
				RasterizerGLES2::register_config();
				RasterizerGLES2::make_current();
				break;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context(GODOT_CANVAS_SELECTOR, &attributes);
	if (emscripten_webgl_make_context_current(ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
		gl_initialization_error = true;
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your browser does not support any of the supported WebGL versions.\n"
								   "Please update your browser version.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;
>>>>>>> master

#ifdef MODULE_WEBSOCKET_ENABLED
	EngineDebugger::register_uri_handler("ws://", RemoteDebuggerPeerWebSocket::create);
	EngineDebugger::register_uri_handler("wss://", RemoteDebuggerPeerWebSocket::create);
#endif

	char locale_ptr[16];
	/* clang-format off */
	EM_ASM({
		stringToUTF8(Module['locale'], $0, 16);
	}, locale_ptr);
	/* clang-format on */
	setenv("LANG", locale_ptr, true);
}

<<<<<<< HEAD
void OS_JavaScript::resume_audio() {
	audio_driver_javascript.resume();
=======
	AudioDriverManager::initialize(p_audio_driver);
	VisualServer *visual_server = memnew(VisualServerRaster());
	input = memnew(InputDefault);

	EMSCRIPTEN_RESULT result;
#define EM_CHECK(ev)                         \
	if (result != EMSCRIPTEN_RESULT_SUCCESS) \
	ERR_PRINTS("Error while setting " #ev " callback: Code " + itos(result))
#define SET_EM_CALLBACK(target, ev, cb)                               \
	result = emscripten_set_##ev##_callback(target, NULL, true, &cb); \
	EM_CHECK(ev)
#define SET_EM_CALLBACK_NOTARGET(ev, cb)                      \
	result = emscripten_set_##ev##_callback(NULL, true, &cb); \
	EM_CHECK(ev)
	// These callbacks from Emscripten's html5.h suffice to access most
	// JavaScript APIs. For APIs that are not (sufficiently) exposed, EM_ASM
	// is used below.
	SET_EM_CALLBACK(EMSCRIPTEN_EVENT_TARGET_WINDOW, mousemove, mousemove_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, mousedown, mouse_button_callback)
	SET_EM_CALLBACK(EMSCRIPTEN_EVENT_TARGET_WINDOW, mouseup, mouse_button_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, wheel, wheel_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, touchstart, touch_press_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, touchmove, touchmove_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, touchend, touch_press_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, touchcancel, touch_press_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, keydown, keydown_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, keypress, keypress_callback)
	SET_EM_CALLBACK(GODOT_CANVAS_SELECTOR, keyup, keyup_callback)
	SET_EM_CALLBACK(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, fullscreenchange, fullscreen_change_callback)
	SET_EM_CALLBACK_NOTARGET(gamepadconnected, gamepad_change_callback)
	SET_EM_CALLBACK_NOTARGET(gamepaddisconnected, gamepad_change_callback)
#undef SET_EM_CALLBACK_NOTARGET
#undef SET_EM_CALLBACK
#undef EM_CHECK

	/* clang-format off */
	EM_ASM_ARGS({
		const send_notification = cwrap('send_notification', null, ['number']);
		const notifications = arguments;
		(['mouseover', 'mouseleave', 'focus', 'blur']).forEach(function(event, index) {
			Module.canvas.addEventListener(event, send_notification.bind(null, notifications[index]));
		});
		// Clipboard
		const update_clipboard = cwrap('update_clipboard', null, ['string']);
		window.addEventListener('paste', function(evt) {
			update_clipboard(evt.clipboardData.getData('text'));
		}, true);
	},
		MainLoop::NOTIFICATION_WM_MOUSE_ENTER,
		MainLoop::NOTIFICATION_WM_MOUSE_EXIT,
		MainLoop::NOTIFICATION_WM_FOCUS_IN,
		MainLoop::NOTIFICATION_WM_FOCUS_OUT
	);
	/* clang-format on */

	visual_server->init();

	return OK;
>>>>>>> master
}

void OS_JavaScript::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
}

MainLoop *OS_JavaScript::get_main_loop() const {
	return main_loop;
}

void OS_JavaScript::main_loop_callback() {
	get_singleton()->main_loop_iterate();
}

bool OS_JavaScript::main_loop_iterate() {
	if (is_userfs_persistent() && sync_wait_time >= 0) {
		int64_t current_time = get_ticks_msec();
		int64_t elapsed_time = current_time - last_sync_check_time;
		last_sync_check_time = current_time;

		sync_wait_time -= elapsed_time;

		if (sync_wait_time < 0) {
			/* clang-format off */
			EM_ASM(
				FS.syncfs(function(error) {
					if (error) { err('Failed to save IDB file system: ' + error.message); }
				});
			);
			/* clang-format on */
		}
	}

	DisplayServer::get_singleton()->process_events();

	return Main::iteration();
}

void OS_JavaScript::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}
	main_loop = nullptr;
}

void OS_JavaScript::finalize_async() {
	finalizing = true;
	audio_driver_javascript.finish_async();
}

void OS_JavaScript::finalize() {
	delete_main_loop();
}

// Miscellaneous

Error OS_JavaScript::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex) {
	Array args;
	for (const List<String>::Element *E = p_arguments.front(); E; E = E->next()) {
		args.push_back(E->get());
	}
	String json_args = JSON::print(args);
	/* clang-format off */
	int failed = EM_ASM_INT({
		const json_args = UTF8ToString($0);
		const args = JSON.parse(json_args);
		if (Module["onExecute"]) {
			Module["onExecute"](args);
			return 0;
		}
		return 1;
	}, json_args.utf8().get_data());
	/* clang-format on */
	ERR_FAIL_COND_V_MSG(failed, ERR_UNAVAILABLE, "OS::execute() must be implemented in JavaScript via 'engine.setOnExecute' if required.");
	return OK;
}

Error OS_JavaScript::kill(const ProcessID &p_pid) {
	ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "OS::kill() is not available on the HTML5 platform.");
}

int OS_JavaScript::get_process_id() const {
	ERR_FAIL_V_MSG(0, "OS::get_process_id() is not available on the HTML5 platform.");
}

bool OS_JavaScript::_check_internal_feature_support(const String &p_feature) {
	if (p_feature == "HTML5" || p_feature == "web")
		return true;

#ifdef JAVASCRIPT_EVAL_ENABLED
	if (p_feature == "JavaScript")
		return true;
#endif

	return false;
}

String OS_JavaScript::get_executable_path() const {
	return OS::get_executable_path();
}

Error OS_JavaScript::shell_open(String p_uri) {
	// Open URI in a new tab, browser will deal with it by protocol.
	/* clang-format off */
	EM_ASM_({
		window.open(UTF8ToString($0), '_blank');
	}, p_uri.utf8().get_data());
	/* clang-format on */
	return OK;
}

String OS_JavaScript::get_name() const {
	return "HTML5";
}

bool OS_JavaScript::can_draw() const {
	return true; // Always?
}

String OS_JavaScript::get_user_data_dir() const {
	return "/userfs";
};

String OS_JavaScript::get_cache_path() const {
	return "/home/web_user/.cache";
}

String OS_JavaScript::get_config_path() const {
	return "/home/web_user/.config";
}

String OS_JavaScript::get_data_path() const {
	return "/home/web_user/.local/share";
}

void OS_JavaScript::file_access_close_callback(const String &p_file, int p_flags) {
	OS_JavaScript *os = get_singleton();
	if (os->is_userfs_persistent() && p_file.begins_with("/userfs") && p_flags & FileAccess::WRITE) {
		os->last_sync_check_time = OS::get_singleton()->get_ticks_msec();
		// Wait five seconds in case more files are about to be closed.
		os->sync_wait_time = 5000;
	}
}

void OS_JavaScript::set_idb_available(bool p_idb_available) {
	idb_available = p_idb_available;
}

bool OS_JavaScript::is_userfs_persistent() const {
	return idb_available;
}

OS_JavaScript *OS_JavaScript::get_singleton() {
	return static_cast<OS_JavaScript *>(OS::get_singleton());
}

<<<<<<< HEAD
void OS_JavaScript::initialize_joypads() {
}
=======
OS_JavaScript::OS_JavaScript(int p_argc, char *p_argv[]) {

	List<String> arguments;
	for (int i = 1; i < p_argc; i++) {
		arguments.push_back(String::utf8(p_argv[i]));
	}
	set_cmdline(p_argv[0], arguments);

	last_click_button_index = -1;
	last_click_ms = 0;
	last_click_pos = Point2(-100, -100);

	window_maximized = false;
	entering_fullscreen = false;
	just_exited_fullscreen = false;
	transparency_enabled = false;

	main_loop = NULL;

	idb_available = false;
	sync_wait_time = -1;
>>>>>>> master

OS_JavaScript::OS_JavaScript() {
	AudioDriverManager::add_driver(&audio_driver_javascript);

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
}
