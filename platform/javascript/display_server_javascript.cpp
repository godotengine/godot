/*************************************************************************/
/*  display_server_javascript.cpp                                        */
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

#include "platform/javascript/display_server_javascript.h"

#include "drivers/dummy/rasterizer_dummy.h"
#include "platform/javascript/os_javascript.h"

#include <emscripten.h>
#include <png.h>

#include "dom_keys.inc"

#define DOM_BUTTON_LEFT 0
#define DOM_BUTTON_MIDDLE 1
#define DOM_BUTTON_RIGHT 2
#define DOM_BUTTON_XBUTTON1 3
#define DOM_BUTTON_XBUTTON2 4

char DisplayServerJavaScript::canvas_id[256] = { 0 };
static bool cursor_inside_canvas = true;

DisplayServerJavaScript *DisplayServerJavaScript::get_singleton() {
	return static_cast<DisplayServerJavaScript *>(DisplayServer::get_singleton());
}

// Window (canvas)
void DisplayServerJavaScript::focus_canvas() {
	/* clang-format off */
	EM_ASM(
		Module['canvas'].focus();
	);
	/* clang-format on */
}

bool DisplayServerJavaScript::is_canvas_focused() {
	/* clang-format off */
	return EM_ASM_INT_V(
		return document.activeElement == Module['canvas'];
	);
	/* clang-format on */
}

bool DisplayServerJavaScript::check_size_force_redraw() {
	int canvas_width;
	int canvas_height;
	emscripten_get_canvas_element_size(DisplayServerJavaScript::canvas_id, &canvas_width, &canvas_height);
	if (last_width != canvas_width || last_height != canvas_height) {
		last_width = canvas_width;
		last_height = canvas_height;
		// Update the framebuffer size for redraw.
		emscripten_set_canvas_element_size(DisplayServerJavaScript::canvas_id, canvas_width, canvas_height);
		return true;
	}
	return false;
}

Point2 DisplayServerJavaScript::compute_position_in_canvas(int p_x, int p_y) {
	int canvas_x = EM_ASM_INT({
		return Module['canvas'].getBoundingClientRect().x;
	});
	int canvas_y = EM_ASM_INT({
		return Module['canvas'].getBoundingClientRect().y;
	});
	int canvas_width;
	int canvas_height;
	emscripten_get_canvas_element_size(canvas_id, &canvas_width, &canvas_height);

	double element_width;
	double element_height;
	emscripten_get_element_css_size(canvas_id, &element_width, &element_height);

	return Point2((int)(canvas_width / element_width * (p_x - canvas_x)),
			(int)(canvas_height / element_height * (p_y - canvas_y)));
}

EM_BOOL DisplayServerJavaScript::fullscreen_change_callback(int p_event_type, const EmscriptenFullscreenChangeEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();
	// Empty ID is canvas.
	String target_id = String::utf8(p_event->id);
	String canvas_str_id = String::utf8(canvas_id);
	if (target_id.empty() || target_id == canvas_str_id) {
		// This event property is the only reliable data on
		// browser fullscreen state.
		if (p_event->isFullscreen) {
			display->window_mode = WINDOW_MODE_FULLSCREEN;
		} else {
			display->window_mode = WINDOW_MODE_WINDOWED;
		}
	}
	return false;
}

// Drag and drop callback (see native/utils.js).
extern "C" EMSCRIPTEN_KEEPALIVE void _drop_files_callback(char *p_filev[], int p_filec) {
	DisplayServerJavaScript *ds = DisplayServerJavaScript::get_singleton();
	if (!ds) {
		ERR_FAIL_MSG("Unable to drop files because the DisplayServer is not active");
	}
	if (ds->drop_files_callback.is_null())
		return;
	Vector<String> files;
	for (int i = 0; i < p_filec; i++) {
		files.push_back(String::utf8(p_filev[i]));
	}
	Variant v = files;
	Variant *vp = &v;
	Variant ret;
	Callable::CallError ce;
	ds->drop_files_callback.call((const Variant **)&vp, 1, ret, ce);
}

// Keys

template <typename T>
void DisplayServerJavaScript::dom2godot_mod(T *emscripten_event_ptr, Ref<InputEventWithModifiers> godot_event) {
	godot_event->set_shift(emscripten_event_ptr->shiftKey);
	godot_event->set_alt(emscripten_event_ptr->altKey);
	godot_event->set_control(emscripten_event_ptr->ctrlKey);
	godot_event->set_metakey(emscripten_event_ptr->metaKey);
}

Ref<InputEventKey> DisplayServerJavaScript::setup_key_event(const EmscriptenKeyboardEvent *emscripten_event) {
	Ref<InputEventKey> ev;
	ev.instance();
	ev->set_echo(emscripten_event->repeat);
	dom2godot_mod(emscripten_event, ev);
	ev->set_keycode(dom_code2godot_scancode(emscripten_event->code, emscripten_event->key, false));
	ev->set_physical_keycode(dom_code2godot_scancode(emscripten_event->code, emscripten_event->key, true));

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

EM_BOOL DisplayServerJavaScript::keydown_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();
	Ref<InputEventKey> ev = setup_key_event(p_event);
	ev->set_pressed(true);
	if (ev->get_unicode() == 0 && keycode_has_unicode(ev->get_keycode())) {
		// Defer to keypress event for legacy unicode retrieval.
		display->deferred_key_event = ev;
		// Do not suppress keypress event.
		return false;
	}
	Input::get_singleton()->parse_input_event(ev);
	return true;
}

EM_BOOL DisplayServerJavaScript::keypress_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();
	display->deferred_key_event->set_unicode(p_event->charCode);
	Input::get_singleton()->parse_input_event(display->deferred_key_event);
	return true;
}

EM_BOOL DisplayServerJavaScript::keyup_callback(int p_event_type, const EmscriptenKeyboardEvent *p_event, void *p_user_data) {
	Ref<InputEventKey> ev = setup_key_event(p_event);
	ev->set_pressed(false);
	Input::get_singleton()->parse_input_event(ev);
	return ev->get_keycode() != KEY_UNKNOWN && ev->get_keycode() != 0;
}

// Mouse

EM_BOOL DisplayServerJavaScript::mouse_button_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();

	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_pressed(p_event_type == EMSCRIPTEN_EVENT_MOUSEDOWN);
	ev->set_position(compute_position_in_canvas(p_event->clientX, p_event->clientY));
	ev->set_global_position(ev->get_position());
	dom2godot_mod(p_event, ev);

	switch (p_event->button) {
		case DOM_BUTTON_LEFT:
			ev->set_button_index(BUTTON_LEFT);
			break;
		case DOM_BUTTON_MIDDLE:
			ev->set_button_index(BUTTON_MIDDLE);
			break;
		case DOM_BUTTON_RIGHT:
			ev->set_button_index(BUTTON_RIGHT);
			break;
		case DOM_BUTTON_XBUTTON1:
			ev->set_button_index(BUTTON_XBUTTON1);
			break;
		case DOM_BUTTON_XBUTTON2:
			ev->set_button_index(BUTTON_XBUTTON2);
			break;
		default:
			return false;
	}

	if (ev->is_pressed()) {
		double diff = emscripten_get_now() - display->last_click_ms;

		if (ev->get_button_index() == display->last_click_button_index) {
			if (diff < 400 && Point2(display->last_click_pos).distance_to(ev->get_position()) < 5) {
				display->last_click_ms = 0;
				display->last_click_pos = Point2(-100, -100);
				display->last_click_button_index = -1;
				ev->set_doubleclick(true);
			}

		} else {
			display->last_click_button_index = ev->get_button_index();
		}

		if (!ev->is_doubleclick()) {
			display->last_click_ms += diff;
			display->last_click_pos = ev->get_position();
		}
	}

	Input *input = Input::get_singleton();
	int mask = input->get_mouse_button_mask();
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

	input->parse_input_event(ev);
	// Prevent multi-click text selection and wheel-click scrolling anchor.
	// Context menu is prevented through contextmenu event.
	return true;
}

EM_BOOL DisplayServerJavaScript::mousemove_callback(int p_event_type, const EmscriptenMouseEvent *p_event, void *p_user_data) {
	Input *input = Input::get_singleton();
	int input_mask = input->get_mouse_button_mask();
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
	input->set_mouse_position(ev->get_position());
	ev->set_speed(input->get_last_mouse_speed());

	input->parse_input_event(ev);
	// Don't suppress mouseover/-leave events.
	return false;
}

// Cursor
const char *DisplayServerJavaScript::godot2dom_cursor(DisplayServer::CursorShape p_shape) {
	switch (p_shape) {
		case DisplayServer::CURSOR_ARROW:
			return "auto";
		case DisplayServer::CURSOR_IBEAM:
			return "text";
		case DisplayServer::CURSOR_POINTING_HAND:
			return "pointer";
		case DisplayServer::CURSOR_CROSS:
			return "crosshair";
		case DisplayServer::CURSOR_WAIT:
			return "progress";
		case DisplayServer::CURSOR_BUSY:
			return "wait";
		case DisplayServer::CURSOR_DRAG:
			return "grab";
		case DisplayServer::CURSOR_CAN_DROP:
			return "grabbing";
		case DisplayServer::CURSOR_FORBIDDEN:
			return "no-drop";
		case DisplayServer::CURSOR_VSIZE:
			return "ns-resize";
		case DisplayServer::CURSOR_HSIZE:
			return "ew-resize";
		case DisplayServer::CURSOR_BDIAGSIZE:
			return "nesw-resize";
		case DisplayServer::CURSOR_FDIAGSIZE:
			return "nwse-resize";
		case DisplayServer::CURSOR_MOVE:
			return "move";
		case DisplayServer::CURSOR_VSPLIT:
			return "row-resize";
		case DisplayServer::CURSOR_HSPLIT:
			return "col-resize";
		case DisplayServer::CURSOR_HELP:
			return "help";
		default:
			return "auto";
	}
}

void DisplayServerJavaScript::set_css_cursor(const char *p_cursor) {
	/* clang-format off */
	EM_ASM_({
		Module['canvas'].style.cursor = UTF8ToString($0);
	}, p_cursor);
	/* clang-format on */
}

bool DisplayServerJavaScript::is_css_cursor_hidden() const {
	/* clang-format off */
	return EM_ASM_INT({
		return Module['canvas'].style.cursor === 'none';
	});
	/* clang-format on */
}

void DisplayServerJavaScript::cursor_set_shape(CursorShape p_shape) {
	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (mouse_get_mode() == MOUSE_MODE_VISIBLE) {
		if (cursors[p_shape] != "") {
			Vector<String> url = cursors[p_shape].split("?");
			set_css_cursor(("url(\"" + url[0] + "\") " + url[1] + ", auto").utf8());
		} else {
			set_css_cursor(godot2dom_cursor(p_shape));
		}
	}

	cursor_shape = p_shape;
}

DisplayServer::CursorShape DisplayServerJavaScript::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerJavaScript::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	if (p_cursor.is_valid()) {
		Map<CursorShape, Vector<Variant>>::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}

			cursors_cache.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_data();
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

		PackedByteArray png;
		size_t len;
		PackedByteArray data = image->get_data();
		ERR_FAIL_COND(!png_image_write_get_memory_size(png_meta, len, 0, data.ptr(), 0, nullptr));

		png.resize(len);
		ERR_FAIL_COND(!png_image_write_to_memory(&png_meta, png.ptrw(), &len, 0, data.ptr(), 0, nullptr));

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
		}, png.ptr(), len, &object_url);
		/* clang-format on */

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

	cursor_set_shape(cursor_shape);
}

// Mouse mode
void DisplayServerJavaScript::mouse_set_mode(MouseMode p_mode) {
	ERR_FAIL_COND_MSG(p_mode == MOUSE_MODE_CONFINED, "MOUSE_MODE_CONFINED is not supported for the HTML5 platform.");
	if (p_mode == mouse_get_mode())
		return;

	if (p_mode == MOUSE_MODE_VISIBLE) {
		// set_css_cursor must be called before set_cursor_shape to make the cursor visible
		set_css_cursor(godot2dom_cursor(cursor_shape));
		cursor_set_shape(cursor_shape);
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_HIDDEN) {
		set_css_cursor("none");
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_CAPTURED) {
		EMSCRIPTEN_RESULT result = emscripten_request_pointerlock("canvas", false);
		ERR_FAIL_COND_MSG(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
		ERR_FAIL_COND_MSG(result != EMSCRIPTEN_RESULT_SUCCESS, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
		// set_css_cursor must be called before cursor_set_shape to make the cursor visible
		set_css_cursor(godot2dom_cursor(cursor_shape));
		cursor_set_shape(cursor_shape);
	}
}

DisplayServer::MouseMode DisplayServerJavaScript::mouse_get_mode() const {
	if (is_css_cursor_hidden())
		return MOUSE_MODE_HIDDEN;

	EmscriptenPointerlockChangeEvent ev;
	emscripten_get_pointerlock_status(&ev);
	return (ev.isActive && String::utf8(ev.id) == "canvas") ? MOUSE_MODE_CAPTURED : MOUSE_MODE_VISIBLE;
}

// Wheel

EM_BOOL DisplayServerJavaScript::wheel_callback(int p_event_type, const EmscriptenWheelEvent *p_event, void *p_user_data) {
	ERR_FAIL_COND_V(p_event_type != EMSCRIPTEN_EVENT_WHEEL, false);
	if (!is_canvas_focused()) {
		if (cursor_inside_canvas) {
			focus_canvas();
		} else {
			return false;
		}
	}

	Input *input = Input::get_singleton();
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
EM_BOOL DisplayServerJavaScript::touch_press_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();
	Ref<InputEventScreenTouch> ev;
	ev.instance();
	int lowest_id_index = -1;
	for (int i = 0; i < p_event->numTouches; ++i) {
		const EmscriptenTouchPoint &touch = p_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < p_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev->set_index(touch.identifier);
		ev->set_position(compute_position_in_canvas(touch.clientX, touch.clientY));
		display->touches[i] = ev->get_position();
		ev->set_pressed(p_event_type == EMSCRIPTEN_EVENT_TOUCHSTART);

		Input::get_singleton()->parse_input_event(ev);
	}
	// Resume audio context after input in case autoplay was denied.
	return true;
}

EM_BOOL DisplayServerJavaScript::touchmove_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data) {
	DisplayServerJavaScript *display = get_singleton();
	Ref<InputEventScreenDrag> ev;
	ev.instance();
	int lowest_id_index = -1;
	for (int i = 0; i < p_event->numTouches; ++i) {
		const EmscriptenTouchPoint &touch = p_event->touches[i];
		if (lowest_id_index == -1 || touch.identifier < p_event->touches[lowest_id_index].identifier)
			lowest_id_index = i;
		if (!touch.isChanged)
			continue;
		ev->set_index(touch.identifier);
		ev->set_position(compute_position_in_canvas(touch.clientX, touch.clientY));
		Point2 &prev = display->touches[i];
		ev->set_relative(ev->get_position() - prev);
		prev = ev->get_position();

		Input::get_singleton()->parse_input_event(ev);
	}
	return true;
}

bool DisplayServerJavaScript::screen_is_touchscreen(int p_screen) const {
	return EM_ASM_INT({ return 'ontouchstart' in window; });
}

// Gamepad

EM_BOOL DisplayServerJavaScript::gamepad_change_callback(int p_event_type, const EmscriptenGamepadEvent *p_event, void *p_user_data) {
	Input *input = Input::get_singleton();
	if (p_event_type == EMSCRIPTEN_EVENT_GAMEPADCONNECTED) {
		String guid = "";
		if (String::utf8(p_event->mapping) == "standard")
			guid = "Default HTML5 Gamepad";
		input->joy_connection_changed(p_event->index, true, String::utf8(p_event->id), guid);
	} else {
		input->joy_connection_changed(p_event->index, false, "");
	}
	return true;
}

void DisplayServerJavaScript::process_joypads() {
	int joypad_count = emscripten_get_num_gamepads();
	Input *input = Input::get_singleton();
	for (int joypad = 0; joypad < joypad_count; joypad++) {
		EmscriptenGamepadEvent state;
		EMSCRIPTEN_RESULT query_result = emscripten_get_gamepad_status(joypad, &state);
		// Chromium reserves gamepads slots, so NO_DATA is an expected result.
		ERR_CONTINUE(query_result != EMSCRIPTEN_RESULT_SUCCESS &&
					 query_result != EMSCRIPTEN_RESULT_NO_DATA);
		if (query_result == EMSCRIPTEN_RESULT_SUCCESS && state.connected) {
			int button_count = MIN(state.numButtons, 18);
			int axis_count = MIN(state.numAxes, 8);
			for (int button = 0; button < button_count; button++) {
				float value = state.analogButton[button];
				input->joy_button(joypad, button, value);
			}
			for (int axis = 0; axis < axis_count; axis++) {
				Input::JoyAxis joy_axis;
				joy_axis.min = -1;
				joy_axis.value = state.axis[axis];
				input->joy_axis(joypad, axis, joy_axis);
			}
		}
	}
}

#if 0
bool DisplayServerJavaScript::is_joy_known(int p_device) {

	return Input::get_singleton()->is_joy_mapped(p_device);
}


String DisplayServerJavaScript::get_joy_guid(int p_device) const {

	return Input::get_singleton()->get_joy_guid_remapped(p_device);
}
#endif

Vector<String> DisplayServerJavaScript::get_rendering_drivers_func() {
	Vector<String> drivers;
	drivers.push_back("dummy");
	return drivers;
}

// Clipboard
extern "C" EMSCRIPTEN_KEEPALIVE void update_clipboard(const char *p_text) {
	// Only call set_clipboard from OS (sets local clipboard)
	DisplayServerJavaScript::get_singleton()->clipboard = p_text;
}

void DisplayServerJavaScript::clipboard_set(const String &p_text) {
	/* clang-format off */
	int err = EM_ASM_INT({
		var text = UTF8ToString($0);
		if (!navigator.clipboard || !navigator.clipboard.writeText)
			return 1;
		navigator.clipboard.writeText(text).catch(function(e) {
			// Setting OS clipboard is only possible from an input callback.
			console.error("Setting OS clipboard is only possible from an input callback for the HTML5 plafrom. Exception:", e);
		});
		return 0;
	}, p_text.utf8().get_data());
	/* clang-format on */
	ERR_FAIL_COND_MSG(err, "Clipboard API is not supported.");
}

String DisplayServerJavaScript::clipboard_get() const {
	/* clang-format off */
	EM_ASM({
		try {
			navigator.clipboard.readText().then(function (result) {
				ccall('update_clipboard', 'void', ['string'], [result]);
			}).catch(function (e) {
				// Fail graciously.
			});
		} catch (e) {
			// Fail graciously.
		}
	});
	/* clang-format on */
	return clipboard;
}

extern "C" EMSCRIPTEN_KEEPALIVE void send_window_event(int p_notification) {
	if (p_notification == DisplayServer::WINDOW_EVENT_MOUSE_ENTER || p_notification == DisplayServer::WINDOW_EVENT_MOUSE_EXIT) {
		cursor_inside_canvas = p_notification == DisplayServer::WINDOW_EVENT_MOUSE_ENTER;
	}
	OS_JavaScript *os = OS_JavaScript::get_singleton();
	if (os->is_finalizing())
		return; // We don't want events anymore.
	DisplayServerJavaScript *ds = DisplayServerJavaScript::get_singleton();
	if (ds && !ds->window_event_callback.is_null()) {
		Variant event = int(p_notification);
		Variant *eventp = &event;
		Variant ret;
		Callable::CallError ce;
		ds->window_event_callback.call((const Variant **)&eventp, 1, ret, ce);
	}
}

void DisplayServerJavaScript::alert(const String &p_alert, const String &p_title) {
	/* clang-format off */
	EM_ASM_({
		window.alert(UTF8ToString($0));
	}, p_alert.utf8().get_data());
	/* clang-format on */
}

void DisplayServerJavaScript::set_icon(const Ref<Image> &p_icon) {
	ERR_FAIL_COND(p_icon.is_null());
	Ref<Image> icon = p_icon;
	if (icon->is_compressed()) {
		icon = icon->duplicate();
		ERR_FAIL_COND(icon->decompress() != OK);
	}
	if (icon->get_format() != Image::FORMAT_RGBA8) {
		if (icon == p_icon)
			icon = icon->duplicate();
		icon->convert(Image::FORMAT_RGBA8);
	}

	png_image png_meta;
	memset(&png_meta, 0, sizeof png_meta);
	png_meta.version = PNG_IMAGE_VERSION;
	png_meta.width = icon->get_width();
	png_meta.height = icon->get_height();
	png_meta.format = PNG_FORMAT_RGBA;

	PackedByteArray png;
	size_t len;
	PackedByteArray data = icon->get_data();
	ERR_FAIL_COND(!png_image_write_get_memory_size(png_meta, len, 0, data.ptr(), 0, nullptr));

	png.resize(len);
	ERR_FAIL_COND(!png_image_write_to_memory(&png_meta, png.ptrw(), &len, 0, data.ptr(), 0, nullptr));

	/* clang-format off */
	EM_ASM({
		var PNG_PTR = $0;
		var PNG_LEN = $1;

		var png = new Blob([HEAPU8.slice(PNG_PTR, PNG_PTR + PNG_LEN)], { type: "image/png" });
		var url = URL.createObjectURL(png);
		var link = document.getElementById('-gd-engine-icon');
		if (link === null) {
			link = document.createElement('link');
			link.rel = 'icon';
			link.id = '-gd-engine-icon';
			document.head.appendChild(link);
		}
		link.href = url;
	}, png.ptr(), len);
	/* clang-format on */
}

void DisplayServerJavaScript::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	OS_JavaScript *os = OS_JavaScript::get_singleton();
	if (os->is_finalizing())
		return; // We don't want events anymore.

	// Resume audio context after input in case autoplay was denied.
	os->resume_audio();

	Callable cb = get_singleton()->input_event_callback;
	if (!cb.is_null()) {
		Variant ev = p_event;
		Variant *evp = &ev;
		Variant ret;
		Callable::CallError ce;
		cb.call((const Variant **)&evp, 1, ret, ce);
	}
}

DisplayServer *DisplayServerJavaScript::create_func(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	return memnew(DisplayServerJavaScript(p_rendering_driver, p_mode, p_flags, p_resolution, r_error));
}

DisplayServerJavaScript::DisplayServerJavaScript(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	r_error = OK; // Always succeeds for now.

	/* clang-format off */
	swap_cancel_ok = EM_ASM_INT({
		const win = (['Windows', 'Win64', 'Win32', 'WinCE']);
		const plat = navigator.platform || "";
		if (win.indexOf(plat) !== -1) {
			return 1;
		}
		return 0;
	}) == 1;
	/* clang-format on */

	RasterizerDummy::make_current(); // TODO GLES2 in Godot 4.0... or webgpu?
#if 0
	EmscriptenWebGLContextAttributes attributes;
	emscripten_webgl_init_context_attributes(&attributes);
	attributes.alpha = GLOBAL_GET("display/window/per_pixel_transparency/allowed");
	attributes.antialias = false;
	ERR_FAIL_INDEX_V(p_video_driver, VIDEO_DRIVER_MAX, ERR_INVALID_PARAMETER);

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

	bool gl_initialization_error = false;

	if (RasterizerGLES2::is_viable() == OK) {
		attributes.majorVersion = 1;
		RasterizerGLES2::register_config();
		RasterizerGLES2::make_current();
	} else {
		gl_initialization_error = true;
	}

	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx = emscripten_webgl_create_context(canvas_id, &attributes);
	if (emscripten_webgl_make_context_current(ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
		gl_initialization_error = true;
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your browser does not seem to support WebGL. Please update your browser version.",
				"Unable to initialize video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;
#endif

	/* clang-format off */
	window_set_mode(p_mode);
	if (EM_ASM_INT_V({ return Module['resizeCanvasOnStart'] })) {
		/* clang-format on */
		window_set_size(p_resolution);
	}

	EMSCRIPTEN_RESULT result;
#define EM_CHECK(ev)                         \
	if (result != EMSCRIPTEN_RESULT_SUCCESS) \
		ERR_PRINT("Error while setting " #ev " callback: Code " + itos(result));
#define SET_EM_CALLBACK(target, ev, cb)                                  \
	result = emscripten_set_##ev##_callback(target, nullptr, true, &cb); \
	EM_CHECK(ev)
#define SET_EM_WINDOW_CALLBACK(ev, cb)                                                         \
	result = emscripten_set_##ev##_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, false, &cb); \
	EM_CHECK(ev)
#define SET_EM_CALLBACK_NOTARGET(ev, cb)                         \
	result = emscripten_set_##ev##_callback(nullptr, true, &cb); \
	EM_CHECK(ev)
	// These callbacks from Emscripten's html5.h suffice to access most
	// JavaScript APIs. For APIs that are not (sufficiently) exposed, EM_ASM
	// is used below.
	SET_EM_CALLBACK(canvas_id, mousedown, mouse_button_callback)
	SET_EM_WINDOW_CALLBACK(mousemove, mousemove_callback)
	SET_EM_WINDOW_CALLBACK(mouseup, mouse_button_callback)
	SET_EM_CALLBACK(canvas_id, wheel, wheel_callback)
	SET_EM_CALLBACK(canvas_id, touchstart, touch_press_callback)
	SET_EM_CALLBACK(canvas_id, touchmove, touchmove_callback)
	SET_EM_CALLBACK(canvas_id, touchend, touch_press_callback)
	SET_EM_CALLBACK(canvas_id, touchcancel, touch_press_callback)
	SET_EM_CALLBACK(canvas_id, keydown, keydown_callback)
	SET_EM_CALLBACK(canvas_id, keypress, keypress_callback)
	SET_EM_CALLBACK(canvas_id, keyup, keyup_callback)
	SET_EM_CALLBACK(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, fullscreenchange, fullscreen_change_callback)
	SET_EM_CALLBACK_NOTARGET(gamepadconnected, gamepad_change_callback)
	SET_EM_CALLBACK_NOTARGET(gamepaddisconnected, gamepad_change_callback)
#undef SET_EM_CALLBACK_NOTARGET
#undef SET_EM_CALLBACK
#undef EM_CHECK

	/* clang-format off */
	EM_ASM_ARGS({
		// Bind native event listeners.
		// Module.listeners, and Module.drop_handler are defined in native/utils.js
		const canvas = Module['canvas'];
		const send_window_event = cwrap('send_window_event', null, ['number']);
		const notifications = arguments;
		(['mouseover', 'mouseleave', 'focus', 'blur']).forEach(function(event, index) {
			Module.listeners.add(canvas, event, send_window_event.bind(null, notifications[index]), true);
		});
		// Clipboard
		const update_clipboard = cwrap('update_clipboard', null, ['string']);
		Module.listeners.add(window, 'paste', function(evt) {
			update_clipboard(evt.clipboardData.getData('text'));
		}, false);
		// Drag an drop
		Module.listeners.add(canvas, 'dragover', function(ev) {
			// Prevent default behavior (which would try to open the file(s))
			ev.preventDefault();
		}, false);
		Module.listeners.add(canvas, 'drop', Module.drop_handler, false);
	},
		WINDOW_EVENT_MOUSE_ENTER,
		WINDOW_EVENT_MOUSE_EXIT,
		WINDOW_EVENT_FOCUS_IN,
		WINDOW_EVENT_FOCUS_OUT
	);
	/* clang-format on */

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_event);
}

DisplayServerJavaScript::~DisplayServerJavaScript() {
	EM_ASM({
		Module.listeners.clear();
	});
	//emscripten_webgl_commit_frame();
	//emscripten_webgl_destroy_context(webgl_ctx);
}

bool DisplayServerJavaScript::has_feature(Feature p_feature) const {
	switch (p_feature) {
		//case FEATURE_CONSOLE_WINDOW:
		//case FEATURE_GLOBAL_MENU:
		//case FEATURE_HIDPI:
		//case FEATURE_IME:
		case FEATURE_ICON:
		case FEATURE_CLIPBOARD:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_CUSTOM_CURSOR_SHAPE:
		case FEATURE_MOUSE:
		case FEATURE_TOUCHSCREEN:
			return true;
		//case FEATURE_MOUSE_WARP:
		//case FEATURE_NATIVE_DIALOG:
		//case FEATURE_NATIVE_ICON:
		//case FEATURE_NATIVE_VIDEO:
		//case FEATURE_WINDOW_TRANSPARENCY:
		//case FEATURE_KEEP_SCREEN_ON:
		//case FEATURE_ORIENTATION:
		//case FEATURE_VIRTUAL_KEYBOARD:
		default:
			return false;
	}
}

void DisplayServerJavaScript::register_javascript_driver() {
	register_create_function("javascript", create_func, get_rendering_drivers_func);
}

String DisplayServerJavaScript::get_name() const {
	return "javascript";
}

int DisplayServerJavaScript::get_screen_count() const {
	return 1;
}

Point2i DisplayServerJavaScript::screen_get_position(int p_screen) const {
	return Point2i(); // TODO offsetX/Y?
}

Size2i DisplayServerJavaScript::screen_get_size(int p_screen) const {
	EmscriptenFullscreenChangeEvent ev;
	EMSCRIPTEN_RESULT result = emscripten_get_fullscreen_status(&ev);
	ERR_FAIL_COND_V(result != EMSCRIPTEN_RESULT_SUCCESS, Size2i());
	return Size2i(ev.screenWidth, ev.screenHeight);
}

Rect2i DisplayServerJavaScript::screen_get_usable_rect(int p_screen) const {
	int canvas[2];
	emscripten_get_canvas_element_size(canvas_id, canvas, canvas + 1);
	return Rect2i(0, 0, canvas[0], canvas[1]);
}

int DisplayServerJavaScript::screen_get_dpi(int p_screen) const {
	return 96; // TODO maybe check pixel ratio via window.devicePixelRatio * 96? Inexact.
}

Vector<DisplayServer::WindowID> DisplayServerJavaScript::get_window_list() const {
	Vector<WindowID> ret;
	ret.push_back(MAIN_WINDOW_ID);
	return ret;
}

DisplayServerJavaScript::WindowID DisplayServerJavaScript::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

void DisplayServerJavaScript::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerJavaScript::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerJavaScript::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	// Not supported.
}

void DisplayServerJavaScript::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callback = p_callable;
}

void DisplayServerJavaScript::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerJavaScript::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callback = p_callable; // TODO unused... do I need this?
}

void DisplayServerJavaScript::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	drop_files_callback = p_callable;
}

void DisplayServerJavaScript::window_set_title(const String &p_title, WindowID p_window) {
	/* clang-format off */
	EM_ASM_({
		document.title = UTF8ToString($0);
	}, p_title.utf8().get_data());
	/* clang-format on */
}

int DisplayServerJavaScript::window_get_current_screen(WindowID p_window) const {
	return 1;
}

void DisplayServerJavaScript::window_set_current_screen(int p_screen, WindowID p_window) {
	// Not implemented.
}

Point2i DisplayServerJavaScript::window_get_position(WindowID p_window) const {
	return Point2i(); // TODO Does this need implementation?
}

void DisplayServerJavaScript::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Not supported.
}

void DisplayServerJavaScript::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Not supported.
}

void DisplayServerJavaScript::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Not supported.
}

Size2i DisplayServerJavaScript::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerJavaScript::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Not supported.
}

Size2i DisplayServerJavaScript::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerJavaScript::window_set_size(const Size2i p_size, WindowID p_window) {
	last_width = p_size.x;
	last_height = p_size.y;
	double scale = EM_ASM_DOUBLE({
		return window.devicePixelRatio || 1;
	});
	emscripten_set_canvas_element_size(canvas_id, p_size.x * scale, p_size.y * scale);
	emscripten_set_element_css_size(canvas_id, p_size.x, p_size.y);
}

Size2i DisplayServerJavaScript::window_get_size(WindowID p_window) const {
	int canvas[2];
	emscripten_get_canvas_element_size(canvas_id, canvas, canvas + 1);
	return Size2(canvas[0], canvas[1]);
}

Size2i DisplayServerJavaScript::window_get_real_size(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerJavaScript::window_set_mode(WindowMode p_mode, WindowID p_window) {
	if (window_mode == p_mode)
		return;

	switch (p_mode) {
		case WINDOW_MODE_WINDOWED: {
			if (window_mode == WINDOW_MODE_FULLSCREEN) {
				emscripten_exit_fullscreen();
			}
			window_mode = WINDOW_MODE_WINDOWED;
			window_set_size(windowed_size);
		} break;
		case WINDOW_MODE_FULLSCREEN: {
			EmscriptenFullscreenStrategy strategy;
			strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
			strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
			strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
			strategy.canvasResizedCallback = nullptr;
			EMSCRIPTEN_RESULT result = emscripten_request_fullscreen_strategy(canvas_id, false, &strategy);
			ERR_FAIL_COND_MSG(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED, "Enabling fullscreen is only possible from an input callback for the HTML5 platform.");
			ERR_FAIL_COND_MSG(result != EMSCRIPTEN_RESULT_SUCCESS, "Enabling fullscreen is only possible from an input callback for the HTML5 platform.");
		} break;
		case WINDOW_MODE_MAXIMIZED:
		case WINDOW_MODE_MINIMIZED:
			WARN_PRINT("WindowMode MAXIMIZED and MINIMIZED are not supported in HTML5 platform.");
			break;
		default:
			break;
	}
}

DisplayServerJavaScript::WindowMode DisplayServerJavaScript::window_get_mode(WindowID p_window) const {
	return window_mode;
}

bool DisplayServerJavaScript::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerJavaScript::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Not supported.
}

bool DisplayServerJavaScript::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerJavaScript::window_request_attention(WindowID p_window) {
	// Not supported.
}

void DisplayServerJavaScript::window_move_to_foreground(WindowID p_window) {
	// Not supported.
}

bool DisplayServerJavaScript::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerJavaScript::can_any_window_draw() const {
	return true;
}

void DisplayServerJavaScript::process_events() {
	if (emscripten_sample_gamepad_data() == EMSCRIPTEN_RESULT_SUCCESS)
		process_joypads();
}

int DisplayServerJavaScript::get_current_video_driver() const {
	return 1;
}

bool DisplayServerJavaScript::get_swap_cancel_ok() {
	return swap_cancel_ok;
}

void DisplayServerJavaScript::swap_buffers() {
	//emscripten_webgl_commit_frame();
}
