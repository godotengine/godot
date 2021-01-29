/*************************************************************************/
/*  os_javascript.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/io/json.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"
#ifndef NO_THREADS
#include "servers/visual/visual_server_wrap_mt.h"
#endif

#include <dlfcn.h>
#include <emscripten.h>
#include <png.h>
#include <stdlib.h>

#include "dom_keys.inc"
#include "godot_js.h"

#define DOM_BUTTON_LEFT 0
#define DOM_BUTTON_MIDDLE 1
#define DOM_BUTTON_RIGHT 2
#define DOM_BUTTON_XBUTTON1 3
#define DOM_BUTTON_XBUTTON2 4

// Quit
void OS_JavaScript::request_quit_callback() {
	OS_JavaScript *os = get_singleton();
	if (os && os->get_main_loop()) {
		os->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
	}
}

// Files drop (implemented in JS for now).
void OS_JavaScript::drop_files_callback(char **p_filev, int p_filec) {
	OS_JavaScript *os = get_singleton();
	if (!os || !os->get_main_loop()) {
		return;
	}
	Vector<String> files;
	for (int i = 0; i < p_filec; i++) {
		files.push_back(String::utf8(p_filev[i]));
	}
	os->get_main_loop()->drop_files(files);
}

void OS_JavaScript::send_notification_callback(int p_notification) {

	OS_JavaScript *os = get_singleton();
	if (!os) {
		return;
	}
	if (p_notification == MainLoop::NOTIFICATION_WM_MOUSE_ENTER || p_notification == MainLoop::NOTIFICATION_WM_MOUSE_EXIT) {
		os->cursor_inside_canvas = p_notification == MainLoop::NOTIFICATION_WM_MOUSE_ENTER;
	}
	MainLoop *loop = os->get_main_loop();
	if (loop) {
		loop->notification(p_notification);
	}
}

// Window (canvas)

Point2 OS_JavaScript::compute_position_in_canvas(int x, int y) {
	OS_JavaScript *os = get_singleton();
	int canvas_x;
	int canvas_y;
	godot_js_display_canvas_bounding_rect_position_get(&canvas_x, &canvas_y);
	int canvas_width;
	int canvas_height;
	emscripten_get_canvas_element_size(os->canvas_id, &canvas_width, &canvas_height);

	double element_width;
	double element_height;
	emscripten_get_element_css_size(os->canvas_id, &element_width, &element_height);

	return Point2((int)(canvas_width / element_width * (x - canvas_x)),
			(int)(canvas_height / element_height * (y - canvas_y)));
}

bool OS_JavaScript::check_size_force_redraw() {
	int canvas_width;
	int canvas_height;
	emscripten_get_canvas_element_size(canvas_id, &canvas_width, &canvas_height);
	if (last_width != canvas_width || last_height != canvas_height) {
		last_width = canvas_width;
		last_height = canvas_height;
		// Update the framebuffer size for redraw.
		emscripten_set_canvas_element_size(canvas_id, canvas_width, canvas_height);
		return true;
	}
	return false;
}

EM_BOOL OS_JavaScript::fullscreen_change_callback(int p_event_type, const EmscriptenFullscreenChangeEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
	// Empty ID is canvas.
	String target_id = String::utf8(p_event->id);
	if (target_id.empty() || target_id == String::utf8(&(os->canvas_id[1]))) {
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
		double scale = godot_js_display_pixel_ratio_get();
		emscripten_set_canvas_element_size(canvas_id, p_size.x, p_size.y);
		emscripten_set_element_css_size(canvas_id, p_size.x / scale, p_size.y / scale);
	}
}

Size2 OS_JavaScript::get_window_size() const {

	int canvas[2];
	emscripten_get_canvas_element_size(canvas_id, canvas, canvas + 1);
	return Size2(canvas[0], canvas[1]);
}

void OS_JavaScript::set_window_maximized(bool p_enabled) {

#ifndef TOOLS_ENABLED
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
		emscripten_enter_soft_fullscreen(canvas_id, &strategy);
		window_maximized = p_enabled;
	}
#endif
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
		EMSCRIPTEN_RESULT result = emscripten_request_fullscreen_strategy(canvas_id, false, &strategy);
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
	os->resume_audio();
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
		godot_js_display_canvas_focus();
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
	os->resume_audio();
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
	if (!os->cursor_inside_canvas && !input_mask)
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

void OS_JavaScript::set_cursor_shape(CursorShape p_shape) {

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);
	if (cursor_shape == p_shape) {
		return;
	}
	cursor_shape = p_shape;
	godot_js_display_cursor_set_shape(godot2dom_cursor(cursor_shape));
}

void OS_JavaScript::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {

	if (p_cursor.is_valid()) {

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
		godot_js_display_cursor_set_custom_shape(godot2dom_cursor(p_shape), r.ptr(), len, p_hotspot.x, p_hotspot.y);
		r = PoolByteArray::Read();

	} else {
		godot_js_display_cursor_set_custom_shape(godot2dom_cursor(p_shape), NULL, 0, 0, 0);
	}
}

void OS_JavaScript::set_mouse_mode(OS::MouseMode p_mode) {

	ERR_FAIL_COND_MSG(p_mode == MOUSE_MODE_CONFINED, "MOUSE_MODE_CONFINED is not supported for the HTML5 platform.");
	if (p_mode == get_mouse_mode())
		return;

	if (p_mode == MOUSE_MODE_VISIBLE) {

		godot_js_display_cursor_set_visible(1);
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_HIDDEN) {

		godot_js_display_cursor_set_visible(0);
		emscripten_exit_pointerlock();

	} else if (p_mode == MOUSE_MODE_CAPTURED) {

		godot_js_display_cursor_set_visible(1);
		EMSCRIPTEN_RESULT result = emscripten_request_pointerlock(canvas_id, false);
		ERR_FAIL_COND_MSG(result == EMSCRIPTEN_RESULT_FAILED_NOT_DEFERRED, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
		ERR_FAIL_COND_MSG(result != EMSCRIPTEN_RESULT_SUCCESS, "MOUSE_MODE_CAPTURED can only be entered from within an appropriate input callback.");
	}
}

OS::MouseMode OS_JavaScript::get_mouse_mode() const {

	if (godot_js_display_cursor_is_hidden())
		return MOUSE_MODE_HIDDEN;

	EmscriptenPointerlockChangeEvent ev;
	emscripten_get_pointerlock_status(&ev);
	return (ev.isActive && String::utf8(ev.id) == String::utf8(&canvas_id[1])) ? MOUSE_MODE_CAPTURED : MOUSE_MODE_VISIBLE;
}

// Wheel

EM_BOOL OS_JavaScript::wheel_callback(int p_event_type, const EmscriptenWheelEvent *p_event, void *p_user_data) {

	ERR_FAIL_COND_V(p_event_type != EMSCRIPTEN_EVENT_WHEEL, false);
	OS_JavaScript *os = get_singleton();
	if (!godot_js_display_canvas_is_focused()) {
		if (os->cursor_inside_canvas) {
			godot_js_display_canvas_focus();
		} else {
			return false;
		}
	}

	InputDefault *input = os->input;
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

bool OS_JavaScript::has_touchscreen_ui_hint() const {
	return godot_js_display_touchscreen_is_available();
}

EM_BOOL OS_JavaScript::touch_press_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
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
		os->touches[i] = ev->get_position();
		ev->set_pressed(p_event_type == EMSCRIPTEN_EVENT_TOUCHSTART);

		os->input->parse_input_event(ev);
	}
	// Resume audio context after input in case autoplay was denied.
	os->resume_audio();
	return true;
}

EM_BOOL OS_JavaScript::touchmove_callback(int p_event_type, const EmscriptenTouchEvent *p_event, void *p_user_data) {

	OS_JavaScript *os = get_singleton();
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
		Point2 &prev = os->touches[i];
		ev->set_relative(ev->get_position() - prev);
		prev = ev->get_position();

		os->input->parse_input_event(ev);
	}
	return true;
}

// Gamepad
void OS_JavaScript::gamepad_callback(int p_index, int p_connected, const char *p_id, const char *p_guid) {
	InputDefault *input = get_singleton()->input;
	if (p_connected) {
		input->joy_connection_changed(p_index, true, String::utf8(p_id), String::utf8(p_guid));
	} else {
		input->joy_connection_changed(p_index, false, "");
	}
}

void OS_JavaScript::process_joypads() {

	int32_t pads = godot_js_display_gamepad_sample_count();
	int32_t s_btns_num = 0;
	int32_t s_axes_num = 0;
	int32_t s_standard = 0;
	float s_btns[16];
	float s_axes[10];
	for (int idx = 0; idx < pads; idx++) {
		int err = godot_js_display_gamepad_sample_get(idx, s_btns, &s_btns_num, s_axes, &s_axes_num, &s_standard);
		if (err) {
			continue;
		}
		for (int b = 0; b < s_btns_num; b++) {
			float value = s_btns[b];
			// Buttons 6 and 7 in the standard mapping need to be
			// axis to be handled as JOY_ANALOG by Godot.
			if (s_standard && (b == 6 || b == 7)) {
				InputDefault::JoyAxis joy_axis;
				joy_axis.min = 0;
				joy_axis.value = value;
				int a = b == 6 ? JOY_ANALOG_L2 : JOY_ANALOG_R2;
				input->joy_axis(idx, a, joy_axis);
			} else {
				input->joy_button(idx, b, value);
			}
		}
		for (int a = 0; a < s_axes_num; a++) {
			InputDefault::JoyAxis joy_axis;
			joy_axis.min = -1;
			joy_axis.value = s_axes[a];
			input->joy_axis(idx, a, joy_axis);
		}
	}
}

bool OS_JavaScript::is_joy_known(int p_device) {

	return input->is_joy_mapped(p_device);
}

String OS_JavaScript::get_joy_guid(int p_device) const {

	return input->get_joy_guid_remapped(p_device);
}

// Video

int OS_JavaScript::get_video_driver_count() const {

	return VIDEO_DRIVER_MAX;
}

const char *OS_JavaScript::get_video_driver_name(int p_driver) const {

	switch (p_driver) {
		case VIDEO_DRIVER_GLES3:
			return "GLES3";
		case VIDEO_DRIVER_GLES2:
			return "GLES2";
	}
	ERR_FAIL_V_MSG(NULL, "Invalid video driver index: " + itos(p_driver) + ".");
}

// Audio

int OS_JavaScript::get_audio_driver_count() const {

	return 1;
}

const char *OS_JavaScript::get_audio_driver_name(int p_driver) const {

	return "JavaScript";
}

// Clipboard
void OS_JavaScript::update_clipboard_callback(const char *p_text) {
	// Only call set_clipboard from OS (sets local clipboard)
	get_singleton()->OS::set_clipboard(p_text);
}

void OS_JavaScript::set_clipboard(const String &p_text) {
	OS::set_clipboard(p_text);
	int err = godot_js_display_clipboard_set(p_text.utf8().get_data());
	ERR_FAIL_COND_MSG(err, "Clipboard API is not supported.");
}

String OS_JavaScript::get_clipboard() const {
	godot_js_display_clipboard_get(update_clipboard_callback);
	return this->OS::get_clipboard();
}

// Lifecycle
int OS_JavaScript::get_current_video_driver() const {
	return video_driver_index;
}

void OS_JavaScript::initialize_core() {

	OS_Unix::initialize_core();
}

Error OS_JavaScript::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {

	swap_ok_cancel = godot_js_display_is_swap_ok_cancel() == 1;

	EmscriptenWebGLContextAttributes attributes;
	emscripten_webgl_init_context_attributes(&attributes);
	attributes.alpha = GLOBAL_GET("display/window/per_pixel_transparency/allowed");
	attributes.antialias = false;
	attributes.explicitSwapControl = true;
	ERR_FAIL_INDEX_V(p_video_driver, VIDEO_DRIVER_MAX, ERR_INVALID_PARAMETER);

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

#ifdef TOOLS_ENABLED
	bool gles3 = false;
#else
	bool gles3 = true;
	if (p_video_driver == VIDEO_DRIVER_GLES2) {
		gles3 = false;
	}
#endif

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

	webgl_ctx = emscripten_webgl_create_context(canvas_id, &attributes);
	if (emscripten_webgl_make_context_current(webgl_ctx) != EMSCRIPTEN_RESULT_SUCCESS) {
		gl_initialization_error = true;
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your browser does not support any of the supported WebGL versions.\n"
								   "Please update your browser version.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;

	video_mode = p_desired;
	// fullscreen_change_callback will correct this if the request is successful.
	video_mode.fullscreen = false;
	// Emscripten only attempts fullscreen requests if the user input callback
	// was registered through one its own functions, so request manually for
	// start-up fullscreen.
	if (p_desired.fullscreen) {
		godot_js_display_window_request_fullscreen();
	}
	if (godot_js_config_is_resize_on_start()) {
		set_window_size(Size2(video_mode.width, video_mode.height));
	} else {
		set_window_size(get_window_size());
	}

	AudioDriverManager::initialize(p_audio_driver);
	visual_server = memnew(VisualServerRaster());
#ifndef NO_THREADS
	visual_server = memnew(VisualServerWrapMT(visual_server, false));
#endif
	input = memnew(InputDefault);

	EMSCRIPTEN_RESULT result;
#define EM_CHECK(ev)                         \
	if (result != EMSCRIPTEN_RESULT_SUCCESS) \
	ERR_PRINTS("Error while setting " #ev " callback: Code " + itos(result))
#define SET_EM_CALLBACK(target, ev, cb)                               \
	result = emscripten_set_##ev##_callback(target, NULL, true, &cb); \
	EM_CHECK(ev)
#define SET_EM_WINDOW_CALLBACK(ev, cb)                                                         \
	result = emscripten_set_##ev##_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, false, &cb); \
	EM_CHECK(ev)
	// These callbacks from Emscripten's html5.h suffice to access most
	// JavaScript APIs.
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
#undef SET_EM_CALLBACK
#undef EM_CHECK

	// For APIs that are not (sufficiently) exposed, a
	// library is used below (implemented in library_godot_display.js).
	godot_js_display_notification_cb(&OS_JavaScript::send_notification_callback,
			MainLoop::NOTIFICATION_WM_MOUSE_ENTER,
			MainLoop::NOTIFICATION_WM_MOUSE_EXIT,
			MainLoop::NOTIFICATION_WM_FOCUS_IN,
			MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	godot_js_display_paste_cb(&OS_JavaScript::update_clipboard_callback);
	godot_js_display_drop_files_cb(&OS_JavaScript::drop_files_callback);
	godot_js_display_gamepad_cb(&OS_JavaScript::gamepad_callback);

	visual_server->init();

	return OK;
}

bool OS_JavaScript::get_swap_ok_cancel() {
	return swap_ok_cancel;
}

void OS_JavaScript::swap_buffers() {
	emscripten_webgl_commit_frame();
}

void OS_JavaScript::set_main_loop(MainLoop *p_main_loop) {

	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

MainLoop *OS_JavaScript::get_main_loop() const {

	return main_loop;
}

void OS_JavaScript::resume_audio() {
	if (audio_driver_javascript) {
		audio_driver_javascript->resume();
	}
}

void OS_JavaScript::fs_sync_callback() {
	get_singleton()->idb_is_syncing = false;
}

bool OS_JavaScript::main_loop_iterate() {

	if (is_userfs_persistent() && idb_needs_sync && !idb_is_syncing) {
		idb_is_syncing = true;
		idb_needs_sync = false;
		godot_js_os_fs_sync(&OS_JavaScript::fs_sync_callback);
	}

	if (godot_js_display_gamepad_sample() == OK)
		process_joypads();

	if (just_exited_fullscreen) {
		if (window_maximized) {
			EmscriptenFullscreenStrategy strategy;
			strategy.scaleMode = EMSCRIPTEN_FULLSCREEN_SCALE_STRETCH;
			strategy.canvasResolutionScaleMode = EMSCRIPTEN_FULLSCREEN_CANVAS_SCALE_STDDEF;
			strategy.filteringMode = EMSCRIPTEN_FULLSCREEN_FILTERING_DEFAULT;
			strategy.canvasResizedCallback = NULL;
			emscripten_enter_soft_fullscreen(canvas_id, &strategy);
		} else {
			set_window_size(Size2(windowed_size.width, windowed_size.height));
		}
		just_exited_fullscreen = false;
	}

	int canvas[2];
	emscripten_get_canvas_element_size(canvas_id, canvas, canvas + 1);
	video_mode.width = canvas[0];
	video_mode.height = canvas[1];
	if (!window_maximized && !video_mode.fullscreen && !just_exited_fullscreen && !entering_fullscreen) {
		windowed_size.width = canvas[0];
		windowed_size.height = canvas[1];
	}

	return Main::iteration();
}

void OS_JavaScript::delete_main_loop() {

	memdelete(main_loop);
	main_loop = NULL;
}

void OS_JavaScript::finalize() {

	memdelete(input);
	visual_server->finish();
	emscripten_webgl_commit_frame();
	memdelete(visual_server);
	emscripten_webgl_destroy_context(webgl_ctx);
	if (audio_driver_javascript) {
		memdelete(audio_driver_javascript);
	}
}

// Miscellaneous

Error OS_JavaScript::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex) {

	Array args;
	for (const List<String>::Element *E = p_arguments.front(); E; E = E->next()) {
		args.push_back(E->get());
	}
	String json_args = JSON::print(args);
	int failed = godot_js_os_execute(json_args.utf8().get_data());
	ERR_FAIL_COND_V_MSG(failed, ERR_UNAVAILABLE, "OS::execute() must be implemented in Javascript via 'engine.setOnExecute' if required.");
	return OK;
}

Error OS_JavaScript::kill(const ProcessID &p_pid) {

	ERR_FAIL_V_MSG(ERR_UNAVAILABLE, "OS::kill() is not available on the HTML5 platform.");
}

int OS_JavaScript::get_process_id() const {

	ERR_FAIL_V_MSG(0, "OS::get_process_id() is not available on the HTML5 platform.");
}

bool OS_JavaScript::_check_internal_feature_support(const String &p_feature) {

	if (p_feature == "HTML5" || p_feature == "web") {
		return true;
	}

#ifdef JAVASCRIPT_EVAL_ENABLED
	if (p_feature == "JavaScript") {
		return true;
	}
#endif
#ifndef NO_THREADS
	if (p_feature == "threads") {
		return true;
	}
#endif
#if WASM_GDNATIVE
	if (p_feature == "wasm32") {
		return true;
	}
#endif

	return false;
}

void OS_JavaScript::alert(const String &p_alert, const String &p_title) {
	godot_js_display_alert(p_alert.utf8().get_data());
}

void OS_JavaScript::set_window_title(const String &p_title) {
	godot_js_display_window_title_set(p_title.utf8().get_data());
}

void OS_JavaScript::set_icon(const Ref<Image> &p_icon) {

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

	PoolByteArray png;
	size_t len;
	PoolByteArray::Read r = icon->get_data().read();
	ERR_FAIL_COND(!png_image_write_get_memory_size(png_meta, len, 0, r.ptr(), 0, NULL));

	png.resize(len);
	PoolByteArray::Write w = png.write();
	ERR_FAIL_COND(!png_image_write_to_memory(&png_meta, w.ptr(), &len, 0, r.ptr(), 0, NULL));
	w = PoolByteArray::Write();

	r = png.read();
	godot_js_display_window_icon_set(r.ptr(), len);
}

String OS_JavaScript::get_executable_path() const {

	return OS::get_executable_path();
}

Error OS_JavaScript::shell_open(String p_uri) {

	// Open URI in a new tab, browser will deal with it by protocol.
	godot_js_os_shell_open(p_uri.utf8().get_data());
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

OS::PowerState OS_JavaScript::get_power_state() {

	WARN_PRINT_ONCE("Power management is not supported for the HTML5 platform, defaulting to POWERSTATE_UNKNOWN");
	return OS::POWERSTATE_UNKNOWN;
}

int OS_JavaScript::get_power_seconds_left() {

	WARN_PRINT_ONCE("Power management is not supported for the HTML5 platform, defaulting to -1");
	return -1;
}

int OS_JavaScript::get_power_percent_left() {

	WARN_PRINT_ONCE("Power management is not supported for the HTML5 platform, defaulting to -1");
	return -1;
}

void OS_JavaScript::file_access_close_callback(const String &p_file, int p_flags) {

	OS_JavaScript *os = get_singleton();

	if (!(os->is_userfs_persistent() && p_flags & FileAccess::WRITE)) {
		return; // FS persistence is not working or we are not writing.
	}
	bool is_file_persistent = p_file.begins_with("/userfs");
#ifdef TOOLS_ENABLED
	// Hack for editor persistence (can we track).
	is_file_persistent = is_file_persistent || p_file.begins_with("/home/web_user/");
#endif
	if (is_file_persistent) {
		os->idb_needs_sync = true;
	}
}

bool OS_JavaScript::is_userfs_persistent() const {

	return idb_available;
}

Error OS_JavaScript::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = p_path.get_file();
	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ". Error: " + dlerror());
	return OK;
}

OS_JavaScript *OS_JavaScript::get_singleton() {

	return static_cast<OS_JavaScript *>(OS::get_singleton());
}

OS_JavaScript::OS_JavaScript() {
	// Expose method for requesting quit.
	godot_js_os_request_quit_cb(&request_quit_callback);
	// Set canvas ID
	godot_js_config_canvas_id_get(canvas_id, sizeof(canvas_id));

	cursor_inside_canvas = true;
	cursor_shape = OS::CURSOR_ARROW;

	last_click_button_index = -1;
	last_click_ms = 0;
	last_click_pos = Point2(-100, -100);

	last_width = 0;
	last_height = 0;

	window_maximized = false;
	entering_fullscreen = false;
	just_exited_fullscreen = false;
	transparency_enabled = false;

	main_loop = NULL;
	visual_server = NULL;
	audio_driver_javascript = NULL;

	swap_ok_cancel = false;
	idb_available = godot_js_os_fs_is_persistent() != 0;
	idb_needs_sync = false;
	idb_is_syncing = false;

	if (AudioDriverJavaScript::is_available()) {
		audio_driver_javascript = memnew(AudioDriverJavaScript);
		AudioDriverManager::add_driver(audio_driver_javascript);
	}

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
}
