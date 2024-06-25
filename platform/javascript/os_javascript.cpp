/**************************************************************************/
/*  os_javascript.cpp                                                     */
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

#include "os_javascript.h"

#include "core/io/json.h"
#include "core/project_settings.h"
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

#include "api/javascript_singleton.h"
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

bool OS_JavaScript::tts_is_speaking() const {
	ERR_FAIL_COND_V_MSG(!tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return godot_js_tts_is_speaking();
}

bool OS_JavaScript::tts_is_paused() const {
	ERR_FAIL_COND_V_MSG(!tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return godot_js_tts_is_paused();
}

void OS_JavaScript::update_voices_callback(int p_size, const char **p_voice) {
	get_singleton()->voices.clear();
	for (int i = 0; i < p_size; i++) {
		Vector<String> tokens = String::utf8(p_voice[i]).split(";", true, 2);
		if (tokens.size() == 2) {
			Dictionary voice_d;
			voice_d["name"] = tokens[1];
			voice_d["id"] = tokens[1];
			voice_d["language"] = tokens[0];
			get_singleton()->voices.push_back(voice_d);
		}
	}
}

Array OS_JavaScript::tts_get_voices() const {
	ERR_FAIL_COND_V_MSG(!tts, Array(), "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	godot_js_tts_get_voices(update_voices_callback);
	return voices;
}

void OS_JavaScript::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	if (p_interrupt) {
		tts_stop();
	}

	if (p_text.empty()) {
		tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, p_utterance_id);
		return;
	}

	CharString string = p_text.utf8();
	utterance_ids[p_utterance_id] = string;

	godot_js_tts_speak(string.get_data(), p_voice.utf8().get_data(), CLAMP(p_volume, 0, 100), CLAMP(p_pitch, 0.f, 2.f), CLAMP(p_rate, 0.1f, 10.f), p_utterance_id, OS_JavaScript::_js_utterance_callback);
}

void OS_JavaScript::tts_pause() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	godot_js_tts_pause();
}

void OS_JavaScript::tts_resume() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	godot_js_tts_resume();
}

void OS_JavaScript::tts_stop() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	for (Map<int, CharString>::Element *E = utterance_ids.front(); E; E = E->next()) {
		tts_post_utterance_event(OS::TTS_UTTERANCE_CANCELED, E->key());
	}
	utterance_ids.clear();
	godot_js_tts_stop();
}

void OS_JavaScript::_js_utterance_callback(int p_event, int p_id, int p_pos) {
	OS_JavaScript *ds = (OS_JavaScript *)OS::get_singleton();
	if (ds->utterance_ids.has(p_id)) {
		int pos = 0;
		if ((TTSUtteranceEvent)p_event == OS::TTS_UTTERANCE_BOUNDARY) {
			// Convert position from UTF-8 to UTF-32.
			const CharString &string = ds->utterance_ids[p_id];
			for (int i = 0; i < MIN(p_pos, string.length()); i++) {
				uint8_t c = string[i];
				if ((c & 0xe0) == 0xc0) {
					i += 1;
				} else if ((c & 0xf0) == 0xe0) {
					i += 2;
				} else if ((c & 0xf8) == 0xf0) {
					i += 3;
				}
				pos++;
			}
		} else if ((TTSUtteranceEvent)p_event != OS::TTS_UTTERANCE_STARTED) {
			ds->utterance_ids.erase(p_id);
		}
		ds->tts_post_utterance_event((TTSUtteranceEvent)p_event, p_id, pos);
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
	if (godot_js_is_ime_focused() && (p_notification == MainLoop::NOTIFICATION_WM_FOCUS_IN || p_notification == MainLoop::NOTIFICATION_WM_FOCUS_OUT)) {
		return;
	}
	MainLoop *loop = os->get_main_loop();
	if (loop) {
		loop->notification(p_notification);
	}
}

// Window (canvas)

bool OS_JavaScript::check_size_force_redraw() {
	return godot_js_display_size_update() != 0;
}

void OS_JavaScript::fullscreen_change_callback(int p_fullscreen) {
	OS_JavaScript *os = get_singleton();
	os->video_mode.fullscreen = p_fullscreen;
}

void OS_JavaScript::window_blur_callback() {
	get_singleton()->input->release_pressed_events();
}

void OS_JavaScript::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
	video_mode = p_video_mode;
}

OS::VideoMode OS_JavaScript::get_video_mode(int p_screen) const {
	return video_mode;
}

Size2 OS_JavaScript::get_screen_size(int p_screen) const {
	int size[2];
	godot_js_display_screen_size_get(size, size + 1);
	return Size2(size[0], size[1]);
}

void OS_JavaScript::set_window_size(const Size2 p_size) {
	if (video_mode.fullscreen) {
		set_window_fullscreen(false);
	}
	godot_js_display_desired_size_set(p_size.x, p_size.y);
}

Size2 OS_JavaScript::get_window_size() const {
	int size[2];
	godot_js_display_window_size_get(size, size + 1);
	return Size2(size[0], size[1]);
}

void OS_JavaScript::set_window_maximized(bool p_enabled) {
}

bool OS_JavaScript::is_window_maximized() const {
	return false;
}

void OS_JavaScript::set_window_fullscreen(bool p_enabled) {
	if (p_enabled == video_mode.fullscreen) {
		return;
	}

	// Just request changes here, if successful, logic continues in
	// fullscreen_change_callback.
	if (p_enabled) {
		int result = godot_js_display_fullscreen_request();
		ERR_FAIL_COND_MSG(result, "The request was denied. Remember that enabling fullscreen is only possible from an input callback for the HTML5 platform.");
	} else {
		// No logic allowed here, since exiting w/ ESC key won't use this function.
		ERR_FAIL_COND(godot_js_display_fullscreen_exit());
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

static void dom2godot_mod(Ref<InputEventWithModifiers> ev, int p_mod) {
	ev->set_shift(p_mod & 1);
	ev->set_alt(p_mod & 2);
	ev->set_control(p_mod & 4);
	ev->set_metakey(p_mod & 8);
}

void OS_JavaScript::key_callback(int p_pressed, int p_repeat, int p_modifiers) {
	OS_JavaScript *os = get_singleton();
	JSKeyEvent &key_event = os->key_event;

	const String code = String::utf8(key_event.code);
	const String key = String::utf8(key_event.key);

	// Resume audio context after input in case autoplay was denied.
	os->resume_audio();

	if (os->ime_started) {
		return;
	}

	wchar_t c = 0x00;
	String unicode = key;
	if (unicode.length() == 1) {
		c = unicode[0];
	}
	uint32_t keycode = dom_code2godot_scancode(code.utf8().get_data(), key.utf8().get_data(), false);
	uint32_t scancode = dom_code2godot_scancode(code.utf8().get_data(), key.utf8().get_data(), true);

	OS_JavaScript::KeyEvent ke;

	ke.pressed = p_pressed;
	ke.echo = p_repeat;
	ke.raw = true;
	ke.keycode = keycode;
	ke.physical_keycode = scancode;
	ke.unicode = c;
	ke.mod = p_modifiers;

	if (os->key_event_pos >= os->key_event_buffer.size()) {
		os->key_event_buffer.resize(1 + os->key_event_pos);
	}
	os->key_event_buffer.write[os->key_event_pos++] = ke;

	// Make sure to flush all events so we can call restricted APIs inside the event.
	os->input->flush_buffered_events();
}

// Mouse

Point2 OS_JavaScript::get_mouse_position() const {
	return input->get_mouse_position();
}

int OS_JavaScript::get_mouse_button_state() const {
	return input->get_mouse_button_mask();
}

int OS_JavaScript::mouse_button_callback(int p_pressed, int p_button, double p_x, double p_y, int p_modifiers) {
	OS_JavaScript *os = get_singleton();

	Ref<InputEventMouseButton> ev;
	ev.instance();
	ev->set_pressed(p_pressed);
	ev->set_position(Point2(p_x, p_y));
	ev->set_global_position(ev->get_position());
	ev->set_pressed(p_pressed);
	dom2godot_mod(ev, p_modifiers);

	switch (p_button) {
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

	if (p_pressed) {
		uint64_t diff = (OS::get_singleton()->get_ticks_usec() / 1000) - os->last_click_ms;

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

	// Make sure to flush all events so we can call restricted APIs inside the event.
	os->input->flush_buffered_events();

	// Prevent multi-click text selection and wheel-click scrolling anchor.
	// Context menu is prevented through contextmenu event.
	return true;
}

void OS_JavaScript::mouse_move_callback(double p_x, double p_y, double p_rel_x, double p_rel_y, int p_modifiers) {
	OS_JavaScript *os = get_singleton();

	int input_mask = os->input->get_mouse_button_mask();
	// For motion outside the canvas, only read mouse movement if dragging
	// started inside the canvas; imitating desktop app behaviour.
	if (!os->cursor_inside_canvas && !input_mask)
		return;

	Ref<InputEventMouseMotion> ev;
	ev.instance();
	dom2godot_mod(ev, p_modifiers);
	ev->set_button_mask(input_mask);

	ev->set_position(Point2(p_x, p_y));
	ev->set_global_position(ev->get_position());

	ev->set_relative(Vector2(p_rel_x, p_rel_y));
	ev->set_speed(os->input->get_last_mouse_speed());

	os->input->parse_input_event(ev);
}

static const char *godot2dom_cursor(OS::CursorShape p_shape) {
	switch (p_shape) {
		case OS::CURSOR_ARROW:
		default:
			return "default";
		case OS::CURSOR_IBEAM:
			return "text";
		case OS::CURSOR_POINTING_HAND:
			return "pointer";
		case OS::CURSOR_CROSS:
			return "crosshair";
		case OS::CURSOR_WAIT:
			return "wait";
		case OS::CURSOR_BUSY:
			return "progress";
		case OS::CURSOR_DRAG:
			return "grab";
		case OS::CURSOR_CAN_DROP:
			return "grabbing";
		case OS::CURSOR_FORBIDDEN:
			return "no-drop";
		case OS::CURSOR_VSIZE:
			return "ns-resize";
		case OS::CURSOR_HSIZE:
			return "ew-resize";
		case OS::CURSOR_BDIAGSIZE:
			return "nesw-resize";
		case OS::CURSOR_FDIAGSIZE:
			return "nwse-resize";
		case OS::CURSOR_MOVE:
			return "move";
		case OS::CURSOR_VSPLIT:
			return "row-resize";
		case OS::CURSOR_HSPLIT:
			return "col-resize";
		case OS::CURSOR_HELP:
			return "help";
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

OS::CursorShape OS_JavaScript::get_cursor_shape() const {
	return cursor_shape;
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
	ERR_FAIL_COND_MSG(p_mode == MOUSE_MODE_CONFINED || p_mode == MOUSE_MODE_CONFINED_HIDDEN, "MOUSE_MODE_CONFINED is not supported for the HTML5 platform.");
	if (p_mode == get_mouse_mode()) {
		return;
	}

	if (p_mode == MOUSE_MODE_VISIBLE) {
		godot_js_display_cursor_set_visible(1);
		godot_js_display_cursor_lock_set(false);

	} else if (p_mode == MOUSE_MODE_HIDDEN) {
		godot_js_display_cursor_set_visible(0);
		godot_js_display_cursor_lock_set(false);

	} else if (p_mode == MOUSE_MODE_CAPTURED) {
		godot_js_display_cursor_set_visible(1);
		godot_js_display_cursor_lock_set(true);
	}
}

OS::MouseMode OS_JavaScript::get_mouse_mode() const {
	if (godot_js_display_cursor_is_hidden()) {
		return MOUSE_MODE_HIDDEN;
	}
	if (godot_js_display_cursor_is_locked()) {
		return MOUSE_MODE_CAPTURED;
	}
	return MOUSE_MODE_VISIBLE;
}

// Wheel

int OS_JavaScript::mouse_wheel_callback(double p_delta_x, double p_delta_y) {
	OS_JavaScript *os = get_singleton();

	if (!godot_js_display_canvas_is_focused() && !godot_js_is_ime_focused()) {
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

	if (p_delta_y < 0) {
		ev->set_button_index(BUTTON_WHEEL_UP);
	} else if (p_delta_y > 0) {
		ev->set_button_index(BUTTON_WHEEL_DOWN);
	} else if (p_delta_x > 0) {
		ev->set_button_index(BUTTON_WHEEL_LEFT);
	} else if (p_delta_x < 0) {
		ev->set_button_index(BUTTON_WHEEL_RIGHT);
	} else {
		return false;
	}

	// Different browsers give wildly different delta values, and we can't
	// interpret deltaMode, so use default value for wheel events' factor.

	int button_flag = 1 << (ev->get_button_index() - 1);

	ev->set_pressed(true);
	ev->set_button_mask(input->get_mouse_button_mask() | button_flag);
	input->parse_input_event(ev);

	Ref<InputEventMouseButton> release = ev->duplicate();
	release->set_pressed(false);
	release->set_button_mask(input->get_mouse_button_mask() & ~button_flag);
	input->parse_input_event(release);

	return true;
}

// Touch

bool OS_JavaScript::has_touchscreen_ui_hint() const {
	return godot_js_display_touchscreen_is_available();
}

void OS_JavaScript::touch_callback(int p_type, int p_count) {
	OS_JavaScript *os = get_singleton();
	// Resume audio context after input in case autoplay was denied.
	os->resume_audio();

	const JSTouchEvent &touch_event = os->touch_event;
	for (int i = 0; i < p_count; i++) {
		Point2 point(touch_event.coords[i * 2], touch_event.coords[i * 2 + 1]);
		if (p_type == 2) {
			// touchmove
			Ref<InputEventScreenDrag> ev;
			ev.instance();
			ev->set_index(touch_event.identifier[i]);
			ev->set_position(point);

			Point2 &prev = os->touches[i];
			ev->set_relative(ev->get_position() - prev);
			prev = ev->get_position();

			os->input->parse_input_event(ev);
		} else {
			// touchstart/touchend
			Ref<InputEventScreenTouch> ev;
			ev.instance();
			ev->set_index(touch_event.identifier[i]);
			ev->set_position(point);
			ev->set_pressed(p_type == 0);
			os->touches[i] = point;

			os->input->parse_input_event(ev);

			// Make sure to flush all events so we can call restricted APIs inside the event.
			os->input->flush_buffered_events();
		}
	}
}

// IME.
void OS_JavaScript::ime_callback(int p_type, const char *p_text) {
	OS_JavaScript *os = get_singleton();

	// Resume audio context after input in case autoplay was denied.
	os->resume_audio();

	switch (p_type) {
		case 0: {
			// IME start.
			os->ime_text = String();
			os->ime_selection = Vector2i();
			for (int i = os->key_event_pos - 1; i >= 0; i--) {
				// Delete last raw keydown event from query.
				if (os->key_event_buffer[i].pressed && os->key_event_buffer[i].raw) {
					os->key_event_buffer.remove(i);
					os->key_event_pos--;
					break;
				}
			}
			os->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
			os->ime_started = true;
		} break;
		case 1: {
			// IME update.
			if (os->ime_active && os->ime_started) {
				os->ime_text = String::utf8(p_text);
				os->ime_selection = Vector2i(os->ime_text.length(), os->ime_text.length());
				os->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
			}
		} break;
		case 2: {
			// IME commit.
			if (os->ime_active && os->ime_started) {
				os->ime_started = false;

				os->ime_text = String();
				os->ime_selection = Vector2i();
				os->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);

				String text = String::utf8(p_text);
				for (int i = 0; i < text.length(); i++) {
					OS_JavaScript::KeyEvent ke;

					ke.pressed = true;
					ke.echo = false;
					ke.raw = false;
					ke.keycode = 0;
					ke.physical_keycode = 0;
					ke.unicode = text[i];
					ke.mod = 0;

					if (os->key_event_pos >= os->key_event_buffer.size()) {
						os->key_event_buffer.resize(1 + os->key_event_pos);
					}
					os->key_event_buffer.write[os->key_event_pos++] = ke;
				}
			}
		} break;
		default:
			break;
	}

	os->process_keys();
	os->input->flush_buffered_events();
}

void OS_JavaScript::set_ime_active(const bool p_active) {
	ime_active = p_active;
	godot_js_set_ime_active(p_active);
}

void OS_JavaScript::set_ime_position(const Point2 &p_pos) {
	godot_js_set_ime_position(p_pos.x, p_pos.y);
}

Point2 OS_JavaScript::get_ime_selection() const {
	return ime_selection;
}

String OS_JavaScript::get_ime_text() const {
	return ime_text;
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
	int32_t pads = godot_js_input_gamepad_sample_count();
	int32_t s_btns_num = 0;
	int32_t s_axes_num = 0;
	int32_t s_standard = 0;
	float s_btns[16];
	float s_axes[10];
	for (int idx = 0; idx < pads; idx++) {
		int err = godot_js_input_gamepad_sample_get(idx, s_btns, &s_btns_num, s_axes, &s_axes_num, &s_standard);
		if (err) {
			continue;
		}
		for (int b = 0; b < s_btns_num; b++) {
			// Buttons 6 and 7 in the standard mapping need to be
			// axis to be handled as JOY_ANALOG by Godot.
			if (s_standard && (b == 6 || b == 7)) {
				input->joy_axis(idx, b, s_btns[b]);
			} else {
				input->joy_button(idx, b, s_btns[b]);
			}
		}
		for (int a = 0; a < s_axes_num; a++) {
			input->joy_axis(idx, a, s_axes[a]);
		}
	}
}

void OS_JavaScript::process_keys() {
	for (int i = 0; i < key_event_pos; i++) {
		const OS_JavaScript::KeyEvent &ke = key_event_buffer[i];

		Ref<InputEventKey> ev;
		ev.instance();
		ev->set_pressed(ke.pressed);
		ev->set_echo(ke.echo);
		ev->set_scancode(ke.keycode);
		ev->set_physical_scancode(ke.physical_keycode);
		ev->set_unicode(ke.unicode);
		if (ke.raw) {
			dom2godot_mod(ev, ke.mod);
		}

		input->parse_input_event(ev);
	}
	key_event_pos = 0;
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
	return audio_drivers.size();
}

const char *OS_JavaScript::get_audio_driver_name(int p_driver) const {
	if (audio_drivers.size() <= p_driver) {
		return "Unknown";
	}
	return audio_drivers[p_driver]->get_name();
}

// Clipboard
void OS_JavaScript::update_clipboard_callback(const char *p_text) {
	// Only call set_clipboard from OS (sets local clipboard)
	get_singleton()->OS::set_clipboard(String::utf8(p_text));
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
	video_mode = p_desired;
	// fullscreen_change_callback will correct this if the request is successful.
	video_mode.fullscreen = false;
	// Handle contextmenu, webglcontextlost, initial canvas setup.
	godot_js_display_setup_canvas(video_mode.width, video_mode.height, video_mode.fullscreen, is_hidpi_allowed() ? 1 : 0);

	swap_ok_cancel = godot_js_display_is_swap_ok_cancel() == 1;

	tts = GLOBAL_GET("audio/general/text_to_speech");

	EmscriptenWebGLContextAttributes attributes;
	emscripten_webgl_init_context_attributes(&attributes);
	attributes.alpha = GLOBAL_GET("display/window/per_pixel_transparency/allowed");
	attributes.antialias = false;
	attributes.explicitSwapControl = true;
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
			if (godot_js_display_has_webgl(2) && RasterizerGLES3::is_viable() == OK) {
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
			if (godot_js_display_has_webgl(1) && RasterizerGLES2::is_viable() == OK) {
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

	AudioDriverManager::initialize(p_audio_driver);
	visual_server = memnew(VisualServerRaster());
#ifndef NO_THREADS
	visual_server = memnew(VisualServerWrapMT(visual_server, false));
#endif
	input = memnew(InputDefault);

	// JS Input interface (js/libs/library_godot_input.js)
	godot_js_input_mouse_button_cb(&OS_JavaScript::mouse_button_callback);
	godot_js_input_mouse_move_cb(&OS_JavaScript::mouse_move_callback);
	godot_js_input_mouse_wheel_cb(&OS_JavaScript::mouse_wheel_callback);
	godot_js_input_touch_cb(&OS_JavaScript::touch_callback, touch_event.identifier, touch_event.coords);
	godot_js_input_key_cb(&OS_JavaScript::key_callback, key_event.code, key_event.key);
	godot_js_input_gamepad_cb(&OS_JavaScript::gamepad_callback);
	godot_js_input_paste_cb(&OS_JavaScript::update_clipboard_callback);
	godot_js_input_drop_files_cb(&OS_JavaScript::drop_files_callback);
	godot_js_set_ime_cb(&OS_JavaScript::ime_callback, &OS_JavaScript::key_callback, key_event.code, key_event.key);

	// JS Display interface (js/libs/library_godot_display.js)
	godot_js_display_fullscreen_cb(&OS_JavaScript::fullscreen_change_callback);
	godot_js_display_window_blur_cb(&window_blur_callback);
	godot_js_display_notification_cb(&OS_JavaScript::send_notification_callback,
			MainLoop::NOTIFICATION_WM_MOUSE_ENTER,
			MainLoop::NOTIFICATION_WM_MOUSE_EXIT,
			MainLoop::NOTIFICATION_WM_FOCUS_IN,
			MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	godot_js_display_vk_cb(&input_text_callback);

	visual_server->init();

	return OK;
}

void OS_JavaScript::input_text_callback(const char *p_text, int p_cursor) {
	OS_JavaScript *os = OS_JavaScript::get_singleton();
	if (!os || !os->get_main_loop()) {
		return;
	}
	os->get_main_loop()->input_text(String::utf8(p_text));
	Ref<InputEventKey> k;
	for (int i = 0; i < p_cursor; i++) {
		k.instance();
		k->set_pressed(true);
		k->set_echo(false);
		k->set_scancode(KEY_RIGHT);
		os->input->parse_input_event(k);
		k.instance();
		k->set_pressed(false);
		k->set_echo(false);
		k->set_scancode(KEY_RIGHT);
		os->input->parse_input_event(k);
	}
}

bool OS_JavaScript::has_virtual_keyboard() const {
	return godot_js_display_vk_available() != 0;
}

void OS_JavaScript::show_virtual_keyboard(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
	godot_js_display_vk_show(p_existing_text.utf8().get_data(), p_type, p_cursor_start, p_cursor_end);
}

void OS_JavaScript::hide_virtual_keyboard() {
	godot_js_display_vk_hide();
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
	AudioDriverJavaScript::resume();
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

	process_keys();
	input->flush_buffered_events();
	if (godot_js_input_gamepad_sample() == OK) {
		process_joypads();
	}

	return Main::iteration();
}

int OS_JavaScript::get_screen_dpi(int p_screen) const {
	return godot_js_display_screen_dpi_get();
}

float OS_JavaScript::get_screen_scale(int p_screen) const {
	return godot_js_display_pixel_ratio_get();
}

float OS_JavaScript::get_screen_max_scale() const {
	return get_screen_scale();
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
	for (int i = 0; i < audio_drivers.size(); i++) {
		memdelete(audio_drivers[i]);
	}
	audio_drivers.clear();
}

// Miscellaneous

Error OS_JavaScript::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
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

bool OS_JavaScript::is_process_running(const ProcessID &p_pid) const {
	return false;
}

int OS_JavaScript::get_processor_count() const {
	return godot_js_os_hw_concurrency_get();
}

String OS_JavaScript::get_unique_id() const {
	ERR_FAIL_V_MSG("", "OS::get_unique_id() is not available on the HTML5 platform.");
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

void OS_JavaScript::vibrate_handheld(int p_duration_ms) {
	godot_js_input_vibrate_handheld(p_duration_ms);
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

void OS_JavaScript::update_pwa_state_callback() {
	if (OS_JavaScript::get_singleton()) {
		OS_JavaScript::get_singleton()->pwa_is_waiting = true;
	}
	if (JavaScript::get_singleton()) {
		JavaScript::get_singleton()->emit_signal("pwa_update_available");
	}
}

Error OS_JavaScript::pwa_update() {
	return godot_js_pwa_update() ? FAILED : OK;
}

void OS_JavaScript::force_fs_sync() {
	if (is_userfs_persistent()) {
		idb_needs_sync = true;
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

	transparency_enabled = false;

	main_loop = NULL;
	visual_server = NULL;

	tts = false;
	swap_ok_cancel = false;
	idb_available = godot_js_os_fs_is_persistent() != 0;
	idb_needs_sync = false;
	idb_is_syncing = false;
	pwa_is_waiting = false;
	godot_js_pwa_cb(&OS_JavaScript::update_pwa_state_callback);

	if (AudioDriverJavaScript::is_available()) {
#ifdef NO_THREADS
		audio_drivers.push_back(memnew(AudioDriverScriptProcessor));
#endif
		audio_drivers.push_back(memnew(AudioDriverWorklet));
	}
	for (int i = 0; i < audio_drivers.size(); i++) {
		AudioDriverManager::add_driver(audio_drivers[i]);
	}

	Vector<Logger *> loggers;
	loggers.push_back(memnew(StdLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	FileAccessUnix::close_notification_func = file_access_close_callback;
}
