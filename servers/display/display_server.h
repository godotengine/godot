/**************************************************************************/
/*  display_server.h                                                      */
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

#pragma once

#include "core/input/input_enums.h"
#include "core/io/image.h"
#include "core/io/resource.h"
#include "core/object/object.h"
#include "core/os/keyboard.h"
#include "core/os/process_id.h"
#include "core/variant/callable.h"
#include "core/variant/typed_array.h"
#include "servers/display/display_server_enums.h"

class NativeMenu;
class Texture2D;

#undef CursorShape
namespace InputClassEnums {
enum MouseMode : int;
enum CursorShape : int;
} //namespace InputClassEnums

// Defined here so it can be forward-declared.
struct TTSUtterance {
	String text;
	String voice;
	int volume = 50;
	float pitch = 1.f;
	float rate = 1.f;
	int64_t id = 0;
};

class DisplayServer : public Object {
	GDCLASS(DisplayServer, Object)

	static DisplayServer *singleton;

public:
	_FORCE_INLINE_ static DisplayServer *get_singleton() {
		return singleton;
	}

	/* CREATE */

private:
	enum {
		MAX_SERVERS = 64
	};

	typedef DisplayServer *(*CreateFunction)(const String &, DisplayServerEnums::WindowMode, DisplayServerEnums::VSyncMode, uint32_t, const Point2i *, const Size2i &, int p_screen, DisplayServerEnums::Context, int64_t p_parent_window, Error &r_error);
	typedef Vector<String> (*GetRenderingDriversFunction)();

	struct DisplayServerCreate {
		const char *name;
		CreateFunction create_function;
		GetRenderingDriversFunction get_rendering_drivers_function;
	};

	static DisplayServerCreate server_create_functions[MAX_SERVERS];
	static int server_create_count;

public:
	static void register_create_function(const char *p_name, CreateFunction p_function, GetRenderingDriversFunction p_get_drivers);
	static int get_create_function_count();
	static const char *get_create_function_name(int p_index);
	static Vector<String> get_create_function_rendering_drivers(int p_index);
	static DisplayServer *create(int p_index, const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error);

	DisplayServer();
	~DisplayServer();

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	static void _bind_compatibility_methods();
#endif

	/* MAIN */

public:
	virtual String get_name() const = 0;

	virtual void set_context(DisplayServerEnums::Context p_context);
	virtual void set_native_icon(const String &p_filename);
	virtual void set_icon(const Ref<Image> &p_icon);

	virtual bool has_feature(DisplayServerEnums::Feature p_feature) const = 0;

	virtual void process_events() = 0;
	virtual void force_process_and_drop_events();

	virtual void release_rendering_thread();
	virtual void swap_buffers();

	virtual void beep() const;

	/* RENDERING DEVICE */

	// Used to cache the result of `can_create_rendering_device()` when RenderingDevice isn't currently being used.
	// This is done as creating a RenderingDevice is quite slow.
	static inline DisplayServerEnums::RenderingDeviceCreationStatus created_rendering_device = DisplayServerEnums::RenderingDeviceCreationStatus::UNKNOWN;
	static bool can_create_rendering_device();

	static inline DisplayServerEnums::RenderingDeviceCreationStatus supported_rendering_device = DisplayServerEnums::RenderingDeviceCreationStatus::UNKNOWN;
	static bool is_rendering_device_supported();

	/* GLOBAL MENU */

public:
	virtual void help_set_search_callbacks(const Callable &p_search_callback = Callable(), const Callable &p_action_callback = Callable());

#ifndef DISABLE_DEPRECATED
private:
	mutable HashMap<String, RID> menu_names;

	RID _get_rid_from_name(NativeMenu *p_nmenu, const String &p_menu_root) const;

public:
	virtual void global_menu_set_popup_callbacks(const String &p_menu_root, const Callable &p_open_callback = Callable(), const Callable &p_close_callback = Callable());

	virtual int global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu, int p_index = -1);
	virtual int global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_icon_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_icon_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_radio_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_icon_radio_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_multistate_item(const String &p_menu_root, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1);
	virtual int global_menu_add_separator(const String &p_menu_root, int p_index = -1);

	virtual int global_menu_get_item_index_from_text(const String &p_menu_root, const String &p_text) const;
	virtual int global_menu_get_item_index_from_tag(const String &p_menu_root, const Variant &p_tag) const;

	virtual bool global_menu_is_item_checked(const String &p_menu_root, int p_idx) const;
	virtual bool global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const;
	virtual bool global_menu_is_item_radio_checkable(const String &p_menu_root, int p_idx) const;
	virtual Callable global_menu_get_item_callback(const String &p_menu_root, int p_idx) const;
	virtual Callable global_menu_get_item_key_callback(const String &p_menu_root, int p_idx) const;
	virtual Variant global_menu_get_item_tag(const String &p_menu_root, int p_idx) const;
	virtual String global_menu_get_item_text(const String &p_menu_root, int p_idx) const;
	virtual String global_menu_get_item_submenu(const String &p_menu_root, int p_idx) const;
	virtual Key global_menu_get_item_accelerator(const String &p_menu_root, int p_idx) const;
	virtual bool global_menu_is_item_disabled(const String &p_menu_root, int p_idx) const;
	virtual bool global_menu_is_item_hidden(const String &p_menu_root, int p_idx) const;
	virtual String global_menu_get_item_tooltip(const String &p_menu_root, int p_idx) const;
	virtual int global_menu_get_item_state(const String &p_menu_root, int p_idx) const;
	virtual int global_menu_get_item_max_states(const String &p_menu_root, int p_idx) const;
	virtual Ref<Texture2D> global_menu_get_item_icon(const String &p_menu_root, int p_idx) const;
	virtual int global_menu_get_item_indentation_level(const String &p_menu_root, int p_idx) const;

	virtual void global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked);
	virtual void global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable);
	virtual void global_menu_set_item_radio_checkable(const String &p_menu_root, int p_idx, bool p_checkable);
	virtual void global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback);
	virtual void global_menu_set_item_key_callback(const String &p_menu_root, int p_idx, const Callable &p_key_callback);
	virtual void global_menu_set_item_hover_callbacks(const String &p_menu_root, int p_idx, const Callable &p_callback);
	virtual void global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag);
	virtual void global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text);
	virtual void global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu);
	virtual void global_menu_set_item_accelerator(const String &p_menu_root, int p_idx, Key p_keycode);
	virtual void global_menu_set_item_disabled(const String &p_menu_root, int p_idx, bool p_disabled);
	virtual void global_menu_set_item_hidden(const String &p_menu_root, int p_idx, bool p_hidden);
	virtual void global_menu_set_item_tooltip(const String &p_menu_root, int p_idx, const String &p_tooltip);
	virtual void global_menu_set_item_state(const String &p_menu_root, int p_idx, int p_state);
	virtual void global_menu_set_item_max_states(const String &p_menu_root, int p_idx, int p_max_states);
	virtual void global_menu_set_item_icon(const String &p_menu_root, int p_idx, const Ref<Texture2D> &p_icon);
	virtual void global_menu_set_item_indentation_level(const String &p_menu_root, int p_idx, int p_level);

	virtual int global_menu_get_item_count(const String &p_menu_root) const;

	virtual void global_menu_remove_item(const String &p_menu_root, int p_idx);
	virtual void global_menu_clear(const String &p_menu_root);

	virtual Dictionary global_menu_get_system_menu_roots() const;
#endif // DISABLE_DEPRECATED

	/* TTS */

private:
	Callable utterance_callback[DisplayServerEnums::TTS_UTTERANCE_MAX];

public:
	virtual bool tts_is_speaking() const;
	virtual bool tts_is_paused() const;
	virtual TypedArray<Dictionary> tts_get_voices() const;
	virtual PackedStringArray tts_get_voices_for_language(const String &p_language) const;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int64_t p_utterance_id = 0, bool p_interrupt = false);
	virtual void tts_pause();
	virtual void tts_resume();
	virtual void tts_stop();

	virtual void tts_set_utterance_callback(DisplayServerEnums::TTSUtteranceEvent p_event, const Callable &p_callable);
	virtual void tts_post_utterance_event(DisplayServerEnums::TTSUtteranceEvent p_event, int64_t p_id, int p_pos = 0);

	/* THEME */

	virtual bool is_dark_mode_supported() const { return false; }
	virtual bool is_dark_mode() const { return false; }
	virtual Color get_accent_color() const { return Color(0, 0, 0, 0); }
	virtual Color get_base_color() const { return Color(0, 0, 0, 0); }
	virtual void set_system_theme_change_callback(const Callable &p_callable) {}
	virtual void set_hardware_keyboard_connection_change_callback(const Callable &p_callable) {}

	/* MOUSE */

private:
	static void _input_set_mouse_mode(InputClassEnums::MouseMode p_mode);
	static InputClassEnums::MouseMode _input_get_mouse_mode();
	static void _input_set_mouse_mode_override(InputClassEnums::MouseMode p_mode);
	static InputClassEnums::MouseMode _input_get_mouse_mode_override();
	static void _input_set_mouse_mode_override_enabled(bool p_enabled);
	static bool _input_is_mouse_mode_override_enabled();
	static void _input_warp(const Vector2 &p_to_pos);
	static InputClassEnums::CursorShape _input_get_current_cursor_shape();
	static void _input_set_custom_mouse_cursor_func(const Ref<Resource> &, InputClassEnums::CursorShape, const Vector2 &p_hotspot);

protected:
	static Ref<Image> _get_cursor_image_from_resource(const Ref<Resource> &p_cursor, const Vector2 &p_hotspot);

public:
	virtual void mouse_set_mode(DisplayServerEnums::MouseMode p_mode);
	virtual DisplayServerEnums::MouseMode mouse_get_mode() const;
	virtual void mouse_set_mode_override(DisplayServerEnums::MouseMode p_mode);
	virtual DisplayServerEnums::MouseMode mouse_get_mode_override() const;
	virtual void mouse_set_mode_override_enabled(bool p_override_enabled);
	virtual bool mouse_is_mode_override_enabled() const;

	virtual void warp_mouse(const Point2i &p_position);
	virtual Point2i mouse_get_position() const;
	virtual BitField<MouseButtonMask> mouse_get_button_state() const;

	virtual void cursor_set_shape(DisplayServerEnums::CursorShape p_shape);
	virtual DisplayServerEnums::CursorShape cursor_get_shape() const;
	virtual void cursor_set_custom_image(const Ref<Resource> &p_cursor, DisplayServerEnums::CursorShape p_shape = DisplayServerEnums::CURSOR_ARROW, const Vector2 &p_hotspot = Vector2());

	/* KEYBOARD */

	virtual Point2i ime_get_selection() const;
	virtual String ime_get_text() const;

	virtual void virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect = Rect2(), DisplayServerEnums::VirtualKeyboardType p_type = DisplayServerEnums::KEYBOARD_TYPE_DEFAULT, int p_max_length = -1, int p_cursor_start = -1, int p_cursor_end = -1);
	virtual void virtual_keyboard_hide();

	// Returns height of the currently shown virtual keyboard (0 if keyboard is hidden).
	virtual int virtual_keyboard_get_height() const;

	virtual bool has_hardware_keyboard() const;

	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const;
	virtual Key keyboard_get_label_from_physical(Key p_keycode) const;

	/* TABLET */

	virtual int tablet_get_driver_count() const { return 1; }
	virtual String tablet_get_driver_name(int p_driver) const { return "default"; }
	virtual String tablet_get_current_driver() const { return "default"; }
	virtual void tablet_set_current_driver(const String &p_driver) {}

	/* CLIPBOARD */

	virtual void clipboard_set(const String &p_text);
	virtual String clipboard_get() const;
	virtual Ref<Image> clipboard_get_image() const;
	virtual bool clipboard_has() const;
	virtual bool clipboard_has_image() const;
	virtual void clipboard_set_primary(const String &p_text);
	virtual String clipboard_get_primary() const;

	/* SCREEN */

	const float SCREEN_REFRESH_RATE_FALLBACK = -1.0; // Returned by screen_get_refresh_rate if the method fails.

	virtual TypedArray<Rect2> get_display_cutouts() const { return TypedArray<Rect2>(); }
	virtual Rect2i get_display_safe_area() const { return screen_get_usable_rect(); }

	int _get_screen_index(int p_screen) const {
		switch (p_screen) {
			case DisplayServerEnums::SCREEN_WITH_MOUSE_FOCUS: {
				const Rect2i rect = Rect2i(mouse_get_position(), Vector2i(1, 1));
				return get_screen_from_rect(rect);
			} break;
			case DisplayServerEnums::SCREEN_WITH_KEYBOARD_FOCUS: {
				return get_keyboard_focus_screen();
			} break;
			case DisplayServerEnums::SCREEN_PRIMARY: {
				return get_primary_screen();
			} break;
			case DisplayServerEnums::SCREEN_OF_MAIN_WINDOW: {
				return window_get_current_screen(DisplayServerEnums::MAIN_WINDOW_ID);
			} break;
			default: {
				return p_screen;
			} break;
		}
	}

	virtual int get_screen_count() const = 0;
	virtual int get_primary_screen() const = 0;
	virtual int get_keyboard_focus_screen() const { return get_primary_screen(); }
	virtual int get_screen_from_rect(const Rect2 &p_rect) const;
	virtual Point2i screen_get_position(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual Size2i screen_get_size(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual Rect2i screen_get_usable_rect(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual int screen_get_dpi(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual float screen_get_scale(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const;
	virtual float screen_get_max_scale() const {
		float max_scale = 1.f;
		int screen_count = get_screen_count();
		for (int i = 0; i < screen_count; i++) {
			max_scale = std::fmax(max_scale, screen_get_scale(i));
		}
		return max_scale;
	}
	virtual float screen_get_refresh_rate(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const = 0;
	virtual Color screen_get_pixel(const Point2i &p_position) const { return Color(); }
	virtual Ref<Image> screen_get_image(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const { return Ref<Image>(); }
	virtual Ref<Image> screen_get_image_rect(const Rect2i &p_rect) const { return Ref<Image>(); }
	virtual bool is_touchscreen_available() const;

	virtual void screen_set_orientation(DisplayServerEnums::ScreenOrientation p_orientation, int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW);
	virtual DisplayServerEnums::ScreenOrientation screen_get_orientation(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const;

	virtual void screen_set_keep_on(bool p_enable); //disable screensaver
	virtual bool screen_is_kept_on() const;

	/* WINDOW */

private:
	static bool window_early_clear_override_enabled;
	static Color window_early_clear_override_color;

protected:
	static bool _get_window_early_clear_override(Color &r_color);

public:
	static void set_early_window_clear_color_override(bool p_enabled, Color p_color = Color(0, 0, 0, 0));

	virtual Vector<DisplayServerEnums::WindowID> get_window_list() const = 0;

	virtual DisplayServerEnums::WindowID create_sub_window(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i(), bool p_exclusive = false, DisplayServerEnums::WindowID p_transient_parent = DisplayServerEnums::INVALID_WINDOW_ID);
	virtual void show_window(DisplayServerEnums::WindowID p_id);
	virtual void delete_sub_window(DisplayServerEnums::WindowID p_id);

	virtual DisplayServerEnums::WindowID window_get_active_popup() const { return DisplayServerEnums::INVALID_WINDOW_ID; }
	virtual void window_set_popup_safe_rect(DisplayServerEnums::WindowID p_window, const Rect2i &p_rect) {}
	virtual Rect2i window_get_popup_safe_rect(DisplayServerEnums::WindowID p_window) const { return Rect2i(); }

	virtual int64_t window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual DisplayServerEnums::WindowID get_window_at_screen_position(const Point2i &p_position) const = 0;

	virtual void window_attach_instance_id(ObjectID p_instance, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0; // Note: internal method used by Window, do not expose.
	virtual ObjectID window_get_attached_instance_id(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;

	virtual void window_set_window_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual void window_set_input_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual void window_set_input_text_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;

	virtual void window_set_drop_files_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;

	virtual void window_set_title(const String &p_title, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_title_size(const String &p_title, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const { return Size2i(); }

	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);

	virtual int window_get_current_screen(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;
	virtual void window_set_current_screen(int p_screen, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;

	virtual Point2i window_get_position(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;
	virtual Point2i window_get_position_with_decorations(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;
	virtual void window_set_position(const Point2i &p_position, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;

	virtual void window_set_transient(DisplayServerEnums::WindowID p_window, DisplayServerEnums::WindowID p_parent) = 0;
	virtual void window_set_exclusive(DisplayServerEnums::WindowID p_window, bool p_exclusive);

	virtual void window_set_max_size(const Size2i p_size, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_max_size(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_set_min_size(const Size2i p_size, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_min_size(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_set_size(const Size2i p_size, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual Size2i window_get_size(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;
	virtual Size2i window_get_size_with_decorations(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual float window_get_scale(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const {
		int screen = window_get_current_screen(p_window);
		return screen_get_scale(screen);
	}

	virtual void window_set_mode(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual DisplayServerEnums::WindowMode window_get_mode(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual DisplayServerEnums::VSyncMode window_get_vsync_mode(DisplayServerEnums::WindowID p_window) const;

	virtual bool window_is_hdr_output_supported(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual void window_request_hdr_output(const bool p_enable, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual bool window_is_hdr_output_requested(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;
	virtual bool window_is_hdr_output_enabled(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual void window_set_hdr_output_reference_luminance(const float p_reference_luminance, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual float window_get_hdr_output_reference_luminance(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;
	virtual float window_get_hdr_output_current_reference_luminance(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual void window_set_hdr_output_max_luminance(const float p_max_luminance, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual float window_get_hdr_output_max_luminance(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;
	virtual float window_get_hdr_output_current_max_luminance(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual void window_set_icon(const Ref<Image> &p_icon, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}

	virtual float window_get_output_max_linear_value(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const;

	virtual bool window_is_maximize_allowed(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_set_flag(DisplayServerEnums::WindowFlags p_flag, bool p_enabled, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual bool window_get_flag(DisplayServerEnums::WindowFlags p_flag, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual void window_request_attention(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual void window_set_taskbar_progress_value(float p_value, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}
	virtual void window_set_taskbar_progress_state(DisplayServerEnums::ProgressState p_state, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}
	virtual void window_move_to_foreground(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) = 0;
	virtual bool window_is_focused(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual DisplayServerEnums::WindowID get_focused_window() const;

	virtual void window_set_window_buttons_offset(const Vector2i &p_offset, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}
	virtual Vector3i window_get_safe_title_margins(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const { return Vector3i(); }

	virtual bool window_can_draw(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const = 0;

	virtual bool can_any_window_draw() const = 0;

	virtual void window_set_ime_active(const bool p_active, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual void window_set_ime_position(const Point2i &p_pos, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID);

	virtual bool window_maximize_on_title_dbl_click() const { return false; }
	virtual bool window_minimize_on_title_dbl_click() const { return false; }

	virtual void window_start_drag(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}

	virtual void window_set_color(const Color &p_color) {}

	virtual void window_start_resize(DisplayServerEnums::WindowResizeEdge p_edge, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}

	// necessary for GL focus, may be able to use one of the existing functions for this, not sure yet
	virtual void gl_window_make_current(DisplayServerEnums::WindowID p_window_id);

	virtual bool is_window_transparency_available() const { return false; }

	/* PROCESS */

	virtual void enable_for_stealing_focus(ProcessID pid);

	virtual Error embed_process(DisplayServerEnums::WindowID p_window, ProcessID p_pid, const Rect2i &p_rect, bool p_visible, bool p_grab_focus);
	virtual Error request_close_embedded_process(ProcessID p_pid);
	virtual Error remove_embedded_process(ProcessID p_pid);
	virtual ProcessID get_focused_process_id();

	/* DIALOGS */

	virtual bool get_swap_cancel_ok();

	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback);
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback);

	virtual Error file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID);
	virtual Error file_dialog_with_options_show(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID);

#ifndef DISABLE_DEPRECATED
	Error _file_dialog_show_bind_compat_98194(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback);
	Error _file_dialog_with_options_show_bind_compat_98194(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback);
#endif

	virtual void show_emoji_and_symbol_picker() const;
	virtual bool color_picker(const Callable &p_callback);

	/* STATUS INDICATOR */

	virtual DisplayServerEnums::IndicatorID create_status_indicator(const Ref<Texture2D> &p_icon, const String &p_tooltip, const Callable &p_callback);
	virtual void status_indicator_set_icon(DisplayServerEnums::IndicatorID p_id, const Ref<Texture2D> &p_icon);
	virtual void status_indicator_set_tooltip(DisplayServerEnums::IndicatorID p_id, const String &p_tooltip);
	virtual void status_indicator_set_menu(DisplayServerEnums::IndicatorID p_id, const RID &p_menu_rid);
	virtual void status_indicator_set_callback(DisplayServerEnums::IndicatorID p_id, const Callable &p_callback);
	virtual Rect2 status_indicator_get_rect(DisplayServerEnums::IndicatorID p_id) const;
	virtual void delete_status_indicator(DisplayServerEnums::IndicatorID p_id);

	/* OUTPUT */

private:
	LocalVector<ObjectID> additional_outputs;

public:
	void register_additional_output(Object *p_output);
	void unregister_additional_output(Object *p_output);
	bool has_additional_outputs() const { return additional_outputs.size() > 0; }

	/* PICTURE_IN_PICTURE */
	virtual bool is_in_pip_mode(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) { return false; }
	virtual void pip_mode_enter(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}
	virtual void pip_mode_set_aspect_ratio(int p_numerator, int p_denominator, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}
	virtual void pip_mode_set_auto_enter_on_background(bool p_auto_enter_on_background, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) {}

	/* ACCESSIBILITY */

#ifndef DISABLE_DEPRECATED
private:
	RID _accessibility_create_sub_text_edit_elements_bind_compat_113459(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos = -1);
#endif

public:
	virtual int accessibility_should_increase_contrast() const { return -1; }
	virtual int accessibility_should_reduce_animation() const { return -1; }
	virtual int accessibility_should_reduce_transparency() const { return -1; }
	virtual int accessibility_screen_reader_active() const { return -1; }

#ifndef DISABLE_DEPRECATED
	virtual RID accessibility_create_element(DisplayServerEnums::WindowID p_window_id, DisplayServerEnums::AccessibilityRole p_role);
	virtual RID accessibility_create_sub_element(const RID &p_parent_rid, DisplayServerEnums::AccessibilityRole p_role, int p_insert_pos = -1);
	virtual RID accessibility_create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos = -1, bool p_is_last_line = false);
	virtual bool accessibility_has_element(const RID &p_id) const;
	virtual void accessibility_free_element(const RID &p_id);

	virtual void accessibility_element_set_meta(const RID &p_id, const Variant &p_meta);
	virtual Variant accessibility_element_get_meta(const RID &p_id) const;

	virtual void accessibility_update_if_active(const Callable &p_callable);

	virtual void accessibility_update_set_focus(const RID &p_id);
	virtual RID accessibility_get_window_root(DisplayServerEnums::WindowID p_window_id) const;

	virtual void accessibility_set_window_rect(DisplayServerEnums::WindowID p_window_id, const Rect2 &p_rect_out, const Rect2 &p_rect_in);
	virtual void accessibility_set_window_focused(DisplayServerEnums::WindowID p_window_id, bool p_focused);
	virtual void accessibility_set_window_callbacks(DisplayServerEnums::WindowID p_window_id, const Callable &p_activate_callable, const Callable &p_deativate_callable); // Note: internal method used by Window, do not expose.
	virtual void accessibility_window_activation_completed(DisplayServerEnums::WindowID p_window_id); // Note: internal method used by Window, do not expose.
	virtual void accessibility_window_deactivation_completed(DisplayServerEnums::WindowID p_window_id); // Note: internal method used by Window, do not expose.

	virtual void accessibility_update_set_role(const RID &p_id, DisplayServerEnums::AccessibilityRole p_role);
	virtual void accessibility_update_set_name(const RID &p_id, const String &p_name);
	virtual void accessibility_update_set_extra_info(const RID &p_id, const String &p_name_extra_info);
	virtual void accessibility_update_set_description(const RID &p_id, const String &p_description);
	virtual void accessibility_update_set_value(const RID &p_id, const String &p_value);
	virtual void accessibility_update_set_tooltip(const RID &p_id, const String &p_tooltip);
	virtual void accessibility_update_set_bounds(const RID &p_id, const Rect2 &p_rect);
	virtual void accessibility_update_set_transform(const RID &p_id, const Transform2D &p_transform);
	virtual void accessibility_update_add_child(const RID &p_id, const RID &p_child_id);
	virtual void accessibility_update_add_related_controls(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_add_related_details(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_add_related_described_by(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_add_related_flow_to(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_add_related_labeled_by(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_add_related_radio_group(const RID &p_id, const RID &p_related_id);
	virtual void accessibility_update_set_active_descendant(const RID &p_id, const RID &p_other_id);
	virtual void accessibility_update_set_next_on_line(const RID &p_id, const RID &p_other_id);
	virtual void accessibility_update_set_previous_on_line(const RID &p_id, const RID &p_other_id);
	virtual void accessibility_update_set_member_of(const RID &p_id, const RID &p_group_id);
	virtual void accessibility_update_set_in_page_link_target(const RID &p_id, const RID &p_other_id);
	virtual void accessibility_update_set_error_message(const RID &p_id, const RID &p_other_id);
	virtual void accessibility_update_set_live(const RID &p_id, DisplayServerEnums::AccessibilityLiveMode p_live);
	virtual void accessibility_update_add_action(const RID &p_id, DisplayServerEnums::AccessibilityAction p_action, const Callable &p_callable);
	virtual void accessibility_update_add_custom_action(const RID &p_id, int p_action_id, const String &p_action_description);
	virtual void accessibility_update_set_table_row_count(const RID &p_id, int p_count);
	virtual void accessibility_update_set_table_column_count(const RID &p_id, int p_count);
	virtual void accessibility_update_set_table_row_index(const RID &p_id, int p_index);
	virtual void accessibility_update_set_table_column_index(const RID &p_id, int p_index);
	virtual void accessibility_update_set_table_cell_position(const RID &p_id, int p_row_index, int p_column_index);
	virtual void accessibility_update_set_table_cell_span(const RID &p_id, int p_row_span, int p_column_span);
	virtual void accessibility_update_set_list_item_count(const RID &p_id, int p_size);
	virtual void accessibility_update_set_list_item_index(const RID &p_id, int p_index);
	virtual void accessibility_update_set_list_item_level(const RID &p_id, int p_level);
	virtual void accessibility_update_set_list_item_selected(const RID &p_id, bool p_selected);
	virtual void accessibility_update_set_list_item_expanded(const RID &p_id, bool p_expanded);
	virtual void accessibility_update_set_popup_type(const RID &p_id, DisplayServerEnums::AccessibilityPopupType p_popup);
	virtual void accessibility_update_set_checked(const RID &p_id, bool p_checekd);
	virtual void accessibility_update_set_num_value(const RID &p_id, double p_position);
	virtual void accessibility_update_set_num_range(const RID &p_id, double p_min, double p_max);
	virtual void accessibility_update_set_num_step(const RID &p_id, double p_step);
	virtual void accessibility_update_set_num_jump(const RID &p_id, double p_jump);
	virtual void accessibility_update_set_scroll_x(const RID &p_id, double p_position);
	virtual void accessibility_update_set_scroll_x_range(const RID &p_id, double p_min, double p_max);
	virtual void accessibility_update_set_scroll_y(const RID &p_id, double p_position);
	virtual void accessibility_update_set_scroll_y_range(const RID &p_id, double p_min, double p_max);
	virtual void accessibility_update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline);
	virtual void accessibility_update_set_text_align(const RID &p_id, HorizontalAlignment p_align);
	virtual void accessibility_update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int p_start_char, const RID &p_text_end_id, int p_end_char);
	virtual void accessibility_update_set_flag(const RID &p_id, DisplayServerEnums::AccessibilityFlags p_flag, bool p_value);
	virtual void accessibility_update_set_classname(const RID &p_id, const String &p_classname);
	virtual void accessibility_update_set_placeholder(const RID &p_id, const String &p_placeholder);
	virtual void accessibility_update_set_language(const RID &p_id, const String &p_language);
	virtual void accessibility_update_set_text_orientation(const RID &p_id, bool p_vertical);
	virtual void accessibility_update_set_list_orientation(const RID &p_id, bool p_vertical);
	virtual void accessibility_update_set_shortcut(const RID &p_id, const String &p_shortcut);
	virtual void accessibility_update_set_url(const RID &p_id, const String &p_url);
	virtual void accessibility_update_set_role_description(const RID &p_id, const String &p_description);
	virtual void accessibility_update_set_state_description(const RID &p_id, const String &p_description);
	virtual void accessibility_update_set_color_value(const RID &p_id, const Color &p_color);
	virtual void accessibility_update_set_background_color(const RID &p_id, const Color &p_color);
	virtual void accessibility_update_set_foreground_color(const RID &p_id, const Color &p_color);
#endif // DISABLE_DEPRECATED
};

VARIANT_ENUM_CAST_EXT(DisplayServerEnums::Feature, DisplayServer::Feature)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::TTSUtteranceEvent, DisplayServer::TTSUtteranceEvent)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::MouseMode, DisplayServer::MouseMode)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::CursorShape, DisplayServer::CursorShape)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::VirtualKeyboardType, DisplayServer::VirtualKeyboardType)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::ScreenOrientation, DisplayServer::ScreenOrientation)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::HandleType, DisplayServer::HandleType)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::WindowMode, DisplayServer::WindowMode)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::WindowFlags, DisplayServer::WindowFlags)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::WindowEvent, DisplayServer::WindowEvent)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::WindowResizeEdge, DisplayServer::WindowResizeEdge)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::VSyncMode, DisplayServer::VSyncMode)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::ProgressState, DisplayServer::ProgressState)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::FileDialogMode, DisplayServer::FileDialogMode)

#ifndef DISABLE_DEPRECATED
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityRole, DisplayServer::AccessibilityRole)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityPopupType, DisplayServer::AccessibilityPopupType)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityFlags, DisplayServer::AccessibilityFlags)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityAction, DisplayServer::AccessibilityAction)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityLiveMode, DisplayServer::AccessibilityLiveMode)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityScrollUnit, DisplayServer::AccessibilityScrollUnit)
VARIANT_ENUM_CAST_EXT(DisplayServerEnums::AccessibilityScrollHint, DisplayServer::AccessibilityScrollHint)
#endif // DISABLE_DEPRECATED
