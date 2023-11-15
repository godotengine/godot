/**************************************************************************/
/*  display_server_macos.h                                                */
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

#ifndef DISPLAY_SERVER_MACOS_H
#define DISPLAY_SERVER_MACOS_H

#include "core/input/input.h"
#include "servers/display_server.h"

#if defined(GLES3_ENABLED)
#include "gl_manager_macos_angle.h"
#include "gl_manager_macos_legacy.h"
#endif // GLES3_ENABLED

#if defined(VULKAN_ENABLED)
#include "vulkan_context_macos.h"

#include "drivers/vulkan/rendering_device_vulkan.h"
#endif // VULKAN_ENABLED

#define BitMap _QDBitMap // Suppress deprecated QuickDraw definition.

#import <AppKit/AppKit.h>
#import <AppKit/NSCursor.h>
#import <ApplicationServices/ApplicationServices.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOKit/pwr_mgt/IOPMLib.h>

#undef BitMap
#undef CursorShape

class DisplayServerMacOS : public DisplayServer {
	// No need to register with GDCLASS, it's platform-specific and nothing is added.

	_THREAD_SAFE_CLASS_

public:
	struct KeyEvent {
		WindowID window_id = INVALID_WINDOW_ID;
		unsigned int macos_state = false;
		bool pressed = false;
		bool echo = false;
		bool raw = false;
		Key keycode = Key::NONE;
		Key physical_keycode = Key::NONE;
		Key key_label = Key::NONE;
		uint32_t unicode = 0;
	};

	struct WindowData {
		id window_delegate;
		id window_object;
		id window_view;
		id window_button_view;

		Vector<Vector2> mpath;

		Point2i mouse_pos;

		Size2i min_size;
		Size2i max_size;
		Size2i size;
		Vector2i wb_offset = Vector2i(14, 14);

		NSRect last_frame_rect;

		bool im_active = false;
		Size2i im_position;

		Callable rect_changed_callback;
		Callable event_callback;
		Callable input_event_callback;
		Callable input_text_callback;
		Callable drop_files_callback;

		ObjectID instance_id;
		bool fs_transition = false;
		bool initial_size = true;

		WindowID transient_parent = INVALID_WINDOW_ID;
		bool exclusive = false;
		HashSet<WindowID> transient_children;

		bool layered_window = false;
		bool fullscreen = false;
		bool exclusive_fullscreen = false;
		bool on_top = false;
		bool borderless = false;
		bool resize_disabled = false;
		bool no_focus = false;
		bool is_popup = false;
		bool mpass = false;
		bool focused = false;
		bool is_visible = true;
		bool extend_to_title = false;

		Rect2i parent_safe_rect;
	};

	List<WindowID> popup_list;
	uint64_t time_since_popup = 0;

private:
#if defined(GLES3_ENABLED)
	GLManagerLegacy_MacOS *gl_manager_legacy = nullptr;
	GLManagerANGLE_MacOS *gl_manager_angle = nullptr;
#endif
#if defined(VULKAN_ENABLED)
	VulkanContextMacOS *context_vulkan = nullptr;
	RenderingDeviceVulkan *rendering_device_vulkan = nullptr;
#endif
	String rendering_driver;

	NSMenu *apple_menu = nullptr;
	NSMenu *dock_menu = nullptr;
	struct MenuData {
		Callable open;
		Callable close;
		NSMenu *menu = nullptr;
		bool is_open = false;
	};
	HashMap<String, MenuData> submenu;
	HashMap<NSMenu *, String> submenu_inv;

	struct WarpEvent {
		NSTimeInterval timestamp;
		NSPoint delta;
	};
	List<WarpEvent> warp_events;
	NSTimeInterval last_warp = 0;
	bool ignore_warp = false;

	Vector<KeyEvent> key_event_buffer;
	int key_event_pos = 0;

	id tts = nullptr;
	id menu_delegate = nullptr;

	Point2i im_selection;
	String im_text;

	CGEventSourceRef event_source;
	MouseMode mouse_mode = MOUSE_MODE_VISIBLE;
	BitField<MouseButtonMask> last_button_state;

	bool drop_events = false;
	bool in_dispatch_input_event = false;

	struct LayoutInfo {
		String name;
		String code;
	};
	Vector<LayoutInfo> kbd_layouts;
	int current_layout = 0;
	bool keyboard_layout_dirty = true;

	WindowID window_mouseover_id = INVALID_WINDOW_ID;
	WindowID last_focused_window = INVALID_WINDOW_ID;
	WindowID window_id_counter = MAIN_WINDOW_ID;
	float display_max_scale = 1.f;
	Point2i origin;
	bool displays_arrangement_dirty = true;
	bool is_resizing = false;

	CursorShape cursor_shape = CURSOR_ARROW;
	NSCursor *cursors[CURSOR_MAX];
	HashMap<CursorShape, Vector<Variant>> cursors_cache;

	HashMap<WindowID, WindowData> windows;

	IOPMAssertionID screen_keep_on_assertion = kIOPMNullAssertionID;

	struct MenuCall {
		Variant tag;
		Callable callback;
	};
	List<MenuCall> deferred_menu_calls;

	const NSMenu *_get_menu_root(const String &p_menu_root) const;
	NSMenu *_get_menu_root(const String &p_menu_root);
	bool _is_menu_opened(NSMenu *p_menu) const;

	WindowID _create_window(WindowMode p_mode, VSyncMode p_vsync_mode, const Rect2i &p_rect);
	void _update_window_style(WindowData p_wd);

	void _update_displays_arrangement();
	Point2i _get_screens_origin() const;
	Point2i _get_native_screen_position(int p_screen) const;
	static void _displays_arrangement_changed(CGDirectDisplayID display_id, CGDisplayChangeSummaryFlags flags, void *user_info);

	static void _dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);
	void _push_input(const Ref<InputEvent> &p_event);
	void _process_key_events();
	void _update_keyboard_layouts();
	static void _keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info);
	NSImage *_convert_to_nsimg(Ref<Image> &p_image) const;

	static NSCursor *_cursor_from_selector(SEL p_selector, SEL p_fallback = nil);

	bool _has_help_menu() const;
	NSMenuItem *_menu_add_item(const String &p_menu_root, const String &p_label, Key p_accel, int p_index, int *r_out);

public:
	NSMenu *get_dock_menu() const;
	void menu_callback(id p_sender);
	void menu_open(NSMenu *p_menu);
	void menu_close(NSMenu *p_menu);

	bool has_window(WindowID p_window) const;
	WindowData &get_window(WindowID p_window);

	void send_event(NSEvent *p_event);
	void send_window_event(const WindowData &p_wd, WindowEvent p_event);
	void release_pressed_events();
	void get_key_modifier_state(unsigned int p_macos_state, Ref<InputEventWithModifiers> r_state) const;
	void update_mouse_pos(WindowData &p_wd, NSPoint p_location_in_window);
	void push_to_key_event_buffer(const KeyEvent &p_event);
	void pop_last_key_event();
	void update_im_text(const Point2i &p_selection, const String &p_text);
	void set_last_focused_window(WindowID p_window);
	bool mouse_process_popups(bool p_close = false);
	void popup_open(WindowID p_window);
	void popup_close(WindowID p_window);
	void set_is_resizing(bool p_is_resizing);
	bool get_is_resizing() const;
	void reparent_check(WindowID p_window);
	WindowID _get_focused_window_or_popup() const;
	void mouse_enter_window(WindowID p_window);
	void mouse_exit_window(WindowID p_window);
	void update_presentation_mode();

	void window_destroy(WindowID p_window);
	void window_resize(WindowID p_window, int p_width, int p_height);
	void window_set_custom_window_buttons(WindowData &p_wd, bool p_enabled);
	void set_window_per_pixel_transparency_enabled(bool p_enabled, WindowID p_window);

	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;

	virtual void global_menu_set_popup_callbacks(const String &p_menu_root, const Callable &p_open_callback = Callable(), const Callable &p_close_callback = Callable()) override;

	virtual int global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu, int p_index = -1) override;
	virtual int global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_icon_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_icon_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_radio_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_icon_radio_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_multistate_item(const String &p_menu_root, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback = Callable(), const Callable &p_key_callback = Callable(), const Variant &p_tag = Variant(), Key p_accel = Key::NONE, int p_index = -1) override;
	virtual int global_menu_add_separator(const String &p_menu_root, int p_index = -1) override;

	virtual int global_menu_get_item_index_from_text(const String &p_menu_root, const String &p_text) const override;
	virtual int global_menu_get_item_index_from_tag(const String &p_menu_root, const Variant &p_tag) const override;

	virtual bool global_menu_is_item_checked(const String &p_menu_root, int p_idx) const override;
	virtual bool global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const override;
	virtual bool global_menu_is_item_radio_checkable(const String &p_menu_root, int p_idx) const override;
	virtual Callable global_menu_get_item_callback(const String &p_menu_root, int p_idx) const override;
	virtual Callable global_menu_get_item_key_callback(const String &p_menu_root, int p_idx) const override;
	virtual Variant global_menu_get_item_tag(const String &p_menu_root, int p_idx) const override;
	virtual String global_menu_get_item_text(const String &p_menu_root, int p_idx) const override;
	virtual String global_menu_get_item_submenu(const String &p_menu_root, int p_idx) const override;
	virtual Key global_menu_get_item_accelerator(const String &p_menu_root, int p_idx) const override;
	virtual bool global_menu_is_item_disabled(const String &p_menu_root, int p_idx) const override;
	virtual bool global_menu_is_item_hidden(const String &p_menu_root, int p_idx) const override;
	virtual String global_menu_get_item_tooltip(const String &p_menu_root, int p_idx) const override;
	virtual int global_menu_get_item_state(const String &p_menu_root, int p_idx) const override;
	virtual int global_menu_get_item_max_states(const String &p_menu_root, int p_idx) const override;
	virtual Ref<Texture2D> global_menu_get_item_icon(const String &p_menu_root, int p_idx) const override;
	virtual int global_menu_get_item_indentation_level(const String &p_menu_root, int p_idx) const override;

	virtual void global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) override;
	virtual void global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) override;
	virtual void global_menu_set_item_radio_checkable(const String &p_menu_root, int p_idx, bool p_checkable) override;
	virtual void global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) override;
	virtual void global_menu_set_item_key_callback(const String &p_menu_root, int p_idx, const Callable &p_key_callback) override;
	virtual void global_menu_set_item_hover_callbacks(const String &p_menu_root, int p_idx, const Callable &p_callback) override;
	virtual void global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) override;
	virtual void global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) override;
	virtual void global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) override;
	virtual void global_menu_set_item_accelerator(const String &p_menu_root, int p_idx, Key p_keycode) override;
	virtual void global_menu_set_item_disabled(const String &p_menu_root, int p_idx, bool p_disabled) override;
	virtual void global_menu_set_item_hidden(const String &p_menu_root, int p_idx, bool p_hidden) override;
	virtual void global_menu_set_item_tooltip(const String &p_menu_root, int p_idx, const String &p_tooltip) override;
	virtual void global_menu_set_item_state(const String &p_menu_root, int p_idx, int p_state) override;
	virtual void global_menu_set_item_max_states(const String &p_menu_root, int p_idx, int p_max_states) override;
	virtual void global_menu_set_item_icon(const String &p_menu_root, int p_idx, const Ref<Texture2D> &p_icon) override;
	virtual void global_menu_set_item_indentation_level(const String &p_menu_root, int p_idx, int p_level) override;

	virtual int global_menu_get_item_count(const String &p_menu_root) const override;

	virtual void global_menu_remove_item(const String &p_menu_root, int p_idx) override;
	virtual void global_menu_clear(const String &p_menu_root) override;

	virtual bool tts_is_speaking() const override;
	virtual bool tts_is_paused() const override;
	virtual TypedArray<Dictionary> tts_get_voices() const override;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int p_utterance_id = 0, bool p_interrupt = false) override;
	virtual void tts_pause() override;
	virtual void tts_resume() override;
	virtual void tts_stop() override;

	virtual bool is_dark_mode_supported() const override;
	virtual bool is_dark_mode() const override;
	virtual Color get_accent_color() const override;

	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) override;
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) override;

	virtual Error file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback) override;

	virtual void mouse_set_mode(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode() const override;

	bool update_mouse_wrap(WindowData &p_wd, NSPoint &r_delta, NSPoint &r_mpos, NSTimeInterval p_timestamp);
	virtual void warp_mouse(const Point2i &p_position) override;
	virtual Point2i mouse_get_position() const override;
	void mouse_set_button_state(BitField<MouseButtonMask> p_state);
	virtual BitField<MouseButtonMask> mouse_get_button_state() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;
	virtual Ref<Image> clipboard_get_image() const override;
	virtual bool clipboard_has() const override;
	virtual bool clipboard_has_image() const override;

	virtual int get_screen_count() const override;
	virtual int get_primary_screen() const override;
	virtual int get_keyboard_focus_screen() const override;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_max_scale() const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_refresh_rate(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Color screen_get_pixel(const Point2i &p_position) const override;
	virtual Ref<Image> screen_get_image(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual void screen_set_keep_on(bool p_enable) override;
	virtual bool screen_is_kept_on() const override;

	virtual Vector<int> get_window_list() const override;

	virtual WindowID create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i()) override;
	virtual void show_window(WindowID p_id) override;
	virtual void delete_sub_window(WindowID p_id) override;

	virtual WindowID window_get_active_popup() const override;
	virtual void window_set_popup_safe_rect(WindowID p_window, const Rect2i &p_rect) override;
	virtual Rect2i window_get_popup_safe_rect(WindowID p_window) const override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_title_size(const String &p_title, WindowID p_window) const override;
	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Point2i window_get_position_with_decorations(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent) override;
	virtual void window_set_exclusive(WindowID p_window, bool p_exclusive) override;

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_size_with_decorations(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_is_focused(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool can_any_window_draw() const override;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual int64_t window_get_native_handle(HandleType p_handle_type, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void gl_window_make_current(DisplayServer::WindowID p_window_id) override;

	virtual void window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual DisplayServer::VSyncMode window_get_vsync_mode(WindowID p_vsync_mode) const override;

	virtual bool window_maximize_on_title_dbl_click() const override;
	virtual bool window_minimize_on_title_dbl_click() const override;

	virtual void window_set_window_buttons_offset(const Vector2i &p_offset, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Vector3i window_get_safe_title_margins(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual Point2i ime_get_selection() const override;
	virtual String ime_get_text() const override;

	void cursor_update_shape();
	virtual void cursor_set_shape(CursorShape p_shape) override;
	virtual CursorShape cursor_get_shape() const override;
	virtual void cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2()) override;

	virtual bool get_swap_cancel_ok() override;

	virtual int keyboard_get_layout_count() const override;
	virtual int keyboard_get_current_layout() const override;
	virtual void keyboard_set_current_layout(int p_index) override;
	virtual String keyboard_get_layout_language(int p_index) const override;
	virtual String keyboard_get_layout_name(int p_index) const override;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const override;
	virtual Key keyboard_get_label_from_physical(Key p_keycode) const override;

	virtual void process_events() override;
	virtual void force_process_and_drop_events() override;

	virtual void release_rendering_thread() override;
	virtual void make_rendering_thread() override;
	virtual void swap_buffers() override;

	virtual void set_native_icon(const String &p_filename) override;
	virtual void set_icon(const Ref<Image> &p_icon) override;

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	static void register_macos_driver();

	DisplayServerMacOS(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error);
	~DisplayServerMacOS();
};

#endif // DISPLAY_SERVER_MACOS_H
