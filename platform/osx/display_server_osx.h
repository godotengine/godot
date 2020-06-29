/*************************************************************************/
/*  display_server_osx.h                                                 */
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

#ifndef DISPLAY_SERVER_OSX_H
#define DISPLAY_SERVER_OSX_H

#define BitMap _QDBitMap // Suppress deprecated QuickDraw definition.

#include "core/input/input.h"
#include "servers/display_server.h"

#if defined(OPENGL_ENABLED)
#include "context_gl_osx.h"
//TODO - reimplement OpenGLES
#endif

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/osx/vulkan_context_osx.h"
#endif

#include <AppKit/AppKit.h>
#include <AppKit/NSCursor.h>
#include <ApplicationServices/ApplicationServices.h>
#include <CoreVideo/CoreVideo.h>

#undef BitMap
#undef CursorShape

class DisplayServerOSX : public DisplayServer {
	GDCLASS(DisplayServerOSX, DisplayServer)

	_THREAD_SAFE_CLASS_

public:
#if defined(OPENGL_ENABLED)
	ContextGL_OSX *context_gles2;
#endif
#if defined(VULKAN_ENABLED)
	VulkanContextOSX *context_vulkan;
	RenderingDeviceVulkan *rendering_device_vulkan;
#endif

	const NSMenu *_get_menu_root(const String &p_menu_root) const;
	NSMenu *_get_menu_root(const String &p_menu_root);

	NSMenu *apple_menu = nullptr;
	NSMenu *dock_menu = nullptr;
	Map<String, NSMenu *> submenu;

	struct KeyEvent {
		WindowID window_id;
		unsigned int osx_state;
		bool pressed;
		bool echo;
		bool raw;
		uint32_t keycode;
		uint32_t physical_keycode;
		uint32_t unicode;
	};

	struct WarpEvent {
		NSTimeInterval timestamp;
		NSPoint delta;
	};

	List<WarpEvent> warp_events;
	NSTimeInterval last_warp = 0;

	Vector<KeyEvent> key_event_buffer;
	int key_event_pos;

	struct WindowData {
		id window_delegate;
		id window_object;
		id window_view;

		Vector<Vector2> mpath;

#if defined(OPENGL_ENABLED)
		ContextGL_OSX *context_gles2 = nullptr;
#endif
		Point2i mouse_pos;

		Size2i min_size;
		Size2i max_size;
		Size2i size;

		bool mouse_down_control = false;

		bool im_active = false;
		Size2i im_position;

		Callable rect_changed_callback;
		Callable event_callback;
		Callable input_event_callback;
		Callable input_text_callback;
		Callable drop_files_callback;

		ObjectID instance_id;

		WindowID transient_parent = INVALID_WINDOW_ID;
		Set<WindowID> transient_children;

		bool layered_window = false;
		bool fullscreen = false;
		bool on_top = false;
		bool borderless = false;
		bool resize_disabled = false;
		bool no_focus = false;
	};

	Point2i im_selection;
	String im_text;

	Map<WindowID, WindowData> windows;

	WindowID window_id_counter = MAIN_WINDOW_ID;

	WindowID _create_window(WindowMode p_mode, const Rect2i &p_rect);
	void _update_window(WindowData p_wd);
	void _send_window_event(const WindowData &wd, WindowEvent p_event);
	static void _dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);
	WindowID _find_window_id(id p_window);

	void _set_window_per_pixel_transparency_enabled(bool p_enabled, WindowID p_window);

	Point2i _get_screens_origin() const;
	Point2i _get_native_screen_position(int p_screen) const;

	void _push_input(const Ref<InputEvent> &p_event);
	void _process_key_events();
	void _release_pressed_events();

	String rendering_driver;

	id delegate;
	id autoreleasePool;
	CGEventSourceRef eventSource;

	CursorShape cursor_shape;
	NSCursor *cursors[CURSOR_MAX];
	Map<CursorShape, Vector<Variant>> cursors_cache;

	MouseMode mouse_mode;
	Point2i last_mouse_pos;
	uint32_t last_button_state;

	bool window_focused;
	bool drop_events;
	bool in_dispatch_input_event = false;

public:
	virtual bool has_feature(Feature p_feature) const override;
	virtual String get_name() const override;

	virtual void global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag = Variant()) override;
	virtual void global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag = Variant()) override;
	virtual void global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu) override;
	virtual void global_menu_add_separator(const String &p_menu_root) override;

	virtual bool global_menu_is_item_checked(const String &p_menu_root, int p_idx) const override;
	virtual bool global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const override;
	virtual Callable global_menu_get_item_callback(const String &p_menu_root, int p_idx) override;
	virtual Variant global_menu_get_item_tag(const String &p_menu_root, int p_idx) override;
	virtual String global_menu_get_item_text(const String &p_menu_root, int p_idx) override;
	virtual String global_menu_get_item_submenu(const String &p_menu_root, int p_idx) override;

	virtual void global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) override;
	virtual void global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) override;
	virtual void global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) override;
	virtual void global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) override;
	virtual void global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) override;
	virtual void global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) override;

	virtual int global_menu_get_item_count(const String &p_menu_root) const override;

	virtual void global_menu_remove_item(const String &p_menu_root, int p_idx) override;
	virtual void global_menu_clear(const String &p_menu_root) override;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!") override;
	virtual Error dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) override;
	virtual Error dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) override;

	virtual void mouse_set_mode(MouseMode p_mode) override;
	virtual MouseMode mouse_get_mode() const override;

	virtual void mouse_warp_to_position(const Point2i &p_to) override;
	virtual Point2i mouse_get_position() const override;
	virtual Point2i mouse_get_absolute_position() const override;
	virtual int mouse_get_button_state() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;

	virtual int get_screen_count() const override;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_max_scale() const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const override;

	virtual Vector<int> get_window_list() const override;

	virtual WindowID create_sub_window(WindowMode p_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i()) override;
	virtual void show_window(WindowID p_id) override;
	virtual void delete_sub_window(WindowID p_id) override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual void window_set_transient(WindowID p_window, WindowID p_parent) override;

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID) override;

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual bool can_any_window_draw() const override;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID) override;

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const override;

	virtual Point2i ime_get_selection() const override;
	virtual String ime_get_text() const override;

	virtual void cursor_set_shape(CursorShape p_shape) override;
	virtual CursorShape cursor_get_shape() const override;
	virtual void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2()) override;

	virtual bool get_swap_cancel_ok() override;

	virtual int keyboard_get_layout_count() const override;
	virtual int keyboard_get_current_layout() const override;
	virtual void keyboard_set_current_layout(int p_index) override;
	virtual String keyboard_get_layout_language(int p_index) const override;
	virtual String keyboard_get_layout_name(int p_index) const override;

	virtual void process_events() override;
	virtual void force_process_and_drop_events() override;

	virtual void release_rendering_thread() override;
	virtual void make_rendering_thread() override;
	virtual void swap_buffers() override;

	virtual void set_native_icon(const String &p_filename) override;
	virtual void set_icon(const Ref<Image> &p_icon) override;

	virtual void console_set_visible(bool p_enabled) override;
	virtual bool is_console_visible() const override;

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	static void register_osx_driver();

	DisplayServerOSX(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerOSX();
};

#endif // DISPLAY_SERVER_OSX_H
