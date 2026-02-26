/**************************************************************************/
/*  display_server_wayland.h                                              */
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

#ifdef WAYLAND_ENABLED

#include "wayland/wayland_thread.h"

#include "servers/display/display_server.h"

class InputEvent;
class NativeMenu;

#ifdef RD_ENABLED
class RenderingDevice;
class RenderingContextDriver;
#endif

#ifdef GLES3_ENABLED
class EGLManager;
#endif

#ifdef DBUS_ENABLED
class FreeDesktopPortalDesktop;
class FreeDesktopAtSPIMonitor;
class FreeDesktopScreenSaver;
#endif

#ifdef SPEECHD_ENABLED
class TTS_Linux;
#endif

class DisplayServerWayland : public DisplayServer {
	GDSOFTCLASS(DisplayServerWayland, DisplayServer);

	struct WindowData {
		DisplayServerEnums::WindowID id = DisplayServerEnums::INVALID_WINDOW_ID;

		DisplayServerEnums::WindowID parent_id = DisplayServerEnums::INVALID_WINDOW_ID;

		// For popups.
		DisplayServerEnums::WindowID root_id = DisplayServerEnums::INVALID_WINDOW_ID;

		// For toplevels.
		List<DisplayServerEnums::WindowID> popup_stack;

		Rect2i rect;
		Size2i max_size;
		Size2i min_size;

		Rect2i safe_rect;

		bool emulate_vsync = false;

#ifdef GLES3_ENABLED
		struct wl_egl_window *wl_egl_window = nullptr;
#endif

		// Flags whether we have allocated a buffer through the video drivers.
		bool visible = false;

		DisplayServerEnums::VSyncMode vsync_mode = DisplayServerEnums::VSYNC_ENABLED;

		uint32_t flags = 0;

		DisplayServerEnums::WindowMode mode = DisplayServerEnums::WINDOW_MODE_WINDOWED;

		Callable rect_changed_callback;
		Callable window_event_callback;
		Callable input_event_callback;
		Callable drop_files_callback;
		Callable input_text_callback;

		String title;
		ObjectID instance_id;
	};

	struct CustomCursor {
		Ref<Resource> resource;
		Point2i hotspot;
	};

	enum class SuspendState {
		NONE, // Unsuspended.
		TIMEOUT, // Legacy fallback.
		CAPABILITY, // New "suspended" wm_capability flag.
	};

	DisplayServerEnums::CursorShape cursor_shape = DisplayServerEnums::CURSOR_ARROW;
	DisplayServerEnums::MouseMode mouse_mode = DisplayServerEnums::MOUSE_MODE_VISIBLE;
	DisplayServerEnums::MouseMode mouse_mode_base = DisplayServerEnums::MOUSE_MODE_VISIBLE;
	DisplayServerEnums::MouseMode mouse_mode_override = DisplayServerEnums::MOUSE_MODE_VISIBLE;
	bool mouse_mode_override_enabled = false;
	void _mouse_update_mode();

	HashMap<DisplayServerEnums::CursorShape, CustomCursor> custom_cursors;

	HashMap<DisplayServerEnums::WindowID, WindowData> windows;
	DisplayServerEnums::WindowID window_id_counter = DisplayServerEnums::MAIN_WINDOW_ID;

	WaylandThread wayland_thread;

	DisplayServerEnums::Context context;
	bool swap_cancel_ok = false;

	// NOTE: These are the based on DisplayServerEnums::WINDOW_FLAG_POPUP, which does NOT imply what it
	// seems. It's particularly confusing for our usecase, but just know that these
	// are the "take all input thx" windows while the `popup_stack` variable keeps
	// track of all the generic floating window concept.
	List<DisplayServerEnums::WindowID> popup_menu_list;
	BitField<MouseButtonMask> last_mouse_monitor_mask = MouseButtonMask::NONE;

	String ime_text;
	Vector2i ime_selection;

	SuspendState suspend_state = SuspendState::NONE;

	String rendering_driver;

#ifdef RD_ENABLED
	RenderingContextDriver *rendering_context = nullptr;
	RenderingDevice *rendering_device = nullptr;
#endif

#ifdef GLES3_ENABLED
	EGLManager *egl_manager = nullptr;
#endif

#ifdef SPEECHD_ENABLED
	TTS_Linux *tts = nullptr;
#endif
	NativeMenu *native_menu = nullptr;

#if DBUS_ENABLED
	FreeDesktopPortalDesktop *portal_desktop = nullptr;
	FreeDesktopAtSPIMonitor *atspi_monitor = nullptr;

	FreeDesktopScreenSaver *screensaver = nullptr;
	bool screensaver_inhibited = false;
#endif
	static String _get_app_id_from_context(DisplayServerEnums::Context p_context);

	void _send_window_event(DisplayServerEnums::WindowEvent p_event, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID);

	static void dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);

	void _update_window_rect(const Rect2i &p_rect, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID);

	void try_suspend();

	void initialize_tts() const;

public:
	virtual bool has_feature(DisplayServerEnums::Feature p_feature) const override;

	virtual String get_name() const override;

#ifdef SPEECHD_ENABLED
	virtual bool tts_is_speaking() const override;
	virtual bool tts_is_paused() const override;
	virtual TypedArray<Dictionary> tts_get_voices() const override;

	virtual void tts_speak(const String &p_text, const String &p_voice, int p_volume = 50, float p_pitch = 1.f, float p_rate = 1.f, int64_t p_utterance_id = 0, bool p_interrupt = false) override;
	virtual void tts_pause() override;
	virtual void tts_resume() override;
	virtual void tts_stop() override;
#endif

#ifdef DBUS_ENABLED
	virtual bool is_dark_mode_supported() const override;
	virtual bool is_dark_mode() const override;
	virtual Color get_accent_color() const override;
	virtual void set_system_theme_change_callback(const Callable &p_callable) override;

	virtual Error file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback, DisplayServerEnums::WindowID p_window_id) override;
	virtual Error file_dialog_with_options_show(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, DisplayServerEnums::FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, DisplayServerEnums::WindowID p_window_id) override;
#endif

	virtual void beep() const override;

	virtual void mouse_set_mode(DisplayServerEnums::MouseMode p_mode) override;
	virtual DisplayServerEnums::MouseMode mouse_get_mode() const override;
	virtual void mouse_set_mode_override(DisplayServerEnums::MouseMode p_mode) override;
	virtual DisplayServerEnums::MouseMode mouse_get_mode_override() const override;
	virtual void mouse_set_mode_override_enabled(bool p_override_enabled) override;
	virtual bool mouse_is_mode_override_enabled() const override;

	virtual void warp_mouse(const Point2i &p_to) override;
	virtual Point2i mouse_get_position() const override;
	virtual BitField<MouseButtonMask> mouse_get_button_state() const override;

	virtual void clipboard_set(const String &p_text) override;
	virtual String clipboard_get() const override;
	virtual Ref<Image> clipboard_get_image() const override;
	virtual void clipboard_set_primary(const String &p_text) override;
	virtual String clipboard_get_primary() const override;

	virtual int get_screen_count() const override;
	virtual int get_primary_screen() const override;
	virtual Point2i screen_get_position(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_refresh_rate(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;

	virtual void screen_set_keep_on(bool p_enable) override;
	virtual bool screen_is_kept_on() const override;

	virtual Vector<DisplayServerEnums::WindowID> get_window_list() const override;

	virtual DisplayServerEnums::WindowID create_sub_window(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i(), bool p_exclusive = false, DisplayServerEnums::WindowID p_transient_parent = DisplayServerEnums::INVALID_WINDOW_ID) override;
	virtual void show_window(DisplayServerEnums::WindowID p_id) override;
	virtual void delete_sub_window(DisplayServerEnums::WindowID p_id) override;

	virtual DisplayServerEnums::WindowID window_get_active_popup() const override;
	virtual void window_set_popup_safe_rect(DisplayServerEnums::WindowID p_window, const Rect2i &p_rect) override;
	virtual Rect2i window_get_popup_safe_rect(DisplayServerEnums::WindowID p_window) const override;

	virtual int64_t window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual DisplayServerEnums::WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_title(const String &p_title, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_mouse_passthrough(const Vector<Vector2> &p_region, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_window_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_drop_files_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual int window_get_current_screen(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual void window_set_current_screen(int p_screen, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual Point2i window_get_position_with_decorations(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual void window_set_position(const Point2i &p_position, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual void window_set_max_size(const Size2i p_size, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual Size2i window_get_max_size(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual void gl_window_make_current(DisplayServerEnums::WindowID p_window_id) override;

	virtual void window_set_transient(DisplayServerEnums::WindowID p_window_id, DisplayServerEnums::WindowID p_parent) override;

	virtual void window_set_min_size(const Size2i p_size, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual Size2i window_get_min_size(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_size(const Size2i p_size, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_size_with_decorations(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual float window_get_scale(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_mode(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual DisplayServerEnums::WindowMode window_get_mode(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual bool window_is_maximize_allowed(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_flag(DisplayServerEnums::WindowFlags p_flag, bool p_enabled, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual bool window_get_flag(DisplayServerEnums::WindowFlags p_flag, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_request_attention(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual void window_move_to_foreground(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual bool window_is_focused(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual bool window_can_draw(DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual bool can_any_window_draw() const override;

	virtual void window_set_ime_active(const bool p_active, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_ime_position(const Point2i &p_pos, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual int accessibility_should_increase_contrast() const override;
	virtual int accessibility_screen_reader_active() const override;

	virtual Point2i ime_get_selection() const override;
	virtual String ime_get_text() const override;

	virtual void window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, DisplayServerEnums::WindowID p_window_id = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual DisplayServerEnums::VSyncMode window_get_vsync_mode(DisplayServerEnums::WindowID p_window_id) const override;

	virtual void window_start_drag(DisplayServerEnums::WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_start_resize(DisplayServerEnums::WindowResizeEdge p_edge, DisplayServerEnums::WindowID p_window) override;

	virtual void cursor_set_shape(DisplayServerEnums::CursorShape p_shape) override;
	virtual DisplayServerEnums::CursorShape cursor_get_shape() const override;
	virtual void cursor_set_custom_image(const Ref<Resource> &p_cursor, DisplayServerEnums::CursorShape p_shape, const Vector2 &p_hotspot) override;

	virtual bool get_swap_cancel_ok() override;

	virtual Error embed_process(DisplayServerEnums::WindowID p_window, OS::ProcessID p_pid, const Rect2i &p_rect, bool p_visible, bool p_grab_focus) override;
	virtual Error request_close_embedded_process(OS::ProcessID p_pid) override;
	virtual Error remove_embedded_process(OS::ProcessID p_pid) override;
	virtual OS::ProcessID get_focused_process_id() override;

	virtual int keyboard_get_layout_count() const override;
	virtual int keyboard_get_current_layout() const override;
	virtual void keyboard_set_current_layout(int p_index) override;
	virtual String keyboard_get_layout_language(int p_index) const override;
	virtual String keyboard_get_layout_name(int p_index) const override;
	virtual Key keyboard_get_keycode_from_physical(Key p_keycode) const override;
	virtual Key keyboard_get_label_from_physical(Key p_keycode) const override;

	virtual bool color_picker(const Callable &p_callback) override;

	virtual void process_events() override;

	virtual void release_rendering_thread() override;
	virtual void swap_buffers() override;

	virtual void set_icon(const Ref<Image> &p_icon) override;

	virtual void set_context(DisplayServerEnums::Context p_context) override;

	virtual bool is_window_transparency_available() const override;

	static DisplayServer *create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Point2i *p_position, const Size2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	static void register_wayland_driver();

	DisplayServerWayland(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error);
	~DisplayServerWayland();
};

#endif // WAYLAND_ENABLED
