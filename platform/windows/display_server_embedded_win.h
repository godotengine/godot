/**************************************************************************/
/*  display_server_embedded_win.h                                         */
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
/* included in all copies or substantial portions of the Software.       */
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

#ifdef GODOT_UWP_EMBED_ENABLED

// Windowless display server for hosting Godot inside a XAML SwapChainPanel
// (UWP or WinUI3). Creates NO Win32 windows: rendering goes to a DXGI
// composition swap chain that the D3D12 driver binds to the host panel via
// ISwapChainPanelNative::SetSwapChain, and all input is injected by the host
// through the godot_uwp_* C ABI (godot_uwp_embed.h).
//
// Inherits the all-stubs DisplayServerHeadless and only overrides what an
// embedded panel actually needs. Overrides mirror the 4.8 DisplayServer
// signatures (display-server enums live in DisplayServerEnums).

#include "servers/display/display_server_headless.h"

#include "core/input/input.h"

#if defined(RD_ENABLED)
class RenderingContextDriver;
class RenderingDevice;
#endif

class DisplayServerEmbeddedWin : public DisplayServerHeadless {
	GDSOFTCLASS(DisplayServerEmbeddedWin, DisplayServerHeadless);

	typedef DisplayServerEnums::WindowID WindowID;

	String rendering_driver;

#if defined(RD_ENABLED)
	RenderingContextDriver *rendering_context = nullptr;
	RenderingDevice *rendering_device = nullptr;
#endif

	HashMap<WindowID, Callable> window_event_callbacks;
	HashMap<WindowID, Callable> window_resize_callbacks;
	HashMap<WindowID, Callable> input_event_callbacks;
	HashMap<WindowID, Callable> input_text_callbacks;
	HashMap<WindowID, ObjectID> window_attached_instance_id;

	Size2i window_size = Size2i(1024, 600);
	Vector2 composition_scale = Vector2(1, 1);

	// Input state assembled from host-injected events.
	Point2 last_mouse_pos;
	BitField<MouseButtonMask> mouse_button_mask;
	bool shift_down = false;
	bool ctrl_down = false;
	bool alt_down = false;
	bool meta_down = false;

	DisplayServerEnums::CursorShape cursor_shape = DisplayServerEnums::CURSOR_ARROW;

	static DisplayServerEmbeddedWin *singleton;

	void _set_modifier_state(Ref<InputEventWithModifiers> p_event);
	void _window_callback(const Callable &p_callable, const Variant &p_arg) const;
	static void _dispatch_input_events(const Ref<InputEvent> &p_event);
	void send_input_event(const Ref<InputEvent> &p_event, WindowID p_id = DisplayServerEnums::MAIN_WINDOW_ID) const;

public:
	static DisplayServerEmbeddedWin *get_embedded_singleton() { return singleton; }

	static void register_embedded_driver();
	static DisplayServer *create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error);
	static Vector<String> get_rendering_drivers_func();

	// ------------------------------------------------------------------
	// Host -> engine (called on the engine thread by godot_uwp_embed.cpp)
	// ------------------------------------------------------------------
	void host_resize(int p_width, int p_height);
	void host_set_composition_scale(float p_sx, float p_sy);
	void host_inject_mouse_button(MouseButton p_button, bool p_pressed, float p_x, float p_y, bool p_double_click);
	void host_inject_mouse_motion(float p_x, float p_y, float p_rel_x, float p_rel_y);
	void host_inject_mouse_wheel(float p_x, float p_y, float p_delta_x, float p_delta_y);
	void host_inject_key(unsigned int p_win_vk, bool p_pressed, bool p_echo, char32_t p_unicode);

	// ------------------------------------------------------------------
	// DisplayServer overrides (signatures mirror DisplayServerHeadless / 4.8)
	// ------------------------------------------------------------------
	virtual bool has_feature(DisplayServerEnums::Feature p_feature) const override;
	virtual String get_name() const override;

	virtual void process_events() override;

	virtual int get_screen_count() const override;
	virtual int get_primary_screen() const override;
	virtual Point2i screen_get_position(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual Size2i screen_get_size(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual Rect2i screen_get_usable_rect(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual int screen_get_dpi(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_scale(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;
	virtual float screen_get_max_scale() const override;
	virtual float screen_get_refresh_rate(int p_screen = DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) const override;

	virtual Vector<WindowID> get_window_list() const override;
	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const override;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;

	virtual Point2i window_get_position(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual Point2i window_get_position_with_decorations(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual void window_set_size(const Size2i p_size, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual Size2i window_get_size(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual Size2i window_get_size_with_decorations(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual DisplayServerEnums::WindowMode window_get_mode(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual bool window_is_focused(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual bool window_can_draw(WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;
	virtual bool can_any_window_draw() const override;

	virtual int64_t window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) const override;

	virtual void window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, WindowID p_window = DisplayServerEnums::MAIN_WINDOW_ID) override;
	virtual DisplayServerEnums::VSyncMode window_get_vsync_mode(WindowID p_window) const override;

	virtual Point2i mouse_get_position() const override;
	virtual void cursor_set_shape(DisplayServerEnums::CursorShape p_shape) override;

	DisplayServerEmbeddedWin(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, Error &r_error);
	~DisplayServerEmbeddedWin();
};

#endif // GODOT_UWP_EMBED_ENABLED
