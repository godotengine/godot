/**************************************************************************/
/*  display_server_embedded_win.cpp                                       */
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

#ifdef GODOT_UWP_EMBED_ENABLED

#include "display_server_embedded_win.h"

#include "godot_uwp_embed.h"
#include "key_mapping_windows.h"

#include "core/input/input_event.h"
#include "core/os/os.h"

#if defined(RD_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif
#endif

DisplayServerEmbeddedWin *DisplayServerEmbeddedWin::singleton = nullptr;

// -----------------------------------------------------------------------
// Creation / registration
// -----------------------------------------------------------------------

DisplayServer *DisplayServerEmbeddedWin::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t /* p_parent_window */, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerEmbeddedWin(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert(
				"Your video card drivers seem not to support Direct3D 12, or no host SwapChainPanel was provided.\n\n"
				"The embedded display driver requires a host panel set via godot_uwp_set_swap_chain_panel() before setup.",
				"Unable to initialize embedded display server");
	}
	return ds;
}

Vector<String> DisplayServerEmbeddedWin::get_rendering_drivers_func() {
	Vector<String> drivers;
#if defined(RD_ENABLED) && defined(D3D12_ENABLED)
	drivers.push_back("d3d12");
#endif
	return drivers;
}

void DisplayServerEmbeddedWin::register_embedded_driver() {
	register_create_function("embedded", create_func, get_rendering_drivers_func);
}

DisplayServerEmbeddedWin::DisplayServerEmbeddedWin(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, Error &r_error) {
	r_error = ERR_CANT_CREATE;
	singleton = this;

	rendering_driver = p_rendering_driver;

	// Take over input dispatch from the headless base.
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	// The host tells us the panel size/scale before setup; fall back to the
	// project resolution otherwise.
	Size2i initial_size = GodotUwpEmbedState::initial_size;
	if (initial_size.width <= 0 || initial_size.height <= 0) {
		initial_size = p_resolution;
	}
	window_size = initial_size;
	composition_scale = GodotUwpEmbedState::initial_scale;

#if defined(RD_ENABLED) && defined(D3D12_ENABLED)
	ERR_FAIL_COND_MSG(rendering_driver != "d3d12", "The embedded display driver only supports the d3d12 rendering driver.");
	ERR_FAIL_NULL_MSG(GodotUwpEmbedState::swap_chain_panel_native, "No SwapChainPanel set. Call godot_uwp_set_swap_chain_panel() before engine setup.");

	rendering_context = memnew(RenderingContextDriverD3D12);
	if (rendering_context->initialize() != OK) {
		memdelete(rendering_context);
		rendering_context = nullptr;
		ERR_FAIL_MSG("Could not initialize D3D12.");
	}

	RenderingContextDriverD3D12::WindowPlatformData wpd;
	wpd.window = nullptr; // Windowless: composition swap chain bound to the host panel.
	wpd.swap_chain_panel_native = GodotUwpEmbedState::swap_chain_panel_native;

	Error err = rendering_context->window_create(DisplayServerEnums::MAIN_WINDOW_ID, &wpd);
	ERR_FAIL_COND_MSG(err != OK, "Could not create a D3D12 composition surface.");

	rendering_context->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, window_size.width, window_size.height);
	rendering_context->window_set_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID, p_vsync_mode);

	// Seed the panel composition scale so the swap chain is created with the
	// correct inverse DPI transform (otherwise output is scaled and cropped).
	((RenderingContextDriverD3D12 *)rendering_context)->surface_set_composition_scale(rendering_context->surface_get_from_window(DisplayServerEnums::MAIN_WINDOW_ID), composition_scale.x, composition_scale.y);

	rendering_device = memnew(RenderingDevice);
	if (rendering_device->initialize(rendering_context, DisplayServerEnums::MAIN_WINDOW_ID) != OK) {
		memdelete(rendering_device);
		rendering_device = nullptr;
		memdelete(rendering_context);
		rendering_context = nullptr;
		ERR_FAIL_MSG("Could not initialize the D3D12 rendering device.");
	}
	rendering_device->screen_create(DisplayServerEnums::MAIN_WINDOW_ID);

	RendererCompositorRD::make_current();
#else
	ERR_FAIL_MSG("The embedded display driver requires the D3D12 rendering driver to be compiled in.");
#endif

	r_error = OK;
}

DisplayServerEmbeddedWin::~DisplayServerEmbeddedWin() {
#if defined(RD_ENABLED)
	if (rendering_device) {
		memdelete(rendering_device);
		rendering_device = nullptr;
	}
	if (rendering_context) {
		memdelete(rendering_context);
		rendering_context = nullptr;
	}
#endif
	singleton = nullptr;
}

// -----------------------------------------------------------------------
// Event plumbing
// -----------------------------------------------------------------------

void DisplayServerEmbeddedWin::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerEmbeddedWin::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	Ref<InputEventFromWindow> event_from_window = p_event;
	WindowID window_id = DisplayServerEnums::INVALID_WINDOW_ID;
	if (event_from_window.is_valid()) {
		window_id = event_from_window->get_window_id();
	}
	DisplayServerEmbeddedWin *ds = (DisplayServerEmbeddedWin *)DisplayServer::get_singleton();
	ds->send_input_event(p_event, window_id);
}

void DisplayServerEmbeddedWin::send_input_event(const Ref<InputEvent> &p_event, WindowID p_id) const {
	if (p_id != DisplayServerEnums::INVALID_WINDOW_ID) {
		const Callable *cb = input_event_callbacks.getptr(p_id);
		if (cb) {
			_window_callback(*cb, p_event);
		}
	} else {
		for (const KeyValue<WindowID, Callable> &E : input_event_callbacks) {
			_window_callback(E.value, p_event);
		}
	}
}

void DisplayServerEmbeddedWin::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (p_callable.is_valid()) {
		p_callable.call(p_arg);
	}
}

void DisplayServerEmbeddedWin::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callbacks[p_window] = p_callable;
}

void DisplayServerEmbeddedWin::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callbacks[p_window] = p_callable;
}

void DisplayServerEmbeddedWin::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callbacks[p_window] = p_callable;
}

void DisplayServerEmbeddedWin::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callbacks[p_window] = p_callable;
}

void DisplayServerEmbeddedWin::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id[p_window] = p_instance;
}

ObjectID DisplayServerEmbeddedWin::window_get_attached_instance_id(WindowID p_window) const {
	const ObjectID *id = window_attached_instance_id.getptr(p_window);
	return id ? *id : ObjectID();
}

// -----------------------------------------------------------------------
// Host -> engine entry points (engine thread)
// -----------------------------------------------------------------------

void DisplayServerEmbeddedWin::host_resize(int p_width, int p_height) {
	if (p_width <= 0 || p_height <= 0) {
		return;
	}
	if (window_size.width == p_width && window_size.height == p_height) {
		return;
	}
	window_size = Size2i(p_width, p_height);

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, p_width, p_height);
	}
#endif

	const Callable *cb = window_resize_callbacks.getptr(DisplayServerEnums::MAIN_WINDOW_ID);
	if (cb) {
		_window_callback(*cb, Rect2i(Point2i(), window_size));
	}
}

void DisplayServerEmbeddedWin::host_set_composition_scale(float p_sx, float p_sy) {
	composition_scale = Vector2(MAX(0.25f, p_sx), MAX(0.25f, p_sy));

#if defined(RD_ENABLED) && defined(D3D12_ENABLED)
	if (rendering_context) {
		((RenderingContextDriverD3D12 *)rendering_context)->surface_set_composition_scale(rendering_context->surface_get_from_window(DisplayServerEnums::MAIN_WINDOW_ID), composition_scale.x, composition_scale.y);
	}
#endif
}

void DisplayServerEmbeddedWin::_set_modifier_state(Ref<InputEventWithModifiers> p_event) {
	p_event->set_shift_pressed(shift_down);
	p_event->set_ctrl_pressed(ctrl_down);
	p_event->set_alt_pressed(alt_down);
	p_event->set_meta_pressed(meta_down);
}

void DisplayServerEmbeddedWin::host_inject_mouse_button(MouseButton p_button, bool p_pressed, float p_x, float p_y, bool p_double_click) {
	if (p_pressed) {
		mouse_button_mask.set_flag(mouse_button_to_mask(p_button));
	} else {
		mouse_button_mask.clear_flag(mouse_button_to_mask(p_button));
	}
	last_mouse_pos = Point2(p_x, p_y);

	Ref<InputEventMouseButton> mb;
	mb.instantiate();
	mb->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	mb->set_button_index(p_button);
	mb->set_pressed(p_pressed);
	mb->set_double_click(p_double_click);
	mb->set_position(last_mouse_pos);
	mb->set_global_position(last_mouse_pos);
	mb->set_button_mask(mouse_button_mask);
	_set_modifier_state(mb);

	Input::get_singleton()->parse_input_event(mb);
}

void DisplayServerEmbeddedWin::host_inject_mouse_motion(float p_x, float p_y, float p_rel_x, float p_rel_y) {
	last_mouse_pos = Point2(p_x, p_y);

	Ref<InputEventMouseMotion> mm;
	mm.instantiate();
	mm->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	mm->set_position(last_mouse_pos);
	mm->set_global_position(last_mouse_pos);
	mm->set_relative(Vector2(p_rel_x, p_rel_y));
	mm->set_relative_screen_position(mm->get_relative());
	mm->set_button_mask(mouse_button_mask);
	_set_modifier_state(mm);

	Input::get_singleton()->parse_input_event(mm);
}

void DisplayServerEmbeddedWin::host_inject_mouse_wheel(float p_x, float p_y, float p_delta_x, float p_delta_y) {
	last_mouse_pos = Point2(p_x, p_y);

	// Vertical then horizontal, each as a press+release pair (Godot's wheel model).
	struct WheelAxis {
		float delta;
		MouseButton up;
		MouseButton down;
	};
	WheelAxis axes[2] = {
		{ p_delta_y, MouseButton::WHEEL_UP, MouseButton::WHEEL_DOWN },
		{ p_delta_x, MouseButton::WHEEL_RIGHT, MouseButton::WHEEL_LEFT },
	};

	for (const WheelAxis &axis : axes) {
		if (axis.delta == 0.0f) {
			continue;
		}
		MouseButton button = axis.delta > 0 ? axis.up : axis.down;

		Ref<InputEventMouseButton> mb;
		mb.instantiate();
		mb->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
		mb->set_button_index(button);
		mb->set_pressed(true);
		mb->set_factor(Math::abs(axis.delta));
		mb->set_position(last_mouse_pos);
		mb->set_global_position(last_mouse_pos);
		BitField<MouseButtonMask> mask = mouse_button_mask;
		mask.set_flag(mouse_button_to_mask(button));
		mb->set_button_mask(mask);
		_set_modifier_state(mb);
		Input::get_singleton()->parse_input_event(mb);

		Ref<InputEventMouseButton> mb_up = mb->duplicate();
		mb_up->set_pressed(false);
		mb_up->set_button_mask(mouse_button_mask);
		Input::get_singleton()->parse_input_event(mb_up);
	}
}

void DisplayServerEmbeddedWin::host_inject_key(unsigned int p_win_vk, bool p_pressed, bool p_echo, char32_t p_unicode) {
	Key keycode = KeyMappingWindows::get_keysym(p_win_vk);

	switch (keycode) {
		case Key::SHIFT:
			shift_down = p_pressed;
			break;
		case Key::CTRL:
			ctrl_down = p_pressed;
			break;
		case Key::ALT:
			alt_down = p_pressed;
			break;
		case Key::META:
			meta_down = p_pressed;
			break;
		default:
			break;
	}

	Ref<InputEventKey> k;
	k.instantiate();
	k->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	k->set_pressed(p_pressed);
	k->set_echo(p_echo);
	k->set_keycode(keycode);
	k->set_physical_keycode(keycode);
	k->set_key_label(keycode);
	if (p_unicode != 0) {
		k->set_unicode(p_unicode);
	}
	_set_modifier_state(k);

	Input::get_singleton()->parse_input_event(k);
}

// -----------------------------------------------------------------------
// Capabilities / metrics
// -----------------------------------------------------------------------

bool DisplayServerEmbeddedWin::has_feature(DisplayServerEnums::Feature p_feature) const {
	switch (p_feature) {
		case DisplayServerEnums::FEATURE_MOUSE:
		case DisplayServerEnums::FEATURE_CLIPBOARD:
		case DisplayServerEnums::FEATURE_HIDPI:
			return true;
		default:
			return false;
	}
}

String DisplayServerEmbeddedWin::get_name() const {
	return "embedded";
}

int DisplayServerEmbeddedWin::get_screen_count() const {
	return 1;
}

int DisplayServerEmbeddedWin::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerEmbeddedWin::screen_get_position(int p_screen) const {
	return Point2i();
}

Size2i DisplayServerEmbeddedWin::screen_get_size(int p_screen) const {
	return window_size;
}

Rect2i DisplayServerEmbeddedWin::screen_get_usable_rect(int p_screen) const {
	return Rect2i(Point2i(), window_size);
}

int DisplayServerEmbeddedWin::screen_get_dpi(int p_screen) const {
	return int(96.0f * composition_scale.x);
}

float DisplayServerEmbeddedWin::screen_get_scale(int p_screen) const {
	return composition_scale.x;
}

float DisplayServerEmbeddedWin::screen_get_max_scale() const {
	return composition_scale.x;
}

float DisplayServerEmbeddedWin::screen_get_refresh_rate(int p_screen) const {
	return 60.0f;
}

Vector<DisplayServerEnums::WindowID> DisplayServerEmbeddedWin::get_window_list() const {
	Vector<WindowID> list;
	list.push_back(DisplayServerEnums::MAIN_WINDOW_ID);
	return list;
}

DisplayServerEnums::WindowID DisplayServerEmbeddedWin::get_window_at_screen_position(const Point2i &p_position) const {
	return DisplayServerEnums::MAIN_WINDOW_ID;
}

Point2i DisplayServerEmbeddedWin::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerEmbeddedWin::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerEmbeddedWin::window_set_size(const Size2i p_size, WindowID p_window) {
	// Engine-initiated resizes are ignored: the host panel owns the size.
}

Size2i DisplayServerEmbeddedWin::window_get_size(WindowID p_window) const {
	return window_size;
}

Size2i DisplayServerEmbeddedWin::window_get_size_with_decorations(WindowID p_window) const {
	return window_size;
}

DisplayServerEnums::WindowMode DisplayServerEmbeddedWin::window_get_mode(WindowID p_window) const {
	return DisplayServerEnums::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerEmbeddedWin::window_is_focused(WindowID p_window) const {
	return true;
}

bool DisplayServerEmbeddedWin::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerEmbeddedWin::can_any_window_draw() const {
	return true;
}

int64_t DisplayServerEmbeddedWin::window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, WindowID p_window) const {
	return 0;
}

void DisplayServerEmbeddedWin::window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, WindowID p_window) {
#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServerEnums::VSyncMode DisplayServerEmbeddedWin::window_get_vsync_mode(WindowID p_window) const {
#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_vsync_mode(p_window);
	}
#endif
	return DisplayServerEnums::VSYNC_ENABLED;
}

Point2i DisplayServerEmbeddedWin::mouse_get_position() const {
	return last_mouse_pos;
}

void DisplayServerEmbeddedWin::cursor_set_shape(DisplayServerEnums::CursorShape p_shape) {
	cursor_shape = p_shape;
}

#endif // GODOT_UWP_EMBED_ENABLED
