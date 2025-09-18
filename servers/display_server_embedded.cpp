/**************************************************************************/
/*  display_server_embedded.cpp                                           */
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

#include "display_server_embedded.h"

#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"
#include "servers/rendering/gl_manager.h"

#ifdef RD_ENABLED
#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/godot_vulkan.h"
#endif // VULKAN_ENABLED
#endif // RD_ENABLED

Ref<RenderingNativeSurface> DisplayServerEmbedded::native_surface = nullptr;

DisplayServerEmbedded *DisplayServerEmbedded::get_singleton() {
	return (DisplayServerEmbedded *)DisplayServer::get_singleton();
}

void DisplayServerEmbedded::set_native_surface(Ref<RenderingNativeSurface> p_native_surface) {
	native_surface = p_native_surface;
}

void DisplayServerEmbedded::set_screen_get_dpi_callback(Callable p_callback) {
	screen_get_dpi_callback = p_callback;
}

void DisplayServerEmbedded::set_screen_get_size_callback(Callable p_callback) {
	screen_get_size_callback = p_callback;
}

void DisplayServerEmbedded::set_screen_get_scale_callback(Callable p_callback) {
	screen_get_scale_callback = p_callback;
}

void DisplayServerEmbedded::_bind_methods() {
	ClassDB::bind_static_method("DisplayServerEmbedded", D_METHOD("set_native_surface", "native_surface"), &DisplayServerEmbedded::set_native_surface);
	ClassDB::bind_static_method("DisplayServerEmbedded", D_METHOD("get_singleton"), &DisplayServerEmbedded::get_singleton);
	ClassDB::bind_static_method("DisplayServerEmbedded", D_METHOD("set_screen_get_dpi_callback", "callback"), &DisplayServerEmbedded::set_screen_get_dpi_callback);
	ClassDB::bind_static_method("DisplayServerEmbedded", D_METHOD("set_screen_get_size_callback", "callback"), &DisplayServerEmbedded::set_screen_get_size_callback);
	ClassDB::bind_static_method("DisplayServerEmbedded", D_METHOD("set_screen_get_scale_callback", "callback"), &DisplayServerEmbedded::set_screen_get_scale_callback);
	ClassDB::bind_method(D_METHOD("resize_window", "size", "id"), &DisplayServerEmbedded::resize_window);
	ClassDB::bind_method(D_METHOD("set_content_scale", "content_scale"), &DisplayServerEmbedded::set_content_scale);
	ClassDB::bind_method(D_METHOD("touch_press", "idx", "x", "y", "pressed", "double_click", "window"), &DisplayServerEmbedded::touch_press);
	ClassDB::bind_method(D_METHOD("touch_drag", "idx", "prev_x", "prev_y", "x", "y", "pressure", "tilt", "window"), &DisplayServerEmbedded::touch_drag);
	ClassDB::bind_method(D_METHOD("touches_canceled", "idx", "window"), &DisplayServerEmbedded::touches_canceled);
	ClassDB::bind_method(D_METHOD("key", "key", "char", "unshifted", "physical", "modifiers", "pressed", "window"), &DisplayServerEmbedded::key, DEFVAL(MAIN_WINDOW_ID));
}

DisplayServerEmbedded::DisplayServerEmbedded(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, Error &r_error) {
	ERR_FAIL_NULL_MSG(native_surface, "Native surface has not been set.");

	rendering_driver = p_rendering_driver;

	native_menu = memnew(NativeMenu);

#if defined(RD_ENABLED)
	rendering_context = nullptr;
	rendering_device = nullptr;

	if (rendering_driver == "vulkan" || rendering_driver == "metal" || rendering_driver == "d3d12") {
		rendering_context = native_surface->create_rendering_context(rendering_driver);
	}

	if (rendering_context) {
		if (rendering_context->initialize() != OK) {
			ERR_PRINT(vformat("Failed to initialize %s context", rendering_driver));
			memdelete(rendering_context);
			rendering_context = nullptr;
			return;
		}

		if (create_native_window(native_surface) != MAIN_WINDOW_ID) {
			ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}

		rendering_device = memnew(RenderingDevice);
		rendering_device->initialize(rendering_context, MAIN_WINDOW_ID);
		rendering_device->screen_create(MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver.contains("opengl3")) {
		gl_manager = native_surface->create_gl_manager(rendering_driver);

		if (gl_manager->initialize() != OK || gl_manager->open_display(nullptr) != OK) {
			memdelete(gl_manager);
			gl_manager = nullptr;
		}
		if (create_native_window(native_surface) != MAIN_WINDOW_ID) {
			ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			r_error = ERR_UNAVAILABLE;
			return;
		}
	}
#endif

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerEmbedded::~DisplayServerEmbedded() {
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}

#if defined(RD_ENABLED)
	if (rendering_device) {
		rendering_device->screen_free(MAIN_WINDOW_ID);
		memdelete(rendering_device);
		rendering_device = nullptr;
	}

	if (rendering_context) {
		rendering_context->window_destroy(MAIN_WINDOW_ID);
		memdelete(rendering_context);
		rendering_context = nullptr;
	}
#endif

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		memdelete(gl_manager);
		gl_manager = nullptr;
	}
#endif
	// Release native surface
	native_surface = nullptr;
}

DisplayServer *DisplayServerEmbedded::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t /* p_parent_window */, Error &r_error) {
	return memnew(DisplayServerEmbedded(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, r_error));
}

Vector<String> DisplayServerEmbedded::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(METAL_ENABLED)
	drivers.push_back("metal");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl3");
#endif

	return drivers;
}

void DisplayServerEmbedded::register_embedded_driver() {
	register_create_function("embedded", create_func, get_rendering_drivers_func);
}

// MARK: Events

void DisplayServerEmbedded::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callbacks[p_window] = p_callable;
}

void DisplayServerEmbedded::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callbacks[p_window] = p_callable;
}
void DisplayServerEmbedded::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callbacks[p_window] = p_callable;
}

void DisplayServerEmbedded::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callbacks[p_window] = p_callable;
}

void DisplayServerEmbedded::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	// Not supported
}

void DisplayServerEmbedded::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerEmbedded::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	Ref<InputEventFromWindow> event_from_window = p_event;
	WindowID window_id = INVALID_WINDOW_ID;
	if (event_from_window.is_valid()) {
		window_id = event_from_window->get_window_id();
	}
	DisplayServerEmbedded::get_singleton()->send_input_event(p_event, window_id);
}

void DisplayServerEmbedded::send_input_event(const Ref<InputEvent> &p_event, WindowID p_id) const {
	if (p_id != INVALID_WINDOW_ID) {
		_window_callback(input_event_callbacks[p_id], p_event);
	} else {
		for (const KeyValue<WindowID, Callable> &E : input_event_callbacks) {
			_window_callback(E.value, p_event);
		}
	}
}

void DisplayServerEmbedded::send_input_text(const String &p_text, WindowID p_id) const {
	_window_callback(input_text_callbacks[p_id], p_text);
}

void DisplayServerEmbedded::send_window_event(DisplayServer::WindowEvent p_event, WindowID p_id) const {
	_window_callback(window_event_callbacks[p_id], int(p_event));
}

void DisplayServerEmbedded::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (!p_callable.is_null()) {
		p_callable.call(p_arg);
	}
}

// MARK: - Input

// MARK: Touches

void DisplayServerEmbedded::touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_double_click, DisplayServer::WindowID p_window) {
	Ref<InputEventScreenTouch> ev;
	ev.instantiate();

	ev->set_window_id(p_window);
	ev->set_index(p_idx);
	ev->set_pressed(p_pressed);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_double_tap(p_double_click);
	perform_event(ev);
}

void DisplayServerEmbedded::touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, float p_pressure, Vector2 p_tilt, DisplayServer::WindowID p_window) {
	Ref<InputEventScreenDrag> ev;
	ev.instantiate();
	ev->set_window_id(p_window);
	ev->set_index(p_idx);
	ev->set_pressure(p_pressure);
	ev->set_tilt(p_tilt);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
	ev->set_relative_screen_position(ev->get_relative());
	perform_event(ev);
}

void DisplayServerEmbedded::perform_event(const Ref<InputEvent> &p_event) {
	Input::get_singleton()->parse_input_event(p_event);
}

void DisplayServerEmbedded::touches_canceled(int p_idx, DisplayServer::WindowID p_window) {
	touch_press(p_idx, -1, -1, false, false, p_window);
}

void DisplayServerEmbedded::key(Key p_key, char32_t p_char, Key p_unshifted, Key p_physical, BitField<KeyModifierMask> p_modifiers, bool p_pressed, DisplayServer::WindowID p_window) {
	Ref<InputEventKey> ev;
	ev.instantiate();
	ev->set_window_id(p_window);
	ev->set_echo(false);
	ev->set_pressed(p_pressed);
	ev->set_keycode(fix_keycode(p_char, p_key));
	if (p_key != Key::SHIFT) {
		ev->set_shift_pressed(p_modifiers.has_flag(KeyModifierMask::SHIFT));
	}
	if (p_key != Key::CTRL) {
		ev->set_ctrl_pressed(p_modifiers.has_flag(KeyModifierMask::CTRL));
	}
	if (p_key != Key::ALT) {
		ev->set_alt_pressed(p_modifiers.has_flag(KeyModifierMask::ALT));
	}
	if (p_key != Key::META) {
		ev->set_meta_pressed(p_modifiers.has_flag(KeyModifierMask::META));
	}
	ev->set_key_label(p_unshifted);
	ev->set_physical_keycode(p_physical);
	ev->set_unicode(fix_unicode(p_char));
	perform_event(ev);
}

// MARK: -

bool DisplayServerEmbedded::has_feature(Feature p_feature) const {
	switch (p_feature) {
#ifndef DISABLE_DEPRECATED
		case FEATURE_GLOBAL_MENU: {
			return (native_menu && native_menu->has_feature(NativeMenu::FEATURE_GLOBAL_MENU));
		} break;
#endif
		// case FEATURE_CURSOR_SHAPE:
		// case FEATURE_CUSTOM_CURSOR_SHAPE:
		// case FEATURE_HIDPI:
		// case FEATURE_ICON:
		// case FEATURE_IME:
		// case FEATURE_MOUSE:
		// case FEATURE_MOUSE_WARP:
		// case FEATURE_NATIVE_DIALOG:
		// case FEATURE_NATIVE_ICON:
		// case FEATURE_WINDOW_TRANSPARENCY:
		//case FEATURE_CLIPBOARD:
		//case FEATURE_KEEP_SCREEN_ON:
		//case FEATURE_ORIENTATION:
		//case FEATURE_VIRTUAL_KEYBOARD:
		//case FEATURE_TEXT_TO_SPEECH:
		case FEATURE_NATIVE_WINDOWS:
		case FEATURE_TOUCHSCREEN:
			return true;
		default:
			return false;
	}
}

String DisplayServerEmbedded::get_name() const {
	return "embedded";
}

int DisplayServerEmbedded::get_screen_count() const {
	return 1;
}

int DisplayServerEmbedded::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerEmbedded::screen_get_position(int p_screen) const {
	return Size2i();
}

Size2i DisplayServerEmbedded::screen_get_size(int p_screen) const {
	if (screen_get_size_callback.is_valid()) {
		return screen_get_size_callback.call(p_screen);
	}
	return window_get_size(MAIN_WINDOW_ID);
}

float DisplayServerEmbedded::screen_get_scale(int p_screen) const {
	if (screen_get_scale_callback.is_valid()) {
		return screen_get_scale_callback.call(p_screen);
	}
	return DisplayServer::screen_get_scale(p_screen);
}

Rect2i DisplayServerEmbedded::screen_get_usable_rect(int p_screen) const {
	return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
}

int DisplayServerEmbedded::screen_get_dpi(int p_screen) const {
	if (screen_get_dpi_callback.is_valid()) {
		return screen_get_dpi_callback.call(p_screen);
	}
	return 96;
}

float DisplayServerEmbedded::screen_get_refresh_rate(int p_screen) const {
	return -1;
}

Vector<DisplayServer::WindowID> DisplayServerEmbedded::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerEmbedded::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

DisplayServer::WindowID DisplayServerEmbedded::create_native_window(Ref<RenderingNativeSurface> p_native_surface) {
	WindowID window_id = window_id_counter++;
	window_surfaces[window_id] = p_native_surface;
	surface_to_window_id[p_native_surface] = window_id;

#if defined(RD_ENABLED)
	if (rendering_context) {
		if (rendering_context->window_create(window_id, p_native_surface) != OK) {
			ERR_PRINT(vformat("Failed to create native window."));
			return INVALID_WINDOW_ID;
		}

		if (rendering_device) {
			rendering_device->screen_create(window_id);
		}
		return window_id;
	}
#endif

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		if (gl_manager->window_create(window_id, p_native_surface, 0, 0) != OK) {
			ERR_FAIL_V_MSG(INVALID_WINDOW_ID, "GL manager failed to create window");
		}
		gl_manager->window_make_current(window_id);
		RasterizerGLES3::make_current(false);
		return window_id;
	}
#endif
	ERR_FAIL_V_MSG(INVALID_WINDOW_ID, "Cannot create native window with current driver.");
}

bool DisplayServerEmbedded::is_native_window(DisplayServer::WindowID p_id) {
	return true;
}

void DisplayServerEmbedded::delete_native_window(DisplayServer::WindowID p_id) {
#if defined(RD_ENABLED)
	if (rendering_device) {
		rendering_device->screen_free(p_id);
	}

	if (rendering_context) {
		rendering_context->window_destroy(p_id);
	}
#endif

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_destroy(p_id);
	}
#endif

	surface_to_window_id.erase(window_surfaces[p_id]);
	window_surfaces.erase(p_id);
}

int64_t DisplayServerEmbedded::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	switch (p_handle_type) {
#if defined(GLES3_ENABLED)
		case OPENGL_FBO: {
			if (gl_manager) {
				return gl_manager->window_get_render_target(p_window);
			}
			return 0;
		}
		case WINDOW_HANDLE: {
			if (gl_manager) {
				return (int64_t)gl_manager->window_get_color_texture(p_window);
			}
			return 0;
		}
#endif
		default: {
			return 0; // Not supported.
		}
	}
}

void DisplayServerEmbedded::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id[p_window] = p_instance;
}

ObjectID DisplayServerEmbedded::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id[p_window];
}

void DisplayServerEmbedded::window_set_title(const String &p_title, WindowID p_window) {
	// Not supported
}

int DisplayServerEmbedded::window_get_current_screen(WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerEmbedded::window_set_current_screen(int p_screen, WindowID p_window) {
	// Not supported
}

Point2i DisplayServerEmbedded::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerEmbedded::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerEmbedded::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerEmbedded::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Not supported
}

void DisplayServerEmbedded::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Not supported
}

Size2i DisplayServerEmbedded::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerEmbedded::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Not supported
}

Size2i DisplayServerEmbedded::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerEmbedded::window_set_size(const Size2i p_size, WindowID p_window) {
	// Not supported
}

Size2i DisplayServerEmbedded::window_get_size(WindowID p_window) const {
#if defined(RD_ENABLED)
	if (rendering_context) {
		uint32_t width = 0;
		uint32_t height = 0;
		rendering_context->window_get_size(p_window, width, height);
		return Size2i(width, height);
	}
#endif
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		return gl_manager->window_get_size(p_window);
	}
#endif
	if (window_sizes.has(p_window)) {
		return window_sizes[p_window];
	}
	return Size2i();
}

Size2i DisplayServerEmbedded::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerEmbedded::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Not supported
}

DisplayServer::WindowMode DisplayServerEmbedded::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerEmbedded::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerEmbedded::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Not supported
}

bool DisplayServerEmbedded::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerEmbedded::window_request_attention(WindowID p_window) {
	// Not supported
}

void DisplayServerEmbedded::window_move_to_foreground(WindowID p_window) {
	// Not supported
}

bool DisplayServerEmbedded::window_is_focused(WindowID p_window) const {
	return true;
}

float DisplayServerEmbedded::screen_get_max_scale() const {
	return screen_get_scale(SCREEN_OF_MAIN_WINDOW);
}

bool DisplayServerEmbedded::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerEmbedded::can_any_window_draw() const {
	return true;
}

bool DisplayServerEmbedded::is_touchscreen_available() const {
	return true;
}

void DisplayServerEmbedded::resize_window(Size2i p_size, WindowID p_id) {
	Size2i size = p_size * content_scale;

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(p_id, size.x, size.y);
	}
#endif

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_resize(p_id, p_size.width, p_size.height);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callbacks[p_id], resize_rect);
}

void DisplayServerEmbedded::set_content_scale(float p_scale) {
	content_scale = p_scale;
}

void DisplayServerEmbedded::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	// Not supported
}

DisplayServer::VSyncMode DisplayServerEmbedded::window_get_vsync_mode(WindowID p_window) const {
	return DisplayServer::VSYNC_ENABLED;
}

void DisplayServerEmbedded::swap_buffers() {
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->swap_buffers();
	}
#endif
}

void DisplayServerEmbedded::gl_window_make_current(DisplayServer::WindowID p_window_id) {
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_make_current(p_window_id);
	}
	current_window = p_window_id;
#endif
}
