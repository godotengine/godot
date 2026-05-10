/**************************************************************************/
/*  display_server_offscreen.cpp                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "display_server_offscreen.h"

#include "core/os/os.h"
#include "servers/rendering/rendering_context_driver.h"

#ifdef RD_ENABLED
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"
#endif

#ifdef VULKAN_ENABLED
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#endif

DisplayServerOffscreen::VirtualWindow *DisplayServerOffscreen::_get_window(DisplayServerEnums::WindowID p_window) {
	return p_window == DisplayServerEnums::MAIN_WINDOW_ID ? &main_window : nullptr;
}

const DisplayServerOffscreen::VirtualWindow *DisplayServerOffscreen::_get_window(DisplayServerEnums::WindowID p_window) const {
	return p_window == DisplayServerEnums::MAIN_WINDOW_ID ? &main_window : nullptr;
}

Size2i DisplayServerOffscreen::_sanitize_size(const Size2i &p_size) const {
	return Size2i(MAX(1, p_size.width), MAX(1, p_size.height));
}

Vector<String> DisplayServerOffscreen::get_rendering_drivers_func() {
	Vector<String> drivers;
#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif
	return drivers;
}

DisplayServer *DisplayServerOffscreen::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Point2i *p_position, const Size2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	DisplayServerOffscreen *ds = memnew(DisplayServerOffscreen(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, r_error));
	if (r_error != OK) {
		memdelete(ds);
		return nullptr;
	}
	return ds;
}

void DisplayServerOffscreen::register_offscreen_driver() {
	for (int i = 0; i < get_create_function_count(); i++) {
		if (String(get_create_function_name(i)) == "offscreen") {
			return;
		}
	}
	register_create_function("offscreen", create_func, get_rendering_drivers_func, false);
}

void DisplayServerOffscreen::_initialize_rendering(const String &p_rendering_driver, Error &r_error) {
#ifndef RD_ENABLED
	r_error = ERR_UNAVAILABLE;
	ERR_PRINT("The offscreen display server requires RenderingDevice support.");
	return;
#else
	if (p_rendering_driver != "vulkan") {
		r_error = ERR_UNAVAILABLE;
		ERR_PRINT(vformat("The offscreen display server only supports the Vulkan rendering driver in this build, not '%s'.", p_rendering_driver));
		return;
	}

#ifndef VULKAN_ENABLED
	r_error = ERR_UNAVAILABLE;
	ERR_PRINT("The offscreen display server requires Vulkan support in this build.");
	return;
#else
	rendering_context = memnew(RenderingContextDriverVulkan);
	if (rendering_context->initialize() != OK) {
		memdelete(rendering_context);
		rendering_context = nullptr;
		r_error = ERR_UNAVAILABLE;
		ERR_PRINT("Could not initialize the offscreen Vulkan rendering context.");
		return;
	}

	rendering_device = memnew(RenderingDevice);
	if (rendering_device->initialize(rendering_context, DisplayServerEnums::INVALID_WINDOW_ID) != OK) {
		memdelete(rendering_device);
		rendering_device = nullptr;
		memdelete(rendering_context);
		rendering_context = nullptr;
		r_error = ERR_UNAVAILABLE;
		ERR_PRINT("Could not initialize the offscreen RenderingDevice.");
		return;
	}

	Error err = rendering_device->screen_create_virtual(DisplayServerEnums::MAIN_WINDOW_ID, main_window.size);
	if (err != OK) {
		memdelete(rendering_device);
		rendering_device = nullptr;
		memdelete(rendering_context);
		rendering_context = nullptr;
		r_error = err;
		ERR_PRINT("Could not create the offscreen virtual screen.");
		return;
	}

	rendering_screen_created = true;
	RendererCompositorRD::make_current();
	r_error = OK;
#endif
#endif
}

Point2i DisplayServerOffscreen::screen_get_position(int p_screen) const {
	return Point2i();
}

Size2i DisplayServerOffscreen::screen_get_size(int p_screen) const {
	return main_window.size;
}

Rect2i DisplayServerOffscreen::screen_get_usable_rect(int p_screen) const {
	return Rect2i(Point2i(), main_window.size);
}

Vector<DisplayServerEnums::WindowID> DisplayServerOffscreen::get_window_list() const {
	Vector<DisplayServerEnums::WindowID> windows;
	windows.push_back(DisplayServerEnums::MAIN_WINDOW_ID);
	return windows;
}

DisplayServerEnums::WindowID DisplayServerOffscreen::create_sub_window(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect, bool p_exclusive, DisplayServerEnums::WindowID p_transient_parent) {
	ERR_PRINT_ONCE("Sub-windows are not supported by the offscreen display server.");
	return DisplayServerEnums::INVALID_WINDOW_ID;
}

DisplayServerEnums::WindowID DisplayServerOffscreen::get_window_at_screen_position(const Point2i &p_position) const {
	Rect2i rect(main_window.position, main_window.size);
	return rect.has_point(p_position) ? DisplayServerEnums::MAIN_WINDOW_ID : DisplayServerEnums::INVALID_WINDOW_ID;
}

void DisplayServerOffscreen::window_attach_instance_id(ObjectID p_instance, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->attached_instance_id = p_instance;
}

ObjectID DisplayServerOffscreen::window_get_attached_instance_id(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->attached_instance_id : ObjectID();
}

void DisplayServerOffscreen::window_set_rect_changed_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->rect_changed_callback = p_callable;
}

void DisplayServerOffscreen::window_set_window_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->window_event_callback = p_callable;
}

void DisplayServerOffscreen::window_set_input_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->input_event_callback = p_callable;
}

void DisplayServerOffscreen::window_set_input_text_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->input_text_callback = p_callable;
}

void DisplayServerOffscreen::window_set_drop_files_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->drop_files_callback = p_callable;
}

void DisplayServerOffscreen::window_set_title(const String &p_title, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->title = p_title;
}

int DisplayServerOffscreen::window_get_current_screen(DisplayServerEnums::WindowID p_window) const {
	return _get_window(p_window) ? 0 : DisplayServerEnums::INVALID_SCREEN;
}

Point2i DisplayServerOffscreen::window_get_position(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->position : Point2i();
}

Point2i DisplayServerOffscreen::window_get_position_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return window_get_position(p_window);
}

void DisplayServerOffscreen::window_set_position(const Point2i &p_position, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	if (window->position == p_position) {
		return;
	}
	window->position = p_position;
	if (window->rect_changed_callback.is_valid()) {
		window->rect_changed_callback.call(Rect2i(window->position, window->size));
	}
}

void DisplayServerOffscreen::window_set_max_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->max_size = p_size;
}

Size2i DisplayServerOffscreen::window_get_max_size(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->max_size : Size2i();
}

void DisplayServerOffscreen::window_set_min_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->min_size = p_size;
}

Size2i DisplayServerOffscreen::window_get_min_size(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->min_size : Size2i();
}

void DisplayServerOffscreen::window_set_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);

	Size2i size = _sanitize_size(p_size);
	if (window->size == size) {
		return;
	}

#ifdef RD_ENABLED
	if (rendering_device && rendering_screen_created) {
		Error err = rendering_device->screen_resize_virtual(p_window, size);
		ERR_FAIL_COND_MSG(err != OK, "Unable to resize the offscreen virtual screen.");
	}
#endif

	window->size = size;
	if (window->rect_changed_callback.is_valid()) {
		window->rect_changed_callback.call(Rect2i(window->position, window->size));
	}
}

Size2i DisplayServerOffscreen::window_get_size(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->size : Size2i();
}

Size2i DisplayServerOffscreen::window_get_size_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerOffscreen::window_set_mode(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->mode = p_mode;
}

DisplayServerEnums::WindowMode DisplayServerOffscreen::window_get_mode(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->mode : DisplayServerEnums::WINDOW_MODE_WINDOWED;
}

void DisplayServerOffscreen::window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	window->vsync_mode = p_vsync_mode;
}

DisplayServerEnums::VSyncMode DisplayServerOffscreen::window_get_vsync_mode(DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? window->vsync_mode : DisplayServerEnums::VSYNC_DISABLED;
}

void DisplayServerOffscreen::window_set_flag(DisplayServerEnums::WindowFlags p_flag, bool p_enabled, DisplayServerEnums::WindowID p_window) {
	VirtualWindow *window = _get_window(p_window);
	ERR_FAIL_NULL(window);
	if (p_enabled) {
		window->flags |= (1 << p_flag);
	} else {
		window->flags &= ~(1 << p_flag);
	}
}

bool DisplayServerOffscreen::window_get_flag(DisplayServerEnums::WindowFlags p_flag, DisplayServerEnums::WindowID p_window) const {
	const VirtualWindow *window = _get_window(p_window);
	return window ? (window->flags & (1 << p_flag)) != 0 : false;
}

bool DisplayServerOffscreen::window_is_focused(DisplayServerEnums::WindowID p_window) const {
	return _get_window(p_window) != nullptr;
}

bool DisplayServerOffscreen::window_can_draw(DisplayServerEnums::WindowID p_window) const {
	return _get_window(p_window) != nullptr;
}

DisplayServerOffscreen::DisplayServerOffscreen(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Point2i *p_position, const Size2i &p_resolution, Error &r_error) {
	main_window.size = _sanitize_size(p_resolution);
	main_window.position = p_position ? *p_position : Point2i();
	main_window.mode = p_mode;
	main_window.vsync_mode = p_vsync_mode;
	main_window.flags = p_flags;

	_initialize_rendering(p_rendering_driver, r_error);
}

DisplayServerOffscreen::~DisplayServerOffscreen() {
#ifdef RD_ENABLED
	if (rendering_device) {
		if (rendering_screen_created) {
			rendering_device->screen_free(DisplayServerEnums::MAIN_WINDOW_ID);
			rendering_screen_created = false;
		}
		memdelete(rendering_device);
		rendering_device = nullptr;
	}
#endif

	if (rendering_context) {
		memdelete(rendering_context);
		rendering_context = nullptr;
	}
}
