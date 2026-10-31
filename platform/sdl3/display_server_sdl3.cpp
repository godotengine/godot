/**************************************************************************/
/*  display_server_sdl3.cpp                                               */
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

#include "display_server_sdl3.h"

#include "key_mapping_sdl3.h"

#ifdef VULKAN_ENABLED
#include "servers/rendering/rendering_device.h"
#include "rendering_context_driver_vulkan_sdl3.h"
#endif

#include "core/input/input.h"
#include "core/os/os.h"
#include "drivers/sdl/joypad_sdl.h"
#include "os_sdl3.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"

#include <SDL3/SDL.h>

#ifdef VULKAN_ENABLED
#include <drivers/vulkan/rendering_context_driver_vulkan.h>
#endif

static DisplayServerSDL3 *sdl3_server = nullptr;

void DisplayServerSDL3::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	static_cast<DisplayServerSDL3 *>(DisplayServer::get_singleton())->_dispatch_input_event(p_event);
}

void DisplayServerSDL3::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	if (input_event_callback.is_valid()) {
		input_event_callback.call(p_event);
	}
}

Vector<String> DisplayServerSDL3::get_rendering_drivers_func() {
	Vector<String> drivers;
#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif
#ifdef GLES3_ENABLED
	drivers.push_back("opengl3");
#endif
	return drivers;
}

void DisplayServerSDL3::register_sdl3_driver() {
	register_create_function("sdl3", create_func, get_rendering_drivers_func);
}

DisplayServer *DisplayServerSDL3::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	DisplayServerSDL3 *server = memnew(DisplayServerSDL3(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
	// Only a server that finished constructing may be reached through _display_for().
	if (r_error == OK) {
		sdl3_server = server;
	}
	return server;
}

DisplayServerSDL3::DisplayServerSDL3(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	r_error = ERR_UNAVAILABLE;
	rendering_driver = p_rendering_driver;
	window_mode = p_mode;
	vsync_mode = p_vsync_mode;

	KeyMappingSDL3::initialize();

	// The one SDL initialisation in the process. The joypad driver reuses it.
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) {
		ERR_PRINT(vformat("SDL_Init failed: %s", SDL_GetError()));
		return;
	}

	Size2i size = p_resolution;
	if (size.x <= 0) {
		size.x = 1280;
	}
	if (size.y <= 0) {
		size.y = 720;
	}

	SDL_WindowFlags flags = SDL_WINDOW_HIGH_PIXEL_DENSITY;
#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		flags |= SDL_WINDOW_VULKAN;
	}
#endif
#ifdef GLES3_ENABLED
	if (rendering_driver == "opengl3") {
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		flags |= SDL_WINDOW_OPENGL;
	}
#endif

	window = SDL_CreateWindow(GODOT_VERSION_FULL_NAME, size.x, size.y, flags);
	if (window == nullptr) {
		ERR_PRINT(vformat("SDL_CreateWindow failed: %s", SDL_GetError()));
		return;
	}

	if (p_mode == DisplayServerEnums::WINDOW_MODE_FULLSCREEN || p_mode == DisplayServerEnums::WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
		SDL_SetWindowFullscreen(window, true);
	}
	if (p_position != nullptr) {
		SDL_SetWindowPosition(window, p_position->x, p_position->y);
	}

	int window_w = 0;
	int window_h = 0;
	SDL_GetWindowSize(window, &window_w, &window_h);
	Size2i position(0, 0);
	SDL_GetWindowPosition(window, &position.x, &position.y);
	window_size = Size2i(window_w, window_h);
	window_position = Point2i(position);

#ifdef VULKAN_ENABLED
	if (rendering_driver == "vulkan") {
		RenderingContextDriverVulkanSDL3 *vulkan_context = memnew(RenderingContextDriverVulkanSDL3);
		rendering_context = vulkan_context;
		if (vulkan_context->initialize() != OK) {
			ERR_PRINT("Failed to initialize the Vulkan rendering context.");
			return;
		}
		RenderingContextDriverVulkanSDL3::WindowPlatformData wpd;
		wpd.window = window;
		if (rendering_context->window_create(DisplayServerEnums::MAIN_WINDOW_ID, &wpd) != OK) {
			ERR_PRINT("Failed to create the Vulkan window.");
			return;
		}
		int pixel_w = 0;
		int pixel_h = 0;
		SDL_GetWindowSizeInPixels(window, &pixel_w, &pixel_h);
		rendering_context->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, pixel_w, pixel_h);
		rendering_context->window_set_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID, vsync_mode);

		rendering_device = memnew(RenderingDevice);
		if (rendering_device->initialize(rendering_context, DisplayServerEnums::MAIN_WINDOW_ID) != OK) {
			ERR_PRINT("Failed to initialize the rendering device.");
			memdelete(rendering_device);
			rendering_device = nullptr;
			return;
		}
		rendering_device->screen_create(DisplayServerEnums::MAIN_WINDOW_ID);
		RendererCompositorRD::make_current();
	}
#endif

#ifdef GLES3_ENABLED
	if (rendering_driver == "opengl3") {
		gl_context = SDL_GL_CreateContext(window);
		if (gl_context == nullptr) {
			ERR_PRINT(vformat("SDL_GL_CreateContext failed: %s", SDL_GetError()));
			return;
		}
		SDL_GL_MakeCurrent(window, gl_context);
		_apply_vsync();
	}
#endif

	if (!SDL_StartTextInput(window)) {
		ERR_PRINT(vformat("SDL_StartTextInput failed: %s", SDL_GetError()));
	}

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerSDL3::~DisplayServerSDL3() {
	for (const KeyValue<DisplayServerEnums::CursorShape, SDL_Cursor *> &cursor : cursors) {
		SDL_DestroyCursor(cursor.value);
	}
	cursors.clear();

	// The Vulkan/RD context owns the instance and surface; both go before the window.
	if (rendering_device != nullptr) {
		rendering_device->screen_free(DisplayServerEnums::MAIN_WINDOW_ID);
	}
	if (rendering_context != nullptr) {
		rendering_context->window_destroy(DisplayServerEnums::MAIN_WINDOW_ID);
	}
	if (rendering_device != nullptr) {
		memdelete(rendering_device);
		rendering_device = nullptr;
	}
	if (rendering_context != nullptr) {
		memdelete(rendering_context);
		rendering_context = nullptr;
	}

	if (gl_context != nullptr) {
		SDL_GL_DestroyContext(gl_context);
		gl_context = nullptr;
	}
	if (window != nullptr) {
		SDL_DestroyWindow(window);
		window = nullptr;
	}
	if (sdl3_server == this) {
		sdl3_server = nullptr;
	}
}

bool DisplayServerSDL3::has_feature(DisplayServerEnums::Feature p_feature) const {
	switch (p_feature) {
		case DisplayServerEnums::FEATURE_MOUSE:
		case DisplayServerEnums::FEATURE_MOUSE_WARP:
		case DisplayServerEnums::FEATURE_CURSOR_SHAPE:
		case DisplayServerEnums::FEATURE_CLIPBOARD:
		case DisplayServerEnums::FEATURE_CLIPBOARD_PRIMARY:
		case DisplayServerEnums::FEATURE_HIDPI:
		case DisplayServerEnums::FEATURE_SWAP_BUFFERS:
		case DisplayServerEnums::FEATURE_KEEP_SCREEN_ON:
			return true;
		default:
			return false;
	}
}

/* INPUT */

void DisplayServerSDL3::_handle_key_event(const SDL_KeyboardEvent &p_event, bool p_down) {
	if (!input_event_callback.is_valid()) {
		return;
	}

	Ref<InputEventKey> key;
	key.instantiate();
	key->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	key->set_pressed(p_down);
	key->set_echo(p_event.repeat != 0);
	key->set_keycode(KeyMappingSDL3::remap_key(p_event.key));
	key->set_physical_keycode(KeyMappingSDL3::remap_key(p_event.scancode));
	key->set_location(KeyMappingSDL3::key_location(p_event.scancode));
	key->set_unicode((char32_t)p_event.key);

	Key mods = Key::NONE;
	if (p_event.mod & (SDL_KMOD_LSHIFT | SDL_KMOD_RSHIFT)) {
		mods = mods | KeyModifierMask::SHIFT;
	}
	if (p_event.mod & (SDL_KMOD_LCTRL | SDL_KMOD_RCTRL)) {
		mods = mods | KeyModifierMask::CTRL;
	}
	if (p_event.mod & (SDL_KMOD_LALT | SDL_KMOD_RALT)) {
		mods = mods | KeyModifierMask::ALT;
	}
	if (p_event.mod & (SDL_KMOD_LGUI | SDL_KMOD_RGUI)) {
		mods = mods | KeyModifierMask::META;
	}
	if (p_event.mod & SDL_KMOD_MODE) {
		mods = mods | KeyModifierMask::GROUP_SWITCH;
	}
	key->set_keycode(key->get_keycode() | mods);
	key->set_physical_keycode(key->get_physical_keycode() | mods);

	Input::get_singleton()->parse_input_event(key);
}

void DisplayServerSDL3::_handle_text_input(const SDL_TextInputEvent &p_event) {
	if (!input_text_callback.is_valid()) {
		return;
	}
	input_text_callback.call(String::utf8(p_event.text));
}

void DisplayServerSDL3::_handle_mouse_motion(const SDL_MouseMotionEvent &p_event) {
	if (!input_event_callback.is_valid()) {
		return;
	}
	Ref<InputEventMouseMotion> motion;
	motion.instantiate();
	motion->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	motion->set_position(Point2(p_event.x, p_event.y));
	motion->set_global_position(Point2(p_event.x, p_event.y));
	motion->set_relative(Point2(p_event.xrel, p_event.yrel));
	Input::get_singleton()->parse_input_event(motion);
}

void DisplayServerSDL3::_handle_mouse_button(const SDL_MouseButtonEvent &p_event) {
	if (!input_event_callback.is_valid()) {
		return;
	}
	Ref<InputEventMouseButton> button;
	button.instantiate();
	button->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
	button->set_position(Point2(p_event.x, p_event.y));
	button->set_global_position(Point2(p_event.x, p_event.y));
	button->set_pressed(p_event.down);
	switch (p_event.button) {
		case SDL_BUTTON_LEFT:
			button->set_button_index(MouseButton::LEFT);
			break;
		case SDL_BUTTON_MIDDLE:
			button->set_button_index(MouseButton::MIDDLE);
			break;
		case SDL_BUTTON_RIGHT:
			button->set_button_index(MouseButton::RIGHT);
			break;
		case SDL_BUTTON_X1:
			button->set_button_index(MouseButton::MB_XBUTTON1);
			break;
		case SDL_BUTTON_X2:
			button->set_button_index(MouseButton::MB_XBUTTON2);
			break;
		default:
			return;
	}
	button->set_factor(1.0);
	Input::get_singleton()->parse_input_event(button);
}

void DisplayServerSDL3::_handle_mouse_wheel(const SDL_MouseWheelEvent &p_event) {
	if (!input_event_callback.is_valid()) {
		return;
	}
	const float wheel_x = p_event.x;
	const float wheel_y = p_event.y;

	const struct {
		MouseButton index;
		float amount;
	} axes[] = {
		{ MouseButton::WHEEL_UP, wheel_y > 0 ? wheel_y : 0 },
		{ MouseButton::WHEEL_DOWN, wheel_y < 0 ? -wheel_y : 0 },
		{ MouseButton::WHEEL_LEFT, wheel_x < 0 ? -wheel_x : 0 },
		{ MouseButton::WHEEL_RIGHT, wheel_x > 0 ? wheel_x : 0 },
	};
	for (const auto &axis : axes) {
		if (axis.amount == 0) {
			continue;
		}
		Ref<InputEventMouseButton> button;
		button.instantiate();
		button->set_window_id(DisplayServerEnums::MAIN_WINDOW_ID);
		button->set_position(Point2(p_event.mouse_x, p_event.mouse_y));
		button->set_global_position(Point2(p_event.mouse_x, p_event.mouse_y));
		button->set_button_index(axis.index);
		button->set_factor(axis.amount);
		button->set_pressed(true);
		Input::get_singleton()->parse_input_event(button);
		button->set_pressed(false);
		Input::get_singleton()->parse_input_event(button);
	}
}

void DisplayServerSDL3::_handle_drop(const char *p_file) {
	if (p_file == nullptr || !drop_files_callback.is_valid()) {
		return;
	}
	// Godot's contract is one callback per drop, carrying every path at once.
	Vector<String> files;
	files.push_back(String::utf8(p_file));
	const Variant v_files = files;
	drop_files_callback.call(v_files);
}

void DisplayServerSDL3::_handle_window_event(SDL_WindowID p_id, uint32_t p_event_type) {
	if (p_id != SDL_GetWindowID(window)) {
		return;
	}

	switch (p_event_type) {
		case SDL_EVENT_WINDOW_RESIZED: {
			int resized_w = 0;
			int resized_h = 0;
			SDL_GetWindowSize(window, &resized_w, &resized_h);
			window_size = Size2i(resized_w, resized_h);
			if (rendering_context != nullptr) {
				int resized_pw = 0;
				int resized_ph = 0;
				SDL_GetWindowSizeInPixels(window, &resized_pw, &resized_ph);
				rendering_context->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, resized_pw, resized_ph);
			}
			// The window may have been dragged to another monitor since the last event.
			SDL_GetWindowPosition(window, &window_position.x, &window_position.y);
			if (rect_changed_callback.is_valid()) {
				rect_changed_callback.call(Rect2i(window_position, window_size));
			}
		} break;
		case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
			if (rendering_context != nullptr) {
				int changed_w = 0;
				int changed_h = 0;
				SDL_GetWindowSizeInPixels(window, &changed_w, &changed_h);
				rendering_context->window_set_size(DisplayServerEnums::MAIN_WINDOW_ID, changed_w, changed_h);
			}
		} break;
		case SDL_EVENT_WINDOW_MINIMIZED:
			minimized = true;
			break;
		case SDL_EVENT_WINDOW_RESTORED:
			minimized = false;
			break;
		case SDL_EVENT_WINDOW_FOCUS_GAINED:
			focused = true;
			if (window_event_callback.is_valid()) {
				window_event_callback.call(DisplayServerEnums::WINDOW_EVENT_FOCUS_IN, DisplayServerEnums::MAIN_WINDOW_ID);
			}
			break;
		case SDL_EVENT_WINDOW_FOCUS_LOST:
			focused = false;
			if (window_event_callback.is_valid()) {
				window_event_callback.call(DisplayServerEnums::WINDOW_EVENT_FOCUS_OUT, DisplayServerEnums::MAIN_WINDOW_ID);
			}
			break;
		case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
			if (window_event_callback.is_valid()) {
				window_event_callback.call(DisplayServerEnums::WINDOW_EVENT_CLOSE_REQUEST, DisplayServerEnums::MAIN_WINDOW_ID);
			}
			break;
		default:
			break;
	}
}

void DisplayServerSDL3::process_events() {
	if (window == nullptr) {
		return;
	}

	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		switch (event.type) {
			case SDL_EVENT_QUIT:
				if (window_event_callback.is_valid()) {
					window_event_callback.call(DisplayServerEnums::WINDOW_EVENT_CLOSE_REQUEST, DisplayServerEnums::MAIN_WINDOW_ID);
				}
				break;
			case SDL_EVENT_WINDOW_RESIZED:
			case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
			case SDL_EVENT_WINDOW_MINIMIZED:
			case SDL_EVENT_WINDOW_RESTORED:
			case SDL_EVENT_WINDOW_FOCUS_GAINED:
			case SDL_EVENT_WINDOW_FOCUS_LOST:
			case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
				_handle_window_event(event.window.windowID, event.type);
				break;
			case SDL_EVENT_KEY_DOWN:
				_handle_key_event(event.key, true);
				break;
			case SDL_EVENT_KEY_UP:
				_handle_key_event(event.key, false);
				break;
			case SDL_EVENT_TEXT_INPUT:
				_handle_text_input(event.text);
				break;
			case SDL_EVENT_MOUSE_MOTION:
				_handle_mouse_motion(event.motion);
				break;
			case SDL_EVENT_MOUSE_BUTTON_DOWN:
			case SDL_EVENT_MOUSE_BUTTON_UP:
				_handle_mouse_button(event.button);
				break;
			case SDL_EVENT_MOUSE_WHEEL:
				_handle_mouse_wheel(event.wheel);
				break;
			case SDL_EVENT_DROP_FILE:
				_handle_drop(event.drop.data);
				break;
			case SDL_EVENT_DROP_POSITION:
				// One DROP_FILE arrives per dropped file; nothing to do per position.
				break;
			default:
				break;
		}
	}

	// Gamepad input stays in the existing SDL driver, which owns its own event
	// handling and only needs to be pumped.
	if (JoypadSDL *joypads = static_cast<OS_SDL3 *>(OS::get_singleton())->get_joypad_driver()) {
		joypads->process_events();
	}
}

/* SCREEN */

int DisplayServerSDL3::get_screen_count() const {
	int count = 0;
	SDL_DisplayID *displays = SDL_GetDisplays(&count);
	if (displays != nullptr) {
		SDL_free(displays);
	}
	return count;
}

static SDL_DisplayID _display_for(int p_screen) {
	if (p_screen == DisplayServerEnums::SCREEN_OF_MAIN_WINDOW) {
		const int64_t handle = sdl3_server != nullptr ? sdl3_server->window_get_native_handle(DisplayServerEnums::WINDOW_HANDLE) : 0;
		return handle != 0 ? SDL_GetDisplayForWindow((SDL_Window *)handle) : SDL_GetPrimaryDisplay();
	}
	int count = 0;
	SDL_DisplayID *displays = SDL_GetDisplays(&count);
	SDL_DisplayID result = SDL_GetPrimaryDisplay();
	if (displays != nullptr) {
		if (p_screen >= 0 && p_screen < count) {
			result = displays[p_screen];
		}
		SDL_free(displays);
	}
	return result;
}

int DisplayServerSDL3::get_primary_screen() const {
	int count = 0;
	SDL_DisplayID *displays = SDL_GetDisplays(&count);
	int result = 0;
	if (displays != nullptr) {
		const SDL_DisplayID primary = SDL_GetPrimaryDisplay();
		for (int i = 0; i < count; i++) {
			if (displays[i] == primary) {
				result = i;
				break;
			}
		}
		SDL_free(displays);
	}
	return result;
}

Point2i DisplayServerSDL3::screen_get_position(int p_screen) const {
	SDL_Rect bounds;
	if (SDL_GetDisplayBounds(_display_for(p_screen), &bounds)) {
		return Point2i(bounds.x, bounds.y);
	}
	return Point2i();
}

Size2i DisplayServerSDL3::screen_get_size(int p_screen) const {
	SDL_Rect bounds;
	if (SDL_GetDisplayBounds(_display_for(p_screen), &bounds)) {
		return Size2i(bounds.w, bounds.h);
	}
	return Size2i();
}

Rect2i DisplayServerSDL3::screen_get_usable_rect(int p_screen) const {
	SDL_Rect bounds;
	if (SDL_GetDisplayUsableBounds(_display_for(p_screen), &bounds)) {
		return Rect2i(bounds.x, bounds.y, bounds.w, bounds.h);
	}
	return Rect2i();
}

int DisplayServerSDL3::screen_get_dpi(int p_screen) const {
	const float scale = SDL_GetDisplayContentScale(_display_for(p_screen));
	return scale > 0 ? Math::round(96.0f * scale) : 96;
}

float DisplayServerSDL3::screen_get_refresh_rate(int p_screen) const {
	const SDL_DisplayMode *mode = SDL_GetCurrentDisplayMode(_display_for(p_screen));
	return mode != nullptr ? mode->refresh_rate : 0.0f;
}

/* WINDOW */

Vector<DisplayServerEnums::WindowID> DisplayServerSDL3::get_window_list() const {
	Vector<DisplayServerEnums::WindowID> list;
	if (window != nullptr) {
		list.push_back(DisplayServerEnums::MAIN_WINDOW_ID);
	}
	return list;
}

DisplayServerEnums::WindowID DisplayServerSDL3::get_window_at_screen_position(const Point2i &p_position) const {
	return DisplayServerEnums::MAIN_WINDOW_ID;
}

void DisplayServerSDL3::window_attach_instance_id(ObjectID p_instance, DisplayServerEnums::WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerSDL3::window_get_attached_instance_id(DisplayServerEnums::WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerSDL3::window_set_rect_changed_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	rect_changed_callback = p_callable;
}

void DisplayServerSDL3::window_set_window_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	window_event_callback = p_callable;
}

void DisplayServerSDL3::window_set_input_event_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerSDL3::window_set_input_text_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerSDL3::window_set_drop_files_callback(const Callable &p_callable, DisplayServerEnums::WindowID p_window) {
	drop_files_callback = p_callable;
}

void DisplayServerSDL3::window_set_title(const String &p_title, DisplayServerEnums::WindowID p_window) {
	if (window != nullptr) {
		SDL_SetWindowTitle(window, p_title.utf8().get_data());
	}
}

int DisplayServerSDL3::window_get_current_screen(DisplayServerEnums::WindowID p_window) const {
	return get_primary_screen();
}

void DisplayServerSDL3::window_set_current_screen(int p_screen, DisplayServerEnums::WindowID p_window) {
}

Point2i DisplayServerSDL3::window_get_position(DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return Point2i();
	}
	int px = 0;
	int py = 0;
	SDL_GetWindowPosition(window, &px, &py);
	const Point2i position(px, py);
	return position;
}

Point2i DisplayServerSDL3::window_get_position_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return window_get_position(p_window);
}

void DisplayServerSDL3::window_set_position(const Point2i &p_position, DisplayServerEnums::WindowID p_window) {
	if (window != nullptr) {
		SDL_SetWindowPosition(window, p_position.x, p_position.y);
	}
}

void DisplayServerSDL3::window_set_transient(DisplayServerEnums::WindowID p_window, DisplayServerEnums::WindowID p_parent) {
}

void DisplayServerSDL3::window_set_max_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	if (window != nullptr && p_size.x > 0 && p_size.y > 0) {
		SDL_SetWindowMaximumSize(window, p_size.x, p_size.y);
	}
}

Size2i DisplayServerSDL3::window_get_max_size(DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return Size2i();
	}
	Size2i size;
	SDL_GetWindowMaximumSize(window, &size.x, &size.y);
	return size;
}

void DisplayServerSDL3::window_set_min_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	if (window != nullptr) {
		SDL_SetWindowMinimumSize(window, p_size.x, p_size.y);
	}
}

Size2i DisplayServerSDL3::window_get_min_size(DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return Size2i();
	}
	Size2i size;
	SDL_GetWindowMinimumSize(window, &size.x, &size.y);
	return size;
}

void DisplayServerSDL3::window_set_size(const Size2i p_size, DisplayServerEnums::WindowID p_window) {
	if (window != nullptr) {
		SDL_SetWindowSize(window, p_size.x, p_size.y);
		window_size = p_size;
	}
}

Size2i DisplayServerSDL3::window_get_size(DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return Size2i();
	}
	int w = 0;
	int h = 0;
	SDL_GetWindowSize(window, &w, &h);
	return Size2i(w, h);
}

Size2i DisplayServerSDL3::window_get_size_with_decorations(DisplayServerEnums::WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerSDL3::window_set_mode(DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::WindowID p_window) {
	if (window == nullptr) {
		return;
	}
	const bool fullscreen = p_mode == DisplayServerEnums::WINDOW_MODE_FULLSCREEN ||
			p_mode == DisplayServerEnums::WINDOW_MODE_EXCLUSIVE_FULLSCREEN;
	SDL_SetWindowFullscreen(window, fullscreen);
	if (p_mode == DisplayServerEnums::WINDOW_MODE_MINIMIZED) {
		SDL_MinimizeWindow(window);
	} else if (p_mode == DisplayServerEnums::WINDOW_MODE_MAXIMIZED) {
		SDL_MaximizeWindow(window);
	}
	window_mode = p_mode;
}

DisplayServerEnums::WindowMode DisplayServerSDL3::window_get_mode(DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return DisplayServerEnums::WINDOW_MODE_WINDOWED;
	}
	const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
	if (flags & SDL_WINDOW_FULLSCREEN) {
		return DisplayServerEnums::WINDOW_MODE_FULLSCREEN;
	}
	if (flags & SDL_WINDOW_MINIMIZED) {
		return DisplayServerEnums::WINDOW_MODE_MINIMIZED;
	}
	if (flags & SDL_WINDOW_MAXIMIZED) {
		return DisplayServerEnums::WINDOW_MODE_MAXIMIZED;
	}
	return DisplayServerEnums::WINDOW_MODE_WINDOWED;
}

bool DisplayServerSDL3::window_is_maximize_allowed(DisplayServerEnums::WindowID p_window) const {
	return window != nullptr;
}

void DisplayServerSDL3::window_set_flag(DisplayServerEnums::WindowFlags p_flag, bool p_enabled, DisplayServerEnums::WindowID p_window) {
	if (window == nullptr) {
		return;
	}
	switch (p_flag) {
		case DisplayServerEnums::WINDOW_FLAG_BORDERLESS:
			SDL_SetWindowBordered(window, !p_enabled);
			break;
		case DisplayServerEnums::WINDOW_FLAG_ALWAYS_ON_TOP:
			SDL_SetWindowAlwaysOnTop(window, p_enabled);
			break;
		case DisplayServerEnums::WINDOW_FLAG_RESIZE_DISABLED:
			SDL_SetWindowResizable(window, !p_enabled);
			break;
		case DisplayServerEnums::WINDOW_FLAG_MOUSE_PASSTHROUGH:
			SDL_SetWindowMouseGrab(window, !p_enabled);
			break;
		default:
			break;
	}
}

bool DisplayServerSDL3::window_get_flag(DisplayServerEnums::WindowFlags p_flag, DisplayServerEnums::WindowID p_window) const {
	if (window == nullptr) {
		return false;
	}
	const SDL_WindowFlags flags = SDL_GetWindowFlags(window);
	switch (p_flag) {
		case DisplayServerEnums::WINDOW_FLAG_BORDERLESS:
			return flags & SDL_WINDOW_BORDERLESS;
		case DisplayServerEnums::WINDOW_FLAG_ALWAYS_ON_TOP:
			return flags & SDL_WINDOW_ALWAYS_ON_TOP;
		case DisplayServerEnums::WINDOW_FLAG_RESIZE_DISABLED:
			return !(flags & SDL_WINDOW_RESIZABLE);
		default:
			return false;
	}
}

void DisplayServerSDL3::window_request_attention(DisplayServerEnums::WindowID p_window) {
}

void DisplayServerSDL3::window_move_to_foreground(DisplayServerEnums::WindowID p_window) {
	if (window != nullptr) {
		SDL_RaiseWindow(window);
	}
}

bool DisplayServerSDL3::window_is_focused(DisplayServerEnums::WindowID p_window) const {
	return focused;
}

bool DisplayServerSDL3::window_can_draw(DisplayServerEnums::WindowID p_window) const {
	return window != nullptr && !minimized;
}

bool DisplayServerSDL3::can_any_window_draw() const {
	return window != nullptr && !minimized;
}

int64_t DisplayServerSDL3::window_get_native_handle(DisplayServerEnums::HandleType p_handle_type, DisplayServerEnums::WindowID p_window) const {
	switch (p_handle_type) {
		case DisplayServerEnums::WINDOW_HANDLE:
			return (int64_t)window;
		case DisplayServerEnums::OPENGL_CONTEXT:
			return (int64_t)gl_context;
		default:
			return 0;
	}
}

void DisplayServerSDL3::window_set_vsync_mode(DisplayServerEnums::VSyncMode p_vsync_mode, DisplayServerEnums::WindowID p_window) {
	vsync_mode = p_vsync_mode;
	_apply_vsync();
}

DisplayServerEnums::VSyncMode DisplayServerSDL3::window_get_vsync_mode(DisplayServerEnums::WindowID p_window) const {
	return vsync_mode;
}

void DisplayServerSDL3::_apply_vsync() {
	if (rendering_context != nullptr) {
		rendering_context->window_set_vsync_mode(DisplayServerEnums::MAIN_WINDOW_ID, vsync_mode);
	}
	if (gl_context != nullptr) {
		SDL_GL_SetSwapInterval(vsync_mode == DisplayServerEnums::VSYNC_DISABLED ? 0 : 1);
	}
}

void DisplayServerSDL3::window_set_ime_active(const bool p_active, DisplayServerEnums::WindowID p_window) {
}

void DisplayServerSDL3::window_set_ime_position(const Point2i &p_pos, DisplayServerEnums::WindowID p_window) {
}

void DisplayServerSDL3::window_set_mouse_passthrough(const Vector<Vector2> &p_region, DisplayServerEnums::WindowID p_window) {
}

/* MOUSE */

void DisplayServerSDL3::mouse_set_mode(DisplayServerEnums::MouseMode p_mode) {
	if (window == nullptr) {
		return;
	}
	mouse_mode = p_mode;
	switch (p_mode) {
		case DisplayServerEnums::MOUSE_MODE_VISIBLE:
			SDL_SetWindowRelativeMouseMode(window, false);
			SDL_ShowCursor();
			break;
		case DisplayServerEnums::MOUSE_MODE_HIDDEN:
			SDL_SetWindowRelativeMouseMode(window, false);
			SDL_HideCursor();
			break;
		case DisplayServerEnums::MOUSE_MODE_CAPTURED:
			SDL_SetWindowRelativeMouseMode(window, true);
			SDL_ShowCursor();
			break;
		case DisplayServerEnums::MOUSE_MODE_CONFINED:
			SDL_SetWindowRelativeMouseMode(window, false);
			SDL_SetWindowMouseGrab(window, true);
			break;
		case DisplayServerEnums::MOUSE_MODE_CONFINED_HIDDEN:
			SDL_SetWindowRelativeMouseMode(window, false);
			SDL_SetWindowMouseGrab(window, true);
			SDL_HideCursor();
			break;
	}
}

DisplayServerEnums::MouseMode DisplayServerSDL3::mouse_get_mode() const {
	// SDL can only distinguish captured from not, so the requested mode is kept.
	return mouse_mode;
}

Point2i DisplayServerSDL3::mouse_get_position() const {
	float x = 0;
	float y = 0;
	SDL_GetGlobalMouseState(&x, &y);
	return Point2i(x, y);
}

void DisplayServerSDL3::warp_mouse(const Point2i &p_position) {
	if (window != nullptr) {
		SDL_WarpMouseInWindow(window, p_position.x, p_position.y);
	}
}

/* CLIPBOARD */

void DisplayServerSDL3::clipboard_set(const String &p_text) {
	clipboard = p_text;
	SDL_SetClipboardText(p_text.utf8().get_data());
}

String DisplayServerSDL3::clipboard_get() const {
	if (SDL_HasClipboardText()) {
		char *text = SDL_GetClipboardText();
		if (text != nullptr) {
			String result = String::utf8(text);
			SDL_free(text);
			return result;
		}
	}
	return clipboard;
}

bool DisplayServerSDL3::clipboard_has() const {
	return SDL_HasClipboardText();
}

void DisplayServerSDL3::clipboard_set_primary(const String &p_text) {
	primary_clipboard = p_text;
	SDL_SetPrimarySelectionText(p_text.utf8().get_data());
}

String DisplayServerSDL3::clipboard_get_primary() const {
	if (SDL_HasPrimarySelectionText()) {
		char *text = SDL_GetPrimarySelectionText();
		if (text != nullptr) {
			String result = String::utf8(text);
			SDL_free(text);
			return result;
		}
	}
	return primary_clipboard;
}

/* CURSOR */

SDL_Cursor *DisplayServerSDL3::_get_cursor(DisplayServerEnums::CursorShape p_shape) {
	SDL_Cursor **cursor = cursors.getptr(p_shape);
	if (cursor != nullptr) {
		return *cursor;
	}

	SDL_SystemCursor id = SDL_SYSTEM_CURSOR_DEFAULT;
	switch (p_shape) {
		case DisplayServerEnums::CURSOR_ARROW:
			id = SDL_SYSTEM_CURSOR_DEFAULT;
			break;
		case DisplayServerEnums::CURSOR_IBEAM:
			id = SDL_SYSTEM_CURSOR_TEXT;
			break;
		case DisplayServerEnums::CURSOR_POINTING_HAND:
			id = SDL_SYSTEM_CURSOR_POINTER;
			break;
		case DisplayServerEnums::CURSOR_CROSS:
			id = SDL_SYSTEM_CURSOR_CROSSHAIR;
			break;
		case DisplayServerEnums::CURSOR_WAIT:
			id = SDL_SYSTEM_CURSOR_WAIT;
			break;
		case DisplayServerEnums::CURSOR_BUSY:
			id = SDL_SYSTEM_CURSOR_PROGRESS;
			break;
		case DisplayServerEnums::CURSOR_DRAG:
			id = SDL_SYSTEM_CURSOR_MOVE;
			break;
		case DisplayServerEnums::CURSOR_CAN_DROP:
			id = SDL_SYSTEM_CURSOR_NOT_ALLOWED;
			break;
		case DisplayServerEnums::CURSOR_FORBIDDEN:
			id = SDL_SYSTEM_CURSOR_NOT_ALLOWED;
			break;
		case DisplayServerEnums::CURSOR_VSIZE:
			id = SDL_SYSTEM_CURSOR_NS_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_HSIZE:
			id = SDL_SYSTEM_CURSOR_EW_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_BDIAGSIZE:
			id = SDL_SYSTEM_CURSOR_NWSE_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_FDIAGSIZE:
			id = SDL_SYSTEM_CURSOR_NESW_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_MOVE:
			id = SDL_SYSTEM_CURSOR_MOVE;
			break;
		case DisplayServerEnums::CURSOR_VSPLIT:
			id = SDL_SYSTEM_CURSOR_EW_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_HSPLIT:
			id = SDL_SYSTEM_CURSOR_NS_RESIZE;
			break;
		case DisplayServerEnums::CURSOR_HELP:
			id = SDL_SYSTEM_CURSOR_DEFAULT;
			break;
		default:
			break;
	}

	SDL_Cursor *created = SDL_CreateSystemCursor(id);
	if (created != nullptr) {
		cursors.insert(p_shape, created);
	}
	return created;
}

void DisplayServerSDL3::cursor_set_shape(DisplayServerEnums::CursorShape p_shape) {
	if (window == nullptr) {
		return;
	}
	SDL_Cursor *cursor = _get_cursor(p_shape);
	if (cursor != nullptr) {
		SDL_SetCursor(cursor);
		current_cursor_shape = p_shape;
	}
}

DisplayServerEnums::CursorShape DisplayServerSDL3::cursor_get_shape() const {
	return current_cursor_shape;
}

void DisplayServerSDL3::cursor_set_custom_image(const Ref<Resource> &p_cursor, DisplayServerEnums::CursorShape p_shape, const Vector2 &p_hotspot) {
	// Custom cursors need image decoding and per-platform bitmap formats.
	// Console targets have no pointer at all, so this stays a no-op.
}

/* MISC */

void DisplayServerSDL3::screen_set_keep_on(bool p_enable) {
	screen_kept_on = p_enable;
	if (p_enable) {
		SDL_DisableScreenSaver();
	} else {
		SDL_EnableScreenSaver();
	}
}

bool DisplayServerSDL3::screen_is_kept_on() const {
	return screen_kept_on;
}

void DisplayServerSDL3::swap_buffers() {
	if (gl_context != nullptr) {
		SDL_GL_SwapWindow(window);
	}
	// Vulkan presents through the RD swap chain.
}

void DisplayServerSDL3::release_rendering_thread() {
	if (gl_context != nullptr) {
		SDL_GL_MakeCurrent(window, nullptr);
	}
}
