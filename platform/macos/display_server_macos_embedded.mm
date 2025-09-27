/**************************************************************************/
/*  display_server_macos_embedded.mm                                      */
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

#import "display_server_macos_embedded.h"

#if defined(GLES3_ENABLED)
#import "embedded_gl_manager.h"
#import "platform_gl.h"

#import "drivers/gles3/rasterizer_gles3.h"
#endif

#if defined(RD_ENABLED)
#import "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#import "servers/rendering/rendering_device.h"

#if defined(VULKAN_ENABLED)
#import "drivers/apple/rendering_context_driver_vulkan_apple.h"
#endif // VULKAN_ENABLED
#if defined(METAL_ENABLED)
#import "drivers/metal/rendering_context_driver_metal.h"
#endif
#include "drivers/apple/rendering_native_surface_apple.h"
#endif // RD_ENABLED

#import "embedded_debugger.h"
#import "macos_quartz_core_spi.h"

#import "core/config/project_settings.h"
#import "core/debugger/engine_debugger.h"
#import "core/io/marshalls.h"

DisplayServerMacOSEmbedded::DisplayServerMacOSEmbedded(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, Error &r_error) {
	EmbeddedDebugger::initialize(this);

	r_error = OK; // default to OK

	native_menu = memnew(NativeMenu);

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	rendering_driver = p_rendering_driver;

#if defined(RD_ENABLED)
#if defined(VULKAN_ENABLED)
#if defined(__x86_64__)
	bool fallback_to_vulkan = GLOBAL_GET("rendering/rendering_device/fallback_to_vulkan");
	if (!fallback_to_vulkan) {
		WARN_PRINT("Metal is not supported on Intel Macs, switching to Vulkan.");
	}
	// Metal rendering driver not available on Intel.
	if (rendering_driver == "metal") {
		rendering_driver = "vulkan";
		OS::get_singleton()->set_current_rendering_driver_name(rendering_driver);
	}
#endif
	if (rendering_driver == "vulkan") {
		rendering_context = memnew(RenderingContextDriverVulkanApple);
	}
#endif
#if defined(METAL_ENABLED)
	if (rendering_driver == "metal") {
		rendering_context = memnew(RenderingContextDriverMetal);
	}
#endif

	if (rendering_context) {
		if (rendering_context->initialize() != OK) {
			memdelete(rendering_context);
			rendering_context = nullptr;
#if defined(GLES3_ENABLED)
			bool fallback_to_opengl3 = GLOBAL_GET("rendering/rendering_device/fallback_to_opengl3");
			if (fallback_to_opengl3 && rendering_driver != "opengl3") {
				WARN_PRINT("Your device does not seem to support MoltenVK or Metal, switching to OpenGL 3.");
				rendering_driver = "opengl3";
				OS::get_singleton()->set_current_rendering_method("gl_compatibility");
				OS::get_singleton()->set_current_rendering_driver_name(rendering_driver);
			} else
#endif
			{
				r_error = ERR_CANT_CREATE;
				ERR_FAIL_MSG("Could not initialize " + rendering_driver);
			}
		}
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3_angle") {
		WARN_PRINT("ANGLE not supported for embedded display, switching to native OpenGL.");
		rendering_driver = "opengl3";
		OS::get_singleton()->set_current_rendering_driver_name(rendering_driver);
	}

	if (rendering_driver == "opengl3") {
		gl_manager = memnew(GLManagerEmbedded);
		if (gl_manager->initialize() != OK) {
			memdelete(gl_manager);
			gl_manager = nullptr;
			r_error = ERR_UNAVAILABLE;
			ERR_FAIL_MSG("Could not initialize native OpenGL.");
		}
		layer = [CALayer new];
		// OpenGL content is flipped, so it must be transformed.
		layer.anchorPoint = CGPointMake(0, 0);
		layer.transform = CATransform3DMakeScale(1.0, -1.0, 1.0);

		Error err = gl_manager->window_create(window_id_counter, layer, p_resolution.width, p_resolution.height);
		if (err != OK) {
			ERR_FAIL_MSG("Could not create OpenGL context.");
		}
		gl_manager->set_vsync_enabled(p_vsync_mode != DisplayServer::VSYNC_DISABLED);
	}
#endif

#if defined(RD_ENABLED)
	if (rendering_context) {
		layer = [CAMetalLayer new];
		layer.anchorPoint = CGPointMake(0, 1);

		Ref<RenderingNativeSurfaceApple> apple_surface;
		if (rendering_driver == "vulkan" || rendering_driver == "metal") {
			apple_surface = RenderingNativeSurfaceApple::create((__bridge void *)layer);
		}

		Error err = rendering_context->window_create(window_id_counter, apple_surface);
		ERR_FAIL_COND_MSG(err != OK, vformat("Can't create a %s context", rendering_driver));

		// The rendering context is always in pixels
		rendering_context->window_set_size(window_id_counter, p_resolution.width, p_resolution.height);
		rendering_context->window_set_vsync_mode(window_id_counter, p_vsync_mode);
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		RasterizerGLES3::make_current(true);
	}
	if (rendering_driver == "opengl3_angle") {
		RasterizerGLES3::make_current(false);
	}
#endif
#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_device = memnew(RenderingDevice);
		rendering_device->initialize(rendering_context, MAIN_WINDOW_ID);
		rendering_device->screen_create(MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
	}
#endif

	CGFloat scale = screen_get_max_scale();
	layer.contentsScale = scale;
	layer.magnificationFilter = kCAFilterNearest;
	layer.minificationFilter = kCAFilterNearest;
	transparent = ((p_flags & WINDOW_FLAG_TRANSPARENT_BIT) == WINDOW_FLAG_TRANSPARENT_BIT);
	layer.opaque = !(OS::get_singleton()->is_layered_allowed() && transparent);
	layer.actions = @{ @"contents" : [NSNull null] }; // Disable implicit animations for contents.
	// AppKit frames, bounds and positions are always in points.
	CGRect bounds = CGRectMake(0, 0, p_resolution.width, p_resolution.height);
	bounds = CGRectApplyAffineTransform(bounds, CGAffineTransformInvert(CGAffineTransformMakeScale(scale, scale)));
	layer.bounds = bounds;

	CGSConnectionID connection_id = CGSMainConnectionID();
	ca_context = [CAContext contextWithCGSConnection:connection_id options:@{ kCAContextCIFilterBehavior : @"ignore" }];
	ca_context.layer = layer;

	{
		Array arr = { ca_context.contextId };
		EngineDebugger::get_singleton()->send_message("game_view:set_context_id", arr);
	}
}

DisplayServerMacOSEmbedded::~DisplayServerMacOSEmbedded() {
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}

	EmbeddedDebugger::deinitialize();

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		memdelete(gl_manager);
		gl_manager = nullptr;
	}
#endif

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
}

DisplayServer *DisplayServerMacOSEmbedded::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t /* p_parent_window */, Error &r_error) {
	return memnew(DisplayServerMacOSEmbedded(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, r_error));
}

Vector<String> DisplayServerMacOSEmbedded::get_rendering_drivers_func() {
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

void DisplayServerMacOSEmbedded::register_embedded_driver() {
	register_create_function("embedded", create_func, get_rendering_drivers_func);
}

void DisplayServerMacOSEmbedded::beep() const {
	NSBeep();
}

// MARK: - Mouse

void DisplayServerMacOSEmbedded::_mouse_update_mode() {
	MouseMode wanted_mouse_mode = mouse_mode_override_enabled
			? mouse_mode_override
			: mouse_mode_base;

	if (wanted_mouse_mode == mouse_mode) {
		return;
	}

	EngineDebugger::get_singleton()->send_message("game_view:mouse_set_mode", { wanted_mouse_mode });

	mouse_mode = wanted_mouse_mode;
}

void DisplayServerMacOSEmbedded::mouse_set_mode(MouseMode p_mode) {
	if (p_mode == mouse_mode_base) {
		return;
	}
	mouse_mode_base = p_mode;
	_mouse_update_mode();
}

DisplayServerMacOSEmbedded::MouseMode DisplayServerMacOSEmbedded::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerMacOSEmbedded::mouse_set_mode_override(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, MouseMode::MOUSE_MODE_MAX);
	if (p_mode == mouse_mode_override) {
		return;
	}
	mouse_mode_override = p_mode;
	_mouse_update_mode();
}

DisplayServer::MouseMode DisplayServerMacOSEmbedded::mouse_get_mode_override() const {
	return mouse_mode_override;
}

void DisplayServerMacOSEmbedded::mouse_set_mode_override_enabled(bool p_override_enabled) {
	if (p_override_enabled == mouse_mode_override_enabled) {
		return;
	}
	mouse_mode_override_enabled = p_override_enabled;
	_mouse_update_mode();
}

bool DisplayServerMacOSEmbedded::mouse_is_mode_override_enabled() const {
	return mouse_mode_override_enabled;
}

void DisplayServerMacOSEmbedded::warp_mouse(const Point2i &p_position) {
	_THREAD_SAFE_METHOD_
	Input::get_singleton()->set_mouse_position(p_position);
	EngineDebugger::get_singleton()->send_message("game_view:warp_mouse", { p_position });
}

Point2i DisplayServerMacOSEmbedded::mouse_get_position() const {
	_THREAD_SAFE_METHOD_

	const NSPoint mouse_pos = [NSEvent mouseLocation];
	const float scale = screen_get_max_scale();

	for (NSScreen *screen in [NSScreen screens]) {
		NSRect frame = [screen frame];
		if (NSMouseInRect(mouse_pos, frame, NO)) {
			Vector2i pos = Vector2i((int)mouse_pos.x, (int)mouse_pos.y);
			pos *= scale;
			// TODO(sgc): fix this
			// pos -= _get_screens_origin();
			pos.y *= -1;
			return pos;
		}
	}
	return Vector2i();
}

BitField<MouseButtonMask> DisplayServerMacOSEmbedded::mouse_get_button_state() const {
	BitField<MouseButtonMask> last_button_state = MouseButtonMask::NONE;

	NSUInteger buttons = [NSEvent pressedMouseButtons];
	if (buttons & (1 << 0)) {
		last_button_state.set_flag(MouseButtonMask::LEFT);
	}
	if (buttons & (1 << 1)) {
		last_button_state.set_flag(MouseButtonMask::RIGHT);
	}
	if (buttons & (1 << 2)) {
		last_button_state.set_flag(MouseButtonMask::MIDDLE);
	}
	if (buttons & (1 << 3)) {
		last_button_state.set_flag(MouseButtonMask::MB_XBUTTON1);
	}
	if (buttons & (1 << 4)) {
		last_button_state.set_flag(MouseButtonMask::MB_XBUTTON2);
	}
	return last_button_state;
}

// MARK: Events

void DisplayServerMacOSEmbedded::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callbacks[p_window] = p_callable;
}

void DisplayServerMacOSEmbedded::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callbacks[p_window] = p_callable;
}
void DisplayServerMacOSEmbedded::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callbacks[p_window] = p_callable;
}

void DisplayServerMacOSEmbedded::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callbacks[p_window] = p_callable;
}

void DisplayServerMacOSEmbedded::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	// Not supported
}

void DisplayServerMacOSEmbedded::process_events() {
	Input *input = Input::get_singleton();
	input->flush_buffered_events();
}

void DisplayServerMacOSEmbedded::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	Ref<InputEventFromWindow> event_from_window = p_event;
	WindowID window_id = INVALID_WINDOW_ID;
	if (event_from_window.is_valid()) {
		window_id = event_from_window->get_window_id();
	}
	DisplayServerMacOSEmbedded *ds = (DisplayServerMacOSEmbedded *)DisplayServer::get_singleton();
	ds->send_input_event(p_event, window_id);
}

void DisplayServerMacOSEmbedded::send_input_event(const Ref<InputEvent> &p_event, WindowID p_id) const {
	if (p_id != INVALID_WINDOW_ID) {
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

void DisplayServerMacOSEmbedded::send_input_text(const String &p_text, WindowID p_id) const {
	const Callable *cb = input_text_callbacks.getptr(p_id);
	if (cb) {
		_window_callback(*cb, p_text);
	}
}

void DisplayServerMacOSEmbedded::send_window_event(DisplayServer::WindowEvent p_event, WindowID p_id) const {
	const Callable *cb = window_event_callbacks.getptr(p_id);
	if (cb) {
		_window_callback(*cb, int(p_event));
	}
}

void DisplayServerMacOSEmbedded::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (p_callable.is_valid()) {
		p_callable.call(p_arg);
	}
}

// MARK: -

bool DisplayServerMacOSEmbedded::has_feature(Feature p_feature) const {
	switch (p_feature) {
#ifndef DISABLE_DEPRECATED
		case FEATURE_GLOBAL_MENU: {
			return (native_menu && native_menu->has_feature(NativeMenu::FEATURE_GLOBAL_MENU));
		} break;
#endif
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_IME:
		case FEATURE_CUSTOM_CURSOR_SHAPE:
			// case FEATURE_HIDPI:
			// case FEATURE_ICON:
			// case FEATURE_MOUSE:
		case FEATURE_MOUSE_WARP:
			// case FEATURE_NATIVE_DIALOG:
			// case FEATURE_NATIVE_ICON:
			// case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_CLIPBOARD:
			// case FEATURE_KEEP_SCREEN_ON:
			// case FEATURE_ORIENTATION:
			// case FEATURE_VIRTUAL_KEYBOARD:
		case FEATURE_TEXT_TO_SPEECH:
			// case FEATURE_TOUCHSCREEN:
			return true;
		default:
			return false;
	}
}

String DisplayServerMacOSEmbedded::get_name() const {
	return "embedded";
}

int DisplayServerMacOSEmbedded::get_screen_count() const {
	return 1;
}

int DisplayServerMacOSEmbedded::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerMacOSEmbedded::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Point2i());

	return Point2i(0, 0);
}

Size2i DisplayServerMacOSEmbedded::screen_get_size(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Size2i());

	return window_get_size(MAIN_WINDOW_ID);
}

Rect2i DisplayServerMacOSEmbedded::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Rect2i());

	return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
}

int DisplayServerMacOSEmbedded::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 72);

	return 96;
}

float DisplayServerMacOSEmbedded::screen_get_scale(int p_screen) const {
	_THREAD_SAFE_METHOD_

	switch (p_screen) {
		case SCREEN_WITH_MOUSE_FOCUS:
		case SCREEN_WITH_KEYBOARD_FOCUS:
		case SCREEN_PRIMARY:
		case SCREEN_OF_MAIN_WINDOW:
		case 0:
			return state.screen_window_scale;
		default:
			return 1.0;
	}
}

float DisplayServerMacOSEmbedded::screen_get_refresh_rate(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

	p_screen = _get_screen_index(p_screen);
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSDictionary *description = [[screenArray objectAtIndex:p_screen] deviceDescription];
		const CGDisplayModeRef displayMode = CGDisplayCopyDisplayMode([[description objectForKey:@"NSScreenNumber"] unsignedIntValue]);
		const double displayRefreshRate = CGDisplayModeGetRefreshRate(displayMode);
		return (float)displayRefreshRate;
	}
	ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
	return SCREEN_REFRESH_RATE_FALLBACK;
}

Vector<DisplayServer::WindowID> DisplayServerMacOSEmbedded::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerMacOSEmbedded::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

void DisplayServerMacOSEmbedded::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id[p_window] = p_instance;
}

ObjectID DisplayServerMacOSEmbedded::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id[p_window];
}

void DisplayServerMacOSEmbedded::window_set_title(const String &p_title, WindowID p_window) {
	// Not supported
}

int DisplayServerMacOSEmbedded::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, INVALID_SCREEN);

	return 0;
}

void DisplayServerMacOSEmbedded::window_set_current_screen(int p_screen, WindowID p_window) {
	// Not supported
}

Point2i DisplayServerMacOSEmbedded::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerMacOSEmbedded::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerMacOSEmbedded::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerMacOSEmbedded::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Not supported
}

void DisplayServerMacOSEmbedded::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Not supported
}

Size2i DisplayServerMacOSEmbedded::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerMacOSEmbedded::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Not supported
}

Size2i DisplayServerMacOSEmbedded::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerMacOSEmbedded::window_set_size(const Size2i p_size, WindowID p_window) {
	[CATransaction begin];
	[CATransaction setDisableActions:YES];

	CGFloat scale = screen_get_max_scale();
	CGRect bounds = CGRectMake(0, 0, p_size.width, p_size.height);
	bounds = CGRectApplyAffineTransform(bounds, CGAffineTransformInvert(CGAffineTransformMakeScale(scale, scale)));
	layer.bounds = bounds;
	layer.contentsScale = scale;

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(p_window, p_size.width, p_size.height);
	}
#endif
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_resize(p_window, p_size.width, p_size.height);
	}
#endif
	[CATransaction commit];

	Callable *cb = window_resize_callbacks.getptr(p_window);
	if (cb) {
		Variant resize_rect = Rect2i(Point2i(), p_size);
		_window_callback(window_resize_callbacks[p_window], resize_rect);
	}
}

Size2i DisplayServerMacOSEmbedded::window_get_size(WindowID p_window) const {
#if defined(RD_ENABLED)
	if (rendering_context) {
		RenderingContextDriver::SurfaceID surface = rendering_context->surface_get_from_window(p_window);
		ERR_FAIL_COND_V_MSG(surface == 0, Size2i(), "Invalid window ID");
		uint32_t width = rendering_context->surface_get_width(surface);
		uint32_t height = rendering_context->surface_get_height(surface);
		return Size2i(width, height);
	}
#endif
#ifdef GLES3_ENABLED
	if (gl_manager) {
		return gl_manager->window_get_size(p_window);
	}
#endif
	return Size2i();
}

Size2i DisplayServerMacOSEmbedded::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerMacOSEmbedded::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Not supported
}

DisplayServer::WindowMode DisplayServerMacOSEmbedded::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_WINDOWED;
}

bool DisplayServerMacOSEmbedded::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerMacOSEmbedded::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	if (p_flag == WINDOW_FLAG_TRANSPARENT && p_window == MAIN_WINDOW_ID) {
		transparent = p_enabled;
		layer.opaque = !(OS::get_singleton()->is_layered_allowed() && transparent);
	}
}

bool DisplayServerMacOSEmbedded::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	if (p_flag == WINDOW_FLAG_TRANSPARENT && p_window == MAIN_WINDOW_ID) {
		return transparent;
	}
	return false;
}

void DisplayServerMacOSEmbedded::window_request_attention(WindowID p_window) {
	// Not supported
}

void DisplayServerMacOSEmbedded::window_move_to_foreground(WindowID p_window) {
	// Not supported
}

bool DisplayServerMacOSEmbedded::window_is_focused(WindowID p_window) const {
	return true;
}

float DisplayServerMacOSEmbedded::screen_get_max_scale() const {
	return state.screen_max_scale;
}

bool DisplayServerMacOSEmbedded::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerMacOSEmbedded::can_any_window_draw() const {
	return true;
}

void DisplayServerMacOSEmbedded::window_set_ime_active(const bool p_active, WindowID p_window) {
	EngineDebugger::get_singleton()->send_message("game_view:window_set_ime_active", { p_active });
}

void DisplayServerMacOSEmbedded::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	if (p_pos == ime_last_position) {
		return;
	}
	EngineDebugger::get_singleton()->send_message("game_view:window_set_ime_position", { p_pos });
	ime_last_position = p_pos;
}

void DisplayServerMacOSEmbedded::set_state(const DisplayServerEmbeddedState &p_state) {
	if (state == p_state) {
		return;
	}

	uint32_t old_display_id = state.display_id;

	state = p_state;

	if (state.display_id != old_display_id) {
#if defined(GLES3_ENABLED)
		if (gl_manager) {
			gl_manager->set_display_id(state.display_id);
		}
#endif
	}
}

void DisplayServerMacOSEmbedded::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->set_vsync_enabled(p_vsync_mode != DisplayServer::VSYNC_DISABLED);
	}
#endif

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerMacOSEmbedded::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		return (gl_manager->is_vsync_enabled() ? DisplayServer::VSyncMode::VSYNC_ENABLED : DisplayServer::VSyncMode::VSYNC_DISABLED);
	}
#endif
#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}

void DisplayServerMacOSEmbedded::update_im_text(const Point2i &p_selection, const String &p_text) {
	im_selection = p_selection;
	im_text = p_text;

	OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
}

Point2i DisplayServerMacOSEmbedded::ime_get_selection() const {
	return im_selection;
}

String DisplayServerMacOSEmbedded::ime_get_text() const {
	return im_text;
}

void DisplayServerMacOSEmbedded::cursor_set_shape(CursorShape p_shape) {
	cursor_shape = p_shape;
	EngineDebugger::get_singleton()->send_message("game_view:cursor_set_shape", { p_shape });
}

DisplayServer::CursorShape DisplayServerMacOSEmbedded::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerMacOSEmbedded::cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	PackedByteArray data;
	if (p_cursor.is_valid()) {
		Ref<Image> image = _get_cursor_image_from_resource(p_cursor, p_hotspot);
		if (image.is_valid()) {
			data = image->save_png_to_buffer();
		}
	}
	EngineDebugger::get_singleton()->send_message("game_view:cursor_set_custom_image", { data, p_shape, p_hotspot });
}

void DisplayServerMacOSEmbedded::swap_buffers() {
#ifdef GLES3_ENABLED
	if (gl_manager) {
		gl_manager->swap_buffers();
	}
#endif
}

void DisplayServerEmbeddedState::serialize(PackedByteArray &r_data) {
	r_data.resize(16);

	uint8_t *data = r_data.ptrw();
	data += encode_float(screen_max_scale, data);
	data += encode_float(screen_dpi, data);
	data += encode_float(screen_window_scale, data);
	data += encode_uint32(display_id, data);

	// Assert we had enough space.
	DEV_ASSERT(r_data.size() >= (data - r_data.ptrw()));
}

Error DisplayServerEmbeddedState::deserialize(const PackedByteArray &p_data) {
	const uint8_t *data = p_data.ptr();

	screen_max_scale = decode_float(data);
	data += sizeof(float);
	screen_dpi = decode_float(data);
	data += sizeof(float);
	screen_window_scale = decode_float(data);
	data += sizeof(float);
	display_id = decode_uint32(data);

	return OK;
}
