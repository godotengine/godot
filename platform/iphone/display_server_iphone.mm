/*************************************************************************/
/*  display_server_iphone.mm                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "display_server_iphone.h"
#import "app_delegate.h"
#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"
#import "device_metrics.h"
#import "godot_view.h"
#include "ios.h"
#import "keyboard_input_view.h"
#include "os_iphone.h"
#import "view_controller.h"

#import <Foundation/Foundation.h>
#import <sys/utsname.h>

static const float kDisplayServerIPhoneAcceleration = 1;

DisplayServerIPhone *DisplayServerIPhone::get_singleton() {
	return (DisplayServerIPhone *)DisplayServer::get_singleton();
}

DisplayServerIPhone::DisplayServerIPhone(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	rendering_driver = p_rendering_driver;

#if defined(GLES3_ENABLED)
	// FIXME: Add support for both OpenGL and Vulkan when OpenGL is implemented
	// again,
	// Note that we should be checking "opengl3" as the driver, might never enable this seeing OpenGL is deprecated on iOS
	// We are hardcoding the rendering_driver to "vulkan" down below

	if (rendering_driver == "opengl_es") {
		bool gl_initialization_error = false;

		// FIXME: Add Vulkan support via MoltenVK. Add fallback code back?

		if (RasterizerGLES3::is_viable() == OK) {
			RasterizerGLES3::register_config();
			RasterizerGLES3::make_current();
		} else {
			gl_initialization_error = true;
		}

		if (gl_initialization_error) {
			OS::get_singleton()->alert("Your device does not support any of the supported OpenGL versions.", "Unable to initialize video driver");
			//        return ERR_UNAVAILABLE;
		}

		//    rendering_server = memnew(RenderingServerDefault);
		//    // FIXME: Reimplement threaded rendering
		//    if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		//        rendering_server = memnew(RenderingServerWrapMT(rendering_server,
		//        false));
		//    }
		//    rendering_server->init();
		// rendering_server->cursor_set_visible(false, 0);

		// reset this to what it should be, it will have been set to 0 after
		// rendering_server->init() is called
		//    RasterizerStorageGLES3system_fbo = gl_view_base_fb;
	}
#endif

#if defined(VULKAN_ENABLED)
	rendering_driver = "vulkan";

	context_vulkan = nullptr;
	rendering_device_vulkan = nullptr;

	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextIPhone);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			ERR_FAIL_MSG("Failed to initialize Vulkan context");
		}

		CALayer *layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"vulkan"];

		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS rendering layer.");
		}

		Size2i size = Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_max_scale();
		if (context_vulkan->window_create(MAIN_WINDOW_ID, p_vsync_mode, layer, size.width, size.height) != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			ERR_FAIL_MSG("Failed to create Vulkan window.");
		}

		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

	bool keep_screen_on = bool(GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true));
	screen_set_keep_on(keep_screen_on);

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerIPhone::~DisplayServerIPhone() {
#if defined(VULKAN_ENABLED)
	if (rendering_device_vulkan) {
		rendering_device_vulkan->finalize();
		memdelete(rendering_device_vulkan);
		rendering_device_vulkan = nullptr;
	}

	if (context_vulkan) {
		context_vulkan->window_destroy(MAIN_WINDOW_ID);
		memdelete(context_vulkan);
		context_vulkan = nullptr;
	}
#endif
}

DisplayServer *DisplayServerIPhone::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	return memnew(DisplayServerIPhone(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
}

Vector<String> DisplayServerIPhone::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl_es");
#endif

	return drivers;
}

void DisplayServerIPhone::register_iphone_driver() {
	register_create_function("iphone", create_func, get_rendering_drivers_func);
}

// MARK: Events

void DisplayServerIPhone::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callback = p_callable;
}

void DisplayServerIPhone::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callback = p_callable;
}
void DisplayServerIPhone::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerIPhone::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerIPhone::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerIPhone::process_events() {
}

void DisplayServerIPhone::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	DisplayServerIPhone::get_singleton()->send_input_event(p_event);
}

void DisplayServerIPhone::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, p_event);
}

void DisplayServerIPhone::send_input_text(const String &p_text) const {
	_window_callback(input_text_callback, p_text);
}

void DisplayServerIPhone::send_window_event(DisplayServer::WindowEvent p_event) const {
	_window_callback(window_event_callback, int(p_event));
}

void DisplayServerIPhone::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (!p_callable.is_null()) {
		const Variant *argp = &p_arg;
		Variant ret;
		Callable::CallError ce;
		p_callable.call((const Variant **)&argp, 1, ret, ce);
	}
}

// MARK: - Input

// MARK: Touches

void DisplayServerIPhone::touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_double_click) {
	if (!GLOBAL_DEF("debug/disable_touch", false)) {
		Ref<InputEventScreenTouch> ev;
		ev.instantiate();

		ev->set_index(p_idx);
		ev->set_pressed(p_pressed);
		ev->set_position(Vector2(p_x, p_y));
		perform_event(ev);
	}
};

void DisplayServerIPhone::touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y) {
	if (!GLOBAL_DEF("debug/disable_touch", false)) {
		Ref<InputEventScreenDrag> ev;
		ev.instantiate();
		ev->set_index(p_idx);
		ev->set_position(Vector2(p_x, p_y));
		ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
		perform_event(ev);
	};
};

void DisplayServerIPhone::perform_event(const Ref<InputEvent> &p_event) {
	Input::get_singleton()->parse_input_event(p_event);
};

void DisplayServerIPhone::touches_cancelled(int p_idx) {
	touch_press(p_idx, -1, -1, false, false);
};

// MARK: Keyboard

void DisplayServerIPhone::key(Key p_key, bool p_pressed) {
	Ref<InputEventKey> ev;
	ev.instantiate();
	ev->set_echo(false);
	ev->set_pressed(p_pressed);
	ev->set_keycode(p_key);
	ev->set_physical_keycode(p_key);
	ev->set_unicode((char32_t)p_key);
	perform_event(ev);
};

// MARK: Motion

void DisplayServerIPhone::update_gravity(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_gravity(Vector3(p_x, p_y, p_z));
};

void DisplayServerIPhone::update_accelerometer(float p_x, float p_y, float p_z) {
	// Found out the Z should not be negated! Pass as is!
	Vector3 v_accelerometer = Vector3(
			p_x / kDisplayServerIPhoneAcceleration,
			p_y / kDisplayServerIPhoneAcceleration,
			p_z / kDisplayServerIPhoneAcceleration);

	Input::get_singleton()->set_accelerometer(v_accelerometer);
};

void DisplayServerIPhone::update_magnetometer(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_magnetometer(Vector3(p_x, p_y, p_z));
};

void DisplayServerIPhone::update_gyroscope(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_gyroscope(Vector3(p_x, p_y, p_z));
};

// MARK: -

bool DisplayServerIPhone::has_feature(Feature p_feature) const {
	switch (p_feature) {
		// case FEATURE_CURSOR_SHAPE:
		// case FEATURE_CUSTOM_CURSOR_SHAPE:
		// case FEATURE_GLOBAL_MENU:
		// case FEATURE_HIDPI:
		// case FEATURE_ICON:
		// case FEATURE_IME:
		// case FEATURE_MOUSE:
		// case FEATURE_MOUSE_WARP:
		// case FEATURE_NATIVE_DIALOG:
		// case FEATURE_NATIVE_ICON:
		// case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_CLIPBOARD:
		case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_ORIENTATION:
		case FEATURE_TOUCHSCREEN:
		case FEATURE_VIRTUAL_KEYBOARD:
			return true;
		default:
			return false;
	}
}

String DisplayServerIPhone::get_name() const {
	return "iPhone";
}

int DisplayServerIPhone::get_screen_count() const {
	return 1;
}

Point2i DisplayServerIPhone::screen_get_position(int p_screen) const {
	return Size2i();
}

Size2i DisplayServerIPhone::screen_get_size(int p_screen) const {
	CALayer *layer = AppDelegate.viewController.godotView.renderingLayer;

	if (!layer) {
		return Size2i();
	}

	return Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_scale(p_screen);
}

Rect2i DisplayServerIPhone::screen_get_usable_rect(int p_screen) const {
	if (@available(iOS 11, *)) {
		UIEdgeInsets insets = UIEdgeInsetsZero;
		UIView *view = AppDelegate.viewController.godotView;

		if ([view respondsToSelector:@selector(safeAreaInsets)]) {
			insets = [view safeAreaInsets];
		}

		float scale = screen_get_scale(p_screen);
		Size2i insets_position = Size2i(insets.left, insets.top) * scale;
		Size2i insets_size = Size2i(insets.left + insets.right, insets.top + insets.bottom) * scale;

		return Rect2i(screen_get_position(p_screen) + insets_position, screen_get_size(p_screen) - insets_size);
	} else {
		return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
	}
}

int DisplayServerIPhone::screen_get_dpi(int p_screen) const {
	struct utsname systemInfo;
	uname(&systemInfo);

	NSString *string = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

	NSDictionary *iOSModelToDPI = [GodotDeviceMetrics dpiList];

	for (NSArray *keyArray in iOSModelToDPI) {
		if ([keyArray containsObject:string]) {
			NSNumber *value = iOSModelToDPI[keyArray];
			return [value intValue];
		}
	}

	// If device wasn't found in dictionary
	// make a best guess from device metrics.
	CGFloat scale = [UIScreen mainScreen].scale;

	UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;

	switch (idiom) {
		case UIUserInterfaceIdiomPad:
			return scale == 2 ? 264 : 132;
		case UIUserInterfaceIdiomPhone: {
			if (scale == 3) {
				CGFloat nativeScale = [UIScreen mainScreen].nativeScale;
				return nativeScale == 3 ? 458 : 401;
			}

			return 326;
		}
		default:
			return 72;
	}
}

float DisplayServerIPhone::screen_get_scale(int p_screen) const {
	return [UIScreen mainScreen].nativeScale;
}

Vector<DisplayServer::WindowID> DisplayServerIPhone::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerIPhone::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

int64_t DisplayServerIPhone::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, 0);
	switch (p_handle_type) {
		case DISPLAY_HANDLE: {
			return 0; // Not supported.
		}
		case WINDOW_HANDLE: {
			return (int64_t)AppDelegate.viewController;
		}
		case WINDOW_VIEW: {
			return (int64_t)AppDelegate.viewController.godotView;
		}
		default: {
			return 0;
		}
	}
}

void DisplayServerIPhone::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerIPhone::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerIPhone::window_set_title(const String &p_title, WindowID p_window) {
	// Probably not supported for iOS
}

int DisplayServerIPhone::window_get_current_screen(WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerIPhone::window_set_current_screen(int p_screen, WindowID p_window) {
	// Probably not supported for iOS
}

Point2i DisplayServerIPhone::window_get_position(WindowID p_window) const {
	return Point2i();
}

void DisplayServerIPhone::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerIPhone::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Probably not supported for iOS
}

void DisplayServerIPhone::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIPhone::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIPhone::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIPhone::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIPhone::window_set_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIPhone::window_get_size(WindowID p_window) const {
	CGRect screenBounds = [UIScreen mainScreen].bounds;
	return Size2i(screenBounds.size.width, screenBounds.size.height) * screen_get_max_scale();
}

Size2i DisplayServerIPhone::window_get_real_size(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerIPhone::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Probably not supported for iOS
}

DisplayServer::WindowMode DisplayServerIPhone::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerIPhone::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerIPhone::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerIPhone::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerIPhone::window_request_attention(WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerIPhone::window_move_to_foreground(WindowID p_window) {
	// Probably not supported for iOS
}

float DisplayServerIPhone::screen_get_max_scale() const {
	return screen_get_scale(SCREEN_OF_MAIN_WINDOW);
};

void DisplayServerIPhone::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	screen_orientation = p_orientation;
}

DisplayServer::ScreenOrientation DisplayServerIPhone::screen_get_orientation(int p_screen) const {
	return screen_orientation;
}

bool DisplayServerIPhone::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerIPhone::can_any_window_draw() const {
	return true;
}

bool DisplayServerIPhone::screen_is_touchscreen(int p_screen) const {
	return true;
}

void DisplayServerIPhone::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, bool p_multiline, int p_max_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
								 multiline:p_multiline
							   cursorStart:p_cursor_start
								 cursorEnd:p_cursor_end];
}

void DisplayServerIPhone::virtual_keyboard_hide() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

void DisplayServerIPhone::virtual_keyboard_set_height(int height) {
	virtual_keyboard_height = height * screen_get_max_scale();
}

int DisplayServerIPhone::virtual_keyboard_get_height() const {
	return virtual_keyboard_height;
}

void DisplayServerIPhone::clipboard_set(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8()];
}

String DisplayServerIPhone::clipboard_get() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}

void DisplayServerIPhone::screen_set_keep_on(bool p_enable) {
	[UIApplication sharedApplication].idleTimerDisabled = p_enable;
}

bool DisplayServerIPhone::screen_is_kept_on() const {
	return [UIApplication sharedApplication].idleTimerDisabled;
}

void DisplayServerIPhone::resize_window(CGSize viewSize) {
	Size2i size = Size2i(viewSize.width, viewSize.height) * screen_get_max_scale();

#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(MAIN_WINDOW_ID, size.x, size.y);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callback, resize_rect);
}

void DisplayServerIPhone::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
#endif
}

DisplayServer::VSyncMode DisplayServerIPhone::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	return context_vulkan->get_vsync_mode(p_window);
#else
	return DisplayServer::VSYNC_ENABLED;
#endif
}
