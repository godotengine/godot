/**************************************************************************/
/*  display_server_ios.mm                                                 */
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

#import "display_server_ios.h"

#import "app_delegate.h"
#import "device_metrics.h"
#import "godot_view.h"
#import "ios.h"
#import "key_mapping_ios.h"
#import "keyboard_input_view.h"
#import "os_ios.h"
#import "tts_ios.h"
#import "view_controller.h"

#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"

#import <sys/utsname.h>

static const float kDisplayServerIOSAcceleration = 1.f;

DisplayServerIOS *DisplayServerIOS::get_singleton() {
	return (DisplayServerIOS *)DisplayServer::get_singleton();
}

DisplayServerIOS::DisplayServerIOS(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	KeyMappingIOS::initialize();

	rendering_driver = p_rendering_driver;

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		tts = [[TTS_IOS alloc] init];
	}

#if defined(VULKAN_ENABLED)
	context_vulkan = nullptr;
	rendering_device_vulkan = nullptr;

	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextIOS);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			ERR_FAIL_MSG("Failed to initialize Vulkan context");
		}

		CALayer *layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"vulkan"];

		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS Vulkan rendering layer.");
		}

		Size2i size = Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_max_scale();
		if (context_vulkan->window_create(MAIN_WINDOW_ID, p_vsync_mode, layer, size.width, size.height) != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_UNAVAILABLE;
			ERR_FAIL_MSG("Failed to create Vulkan window.");
		}

		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		CALayer *layer = [AppDelegate.viewController.godotView initializeRenderingForDriver:@"opengl3"];

		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS OpenGLES rendering layer.");
		}

		RasterizerGLES3::make_current(false);
	}
#endif

	bool keep_screen_on = bool(GLOBAL_GET("display/window/energy_saving/keep_screen_on"));
	screen_set_keep_on(keep_screen_on);

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerIOS::~DisplayServerIOS() {
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

DisplayServer *DisplayServerIOS::create_func(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	return memnew(DisplayServerIOS(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, r_error));
}

Vector<String> DisplayServerIOS::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl3");
#endif

	return drivers;
}

void DisplayServerIOS::register_ios_driver() {
	register_create_function("iOS", create_func, get_rendering_drivers_func);
}

// MARK: Events

void DisplayServerIOS::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callback = p_callable;
}

void DisplayServerIOS::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callback = p_callable;
}
void DisplayServerIOS::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerIOS::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerIOS::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerIOS::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerIOS::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	DisplayServerIOS::get_singleton()->send_input_event(p_event);
}

void DisplayServerIOS::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, p_event);
}

void DisplayServerIOS::send_input_text(const String &p_text) const {
	_window_callback(input_text_callback, p_text);
}

void DisplayServerIOS::send_window_event(DisplayServer::WindowEvent p_event) const {
	_window_callback(window_event_callback, int(p_event));
}

void DisplayServerIOS::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (!p_callable.is_null()) {
		p_callable.call(p_arg);
	}
}

// MARK: - Input

// MARK: Touches

void DisplayServerIOS::touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_double_click) {
	Ref<InputEventScreenTouch> ev;
	ev.instantiate();

	ev->set_index(p_idx);
	ev->set_pressed(p_pressed);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_double_tap(p_double_click);
	perform_event(ev);
}

void DisplayServerIOS::touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, float p_pressure, Vector2 p_tilt) {
	Ref<InputEventScreenDrag> ev;
	ev.instantiate();
	ev->set_index(p_idx);
	ev->set_pressure(p_pressure);
	ev->set_tilt(p_tilt);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
	perform_event(ev);
}

void DisplayServerIOS::perform_event(const Ref<InputEvent> &p_event) {
	Input::get_singleton()->parse_input_event(p_event);
}

void DisplayServerIOS::touches_canceled(int p_idx) {
	touch_press(p_idx, -1, -1, false, false);
}

// MARK: Keyboard

void DisplayServerIOS::key(Key p_key, char32_t p_char, Key p_unshifted, Key p_physical, NSInteger p_modifier, bool p_pressed) {
	Ref<InputEventKey> ev;
	ev.instantiate();
	ev->set_echo(false);
	ev->set_pressed(p_pressed);
	ev->set_keycode(fix_keycode(p_char, p_key));
	if (@available(iOS 13.4, *)) {
		if (p_key != Key::SHIFT) {
			ev->set_shift_pressed(p_modifier & UIKeyModifierShift);
		}
		if (p_key != Key::CTRL) {
			ev->set_ctrl_pressed(p_modifier & UIKeyModifierControl);
		}
		if (p_key != Key::ALT) {
			ev->set_alt_pressed(p_modifier & UIKeyModifierAlternate);
		}
		if (p_key != Key::META) {
			ev->set_meta_pressed(p_modifier & UIKeyModifierCommand);
		}
	}
	ev->set_key_label(p_unshifted);
	ev->set_physical_keycode(p_physical);
	ev->set_unicode(fix_unicode(p_char));
	perform_event(ev);
}

// MARK: Motion

void DisplayServerIOS::update_gravity(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_gravity(Vector3(p_x, p_y, p_z));
}

void DisplayServerIOS::update_accelerometer(float p_x, float p_y, float p_z) {
	// Found out the Z should not be negated! Pass as is!
	Vector3 v_accelerometer = Vector3(
			p_x / kDisplayServerIOSAcceleration,
			p_y / kDisplayServerIOSAcceleration,
			p_z / kDisplayServerIOSAcceleration);

	Input::get_singleton()->set_accelerometer(v_accelerometer);
}

void DisplayServerIOS::update_magnetometer(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_magnetometer(Vector3(p_x, p_y, p_z));
}

void DisplayServerIOS::update_gyroscope(float p_x, float p_y, float p_z) {
	Input::get_singleton()->set_gyroscope(Vector3(p_x, p_y, p_z));
}

// MARK: -

bool DisplayServerIOS::has_feature(Feature p_feature) const {
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
		case FEATURE_TEXT_TO_SPEECH:
			return true;
		default:
			return false;
	}
}

String DisplayServerIOS::get_name() const {
	return "iOS";
}

bool DisplayServerIOS::tts_is_speaking() const {
	ERR_FAIL_NULL_V_MSG(tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts isSpeaking];
}

bool DisplayServerIOS::tts_is_paused() const {
	ERR_FAIL_NULL_V_MSG(tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts isPaused];
}

TypedArray<Dictionary> DisplayServerIOS::tts_get_voices() const {
	ERR_FAIL_NULL_V_MSG(tts, TypedArray<Dictionary>(), "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts getVoices];
}

void DisplayServerIOS::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts speak:p_text voice:p_voice volume:p_volume pitch:p_pitch rate:p_rate utterance_id:p_utterance_id interrupt:p_interrupt];
}

void DisplayServerIOS::tts_pause() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts pauseSpeaking];
}

void DisplayServerIOS::tts_resume() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts resumeSpeaking];
}

void DisplayServerIOS::tts_stop() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts stopSpeaking];
}

bool DisplayServerIOS::is_dark_mode_supported() const {
	if (@available(iOS 13.0, *)) {
		return true;
	} else {
		return false;
	}
}

bool DisplayServerIOS::is_dark_mode() const {
	if (@available(iOS 13.0, *)) {
		return [UITraitCollection currentTraitCollection].userInterfaceStyle == UIUserInterfaceStyleDark;
	} else {
		return false;
	}
}

Rect2i DisplayServerIOS::get_display_safe_area() const {
	UIEdgeInsets insets = UIEdgeInsetsZero;
	UIView *view = AppDelegate.viewController.godotView;
	if ([view respondsToSelector:@selector(safeAreaInsets)]) {
		insets = [view safeAreaInsets];
	}
	float scale = screen_get_scale();
	Size2i insets_position = Size2i(insets.left, insets.top) * scale;
	Size2i insets_size = Size2i(insets.left + insets.right, insets.top + insets.bottom) * scale;
	return Rect2i(screen_get_position() + insets_position, screen_get_size() - insets_size);
}

int DisplayServerIOS::get_screen_count() const {
	return 1;
}

int DisplayServerIOS::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerIOS::screen_get_position(int p_screen) const {
	return Size2i();
}

Size2i DisplayServerIOS::screen_get_size(int p_screen) const {
	CALayer *layer = AppDelegate.viewController.godotView.renderingLayer;

	if (!layer) {
		return Size2i();
	}

	return Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_scale(p_screen);
}

Rect2i DisplayServerIOS::screen_get_usable_rect(int p_screen) const {
	return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
}

int DisplayServerIOS::screen_get_dpi(int p_screen) const {
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

float DisplayServerIOS::screen_get_refresh_rate(int p_screen) const {
	return [UIScreen mainScreen].maximumFramesPerSecond;
}

float DisplayServerIOS::screen_get_scale(int p_screen) const {
	return [UIScreen mainScreen].scale;
}

Vector<DisplayServer::WindowID> DisplayServerIOS::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerIOS::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

int64_t DisplayServerIOS::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
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

void DisplayServerIOS::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerIOS::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerIOS::window_set_title(const String &p_title, WindowID p_window) {
	// Probably not supported for iOS
}

int DisplayServerIOS::window_get_current_screen(WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerIOS::window_set_current_screen(int p_screen, WindowID p_window) {
	// Probably not supported for iOS
}

Point2i DisplayServerIOS::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerIOS::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerIOS::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerIOS::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Probably not supported for iOS
}

void DisplayServerIOS::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIOS::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerIOS::window_set_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerIOS::window_get_size(WindowID p_window) const {
	CGRect screenBounds = [UIScreen mainScreen].bounds;
	return Size2i(screenBounds.size.width, screenBounds.size.height) * screen_get_max_scale();
}

Size2i DisplayServerIOS::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerIOS::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Probably not supported for iOS
}

DisplayServer::WindowMode DisplayServerIOS::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerIOS::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerIOS::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerIOS::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerIOS::window_request_attention(WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerIOS::window_move_to_foreground(WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerIOS::window_is_focused(WindowID p_window) const {
	return true;
}

float DisplayServerIOS::screen_get_max_scale() const {
	return screen_get_scale(SCREEN_OF_MAIN_WINDOW);
}

void DisplayServerIOS::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	screen_orientation = p_orientation;
	if (@available(iOS 16.0, *)) {
		[AppDelegate.viewController setNeedsUpdateOfSupportedInterfaceOrientations];
	} else {
		[UIViewController attemptRotationToDeviceOrientation];
	}
}

DisplayServer::ScreenOrientation DisplayServerIOS::screen_get_orientation(int p_screen) const {
	return screen_orientation;
}

bool DisplayServerIOS::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerIOS::can_any_window_draw() const {
	return true;
}

bool DisplayServerIOS::is_touchscreen_available() const {
	return true;
}

_FORCE_INLINE_ int _convert_utf32_offset_to_utf16(const String &p_existing_text, int p_pos) {
	int limit = p_pos;
	for (int i = 0; i < MIN(p_existing_text.length(), p_pos); i++) {
		if (p_existing_text[i] > 0xffff) {
			limit++;
		}
	}
	return limit;
}

void DisplayServerIOS::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
	AppDelegate.viewController.keyboardView.textContentType = nil;
	switch (p_type) {
		case KEYBOARD_TYPE_DEFAULT: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_MULTILINE: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_NUMBER: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeNumberPad;
		} break;
		case KEYBOARD_TYPE_NUMBER_DECIMAL: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDecimalPad;
		} break;
		case KEYBOARD_TYPE_PHONE: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypePhonePad;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeTelephoneNumber;
		} break;
		case KEYBOARD_TYPE_EMAIL_ADDRESS: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeEmailAddress;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeEmailAddress;
		} break;
		case KEYBOARD_TYPE_PASSWORD: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypePassword;
		} break;
		case KEYBOARD_TYPE_URL: {
			AppDelegate.viewController.keyboardView.keyboardType = UIKeyboardTypeWebSearch;
			AppDelegate.viewController.keyboardView.textContentType = UITextContentTypeURL;
		} break;
	}

	[AppDelegate.viewController.keyboardView
			becomeFirstResponderWithString:existingString
							   cursorStart:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_start)
								 cursorEnd:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_end)];
}

bool DisplayServerIOS::is_keyboard_active() const {
	return [AppDelegate.viewController.keyboardView isFirstResponder];
}

void DisplayServerIOS::virtual_keyboard_hide() {
	[AppDelegate.viewController.keyboardView resignFirstResponder];
}

void DisplayServerIOS::virtual_keyboard_set_height(int height) {
	virtual_keyboard_height = height * screen_get_max_scale();
}

int DisplayServerIOS::virtual_keyboard_get_height() const {
	return virtual_keyboard_height;
}

void DisplayServerIOS::clipboard_set(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8()];
}

String DisplayServerIOS::clipboard_get() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}

void DisplayServerIOS::screen_set_keep_on(bool p_enable) {
	[UIApplication sharedApplication].idleTimerDisabled = p_enable;
}

bool DisplayServerIOS::screen_is_kept_on() const {
	return [UIApplication sharedApplication].idleTimerDisabled;
}

void DisplayServerIOS::resize_window(CGSize viewSize) {
	Size2i size = Size2i(viewSize.width, viewSize.height) * screen_get_max_scale();

#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(MAIN_WINDOW_ID, size.x, size.y);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callback, resize_rect);
}

void DisplayServerIOS::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerIOS::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		return context_vulkan->get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}
