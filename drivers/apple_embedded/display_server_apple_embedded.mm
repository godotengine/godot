/**************************************************************************/
/*  display_server_apple_embedded.mm                                      */
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

#import "display_server_apple_embedded.h"

#import "apple_embedded.h"
#import "godot_app_delegate_service_apple_embedded.h"
#import "godot_keyboard_input_view.h"
#import "godot_view_apple_embedded.h"
#import "godot_view_controller.h"
#import "key_mapping_apple_embedded.h"
#import "os_apple_embedded.h"
#import "tts_apple_embedded.h"

#include "core/config/project_settings.h"
#include "core/io/file_access_pack.h"

#import <GameController/GameController.h>

static const float kDisplayServerIOSAcceleration = 1.f;

DisplayServerAppleEmbedded *DisplayServerAppleEmbedded::get_singleton() {
	return (DisplayServerAppleEmbedded *)DisplayServer::get_singleton();
}

DisplayServerAppleEmbedded::DisplayServerAppleEmbedded(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	KeyMappingAppleEmbedded::initialize();

	rendering_driver = p_rendering_driver;

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		initialize_tts();
	}
	native_menu = memnew(NativeMenu);

	bool has_made_render_compositor_current = false;

#if defined(RD_ENABLED)
	rendering_context = nullptr;
	rendering_device = nullptr;

	CALayer *layer = nullptr;

	union {
#ifdef VULKAN_ENABLED
		RenderingContextDriverVulkanAppleEmbedded::WindowPlatformData vulkan;
#endif
#ifdef METAL_ENABLED
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
		// Eliminate "RenderingContextDriverMetal is only available on iOS 14.0 or newer".
		RenderingContextDriverMetal::WindowPlatformData metal;
		GODOT_CLANG_WARNING_POP
#endif
	} wpd;

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		layer = [GDTAppDelegateService.viewController.godotView initializeRenderingForDriver:@"vulkan"];
		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS Vulkan rendering layer.");
		}
		wpd.vulkan.layer_ptr = (CAMetalLayer *const *)&layer;
		rendering_context = memnew(RenderingContextDriverVulkanAppleEmbedded);
	}
#endif
#ifdef METAL_ENABLED
	if (rendering_driver == "metal") {
		if (@available(iOS 14.0, *)) {
			layer = [GDTAppDelegateService.viewController.godotView initializeRenderingForDriver:@"metal"];
			wpd.metal.layer = (CAMetalLayer *)layer;
			rendering_context = memnew(RenderingContextDriverMetal);
		} else {
			OS::get_singleton()->alert("Metal is only supported on iOS 14.0 and later.");
			r_error = ERR_UNAVAILABLE;
			return;
		}
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
				ERR_PRINT(vformat("Failed to initialize %s context", rendering_driver));
				r_error = ERR_UNAVAILABLE;
				return;
			}
		}
	}

	if (rendering_context) {
		if (rendering_context->window_create(MAIN_WINDOW_ID, &wpd) != OK) {
			ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}

		Size2i size = Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_max_scale();
		rendering_context->window_set_size(MAIN_WINDOW_ID, size.width, size.height);
		rendering_context->window_set_vsync_mode(MAIN_WINDOW_ID, p_vsync_mode);

		rendering_device = memnew(RenderingDevice);
		if (rendering_device->initialize(rendering_context, MAIN_WINDOW_ID) != OK) {
			rendering_device = nullptr;
			memdelete(rendering_context);
			rendering_context = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}
		rendering_device->screen_create(MAIN_WINDOW_ID);

		RendererCompositorRD::make_current();
		has_made_render_compositor_current = true;
	}
#endif

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		CALayer *layer = [GDTAppDelegateService.viewController.godotView initializeRenderingForDriver:@"opengl3"];

		if (!layer) {
			ERR_FAIL_MSG("Failed to create iOS OpenGLES rendering layer.");
		}

		RasterizerGLES3::make_current(false);
		has_made_render_compositor_current = true;
	}
#endif

	ERR_FAIL_COND_MSG(!has_made_render_compositor_current, vformat("Failed to make RendererCompositor current for rendering driver %s", rendering_driver));

	bool keep_screen_on = bool(GLOBAL_GET("display/window/energy_saving/keep_screen_on"));
	screen_set_keep_on(keep_screen_on);

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerAppleEmbedded::~DisplayServerAppleEmbedded() {
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
}

Vector<String> DisplayServerAppleEmbedded::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(METAL_ENABLED)
	if (@available(ios 14.0, *)) {
		drivers.push_back("metal");
	}
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl3");
#endif

	return drivers;
}

// MARK: Events

void DisplayServerAppleEmbedded::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	window_resize_callback = p_callable;
}

void DisplayServerAppleEmbedded::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	window_event_callback = p_callable;
}
void DisplayServerAppleEmbedded::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerAppleEmbedded::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerAppleEmbedded::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerAppleEmbedded::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

void DisplayServerAppleEmbedded::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	DisplayServerAppleEmbedded::get_singleton()->send_input_event(p_event);
}

void DisplayServerAppleEmbedded::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, p_event);
}

void DisplayServerAppleEmbedded::send_input_text(const String &p_text) const {
	_window_callback(input_text_callback, p_text);
}

void DisplayServerAppleEmbedded::send_window_event(DisplayServer::WindowEvent p_event) const {
	_window_callback(window_event_callback, int(p_event));
}

void DisplayServerAppleEmbedded::_window_callback(const Callable &p_callable, const Variant &p_arg) const {
	if (p_callable.is_valid()) {
		p_callable.call(p_arg);
	}
}

// MARK: - Input

// MARK: Touches

void DisplayServerAppleEmbedded::touch_press(int p_idx, int p_x, int p_y, bool p_pressed, bool p_double_click) {
	Ref<InputEventScreenTouch> ev;
	ev.instantiate();

	ev->set_index(p_idx);
	ev->set_pressed(p_pressed);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_double_tap(p_double_click);
	perform_event(ev);
}

void DisplayServerAppleEmbedded::touch_drag(int p_idx, int p_prev_x, int p_prev_y, int p_x, int p_y, float p_pressure, Vector2 p_tilt) {
	Ref<InputEventScreenDrag> ev;
	ev.instantiate();
	ev->set_index(p_idx);
	ev->set_pressure(p_pressure);
	ev->set_tilt(p_tilt);
	ev->set_position(Vector2(p_x, p_y));
	ev->set_relative(Vector2(p_x - p_prev_x, p_y - p_prev_y));
	ev->set_relative_screen_position(ev->get_relative());
	perform_event(ev);
}

void DisplayServerAppleEmbedded::perform_event(const Ref<InputEvent> &p_event) {
	Input *input_singleton = Input::get_singleton();
	if (input_singleton == nullptr) {
		return;
	}

	input_singleton->parse_input_event(p_event);
}

void DisplayServerAppleEmbedded::touches_canceled(int p_idx) {
	touch_press(p_idx, -1, -1, false, false);
}

// MARK: Keyboard

void DisplayServerAppleEmbedded::key(Key p_key, char32_t p_char, Key p_unshifted, Key p_physical, NSInteger p_modifier, bool p_pressed, KeyLocation p_location) {
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
	ev->set_location(p_location);
	perform_event(ev);
}

// MARK: Motion

void DisplayServerAppleEmbedded::update_gravity(const Vector3 &p_gravity) {
	Input::get_singleton()->set_gravity(p_gravity);
}

void DisplayServerAppleEmbedded::update_accelerometer(const Vector3 &p_accelerometer) {
	Input::get_singleton()->set_accelerometer(p_accelerometer / kDisplayServerIOSAcceleration);
}

void DisplayServerAppleEmbedded::update_magnetometer(const Vector3 &p_magnetometer) {
	Input::get_singleton()->set_magnetometer(p_magnetometer);
}

void DisplayServerAppleEmbedded::update_gyroscope(const Vector3 &p_gyroscope) {
	Input::get_singleton()->set_gyroscope(p_gyroscope);
}

// MARK: -

bool DisplayServerAppleEmbedded::has_feature(Feature p_feature) const {
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
		// case FEATURE_NATIVE_DIALOG_INPUT:
		// case FEATURE_NATIVE_DIALOG_FILE:
		// case FEATURE_NATIVE_DIALOG_FILE_EXTRA:
		// case FEATURE_NATIVE_DIALOG_FILE_MIME:
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

void DisplayServerAppleEmbedded::initialize_tts() const {
	const_cast<DisplayServerAppleEmbedded *>(this)->tts = [[GDTTTS alloc] init];
}

bool DisplayServerAppleEmbedded::tts_is_speaking() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, false);
	return [tts isSpeaking];
}

bool DisplayServerAppleEmbedded::tts_is_paused() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, false);
	return [tts isPaused];
}

TypedArray<Dictionary> DisplayServerAppleEmbedded::tts_get_voices() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, TypedArray<Dictionary>());
	return [tts getVoices];
}

void DisplayServerAppleEmbedded::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts speak:p_text voice:p_voice volume:p_volume pitch:p_pitch rate:p_rate utterance_id:p_utterance_id interrupt:p_interrupt];
}

void DisplayServerAppleEmbedded::tts_pause() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts pauseSpeaking];
}

void DisplayServerAppleEmbedded::tts_resume() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts resumeSpeaking];
}

void DisplayServerAppleEmbedded::tts_stop() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts stopSpeaking];
}

bool DisplayServerAppleEmbedded::is_dark_mode_supported() const {
	if (@available(iOS 13.0, *)) {
		return true;
	} else {
		return false;
	}
}

bool DisplayServerAppleEmbedded::is_dark_mode() const {
	if (@available(iOS 13.0, *)) {
		return [UITraitCollection currentTraitCollection].userInterfaceStyle == UIUserInterfaceStyleDark;
	} else {
		return false;
	}
}

void DisplayServerAppleEmbedded::set_system_theme_change_callback(const Callable &p_callable) {
	system_theme_changed = p_callable;
}

void DisplayServerAppleEmbedded::emit_system_theme_changed() {
	if (system_theme_changed.is_valid()) {
		Variant ret;
		Callable::CallError ce;
		system_theme_changed.callp(nullptr, 0, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute system theme changed callback: %s.", Variant::get_callable_error_text(system_theme_changed, nullptr, 0, ce)));
		}
	}
}

Rect2i DisplayServerAppleEmbedded::get_display_safe_area() const {
	UIEdgeInsets insets = UIEdgeInsetsZero;
	UIView *view = GDTAppDelegateService.viewController.godotView;
	if ([view respondsToSelector:@selector(safeAreaInsets)]) {
		insets = [view safeAreaInsets];
	}
	float scale = screen_get_scale();
	Size2i insets_position = Size2i(insets.left, insets.top) * scale;
	Size2i insets_size = Size2i(insets.left + insets.right, insets.top + insets.bottom) * scale;
	return Rect2i(screen_get_position() + insets_position, screen_get_size() - insets_size);
}

int DisplayServerAppleEmbedded::get_screen_count() const {
	return 1;
}

int DisplayServerAppleEmbedded::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerAppleEmbedded::screen_get_position(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Point2i());

	return Point2i(0, 0);
}

Size2i DisplayServerAppleEmbedded::screen_get_size(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Size2i());

	CALayer *layer = GDTAppDelegateService.viewController.godotView.renderingLayer;

	if (!layer) {
		return Size2i();
	}

	return Size2i(layer.bounds.size.width, layer.bounds.size.height) * screen_get_scale(p_screen);
}

Rect2i DisplayServerAppleEmbedded::screen_get_usable_rect(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Rect2i());

	return Rect2i(screen_get_position(p_screen), screen_get_size(p_screen));
}

Vector<DisplayServer::WindowID> DisplayServerAppleEmbedded::get_window_list() const {
	Vector<DisplayServer::WindowID> list;
	list.push_back(MAIN_WINDOW_ID);
	return list;
}

DisplayServer::WindowID DisplayServerAppleEmbedded::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

int64_t DisplayServerAppleEmbedded::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, 0);
	switch (p_handle_type) {
		case DISPLAY_HANDLE: {
			return 0; // Not supported.
		}
		case WINDOW_HANDLE: {
			return (int64_t)GDTAppDelegateService.viewController;
		}
		case WINDOW_VIEW: {
			return (int64_t)GDTAppDelegateService.viewController.godotView;
		}
		default: {
			return 0;
		}
	}
}

void DisplayServerAppleEmbedded::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerAppleEmbedded::window_get_attached_instance_id(WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerAppleEmbedded::window_set_title(const String &p_title, WindowID p_window) {
	// Probably not supported for iOS
}

int DisplayServerAppleEmbedded::window_get_current_screen(WindowID p_window) const {
	ERR_FAIL_COND_V(p_window != MAIN_WINDOW_ID, INVALID_SCREEN);
	return 0;
}

void DisplayServerAppleEmbedded::window_set_current_screen(int p_screen, WindowID p_window) {
	// Probably not supported for iOS
}

Point2i DisplayServerAppleEmbedded::window_get_position(WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerAppleEmbedded::window_get_position_with_decorations(WindowID p_window) const {
	return Point2i();
}

void DisplayServerAppleEmbedded::window_set_position(const Point2i &p_position, WindowID p_window) {
	// Probably not supported for single window iOS app
}

void DisplayServerAppleEmbedded::window_set_transient(WindowID p_window, WindowID p_parent) {
	// Probably not supported for iOS
}

void DisplayServerAppleEmbedded::window_set_max_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerAppleEmbedded::window_get_max_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerAppleEmbedded::window_set_min_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerAppleEmbedded::window_get_min_size(WindowID p_window) const {
	return Size2i();
}

void DisplayServerAppleEmbedded::window_set_size(const Size2i p_size, WindowID p_window) {
	// Probably not supported for iOS
}

Size2i DisplayServerAppleEmbedded::window_get_size(WindowID p_window) const {
	CGRect viewBounds = GDTAppDelegateService.viewController.view.bounds;
	return Size2i(viewBounds.size.width, viewBounds.size.height) * screen_get_max_scale();
}

Size2i DisplayServerAppleEmbedded::window_get_size_with_decorations(WindowID p_window) const {
	return window_get_size(p_window);
}

void DisplayServerAppleEmbedded::window_set_mode(WindowMode p_mode, WindowID p_window) {
	// Probably not supported for iOS
}

DisplayServer::WindowMode DisplayServerAppleEmbedded::window_get_mode(WindowID p_window) const {
	return WindowMode::WINDOW_MODE_FULLSCREEN;
}

bool DisplayServerAppleEmbedded::window_is_maximize_allowed(WindowID p_window) const {
	return false;
}

void DisplayServerAppleEmbedded::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerAppleEmbedded::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	return false;
}

void DisplayServerAppleEmbedded::window_request_attention(WindowID p_window) {
	// Probably not supported for iOS
}

void DisplayServerAppleEmbedded::window_move_to_foreground(WindowID p_window) {
	// Probably not supported for iOS
}

bool DisplayServerAppleEmbedded::window_is_focused(WindowID p_window) const {
	return true;
}

float DisplayServerAppleEmbedded::screen_get_max_scale() const {
	return screen_get_scale(SCREEN_OF_MAIN_WINDOW);
}

void DisplayServerAppleEmbedded::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX(p_screen, screen_count);

	screen_orientation = p_orientation;
	if (@available(iOS 16.0, *)) {
		[GDTAppDelegateService.viewController setNeedsUpdateOfSupportedInterfaceOrientations];
	}
#if !defined(VISIONOS_ENABLED)
	else {
		[UIViewController attemptRotationToDeviceOrientation];
	}
#endif
}

DisplayServer::ScreenOrientation DisplayServerAppleEmbedded::screen_get_orientation(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_LANDSCAPE);

	return screen_orientation;
}

bool DisplayServerAppleEmbedded::window_can_draw(WindowID p_window) const {
	return true;
}

bool DisplayServerAppleEmbedded::can_any_window_draw() const {
	return true;
}

bool DisplayServerAppleEmbedded::is_touchscreen_available() const {
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

void DisplayServerAppleEmbedded::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	NSString *existingString = [[NSString alloc] initWithUTF8String:p_existing_text.utf8().get_data()];

	GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
	GDTAppDelegateService.viewController.keyboardView.textContentType = nil;
	switch (p_type) {
		case KEYBOARD_TYPE_DEFAULT: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_MULTILINE: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
		} break;
		case KEYBOARD_TYPE_NUMBER: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeNumberPad;
		} break;
		case KEYBOARD_TYPE_NUMBER_DECIMAL: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeDecimalPad;
		} break;
		case KEYBOARD_TYPE_PHONE: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypePhonePad;
			GDTAppDelegateService.viewController.keyboardView.textContentType = UITextContentTypeTelephoneNumber;
		} break;
		case KEYBOARD_TYPE_EMAIL_ADDRESS: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeEmailAddress;
			GDTAppDelegateService.viewController.keyboardView.textContentType = UITextContentTypeEmailAddress;
		} break;
		case KEYBOARD_TYPE_PASSWORD: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeDefault;
			GDTAppDelegateService.viewController.keyboardView.textContentType = UITextContentTypePassword;
		} break;
		case KEYBOARD_TYPE_URL: {
			GDTAppDelegateService.viewController.keyboardView.keyboardType = UIKeyboardTypeWebSearch;
			GDTAppDelegateService.viewController.keyboardView.textContentType = UITextContentTypeURL;
		} break;
	}

	[GDTAppDelegateService.viewController.keyboardView
			becomeFirstResponderWithString:existingString
							   cursorStart:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_start)
								 cursorEnd:_convert_utf32_offset_to_utf16(p_existing_text, p_cursor_end)];
}

bool DisplayServerAppleEmbedded::is_keyboard_active() const {
	return [GDTAppDelegateService.viewController.keyboardView isFirstResponder];
}

void DisplayServerAppleEmbedded::virtual_keyboard_hide() {
	[GDTAppDelegateService.viewController.keyboardView resignFirstResponder];
}

void DisplayServerAppleEmbedded::virtual_keyboard_set_height(int height) {
	virtual_keyboard_height = height * screen_get_max_scale();
}

int DisplayServerAppleEmbedded::virtual_keyboard_get_height() const {
	return virtual_keyboard_height;
}

bool DisplayServerAppleEmbedded::has_hardware_keyboard() const {
	if (@available(iOS 14.0, *)) {
		return [GCKeyboard coalescedKeyboard];
	} else {
		return false;
	}
}

void DisplayServerAppleEmbedded::clipboard_set(const String &p_text) {
	[UIPasteboard generalPasteboard].string = [NSString stringWithUTF8String:p_text.utf8().get_data()];
}

String DisplayServerAppleEmbedded::clipboard_get() const {
	NSString *text = [UIPasteboard generalPasteboard].string;

	return String::utf8([text UTF8String]);
}

void DisplayServerAppleEmbedded::screen_set_keep_on(bool p_enable) {
	[UIApplication sharedApplication].idleTimerDisabled = p_enable;
}

bool DisplayServerAppleEmbedded::screen_is_kept_on() const {
	return [UIApplication sharedApplication].idleTimerDisabled;
}

void DisplayServerAppleEmbedded::resize_window(CGSize viewSize) {
	Size2i size = Size2i(viewSize.width, viewSize.height) * screen_get_max_scale();

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(MAIN_WINDOW_ID, size.x, size.y);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callback, resize_rect);
}

void DisplayServerAppleEmbedded::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerAppleEmbedded::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}

void DisplayServerAppleEmbedded::set_native_icon(const String &p_filename) {
	// Not supported on Apple embedded platforms.
}

void DisplayServerAppleEmbedded::set_icon(const Ref<Image> &p_icon) {
	// Not supported on Apple embedded platforms.
}
