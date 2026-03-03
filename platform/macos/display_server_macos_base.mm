/**************************************************************************/
/*  display_server_macos_base.mm                                          */
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

#import "display_server_macos_base.h"
#import "godot_application_delegate.h"
#import "key_mapping_macos.h"
#import "tts_macos.h"

#include "core/config/project_settings.h"
#include "core/os/main_loop.h"
#include "drivers/png/png_driver_common.h"

#if defined(RD_ENABLED)
#include "servers/rendering/rendering_context_driver.h"
#include "servers/rendering/rendering_device.h"
#endif

#import <Carbon/Carbon.h>

void DisplayServerMacOSBase::_mouse_update_mode() {
	MouseMode wanted_mouse_mode = mouse_mode_override_enabled
			? mouse_mode_override
			: mouse_mode_base;

	if (wanted_mouse_mode == mouse_mode) {
		return;
	}

	MouseMode prev_mode = mouse_mode;
	mouse_mode = wanted_mouse_mode;
	_mouse_apply_mode(prev_mode, wanted_mouse_mode);
}

void DisplayServerMacOSBase::mouse_set_mode(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, MouseMode::MOUSE_MODE_MAX);
	if (p_mode == mouse_mode_base) {
		return;
	}
	mouse_mode_base = p_mode;
	_mouse_update_mode();
}

DisplayServer::MouseMode DisplayServerMacOSBase::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerMacOSBase::mouse_set_mode_override(MouseMode p_mode) {
	ERR_FAIL_INDEX(p_mode, MouseMode::MOUSE_MODE_MAX);
	if (p_mode == mouse_mode_override) {
		return;
	}
	mouse_mode_override = p_mode;
	_mouse_update_mode();
}

DisplayServer::MouseMode DisplayServerMacOSBase::mouse_get_mode_override() const {
	return mouse_mode_override;
}

void DisplayServerMacOSBase::mouse_set_mode_override_enabled(bool p_override_enabled) {
	if (p_override_enabled == mouse_mode_override_enabled) {
		return;
	}
	mouse_mode_override_enabled = p_override_enabled;
	_mouse_update_mode();
}

bool DisplayServerMacOSBase::mouse_is_mode_override_enabled() const {
	return mouse_mode_override_enabled;
}

void DisplayServerMacOSBase::clipboard_set(const String &p_text) {
	_THREAD_SAFE_METHOD_

	NSString *copiedString = [NSString stringWithUTF8String:p_text.utf8().get_data()];
	NSArray *copiedStringArray = [NSArray arrayWithObject:copiedString];

	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	[pasteboard clearContents];
	[pasteboard writeObjects:copiedStringArray];
}

String DisplayServerMacOSBase::clipboard_get() const {
	_THREAD_SAFE_METHOD_

	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSArray *classArray = [NSArray arrayWithObject:[NSString class]];
	NSDictionary *options = [NSDictionary dictionary];

	BOOL ok = [pasteboard canReadObjectForClasses:classArray options:options];

	if (!ok) {
		return "";
	}

	NSArray *objectsToPaste = [pasteboard readObjectsForClasses:classArray options:options];
	if (!objectsToPaste || [objectsToPaste count] < 1) {
		return "";
	}
	NSString *string = [objectsToPaste objectAtIndex:0];

	String ret;
	ret.append_utf8([string UTF8String]);
	return ret;
}

Ref<Image> DisplayServerMacOSBase::clipboard_get_image() const {
	Ref<Image> image;
	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSString *result = [pasteboard availableTypeFromArray:[NSArray arrayWithObjects:NSPasteboardTypeTIFF, NSPasteboardTypePNG, nil]];
	if (!result) {
		return image;
	}
	NSData *data = [pasteboard dataForType:result];
	if (!data) {
		return image;
	}
	NSBitmapImageRep *bitmap = [NSBitmapImageRep imageRepWithData:data];
	NSData *pngData = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
	image.instantiate();
	PNGDriverCommon::png_to_image((const uint8_t *)pngData.bytes, pngData.length, false, image);
	return image;
}

bool DisplayServerMacOSBase::clipboard_has() const {
	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSArray *classArray = [NSArray arrayWithObject:[NSString class]];
	NSDictionary *options = [NSDictionary dictionary];
	return [pasteboard canReadObjectForClasses:classArray options:options];
}

bool DisplayServerMacOSBase::clipboard_has_image() const {
	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSString *result = [pasteboard availableTypeFromArray:[NSArray arrayWithObjects:NSPasteboardTypeTIFF, NSPasteboardTypePNG, nil]];
	return result;
}

CGDirectDisplayID DisplayServerMacOSBase::_get_display_id_for_screen(NSScreen *p_screen) {
	if (@available(macOS 26.0, *)) {
		return [p_screen CGDirectDisplayID];
	} else {
		return [[p_screen deviceDescription][@"NSScreenNumber"] unsignedIntValue];
	}
}

void DisplayServerMacOSBase::initialize_tts() const {
	const_cast<DisplayServerMacOSBase *>(this)->tts = [[TTS_MacOS alloc] init];
}

bool DisplayServerMacOSBase::tts_is_speaking() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, false);
	return [tts isSpeaking];
}

bool DisplayServerMacOSBase::tts_is_paused() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, false);
	return [tts isPaused];
}

TypedArray<Dictionary> DisplayServerMacOSBase::tts_get_voices() const {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL_V(tts, TypedArray<Dictionary>());
	return [tts getVoices];
}

void DisplayServerMacOSBase::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int64_t p_utterance_id, bool p_interrupt) {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts speak:p_text voice:p_voice volume:p_volume pitch:p_pitch rate:p_rate utterance_id:p_utterance_id interrupt:p_interrupt];
}

void DisplayServerMacOSBase::tts_pause() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts pauseSpeaking];
}

void DisplayServerMacOSBase::tts_resume() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts resumeSpeaking];
}

void DisplayServerMacOSBase::tts_stop() {
	if (unlikely(!tts)) {
		initialize_tts();
	}
	ERR_FAIL_NULL(tts);
	[tts stopSpeaking];
}

void DisplayServerMacOSBase::_update_keyboard_layouts() const {
	kbd_layouts.clear();
	current_layout = 0;

	TISInputSourceRef cur_source = TISCopyCurrentKeyboardInputSource();
	NSString *cur_name = (__bridge NSString *)TISGetInputSourceProperty(cur_source, kTISPropertyLocalizedName);
	CFRelease(cur_source);

	// Enum IME layouts.
	NSDictionary *filter_ime = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardInputMode };
	NSArray *list_ime = (__bridge NSArray *)TISCreateInputSourceList((__bridge CFDictionaryRef)filter_ime, false);
	for (NSUInteger i = 0; i < [list_ime count]; i++) {
		LayoutInfo ly;
		NSString *name = (__bridge NSString *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyLocalizedName);
		ly.name.append_utf8([name UTF8String]);

		NSArray *langs = (__bridge NSArray *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyInputSourceLanguages);
		ly.code.append_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
		kbd_layouts.push_back(ly);

		if ([name isEqualToString:cur_name]) {
			current_layout = kbd_layouts.size() - 1;
		}
	}

	// Enum plain keyboard layouts.
	NSDictionary *filter_kbd = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardLayout };
	NSArray *list_kbd = (__bridge NSArray *)TISCreateInputSourceList((__bridge CFDictionaryRef)filter_kbd, false);
	for (NSUInteger i = 0; i < [list_kbd count]; i++) {
		LayoutInfo ly;
		NSString *name = (__bridge NSString *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyLocalizedName);
		ly.name.append_utf8([name UTF8String]);

		NSArray *langs = (__bridge NSArray *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyInputSourceLanguages);
		ly.code.append_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
		kbd_layouts.push_back(ly);

		if ([name isEqualToString:cur_name]) {
			current_layout = kbd_layouts.size() - 1;
		}
	}

	keyboard_layout_dirty = false;
}

void DisplayServerMacOSBase::_keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info) {
	DisplayServerMacOSBase *ds = (DisplayServerMacOSBase *)DisplayServer::get_singleton();
	if (ds) {
		ds->keyboard_layout_dirty = true;
	}
}

int DisplayServerMacOSBase::keyboard_get_layout_count() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}
	return kbd_layouts.size();
}

void DisplayServerMacOSBase::keyboard_set_current_layout(int p_index) {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX(p_index, kbd_layouts.size());

	NSString *cur_name = [NSString stringWithUTF8String:kbd_layouts[p_index].name.utf8().get_data()];

	NSDictionary *filter_kbd = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardLayout };
	NSArray *list_kbd = (__bridge NSArray *)TISCreateInputSourceList((__bridge CFDictionaryRef)filter_kbd, false);
	for (NSUInteger i = 0; i < [list_kbd count]; i++) {
		NSString *name = (__bridge NSString *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyLocalizedName);
		if ([name isEqualToString:cur_name]) {
			TISSelectInputSource((__bridge TISInputSourceRef)[list_kbd objectAtIndex:i]);
			break;
		}
	}

	NSDictionary *filter_ime = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardInputMode };
	NSArray *list_ime = (__bridge NSArray *)TISCreateInputSourceList((__bridge CFDictionaryRef)filter_ime, false);
	for (NSUInteger i = 0; i < [list_ime count]; i++) {
		NSString *name = (__bridge NSString *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyLocalizedName);
		if ([name isEqualToString:cur_name]) {
			TISSelectInputSource((__bridge TISInputSourceRef)[list_ime objectAtIndex:i]);
			break;
		}
	}
}

int DisplayServerMacOSBase::keyboard_get_current_layout() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	return current_layout;
}

String DisplayServerMacOSBase::keyboard_get_layout_language(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].code;
}

String DisplayServerMacOSBase::keyboard_get_layout_name(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].name;
}

Key DisplayServerMacOSBase::keyboard_get_keycode_from_physical(Key p_keycode) const {
	if (p_keycode == Key::PAUSE || p_keycode == Key::NONE) {
		return p_keycode;
	}

	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = p_keycode & KeyModifierMask::CODE_MASK;
	unsigned int macos_keycode = KeyMappingMacOS::unmap_key(keycode_no_mod);
	return (Key)(KeyMappingMacOS::remap_key(macos_keycode, 0, false) | modifiers);
}

Key DisplayServerMacOSBase::keyboard_get_label_from_physical(Key p_keycode) const {
	if (p_keycode == Key::PAUSE || p_keycode == Key::NONE) {
		return p_keycode;
	}

	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = p_keycode & KeyModifierMask::CODE_MASK;
	unsigned int macos_keycode = KeyMappingMacOS::unmap_key(keycode_no_mod);
	return (Key)(KeyMappingMacOS::remap_key(macos_keycode, 0, true) | modifiers);
}

void DisplayServerMacOSBase::show_emoji_and_symbol_picker() const {
	[[NSApplication sharedApplication] orderFrontCharacterPalette:nil];
}

bool DisplayServerMacOSBase::is_dark_mode_supported() const {
	if (@available(macOS 10.14, *)) {
		return true;
	} else {
		return false;
	}
}

bool DisplayServerMacOSBase::is_dark_mode() const {
	if (@available(macOS 10.14, *)) {
		if (![[NSUserDefaults standardUserDefaults] objectForKey:@"AppleInterfaceStyle"]) {
			return false;
		} else {
			return ([[[NSUserDefaults standardUserDefaults] stringForKey:@"AppleInterfaceStyle"] isEqual:@"Dark"]);
		}
	} else {
		return false;
	}
}

Color DisplayServerMacOSBase::get_accent_color() const {
	if (@available(macOS 10.14, *)) {
		__block NSColor *color = nullptr;
		if (@available(macOS 11.0, *)) {
			[NSApp.effectiveAppearance performAsCurrentDrawingAppearance:^{
				color = [[NSColor controlAccentColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			}];
			if (!color) {
				color = [[NSColor controlAccentColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			}
		} else {
			NSAppearance *saved_appearance = [NSAppearance currentAppearance];
			[NSAppearance setCurrentAppearance:[NSApp effectiveAppearance]];
			color = [[NSColor controlAccentColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			[NSAppearance setCurrentAppearance:saved_appearance];
		}
		if (color) {
			CGFloat components[4];
			[color getRed:&components[0] green:&components[1] blue:&components[2] alpha:&components[3]];
			return Color(components[0], components[1], components[2], components[3]);
		} else {
			return Color(0, 0, 0, 0);
		}
	} else {
		return Color(0, 0, 0, 0);
	}
}

Color DisplayServerMacOSBase::get_base_color() const {
	if (@available(macOS 10.14, *)) {
		__block NSColor *color = nullptr;
		if (@available(macOS 11.0, *)) {
			[NSApp.effectiveAppearance performAsCurrentDrawingAppearance:^{
				color = [[NSColor windowBackgroundColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			}];
			if (!color) {
				color = [[NSColor controlColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			}
		} else {
			NSAppearance *saved_appearance = [NSAppearance currentAppearance];
			[NSAppearance setCurrentAppearance:[NSApp effectiveAppearance]];
			color = [[NSColor controlColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
			[NSAppearance setCurrentAppearance:saved_appearance];
		}
		if (color) {
			CGFloat components[4];
			[color getRed:&components[0] green:&components[1] blue:&components[2] alpha:&components[3]];
			return Color(components[0], components[1], components[2], components[3]);
		} else {
			return Color(0, 0, 0, 0);
		}
	} else {
		return Color(0, 0, 0, 0);
	}
}

void DisplayServerMacOSBase::set_system_theme_change_callback(const Callable &p_callable) {
	system_theme_changed = p_callable;
}

bool DisplayServerMacOSBase::screen_is_kept_on() const {
	return (screen_keep_on_assertion);
}

void DisplayServerMacOSBase::screen_set_keep_on(bool p_enable) {
	if (screen_keep_on_assertion) {
		IOPMAssertionRelease(screen_keep_on_assertion);
		screen_keep_on_assertion = kIOPMNullAssertionID;
	}

	if (p_enable) {
		String app_name_string = GLOBAL_GET("application/config/name");
		NSString *name = [NSString stringWithUTF8String:(app_name_string.is_empty() ? "Godot Engine" : app_name_string.utf8().get_data())];
		NSString *reason = @"Godot Engine running with display/window/energy_saving/keep_screen_on = true";
		IOPMAssertionCreateWithDescription(kIOPMAssertPreventUserIdleDisplaySleep, (__bridge CFStringRef)name, (__bridge CFStringRef)reason, (__bridge CFStringRef)reason, nullptr, 0, nullptr, &screen_keep_on_assertion);
	}
}

int DisplayServerMacOSBase::accessibility_should_increase_contrast() const {
	return [(GodotApplicationDelegate *)[[NSApplication sharedApplication] delegate] getHighContrast];
}

int DisplayServerMacOSBase::accessibility_should_reduce_animation() const {
	return [(GodotApplicationDelegate *)[[NSApplication sharedApplication] delegate] getReduceMotion];
}

int DisplayServerMacOSBase::accessibility_should_reduce_transparency() const {
	return [(GodotApplicationDelegate *)[[NSApplication sharedApplication] delegate] getReduceTransparency];
}

int DisplayServerMacOSBase::accessibility_screen_reader_active() const {
	return [(GodotApplicationDelegate *)[[NSApplication sharedApplication] delegate] getVoiceOver];
}

void DisplayServerMacOSBase::update_im_text(const Point2i &p_selection, const String &p_text) {
	im_selection = p_selection;
	im_text = p_text;

	OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
}

Point2i DisplayServerMacOSBase::ime_get_selection() const {
	return im_selection;
}

String DisplayServerMacOSBase::ime_get_text() const {
	return im_text;
}

DisplayServer::CursorShape DisplayServerMacOSBase::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerMacOSBase::beep() const {
	NSBeep();
}

int DisplayServerMacOSBase::get_primary_screen() const {
	return 0;
}

float DisplayServerMacOSBase::screen_get_refresh_rate(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSScreen *screen = [screenArray objectAtIndex:p_screen];
		const CGDisplayModeRef displayMode = CGDisplayCopyDisplayMode(_get_display_id_for_screen(screen));
		const double displayRefreshRate = CGDisplayModeGetRefreshRate(displayMode);
		return (float)displayRefreshRate;
	}
	ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
	return SCREEN_REFRESH_RATE_FALLBACK;
}

void DisplayServerMacOSBase::emit_system_theme_changed() {
	if (system_theme_changed.is_valid()) {
		Variant ret;
		Callable::CallError ce;
		system_theme_changed.callp(nullptr, 0, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute system theme changed callback: %s.", Variant::get_callable_error_text(system_theme_changed, nullptr, 0, ce)));
		}
	}
}

// MARK: - HDR / EDR

void DisplayServerMacOSBase::_update_hdr_output(WindowID p_window, const HDROutput &p_hdr) {
#ifdef RD_ENABLED
	if (!rendering_context) {
		return;
	}

	CGFloat max_potential_edr, max_edr;
	window_get_edr_values(p_window, &max_potential_edr, &max_edr);
	bool desired_hdr_enabled = p_hdr.requested && max_potential_edr > 1.0f;
	bool current_hdr_enabled = rendering_context->window_get_hdr_output_enabled(p_window);
	if (current_hdr_enabled != desired_hdr_enabled) {
		rendering_context->window_set_hdr_output_enabled(p_window, desired_hdr_enabled);
	}

	float reference_luminance = _calculate_current_reference_luminance(max_potential_edr, max_edr);
	rendering_context->window_set_hdr_output_reference_luminance(p_window, reference_luminance);

	float max_luminance = p_hdr.is_auto_max_luminance() ? max_potential_edr * HARDWARE_REFERENCE_LUMINANCE_NITS : p_hdr.max_luminance;
	rendering_context->window_set_hdr_output_max_luminance(p_window, max_luminance);
#endif
}

bool DisplayServerMacOSBase::window_is_hdr_output_supported(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

#if defined(RD_ENABLED)
	if (rendering_device && !rendering_device->has_feature(RenderingDevice::Features::SUPPORTS_HDR_OUTPUT)) {
		return false;
	}
#endif
	CGFloat max_potential_edr;
	window_get_edr_values(p_window, &max_potential_edr, nullptr);
	return max_potential_edr > 1.0f;
}

void DisplayServerMacOSBase::window_request_hdr_output(const bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

#if defined(RD_ENABLED)
	ERR_FAIL_COND_MSG(p_enabled && rendering_device && !rendering_device->has_feature(RenderingDevice::Features::SUPPORTS_HDR_OUTPUT), "HDR output is not supported by the rendering device.");
#endif

	HDROutput &hdr = _get_hdr_output(p_window);
	hdr.requested = p_enabled;
	_update_hdr_output(p_window, hdr);
}

bool DisplayServerMacOSBase::window_is_hdr_output_requested(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	return _get_hdr_output(p_window).requested;
}

bool DisplayServerMacOSBase::window_is_hdr_output_enabled(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_hdr_output_enabled(p_window);
	}
#endif

	return false;
}

void DisplayServerMacOSBase::window_set_hdr_output_reference_luminance(const float p_reference_luminance, WindowID p_window) {
	ERR_PRINT_ONCE("Manually setting reference white luminance is not supported on Apple devices, as they provide a user-facing brightness setting that directly controls reference white luminance.");
}

float DisplayServerMacOSBase::window_get_hdr_output_reference_luminance(WindowID p_window) const {
	return -1.0f; // Always auto-adjusted by the OS on Apple platforms.
}

constexpr float DisplayServerMacOSBase::_calculate_current_reference_luminance(CGFloat p_max_potential_edr_value, CGFloat p_max_edr_value) const {
	return (p_max_potential_edr_value * HARDWARE_REFERENCE_LUMINANCE_NITS) / p_max_edr_value;
}

float DisplayServerMacOSBase::window_get_hdr_output_current_reference_luminance(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_hdr_output_reference_luminance(p_window);
	}
#endif
	return 200.0f;
}

void DisplayServerMacOSBase::window_set_hdr_output_max_luminance(const float p_max_luminance, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	HDROutput &hdr = _get_hdr_output(p_window);

	if (hdr.max_luminance == p_max_luminance) {
		return;
	}
	hdr.max_luminance = p_max_luminance;
	_update_hdr_output(p_window, hdr);
}

float DisplayServerMacOSBase::window_get_hdr_output_max_luminance(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	return _get_hdr_output(p_window).max_luminance;
}

float DisplayServerMacOSBase::window_get_hdr_output_current_max_luminance(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	const HDROutput &hdr = _get_hdr_output(p_window);
	if (hdr.is_auto_max_luminance()) {
		CGFloat max_potential_edr;
		window_get_edr_values(p_window, &max_potential_edr, nullptr);
		return max_potential_edr * HARDWARE_REFERENCE_LUMINANCE_NITS;
	}
	return hdr.max_luminance;
}

float DisplayServerMacOSBase::window_get_output_max_linear_value(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

#if defined(RD_ENABLED)
	if (rendering_context) {
		return rendering_context->window_get_output_max_linear_value(p_window);
	}
#endif

	return 1.0f; // SDR
}

DisplayServerMacOSBase::DisplayServerMacOSBase() {
	KeyMappingMacOS::initialize();

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		initialize_tts();
	}

	screen_set_keep_on(GLOBAL_GET("display/window/energy_saving/keep_screen_on"));

	// Register to be notified on keyboard layout changes.
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
			nullptr, _keyboard_layout_changed,
			kTISNotifySelectedKeyboardInputSourceChanged, nullptr,
			CFNotificationSuspensionBehaviorDeliverImmediately);
}

DisplayServerMacOSBase::~DisplayServerMacOSBase() {
	if (screen_keep_on_assertion) {
		IOPMAssertionRelease(screen_keep_on_assertion);
		screen_keep_on_assertion = kIOPMNullAssertionID;
	}

	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), nullptr, kTISNotifySelectedKeyboardInputSourceChanged, nullptr);
}
