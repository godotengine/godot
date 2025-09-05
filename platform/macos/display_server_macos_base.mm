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
#import "key_mapping_macos.h"
#import "tts_macos.h"

#include "core/config/project_settings.h"
#include "drivers/png/png_driver_common.h"

#import <Carbon/Carbon.h>

void DisplayServerMacOSBase::_update_displays_arrangement() const {
	origin = Point2i();

	for (int i = 0; i < get_screen_count(); i++) {
		Point2i position = _get_native_screen_position(i);
		if (position.x < origin.x) {
			origin.x = position.x;
		}
		if (position.y > origin.y) {
			origin.y = position.y;
		}
	}
	displays_arrangement_dirty = false;
}

Point2i DisplayServerMacOSBase::_get_screens_origin() const {
	// Returns the native top-left screen coordinate of the smallest rectangle
	// that encompasses all screens. Needed in get_screen_position(),
	// window_get_position, and window_set_position()
	// to convert between macOS native screen coordinates and the ones expected by Godot.

	if (displays_arrangement_dirty) {
		_update_displays_arrangement();
	}

	return origin;
}

Point2i DisplayServerMacOSBase::_get_native_screen_position(int p_screen) const {
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		// Return the top-left corner of the screen, for macOS the y starts at the bottom.
		return Point2i(nsrect.origin.x, nsrect.origin.y + nsrect.size.height) * screen_get_max_scale();
	}

	return Point2i();
}

void DisplayServerMacOSBase::_displays_arrangement_changed(CGDirectDisplayID display_id, CGDisplayChangeSummaryFlags flags, void *user_info) {
	DisplayServerMacOSBase *ds = (DisplayServerMacOSBase *)DisplayServer::get_singleton();
	if (ds) {
		ds->displays_arrangement_dirty = true;
	}
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

int DisplayServerMacOSBase::get_screen_count() const {
	_THREAD_SAFE_METHOD_

	NSArray *screenArray = [NSScreen screens];
	return [screenArray count];
}

int DisplayServerMacOSBase::get_primary_screen() const {
	return 0;
}

int DisplayServerMacOSBase::get_keyboard_focus_screen() const {
	const NSUInteger index = [[NSScreen screens] indexOfObject:[NSScreen mainScreen]];
	return (index == NSNotFound) ? get_primary_screen() : index;
}

Point2i DisplayServerMacOSBase::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Point2i());

	Point2i position = _get_native_screen_position(p_screen) - _get_screens_origin();
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value.
	position.y *= -1;
	return position;
}

Size2i DisplayServerMacOSBase::screen_get_size(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Size2i());

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		// Note: Use frame to get the whole screen size.
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		return Size2i(nsrect.size.width, nsrect.size.height) * screen_get_max_scale();
	}

	return Size2i();
}

int DisplayServerMacOSBase::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 72);

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSDictionary *description = [[screenArray objectAtIndex:p_screen] deviceDescription];

		const NSSize displayPixelSize = [[description objectForKey:NSDeviceSize] sizeValue];
		const CGSize displayPhysicalSize = CGDisplayScreenSize([[description objectForKey:@"NSScreenNumber"] unsignedIntValue]);
		float scale = [[screenArray objectAtIndex:p_screen] backingScaleFactor];

		float den2 = (displayPhysicalSize.width / 25.4f) * (displayPhysicalSize.width / 25.4f) + (displayPhysicalSize.height / 25.4f) * (displayPhysicalSize.height / 25.4f);
		if (den2 > 0.0f) {
			return std::ceil(std::sqrt(displayPixelSize.width * displayPixelSize.width + displayPixelSize.height * displayPixelSize.height) / std::sqrt(den2) * scale);
		}
	}

	return 72;
}

float DisplayServerMacOSBase::screen_get_scale(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, 1.0f);

	if (OS::get_singleton()->is_hidpi_allowed()) {
		NSArray<NSScreen *> *screens = NSScreen.screens;
		NSUInteger index = (NSUInteger)p_screen;
		if (index < screens.count) {
			return std::fmax(1.0f, screens[index].backingScaleFactor);
		}
	}

	return 1.f;
}

float DisplayServerMacOSBase::screen_get_max_scale() const {
	_THREAD_SAFE_METHOD_

	// Note: Do not update max display scale on screen configuration change, existing editor windows can't be rescaled on the fly.
	return display_max_scale;
}

Rect2i DisplayServerMacOSBase::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Rect2i());

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		const float scale = screen_get_max_scale();
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] visibleFrame];

		Point2i position = Point2i(nsrect.origin.x, nsrect.origin.y + nsrect.size.height) * scale - _get_screens_origin();
		position.y *= -1;
		Size2i size = Size2i(nsrect.size.width, nsrect.size.height) * scale;

		return Rect2i(position, size);
	}

	return Rect2i();
}

Color DisplayServerMacOSBase::screen_get_pixel(const Point2i &p_position) const {
	HashSet<CGWindowID> exclude_windows = _get_exclude_windows();

	CFArrayRef on_screen_windows = CGWindowListCreate(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
	CFMutableArrayRef capture_windows = CFArrayCreateMutableCopy(nullptr, 0, on_screen_windows);
	for (long i = CFArrayGetCount(on_screen_windows) - 1; i >= 0; i--) {
		CGWindowID window = (CGWindowID)(uintptr_t)CFArrayGetValueAtIndex(capture_windows, i);
		if (exclude_windows.has(window)) {
			CFArrayRemoveValueAtIndex(capture_windows, i);
		}
	}

	Point2i position = p_position - Vector2i(1, 1);
	position -= screen_get_position(0); // Note: coordinates where the screen origin is in the upper-left corner of the main display and y-axis values increase downward.
	position /= screen_get_max_scale();

	Color color;
	CGImageRef image = CGWindowListCreateImageFromArray(CGRectMake(position.x, position.y, 1, 1), capture_windows, kCGWindowListOptionAll);
	if (image) {
		CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
		if (color_space) {
			uint8_t img_data[4];
			CGContextRef context = CGBitmapContextCreate(img_data, 1, 1, 8, 4, color_space, (uint32_t)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
			if (context) {
				CGContextDrawImage(context, CGRectMake(0, 0, 1, 1), image);
				color = Color(img_data[0] / 255.0f, img_data[1] / 255.0f, img_data[2] / 255.0f, img_data[3] / 255.0f);
				CGContextRelease(context);
			}
			CGColorSpaceRelease(color_space);
		}
		CGImageRelease(image);
	}
	return color;
}

Ref<Image> DisplayServerMacOSBase::screen_get_image(int p_screen) const {
	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, Ref<Image>());

	HashSet<CGWindowID> exclude_windows = _get_exclude_windows();

	CFArrayRef on_screen_windows = CGWindowListCreate(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
	CFMutableArrayRef capture_windows = CFArrayCreateMutableCopy(nullptr, 0, on_screen_windows);
	for (long i = CFArrayGetCount(on_screen_windows) - 1; i >= 0; i--) {
		CGWindowID window = (CGWindowID)(uintptr_t)CFArrayGetValueAtIndex(capture_windows, i);
		if (exclude_windows.has(window)) {
			CFArrayRemoveValueAtIndex(capture_windows, i);
		}
	}

	Point2i position = screen_get_position(p_screen);
	position -= screen_get_position(0); // Note: coordinates where the screen origin is in the upper-left corner of the main display and y-axis values increase downward.
	position /= screen_get_max_scale();

	Size2i size = screen_get_size(p_screen);
	size /= screen_get_max_scale();

	Ref<Image> img;
	CGImageRef image = CGWindowListCreateImageFromArray(CGRectMake(position.x, position.y, size.width, size.height), capture_windows, kCGWindowListOptionAll);
	if (image) {
		CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
		if (color_space) {
			NSUInteger width = CGImageGetWidth(image);
			NSUInteger height = CGImageGetHeight(image);

			Vector<uint8_t> img_data;
			img_data.resize(height * width * 4);
			CGContextRef context = CGBitmapContextCreate(img_data.ptrw(), width, height, 8, 4 * width, color_space, (uint32_t)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
			if (context) {
				CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
				img = Image::create_from_data(width, height, false, Image::FORMAT_RGBA8, img_data);
				CGContextRelease(context);
				img->resize(screen_get_size(p_screen).x, screen_get_size(p_screen).y, Image::INTERPOLATE_NEAREST);
			}
			CGColorSpaceRelease(color_space);
		}
		CGImageRelease(image);
	}
	return img;
}

Ref<Image> DisplayServerMacOSBase::screen_get_image_rect(const Rect2i &p_rect) const {
	HashSet<CGWindowID> exclude_windows = _get_exclude_windows();

	CFArrayRef on_screen_windows = CGWindowListCreate(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
	CFMutableArrayRef capture_windows = CFArrayCreateMutableCopy(nullptr, 0, on_screen_windows);
	for (long i = CFArrayGetCount(on_screen_windows) - 1; i >= 0; i--) {
		CGWindowID window = (CGWindowID)(uintptr_t)CFArrayGetValueAtIndex(capture_windows, i);
		if (exclude_windows.has(window)) {
			CFArrayRemoveValueAtIndex(capture_windows, i);
		}
	}

	Point2i position = p_rect.position;
	position -= screen_get_position(0); // Note: coordinates where the screen origin is in the upper-left corner of the main display and y-axis values increase downward.
	position /= screen_get_max_scale();

	Size2i size = p_rect.size;
	size /= screen_get_max_scale();

	Ref<Image> img;
	CGImageRef image = CGWindowListCreateImageFromArray(CGRectMake(position.x, position.y, size.width, size.height), capture_windows, kCGWindowListOptionAll);
	if (image) {
		CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
		if (color_space) {
			NSUInteger width = CGImageGetWidth(image);
			NSUInteger height = CGImageGetHeight(image);

			Vector<uint8_t> img_data;
			img_data.resize_initialized(height * width * 4);
			CGContextRef context = CGBitmapContextCreate(img_data.ptrw(), width, height, 8, 4 * width, color_space, (uint32_t)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
			if (context) {
				CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
				img = Image::create_from_data(width, height, false, Image::FORMAT_RGBA8, img_data);
				CGContextRelease(context);
				img->resize(p_rect.size.x, p_rect.size.y, Image::INTERPOLATE_NEAREST);
			}
			CGColorSpaceRelease(color_space);
		}
		CGImageRelease(image);
	}
	return img;
}

float DisplayServerMacOSBase::screen_get_refresh_rate(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	int screen_count = get_screen_count();
	ERR_FAIL_INDEX_V(p_screen, screen_count, SCREEN_REFRESH_RATE_FALLBACK);

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

Point2i DisplayServerMacOSBase::mouse_get_position() const {
	_THREAD_SAFE_METHOD_

	const NSPoint mouse_pos = [NSEvent mouseLocation];
	const float scale = screen_get_max_scale();

	for (NSScreen *screen in [NSScreen screens]) {
		NSRect frame = [screen frame];
		if (NSMouseInRect(mouse_pos, frame, NO)) {
			Vector2i pos = Vector2i((int)mouse_pos.x, (int)mouse_pos.y);
			pos *= scale;
			pos -= _get_screens_origin();
			pos.y *= -1;
			return pos;
		}
	}
	return Vector2i();
}

BitField<MouseButtonMask> DisplayServerMacOSBase::mouse_get_button_state() const {
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

void DisplayServerMacOSBase::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
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

DisplayServerMacOSBase::DisplayServerMacOSBase() {
	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		initialize_tts();
	}

	// Register to be notified on keyboard layout changes.
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
			nullptr, _keyboard_layout_changed,
			kTISNotifySelectedKeyboardInputSourceChanged, nullptr,
			CFNotificationSuspensionBehaviorDeliverImmediately);

	int screen_count = get_screen_count();
	for (int i = 0; i < screen_count; i++) {
		display_max_scale = std::fmax(display_max_scale, screen_get_scale(i));
	}

	// Register to be notified on displays arrangement changes.
	CGDisplayRegisterReconfigurationCallback(_displays_arrangement_changed, nullptr);
}

DisplayServerMacOSBase::~DisplayServerMacOSBase() {
	if (screen_keep_on_assertion) {
		IOPMAssertionRelease(screen_keep_on_assertion);
		screen_keep_on_assertion = kIOPMNullAssertionID;
	}

	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), nullptr, kTISNotifySelectedKeyboardInputSourceChanged, nullptr);

	CGDisplayRemoveReconfigurationCallback(_displays_arrangement_changed, nullptr);
}
