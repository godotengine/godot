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
#import "godot_menu_delegate.h"
#import "godot_menu_item.h"
#import "godot_status_item.h"
#import "key_mapping_macos.h"
#import "native_menu_macos.h"
#import "tts_macos.h"

#include "core/config/project_settings.h"
#include "drivers/png/png_driver_common.h"
#include "scene/resources/image_texture.h"

#import <Carbon/Carbon.h>

NSImage *DisplayServerMacOSBase::_convert_to_nsimg(Ref<Image> &p_image) const {
	p_image->convert(Image::FORMAT_RGBA8);
	NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc]
			initWithBitmapDataPlanes:nullptr
						  pixelsWide:p_image->get_width()
						  pixelsHigh:p_image->get_height()
					   bitsPerSample:8
					 samplesPerPixel:4
							hasAlpha:YES
							isPlanar:NO
					  colorSpaceName:NSDeviceRGBColorSpace
						 bytesPerRow:int(p_image->get_width()) * 4
						bitsPerPixel:32];
	ERR_FAIL_NULL_V(imgrep, nil);
	uint8_t *pixels = [imgrep bitmapData];

	int len = p_image->get_width() * p_image->get_height();
	const uint8_t *r = p_image->get_data().ptr();

	/* Premultiply the alpha channel */
	for (int i = 0; i < len; i++) {
		uint8_t alpha = r[i * 4 + 3];
		pixels[i * 4 + 0] = (uint8_t)(((uint16_t)r[i * 4 + 0] * alpha) / 255);
		pixels[i * 4 + 1] = (uint8_t)(((uint16_t)r[i * 4 + 1] * alpha) / 255);
		pixels[i * 4 + 2] = (uint8_t)(((uint16_t)r[i * 4 + 2] * alpha) / 255);
		pixels[i * 4 + 3] = alpha;
	}

	NSImage *nsimg = [[NSImage alloc] initWithSize:NSMakeSize(p_image->get_width(), p_image->get_height())];
	ERR_FAIL_NULL_V(nsimg, nil);
	[nsimg addRepresentation:imgrep];
	return nsimg;
}

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

void DisplayServerMacOSBase::beep() const {
	NSBeep();
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

bool DisplayServerMacOSBase::get_swap_cancel_ok() {
	return false;
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

DisplayServer::IndicatorID DisplayServerMacOSBase::create_status_indicator(const Ref<Texture2D> &p_icon, const String &p_tooltip, const Callable &p_callback) {
	NSImage *nsimg = nullptr;
	if (p_icon.is_valid() && p_icon->get_width() > 0 && p_icon->get_height() > 0 && p_icon->get_image().is_valid()) {
		Ref<Image> img = p_icon->get_image();
		img = img->duplicate();
		if (img->is_compressed()) {
			img->decompress();
		}
		nsimg = _convert_to_nsimg(img);
	}

	IndicatorData idat;

	NSStatusItem *item = [[NSStatusBar systemStatusBar] statusItemWithLength:NSSquareStatusItemLength];
	idat.item = item;
	idat.delegate = [[GodotStatusItemDelegate alloc] init];
	[idat.delegate setCallback:p_callback];

	item.button.image = nsimg;
	item.button.imagePosition = NSImageOnly;
	item.button.imageScaling = NSImageScaleProportionallyUpOrDown;
	item.button.target = idat.delegate;
	item.button.action = @selector(click:);
	[item.button sendActionOn:(NSEventMaskLeftMouseDown | NSEventMaskRightMouseDown | NSEventMaskOtherMouseDown)];
	item.button.toolTip = [NSString stringWithUTF8String:p_tooltip.utf8().get_data()];

	IndicatorID iid = indicator_id_counter++;
	indicators[iid] = idat;

	return iid;
}

void DisplayServerMacOSBase::status_indicator_set_icon(IndicatorID p_id, const Ref<Texture2D> &p_icon) {
	ERR_FAIL_COND(!indicators.has(p_id));

	NSImage *nsimg = nullptr;
	if (p_icon.is_valid() && p_icon->get_width() > 0 && p_icon->get_height() > 0 && p_icon->get_image().is_valid()) {
		Ref<Image> img = p_icon->get_image();
		img = img->duplicate();
		if (img->is_compressed()) {
			img->decompress();
		}
		nsimg = _convert_to_nsimg(img);
	}

	NSStatusItem *item = indicators[p_id].item;
	item.button.image = nsimg;
}

void DisplayServerMacOSBase::status_indicator_set_tooltip(IndicatorID p_id, const String &p_tooltip) {
	ERR_FAIL_COND(!indicators.has(p_id));

	NSStatusItem *item = indicators[p_id].item;
	item.button.toolTip = [NSString stringWithUTF8String:p_tooltip.utf8().get_data()];
}

void DisplayServerMacOSBase::status_indicator_set_menu(IndicatorID p_id, const RID &p_menu_rid) {
	ERR_FAIL_COND(!indicators.has(p_id));

	NSStatusItem *item = indicators[p_id].item;
	if (p_menu_rid.is_valid() && native_menu->has_menu(p_menu_rid)) {
		NSMenu *menu = native_menu->get_native_menu_handle(p_menu_rid);
		item.menu = menu;
	} else {
		item.menu = nullptr;
	}
}

void DisplayServerMacOSBase::status_indicator_set_callback(IndicatorID p_id, const Callable &p_callback) {
	ERR_FAIL_COND(!indicators.has(p_id));

	[indicators[p_id].delegate setCallback:p_callback];
}

Rect2 DisplayServerMacOSBase::status_indicator_get_rect(IndicatorID p_id) const {
	ERR_FAIL_COND_V(!indicators.has(p_id), Rect2());

	NSStatusItem *item = indicators[p_id].item;
	NSView *v = item.button;
	const NSRect contentRect = [v frame];
	const NSRect nsrect = [v.window convertRectToScreen:contentRect];
	Rect2 rect;

	// Return the position of the top-left corner, for macOS the y starts at the bottom.
	const float scale = screen_get_max_scale();
	rect.size.x = nsrect.size.width;
	rect.size.y = nsrect.size.height;
	rect.size *= scale;
	rect.position.x = nsrect.origin.x;
	rect.position.y = (nsrect.origin.y + nsrect.size.height);
	rect.position *= scale;
	rect.position -= _get_screens_origin();
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value.
	rect.position.y *= -1;
	return rect;
}

void DisplayServerMacOSBase::delete_status_indicator(IndicatorID p_id) {
	ERR_FAIL_COND(!indicators.has(p_id));

	[[NSStatusBar systemStatusBar] removeStatusItem:indicators[p_id].item];
	indicators.erase(p_id);
}

DisplayServerMacOSBase::DisplayServerMacOSBase() {
	native_menu = memnew(NativeMenuMacOS);

	// Register to be notified on displays arrangement changes.
	CGDisplayRegisterReconfigurationCallback(_displays_arrangement_changed, nullptr);

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
}

DisplayServerMacOSBase::~DisplayServerMacOSBase() {
	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), nullptr, kTISNotifySelectedKeyboardInputSourceChanged, nullptr);

	// Destroy all status indicators.
	for (HashMap<IndicatorID, IndicatorData>::Iterator E = indicators.begin(); E; ++E) {
		[[NSStatusBar systemStatusBar] removeStatusItem:E->value.item];
	}

	// Destroy native menu.
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}

	CGDisplayRemoveReconfigurationCallback(_displays_arrangement_changed, nullptr);
}
