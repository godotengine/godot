/**************************************************************************/
/*  display_server_macos.mm                                               */
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

#include "display_server_macos.h"

#include "godot_button_view.h"
#include "godot_content_view.h"
#include "godot_menu_delegate.h"
#include "godot_menu_item.h"
#include "godot_window.h"
#include "godot_window_delegate.h"
#include "key_mapping_macos.h"
#include "os_macos.h"
#include "tts_macos.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "drivers/png/png_driver_common.h"
#include "main/main.h"
#include "scene/resources/atlas_texture.h"
#include "scene/resources/image_texture.h"

#if defined(GLES3_ENABLED)
#include "drivers/gles3/rasterizer_gles3.h"
#endif

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#endif

#import <Carbon/Carbon.h>
#import <Cocoa/Cocoa.h>
#import <IOKit/IOCFPlugIn.h>
#import <IOKit/IOKitLib.h>
#import <IOKit/hid/IOHIDKeys.h>
#import <IOKit/hid/IOHIDLib.h>

const NSMenu *DisplayServerMacOS::_get_menu_root(const String &p_menu_root) const {
	const NSMenu *menu = nullptr;
	if (p_menu_root == "" || p_menu_root.to_lower() == "_main") {
		// Main menu.
		menu = [NSApp mainMenu];
	} else if (p_menu_root.to_lower() == "_dock") {
		// macOS dock menu.
		menu = dock_menu;
	} else {
		// Submenu.
		if (submenu.has(p_menu_root)) {
			menu = submenu[p_menu_root].menu;
		}
	}
	if (menu == apple_menu) {
		// Do not allow to change Apple menu.
		return nullptr;
	}
	return menu;
}

NSMenu *DisplayServerMacOS::_get_menu_root(const String &p_menu_root) {
	NSMenu *menu = nullptr;
	if (p_menu_root == "" || p_menu_root.to_lower() == "_main") {
		// Main menu.
		menu = [NSApp mainMenu];
	} else if (p_menu_root.to_lower() == "_dock") {
		// macOS dock menu.
		menu = dock_menu;
	} else {
		// Submenu.
		if (!submenu.has(p_menu_root)) {
			NSMenu *n_menu = [[NSMenu alloc] initWithTitle:[NSString stringWithUTF8String:p_menu_root.utf8().get_data()]];
			[n_menu setAutoenablesItems:NO];
			[n_menu setDelegate:menu_delegate];
			submenu[p_menu_root].menu = n_menu;
			submenu_inv[n_menu] = p_menu_root;
		}
		menu = submenu[p_menu_root].menu;
	}
	if (menu == apple_menu) {
		// Do not allow to change Apple menu.
		return nullptr;
	}
	return menu;
}

DisplayServerMacOS::WindowID DisplayServerMacOS::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, const Rect2i &p_rect) {
	WindowID id;
	const float scale = screen_get_max_scale();
	{
		WindowData wd;

		wd.window_delegate = [[GodotWindowDelegate alloc] init];
		ERR_FAIL_NULL_V_MSG(wd.window_delegate, INVALID_WINDOW_ID, "Can't create a window delegate");
		[wd.window_delegate setWindowID:window_id_counter];

		int rq_screen = get_screen_from_rect(p_rect);
		if (rq_screen < 0) {
			rq_screen = get_primary_screen(); // Requested window rect is outside any screen bounds.
		}

		Rect2i srect = screen_get_usable_rect(rq_screen);
		Point2i wpos = p_rect.position;
		if (srect != Rect2i()) {
			wpos.x = CLAMP(wpos.x, srect.position.x, srect.position.x + srect.size.width - p_rect.size.width / 3);
			wpos.y = CLAMP(wpos.y, srect.position.y, srect.position.y + srect.size.height - p_rect.size.height / 3);
		}
		// macOS native y-coordinate relative to _get_screens_origin() is negative,
		// Godot passes a positive value.
		wpos.y *= -1;
		wpos += _get_screens_origin();
		wpos /= scale;

		// initWithContentRect uses bottom-left corner of the window’s frame as origin.
		wd.window_object = [[GodotWindow alloc]
				initWithContentRect:NSMakeRect(100, 100, MAX(1, p_rect.size.width / scale), MAX(1, p_rect.size.height / scale))
						  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
							backing:NSBackingStoreBuffered
							  defer:NO];
		ERR_FAIL_NULL_V_MSG(wd.window_object, INVALID_WINDOW_ID, "Can't create a window");
		[wd.window_object setWindowID:window_id_counter];

		wd.window_view = [[GodotContentView alloc] init];
		ERR_FAIL_NULL_V_MSG(wd.window_view, INVALID_WINDOW_ID, "Can't create a window view");
		[wd.window_view setWindowID:window_id_counter];
		[wd.window_view setWantsLayer:TRUE];

		[wd.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
		[wd.window_object setContentView:wd.window_view];
		[wd.window_object setDelegate:wd.window_delegate];
		[wd.window_object setAcceptsMouseMovedEvents:YES];
		[wd.window_object setRestorable:NO];
		[wd.window_object setColorSpace:[NSColorSpace sRGBColorSpace]];

		if ([wd.window_object respondsToSelector:@selector(setTabbingMode:)]) {
			[wd.window_object setTabbingMode:NSWindowTabbingModeDisallowed];
		}

		CALayer *layer = [(NSView *)wd.window_view layer];
		if (layer) {
			layer.contentsScale = scale;
		}

		NSColor *bg_color = [NSColor windowBackgroundColor];
		Color _bg_color;
		if (_get_window_early_clear_override(_bg_color)) {
			bg_color = [NSColor colorWithCalibratedRed:_bg_color.r green:_bg_color.g blue:_bg_color.b alpha:1.f];
		}

		[wd.window_object setBackgroundColor:bg_color];
		if (layer) {
			[layer setBackgroundColor:bg_color.CGColor];
		}

#if defined(VULKAN_ENABLED)
		if (context_vulkan) {
			Error err = context_vulkan->window_create(window_id_counter, p_vsync_mode, wd.window_view, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create a Vulkan context");
		}
#endif
#if defined(GLES3_ENABLED)
		if (gl_manager_legacy) {
			Error err = gl_manager_legacy->window_create(window_id_counter, wd.window_view, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create an OpenGL context.");
		}
		if (gl_manager_angle) {
			CALayer *layer = [(NSView *)wd.window_view layer];
			Error err = gl_manager_angle->window_create(window_id_counter, nullptr, (__bridge void *)layer, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create an OpenGL context.");
		}
		window_set_vsync_mode(p_vsync_mode, window_id_counter);
#endif
		[wd.window_view updateLayerDelegate];

		const NSRect contentRect = [wd.window_view frame];
		const NSRect windowRect = [wd.window_object frame];
		const NSRect nsrect = [wd.window_object convertRectToScreen:contentRect];
		Point2i offset;
		offset.x = (nsrect.origin.x - windowRect.origin.x);
		offset.y = (nsrect.origin.y + nsrect.size.height);
		offset.y -= (windowRect.origin.y + windowRect.size.height);
		[wd.window_object setFrameTopLeftPoint:NSMakePoint(wpos.x - offset.x, wpos.y - offset.y)];

		id = window_id_counter++;
		windows[id] = wd;
	}

	WindowData &wd = windows[id];
	window_set_mode(p_mode, id);

	const NSRect contentRect = [wd.window_view frame];
	wd.size.width = contentRect.size.width * scale;
	wd.size.height = contentRect.size.height * scale;

	CALayer *layer = [(NSView *)wd.window_view layer];
	if (layer) {
		layer.contentsScale = scale;
	}

#if defined(GLES3_ENABLED)
	if (gl_manager_legacy) {
		gl_manager_legacy->window_resize(id, wd.size.width, wd.size.height);
	}
	if (gl_manager_angle) {
		gl_manager_angle->window_resize(id, wd.size.width, wd.size.height);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(id, wd.size.width, wd.size.height);
	}
#endif

	return id;
}

void DisplayServerMacOS::_update_window_style(WindowData p_wd) {
	bool borderless_full = false;

	if (p_wd.borderless) {
		NSRect frameRect = [p_wd.window_object frame];
		NSRect screenRect = [[p_wd.window_object screen] frame];

		// Check if our window covers up the screen.
		if (frameRect.origin.x <= screenRect.origin.x && frameRect.origin.y <= frameRect.origin.y &&
				frameRect.size.width >= screenRect.size.width && frameRect.size.height >= screenRect.size.height) {
			borderless_full = true;
		}
	}

	if (borderless_full) {
		// If the window covers up the screen set the level to above the main menu and hide on deactivate.
		[(NSWindow *)p_wd.window_object setLevel:NSMainMenuWindowLevel + 1];
		[(NSWindow *)p_wd.window_object setHidesOnDeactivate:YES];
	} else {
		// Reset these when our window is not a borderless window that covers up the screen.
		if (p_wd.on_top && !p_wd.fullscreen) {
			[(NSWindow *)p_wd.window_object setLevel:NSFloatingWindowLevel];
		} else {
			[(NSWindow *)p_wd.window_object setLevel:NSNormalWindowLevel];
		}
		[(NSWindow *)p_wd.window_object setHidesOnDeactivate:NO];
	}
}

void DisplayServerMacOS::set_window_per_pixel_transparency_enabled(bool p_enabled, WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (!OS::get_singleton()->is_layered_allowed()) {
		return;
	}
	if (p_enabled) {
		[wd.window_object setBackgroundColor:[NSColor clearColor]];
		[wd.window_object setOpaque:NO];
		[wd.window_object setHasShadow:NO];
		CALayer *layer = [(NSView *)wd.window_view layer];
		if (layer) {
			[layer setBackgroundColor:[NSColor clearColor].CGColor];
			[layer setOpaque:NO];
		}
#if defined(GLES3_ENABLED)
		if (gl_manager_legacy) {
			gl_manager_legacy->window_set_per_pixel_transparency_enabled(p_window, true);
		}
#endif
	} else {
		NSColor *bg_color = [NSColor windowBackgroundColor];
		Color _bg_color;
		if (_get_window_early_clear_override(_bg_color)) {
			bg_color = [NSColor colorWithCalibratedRed:_bg_color.r green:_bg_color.g blue:_bg_color.b alpha:1.f];
		}
		[wd.window_object setBackgroundColor:bg_color];
		[wd.window_object setOpaque:YES];
		[wd.window_object setHasShadow:YES];
		CALayer *layer = [(NSView *)wd.window_view layer];
		if (layer) {
			[layer setBackgroundColor:bg_color.CGColor];
			[layer setOpaque:YES];
		}
#if defined(GLES3_ENABLED)
		if (gl_manager_legacy) {
			gl_manager_legacy->window_set_per_pixel_transparency_enabled(p_window, false);
		}
#endif
	}
}

void DisplayServerMacOS::_update_displays_arrangement() {
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

Point2i DisplayServerMacOS::_get_screens_origin() const {
	// Returns the native top-left screen coordinate of the smallest rectangle
	// that encompasses all screens. Needed in get_screen_position(),
	// window_get_position, and window_set_position()
	// to convert between macOS native screen coordinates and the ones expected by Godot.

	if (displays_arrangement_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_displays_arrangement();
	}

	return origin;
}

Point2i DisplayServerMacOS::_get_native_screen_position(int p_screen) const {
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		// Return the top-left corner of the screen, for macOS the y starts at the bottom.
		return Point2i(nsrect.origin.x, nsrect.origin.y + nsrect.size.height) * screen_get_max_scale();
	}

	return Point2i();
}

void DisplayServerMacOS::_displays_arrangement_changed(CGDirectDisplayID display_id, CGDisplayChangeSummaryFlags flags, void *user_info) {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		ds->displays_arrangement_dirty = true;
	}
}

DisplayServer::WindowID DisplayServerMacOS::_get_focused_window_or_popup() const {
	const List<WindowID>::Element *E = popup_list.back();
	if (E) {
		return E->get();
	}

	return last_focused_window;
}

void DisplayServerMacOS::mouse_enter_window(WindowID p_window) {
	if (window_mouseover_id != p_window) {
		if (window_mouseover_id != INVALID_WINDOW_ID) {
			send_window_event(windows[window_mouseover_id], WINDOW_EVENT_MOUSE_EXIT);
		}
		window_mouseover_id = p_window;
		if (p_window != INVALID_WINDOW_ID) {
			send_window_event(windows[p_window], WINDOW_EVENT_MOUSE_ENTER);
		}
	}
}

void DisplayServerMacOS::mouse_exit_window(WindowID p_window) {
	if (window_mouseover_id == p_window && p_window != INVALID_WINDOW_ID) {
		send_window_event(windows[p_window], WINDOW_EVENT_MOUSE_EXIT);
	}
	window_mouseover_id = INVALID_WINDOW_ID;
}

void DisplayServerMacOS::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerMacOS *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerMacOS::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	_THREAD_SAFE_METHOD_
	if (!in_dispatch_input_event) {
		in_dispatch_input_event = true;

		{
			List<WindowID>::Element *E = popup_list.back();
			if (E && Object::cast_to<InputEventKey>(*p_event)) {
				// Redirect keyboard input to active popup.
				if (windows.has(E->get())) {
					Callable callable = windows[E->get()].input_event_callback;
					if (callable.is_valid()) {
						callable.call(p_event);
					}
				}
				in_dispatch_input_event = false;
				return;
			}
		}

		Ref<InputEventFromWindow> event_from_window = p_event;
		if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
			// Send to a window.
			if (windows.has(event_from_window->get_window_id())) {
				Callable callable = windows[event_from_window->get_window_id()].input_event_callback;
				if (callable.is_valid()) {
					callable.call(p_event);
				}
			}
		} else {
			// Send to all windows.
			for (KeyValue<WindowID, WindowData> &E : windows) {
				Callable callable = E.value.input_event_callback;
				if (callable.is_valid()) {
					callable.call(p_event);
				}
			}
		}
		in_dispatch_input_event = false;
	}
}

void DisplayServerMacOS::_push_input(const Ref<InputEvent> &p_event) {
	Ref<InputEvent> ev = p_event;
	Input::get_singleton()->parse_input_event(ev);
}

void DisplayServerMacOS::_process_key_events() {
	Ref<InputEventKey> k;
	for (int i = 0; i < key_event_pos; i++) {
		const KeyEvent &ke = key_event_buffer[i];
		if (ke.raw) {
			// Non IME input - no composite characters, pass events as is.
			k.instantiate();

			k->set_window_id(ke.window_id);
			get_key_modifier_state(ke.macos_state, k);
			k->set_pressed(ke.pressed);
			k->set_echo(ke.echo);
			k->set_keycode(ke.keycode);
			k->set_physical_keycode(ke.physical_keycode);
			k->set_key_label(ke.key_label);
			k->set_unicode(ke.unicode);

			_push_input(k);
		} else {
			// IME input.
			if ((i == 0 && ke.keycode == Key::NONE) || (i > 0 && key_event_buffer[i - 1].keycode == Key::NONE)) {
				k.instantiate();

				k->set_window_id(ke.window_id);
				get_key_modifier_state(ke.macos_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_keycode(Key::NONE);
				k->set_physical_keycode(Key::NONE);
				k->set_key_label(Key::NONE);
				k->set_unicode(ke.unicode);

				_push_input(k);
			}
			if (ke.keycode != Key::NONE) {
				k.instantiate();

				k->set_window_id(ke.window_id);
				get_key_modifier_state(ke.macos_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_keycode(ke.keycode);
				k->set_physical_keycode(ke.physical_keycode);
				k->set_key_label(ke.key_label);

				if (i + 1 < key_event_pos && key_event_buffer[i + 1].keycode == Key::NONE) {
					k->set_unicode(key_event_buffer[i + 1].unicode);
				}

				_push_input(k);
			}
		}
	}

	key_event_pos = 0;
}

void DisplayServerMacOS::_update_keyboard_layouts() {
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
		ly.name.parse_utf8([name UTF8String]);

		NSArray *langs = (__bridge NSArray *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyInputSourceLanguages);
		ly.code.parse_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
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
		ly.name.parse_utf8([name UTF8String]);

		NSArray *langs = (__bridge NSArray *)TISGetInputSourceProperty((__bridge TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyInputSourceLanguages);
		ly.code.parse_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
		kbd_layouts.push_back(ly);

		if ([name isEqualToString:cur_name]) {
			current_layout = kbd_layouts.size() - 1;
		}
	}

	keyboard_layout_dirty = false;
}

void DisplayServerMacOS::_keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info) {
	DisplayServerMacOS *ds = (DisplayServerMacOS *)DisplayServer::get_singleton();
	if (ds) {
		ds->keyboard_layout_dirty = true;
	}
}

NSImage *DisplayServerMacOS::_convert_to_nsimg(Ref<Image> &p_image) const {
	p_image->convert(Image::FORMAT_RGBA8);
	NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc]
			initWithBitmapDataPlanes:NULL
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

NSCursor *DisplayServerMacOS::_cursor_from_selector(SEL p_selector, SEL p_fallback) {
	if ([NSCursor respondsToSelector:p_selector]) {
		id object = [NSCursor performSelector:p_selector];
		if ([object isKindOfClass:[NSCursor class]]) {
			return object;
		}
	}
	if (p_fallback) {
		// Fallback should be a reasonable default, no need to check.
		return [NSCursor performSelector:p_fallback];
	}
	return [NSCursor arrowCursor];
}

NSMenu *DisplayServerMacOS::get_dock_menu() const {
	return dock_menu;
}

void DisplayServerMacOS::menu_open(NSMenu *p_menu) {
	if (submenu_inv.has(p_menu)) {
		MenuData &md = submenu[submenu_inv[p_menu]];
		md.is_open = true;
		if (md.open.is_valid()) {
			Variant ret;
			Callable::CallError ce;

			md.open.callp(nullptr, 0, ret, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				ERR_PRINT(vformat("Failed to execute menu open callback: %s.", Variant::get_callable_error_text(md.open, nullptr, 0, ce)));
			}
		}
	}
}

void DisplayServerMacOS::menu_close(NSMenu *p_menu) {
	if (submenu_inv.has(p_menu)) {
		MenuData &md = submenu[submenu_inv[p_menu]];
		md.is_open = false;
		if (md.close.is_valid()) {
			Variant ret;
			Callable::CallError ce;

			md.close.callp(nullptr, 0, ret, ce);
			if (ce.error != Callable::CallError::CALL_OK) {
				ERR_PRINT(vformat("Failed to execute menu close callback: %s.", Variant::get_callable_error_text(md.close, nullptr, 0, ce)));
			}
		}
	}
}

void DisplayServerMacOS::menu_callback(id p_sender) {
	if (![p_sender representedObject]) {
		return;
	}

	GodotMenuItem *value = [p_sender representedObject];

	if (value) {
		if (value->max_states > 0) {
			value->state++;
			if (value->state >= value->max_states) {
				value->state = 0;
			}
		}

		if (value->checkable_type == CHECKABLE_TYPE_CHECK_BOX) {
			if ([p_sender state] == NSControlStateValueOff) {
				[p_sender setState:NSControlStateValueOn];
			} else {
				[p_sender setState:NSControlStateValueOff];
			}
		}

		if (value->callback.is_valid()) {
			MenuCall mc;
			mc.tag = value->meta;
			mc.callback = value->callback;
			deferred_menu_calls.push_back(mc);
			// Do not run callback from here! If it is opening a new window or calling process_events, it will corrupt OS event queue and crash.
		}
	}
}

bool DisplayServerMacOS::has_window(WindowID p_window) const {
	return windows.has(p_window);
}

DisplayServerMacOS::WindowData &DisplayServerMacOS::get_window(WindowID p_window) {
	return windows[p_window];
}

void DisplayServerMacOS::send_event(NSEvent *p_event) {
	// Special case handling of command-period, which is traditionally a special
	// shortcut in macOS and doesn't arrive at our regular keyDown handler.
	if ([p_event type] == NSEventTypeKeyDown) {
		if ((([p_event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask) == NSEventModifierFlagCommand) && [p_event keyCode] == 0x2f) {
			Ref<InputEventKey> k;
			k.instantiate();

			get_key_modifier_state([p_event modifierFlags], k);
			k->set_window_id(DisplayServerMacOS::INVALID_WINDOW_ID);
			k->set_pressed(true);
			k->set_keycode(Key::PERIOD);
			k->set_physical_keycode(Key::PERIOD);
			k->set_key_label(Key::PERIOD);
			k->set_echo([p_event isARepeat]);

			Input::get_singleton()->parse_input_event(k);
		}
	}
}

void DisplayServerMacOS::send_window_event(const WindowData &wd, WindowEvent p_event) {
	_THREAD_SAFE_METHOD_

	if (wd.event_callback.is_valid()) {
		Variant event = int(p_event);
		wd.event_callback.call(event);
	}
}

void DisplayServerMacOS::release_pressed_events() {
	_THREAD_SAFE_METHOD_
	if (Input::get_singleton()) {
		Input::get_singleton()->release_pressed_events();
	}
}

void DisplayServerMacOS::get_key_modifier_state(unsigned int p_macos_state, Ref<InputEventWithModifiers> r_state) const {
	r_state->set_shift_pressed((p_macos_state & NSEventModifierFlagShift));
	r_state->set_ctrl_pressed((p_macos_state & NSEventModifierFlagControl));
	r_state->set_alt_pressed((p_macos_state & NSEventModifierFlagOption));
	r_state->set_meta_pressed((p_macos_state & NSEventModifierFlagCommand));
}

void DisplayServerMacOS::update_mouse_pos(DisplayServerMacOS::WindowData &p_wd, NSPoint p_location_in_window) {
	const NSRect content_rect = [p_wd.window_view frame];
	const float scale = screen_get_max_scale();
	p_wd.mouse_pos.x = p_location_in_window.x * scale;
	p_wd.mouse_pos.y = (content_rect.size.height - p_location_in_window.y) * scale;
	Input::get_singleton()->set_mouse_position(p_wd.mouse_pos);
}

void DisplayServerMacOS::pop_last_key_event() {
	// Does not pop last key event when it is an IME key event.
	if (key_event_pos > 0 && key_event_buffer[key_event_pos - 1].raw) {
		key_event_pos--;
	}
}

void DisplayServerMacOS::push_to_key_event_buffer(const DisplayServerMacOS::KeyEvent &p_event) {
	if (key_event_pos >= key_event_buffer.size()) {
		key_event_buffer.resize(1 + key_event_pos);
	}
	key_event_buffer.write[key_event_pos++] = p_event;
}

void DisplayServerMacOS::update_im_text(const Point2i &p_selection, const String &p_text) {
	im_selection = p_selection;
	im_text = p_text;

	OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
}

void DisplayServerMacOS::set_last_focused_window(WindowID p_window) {
	last_focused_window = p_window;
}

void DisplayServerMacOS::set_is_resizing(bool p_is_resizing) {
	is_resizing = p_is_resizing;
}

bool DisplayServerMacOS::get_is_resizing() const {
	return is_resizing;
}

void DisplayServerMacOS::window_destroy(WindowID p_window) {
#if defined(GLES3_ENABLED)
	if (gl_manager_legacy) {
		gl_manager_legacy->window_destroy(p_window);
	}
#endif
#ifdef VULKAN_ENABLED
	if (context_vulkan) {
		context_vulkan->window_destroy(p_window);
	}
#endif
	windows.erase(p_window);
	update_presentation_mode();
}

void DisplayServerMacOS::window_resize(WindowID p_window, int p_width, int p_height) {
#if defined(GLES3_ENABLED)
	if (gl_manager_legacy) {
		gl_manager_legacy->window_resize(p_window, p_width, p_height);
	}
	if (gl_manager_angle) {
		gl_manager_angle->window_resize(p_window, p_width, p_height);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(p_window, p_width, p_height);
	}
#endif
}

bool DisplayServerMacOS::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_GLOBAL_MENU:
		case FEATURE_SUBWINDOWS:
		//case FEATURE_TOUCHSCREEN:
		case FEATURE_MOUSE:
		case FEATURE_MOUSE_WARP:
		case FEATURE_CLIPBOARD:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_CUSTOM_CURSOR_SHAPE:
		case FEATURE_NATIVE_DIALOG:
		case FEATURE_IME:
		case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_HIDPI:
		case FEATURE_ICON:
		case FEATURE_NATIVE_ICON:
		//case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_SWAP_BUFFERS:
		case FEATURE_TEXT_TO_SPEECH:
		case FEATURE_EXTEND_TO_TITLE:
		case FEATURE_SCREEN_CAPTURE:
			return true;
		default: {
		}
	}
	return false;
}

String DisplayServerMacOS::get_name() const {
	return "macOS";
}

bool DisplayServerMacOS::_has_help_menu() const {
	if ([NSApp helpMenu]) {
		return true;
	} else {
		NSMenu *menu = [NSApp mainMenu];
		const NSMenuItem *menu_item = [menu itemAtIndex:[menu numberOfItems] - 1];
		if (menu_item) {
			String menu_name = String::utf8([[menu_item title] UTF8String]);
			if (menu_name == "Help" || menu_name == RTR("Help")) {
				return true;
			}
		}
		return false;
	}
}

bool DisplayServerMacOS::_is_menu_opened(NSMenu *p_menu) const {
	if (submenu_inv.has(p_menu)) {
		const MenuData &md = submenu[submenu_inv[p_menu]];
		if (md.is_open) {
			return true;
		}
	}
	for (NSInteger i = (p_menu == [NSApp mainMenu]) ? 1 : 0; i < [p_menu numberOfItems]; i++) {
		const NSMenuItem *menu_item = [p_menu itemAtIndex:i];
		if ([menu_item submenu]) {
			if (_is_menu_opened([menu_item submenu])) {
				return true;
			}
		}
	}
	return false;
}

NSMenuItem *DisplayServerMacOS::_menu_add_item(const String &p_menu_root, const String &p_label, Key p_accel, int p_index, int *r_out) {
	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		String keycode = KeyMappingMacOS::keycode_get_native_string(p_accel & KeyModifierMask::CODE_MASK);
		NSMenuItem *menu_item;
		int item_count = ((menu == [NSApp mainMenu]) && _has_help_menu()) ? [menu numberOfItems] - 1 : [menu numberOfItems];
		if ((menu == [NSApp mainMenu]) && (p_label == "Help" || p_label == RTR("Help"))) {
			p_index = [menu numberOfItems];
		} else if (p_index < 0) {
			p_index = item_count;
		} else {
			if (menu == [NSApp mainMenu]) { // Skip Apple menu.
				p_index++;
			}
			p_index = CLAMP(p_index, 0, item_count);
		}
		menu_item = [menu insertItemWithTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()] action:@selector(globalMenuCallback:) keyEquivalent:[NSString stringWithUTF8String:keycode.utf8().get_data()] atIndex:p_index];
		*r_out = (menu == [NSApp mainMenu]) ? p_index - 1 : p_index;
		return menu_item;
	}
	return nullptr;
}

int DisplayServerMacOS::global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_NONE;
		obj->max_states = 0;
		obj->state = 0;
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_CHECK_BOX;
		obj->max_states = 0;
		obj->state = 0;
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_icon_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_NONE;
		obj->max_states = 0;
		obj->state = 0;
		if (p_icon.is_valid()) {
			obj->img = p_icon->get_image();
			obj->img = obj->img->duplicate();
			if (obj->img->is_compressed()) {
				obj->img->decompress();
			}
			obj->img->resize(16, 16, Image::INTERPOLATE_LANCZOS);
			[menu_item setImage:_convert_to_nsimg(obj->img)];
		}
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_icon_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_CHECK_BOX;
		obj->max_states = 0;
		obj->state = 0;
		if (p_icon.is_valid()) {
			obj->img = p_icon->get_image();
			obj->img = obj->img->duplicate();
			if (obj->img->is_compressed()) {
				obj->img->decompress();
			}
			obj->img->resize(16, 16, Image::INTERPOLATE_LANCZOS);
			[menu_item setImage:_convert_to_nsimg(obj->img)];
		}
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_radio_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_RADIO_BUTTON;
		obj->max_states = 0;
		obj->state = 0;
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_icon_radio_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_RADIO_BUTTON;
		obj->max_states = 0;
		obj->state = 0;
		if (p_icon.is_valid()) {
			obj->img = p_icon->get_image();
			obj->img = obj->img->duplicate();
			if (obj->img->is_compressed()) {
				obj->img->decompress();
			}
			obj->img->resize(16, 16, Image::INTERPOLATE_LANCZOS);
			[menu_item setImage:_convert_to_nsimg(obj->img)];
		}
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_multistate_item(const String &p_menu_root, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	_THREAD_SAFE_METHOD_

	int out = -1;
	NSMenuItem *menu_item = _menu_add_item(p_menu_root, p_label, p_accel, p_index, &out);
	if (menu_item) {
		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = p_callback;
		obj->key_callback = p_key_callback;
		obj->meta = p_tag;
		obj->checkable_type = CHECKABLE_TYPE_NONE;
		obj->max_states = p_max_states;
		obj->state = p_default_state;
		[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_accel)];
		[menu_item setRepresentedObject:obj];
	}
	return out;
}

void DisplayServerMacOS::global_menu_set_popup_callbacks(const String &p_menu_root, const Callable &p_open_callback, const Callable &p_close_callback) {
	_THREAD_SAFE_METHOD_

	if (p_menu_root != "" && p_menu_root.to_lower() != "_main" && p_menu_root.to_lower() != "_dock") {
		// Submenu.
		if (!submenu.has(p_menu_root)) {
			NSMenu *n_menu = [[NSMenu alloc] initWithTitle:[NSString stringWithUTF8String:p_menu_root.utf8().get_data()]];
			[n_menu setAutoenablesItems:NO];
			[n_menu setDelegate:menu_delegate];
			submenu[p_menu_root].menu = n_menu;
			submenu_inv[n_menu] = p_menu_root;
		}
		submenu[p_menu_root].open = p_open_callback;
		submenu[p_menu_root].close = p_close_callback;
	}
}

int DisplayServerMacOS::global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu, int p_index) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	NSMenu *sub_menu = _get_menu_root(p_submenu);
	int out = -1;
	if (menu && sub_menu) {
		if (sub_menu == menu) {
			ERR_PRINT("Can't set submenu to self!");
			return -1;
		}
		if ([sub_menu supermenu]) {
			ERR_PRINT("Can't set submenu to menu that is already a submenu of some other menu!");
			return -1;
		}
		NSMenuItem *menu_item;
		int item_count = ((menu == [NSApp mainMenu]) && _has_help_menu()) ? [menu numberOfItems] - 1 : [menu numberOfItems];
		if ((menu == [NSApp mainMenu]) && (p_label == "Help" || p_label == RTR("Help"))) {
			p_index = [menu numberOfItems];
		} else if (p_index < 0) {
			p_index = item_count;
		} else {
			if (menu == [NSApp mainMenu]) { // Skip Apple menu.
				p_index++;
			}
			p_index = CLAMP(p_index, 0, item_count);
		}
		menu_item = [menu insertItemWithTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()] action:nil keyEquivalent:@"" atIndex:p_index];
		out = (menu == [NSApp mainMenu]) ? p_index - 1 : p_index;

		GodotMenuItem *obj = [[GodotMenuItem alloc] init];
		obj->callback = Callable();
		obj->checkable_type = CHECKABLE_TYPE_NONE;
		obj->max_states = 0;
		obj->state = 0;
		[menu_item setRepresentedObject:obj];

		[sub_menu setTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()]];
		[menu setSubmenu:sub_menu forItem:menu_item];
	}
	return out;
}

int DisplayServerMacOS::global_menu_add_separator(const String &p_menu_root, int p_index) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if (menu == [NSApp mainMenu]) { // Do not add separators into main menu.
			return -1;
		}
		if (p_index < 0) {
			p_index = [menu numberOfItems];
		} else {
			p_index = CLAMP(p_index, 0, [menu numberOfItems]);
		}
		[menu insertItem:[NSMenuItem separatorItem] atIndex:p_index];
		return p_index;
	}
	return -1;
}

int DisplayServerMacOS::global_menu_get_item_index_from_text(const String &p_menu_root, const String &p_text) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			return [menu indexOfItemWithTitle:[NSString stringWithUTF8String:p_text.utf8().get_data()]] - 1;
		} else {
			return [menu indexOfItemWithTitle:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
		}
	}

	return -1;
}

int DisplayServerMacOS::global_menu_get_item_index_from_tag(const String &p_menu_root, const Variant &p_tag) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		for (NSInteger i = (menu == [NSApp mainMenu]) ? 1 : 0; i < [menu numberOfItems]; i++) {
			const NSMenuItem *menu_item = [menu itemAtIndex:i];
			if (menu_item) {
				const GodotMenuItem *obj = [menu_item representedObject];
				if (obj && obj->meta == p_tag) {
					if (menu == [NSApp mainMenu]) { // Skip Apple menu.
						return i - 1;
					} else {
						return i;
					}
				}
			}
		}
	}

	return -1;
}

bool DisplayServerMacOS::global_menu_is_item_checked(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, false);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], false);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return ([menu_item state] == NSControlStateValueOn);
		}
	}
	return false;
}

bool DisplayServerMacOS::global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, false);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], false);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->checkable_type == CHECKABLE_TYPE_CHECK_BOX;
			}
		}
	}
	return false;
}

bool DisplayServerMacOS::global_menu_is_item_radio_checkable(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, false);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], false);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->checkable_type == CHECKABLE_TYPE_RADIO_BUTTON;
			}
		}
	}
	return false;
}

Callable DisplayServerMacOS::global_menu_get_item_callback(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, Callable());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], Callable());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->callback;
			}
		}
	}
	return Callable();
}

Callable DisplayServerMacOS::global_menu_get_item_key_callback(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, Callable());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], Callable());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->key_callback;
			}
		}
	}
	return Callable();
}

Variant DisplayServerMacOS::global_menu_get_item_tag(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, Variant());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], Variant());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->meta;
			}
		}
	}
	return Variant();
}

String DisplayServerMacOS::global_menu_get_item_text(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, String());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], String());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return String::utf8([[menu_item title] UTF8String]);
		}
	}
	return String();
}

String DisplayServerMacOS::global_menu_get_item_submenu(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, String());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], String());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			NSMenu *sub_menu = [menu_item submenu];
			if (sub_menu && submenu_inv.has(sub_menu)) {
				return submenu_inv[sub_menu];
			}
		}
	}
	return String();
}

Key DisplayServerMacOS::global_menu_get_item_accelerator(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, Key::NONE);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], Key::NONE);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			String ret = String::utf8([[menu_item keyEquivalent] UTF8String]);
			Key keycode = find_keycode(ret);
			NSUInteger mask = [menu_item keyEquivalentModifierMask];
			if (mask & NSEventModifierFlagControl) {
				keycode |= KeyModifierMask::CTRL;
			}
			if (mask & NSEventModifierFlagOption) {
				keycode |= KeyModifierMask::ALT;
			}
			if (mask & NSEventModifierFlagShift) {
				keycode |= KeyModifierMask::SHIFT;
			}
			if (mask & NSEventModifierFlagCommand) {
				keycode |= KeyModifierMask::META;
			}
			if (mask & NSEventModifierFlagNumericPad) {
				keycode |= KeyModifierMask::KPAD;
			}
			return keycode;
		}
	}
	return Key::NONE;
}

bool DisplayServerMacOS::global_menu_is_item_disabled(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, false);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], false);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return ![menu_item isEnabled];
		}
	}
	return false;
}

bool DisplayServerMacOS::global_menu_is_item_hidden(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, false);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], false);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return [menu_item isHidden];
		}
	}
	return false;
}

String DisplayServerMacOS::global_menu_get_item_tooltip(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, String());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], String());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return String::utf8([[menu_item toolTip] UTF8String]);
		}
	}
	return String();
}

int DisplayServerMacOS::global_menu_get_item_state(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], 0);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->state;
			}
		}
	}
	return 0;
}

int DisplayServerMacOS::global_menu_get_item_max_states(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], 0);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->max_states;
			}
		}
	}
	return 0;
}

Ref<Texture2D> DisplayServerMacOS::global_menu_get_item_icon(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, Ref<Texture2D>());
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], Ref<Texture2D>());
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			if (obj) {
				if (obj->img.is_valid()) {
					return ImageTexture::create_from_image(obj->img);
				}
			}
		}
	}
	return Ref<Texture2D>();
}

int DisplayServerMacOS::global_menu_get_item_indentation_level(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND_V(p_idx < 0, 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND_V(p_idx >= [menu numberOfItems], 0);
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return [menu_item indentationLevel];
		}
	}
	return 0;
}

void DisplayServerMacOS::global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			if (p_checked) {
				[menu_item setState:NSControlStateValueOn];
			} else {
				[menu_item setState:NSControlStateValueOff];
			}
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->checkable_type = (p_checkable) ? CHECKABLE_TYPE_CHECK_BOX : CHECKABLE_TYPE_NONE;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_radio_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->checkable_type = (p_checkable) ? CHECKABLE_TYPE_RADIO_BUTTON : CHECKABLE_TYPE_NONE;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->callback = p_callback;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_hover_callbacks(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->hover_callback = p_callback;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_key_callback(const String &p_menu_root, int p_idx, const Callable &p_key_callback) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->key_callback = p_key_callback;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->meta = p_tag;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setTitle:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
			NSMenu *sub_menu = [menu_item submenu];
			if (sub_menu) {
				[sub_menu setTitle:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
			}
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu && p_submenu.is_empty()) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			if ([menu_item submenu] && _is_menu_opened([menu_item submenu])) {
				ERR_PRINT("Can't remove open menu!");
				return;
			}
			[menu setSubmenu:nil forItem:menu_item];
		}
		return;
	}

	NSMenu *sub_menu = _get_menu_root(p_submenu);
	if (menu && sub_menu) {
		if (sub_menu == menu) {
			ERR_PRINT("Can't set submenu to self!");
			return;
		}
		if ([sub_menu supermenu]) {
			ERR_PRINT("Can't set submenu to menu that is already a submenu of some other menu!");
			return;
		}
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu setSubmenu:sub_menu forItem:menu_item];
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_accelerator(const String &p_menu_root, int p_idx, Key p_keycode) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			if (p_keycode == Key::NONE) {
				[menu_item setKeyEquivalent:@""];
			} else {
				[menu_item setKeyEquivalentModifierMask:KeyMappingMacOS::keycode_get_native_mask(p_keycode)];
				String keycode = KeyMappingMacOS::keycode_get_native_string(p_keycode & KeyModifierMask::CODE_MASK);
				[menu_item setKeyEquivalent:[NSString stringWithUTF8String:keycode.utf8().get_data()]];
			}
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_disabled(const String &p_menu_root, int p_idx, bool p_disabled) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setEnabled:(!p_disabled)];
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_hidden(const String &p_menu_root, int p_idx, bool p_hidden) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setHidden:p_hidden];
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_tooltip(const String &p_menu_root, int p_idx, const String &p_tooltip) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setToolTip:[NSString stringWithUTF8String:p_tooltip.utf8().get_data()]];
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_state(const String &p_menu_root, int p_idx, int p_state) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->state = p_state;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_max_states(const String &p_menu_root, int p_idx, int p_max_states) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			obj->max_states = p_max_states;
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_icon(const String &p_menu_root, int p_idx, const Ref<Texture2D> &p_icon) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GodotMenuItem *obj = [menu_item representedObject];
			ERR_FAIL_NULL(obj);
			if (p_icon.is_valid()) {
				obj->img = p_icon->get_image();
				obj->img = obj->img->duplicate();
				if (obj->img->is_compressed()) {
					obj->img->decompress();
				}
				obj->img->resize(16, 16, Image::INTERPOLATE_LANCZOS);
				[menu_item setImage:_convert_to_nsimg(obj->img)];
			} else {
				obj->img = Ref<Image>();
				[menu_item setImage:nil];
			}
		}
	}
}

void DisplayServerMacOS::global_menu_set_item_indentation_level(const String &p_menu_root, int p_idx, int p_level) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setIndentationLevel:p_level];
		}
	}
}

int DisplayServerMacOS::global_menu_get_item_count(const String &p_menu_root) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			return [menu numberOfItems] - 1;
		} else {
			return [menu numberOfItems];
		}
	} else {
		return 0;
	}
}

void DisplayServerMacOS::global_menu_remove_item(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		ERR_FAIL_COND(p_idx < 0);
		if (menu == [NSApp mainMenu]) { // Skip Apple menu.
			p_idx++;
		}
		ERR_FAIL_COND(p_idx >= [menu numberOfItems]);
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if ([menu_item submenu] && _is_menu_opened([menu_item submenu])) {
			ERR_PRINT("Can't remove open menu!");
			return;
		}
		[menu removeItemAtIndex:p_idx];
	}
}

void DisplayServerMacOS::global_menu_clear(const String &p_menu_root) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if (_is_menu_opened(menu)) {
			ERR_PRINT("Can't remove open menu!");
			return;
		}
		[menu removeAllItems];
		// Restore Apple menu.
		if (menu == [NSApp mainMenu]) {
			NSMenuItem *menu_item = [menu addItemWithTitle:@"" action:nil keyEquivalent:@""];
			[menu setSubmenu:apple_menu forItem:menu_item];
		}
		if (submenu.has(p_menu_root)) {
			submenu_inv.erase(submenu[p_menu_root].menu);
			submenu.erase(p_menu_root);
		}
	}
}

bool DisplayServerMacOS::tts_is_speaking() const {
	ERR_FAIL_NULL_V_MSG(tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts isSpeaking];
}

bool DisplayServerMacOS::tts_is_paused() const {
	ERR_FAIL_NULL_V_MSG(tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts isPaused];
}

TypedArray<Dictionary> DisplayServerMacOS::tts_get_voices() const {
	ERR_FAIL_NULL_V_MSG(tts, Array(), "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	return [tts getVoices];
}

void DisplayServerMacOS::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts speak:p_text voice:p_voice volume:p_volume pitch:p_pitch rate:p_rate utterance_id:p_utterance_id interrupt:p_interrupt];
}

void DisplayServerMacOS::tts_pause() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts pauseSpeaking];
}

void DisplayServerMacOS::tts_resume() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts resumeSpeaking];
}

void DisplayServerMacOS::tts_stop() {
	ERR_FAIL_NULL_MSG(tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	[tts stopSpeaking];
}

bool DisplayServerMacOS::is_dark_mode_supported() const {
	if (@available(macOS 10.14, *)) {
		return true;
	} else {
		return false;
	}
}

bool DisplayServerMacOS::is_dark_mode() const {
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

Color DisplayServerMacOS::get_accent_color() const {
	if (@available(macOS 10.14, *)) {
		NSColor *color = [[NSColor controlAccentColor] colorUsingColorSpace:[NSColorSpace genericRGBColorSpace]];
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

Error DisplayServerMacOS::dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	NSAlert *window = [[NSAlert alloc] init];
	NSString *ns_title = [NSString stringWithUTF8String:p_title.utf8().get_data()];
	NSString *ns_description = [NSString stringWithUTF8String:p_description.utf8().get_data()];

	for (int i = 0; i < p_buttons.size(); i++) {
		NSString *ns_button = [NSString stringWithUTF8String:p_buttons[i].utf8().get_data()];
		[window addButtonWithTitle:ns_button];
	}
	[window setMessageText:ns_title];
	[window setInformativeText:ns_description];
	[window setAlertStyle:NSAlertStyleInformational];

	Variant button_pressed;
	NSInteger ret = [window runModal];
	if (ret == NSAlertFirstButtonReturn) {
		button_pressed = int64_t(0);
	} else if (ret == NSAlertSecondButtonReturn) {
		button_pressed = int64_t(1);
	} else if (ret == NSAlertThirdButtonReturn) {
		button_pressed = int64_t(2);
	} else {
		button_pressed = int64_t(2 + (ret - NSAlertThirdButtonReturn));
	}

	if (!p_callback.is_null()) {
		Variant ret;
		Callable::CallError ce;
		const Variant *args[1] = { &button_pressed };

		p_callback.callp(args, 1, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute dialog callback: %s.", Variant::get_callable_error_text(p_callback, args, 1, ce)));
		}
	}

	return OK;
}

@interface FileDialogDropdown : NSObject {
	NSSavePanel *dialog;
	NSMutableArray *allowed_types;
	int cur_index;
}

- (instancetype)initWithDialog:(NSSavePanel *)p_dialog fileTypes:(NSMutableArray *)p_allowed_types;
- (void)popupAction:(id)sender;
- (int)getIndex;

@end

@implementation FileDialogDropdown

- (int)getIndex {
	return cur_index;
}

- (instancetype)initWithDialog:(NSSavePanel *)p_dialog fileTypes:(NSMutableArray *)p_allowed_types {
	if ((self = [super init])) {
		dialog = p_dialog;
		allowed_types = p_allowed_types;
		cur_index = 0;
	}
	return self;
}

- (void)popupAction:(id)sender {
	NSUInteger index = [sender indexOfSelectedItem];
	if (index < [allowed_types count]) {
		[dialog setAllowedFileTypes:[allowed_types objectAtIndex:index]];
		cur_index = index;
	} else {
		[dialog setAllowedFileTypes:@[]];
		cur_index = -1;
	}
}

@end

FileDialogDropdown *_make_accessory_view(NSSavePanel *p_panel, const Vector<String> &p_filters) {
	NSView *group = [[NSView alloc] initWithFrame:NSZeroRect];
	group.translatesAutoresizingMaskIntoConstraints = NO;

	NSTextField *label = [NSTextField labelWithString:[NSString stringWithUTF8String:RTR("Format").utf8().get_data()]];
	label.translatesAutoresizingMaskIntoConstraints = NO;
	if (@available(macOS 10.14, *)) {
		label.textColor = NSColor.secondaryLabelColor;
	}
	if (@available(macOS 11.10, *)) {
		label.font = [NSFont systemFontOfSize:[NSFont smallSystemFontSize]];
	}
	[group addSubview:label];

	NSPopUpButton *popup = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
	popup.translatesAutoresizingMaskIntoConstraints = NO;

	NSMutableArray *allowed_types = [[NSMutableArray alloc] init];
	bool allow_other = false;
	for (int i = 0; i < p_filters.size(); i++) {
		Vector<String> tokens = p_filters[i].split(";");
		if (tokens.size() >= 1) {
			String flt = tokens[0].strip_edges();
			int filter_slice_count = flt.get_slice_count(",");

			NSMutableArray *type_filters = [[NSMutableArray alloc] init];
			for (int j = 0; j < filter_slice_count; j++) {
				String str = (flt.get_slice(",", j).strip_edges());
				if (str.strip_edges() == "*.*" || str.strip_edges() == "*") {
					allow_other = true;
				} else if (!str.is_empty()) {
					[type_filters addObject:[NSString stringWithUTF8String:str.replace("*.", "").strip_edges().utf8().get_data()]];
				}
			}

			if ([type_filters count] > 0) {
				NSString *name_str = [NSString stringWithUTF8String:((tokens.size() == 1) ? tokens[0] : vformat("%s (%s)", tokens[1], tokens[0])).strip_edges().utf8().get_data()];
				[allowed_types addObject:type_filters];
				[popup addItemWithTitle:name_str];
			}
		}
	}
	FileDialogDropdown *handler = [[FileDialogDropdown alloc] initWithDialog:p_panel fileTypes:allowed_types];
	popup.target = handler;
	popup.action = @selector(popupAction:);

	[group addSubview:popup];

	NSView *view = [[NSView alloc] initWithFrame:NSZeroRect];
	view.translatesAutoresizingMaskIntoConstraints = NO;
	[view addSubview:group];

	NSMutableArray *constraints = [NSMutableArray array];
	[constraints addObject:[popup.topAnchor constraintEqualToAnchor:group.topAnchor constant:10]];
	[constraints addObject:[label.leadingAnchor constraintEqualToAnchor:group.leadingAnchor constant:10]];
	[constraints addObject:[popup.leadingAnchor constraintEqualToAnchor:label.trailingAnchor constant:10]];
	[constraints addObject:[popup.firstBaselineAnchor constraintEqualToAnchor:label.firstBaselineAnchor]];
	[constraints addObject:[group.trailingAnchor constraintEqualToAnchor:popup.trailingAnchor constant:10]];
	[constraints addObject:[group.bottomAnchor constraintEqualToAnchor:popup.bottomAnchor constant:10]];
	[constraints addObject:[group.topAnchor constraintEqualToAnchor:view.topAnchor]];
	[constraints addObject:[group.centerXAnchor constraintEqualToAnchor:view.centerXAnchor]];
	[constraints addObject:[view.bottomAnchor constraintEqualToAnchor:group.bottomAnchor]];
	[NSLayoutConstraint activateConstraints:constraints];

	[p_panel setAllowsOtherFileTypes:allow_other];
	if ([allowed_types count] > 0) {
		[p_panel setAccessoryView:view];
		[p_panel setAllowedFileTypes:[allowed_types objectAtIndex:0]];
	}

	return handler;
}

Error DisplayServerMacOS::file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX_V(int(p_mode), FILE_DIALOG_MODE_SAVE_MAX, FAILED);

	NSString *url = [NSString stringWithUTF8String:p_current_directory.utf8().get_data()];
	FileDialogDropdown *handler = nullptr;

	WindowID prev_focus = last_focused_window;

	Callable callback = p_callback; // Make a copy for async completion handler.
	if (p_mode == FILE_DIALOG_MODE_SAVE_FILE) {
		NSSavePanel *panel = [NSSavePanel savePanel];

		[panel setDirectoryURL:[NSURL fileURLWithPath:url]];
		handler = _make_accessory_view(panel, p_filters);
		[panel setExtensionHidden:YES];
		[panel setCanSelectHiddenExtension:YES];
		[panel setCanCreateDirectories:YES];
		[panel setShowsHiddenFiles:p_show_hidden];
		if (p_filename != "") {
			NSString *fileurl = [NSString stringWithUTF8String:p_filename.utf8().get_data()];
			[panel setNameFieldStringValue:fileurl];
		}

		[panel beginSheetModalForWindow:[[NSApplication sharedApplication] mainWindow]
					  completionHandler:^(NSInteger ret) {
						  if (ret == NSModalResponseOK) {
							  // Save bookmark for folder.
							  if (OS::get_singleton()->is_sandboxed()) {
								  NSArray *bookmarks = [[NSUserDefaults standardUserDefaults] arrayForKey:@"sec_bookmarks"];
								  bool skip = false;
								  for (id bookmark in bookmarks) {
									  NSError *error = nil;
									  BOOL isStale = NO;
									  NSURL *exurl = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&isStale error:&error];
									  if (!error && !isStale && ([[exurl path] compare:[[panel directoryURL] path]] == NSOrderedSame)) {
										  skip = true;
										  break;
									  }
								  }
								  if (!skip) {
									  NSError *error = nil;
									  NSData *bookmark = [[panel directoryURL] bookmarkDataWithOptions:NSURLBookmarkCreationWithSecurityScope includingResourceValuesForKeys:nil relativeToURL:nil error:&error];
									  if (!error) {
										  NSArray *new_bookmarks = [bookmarks arrayByAddingObject:bookmark];
										  [[NSUserDefaults standardUserDefaults] setObject:new_bookmarks forKey:@"sec_bookmarks"];
									  }
								  }
							  }
							  // Callback.
							  Vector<String> files;
							  String url;
							  url.parse_utf8([[[panel URL] path] UTF8String]);
							  files.push_back(url);
							  if (!callback.is_null()) {
								  Variant v_result = true;
								  Variant v_files = files;
								  Variant v_index = [handler getIndex];
								  Variant ret;
								  Callable::CallError ce;
								  const Variant *args[3] = { &v_result, &v_files, &v_index };

								  callback.callp(args, 3, ret, ce);
								  if (ce.error != Callable::CallError::CALL_OK) {
									  ERR_PRINT(vformat("Failed to execute file dialog callback: %s.", Variant::get_callable_error_text(callback, args, 3, ce)));
								  }
							  }
						  } else {
							  if (!callback.is_null()) {
								  Variant v_result = false;
								  Variant v_files = Vector<String>();
								  Variant v_index = [handler getIndex];
								  Variant ret;
								  Callable::CallError ce;
								  const Variant *args[3] = { &v_result, &v_files, &v_index };

								  callback.callp(args, 3, ret, ce);
								  if (ce.error != Callable::CallError::CALL_OK) {
									  ERR_PRINT(vformat("Failed to execute file dialogs callback: %s.", Variant::get_callable_error_text(callback, args, 3, ce)));
								  }
							  }
						  }
						  if (prev_focus != INVALID_WINDOW_ID) {
							  callable_mp(DisplayServer::get_singleton(), &DisplayServer::window_move_to_foreground).call_deferred(prev_focus);
						  }
					  }];
	} else {
		NSOpenPanel *panel = [NSOpenPanel openPanel];

		[panel setDirectoryURL:[NSURL fileURLWithPath:url]];
		handler = _make_accessory_view(panel, p_filters);
		[panel setExtensionHidden:YES];
		[panel setCanSelectHiddenExtension:YES];
		[panel setCanCreateDirectories:YES];
		[panel setCanChooseFiles:(p_mode != FILE_DIALOG_MODE_OPEN_DIR)];
		[panel setCanChooseDirectories:(p_mode == FILE_DIALOG_MODE_OPEN_DIR || p_mode == FILE_DIALOG_MODE_OPEN_ANY)];
		[panel setShowsHiddenFiles:p_show_hidden];
		if (p_filename != "") {
			NSString *fileurl = [NSString stringWithUTF8String:p_filename.utf8().get_data()];
			[panel setNameFieldStringValue:fileurl];
		}
		[panel setAllowsMultipleSelection:(p_mode == FILE_DIALOG_MODE_OPEN_FILES)];

		[panel beginSheetModalForWindow:[[NSApplication sharedApplication] mainWindow]
					  completionHandler:^(NSInteger ret) {
						  if (ret == NSModalResponseOK) {
							  // Save bookmark for folder.
							  NSArray *urls = [(NSOpenPanel *)panel URLs];
							  if (OS::get_singleton()->is_sandboxed()) {
								  NSArray *bookmarks = [[NSUserDefaults standardUserDefaults] arrayForKey:@"sec_bookmarks"];
								  NSMutableArray *new_bookmarks = [bookmarks mutableCopy];
								  for (NSUInteger i = 0; i != [urls count]; ++i) {
									  bool skip = false;
									  for (id bookmark in bookmarks) {
										  NSError *error = nil;
										  BOOL isStale = NO;
										  NSURL *exurl = [NSURL URLByResolvingBookmarkData:bookmark options:NSURLBookmarkResolutionWithSecurityScope relativeToURL:nil bookmarkDataIsStale:&isStale error:&error];
										  if (!error && !isStale && ([[exurl path] compare:[[urls objectAtIndex:i] path]] == NSOrderedSame)) {
											  skip = true;
											  break;
										  }
									  }
									  if (!skip) {
										  NSError *error = nil;
										  NSData *bookmark = [[urls objectAtIndex:i] bookmarkDataWithOptions:NSURLBookmarkCreationWithSecurityScope includingResourceValuesForKeys:nil relativeToURL:nil error:&error];
										  if (!error) {
											  [new_bookmarks addObject:bookmark];
										  }
									  }
								  }
								  [[NSUserDefaults standardUserDefaults] setObject:new_bookmarks forKey:@"sec_bookmarks"];
							  }
							  // Callback.
							  Vector<String> files;
							  for (NSUInteger i = 0; i != [urls count]; ++i) {
								  String url;
								  url.parse_utf8([[[urls objectAtIndex:i] path] UTF8String]);
								  files.push_back(url);
							  }
							  if (!callback.is_null()) {
								  Variant v_result = true;
								  Variant v_files = files;
								  Variant v_index = [handler getIndex];
								  Variant ret;
								  Callable::CallError ce;
								  const Variant *args[3] = { &v_result, &v_files, &v_index };

								  callback.callp(args, 3, ret, ce);
								  if (ce.error != Callable::CallError::CALL_OK) {
									  ERR_PRINT(vformat("Failed to execute file dialog callback: %s.", Variant::get_callable_error_text(callback, args, 3, ce)));
								  }
							  }
						  } else {
							  if (!callback.is_null()) {
								  Variant v_result = false;
								  Variant v_files = Vector<String>();
								  Variant v_index = [handler getIndex];
								  Variant ret;
								  Callable::CallError ce;
								  const Variant *args[3] = { &v_result, &v_files, &v_index };

								  callback.callp(args, 3, ret, ce);
								  if (ce.error != Callable::CallError::CALL_OK) {
									  ERR_PRINT(vformat("Failed to execute file dialogs callback: %s.", Variant::get_callable_error_text(callback, args, 3, ce)));
								  }
							  }
						  }
						  if (prev_focus != INVALID_WINDOW_ID) {
							  callable_mp(DisplayServer::get_singleton(), &DisplayServer::window_move_to_foreground).call_deferred(prev_focus);
						  }
					  }];
	}

	return OK;
}

Error DisplayServerMacOS::dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	NSAlert *window = [[NSAlert alloc] init];
	NSString *ns_title = [NSString stringWithUTF8String:p_title.utf8().get_data()];
	NSString *ns_description = [NSString stringWithUTF8String:p_description.utf8().get_data()];
	NSTextField *input = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, 250, 30)];

	[window addButtonWithTitle:@"OK"];
	[window setMessageText:ns_title];
	[window setInformativeText:ns_description];
	[window setAlertStyle:NSAlertStyleInformational];

	[input setStringValue:[NSString stringWithUTF8String:p_partial.utf8().get_data()]];
	[window setAccessoryView:input];

	[window runModal];

	String ret;
	ret.parse_utf8([[input stringValue] UTF8String]);

	if (!p_callback.is_null()) {
		Variant v_result = ret;
		Variant ret;
		Callable::CallError ce;
		const Variant *args[1] = { &v_result };

		p_callback.callp(args, 1, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute input dialog callback: %s.", Variant::get_callable_error_text(p_callback, args, 1, ce)));
		}
	}

	return OK;
}

void DisplayServerMacOS::mouse_set_mode(MouseMode p_mode) {
	_THREAD_SAFE_METHOD_

	if (p_mode == mouse_mode) {
		return;
	}

	WindowID window_id = _get_focused_window_or_popup();
	if (!windows.has(window_id)) {
		window_id = MAIN_WINDOW_ID;
	}
	WindowData &wd = windows[window_id];

	bool show_cursor = (p_mode == MOUSE_MODE_VISIBLE || p_mode == MOUSE_MODE_CONFINED);
	bool previously_shown = (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED);

	if (show_cursor && !previously_shown) {
		window_id = get_window_at_screen_position(mouse_get_position());
		mouse_enter_window(window_id);
	}

	if (p_mode == MOUSE_MODE_CAPTURED) {
		// Apple Docs state that the display parameter is not used.
		// "This parameter is not used. By default, you may pass kCGDirectMainDisplay."
		// https://developer.apple.com/library/mac/documentation/graphicsimaging/reference/Quartz_Services_Ref/Reference/reference.html
		if (previously_shown) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		CGAssociateMouseAndMouseCursorPosition(false);
		[wd.window_object setMovable:NO];
		const NSRect contentRect = [wd.window_view frame];
		NSRect pointInWindowRect = NSMakeRect(contentRect.size.width / 2, contentRect.size.height / 2, 0, 0);
		NSPoint pointOnScreen = [[wd.window_view window] convertRectToScreen:pointInWindowRect].origin;
		CGPoint lMouseWarpPos = { pointOnScreen.x, CGDisplayBounds(CGMainDisplayID()).size.height - pointOnScreen.y };
		CGWarpMouseCursorPosition(lMouseWarpPos);
	} else if (p_mode == MOUSE_MODE_HIDDEN) {
		if (previously_shown) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		[wd.window_object setMovable:YES];
		CGAssociateMouseAndMouseCursorPosition(true);
	} else if (p_mode == MOUSE_MODE_CONFINED) {
		CGDisplayShowCursor(kCGDirectMainDisplay);
		[wd.window_object setMovable:NO];
		CGAssociateMouseAndMouseCursorPosition(false);
	} else if (p_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		if (previously_shown) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		[wd.window_object setMovable:NO];
		CGAssociateMouseAndMouseCursorPosition(false);
	} else { // MOUSE_MODE_VISIBLE
		CGDisplayShowCursor(kCGDirectMainDisplay);
		[wd.window_object setMovable:YES];
		CGAssociateMouseAndMouseCursorPosition(true);
	}

	last_warp = [[NSProcessInfo processInfo] systemUptime];
	ignore_warp = true;
	warp_events.clear();
	mouse_mode = p_mode;

	if (show_cursor) {
		cursor_update_shape();
	}
}

DisplayServer::MouseMode DisplayServerMacOS::mouse_get_mode() const {
	return mouse_mode;
}

bool DisplayServerMacOS::update_mouse_wrap(WindowData &p_wd, NSPoint &r_delta, NSPoint &r_mpos, NSTimeInterval p_timestamp) {
	_THREAD_SAFE_METHOD_

	if (ignore_warp) {
		// Discard late events, before warp.
		if (p_timestamp < last_warp) {
			return true;
		}
		ignore_warp = false;
		return true;
	}

	if (mouse_mode == DisplayServer::MOUSE_MODE_CONFINED || mouse_mode == DisplayServer::MOUSE_MODE_CONFINED_HIDDEN) {
		// Discard late events.
		if (p_timestamp < last_warp) {
			return true;
		}

		// Warp affects next event delta, subtract previous warp deltas.
		List<WarpEvent>::Element *F = warp_events.front();
		while (F) {
			if (F->get().timestamp < p_timestamp) {
				List<DisplayServerMacOS::WarpEvent>::Element *E = F;
				r_delta.x -= E->get().delta.x;
				r_delta.y -= E->get().delta.y;
				F = F->next();
				warp_events.erase(E);
			} else {
				F = F->next();
			}
		}

		// Confine mouse position to the window, and update delta.
		NSRect frame = [p_wd.window_object frame];
		NSPoint conf_pos = r_mpos;
		conf_pos.x = CLAMP(conf_pos.x + r_delta.x, 0.f, frame.size.width);
		conf_pos.y = CLAMP(conf_pos.y - r_delta.y, 0.f, frame.size.height);
		r_delta.x = conf_pos.x - r_mpos.x;
		r_delta.y = r_mpos.y - conf_pos.y;
		r_mpos = conf_pos;

		// Move mouse cursor.
		NSRect point_in_window_rect = NSMakeRect(conf_pos.x, conf_pos.y, 0, 0);
		conf_pos = [[p_wd.window_view window] convertRectToScreen:point_in_window_rect].origin;
		conf_pos.y = CGDisplayBounds(CGMainDisplayID()).size.height - conf_pos.y;
		CGWarpMouseCursorPosition(conf_pos);

		// Save warp data.
		last_warp = [[NSProcessInfo processInfo] systemUptime];

		DisplayServerMacOS::WarpEvent ev;
		ev.timestamp = last_warp;
		ev.delta = r_delta;
		warp_events.push_back(ev);
	}

	return false;
}

void DisplayServerMacOS::warp_mouse(const Point2i &p_position) {
	_THREAD_SAFE_METHOD_

	if (mouse_mode != MOUSE_MODE_CAPTURED) {
		WindowID window_id = _get_focused_window_or_popup();
		if (!windows.has(window_id)) {
			window_id = MAIN_WINDOW_ID;
		}
		WindowData &wd = windows[window_id];

		// Local point in window coords.
		const NSRect contentRect = [wd.window_view frame];
		const float scale = screen_get_max_scale();
		NSRect pointInWindowRect = NSMakeRect(p_position.x / scale, contentRect.size.height - (p_position.y / scale), scale, scale);
		NSPoint pointOnScreen = [[wd.window_view window] convertRectToScreen:pointInWindowRect].origin;

		// Point in screen coords.
		CGPoint lMouseWarpPos = { pointOnScreen.x, CGDisplayBounds(CGMainDisplayID()).size.height - pointOnScreen.y };

		// Do the warping.
		CGEventSourceRef lEventRef = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState);
		CGEventSourceSetLocalEventsSuppressionInterval(lEventRef, 0.0);
		CGAssociateMouseAndMouseCursorPosition(false);
		CGWarpMouseCursorPosition(lMouseWarpPos);
		if (mouse_mode != MOUSE_MODE_CONFINED && mouse_mode != MOUSE_MODE_CONFINED_HIDDEN) {
			CGAssociateMouseAndMouseCursorPosition(true);
		}
	}
}

Point2i DisplayServerMacOS::mouse_get_position() const {
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

void DisplayServerMacOS::mouse_set_button_state(BitField<MouseButtonMask> p_state) {
	last_button_state = p_state;
}

BitField<MouseButtonMask> DisplayServerMacOS::mouse_get_button_state() const {
	return last_button_state;
}

void DisplayServerMacOS::clipboard_set(const String &p_text) {
	_THREAD_SAFE_METHOD_

	NSString *copiedString = [NSString stringWithUTF8String:p_text.utf8().get_data()];
	NSArray *copiedStringArray = [NSArray arrayWithObject:copiedString];

	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	[pasteboard clearContents];
	[pasteboard writeObjects:copiedStringArray];
}

String DisplayServerMacOS::clipboard_get() const {
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
	ret.parse_utf8([string UTF8String]);
	return ret;
}

Ref<Image> DisplayServerMacOS::clipboard_get_image() const {
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
	NSData *pngData = [bitmap representationUsingType:NSPNGFileType properties:@{}];
	image.instantiate();
	PNGDriverCommon::png_to_image((const uint8_t *)pngData.bytes, pngData.length, false, image);
	return image;
}

bool DisplayServerMacOS::clipboard_has() const {
	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSArray *classArray = [NSArray arrayWithObject:[NSString class]];
	NSDictionary *options = [NSDictionary dictionary];
	return [pasteboard canReadObjectForClasses:classArray options:options];
}

bool DisplayServerMacOS::clipboard_has_image() const {
	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	NSString *result = [pasteboard availableTypeFromArray:[NSArray arrayWithObjects:NSPasteboardTypeTIFF, NSPasteboardTypePNG, nil]];
	return result;
}

int DisplayServerMacOS::get_screen_count() const {
	_THREAD_SAFE_METHOD_

	NSArray *screenArray = [NSScreen screens];
	return [screenArray count];
}

int DisplayServerMacOS::get_primary_screen() const {
	return 0;
}

int DisplayServerMacOS::get_keyboard_focus_screen() const {
	const NSUInteger index = [[NSScreen screens] indexOfObject:[NSScreen mainScreen]];
	return (index == NSNotFound) ? 0 : index;
}

Point2i DisplayServerMacOS::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	Point2i position = _get_native_screen_position(p_screen) - _get_screens_origin();
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value.
	position.y *= -1;
	return position;
}

Size2i DisplayServerMacOS::screen_get_size(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		// Note: Use frame to get the whole screen size.
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		return Size2i(nsrect.size.width, nsrect.size.height) * screen_get_max_scale();
	}

	return Size2i();
}

int DisplayServerMacOS::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSDictionary *description = [[screenArray objectAtIndex:p_screen] deviceDescription];

		const NSSize displayPixelSize = [[description objectForKey:NSDeviceSize] sizeValue];
		const CGSize displayPhysicalSize = CGDisplayScreenSize([[description objectForKey:@"NSScreenNumber"] unsignedIntValue]);
		float scale = [[screenArray objectAtIndex:p_screen] backingScaleFactor];

		float den2 = (displayPhysicalSize.width / 25.4f) * (displayPhysicalSize.width / 25.4f) + (displayPhysicalSize.height / 25.4f) * (displayPhysicalSize.height / 25.4f);
		if (den2 > 0.0f) {
			return ceil(sqrt(displayPixelSize.width * displayPixelSize.width + displayPixelSize.height * displayPixelSize.height) / sqrt(den2) * scale);
		}
	}

	return 72;
}

float DisplayServerMacOS::screen_get_scale(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	if (OS::get_singleton()->is_hidpi_allowed()) {
		NSArray *screenArray = [NSScreen screens];
		if ((NSUInteger)p_screen < [screenArray count]) {
			if ([[screenArray objectAtIndex:p_screen] respondsToSelector:@selector(backingScaleFactor)]) {
				return fmax(1.0, [[screenArray objectAtIndex:p_screen] backingScaleFactor]);
			}
		}
	}

	return 1.f;
}

float DisplayServerMacOS::screen_get_max_scale() const {
	_THREAD_SAFE_METHOD_

	// Note: Do not update max display scale on screen configuration change, existing editor windows can't be rescaled on the fly.
	return display_max_scale;
}

Rect2i DisplayServerMacOS::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
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

Color DisplayServerMacOS::screen_get_pixel(const Point2i &p_position) const {
	Point2i position = p_position;
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot passes a positive value.
	position.y *= -1;
	position += _get_screens_origin();
	position /= screen_get_max_scale();

	for (NSScreen *screen in [NSScreen screens]) {
		NSRect frame = [screen frame];
		if (NSMouseInRect(NSMakePoint(position.x, position.y), frame, NO)) {
			NSDictionary *screenDescription = [screen deviceDescription];
			CGDirectDisplayID display_id = [[screenDescription objectForKey:@"NSScreenNumber"] unsignedIntValue];
			CGImageRef image = CGDisplayCreateImageForRect(display_id, CGRectMake(position.x - frame.origin.x, frame.size.height - (position.y - frame.origin.y), 1, 1));
			if (image) {
				NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithCGImage:image];
				CGImageRelease(image);
				NSColor *color = [bitmap colorAtX:0 y:0];
				if (color) {
					CGFloat components[4];
					[color getRed:&components[0] green:&components[1] blue:&components[2] alpha:&components[3]];
					return Color(components[0], components[1], components[2], components[3]);
				}
			}
		}
	}
	return Color();
}

Ref<Image> DisplayServerMacOS::screen_get_image(int p_screen) const {
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), Ref<Image>());

	switch (p_screen) {
		case SCREEN_PRIMARY: {
			p_screen = get_primary_screen();
		} break;
		case SCREEN_OF_MAIN_WINDOW: {
			p_screen = window_get_current_screen(MAIN_WINDOW_ID);
		} break;
		default:
			break;
	}

	Ref<Image> img;
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		NSDictionary *screenDescription = [[screenArray objectAtIndex:p_screen] deviceDescription];
		CGDirectDisplayID display_id = [[screenDescription objectForKey:@"NSScreenNumber"] unsignedIntValue];
		CGImageRef image = CGDisplayCreateImageForRect(display_id, CGRectMake(0, 0, nsrect.size.width, nsrect.size.height));
		if (image) {
			CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
			if (color_space) {
				NSUInteger width = CGImageGetWidth(image);
				NSUInteger height = CGImageGetHeight(image);

				Vector<uint8_t> img_data;
				img_data.resize(height * width * 4);
				CGContextRef context = CGBitmapContextCreate(img_data.ptrw(), width, height, 8, 4 * width, color_space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
				if (context) {
					CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
					img = Image::create_from_data(width, height, false, Image::FORMAT_RGBA8, img_data);
					CGContextRelease(context);
				}
				CGColorSpaceRelease(color_space);
			}
			CGImageRelease(image);
		}
	}
	return img;
}

float DisplayServerMacOS::screen_get_refresh_rate(int p_screen) const {
	_THREAD_SAFE_METHOD_

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

bool DisplayServerMacOS::screen_is_kept_on() const {
	return (screen_keep_on_assertion);
}

void DisplayServerMacOS::screen_set_keep_on(bool p_enable) {
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

Vector<DisplayServer::WindowID> DisplayServerMacOS::get_window_list() const {
	_THREAD_SAFE_METHOD_

	Vector<int> ret;
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		ret.push_back(E.key);
	}
	return ret;
}

DisplayServer::WindowID DisplayServerMacOS::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	WindowID id = _create_window(p_mode, p_vsync_mode, p_rect);
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, id);
		}
	}

	return id;
}

void DisplayServerMacOS::show_window(WindowID p_id) {
	WindowData &wd = windows[p_id];

	popup_open(p_id);
	if ([wd.window_object isMiniaturized]) {
		return;
	} else if (wd.no_focus) {
		[wd.window_object orderFront:nil];
	} else {
		[wd.window_object makeKeyAndOrderFront:nil];
	}
}

void DisplayServerMacOS::delete_sub_window(WindowID p_id) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_id));
	ERR_FAIL_COND_MSG(p_id == MAIN_WINDOW_ID, "Main window can't be deleted");

	WindowData &wd = windows[p_id];

	[wd.window_object setContentView:nil];
	[wd.window_object close];
}

void DisplayServerMacOS::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.rect_changed_callback = p_callable;
}

void DisplayServerMacOS::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.event_callback = p_callable;
}

void DisplayServerMacOS::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_event_callback = p_callable;
}

void DisplayServerMacOS::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_text_callback = p_callable;
}

void DisplayServerMacOS::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.drop_files_callback = p_callable;
}

void DisplayServerMacOS::window_set_title(const String &p_title, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	[wd.window_object setTitle:[NSString stringWithUTF8String:p_title.utf8().get_data()]];
}

Size2i DisplayServerMacOS::window_get_title_size(const String &p_title, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	Size2i size;
	ERR_FAIL_COND_V(!windows.has(p_window), size);

	const WindowData &wd = windows[p_window];
	if (wd.fullscreen || wd.borderless) {
		return size;
	}
	if ([wd.window_object respondsToSelector:@selector(isMiniaturized)]) {
		if ([wd.window_object isMiniaturized]) {
			return size;
		}
	}

	float scale = screen_get_max_scale();

	if (wd.window_button_view) {
		size.x = ([wd.window_button_view getOffset].x + [wd.window_button_view frame].size.width);
		size.y = ([wd.window_button_view getOffset].y + [wd.window_button_view frame].size.height);
	} else {
		NSButton *cb = [wd.window_object standardWindowButton:NSWindowCloseButton];
		NSButton *mb = [wd.window_object standardWindowButton:NSWindowMiniaturizeButton];
		float cb_frame = NSMinX([cb frame]);
		float mb_frame = NSMinX([mb frame]);
		bool is_rtl = ([wd.window_object windowTitlebarLayoutDirection] == NSUserInterfaceLayoutDirectionRightToLeft);

		float window_buttons_spacing = (is_rtl) ? (cb_frame - mb_frame) : (mb_frame - cb_frame);
		size.x = window_buttons_spacing * 4;
		size.y = [cb frame].origin.y + [cb frame].size.height;
	}

	NSDictionary *attributes = [NSDictionary dictionaryWithObjectsAndKeys:[NSFont titleBarFontOfSize:0], NSFontAttributeName, nil];
	NSSize text_size = [[[NSAttributedString alloc] initWithString:[NSString stringWithUTF8String:p_title.utf8().get_data()] attributes:attributes] size];
	size.x += text_size.width;
	size.y = MAX(size.y, text_size.height);

	return size * scale;
}

void DisplayServerMacOS::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.mpath = p_region;
}

int DisplayServerMacOS::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	const WindowData &wd = windows[p_window];

	const NSUInteger index = [[NSScreen screens] indexOfObject:[wd.window_object screen]];
	return (index == NSNotFound) ? 0 : index;
}

void DisplayServerMacOS::window_set_current_screen(int p_screen, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (window_get_current_screen(p_window) == p_screen) {
		return;
	}

	bool was_fullscreen = false;
	if (wd.fullscreen) {
		// Temporary exit fullscreen mode to move window.
		[wd.window_object toggleFullScreen:nil];
		was_fullscreen = true;
	}

	Rect2i srect = screen_get_usable_rect(p_screen);
	Point2i wpos = window_get_position(p_window) - screen_get_position(window_get_current_screen(p_window));
	Size2i wsize = window_get_size(p_window);
	wpos += srect.position;

	wpos.x = CLAMP(wpos.x, srect.position.x, srect.position.x + srect.size.width - wsize.width / 3);
	wpos.y = CLAMP(wpos.y, srect.position.y, srect.position.y + srect.size.height - wsize.height / 3);
	window_set_position(wpos, p_window);

	if (was_fullscreen) {
		// Re-enter fullscreen mode.
		[wd.window_object toggleFullScreen:nil];
	}
}

void DisplayServerMacOS::reparent_check(WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	NSScreen *screen = [wd.window_object screen];

	if (wd.transient_parent != INVALID_WINDOW_ID) {
		WindowData &wd_parent = windows[wd.transient_parent];
		NSScreen *parent_screen = [wd_parent.window_object screen];

		if (parent_screen == screen) {
			if (![[wd_parent.window_object childWindows] containsObject:wd.window_object]) {
				[wd.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenAuxiliary];
				[wd_parent.window_object addChildWindow:wd.window_object ordered:NSWindowAbove];
			}
		} else {
			if ([[wd_parent.window_object childWindows] containsObject:wd.window_object]) {
				[wd_parent.window_object removeChildWindow:wd.window_object];
				[wd.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
				[wd.window_object orderFront:nil];
			}
		}
	}

	for (const WindowID &child : wd.transient_children) {
		WindowData &wd_child = windows[child];
		NSScreen *child_screen = [wd_child.window_object screen];

		if (child_screen == screen) {
			if (![[wd.window_object childWindows] containsObject:wd_child.window_object]) {
				if (wd_child.fullscreen) {
					[wd_child.window_object toggleFullScreen:nil];
				}
				[wd_child.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenAuxiliary];
				[wd.window_object addChildWindow:wd_child.window_object ordered:NSWindowAbove];
			}
		} else {
			if ([[wd.window_object childWindows] containsObject:wd_child.window_object]) {
				[wd.window_object removeChildWindow:wd_child.window_object];
				[wd_child.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
			}
		}
	}
}

void DisplayServerMacOS::window_set_exclusive(WindowID p_window, bool p_exclusive) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	if (wd.exclusive != p_exclusive) {
		wd.exclusive = p_exclusive;
		reparent_check(p_window);
	}
}

Point2i DisplayServerMacOS::window_get_position(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());
	const WindowData &wd = windows[p_window];

	// Use content rect position (without titlebar / window border).
	const NSRect contentRect = [wd.window_view frame];
	const NSRect nsrect = [wd.window_object convertRectToScreen:contentRect];
	Point2i pos;

	// Return the position of the top-left corner, for macOS the y starts at the bottom.
	const float scale = screen_get_max_scale();
	pos.x = nsrect.origin.x;
	pos.y = (nsrect.origin.y + nsrect.size.height);
	pos *= scale;
	pos -= _get_screens_origin();
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value.
	pos.y *= -1;
	return pos;
}

Point2i DisplayServerMacOS::window_get_position_with_decorations(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());
	const WindowData &wd = windows[p_window];

	const NSRect nsrect = [wd.window_object frame];
	Point2i pos;

	// Return the position of the top-left corner, for macOS the y starts at the bottom.
	const float scale = screen_get_max_scale();
	pos.x = nsrect.origin.x;
	pos.y = (nsrect.origin.y + nsrect.size.height);
	pos *= scale;
	pos -= _get_screens_origin();
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value.
	pos.y *= -1;
	return pos;
}

void DisplayServerMacOS::window_set_position(const Point2i &p_position, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (NSEqualRects([wd.window_object frame], [[wd.window_object screen] visibleFrame])) {
		return;
	}

	Point2i position = p_position;
	// macOS native y-coordinate relative to _get_screens_origin() is negative,
	// Godot passes a positive value.
	position.y *= -1;
	position += _get_screens_origin();
	position /= screen_get_max_scale();

	// Remove titlebar / window border size.
	const NSRect contentRect = [wd.window_view frame];
	const NSRect windowRect = [wd.window_object frame];
	const NSRect nsrect = [wd.window_object convertRectToScreen:contentRect];
	Point2i offset;
	offset.x = (nsrect.origin.x - windowRect.origin.x);
	offset.y = (nsrect.origin.y + nsrect.size.height);
	offset.y -= (windowRect.origin.y + windowRect.size.height);

	[wd.window_object setFrameTopLeftPoint:NSMakePoint(position.x - offset.x, position.y - offset.y)];

	_update_window_style(wd);
	update_mouse_pos(wd, [wd.window_object mouseLocationOutsideOfEventStream]);
}

void DisplayServerMacOS::window_set_transient(WindowID p_window, WindowID p_parent) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(p_window == p_parent);

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd_window = windows[p_window];

	ERR_FAIL_COND(wd_window.transient_parent == p_parent);

	ERR_FAIL_COND_MSG(wd_window.on_top, "Windows with the 'on top' can't become transient.");
	if (p_parent == INVALID_WINDOW_ID) {
		// Remove transient.
		ERR_FAIL_COND(wd_window.transient_parent == INVALID_WINDOW_ID);
		ERR_FAIL_COND(!windows.has(wd_window.transient_parent));

		WindowData &wd_parent = windows[wd_window.transient_parent];

		wd_window.transient_parent = INVALID_WINDOW_ID;
		wd_parent.transient_children.erase(p_window);
		if ([[wd_parent.window_object childWindows] containsObject:wd_window.window_object]) {
			[wd_parent.window_object removeChildWindow:wd_window.window_object];
		}
		[wd_window.window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
	} else {
		ERR_FAIL_COND(!windows.has(p_parent));
		ERR_FAIL_COND_MSG(wd_window.transient_parent != INVALID_WINDOW_ID, "Window already has a transient parent");
		WindowData &wd_parent = windows[p_parent];

		wd_window.transient_parent = p_parent;
		wd_parent.transient_children.insert(p_window);
		reparent_check(p_window);
	}
}

void DisplayServerMacOS::window_set_max_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2i()) && ((p_size.x < wd.min_size.x) || (p_size.y < wd.min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	wd.max_size = p_size;

	if ((wd.max_size != Size2i()) && !wd.fullscreen) {
		Size2i size = wd.max_size / screen_get_max_scale();
		[wd.window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
	} else {
		[wd.window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
	}
}

Size2i DisplayServerMacOS::window_get_max_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.max_size;
}

void DisplayServerMacOS::update_presentation_mode() {
	for (const KeyValue<WindowID, WindowData> &wd : windows) {
		if (wd.value.fullscreen && wd.value.exclusive_fullscreen) {
			return;
		}
	}
	[NSApp setPresentationOptions:NSApplicationPresentationDefault];
}

void DisplayServerMacOS::window_set_min_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2i()) && (wd.max_size != Size2i()) && ((p_size.x > wd.max_size.x) || (p_size.y > wd.max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	wd.min_size = p_size;

	if ((wd.min_size != Size2i()) && !wd.fullscreen) {
		Size2i size = wd.min_size / screen_get_max_scale();
		[wd.window_object setContentMinSize:NSMakeSize(size.x, size.y)];
	} else {
		[wd.window_object setContentMinSize:NSMakeSize(0, 0)];
	}
}

Size2i DisplayServerMacOS::window_get_min_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	return wd.min_size;
}

void DisplayServerMacOS::window_set_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (NSEqualRects([wd.window_object frame], [[wd.window_object screen] visibleFrame])) {
		return;
	}

	Size2i size = p_size / screen_get_max_scale();

	NSPoint top_left;
	NSRect old_frame = [wd.window_object frame];
	top_left.x = old_frame.origin.x;
	top_left.y = NSMaxY(old_frame);

	NSRect new_frame = NSMakeRect(0, 0, MAX(1, size.x), MAX(1, size.y));
	new_frame = [wd.window_object frameRectForContentRect:new_frame];

	new_frame.origin.x = top_left.x;
	new_frame.origin.y = top_left.y - new_frame.size.height;

	[wd.window_object setFrame:new_frame display:YES];

	_update_window_style(wd);
}

Size2i DisplayServerMacOS::window_get_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.size;
}

Size2i DisplayServerMacOS::window_get_size_with_decorations(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	NSRect frame = [wd.window_object frame];
	return Size2i(frame.size.width, frame.size.height) * screen_get_max_scale();
}

void DisplayServerMacOS::window_set_mode(WindowMode p_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	WindowMode old_mode = window_get_mode(p_window);
	if (old_mode == p_mode) {
		return; // Do nothing.
	}

	switch (old_mode) {
		case WINDOW_MODE_WINDOWED: {
			// Do nothing.
		} break;
		case WINDOW_MODE_MINIMIZED: {
			[wd.window_object deminiaturize:nil];
		} break;
		case WINDOW_MODE_EXCLUSIVE_FULLSCREEN:
		case WINDOW_MODE_FULLSCREEN: {
			if (p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN || p_mode == WINDOW_MODE_FULLSCREEN) {
				if (p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
					const NSUInteger presentationOptions = NSApplicationPresentationHideDock | NSApplicationPresentationHideMenuBar;
					[NSApp setPresentationOptions:presentationOptions];
					wd.exclusive_fullscreen = true;
				} else {
					wd.exclusive_fullscreen = false;
					update_presentation_mode();
				}
				return;
			}

			[(NSWindow *)wd.window_object setLevel:NSNormalWindowLevel];
			if (wd.resize_disabled) { // Restore resize disabled.
				[wd.window_object setStyleMask:[wd.window_object styleMask] & ~NSWindowStyleMaskResizable];
			}
			if (wd.min_size != Size2i()) {
				Size2i size = wd.min_size / screen_get_max_scale();
				[wd.window_object setContentMinSize:NSMakeSize(size.x, size.y)];
			}
			if (wd.max_size != Size2i()) {
				Size2i size = wd.max_size / screen_get_max_scale();
				[wd.window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
			}
			[wd.window_object toggleFullScreen:nil];

			if (old_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
				update_presentation_mode();
			}

			wd.fullscreen = false;
			wd.exclusive_fullscreen = false;
		} break;
		case WINDOW_MODE_MAXIMIZED: {
			if (NSEqualRects([wd.window_object frame], [[wd.window_object screen] visibleFrame])) {
				[wd.window_object zoom:nil];
			}
		} break;
	}

	switch (p_mode) {
		case WINDOW_MODE_WINDOWED: {
			// Do nothing.
		} break;
		case WINDOW_MODE_MINIMIZED: {
			[wd.window_object performMiniaturize:nil];
		} break;
		case WINDOW_MODE_EXCLUSIVE_FULLSCREEN:
		case WINDOW_MODE_FULLSCREEN: {
			if (wd.resize_disabled) { // Fullscreen window should be resizable to work.
				[wd.window_object setStyleMask:[wd.window_object styleMask] | NSWindowStyleMaskResizable];
			}
			[wd.window_object setContentMinSize:NSMakeSize(0, 0)];
			[wd.window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
			[wd.window_object toggleFullScreen:nil];

			wd.fullscreen = true;
			if (p_mode == WINDOW_MODE_EXCLUSIVE_FULLSCREEN) {
				const NSUInteger presentationOptions = NSApplicationPresentationHideDock | NSApplicationPresentationHideMenuBar;
				[NSApp setPresentationOptions:presentationOptions];
				wd.exclusive_fullscreen = true;
			} else {
				wd.exclusive_fullscreen = false;
				update_presentation_mode();
			}
		} break;
		case WINDOW_MODE_MAXIMIZED: {
			if (!NSEqualRects([wd.window_object frame], [[wd.window_object screen] visibleFrame])) {
				[wd.window_object zoom:nil];
			}
		} break;
	}
}

DisplayServer::WindowMode DisplayServerMacOS::window_get_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), WINDOW_MODE_WINDOWED);
	const WindowData &wd = windows[p_window];

	if (wd.fullscreen) { // If fullscreen, it's not in another mode.
		if (wd.exclusive_fullscreen) {
			return WINDOW_MODE_EXCLUSIVE_FULLSCREEN;
		} else {
			return WINDOW_MODE_FULLSCREEN;
		}
	}
	if (NSEqualRects([wd.window_object frame], [[wd.window_object screen] visibleFrame])) {
		return WINDOW_MODE_MAXIMIZED;
	}
	if ([wd.window_object respondsToSelector:@selector(isMiniaturized)]) {
		if ([wd.window_object isMiniaturized]) {
			return WINDOW_MODE_MINIMIZED;
		}
	}

	// All other discarded, return windowed.
	return WINDOW_MODE_WINDOWED;
}

bool DisplayServerMacOS::window_is_maximize_allowed(WindowID p_window) const {
	return true;
}

bool DisplayServerMacOS::window_maximize_on_title_dbl_click() const {
	id value = [[NSUserDefaults standardUserDefaults] objectForKey:@"AppleActionOnDoubleClick"];
	if ([value isKindOfClass:[NSString class]]) {
		return [value isEqualToString:@"Maximize"];
	}
	return false;
}

bool DisplayServerMacOS::window_minimize_on_title_dbl_click() const {
	id value = [[NSUserDefaults standardUserDefaults] objectForKey:@"AppleActionOnDoubleClick"];
	if ([value isKindOfClass:[NSString class]]) {
		return [value isEqualToString:@"Minimize"];
	}
	return false;
}

void DisplayServerMacOS::window_set_window_buttons_offset(const Vector2i &p_offset, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	float scale = screen_get_max_scale();
	wd.wb_offset = p_offset / scale;
	wd.wb_offset.x = MAX(wd.wb_offset.x, 12);
	wd.wb_offset.y = MAX(wd.wb_offset.y, 12);
	if (wd.window_button_view) {
		[wd.window_button_view setOffset:NSMakePoint(wd.wb_offset.x, wd.wb_offset.y)];
	}
}

Vector3i DisplayServerMacOS::window_get_safe_title_margins(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Vector3i());
	const WindowData &wd = windows[p_window];

	if (!wd.window_button_view) {
		return Vector3i();
	}

	float scale = screen_get_max_scale();
	float max_x = [wd.window_button_view getOffset].x + [wd.window_button_view frame].size.width;
	float max_y = [wd.window_button_view getOffset].y + [wd.window_button_view frame].size.height;

	if ([wd.window_object windowTitlebarLayoutDirection] == NSUserInterfaceLayoutDirectionRightToLeft) {
		return Vector3i(0, max_x * scale, max_y * scale);
	} else {
		return Vector3i(max_x * scale, 0, max_y * scale);
	}
}

void DisplayServerMacOS::window_set_custom_window_buttons(WindowData &p_wd, bool p_enabled) {
	if (p_wd.window_button_view) {
		[p_wd.window_button_view removeFromSuperview];
		p_wd.window_button_view = nil;
	}
	if (p_enabled) {
		float cb_frame = NSMinX([[p_wd.window_object standardWindowButton:NSWindowCloseButton] frame]);
		float mb_frame = NSMinX([[p_wd.window_object standardWindowButton:NSWindowMiniaturizeButton] frame]);
		bool is_rtl = ([p_wd.window_object windowTitlebarLayoutDirection] == NSUserInterfaceLayoutDirectionRightToLeft);

		float window_buttons_spacing = (is_rtl) ? (cb_frame - mb_frame) : (mb_frame - cb_frame);

		[p_wd.window_object setTitleVisibility:NSWindowTitleHidden];
		[[p_wd.window_object standardWindowButton:NSWindowZoomButton] setHidden:YES];
		[[p_wd.window_object standardWindowButton:NSWindowMiniaturizeButton] setHidden:YES];
		[[p_wd.window_object standardWindowButton:NSWindowCloseButton] setHidden:YES];

		p_wd.window_button_view = [[GodotButtonView alloc] initWithFrame:NSZeroRect];
		[p_wd.window_button_view initButtons:window_buttons_spacing offset:NSMakePoint(p_wd.wb_offset.x, p_wd.wb_offset.y) rtl:is_rtl];
		[p_wd.window_view addSubview:p_wd.window_button_view];
	} else {
		[p_wd.window_object setTitleVisibility:NSWindowTitleVisible];
		[[p_wd.window_object standardWindowButton:NSWindowZoomButton] setHidden:NO];
		[[p_wd.window_object standardWindowButton:NSWindowMiniaturizeButton] setHidden:NO];
		[[p_wd.window_object standardWindowButton:NSWindowCloseButton] setHidden:NO];
	}
}

void DisplayServerMacOS::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			wd.resize_disabled = p_enabled;
			if (wd.fullscreen) { // Fullscreen window should be resizable, style will be applied on exiting fullscreen.
				return;
			}
			if (p_enabled) {
				[wd.window_object setStyleMask:[wd.window_object styleMask] & ~NSWindowStyleMaskResizable];
				[[wd.window_object standardWindowButton:NSWindowZoomButton] setEnabled:NO];
			} else {
				[wd.window_object setStyleMask:[wd.window_object styleMask] | NSWindowStyleMaskResizable];
				[[wd.window_object standardWindowButton:NSWindowZoomButton] setEnabled:YES];
			}
		} break;
		case WINDOW_FLAG_EXTEND_TO_TITLE: {
			NSRect rect = [wd.window_object frame];
			wd.extend_to_title = p_enabled;
			if (p_enabled) {
				[wd.window_object setTitlebarAppearsTransparent:YES];
				[wd.window_object setStyleMask:[wd.window_object styleMask] | NSWindowStyleMaskFullSizeContentView];

				if (!wd.fullscreen) {
					window_set_custom_window_buttons(wd, true);
				}
			} else {
				[wd.window_object setTitlebarAppearsTransparent:NO];
				[wd.window_object setStyleMask:[wd.window_object styleMask] & ~NSWindowStyleMaskFullSizeContentView];

				if (!wd.fullscreen) {
					window_set_custom_window_buttons(wd, false);
				}
			}
			[wd.window_object setFrame:rect display:YES];
			send_window_event(wd, DisplayServerMacOS::WINDOW_EVENT_TITLEBAR_CHANGE);
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			if (wd.fullscreen) {
				return;
			}
			// OrderOut prevents a lose focus bug with the window.
			bool was_visible = false;
			if ([wd.window_object isVisible]) {
				was_visible = true;
				[wd.window_object orderOut:nil];
			}
			wd.borderless = p_enabled;
			if (p_enabled) {
				[wd.window_object setStyleMask:NSWindowStyleMaskBorderless];
			} else {
				if (wd.layered_window) {
					wd.layered_window = false;
					set_window_per_pixel_transparency_enabled(false, p_window);
				}
				[wd.window_object setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (wd.extend_to_title ? NSWindowStyleMaskFullSizeContentView : 0) | (wd.resize_disabled ? 0 : NSWindowStyleMaskResizable)];
				// Force update of the window styles.
				NSRect frameRect = [wd.window_object frame];
				[wd.window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
				[wd.window_object setFrame:frameRect display:NO];
			}
			_update_window_style(wd);
			if (was_visible || [wd.window_object isVisible]) {
				if ([wd.window_object isMiniaturized]) {
					return;
				} else if (wd.no_focus) {
					[wd.window_object orderFront:nil];
				} else {
					[wd.window_object makeKeyAndOrderFront:nil];
				}
			}
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			wd.on_top = p_enabled;
			if (wd.fullscreen) {
				return;
			}
			if (p_enabled) {
				[(NSWindow *)wd.window_object setLevel:NSFloatingWindowLevel];
			} else {
				[(NSWindow *)wd.window_object setLevel:NSNormalWindowLevel];
			}
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			if (wd.fullscreen) {
				return;
			}
			if (p_enabled) {
				[wd.window_object setStyleMask:NSWindowStyleMaskBorderless]; // Force borderless.
			} else if (!wd.borderless) {
				[wd.window_object setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (wd.extend_to_title ? NSWindowStyleMaskFullSizeContentView : 0) | (wd.resize_disabled ? 0 : NSWindowStyleMaskResizable)];
			}
			wd.layered_window = p_enabled;
			set_window_per_pixel_transparency_enabled(p_enabled, p_window);
			// Force update of the window styles.
			NSRect frameRect = [wd.window_object frame];
			[wd.window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
			[wd.window_object setFrame:frameRect display:NO];
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			wd.no_focus = p_enabled;
		} break;
		case WINDOW_FLAG_MOUSE_PASSTHROUGH: {
			wd.mpass = p_enabled;
		} break;
		case WINDOW_FLAG_POPUP: {
			ERR_FAIL_COND_MSG(p_window == MAIN_WINDOW_ID, "Main window can't be popup.");
			ERR_FAIL_COND_MSG([wd.window_object isVisible] && (wd.is_popup != p_enabled), "Popup flag can't changed while window is opened.");
			wd.is_popup = p_enabled;
		} break;
		default: {
		}
	}
}

bool DisplayServerMacOS::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			return wd.resize_disabled;
		} break;
		case WINDOW_FLAG_EXTEND_TO_TITLE: {
			return [wd.window_object styleMask] & NSWindowStyleMaskFullSizeContentView;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			return [wd.window_object styleMask] == NSWindowStyleMaskBorderless;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			if (wd.fullscreen) {
				return wd.on_top;
			} else {
				return [(NSWindow *)wd.window_object level] == NSFloatingWindowLevel;
			}
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			return wd.layered_window;
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			return wd.no_focus;
		} break;
		case WINDOW_FLAG_MOUSE_PASSTHROUGH: {
			return wd.mpass;
		} break;
		case WINDOW_FLAG_POPUP: {
			return wd.is_popup;
		} break;
		default: {
		}
	}

	return false;
}

void DisplayServerMacOS::window_request_attention(WindowID p_window) {
	// It's app global, ignore window id.
	[NSApp requestUserAttention:NSCriticalRequest];
}

void DisplayServerMacOS::window_move_to_foreground(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	const WindowData &wd = windows[p_window];

	[[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
	if (wd.no_focus || wd.is_popup) {
		[wd.window_object orderFront:nil];
	} else {
		[wd.window_object makeKeyAndOrderFront:nil];
	}
}

bool DisplayServerMacOS::window_is_focused(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	return wd.focused;
}

bool DisplayServerMacOS::window_can_draw(WindowID p_window) const {
	return windows[p_window].is_visible;
}

bool DisplayServerMacOS::can_any_window_draw() const {
	_THREAD_SAFE_METHOD_

	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (E.value.is_visible) {
			return true;
		}
	}
	return false;
}

void DisplayServerMacOS::window_set_ime_active(const bool p_active, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_active = p_active;

	if (!p_active) {
		[wd.window_view cancelComposition];
	}
}

void DisplayServerMacOS::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_position = p_pos;
}

DisplayServer::WindowID DisplayServerMacOS::get_window_at_screen_position(const Point2i &p_position) const {
	Point2i position = p_position;
	position.y *= -1;
	position += _get_screens_origin();
	position /= screen_get_max_scale();

	NSInteger wnum = [NSWindow windowNumberAtPoint:NSMakePoint(position.x, position.y) belowWindowWithWindowNumber:0 /*topmost*/];
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if ([E.value.window_object windowNumber] == wnum) {
			return E.key;
		}
	}
	return INVALID_WINDOW_ID;
}

int64_t DisplayServerMacOS::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	ERR_FAIL_COND_V(!windows.has(p_window), 0);
	switch (p_handle_type) {
		case DISPLAY_HANDLE: {
			return 0; // Not supported.
		}
		case WINDOW_HANDLE: {
			return (int64_t)windows[p_window].window_object;
		}
		case WINDOW_VIEW: {
			return (int64_t)windows[p_window].window_view;
		}
#ifdef GLES3_ENABLED
		case OPENGL_CONTEXT: {
			if (gl_manager_legacy) {
				return (int64_t)gl_manager_legacy->get_context(p_window);
			}
			if (gl_manager_angle) {
				return (int64_t)gl_manager_angle->get_context(p_window);
			}
			return 0;
		}
#endif
		default: {
			return 0;
		}
	}
}

void DisplayServerMacOS::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].instance_id = p_instance;
}

ObjectID DisplayServerMacOS::window_get_attached_instance_id(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), ObjectID());
	return windows[p_window].instance_id;
}

void DisplayServerMacOS::gl_window_make_current(DisplayServer::WindowID p_window_id) {
#if defined(GLES3_ENABLED)
	if (gl_manager_legacy) {
		gl_manager_legacy->window_make_current(p_window_id);
	}
	if (gl_manager_angle) {
		gl_manager_angle->window_make_current(p_window_id);
	}
#endif
}

void DisplayServerMacOS::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(GLES3_ENABLED)
	if (gl_manager_angle) {
		gl_manager_angle->set_use_vsync(p_vsync_mode != DisplayServer::VSYNC_DISABLED);
	}
	if (gl_manager_legacy) {
		gl_manager_legacy->set_use_vsync(p_vsync_mode != DisplayServer::VSYNC_DISABLED);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerMacOS::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(GLES3_ENABLED)
	if (gl_manager_angle) {
		return (gl_manager_angle->is_using_vsync() ? DisplayServer::VSyncMode::VSYNC_ENABLED : DisplayServer::VSyncMode::VSYNC_DISABLED);
	}
	if (gl_manager_legacy) {
		return (gl_manager_legacy->is_using_vsync() ? DisplayServer::VSyncMode::VSYNC_ENABLED : DisplayServer::VSyncMode::VSYNC_DISABLED);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		return context_vulkan->get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}

Point2i DisplayServerMacOS::ime_get_selection() const {
	return im_selection;
}

String DisplayServerMacOS::ime_get_text() const {
	return im_text;
}

void DisplayServerMacOS::cursor_update_shape() {
	_THREAD_SAFE_METHOD_

	if (cursors[cursor_shape] != nullptr) {
		[cursors[cursor_shape] set];
	} else {
		switch (cursor_shape) {
			case CURSOR_ARROW:
				[[NSCursor arrowCursor] set];
				break;
			case CURSOR_IBEAM:
				[[NSCursor IBeamCursor] set];
				break;
			case CURSOR_POINTING_HAND:
				[[NSCursor pointingHandCursor] set];
				break;
			case CURSOR_CROSS:
				[[NSCursor crosshairCursor] set];
				break;
			case CURSOR_WAIT:
				[[NSCursor arrowCursor] set];
				break;
			case CURSOR_BUSY:
				[[NSCursor arrowCursor] set];
				break;
			case CURSOR_DRAG:
				[[NSCursor closedHandCursor] set];
				break;
			case CURSOR_CAN_DROP:
				[[NSCursor openHandCursor] set];
				break;
			case CURSOR_FORBIDDEN:
				[[NSCursor operationNotAllowedCursor] set];
				break;
			case CURSOR_VSIZE:
				[_cursor_from_selector(@selector(_windowResizeNorthSouthCursor), @selector(resizeUpDownCursor)) set];
				break;
			case CURSOR_HSIZE:
				[_cursor_from_selector(@selector(_windowResizeEastWestCursor), @selector(resizeLeftRightCursor)) set];
				break;
			case CURSOR_BDIAGSIZE:
				[_cursor_from_selector(@selector(_windowResizeNorthEastSouthWestCursor)) set];
				break;
			case CURSOR_FDIAGSIZE:
				[_cursor_from_selector(@selector(_windowResizeNorthWestSouthEastCursor)) set];
				break;
			case CURSOR_MOVE:
				[[NSCursor arrowCursor] set];
				break;
			case CURSOR_VSPLIT:
				[[NSCursor resizeUpDownCursor] set];
				break;
			case CURSOR_HSPLIT:
				[[NSCursor resizeLeftRightCursor] set];
				break;
			case CURSOR_HELP:
				[_cursor_from_selector(@selector(_helpCursor)) set];
				break;
			default: {
			}
		}
	}
}

void DisplayServerMacOS::cursor_set_shape(CursorShape p_shape) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (cursor_shape == p_shape) {
		return;
	}

	cursor_shape = p_shape;

	if (mouse_mode != MOUSE_MODE_VISIBLE && mouse_mode != MOUSE_MODE_CONFINED) {
		return;
	}

	cursor_update_shape();
}

DisplayServerMacOS::CursorShape DisplayServerMacOS::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerMacOS::cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (p_cursor.is_valid()) {
		HashMap<CursorShape, Vector<Variant>>::Iterator cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->value[0] == p_cursor && cursor_c->value[1] == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}
			cursors_cache.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		ERR_FAIL_COND(!texture.is_valid());
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (atlas_texture.is_valid()) {
			texture = atlas_texture->get_atlas();

			atlas_rect.size.width = texture->get_width();
			atlas_rect.size.height = texture->get_height();
			atlas_rect.position.x = atlas_texture->get_region().position.x;
			atlas_rect.position.y = atlas_texture->get_region().position.y;

			texture_size.width = atlas_texture->get_region().size.x;
			texture_size.height = atlas_texture->get_region().size.y;
		} else {
			texture_size.width = texture->get_width();
			texture_size.height = texture->get_height();
		}

		ERR_FAIL_COND(p_hotspot.x < 0 || p_hotspot.y < 0);
		ERR_FAIL_COND(texture_size.width > 256 || texture_size.height > 256);
		ERR_FAIL_COND(p_hotspot.x > texture_size.width || p_hotspot.y > texture_size.height);

		Ref<Image> image = texture->get_image();

		ERR_FAIL_COND(!image.is_valid());
		if (image->is_compressed()) {
			image = image->duplicate(true);
			Error err = image->decompress();
			ERR_FAIL_COND_MSG(err != OK, "Couldn't decompress VRAM-compressed custom mouse cursor image. Switch to a lossless compression mode in the Import dock.");
		}

		NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc]
				initWithBitmapDataPlanes:nullptr
							  pixelsWide:int(texture_size.width)
							  pixelsHigh:int(texture_size.height)
						   bitsPerSample:8
						 samplesPerPixel:4
								hasAlpha:YES
								isPlanar:NO
						  colorSpaceName:NSDeviceRGBColorSpace
							 bytesPerRow:int(texture_size.width) * 4
							bitsPerPixel:32];

		ERR_FAIL_NULL(imgrep);
		uint8_t *pixels = [imgrep bitmapData];

		int len = int(texture_size.width * texture_size.height);

		for (int i = 0; i < len; i++) {
			int row_index = floor(i / texture_size.width) + atlas_rect.position.y;
			int column_index = (i % int(texture_size.width)) + atlas_rect.position.x;

			if (atlas_texture.is_valid()) {
				column_index = MIN(column_index, atlas_rect.size.width - 1);
				row_index = MIN(row_index, atlas_rect.size.height - 1);
			}

			uint32_t color = image->get_pixel(column_index, row_index).to_argb32();

			uint8_t alpha = (color >> 24) & 0xFF;
			pixels[i * 4 + 0] = ((color >> 16) & 0xFF) * alpha / 255;
			pixels[i * 4 + 1] = ((color >> 8) & 0xFF) * alpha / 255;
			pixels[i * 4 + 2] = ((color)&0xFF) * alpha / 255;
			pixels[i * 4 + 3] = alpha;
		}

		NSImage *nsimage = [[NSImage alloc] initWithSize:NSMakeSize(texture_size.width, texture_size.height)];
		[nsimage addRepresentation:imgrep];

		NSCursor *cursor = [[NSCursor alloc] initWithImage:nsimage hotSpot:NSMakePoint(p_hotspot.x, p_hotspot.y)];

		cursors[p_shape] = cursor;

		Vector<Variant> params;
		params.push_back(p_cursor);
		params.push_back(p_hotspot);
		cursors_cache.insert(p_shape, params);

		if (p_shape == cursor_shape) {
			if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
				[cursor set];
			}
		}
	} else {
		// Reset to default system cursor.
		if (cursors[p_shape] != nullptr) {
			cursors[p_shape] = nullptr;
		}

		cursors_cache.erase(p_shape);

		cursor_update_shape();
	}
}

bool DisplayServerMacOS::get_swap_cancel_ok() {
	return false;
}

int DisplayServerMacOS::keyboard_get_layout_count() const {
	if (keyboard_layout_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_keyboard_layouts();
	}
	return kbd_layouts.size();
}

void DisplayServerMacOS::keyboard_set_current_layout(int p_index) {
	if (keyboard_layout_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_keyboard_layouts();
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

int DisplayServerMacOS::keyboard_get_current_layout() const {
	if (keyboard_layout_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_keyboard_layouts();
	}

	return current_layout;
}

String DisplayServerMacOS::keyboard_get_layout_language(int p_index) const {
	if (keyboard_layout_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].code;
}

String DisplayServerMacOS::keyboard_get_layout_name(int p_index) const {
	if (keyboard_layout_dirty) {
		const_cast<DisplayServerMacOS *>(this)->_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].name;
}

Key DisplayServerMacOS::keyboard_get_keycode_from_physical(Key p_keycode) const {
	if (p_keycode == Key::PAUSE || p_keycode == Key::NONE) {
		return p_keycode;
	}

	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = p_keycode & KeyModifierMask::CODE_MASK;
	unsigned int macos_keycode = KeyMappingMacOS::unmap_key(keycode_no_mod);
	return (Key)(KeyMappingMacOS::remap_key(macos_keycode, 0, false) | modifiers);
}

Key DisplayServerMacOS::keyboard_get_label_from_physical(Key p_keycode) const {
	if (p_keycode == Key::PAUSE || p_keycode == Key::NONE) {
		return p_keycode;
	}

	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = p_keycode & KeyModifierMask::CODE_MASK;
	unsigned int macos_keycode = KeyMappingMacOS::unmap_key(keycode_no_mod);
	return (Key)(KeyMappingMacOS::remap_key(macos_keycode, 0, true) | modifiers);
}

void DisplayServerMacOS::process_events() {
	_THREAD_SAFE_METHOD_

	while (true) {
		NSEvent *event = [NSApp
				nextEventMatchingMask:NSEventMaskAny
							untilDate:[NSDate distantPast]
							   inMode:NSDefaultRunLoopMode
							  dequeue:YES];

		if (event == nil) {
			break;
		}

		[NSApp sendEvent:event];
	}

	// Process "menu_callback"s.
	while (List<MenuCall>::Element *call_p = deferred_menu_calls.front()) {
		MenuCall call = call_p->get();
		deferred_menu_calls.pop_front(); // Remove before call to avoid infinite loop in case callback is using `process_events` (e.g. EditorProgress).

		Variant ret;
		Callable::CallError ce;
		const Variant *args[1] = { &call.tag };

		call.callback.callp(args, 1, ret, ce);
		if (ce.error != Callable::CallError::CALL_OK) {
			ERR_PRINT(vformat("Failed to execute menu callback: %s.", Variant::get_callable_error_text(call.callback, args, 1, ce)));
		}
	}

	if (!drop_events) {
		_process_key_events();
		Input::get_singleton()->flush_buffered_events();
	}

	for (KeyValue<WindowID, WindowData> &E : windows) {
		WindowData &wd = E.value;
		if (wd.mpass) {
			if (![wd.window_object ignoresMouseEvents]) {
				[wd.window_object setIgnoresMouseEvents:YES];
			}
		} else if (wd.mpath.size() > 0) {
			update_mouse_pos(wd, [wd.window_object mouseLocationOutsideOfEventStream]);
			if (Geometry2D::is_point_in_polygon(wd.mouse_pos, wd.mpath)) {
				if ([wd.window_object ignoresMouseEvents]) {
					[wd.window_object setIgnoresMouseEvents:NO];
				}
			} else {
				if (![wd.window_object ignoresMouseEvents]) {
					[wd.window_object setIgnoresMouseEvents:YES];
				}
			}
		} else {
			if ([wd.window_object ignoresMouseEvents]) {
				[wd.window_object setIgnoresMouseEvents:NO];
			}
		}
	}
}

void DisplayServerMacOS::force_process_and_drop_events() {
	_THREAD_SAFE_METHOD_

	drop_events = true;
	process_events();
	drop_events = false;
}

void DisplayServerMacOS::release_rendering_thread() {
}

void DisplayServerMacOS::make_rendering_thread() {
}

void DisplayServerMacOS::swap_buffers() {
#if defined(GLES3_ENABLED)
	if (gl_manager_angle) {
		gl_manager_angle->swap_buffers();
	}
	if (gl_manager_legacy) {
		gl_manager_legacy->swap_buffers();
	}
#endif
}

void DisplayServerMacOS::set_native_icon(const String &p_filename) {
	_THREAD_SAFE_METHOD_

	Ref<FileAccess> f = FileAccess::open(p_filename, FileAccess::READ);
	ERR_FAIL_COND(f.is_null());

	Vector<uint8_t> data;
	uint64_t len = f->get_length();
	ERR_FAIL_COND_MSG(len < 8, "Error reading icon data."); // "icns" + 32-bit length

	data.resize(len);
	f->get_buffer((uint8_t *)&data.write[0], len);

	@try {
		NSData *icon_data = [[NSData alloc] initWithBytes:&data.write[0] length:len];
		ERR_FAIL_NULL_MSG(icon_data, "Error reading icon data.");

		NSImage *icon = [[NSImage alloc] initWithData:icon_data];
		ERR_FAIL_NULL_MSG(icon, "Error loading icon.");

		[NSApp setApplicationIconImage:icon];
	} @catch (NSException *exception) {
		ERR_FAIL_MSG("NSException: " + String::utf8([exception reason].UTF8String));
	}
}

void DisplayServerMacOS::set_icon(const Ref<Image> &p_icon) {
	_THREAD_SAFE_METHOD_

	if (p_icon.is_valid()) {
		ERR_FAIL_COND(p_icon->get_width() <= 0 || p_icon->get_height() <= 0);

		Ref<Image> img = p_icon->duplicate();
		img->convert(Image::FORMAT_RGBA8);

		NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc]
				initWithBitmapDataPlanes:nullptr
							  pixelsWide:img->get_width()
							  pixelsHigh:img->get_height()
						   bitsPerSample:8
						 samplesPerPixel:4
								hasAlpha:YES
								isPlanar:NO
						  colorSpaceName:NSDeviceRGBColorSpace
							 bytesPerRow:img->get_width() * 4
							bitsPerPixel:32];
		ERR_FAIL_NULL(imgrep);
		uint8_t *pixels = [imgrep bitmapData];

		int len = img->get_width() * img->get_height();
		const uint8_t *r = img->get_data().ptr();

		/* Premultiply the alpha channel */
		for (int i = 0; i < len; i++) {
			uint8_t alpha = r[i * 4 + 3];
			pixels[i * 4 + 0] = (uint8_t)(((uint16_t)r[i * 4 + 0] * alpha) / 255);
			pixels[i * 4 + 1] = (uint8_t)(((uint16_t)r[i * 4 + 1] * alpha) / 255);
			pixels[i * 4 + 2] = (uint8_t)(((uint16_t)r[i * 4 + 2] * alpha) / 255);
			pixels[i * 4 + 3] = alpha;
		}

		NSImage *nsimg = [[NSImage alloc] initWithSize:NSMakeSize(img->get_width(), img->get_height())];
		ERR_FAIL_NULL(nsimg);

		[nsimg addRepresentation:imgrep];
		[NSApp setApplicationIconImage:nsimg];
	} else {
		[NSApp setApplicationIconImage:nil];
	}
}

DisplayServer *DisplayServerMacOS::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerMacOS(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, r_error));
	if (r_error != OK) {
		if (p_rendering_driver == "vulkan") {
			String executable_command;
			if (OS::get_singleton()->get_bundle_resource_dir() == OS::get_singleton()->get_executable_path().get_base_dir()) {
				executable_command = vformat("\"%s\" --rendering-driver opengl3", OS::get_singleton()->get_executable_path());
			} else {
				executable_command = vformat("open \"%s\" --args --rendering-driver opengl3", OS::get_singleton()->get_bundle_resource_dir().path_join("../..").simplify_path());
			}
			OS::get_singleton()->alert(
					vformat("Your video card drivers seem not to support the required Vulkan version.\n\n"
							"If possible, consider updating your macOS version or using the OpenGL 3 driver.\n\n"
							"You can enable the OpenGL 3 driver by starting the engine from the\n"
							"command line with the command:\n\n    %s",
							executable_command),
					"Unable to initialize Vulkan video driver");
		} else {
			OS::get_singleton()->alert(
					"Your video card drivers seem not to support the required OpenGL 3.3 version.\n\n"
					"If possible, consider updating your macOS version.",
					"Unable to initialize OpenGL video driver");
		}
	}
	return ds;
}

Vector<String> DisplayServerMacOS::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl3");
	drivers.push_back("opengl3_angle");
#endif

	return drivers;
}

void DisplayServerMacOS::register_macos_driver() {
	register_create_function("macos", create_func, get_rendering_drivers_func);
}

DisplayServer::WindowID DisplayServerMacOS::window_get_active_popup() const {
	const List<WindowID>::Element *E = popup_list.back();
	if (E) {
		return E->get();
	} else {
		return INVALID_WINDOW_ID;
	}
}

void DisplayServerMacOS::window_set_popup_safe_rect(WindowID p_window, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.parent_safe_rect = p_rect;
}

Rect2i DisplayServerMacOS::window_get_popup_safe_rect(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Rect2i());
	const WindowData &wd = windows[p_window];
	return wd.parent_safe_rect;
}

void DisplayServerMacOS::popup_open(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	bool has_popup_ancestor = false;
	WindowID transient_root = p_window;
	while (true) {
		WindowID parent = windows[transient_root].transient_parent;
		if (parent == INVALID_WINDOW_ID) {
			break;
		} else {
			transient_root = parent;
			if (windows[parent].is_popup) {
				has_popup_ancestor = true;
				break;
			}
		}
	}

	WindowData &wd = windows[p_window];
	if (wd.is_popup || has_popup_ancestor) {
		bool was_empty = popup_list.is_empty();
		// Find current popup parent, or root popup if new window is not transient.
		List<WindowID>::Element *C = nullptr;
		List<WindowID>::Element *E = popup_list.back();
		while (E) {
			if (wd.transient_parent != E->get() || wd.transient_parent == INVALID_WINDOW_ID) {
				C = E;
				E = E->prev();
			} else {
				break;
			}
		}
		if (C) {
			send_window_event(windows[C->get()], DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
		}

		if (was_empty && popup_list.is_empty()) {
			// Inform OS that popup was opened, to close other native popups.
			[[NSDistributedNotificationCenter defaultCenter] postNotificationName:@"com.apple.HIToolbox.beginMenuTrackingNotification" object:@"org.godotengine.godot.popup_window"];
		}
		time_since_popup = OS::get_singleton()->get_ticks_msec();
		popup_list.push_back(p_window);
	}
}

void DisplayServerMacOS::popup_close(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	bool was_empty = popup_list.is_empty();
	List<WindowID>::Element *E = popup_list.find(p_window);
	while (E) {
		List<WindowID>::Element *F = E->next();
		WindowID win_id = E->get();
		popup_list.erase(E);

		send_window_event(windows[win_id], DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
		E = F;
	}
	if (!was_empty && popup_list.is_empty()) {
		// Inform OS that all popups are closed.
		[[NSDistributedNotificationCenter defaultCenter] postNotificationName:@"com.apple.HIToolbox.endMenuTrackingNotification" object:@"org.godotengine.godot.popup_window"];
	}
}

bool DisplayServerMacOS::mouse_process_popups(bool p_close) {
	_THREAD_SAFE_METHOD_

	bool was_empty = popup_list.is_empty();
	bool closed = false;
	if (p_close) {
		// Close all popups.
		List<WindowID>::Element *E = popup_list.front();
		if (E) {
			send_window_event(windows[E->get()], DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
			closed = true;
		}
		if (!was_empty) {
			// Inform OS that all popups are closed.
			[[NSDistributedNotificationCenter defaultCenter] postNotificationName:@"com.apple.HIToolbox.endMenuTrackingNotification" object:@"org.godotengine.godot.popup_window"];
		}
	} else {
		uint64_t delta = OS::get_singleton()->get_ticks_msec() - time_since_popup;
		if (delta < 250) {
			return false;
		}

		Point2i pos = mouse_get_position();
		List<WindowID>::Element *C = nullptr;
		List<WindowID>::Element *E = popup_list.back();
		// Find top popup to close.
		while (E) {
			// Popup window area.
			Rect2i win_rect = Rect2i(window_get_position_with_decorations(E->get()), window_get_size_with_decorations(E->get()));
			// Area of the parent window, which responsible for opening sub-menu.
			Rect2i safe_rect = window_get_popup_safe_rect(E->get());
			if (win_rect.has_point(pos)) {
				break;
			} else if (safe_rect != Rect2i() && safe_rect.has_point(pos)) {
				break;
			} else {
				C = E;
				E = E->prev();
			}
		}
		if (C) {
			send_window_event(windows[C->get()], DisplayServerMacOS::WINDOW_EVENT_CLOSE_REQUEST);
			closed = true;
		}
		if (!was_empty && popup_list.is_empty()) {
			// Inform OS that all popups are closed.
			[[NSDistributedNotificationCenter defaultCenter] postNotificationName:@"com.apple.HIToolbox.endMenuTrackingNotification" object:@"org.godotengine.godot.popup_window"];
		}
	}
	return closed;
}

DisplayServerMacOS::DisplayServerMacOS(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Error &r_error) {
	KeyMappingMacOS::initialize();

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;

	memset(cursors, 0, sizeof(cursors));

	event_source = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
	ERR_FAIL_COND(!event_source);

	CGEventSourceSetLocalEventsSuppressionInterval(event_source, 0.0);

	int screen_count = get_screen_count();
	for (int i = 0; i < screen_count; i++) {
		display_max_scale = fmax(display_max_scale, screen_get_scale(i));
	}

	// Register to be notified on keyboard layout changes.
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
			nullptr, _keyboard_layout_changed,
			kTISNotifySelectedKeyboardInputSourceChanged, nullptr,
			CFNotificationSuspensionBehaviorDeliverImmediately);

	// Register to be notified on displays arrangement changes.
	CGDisplayRegisterReconfigurationCallback(_displays_arrangement_changed, nullptr);

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		tts = [[TTS_MacOS alloc] init];
	}

	NSMenuItem *menu_item;
	NSString *title;

	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname == nil) {
		nsappname = [[NSProcessInfo processInfo] processName];
	}

	menu_delegate = [[GodotMenuDelegate alloc] init];

	// Setup Dock menu.
	dock_menu = [[NSMenu alloc] initWithTitle:@"_dock"];
	[dock_menu setAutoenablesItems:NO];
	[dock_menu setDelegate:menu_delegate];

	// Setup Apple menu.
	apple_menu = [[NSMenu alloc] initWithTitle:@""];
	title = [NSString stringWithFormat:NSLocalizedString(@"About %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(showAbout:) keyEquivalent:@""];
	[apple_menu setAutoenablesItems:NO];
	[apple_menu setDelegate:menu_delegate];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	NSMenu *services = [[NSMenu alloc] initWithTitle:@""];
	menu_item = [apple_menu addItemWithTitle:NSLocalizedString(@"Services", nil) action:nil keyEquivalent:@""];
	[apple_menu setSubmenu:services forItem:menu_item];
	[NSApp setServicesMenu:services];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	title = [NSString stringWithFormat:NSLocalizedString(@"Hide %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(hide:) keyEquivalent:@"h"];

	menu_item = [apple_menu addItemWithTitle:NSLocalizedString(@"Hide Others", nil) action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
	[menu_item setKeyEquivalentModifierMask:(NSEventModifierFlagOption | NSEventModifierFlagCommand)];

	[apple_menu addItemWithTitle:NSLocalizedString(@"Show all", nil) action:@selector(unhideAllApplications:) keyEquivalent:@""];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	title = [NSString stringWithFormat:NSLocalizedString(@"Quit %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(terminate:) keyEquivalent:@"q"];

	// Add items to the menu bar.
	NSMenu *main_menu = [NSApp mainMenu];
	menu_item = [main_menu addItemWithTitle:@"" action:nil keyEquivalent:@""];
	[main_menu setSubmenu:apple_menu forItem:menu_item];
	[main_menu setAutoenablesItems:NO];

	//!!!!!!!!!!!!!!!!!!!!!!!!!!
	//TODO - do Vulkan and OpenGL support checks, driver selection and fallback
	rendering_driver = p_rendering_driver;

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3_angle") {
		gl_manager_angle = memnew(GLManagerANGLE_MacOS);
		if (gl_manager_angle->initialize() != OK || gl_manager_angle->open_display(nullptr) != OK) {
			memdelete(gl_manager_angle);
			gl_manager_angle = nullptr;
			bool fallback = GLOBAL_GET("rendering/gl_compatibility/fallback_to_native");
			if (fallback) {
				WARN_PRINT("Your video card drivers seem not to support the required Metal version, switching to native OpenGL.");
				rendering_driver = "opengl3";
			} else {
				r_error = ERR_UNAVAILABLE;
				ERR_FAIL_MSG("Could not initialize ANGLE OpenGL.");
			}
		}
	}

	if (rendering_driver == "opengl3") {
		gl_manager_legacy = memnew(GLManagerLegacy_MacOS);
		if (gl_manager_legacy->initialize() != OK) {
			memdelete(gl_manager_legacy);
			gl_manager_legacy = nullptr;
			r_error = ERR_UNAVAILABLE;
			ERR_FAIL_MSG("Could not initialize native OpenGL.");
		}
	}
#endif
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextMacOS);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan");
		}
	}
#endif

	Point2i window_position;
	if (p_position != nullptr) {
		window_position = *p_position;
	} else {
		if (p_screen == SCREEN_OF_MAIN_WINDOW) {
			p_screen = SCREEN_PRIMARY;
		}
		window_position = screen_get_position(p_screen) + (screen_get_size(p_screen) - p_resolution) / 2;
	}

	WindowID main_window = _create_window(p_mode, p_vsync_mode, Rect2i(window_position, p_resolution));
	ERR_FAIL_COND(main_window == INVALID_WINDOW_ID);
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, main_window);
		}
	}
	show_window(MAIN_WINDOW_ID);
	force_process_and_drop_events();

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		RasterizerGLES3::make_current(true);
	}
	if (rendering_driver == "opengl3_angle") {
		RasterizerGLES3::make_current(false);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

	screen_set_keep_on(GLOBAL_GET("display/window/energy_saving/keep_screen_on"));
}

DisplayServerMacOS::~DisplayServerMacOS() {
	if (screen_keep_on_assertion) {
		IOPMAssertionRelease(screen_keep_on_assertion);
		screen_keep_on_assertion = kIOPMNullAssertionID;
	}

	// Destroy all windows.
	for (HashMap<WindowID, WindowData>::Iterator E = windows.begin(); E;) {
		HashMap<WindowID, WindowData>::Iterator F = E;
		++E;
		[F->value.window_object setContentView:nil];
		[F->value.window_object close];
	}

	// Destroy drivers.
#if defined(GLES3_ENABLED)
	if (gl_manager_legacy) {
		memdelete(gl_manager_legacy);
		gl_manager_legacy = nullptr;
	}
	if (gl_manager_angle) {
		memdelete(gl_manager_angle);
		gl_manager_angle = nullptr;
	}
#endif
#if defined(VULKAN_ENABLED)
	if (rendering_device_vulkan) {
		rendering_device_vulkan->finalize();
		memdelete(rendering_device_vulkan);
		rendering_device_vulkan = nullptr;
	}

	if (context_vulkan) {
		memdelete(context_vulkan);
		context_vulkan = nullptr;
	}
#endif

	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), nullptr, kTISNotifySelectedKeyboardInputSourceChanged, nullptr);
	CGDisplayRemoveReconfigurationCallback(_displays_arrangement_changed, nullptr);

	cursors_cache.clear();
}
