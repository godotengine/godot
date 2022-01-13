/*************************************************************************/
/*  display_server_osx.mm                                                */
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

#include "display_server_osx.h"

#include "os_osx.h"

#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "main/main.h"
#include "scene/resources/texture.h"

#include <Carbon/Carbon.h>
#include <Cocoa/Cocoa.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hid/IOHIDLib.h>

#if defined(GLES3_ENABLED)
#include "drivers/gles3/rasterizer_gles3.h"

#import <AppKit/NSOpenGLView.h>
#endif

#if defined(VULKAN_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"

#include <QuartzCore/CAMetalLayer.h>
#endif

#ifndef NSAppKitVersionNumber10_14
#define NSAppKitVersionNumber10_14 1671
#endif

#define DS_OSX ((DisplayServerOSX *)(DisplayServerOSX::get_singleton()))

static bool ignore_momentum_scroll = false;

static void _get_key_modifier_state(unsigned int p_osx_state, Ref<InputEventWithModifiers> r_state) {
	r_state->set_shift_pressed((p_osx_state & NSEventModifierFlagShift));
	r_state->set_ctrl_pressed((p_osx_state & NSEventModifierFlagControl));
	r_state->set_alt_pressed((p_osx_state & NSEventModifierFlagOption));
	r_state->set_meta_pressed((p_osx_state & NSEventModifierFlagCommand));
}

static Vector2i _get_mouse_pos(DisplayServerOSX::WindowData &p_wd, NSPoint p_locationInWindow) {
	const NSRect contentRect = [p_wd.window_view frame];
	const float scale = DS_OSX->screen_get_max_scale();
	p_wd.mouse_pos.x = p_locationInWindow.x * scale;
	p_wd.mouse_pos.y = (contentRect.size.height - p_locationInWindow.y) * scale;
	DS_OSX->last_mouse_pos = p_wd.mouse_pos;
	Input::get_singleton()->set_mouse_position(p_wd.mouse_pos);
	return p_wd.mouse_pos;
}

static void _push_to_key_event_buffer(const DisplayServerOSX::KeyEvent &p_event) {
	Vector<DisplayServerOSX::KeyEvent> &buffer = DS_OSX->key_event_buffer;
	if (DS_OSX->key_event_pos >= buffer.size()) {
		buffer.resize(1 + DS_OSX->key_event_pos);
	}
	buffer.write[DS_OSX->key_event_pos++] = p_event;
}

static NSCursor *_cursorFromSelector(SEL selector, SEL fallback = nil) {
	if ([NSCursor respondsToSelector:selector]) {
		id object = [NSCursor performSelector:selector];
		if ([object isKindOfClass:[NSCursor class]]) {
			return object;
		}
	}
	if (fallback) {
		// Fallback should be a reasonable default, no need to check.
		return [NSCursor performSelector:fallback];
	}
	return [NSCursor arrowCursor];
}

/*************************************************************************/
/* GlobalMenuItem                                                       */
/*************************************************************************/

@interface GlobalMenuItem : NSObject {
@public
	Callable callback;
	Variant meta;
	bool checkable;
}

@end

@implementation GlobalMenuItem
@end

/*************************************************************************/
/* GodotWindowDelegate                                                   */
/*************************************************************************/

@interface GodotWindowDelegate : NSObject {
	DisplayServerOSX::WindowID window_id;
}

- (void)windowWillClose:(NSNotification *)notification;
- (void)setWindowID:(DisplayServerOSX::WindowID)wid;

@end

@implementation GodotWindowDelegate

- (void)setWindowID:(DisplayServerOSX::WindowID)wid {
	window_id = wid;
}

- (BOOL)windowShouldClose:(id)sender {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return YES;
	}
	DS_OSX->_send_window_event(DS_OSX->windows[window_id], DisplayServerOSX::WINDOW_EVENT_CLOSE_REQUEST);
	return NO;
}

- (void)windowWillClose:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	while (wd.transient_children.size()) {
		DS_OSX->window_set_transient(wd.transient_children.front()->get(), DisplayServerOSX::INVALID_WINDOW_ID);
	}

	if (wd.transient_parent != DisplayServerOSX::INVALID_WINDOW_ID) {
		DisplayServerOSX::WindowData &pwd = DS_OSX->windows[wd.transient_parent];
		[pwd.window_object makeKeyAndOrderFront:nil]; // Move focus back to parent.
		DS_OSX->window_set_transient(window_id, DisplayServerOSX::INVALID_WINDOW_ID);
	} else if ((window_id != DisplayServerOSX::MAIN_WINDOW_ID) && (DS_OSX->windows.size() == 1)) {
		DisplayServerOSX::WindowData &pwd = DS_OSX->windows[DisplayServerOSX::MAIN_WINDOW_ID];
		[pwd.window_object makeKeyAndOrderFront:nil]; // Move focus back to main window if there is no parent or other windows left.
	}

#if defined(GLES3_ENABLED)
	if (DS_OSX->gl_manager) {
		DS_OSX->gl_manager->window_destroy(window_id);
	}
#endif
#ifdef VULKAN_ENABLED
	if (DS_OSX->context_vulkan) {
		DS_OSX->context_vulkan->window_destroy(window_id);
	}
#endif

	DS_OSX->windows.erase(window_id);
}

- (void)windowDidEnterFullScreen:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	wd.fullscreen = true;

	[wd.window_object setContentMinSize:NSMakeSize(0, 0)];
	[wd.window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
	// Force window resize event.
	[self windowDidResize:notification];
}

- (void)windowDidExitFullScreen:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	wd.fullscreen = false;

	const float scale = DS_OSX->screen_get_max_scale();
	if (wd.min_size != Size2i()) {
		Size2i size = wd.min_size / scale;
		[wd.window_object setContentMinSize:NSMakeSize(size.x, size.y)];
	}
	if (wd.max_size != Size2i()) {
		Size2i size = wd.max_size / scale;
		[wd.window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
	}

	if (wd.resize_disabled) {
		[wd.window_object setStyleMask:[wd.window_object styleMask] & ~NSWindowStyleMaskResizable];
	}

	if (wd.on_top) {
		[wd.window_object setLevel:NSFloatingWindowLevel];
	}
	// Force window resize event.
	[self windowDidResize:notification];
}

- (void)windowDidChangeBackingProperties:(NSNotification *)notification {
	if (!DisplayServerOSX::get_singleton()) {
		return;
	}

	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	CGFloat newBackingScaleFactor = [wd.window_object backingScaleFactor];
	CGFloat oldBackingScaleFactor = [[[notification userInfo] objectForKey:@"NSBackingPropertyOldScaleFactorKey"] doubleValue];

	if (newBackingScaleFactor != oldBackingScaleFactor) {
		//Set new display scale and window size
		const float scale = DS_OSX->screen_get_max_scale();
		const NSRect contentRect = [wd.window_view frame];

		wd.size.width = contentRect.size.width * scale;
		wd.size.height = contentRect.size.height * scale;

		DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_DPI_CHANGE);

		CALayer *layer = [wd.window_view layer];
		if (layer) {
			layer.contentsScale = scale;
		}

		//Force window resize event
		[self windowDidResize:notification];
	}
}

- (void)windowDidResize:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	const NSRect contentRect = [wd.window_view frame];

	const float scale = DS_OSX->screen_get_max_scale();
	wd.size.width = contentRect.size.width * scale;
	wd.size.height = contentRect.size.height * scale;

	CALayer *layer = [wd.window_view layer];
	if (layer) {
		layer.contentsScale = scale;
	}

#if defined(GLES3_ENABLED)
	if (DS_OSX->gl_manager) {
		DS_OSX->gl_manager->window_resize(window_id, wd.size.width, wd.size.height);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (DS_OSX->context_vulkan) {
		DS_OSX->context_vulkan->window_resize(window_id, wd.size.width, wd.size.height);
	}
#endif

	if (!wd.rect_changed_callback.is_null()) {
		Variant size = Rect2i(DS_OSX->window_get_position(window_id), DS_OSX->window_get_size(window_id));
		Variant *sizep = &size;
		Variant ret;
		Callable::CallError ce;
		wd.rect_changed_callback.call((const Variant **)&sizep, 1, ret, ce);
	}
}

- (void)windowDidMove:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	DS_OSX->_release_pressed_events();

	if (!wd.rect_changed_callback.is_null()) {
		Variant size = Rect2i(DS_OSX->window_get_position(window_id), DS_OSX->window_get_size(window_id));
		Variant *sizep = &size;
		Variant ret;
		Callable::CallError ce;
		wd.rect_changed_callback.call((const Variant **)&sizep, 1, ret, ce);
	}
}

- (void)windowDidBecomeKey:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (DS_OSX->mouse_mode == DisplayServer::MOUSE_MODE_CAPTURED) {
		const NSRect contentRect = [wd.window_view frame];
		NSRect pointInWindowRect = NSMakeRect(contentRect.size.width / 2, contentRect.size.height / 2, 0, 0);
		NSPoint pointOnScreen = [[wd.window_view window] convertRectToScreen:pointInWindowRect].origin;
		CGPoint lMouseWarpPos = { pointOnScreen.x, CGDisplayBounds(CGMainDisplayID()).size.height - pointOnScreen.y };
		CGWarpMouseCursorPosition(lMouseWarpPos);
	} else {
		_get_mouse_pos(wd, [wd.window_object mouseLocationOutsideOfEventStream]);
		Input::get_singleton()->set_mouse_position(wd.mouse_pos);
	}

	DS_OSX->window_focused = true;
	DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_FOCUS_IN);
}

- (void)windowDidResignKey:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	DS_OSX->window_focused = false;

	DS_OSX->_release_pressed_events();
	DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_FOCUS_OUT);
}

- (void)windowDidMiniaturize:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	DS_OSX->window_focused = false;

	DS_OSX->_release_pressed_events();
	DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_FOCUS_OUT);
}

- (void)windowDidDeminiaturize:(NSNotification *)notification {
	if (!DS_OSX || !DS_OSX->windows.has(window_id)) {
		return;
	}
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	DS_OSX->window_focused = true;
	DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_FOCUS_IN);
}

@end

/*************************************************************************/
/* GodotContentView                                                      */
/*************************************************************************/

#if defined(GLES3_ENABLED)
@interface GodotContentView : NSOpenGLView <NSTextInputClient> {
#else
@interface GodotContentView : NSView <NSTextInputClient> {
#endif

	DisplayServerOSX::WindowID window_id;
	NSTrackingArea *trackingArea;
	NSMutableAttributedString *markedText;
	bool imeInputEventInProgress;
}

- (void)cancelComposition;
- (CALayer *)makeBackingLayer;
- (BOOL)wantsUpdateLayer;
- (void)updateLayer;
- (void)setWindowID:(DisplayServerOSX::WindowID)wid;

@end

@implementation GodotContentView

- (void)setWindowID:(DisplayServerOSX::WindowID)wid {
	window_id = wid;
}

+ (void)initialize {
	if (self == [GodotContentView class]) {
		// nothing left to do here at the moment..
	}
}

- (CALayer *)makeBackingLayer {
#if defined(VULKAN_ENABLED)
	if (DS_OSX->context_vulkan) {
		CALayer *layer = [[CAMetalLayer class] layer];
		return layer;
	}
#endif
	return [super makeBackingLayer];
}

- (void)updateLayer {
#if defined(GLES3_ENABLED)
	if (DS_OSX->gl_manager) {
		DS_OSX->gl_manager->window_update(window_id);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (DS_OSX->context_vulkan) {
		[super updateLayer];
	}
#endif
}

- (BOOL)wantsUpdateLayer {
	return YES;
}

- (id)init {
	self = [super init];
	trackingArea = nil;
	imeInputEventInProgress = false;
	[self updateTrackingAreas];
#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101400
	[self registerForDraggedTypes:[NSArray arrayWithObject:NSPasteboardTypeFileURL]];
#else
	[self registerForDraggedTypes:[NSArray arrayWithObject:NSFilenamesPboardType]];
#endif
	markedText = [[NSMutableAttributedString alloc] init];
	return self;
}

- (void)dealloc {
	[trackingArea release];
	[markedText release];
	[super dealloc];
}

static const NSRange kEmptyRange = { NSNotFound, 0 };

- (BOOL)hasMarkedText {
	return (markedText.length > 0);
}

- (NSRange)markedRange {
	return NSMakeRange(0, markedText.length);
}

- (NSRange)selectedRange {
	return kEmptyRange;
}

- (void)setMarkedText:(id)aString selectedRange:(NSRange)selectedRange replacementRange:(NSRange)replacementRange {
	if ([aString isKindOfClass:[NSAttributedString class]]) {
		[markedText initWithAttributedString:aString];
	} else {
		[markedText initWithString:aString];
	}
	if (markedText.length == 0) {
		[self unmarkText];
		return;
	}

	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (wd.im_active) {
		imeInputEventInProgress = true;
		DS_OSX->im_text.parse_utf8([[markedText mutableString] UTF8String]);
		DS_OSX->im_selection = Point2i(selectedRange.location, selectedRange.length);

		OS_OSX::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
	}
}

- (void)doCommandBySelector:(SEL)aSelector {
	if ([self respondsToSelector:aSelector]) {
		[self performSelector:aSelector];
	}
}

- (void)unmarkText {
	imeInputEventInProgress = false;
	[[markedText mutableString] setString:@""];

	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (wd.im_active) {
		DS_OSX->im_text = String();
		DS_OSX->im_selection = Point2i();

		OS_OSX::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
	}
}

- (NSArray *)validAttributesForMarkedText {
	return [NSArray array];
}

- (NSAttributedString *)attributedSubstringForProposedRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange {
	return nil;
}

- (NSUInteger)characterIndexForPoint:(NSPoint)aPoint {
	return 0;
}

- (NSRect)firstRectForCharacterRange:(NSRange)aRange actualRange:(NSRangePointer)actualRange {
	ERR_FAIL_COND_V(!DS_OSX->windows.has(window_id), NSMakeRect(0, 0, 0, 0));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	const NSRect contentRect = [wd.window_view frame];
	const float scale = DS_OSX->screen_get_max_scale();
	NSRect pointInWindowRect = NSMakeRect(wd.im_position.x / scale, contentRect.size.height - (wd.im_position.y / scale) - 1, 0, 0);
	NSPoint pointOnScreen = [wd.window_object convertRectToScreen:pointInWindowRect].origin;

	return NSMakeRect(pointOnScreen.x, pointOnScreen.y, 0, 0);
}

- (void)cancelComposition {
	[self unmarkText];
	NSTextInputContext *currentInputContext = [NSTextInputContext currentInputContext];
	[currentInputContext discardMarkedText];
}

- (void)insertText:(id)aString {
	[self insertText:aString replacementRange:NSMakeRange(0, 0)];
}

- (void)insertText:(id)aString replacementRange:(NSRange)replacementRange {
	NSEvent *event = [NSApp currentEvent];

	NSString *characters;
	if ([aString isKindOfClass:[NSAttributedString class]]) {
		characters = [aString string];
	} else {
		characters = (NSString *)aString;
	}

	NSCharacterSet *ctrlChars = [NSCharacterSet controlCharacterSet];
	NSCharacterSet *wsnlChars = [NSCharacterSet whitespaceAndNewlineCharacterSet];
	if ([characters rangeOfCharacterFromSet:ctrlChars].length && [characters rangeOfCharacterFromSet:wsnlChars].length == 0) {
		NSTextInputContext *currentInputContext = [NSTextInputContext currentInputContext];
		[currentInputContext discardMarkedText];
		[self cancelComposition];
		return;
	}

	Char16String text;
	text.resize([characters length] + 1);
	[characters getCharacters:(unichar *)text.ptrw() range:NSMakeRange(0, [characters length])];

	String u32text;
	u32text.parse_utf16(text.ptr(), text.length());

	for (int i = 0; i < u32text.length(); i++) {
		const char32_t codepoint = u32text[i];
		if ((codepoint & 0xFF00) == 0xF700) {
			continue;
		}

		DisplayServerOSX::KeyEvent ke;

		ke.window_id = window_id;
		ke.osx_state = [event modifierFlags];
		ke.pressed = true;
		ke.echo = false;
		ke.raw = false; // IME input event
		ke.keycode = Key::NONE;
		ke.physical_keycode = Key::NONE;
		ke.unicode = codepoint;

		_push_to_key_event_buffer(ke);
	}
	[self cancelComposition];
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
	return NSDragOperationCopy;
}

- (NSDragOperation)draggingUpdated:(id<NSDraggingInfo>)sender {
	return NSDragOperationCopy;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
	ERR_FAIL_COND_V(!DS_OSX->windows.has(window_id), NO);
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (!wd.drop_files_callback.is_null()) {
		Vector<String> files;
		NSPasteboard *pboard = [sender draggingPasteboard];

#if MAC_OS_X_VERSION_MIN_REQUIRED >= 101400
		NSArray *items = pboard.pasteboardItems;
		for (NSPasteboardItem *item in items) {
			NSString *path = [item stringForType:NSPasteboardTypeFileURL];
			NSString *ns = [NSURL URLWithString:path].path;
			char *utfs = strdup([ns UTF8String]);
			String ret;
			ret.parse_utf8(utfs);
			free(utfs);
			files.push_back(ret);
		}
#else
		NSArray *filenames = [pboard propertyListForType:NSFilenamesPboardType];
		for (NSString *ns in filenames) {
			char *utfs = strdup([ns UTF8String]);
			String ret;
			ret.parse_utf8(utfs);
			free(utfs);
			files.push_back(ret);
		}
#endif

		Variant v = files;
		Variant *vp = &v;
		Variant ret;
		Callable::CallError ce;
		wd.drop_files_callback.call((const Variant **)&vp, 1, ret, ce);
	}

	return NO;
}

- (BOOL)isOpaque {
	return YES;
}

- (BOOL)canBecomeKeyView {
	if (DS_OSX->windows.has(window_id)) {
		DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];
		if (wd.no_focus) {
			return NO;
		}
	}
	return YES;
}

- (BOOL)acceptsFirstResponder {
	return YES;
}

- (void)cursorUpdate:(NSEvent *)event {
	DisplayServer::CursorShape p_shape = DS_OSX->cursor_shape;
	DS_OSX->cursor_shape = DisplayServer::CURSOR_MAX;
	DS_OSX->cursor_set_shape(p_shape);
}

static void _mouseDownEvent(DisplayServer::WindowID window_id, NSEvent *event, MouseButton index, MouseButton mask, bool pressed) {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (pressed) {
		DS_OSX->last_button_state |= mask;
	} else {
		DS_OSX->last_button_state &= (MouseButton)~mask;
	}

	Ref<InputEventMouseButton> mb;
	mb.instantiate();
	mb->set_window_id(window_id);
	const Vector2 pos = _get_mouse_pos(wd, [event locationInWindow]);
	_get_key_modifier_state([event modifierFlags], mb);
	mb->set_button_index(index);
	mb->set_pressed(pressed);
	mb->set_position(pos);
	mb->set_global_position(pos);
	mb->set_button_mask(DS_OSX->last_button_state);
	if (index == MouseButton::LEFT && pressed) {
		mb->set_double_click([event clickCount] == 2);
	}

	Input::get_singleton()->parse_input_event(mb);
}

- (void)mouseDown:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (([event modifierFlags] & NSEventModifierFlagControl)) {
		wd.mouse_down_control = true;
		_mouseDownEvent(window_id, event, MouseButton::RIGHT, MouseButton::MASK_RIGHT, true);
	} else {
		wd.mouse_down_control = false;
		_mouseDownEvent(window_id, event, MouseButton::LEFT, MouseButton::MASK_LEFT, true);
	}
}

- (void)mouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)mouseUp:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (wd.mouse_down_control) {
		_mouseDownEvent(window_id, event, MouseButton::RIGHT, MouseButton::MASK_RIGHT, false);
	} else {
		_mouseDownEvent(window_id, event, MouseButton::LEFT, MouseButton::MASK_LEFT, false);
	}
}

- (void)mouseMoved:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	NSPoint delta = NSMakePoint([event deltaX], [event deltaY]);
	NSPoint mpos = [event locationInWindow];

	if (DS_OSX->ignore_warp) {
		// Discard late events, before warp
		if (([event timestamp]) < DS_OSX->last_warp) {
			return;
		}
		DS_OSX->ignore_warp = false;
		return;
	}

	if (DS_OSX->mouse_mode == DisplayServer::MOUSE_MODE_CONFINED || DS_OSX->mouse_mode == DisplayServer::MOUSE_MODE_CONFINED_HIDDEN) {
		// Discard late events
		if (([event timestamp]) < DS_OSX->last_warp) {
			return;
		}

		// Warp affects next event delta, subtract previous warp deltas
		List<DisplayServerOSX::WarpEvent>::Element *F = DS_OSX->warp_events.front();
		while (F) {
			if (F->get().timestamp < [event timestamp]) {
				List<DisplayServerOSX::WarpEvent>::Element *E = F;
				delta.x -= E->get().delta.x;
				delta.y -= E->get().delta.y;
				F = F->next();
				DS_OSX->warp_events.erase(E);
			} else {
				F = F->next();
			}
		}

		// Confine mouse position to the window, and update delta
		NSRect frame = [wd.window_object frame];
		NSPoint conf_pos = mpos;
		conf_pos.x = CLAMP(conf_pos.x + delta.x, 0.f, frame.size.width);
		conf_pos.y = CLAMP(conf_pos.y - delta.y, 0.f, frame.size.height);
		delta.x = conf_pos.x - mpos.x;
		delta.y = mpos.y - conf_pos.y;
		mpos = conf_pos;

		// Move mouse cursor
		NSRect pointInWindowRect = NSMakeRect(conf_pos.x, conf_pos.y, 0, 0);
		conf_pos = [[wd.window_view window] convertRectToScreen:pointInWindowRect].origin;
		conf_pos.y = CGDisplayBounds(CGMainDisplayID()).size.height - conf_pos.y;
		CGWarpMouseCursorPosition(conf_pos);

		// Save warp data
		DS_OSX->last_warp = [[NSProcessInfo processInfo] systemUptime];
		DisplayServerOSX::WarpEvent ev;
		ev.timestamp = DS_OSX->last_warp;
		ev.delta = delta;
		DS_OSX->warp_events.push_back(ev);
	}

	Ref<InputEventMouseMotion> mm;
	mm.instantiate();

	mm->set_window_id(window_id);
	mm->set_button_mask(DS_OSX->last_button_state);
	const Vector2i pos = _get_mouse_pos(wd, mpos);
	mm->set_position(pos);
	mm->set_pressure([event pressure]);
	if ([event subtype] == NSEventSubtypeTabletPoint) {
		const NSPoint p = [event tilt];
		mm->set_tilt(Vector2(p.x, p.y));
	}
	mm->set_global_position(pos);
	mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());
	const Vector2i relativeMotion = Vector2i(delta.x, delta.y) * DS_OSX->screen_get_max_scale();
	mm->set_relative(relativeMotion);
	_get_key_modifier_state([event modifierFlags], mm);

	Input::get_singleton()->parse_input_event(mm);
}

- (void)rightMouseDown:(NSEvent *)event {
	_mouseDownEvent(window_id, event, MouseButton::RIGHT, MouseButton::MASK_RIGHT, true);
}

- (void)rightMouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)rightMouseUp:(NSEvent *)event {
	_mouseDownEvent(window_id, event, MouseButton::RIGHT, MouseButton::MASK_RIGHT, false);
}

- (void)otherMouseDown:(NSEvent *)event {
	if ((int)[event buttonNumber] == 2) {
		_mouseDownEvent(window_id, event, MouseButton::MIDDLE, MouseButton::MASK_MIDDLE, true);
	} else if ((int)[event buttonNumber] == 3) {
		_mouseDownEvent(window_id, event, MouseButton::MB_XBUTTON1, MouseButton::MASK_XBUTTON1, true);
	} else if ((int)[event buttonNumber] == 4) {
		_mouseDownEvent(window_id, event, MouseButton::MB_XBUTTON2, MouseButton::MASK_XBUTTON2, true);
	} else {
		return;
	}
}

- (void)otherMouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)otherMouseUp:(NSEvent *)event {
	if ((int)[event buttonNumber] == 2) {
		_mouseDownEvent(window_id, event, MouseButton::MIDDLE, MouseButton::MASK_MIDDLE, false);
	} else if ((int)[event buttonNumber] == 3) {
		_mouseDownEvent(window_id, event, MouseButton::MB_XBUTTON1, MouseButton::MASK_XBUTTON1, false);
	} else if ((int)[event buttonNumber] == 4) {
		_mouseDownEvent(window_id, event, MouseButton::MB_XBUTTON2, MouseButton::MASK_XBUTTON2, false);
	} else {
		return;
	}
}

- (void)mouseExited:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (DS_OSX->mouse_mode != DisplayServer::MOUSE_MODE_CAPTURED) {
		DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_MOUSE_EXIT);
	}
}

- (void)mouseEntered:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	if (DS_OSX->mouse_mode != DisplayServer::MOUSE_MODE_CAPTURED) {
		DS_OSX->_send_window_event(wd, DisplayServerOSX::WINDOW_EVENT_MOUSE_ENTER);
	}

	DisplayServer::CursorShape p_shape = DS_OSX->cursor_shape;
	DS_OSX->cursor_shape = DisplayServer::CURSOR_MAX;
	DS_OSX->cursor_set_shape(p_shape);
}

- (void)magnifyWithEvent:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	Ref<InputEventMagnifyGesture> ev;
	ev.instantiate();
	ev->set_window_id(window_id);
	_get_key_modifier_state([event modifierFlags], ev);
	ev->set_position(_get_mouse_pos(wd, [event locationInWindow]));
	ev->set_factor([event magnification] + 1.0);

	Input::get_singleton()->parse_input_event(ev);
}

- (void)viewDidChangeBackingProperties {
	// nothing left to do here
}

- (void)updateTrackingAreas {
	if (trackingArea != nil) {
		[self removeTrackingArea:trackingArea];
		[trackingArea release];
	}

	NSTrackingAreaOptions options = NSTrackingMouseEnteredAndExited | NSTrackingActiveInKeyWindow | NSTrackingCursorUpdate | NSTrackingInVisibleRect;
	trackingArea = [[NSTrackingArea alloc] initWithRect:[self bounds] options:options owner:self userInfo:nil];

	[self addTrackingArea:trackingArea];
	[super updateTrackingAreas];
}

static bool isNumpadKey(unsigned int key) {
	static const unsigned int table[] = {
		0x41, /* kVK_ANSI_KeypadDecimal */
		0x43, /* kVK_ANSI_KeypadMultiply */
		0x45, /* kVK_ANSI_KeypadPlus */
		0x47, /* kVK_ANSI_KeypadClear */
		0x4b, /* kVK_ANSI_KeypadDivide */
		0x4c, /* kVK_ANSI_KeypadEnter */
		0x4e, /* kVK_ANSI_KeypadMinus */
		0x51, /* kVK_ANSI_KeypadEquals */
		0x52, /* kVK_ANSI_Keypad0 */
		0x53, /* kVK_ANSI_Keypad1 */
		0x54, /* kVK_ANSI_Keypad2 */
		0x55, /* kVK_ANSI_Keypad3 */
		0x56, /* kVK_ANSI_Keypad4 */
		0x57, /* kVK_ANSI_Keypad5 */
		0x58, /* kVK_ANSI_Keypad6 */
		0x59, /* kVK_ANSI_Keypad7 */
		0x5b, /* kVK_ANSI_Keypad8 */
		0x5c, /* kVK_ANSI_Keypad9 */
		0x5f, /* kVK_JIS_KeypadComma */
		0x00
	};
	for (int i = 0; table[i] != 0; i++) {
		if (key == table[i]) {
			return true;
		}
	}
	return false;
}

// Keyboard symbol translation table
static const Key _osx_to_godot_table[128] = {
	/* 00 */ Key::A,
	/* 01 */ Key::S,
	/* 02 */ Key::D,
	/* 03 */ Key::F,
	/* 04 */ Key::H,
	/* 05 */ Key::G,
	/* 06 */ Key::Z,
	/* 07 */ Key::X,
	/* 08 */ Key::C,
	/* 09 */ Key::V,
	/* 0a */ Key::SECTION, /* ISO Section */
	/* 0b */ Key::B,
	/* 0c */ Key::Q,
	/* 0d */ Key::W,
	/* 0e */ Key::E,
	/* 0f */ Key::R,
	/* 10 */ Key::Y,
	/* 11 */ Key::T,
	/* 12 */ Key::KEY_1,
	/* 13 */ Key::KEY_2,
	/* 14 */ Key::KEY_3,
	/* 15 */ Key::KEY_4,
	/* 16 */ Key::KEY_6,
	/* 17 */ Key::KEY_5,
	/* 18 */ Key::EQUAL,
	/* 19 */ Key::KEY_9,
	/* 1a */ Key::KEY_7,
	/* 1b */ Key::MINUS,
	/* 1c */ Key::KEY_8,
	/* 1d */ Key::KEY_0,
	/* 1e */ Key::BRACERIGHT,
	/* 1f */ Key::O,
	/* 20 */ Key::U,
	/* 21 */ Key::BRACELEFT,
	/* 22 */ Key::I,
	/* 23 */ Key::P,
	/* 24 */ Key::ENTER,
	/* 25 */ Key::L,
	/* 26 */ Key::J,
	/* 27 */ Key::APOSTROPHE,
	/* 28 */ Key::K,
	/* 29 */ Key::SEMICOLON,
	/* 2a */ Key::BACKSLASH,
	/* 2b */ Key::COMMA,
	/* 2c */ Key::SLASH,
	/* 2d */ Key::N,
	/* 2e */ Key::M,
	/* 2f */ Key::PERIOD,
	/* 30 */ Key::TAB,
	/* 31 */ Key::SPACE,
	/* 32 */ Key::QUOTELEFT,
	/* 33 */ Key::BACKSPACE,
	/* 34 */ Key::UNKNOWN,
	/* 35 */ Key::ESCAPE,
	/* 36 */ Key::META,
	/* 37 */ Key::META,
	/* 38 */ Key::SHIFT,
	/* 39 */ Key::CAPSLOCK,
	/* 3a */ Key::ALT,
	/* 3b */ Key::CTRL,
	/* 3c */ Key::SHIFT,
	/* 3d */ Key::ALT,
	/* 3e */ Key::CTRL,
	/* 3f */ Key::UNKNOWN, /* Function */
	/* 40 */ Key::UNKNOWN, /* F17 */
	/* 41 */ Key::KP_PERIOD,
	/* 42 */ Key::UNKNOWN,
	/* 43 */ Key::KP_MULTIPLY,
	/* 44 */ Key::UNKNOWN,
	/* 45 */ Key::KP_ADD,
	/* 46 */ Key::UNKNOWN,
	/* 47 */ Key::NUMLOCK, /* Really KeypadClear... */
	/* 48 */ Key::VOLUMEUP, /* VolumeUp */
	/* 49 */ Key::VOLUMEDOWN, /* VolumeDown */
	/* 4a */ Key::VOLUMEMUTE, /* Mute */
	/* 4b */ Key::KP_DIVIDE,
	/* 4c */ Key::KP_ENTER,
	/* 4d */ Key::UNKNOWN,
	/* 4e */ Key::KP_SUBTRACT,
	/* 4f */ Key::UNKNOWN, /* F18 */
	/* 50 */ Key::UNKNOWN, /* F19 */
	/* 51 */ Key::EQUAL, /* KeypadEqual */
	/* 52 */ Key::KP_0,
	/* 53 */ Key::KP_1,
	/* 54 */ Key::KP_2,
	/* 55 */ Key::KP_3,
	/* 56 */ Key::KP_4,
	/* 57 */ Key::KP_5,
	/* 58 */ Key::KP_6,
	/* 59 */ Key::KP_7,
	/* 5a */ Key::UNKNOWN, /* F20 */
	/* 5b */ Key::KP_8,
	/* 5c */ Key::KP_9,
	/* 5d */ Key::YEN, /* JIS Yen */
	/* 5e */ Key::UNDERSCORE, /* JIS Underscore */
	/* 5f */ Key::COMMA, /* JIS KeypadComma */
	/* 60 */ Key::F5,
	/* 61 */ Key::F6,
	/* 62 */ Key::F7,
	/* 63 */ Key::F3,
	/* 64 */ Key::F8,
	/* 65 */ Key::F9,
	/* 66 */ Key::UNKNOWN, /* JIS Eisu */
	/* 67 */ Key::F11,
	/* 68 */ Key::UNKNOWN, /* JIS Kana */
	/* 69 */ Key::F13,
	/* 6a */ Key::F16,
	/* 6b */ Key::F14,
	/* 6c */ Key::UNKNOWN,
	/* 6d */ Key::F10,
	/* 6e */ Key::MENU,
	/* 6f */ Key::F12,
	/* 70 */ Key::UNKNOWN,
	/* 71 */ Key::F15,
	/* 72 */ Key::INSERT, /* Really Help... */
	/* 73 */ Key::HOME,
	/* 74 */ Key::PAGEUP,
	/* 75 */ Key::KEY_DELETE,
	/* 76 */ Key::F4,
	/* 77 */ Key::END,
	/* 78 */ Key::F2,
	/* 79 */ Key::PAGEDOWN,
	/* 7a */ Key::F1,
	/* 7b */ Key::LEFT,
	/* 7c */ Key::RIGHT,
	/* 7d */ Key::DOWN,
	/* 7e */ Key::UP,
	/* 7f */ Key::UNKNOWN,
};

// Translates a OS X keycode to a Godot keycode
static Key translateKey(unsigned int key) {
	if (key >= 128) {
		return Key::UNKNOWN;
	}

	return _osx_to_godot_table[key];
}

// Translates a Godot keycode back to a OSX keycode
static unsigned int unmapKey(Key key) {
	for (int i = 0; i <= 126; i++) {
		if (_osx_to_godot_table[i] == key) {
			return i;
		}
	}
	return 127;
}

struct _KeyCodeMap {
	UniChar kchar;
	Key kcode;
};

static const _KeyCodeMap _keycodes[55] = {
	{ '`', Key::QUOTELEFT },
	{ '~', Key::ASCIITILDE },
	{ '0', Key::KEY_0 },
	{ '1', Key::KEY_1 },
	{ '2', Key::KEY_2 },
	{ '3', Key::KEY_3 },
	{ '4', Key::KEY_4 },
	{ '5', Key::KEY_5 },
	{ '6', Key::KEY_6 },
	{ '7', Key::KEY_7 },
	{ '8', Key::KEY_8 },
	{ '9', Key::KEY_9 },
	{ '-', Key::MINUS },
	{ '_', Key::UNDERSCORE },
	{ '=', Key::EQUAL },
	{ '+', Key::PLUS },
	{ 'q', Key::Q },
	{ 'w', Key::W },
	{ 'e', Key::E },
	{ 'r', Key::R },
	{ 't', Key::T },
	{ 'y', Key::Y },
	{ 'u', Key::U },
	{ 'i', Key::I },
	{ 'o', Key::O },
	{ 'p', Key::P },
	{ '[', Key::BRACELEFT },
	{ ']', Key::BRACERIGHT },
	{ '{', Key::BRACELEFT },
	{ '}', Key::BRACERIGHT },
	{ 'a', Key::A },
	{ 's', Key::S },
	{ 'd', Key::D },
	{ 'f', Key::F },
	{ 'g', Key::G },
	{ 'h', Key::H },
	{ 'j', Key::J },
	{ 'k', Key::K },
	{ 'l', Key::L },
	{ ';', Key::SEMICOLON },
	{ ':', Key::COLON },
	{ '\'', Key::APOSTROPHE },
	{ '\"', Key::QUOTEDBL },
	{ '\\', Key::BACKSLASH },
	{ '#', Key::NUMBERSIGN },
	{ 'z', Key::Z },
	{ 'x', Key::X },
	{ 'c', Key::C },
	{ 'v', Key::V },
	{ 'b', Key::B },
	{ 'n', Key::N },
	{ 'm', Key::M },
	{ ',', Key::COMMA },
	{ '.', Key::PERIOD },
	{ '/', Key::SLASH }
};

static Key remapKey(unsigned int key, unsigned int state) {
	if (isNumpadKey(key)) {
		return translateKey(key);
	}

	TISInputSourceRef currentKeyboard = TISCopyCurrentKeyboardInputSource();
	if (!currentKeyboard) {
		return translateKey(key);
	}

	CFDataRef layoutData = (CFDataRef)TISGetInputSourceProperty(currentKeyboard, kTISPropertyUnicodeKeyLayoutData);
	if (!layoutData) {
		return translateKey(key);
	}

	const UCKeyboardLayout *keyboardLayout = (const UCKeyboardLayout *)CFDataGetBytePtr(layoutData);

	UInt32 keysDown = 0;
	UniChar chars[4];
	UniCharCount realLength;

	OSStatus err = UCKeyTranslate(keyboardLayout,
			key,
			kUCKeyActionDisplay,
			(state >> 8) & 0xFF,
			LMGetKbdType(),
			kUCKeyTranslateNoDeadKeysBit,
			&keysDown,
			sizeof(chars) / sizeof(chars[0]),
			&realLength,
			chars);

	if (err != noErr) {
		return translateKey(key);
	}

	for (unsigned int i = 0; i < 55; i++) {
		if (_keycodes[i].kchar == chars[0]) {
			return _keycodes[i].kcode;
		}
	}
	return translateKey(key);
}

- (void)keyDown:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	ignore_momentum_scroll = true;

	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		NSString *characters = [event characters];
		NSUInteger length = [characters length];

		if (!wd.im_active && length > 0 && keycode_has_unicode(remapKey([event keyCode], [event modifierFlags]))) {
			// Fallback unicode character handler used if IME is not active
			Char16String text;
			text.resize([characters length] + 1);
			[characters getCharacters:(unichar *)text.ptrw() range:NSMakeRange(0, [characters length])];

			String u32text;
			u32text.parse_utf16(text.ptr(), text.length());

			for (int i = 0; i < u32text.length(); i++) {
				const char32_t codepoint = u32text[i];

				DisplayServerOSX::KeyEvent ke;

				ke.window_id = window_id;
				ke.osx_state = [event modifierFlags];
				ke.pressed = true;
				ke.echo = [event isARepeat];
				ke.keycode = remapKey([event keyCode], [event modifierFlags]);
				ke.physical_keycode = translateKey([event keyCode]);
				ke.raw = true;
				ke.unicode = codepoint;

				_push_to_key_event_buffer(ke);
			}
		} else {
			DisplayServerOSX::KeyEvent ke;

			ke.window_id = window_id;
			ke.osx_state = [event modifierFlags];
			ke.pressed = true;
			ke.echo = [event isARepeat];
			ke.keycode = remapKey([event keyCode], [event modifierFlags]);
			ke.physical_keycode = translateKey([event keyCode]);
			ke.raw = false;
			ke.unicode = 0;

			_push_to_key_event_buffer(ke);
		}
	}

	// Pass events to IME handler
	if (wd.im_active) {
		[self interpretKeyEvents:[NSArray arrayWithObject:event]];
	}
}

- (void)flagsChanged:(NSEvent *)event {
	ignore_momentum_scroll = true;

	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		DisplayServerOSX::KeyEvent ke;

		ke.window_id = window_id;
		ke.echo = false;
		ke.raw = true;

		int key = [event keyCode];
		int mod = [event modifierFlags];

		if (key == 0x36 || key == 0x37) {
			if (mod & NSEventModifierFlagCommand) {
				mod &= ~NSEventModifierFlagCommand;
				ke.pressed = true;
			} else {
				ke.pressed = false;
			}
		} else if (key == 0x38 || key == 0x3c) {
			if (mod & NSEventModifierFlagShift) {
				mod &= ~NSEventModifierFlagShift;
				ke.pressed = true;
			} else {
				ke.pressed = false;
			}
		} else if (key == 0x3a || key == 0x3d) {
			if (mod & NSEventModifierFlagOption) {
				mod &= ~NSEventModifierFlagOption;
				ke.pressed = true;
			} else {
				ke.pressed = false;
			}
		} else if (key == 0x3b || key == 0x3e) {
			if (mod & NSEventModifierFlagControl) {
				mod &= ~NSEventModifierFlagControl;
				ke.pressed = true;
			} else {
				ke.pressed = false;
			}
		} else {
			return;
		}

		ke.osx_state = mod;
		ke.keycode = remapKey(key, mod);
		ke.physical_keycode = translateKey(key);
		ke.unicode = 0;

		_push_to_key_event_buffer(ke);
	}
}

- (void)keyUp:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		NSString *characters = [event characters];
		NSUInteger length = [characters length];

		// Fallback unicode character handler used if IME is not active
		if (!wd.im_active && length > 0 && keycode_has_unicode(remapKey([event keyCode], [event modifierFlags]))) {
			Char16String text;
			text.resize([characters length] + 1);
			[characters getCharacters:(unichar *)text.ptrw() range:NSMakeRange(0, [characters length])];

			String u32text;
			u32text.parse_utf16(text.ptr(), text.length());

			for (int i = 0; i < u32text.length(); i++) {
				const char32_t codepoint = u32text[i];
				DisplayServerOSX::KeyEvent ke;

				ke.window_id = window_id;
				ke.osx_state = [event modifierFlags];
				ke.pressed = false;
				ke.echo = [event isARepeat];
				ke.keycode = remapKey([event keyCode], [event modifierFlags]);
				ke.physical_keycode = translateKey([event keyCode]);
				ke.raw = true;
				ke.unicode = codepoint;

				_push_to_key_event_buffer(ke);
			}
		} else {
			DisplayServerOSX::KeyEvent ke;

			ke.window_id = window_id;
			ke.osx_state = [event modifierFlags];
			ke.pressed = false;
			ke.echo = [event isARepeat];
			ke.keycode = remapKey([event keyCode], [event modifierFlags]);
			ke.physical_keycode = translateKey([event keyCode]);
			ke.raw = true;
			ke.unicode = 0;

			_push_to_key_event_buffer(ke);
		}
	}
}

inline void sendScrollEvent(DisplayServer::WindowID window_id, MouseButton button, double factor, int modifierFlags) {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	MouseButton mask = mouse_button_to_mask(button);

	Ref<InputEventMouseButton> sc;
	sc.instantiate();

	sc->set_window_id(window_id);
	_get_key_modifier_state(modifierFlags, sc);
	sc->set_button_index(button);
	sc->set_factor(factor);
	sc->set_pressed(true);
	sc->set_position(wd.mouse_pos);
	sc->set_global_position(wd.mouse_pos);
	DS_OSX->last_button_state |= (MouseButton)mask;
	sc->set_button_mask(DS_OSX->last_button_state);

	Input::get_singleton()->parse_input_event(sc);

	sc.instantiate();
	sc->set_window_id(window_id);
	sc->set_button_index(button);
	sc->set_factor(factor);
	sc->set_pressed(false);
	sc->set_position(wd.mouse_pos);
	sc->set_global_position(wd.mouse_pos);
	DS_OSX->last_button_state &= (MouseButton)~mask;
	sc->set_button_mask(DS_OSX->last_button_state);

	Input::get_singleton()->parse_input_event(sc);
}

inline void sendPanEvent(DisplayServer::WindowID window_id, double dx, double dy, int modifierFlags) {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	Ref<InputEventPanGesture> pg;
	pg.instantiate();

	pg->set_window_id(window_id);
	_get_key_modifier_state(modifierFlags, pg);
	pg->set_position(wd.mouse_pos);
	pg->set_delta(Vector2(-dx, -dy));

	Input::get_singleton()->parse_input_event(pg);
}

- (void)scrollWheel:(NSEvent *)event {
	ERR_FAIL_COND(!DS_OSX->windows.has(window_id));
	DisplayServerOSX::WindowData &wd = DS_OSX->windows[window_id];

	double deltaX, deltaY;

	_get_mouse_pos(wd, [event locationInWindow]);

	deltaX = [event scrollingDeltaX];
	deltaY = [event scrollingDeltaY];

	if ([event hasPreciseScrollingDeltas]) {
		deltaX *= 0.03;
		deltaY *= 0.03;
	}

	if ([event momentumPhase] != NSEventPhaseNone) {
		if (ignore_momentum_scroll) {
			return;
		}
	} else {
		ignore_momentum_scroll = false;
	}

	if ([event phase] != NSEventPhaseNone || [event momentumPhase] != NSEventPhaseNone) {
		sendPanEvent(window_id, deltaX, deltaY, [event modifierFlags]);
	} else {
		if (fabs(deltaX)) {
			sendScrollEvent(window_id, 0 > deltaX ? MouseButton::WHEEL_RIGHT : MouseButton::WHEEL_LEFT, fabs(deltaX * 0.3), [event modifierFlags]);
		}
		if (fabs(deltaY)) {
			sendScrollEvent(window_id, 0 < deltaY ? MouseButton::WHEEL_UP : MouseButton::WHEEL_DOWN, fabs(deltaY * 0.3), [event modifierFlags]);
		}
	}
}

@end

/*************************************************************************/
/* GodotWindow                                                           */
/*************************************************************************/

@interface GodotWindow : NSWindow {
}

@end

@implementation GodotWindow

- (BOOL)canBecomeKeyWindow {
	// Required for NSBorderlessWindowMask windows
	for (Map<DisplayServer::WindowID, DisplayServerOSX::WindowData>::Element *E = DS_OSX->windows.front(); E; E = E->next()) {
		if (E->get().window_object == self) {
			if (E->get().no_focus) {
				return NO;
			}
		}
	}
	return YES;
}

- (BOOL)canBecomeMainWindow {
	// Required for NSBorderlessWindowMask windows
	for (Map<DisplayServer::WindowID, DisplayServerOSX::WindowData>::Element *E = DS_OSX->windows.front(); E; E = E->next()) {
		if (E->get().window_object == self) {
			if (E->get().no_focus) {
				return NO;
			}
		}
	}
	return YES;
}

@end

/*************************************************************************/
/* DisplayServerOSX                                                      */
/*************************************************************************/

bool DisplayServerOSX::has_feature(Feature p_feature) const {
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
		//case FEATURE_CONSOLE_WINDOW:
		case FEATURE_IME:
		case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_HIDPI:
		case FEATURE_ICON:
		case FEATURE_NATIVE_ICON:
		//case FEATURE_KEEP_SCREEN_ON:
		case FEATURE_SWAP_BUFFERS:
			return true;
		default: {
		}
	}
	return false;
}

String DisplayServerOSX::get_name() const {
	return "OSX";
}

const NSMenu *DisplayServerOSX::_get_menu_root(const String &p_menu_root) const {
	const NSMenu *menu = nullptr;
	if (p_menu_root == "") {
		// Main menu.x
		menu = [NSApp mainMenu];
	} else if (p_menu_root.to_lower() == "_dock") {
		// macOS dock menu.
		menu = dock_menu;
	} else {
		// Submenu.
		if (submenu.has(p_menu_root)) {
			menu = submenu[p_menu_root];
		}
	}
	if (menu == apple_menu) {
		// Do not allow to change Apple menu.
		return nullptr;
	}
	return menu;
}

NSMenu *DisplayServerOSX::_get_menu_root(const String &p_menu_root) {
	NSMenu *menu = nullptr;
	if (p_menu_root == "") {
		// Main menu.
		menu = [NSApp mainMenu];
	} else if (p_menu_root.to_lower() == "_dock") {
		// macOS dock menu.
		menu = dock_menu;
	} else {
		// Submenu.
		if (!submenu.has(p_menu_root)) {
			NSMenu *n_menu = [[NSMenu alloc] initWithTitle:[NSString stringWithUTF8String:p_menu_root.utf8().get_data()]];
			submenu[p_menu_root] = n_menu;
		}
		menu = submenu[p_menu_root];
	}
	if (menu == apple_menu) {
		// Do not allow to change Apple menu.
		return nullptr;
	}
	return menu;
}

void DisplayServerOSX::global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		NSMenuItem *menu_item = [menu addItemWithTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()] action:@selector(globalMenuCallback:) keyEquivalent:@""];
		GlobalMenuItem *obj = [[[GlobalMenuItem alloc] init] autorelease];
		obj->callback = p_callback;
		obj->meta = p_tag;
		obj->checkable = false;
		[menu_item setRepresentedObject:obj];
	}
}

void DisplayServerOSX::global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Variant &p_tag) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		NSMenuItem *menu_item = [menu addItemWithTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()] action:@selector(globalMenuCallback:) keyEquivalent:@""];
		GlobalMenuItem *obj = [[[GlobalMenuItem alloc] init] autorelease];
		obj->callback = p_callback;
		obj->meta = p_tag;
		obj->checkable = true;
		[menu_item setRepresentedObject:obj];
	}
}

void DisplayServerOSX::global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
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
		NSMenuItem *menu_item = [menu addItemWithTitle:[NSString stringWithUTF8String:p_label.utf8().get_data()] action:nil keyEquivalent:@""];
		[menu setSubmenu:sub_menu forItem:menu_item];
	}
}

void DisplayServerOSX::global_menu_add_separator(const String &p_menu_root) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		[menu addItem:[NSMenuItem separatorItem]];
	}
}

bool DisplayServerOSX::global_menu_is_item_checked(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			return ([menu_item state] == NSControlStateValueOn);
		}
	}
	return false;
}

bool DisplayServerOSX::global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->checkable;
			}
		}
	}
	return false;
}

Callable DisplayServerOSX::global_menu_get_item_callback(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->callback;
			}
		}
	}
	return Callable();
}

Variant DisplayServerOSX::global_menu_get_item_tag(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			if (obj) {
				return obj->meta;
			}
		}
	}
	return Variant();
}

String DisplayServerOSX::global_menu_get_item_text(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			char *utfs = strdup([[menu_item title] UTF8String]);
			String ret;
			ret.parse_utf8(utfs);
			free(utfs);
			return ret;
		}
	}
	return String();
}

String DisplayServerOSX::global_menu_get_item_submenu(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		const NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			const NSMenu *sub_menu = [menu_item submenu];
			if (sub_menu) {
				for (Map<String, NSMenu *>::Element *E = submenu.front(); E; E = E->next()) {
					if (E->get() == sub_menu)
						return E->key();
				}
			}
		}
	}
	return String();
}

void DisplayServerOSX::global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
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

void DisplayServerOSX::global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			obj->checkable = p_checkable;
		}
	}
}

void DisplayServerOSX::global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			obj->callback = p_callback;
		}
	}
}

void DisplayServerOSX::global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			GlobalMenuItem *obj = [menu_item representedObject];
			obj->meta = p_tag;
		}
	}
}

void DisplayServerOSX::global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu_item setTitle:[NSString stringWithUTF8String:p_text.utf8().get_data()]];
		}
	}
}

void DisplayServerOSX::global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
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
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not edit Apple menu.
			return;
		}
		NSMenuItem *menu_item = [menu itemAtIndex:p_idx];
		if (menu_item) {
			[menu setSubmenu:sub_menu forItem:menu_item];
		}
	}
}

int DisplayServerOSX::global_menu_get_item_count(const String &p_menu_root) const {
	_THREAD_SAFE_METHOD_

	const NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		return [menu numberOfItems];
	} else {
		return 0;
	}
}

void DisplayServerOSX::global_menu_remove_item(const String &p_menu_root, int p_idx) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		if ((menu == [NSApp mainMenu]) && (p_idx == 0)) { // Do not delete Apple menu.
			return;
		}
		[menu removeItemAtIndex:p_idx];
	}
}

void DisplayServerOSX::global_menu_clear(const String &p_menu_root) {
	_THREAD_SAFE_METHOD_

	NSMenu *menu = _get_menu_root(p_menu_root);
	if (menu) {
		[menu removeAllItems];
		// Restore Apple menu.
		if (menu == [NSApp mainMenu]) {
			NSMenuItem *menu_item = [menu addItemWithTitle:@"" action:nil keyEquivalent:@""];
			[menu setSubmenu:apple_menu forItem:menu_item];
		}
	}
}

Error DisplayServerOSX::dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) {
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

	int button_pressed;
	NSInteger ret = [window runModal];
	if (ret == NSAlertFirstButtonReturn) {
		button_pressed = 0;
	} else if (ret == NSAlertSecondButtonReturn) {
		button_pressed = 1;
	} else if (ret == NSAlertThirdButtonReturn) {
		button_pressed = 2;
	} else {
		button_pressed = 2 + (ret - NSAlertThirdButtonReturn);
	}

	if (!p_callback.is_null()) {
		Variant button = button_pressed;
		Variant *buttonp = &button;
		Variant ret;
		Callable::CallError ce;
		p_callback.call((const Variant **)&buttonp, 1, ret, ce);
	}

	[window release];
	return OK;
}

Error DisplayServerOSX::dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) {
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

	char *utfs = strdup([[input stringValue] UTF8String]);
	String ret;
	ret.parse_utf8(utfs);
	free(utfs);

	if (!p_callback.is_null()) {
		Variant text = ret;
		Variant *textp = &text;
		Variant ret;
		Callable::CallError ce;
		p_callback.call((const Variant **)&textp, 1, ret, ce);
	}

	[window release];
	return OK;
}

void DisplayServerOSX::mouse_set_mode(MouseMode p_mode) {
	_THREAD_SAFE_METHOD_

	if (p_mode == mouse_mode) {
		return;
	}

	WindowData &wd = windows[MAIN_WINDOW_ID];
	if (p_mode == MOUSE_MODE_CAPTURED) {
		// Apple Docs state that the display parameter is not used.
		// "This parameter is not used. By default, you may pass kCGDirectMainDisplay."
		// https://developer.apple.com/library/mac/documentation/graphicsimaging/reference/Quartz_Services_Ref/Reference/reference.html
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
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
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		[wd.window_object setMovable:YES];
		CGAssociateMouseAndMouseCursorPosition(true);
	} else if (p_mode == MOUSE_MODE_CONFINED) {
		CGDisplayShowCursor(kCGDirectMainDisplay);
		[wd.window_object setMovable:NO];
		CGAssociateMouseAndMouseCursorPosition(false);
	} else if (p_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
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

	if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
		CursorShape p_shape = cursor_shape;
		cursor_shape = DisplayServer::CURSOR_MAX;
		cursor_set_shape(p_shape);
	}
}

DisplayServer::MouseMode DisplayServerOSX::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerOSX::mouse_warp_to_position(const Point2i &p_to) {
	_THREAD_SAFE_METHOD_

	if (mouse_mode == MOUSE_MODE_CAPTURED) {
		last_mouse_pos = p_to;
	} else {
		WindowData &wd = windows[MAIN_WINDOW_ID];

		//local point in window coords
		const NSRect contentRect = [wd.window_view frame];
		const float scale = screen_get_max_scale();
		NSRect pointInWindowRect = NSMakeRect(p_to.x / scale, contentRect.size.height - (p_to.y / scale - 1), 0, 0);
		NSPoint pointOnScreen = [[wd.window_view window] convertRectToScreen:pointInWindowRect].origin;

		//point in scren coords
		CGPoint lMouseWarpPos = { pointOnScreen.x, CGDisplayBounds(CGMainDisplayID()).size.height - pointOnScreen.y };

		//do the warping
		CGEventSourceRef lEventRef = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState);
		CGEventSourceSetLocalEventsSuppressionInterval(lEventRef, 0.0);
		CGAssociateMouseAndMouseCursorPosition(false);
		CGWarpMouseCursorPosition(lMouseWarpPos);
		if (mouse_mode != MOUSE_MODE_CONFINED && mouse_mode != MOUSE_MODE_CONFINED_HIDDEN) {
			CGAssociateMouseAndMouseCursorPosition(true);
		}
	}
}

Point2i DisplayServerOSX::mouse_get_position() const {
	return last_mouse_pos;
}

Point2i DisplayServerOSX::mouse_get_absolute_position() const {
	_THREAD_SAFE_METHOD_

	const NSPoint mouse_pos = [NSEvent mouseLocation];
	const float scale = screen_get_max_scale();

	for (NSScreen *screen in [NSScreen screens]) {
		NSRect frame = [screen frame];
		if (NSMouseInRect(mouse_pos, frame, NO)) {
			return Vector2i((int)mouse_pos.x, (int)-mouse_pos.y) * scale + _get_screens_origin();
		}
	}
	return Vector2i();
}

MouseButton DisplayServerOSX::mouse_get_button_state() const {
	return last_button_state;
}

void DisplayServerOSX::clipboard_set(const String &p_text) {
	_THREAD_SAFE_METHOD_

	NSString *copiedString = [NSString stringWithUTF8String:p_text.utf8().get_data()];
	NSArray *copiedStringArray = [NSArray arrayWithObject:copiedString];

	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	[pasteboard clearContents];
	[pasteboard writeObjects:copiedStringArray];
}

String DisplayServerOSX::clipboard_get() const {
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

	char *utfs = strdup([string UTF8String]);
	String ret;
	ret.parse_utf8(utfs);
	free(utfs);

	return ret;
}

int DisplayServerOSX::get_screen_count() const {
	_THREAD_SAFE_METHOD_

	NSArray *screenArray = [NSScreen screens];
	return [screenArray count];
}

// Returns the native top-left screen coordinate of the smallest rectangle
// that encompasses all screens. Needed in get_screen_position(),
// window_get_position, and window_set_position()
// to convert between OS X native screen coordinates and the ones expected by Godot

static bool displays_arrangement_dirty = true;
static bool displays_scale_dirty = true;
static void displays_arrangement_changed(CGDirectDisplayID display_id, CGDisplayChangeSummaryFlags flags, void *user_info) {
	displays_arrangement_dirty = true;
	displays_scale_dirty = true;
}

Point2i DisplayServerOSX::_get_screens_origin() const {
	static Point2i origin;

	if (displays_arrangement_dirty) {
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

	return origin;
}

Point2i DisplayServerOSX::_get_native_screen_position(int p_screen) const {
	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		// Return the top-left corner of the screen, for OS X the y starts at the bottom
		return Point2i(nsrect.origin.x, nsrect.origin.y + nsrect.size.height) * screen_get_max_scale();
	}

	return Point2i();
}

Point2i DisplayServerOSX::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	Point2i position = _get_native_screen_position(p_screen) - _get_screens_origin();
	// OS X native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value
	position.y *= -1;
	return position;
}

Size2i DisplayServerOSX::screen_get_size(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		// Note: Use frame to get the whole screen size
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		return Size2i(nsrect.size.width, nsrect.size.height) * screen_get_max_scale();
	}

	return Size2i();
}

int DisplayServerOSX::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

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

float DisplayServerOSX::screen_get_scale(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}
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

float DisplayServerOSX::screen_get_max_scale() const {
	_THREAD_SAFE_METHOD_

	static float scale = 1.f;
	if (displays_scale_dirty) {
		int screen_count = get_screen_count();
		for (int i = 0; i < screen_count; i++) {
			scale = fmax(scale, screen_get_scale(i));
		}
		displays_scale_dirty = false;
	}
	return scale;
}

Rect2i DisplayServerOSX::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	if (p_screen == SCREEN_OF_MAIN_WINDOW) {
		p_screen = window_get_current_screen();
	}

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

Vector<DisplayServer::WindowID> DisplayServerOSX::get_window_list() const {
	_THREAD_SAFE_METHOD_

	Vector<int> ret;
	for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
		ret.push_back(E->key());
	}
	return ret;
}

DisplayServer::WindowID DisplayServerOSX::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	WindowID id = _create_window(p_mode, p_vsync_mode, p_rect);
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, id);
		}
	}

	return id;
}

void DisplayServerOSX::show_window(WindowID p_id) {
	WindowData &wd = windows[p_id];

	if (wd.no_focus) {
		[wd.window_object orderFront:nil];
	} else {
		[wd.window_object makeKeyAndOrderFront:nil];
	}
}

void DisplayServerOSX::_send_window_event(const WindowData &wd, WindowEvent p_event) {
	_THREAD_SAFE_METHOD_

	if (!wd.event_callback.is_null()) {
		Variant event = int(p_event);
		Variant *eventp = &event;
		Variant ret;
		Callable::CallError ce;
		wd.event_callback.call((const Variant **)&eventp, 1, ret, ce);
	}
}

DisplayServerOSX::WindowID DisplayServerOSX::_find_window_id(id p_window) {
	for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
		if (E->get().window_object == p_window) {
			return E->key();
		}
	}
	return INVALID_WINDOW_ID;
}

void DisplayServerOSX::_update_window(WindowData p_wd) {
	bool borderless_full = false;

	if (p_wd.borderless) {
		NSRect frameRect = [p_wd.window_object frame];
		NSRect screenRect = [[p_wd.window_object screen] frame];

		// Check if our window covers up the screen
		if (frameRect.origin.x <= screenRect.origin.x && frameRect.origin.y <= frameRect.origin.y &&
				frameRect.size.width >= screenRect.size.width && frameRect.size.height >= screenRect.size.height) {
			borderless_full = true;
		}
	}

	if (borderless_full) {
		// If the window covers up the screen set the level to above the main menu and hide on deactivate
		[p_wd.window_object setLevel:NSMainMenuWindowLevel + 1];
		[p_wd.window_object setHidesOnDeactivate:YES];
	} else {
		// Reset these when our window is not a borderless window that covers up the screen
		if (p_wd.on_top && !p_wd.fullscreen) {
			[p_wd.window_object setLevel:NSFloatingWindowLevel];
		} else {
			[p_wd.window_object setLevel:NSNormalWindowLevel];
		}
		[p_wd.window_object setHidesOnDeactivate:NO];
	}
}

void DisplayServerOSX::delete_sub_window(WindowID p_id) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_id));
	ERR_FAIL_COND_MSG(p_id == MAIN_WINDOW_ID, "Main window can't be deleted");

	WindowData &wd = windows[p_id];

	[wd.window_object setContentView:nil];
	[wd.window_object close];
}

void DisplayServerOSX::window_set_title(const String &p_title, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	[wd.window_object setTitle:[NSString stringWithUTF8String:p_title.utf8().get_data()]];
}

void DisplayServerOSX::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.mpath = p_region;
}

void DisplayServerOSX::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.rect_changed_callback = p_callable;
}

void DisplayServerOSX::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.event_callback = p_callable;
}

void DisplayServerOSX::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_event_callback = p_callable;
}

void DisplayServerOSX::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.input_text_callback = p_callable;
}

void DisplayServerOSX::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	wd.drop_files_callback = p_callable;
}

int DisplayServerOSX::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND_V(!windows.has(p_window), -1);
	const WindowData &wd = windows[p_window];

	const NSUInteger index = [[NSScreen screens] indexOfObject:[wd.window_object screen]];
	return (index == NSNotFound) ? 0 : index;
}

void DisplayServerOSX::window_set_current_screen(int p_screen, WindowID p_window) {
	_THREAD_SAFE_METHOD_
	Point2i wpos = window_get_position(p_window) - screen_get_position(window_get_current_screen(p_window));
	window_set_position(wpos + screen_get_position(p_screen), p_window);
}

void DisplayServerOSX::window_set_transient(WindowID p_window, WindowID p_parent) {
	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(p_window == p_parent);

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd_window = windows[p_window];

	ERR_FAIL_COND(wd_window.transient_parent == p_parent);

	ERR_FAIL_COND_MSG(wd_window.on_top, "Windows with the 'on top' can't become transient.");
	if (p_parent == INVALID_WINDOW_ID) {
		//remove transient
		ERR_FAIL_COND(wd_window.transient_parent == INVALID_WINDOW_ID);
		ERR_FAIL_COND(!windows.has(wd_window.transient_parent));

		WindowData &wd_parent = windows[wd_window.transient_parent];

		wd_window.transient_parent = INVALID_WINDOW_ID;
		wd_parent.transient_children.erase(p_window);

		[wd_parent.window_object removeChildWindow:wd_window.window_object];
	} else {
		ERR_FAIL_COND(!windows.has(p_parent));
		ERR_FAIL_COND_MSG(wd_window.transient_parent != INVALID_WINDOW_ID, "Window already has a transient parent");
		WindowData &wd_parent = windows[p_parent];

		wd_window.transient_parent = p_parent;
		wd_parent.transient_children.insert(p_window);

		[wd_parent.window_object addChildWindow:wd_window.window_object ordered:NSWindowAbove];
	}
}

Point2i DisplayServerOSX::window_get_position(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());
	const WindowData &wd = windows[p_window];

	NSRect nsrect = [wd.window_object frame];
	Point2i pos;

	// Return the position of the top-left corner, for OS X the y starts at the bottom
	const float scale = screen_get_max_scale();
	pos.x = nsrect.origin.x;
	pos.y = (nsrect.origin.y + nsrect.size.height);
	pos *= scale;
	pos -= _get_screens_origin();
	// OS X native y-coordinate relative to _get_screens_origin() is negative,
	// Godot expects a positive value
	pos.y *= -1;
	return pos;
}

void DisplayServerOSX::window_set_position(const Point2i &p_position, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	Point2i position = p_position;
	// OS X native y-coordinate relative to _get_screens_origin() is negative,
	// Godot passes a positive value
	position.y *= -1;
	position += _get_screens_origin();
	position /= screen_get_max_scale();

	[wd.window_object setFrameTopLeftPoint:NSMakePoint(position.x, position.y)];

	_update_window(wd);
	_get_mouse_pos(wd, [wd.window_object mouseLocationOutsideOfEventStream]);
}

void DisplayServerOSX::window_set_max_size(const Size2i p_size, WindowID p_window) {
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

Size2i DisplayServerOSX::window_get_max_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.max_size;
}

void DisplayServerOSX::window_set_min_size(const Size2i p_size, WindowID p_window) {
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

Size2i DisplayServerOSX::window_get_min_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	return wd.min_size;
}

void DisplayServerOSX::window_set_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	Size2i size = p_size / screen_get_max_scale();

	NSPoint top_left;
	NSRect old_frame = [wd.window_object frame];
	top_left.x = old_frame.origin.x;
	top_left.y = NSMaxY(old_frame);

	NSRect new_frame = NSMakeRect(0, 0, size.x, size.y);
	new_frame = [wd.window_object frameRectForContentRect:new_frame];

	new_frame.origin.x = top_left.x;
	new_frame.origin.y = top_left.y - new_frame.size.height;

	[wd.window_object setFrame:new_frame display:YES];

	_update_window(wd);
}

Size2i DisplayServerOSX::window_get_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.size;
}

Size2i DisplayServerOSX::window_get_real_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	NSRect frame = [wd.window_object frame];
	return Size2i(frame.size.width, frame.size.height) * screen_get_max_scale();
}

bool DisplayServerOSX::window_is_maximize_allowed(WindowID p_window) const {
	return true;
}

void DisplayServerOSX::_set_window_per_pixel_transparency_enabled(bool p_enabled, WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (!OS_OSX::get_singleton()->is_layered_allowed()) {
		return;
	}
	if (wd.layered_window != p_enabled) {
		if (p_enabled) {
			[wd.window_object setBackgroundColor:[NSColor clearColor]];
			[wd.window_object setOpaque:NO];
			[wd.window_object setHasShadow:NO];
			CALayer *layer = [wd.window_view layer];
			if (layer) {
				[layer setOpaque:NO];
			}
#if defined(VULKAN_ENABLED)
			if (context_vulkan) {
				//TODO - implement transparency for Vulkan
			}
#endif
#if defined(GLES3_ENABLED)
			if (gl_manager) {
				//TODO - reimplement OpenGLES
			}
#endif
			wd.layered_window = true;
		} else {
			[wd.window_object setBackgroundColor:[NSColor colorWithCalibratedWhite:1 alpha:1]];
			[wd.window_object setOpaque:YES];
			[wd.window_object setHasShadow:YES];
			CALayer *layer = [wd.window_view layer];
			if (layer) {
				[layer setOpaque:YES];
			}
#if defined(VULKAN_ENABLED)
			if (context_vulkan) {
				//TODO - implement transparency for Vulkan
			}
#endif
#if defined(GLES3_ENABLED)
			if (gl_manager) {
				//TODO - reimplement OpenGLES
			}
#endif
			wd.layered_window = false;
		}
#if defined(GLES3_ENABLED)
		if (gl_manager) {
			//TODO - reimplement OpenGLES
		}
#endif
#if defined(VULKAN_ENABLED)
		if (context_vulkan) {
			//TODO - implement transparency for Vulkan
		}
#endif
		NSRect frameRect = [wd.window_object frame];
		[wd.window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:YES];
		[wd.window_object setFrame:frameRect display:YES];
	}
}

void DisplayServerOSX::window_set_mode(WindowMode p_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	WindowMode old_mode = window_get_mode(p_window);
	if (old_mode == p_mode) {
		return; // do nothing
	}

	switch (old_mode) {
		case WINDOW_MODE_WINDOWED: {
			//do nothing
		} break;
		case WINDOW_MODE_MINIMIZED: {
			[wd.window_object deminiaturize:nil];
		} break;
		case WINDOW_MODE_FULLSCREEN: {
			[wd.window_object setLevel:NSNormalWindowLevel];
			if (wd.layered_window) {
				_set_window_per_pixel_transparency_enabled(true, p_window);
			}
			if (wd.resize_disabled) { //restore resize disabled
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
			wd.fullscreen = false;
		} break;
		case WINDOW_MODE_MAXIMIZED: {
			if ([wd.window_object isZoomed]) {
				[wd.window_object zoom:nil];
			}
		} break;
	}

	switch (p_mode) {
		case WINDOW_MODE_WINDOWED: {
			//do nothing
		} break;
		case WINDOW_MODE_MINIMIZED: {
			[wd.window_object performMiniaturize:nil];
		} break;
		case WINDOW_MODE_FULLSCREEN: {
			if (wd.layered_window)
				_set_window_per_pixel_transparency_enabled(false, p_window);
			if (wd.resize_disabled) //fullscreen window should be resizable to work
				[wd.window_object setStyleMask:[wd.window_object styleMask] | NSWindowStyleMaskResizable];
			[wd.window_object setContentMinSize:NSMakeSize(0, 0)];
			[wd.window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
			[wd.window_object toggleFullScreen:nil];
			wd.fullscreen = true;
		} break;
		case WINDOW_MODE_MAXIMIZED: {
			if (![wd.window_object isZoomed]) {
				[wd.window_object zoom:nil];
			}
		} break;
	}
}

DisplayServer::WindowMode DisplayServerOSX::window_get_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), WINDOW_MODE_WINDOWED);
	const WindowData &wd = windows[p_window];

	if (wd.fullscreen) { //if fullscreen, it's not in another mode
		return WINDOW_MODE_FULLSCREEN;
	}
	if ([wd.window_object isZoomed] && !wd.resize_disabled) {
		return WINDOW_MODE_MAXIMIZED;
	}
	if ([wd.window_object respondsToSelector:@selector(isMiniaturized)]) {
		if ([wd.window_object isMiniaturized]) {
			return WINDOW_MODE_MINIMIZED;
		}
	}

	// all other discarded, return windowed.
	return WINDOW_MODE_WINDOWED;
}

void DisplayServerOSX::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			wd.resize_disabled = p_enabled;
			if (wd.fullscreen) { //fullscreen window should be resizable, style will be applied on exiting fs
				return;
			}
			if (p_enabled) {
				[wd.window_object setStyleMask:[wd.window_object styleMask] & ~NSWindowStyleMaskResizable];
			} else {
				[wd.window_object setStyleMask:[wd.window_object styleMask] | NSWindowStyleMaskResizable];
			}
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			// OrderOut prevents a lose focus bug with the window
			if ([wd.window_object isVisible]) {
				[wd.window_object orderOut:nil];
			}
			wd.borderless = p_enabled;
			if (p_enabled) {
				[wd.window_object setStyleMask:NSWindowStyleMaskBorderless];
			} else {
				if (wd.layered_window)
					_set_window_per_pixel_transparency_enabled(false, p_window);
				[wd.window_object setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (wd.resize_disabled ? 0 : NSWindowStyleMaskResizable)];
				// Force update of the window styles
				NSRect frameRect = [wd.window_object frame];
				[wd.window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
				[wd.window_object setFrame:frameRect display:NO];
			}
			_update_window(wd);
			if ([wd.window_object isVisible]) {
				if (wd.no_focus) {
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
				[wd.window_object setLevel:NSFloatingWindowLevel];
			} else {
				[wd.window_object setLevel:NSNormalWindowLevel];
			}
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			wd.layered_window = p_enabled;
			if (p_enabled) {
				[wd.window_object setStyleMask:NSWindowStyleMaskBorderless]; // force borderless
			} else if (!wd.borderless) {
				[wd.window_object setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (wd.resize_disabled ? 0 : NSWindowStyleMaskResizable)];
			}
			_set_window_per_pixel_transparency_enabled(p_enabled, p_window);
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			wd.no_focus = p_enabled;
		} break;
		default: {
		}
	}
}

bool DisplayServerOSX::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];

	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			return wd.resize_disabled;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			return [wd.window_object styleMask] == NSWindowStyleMaskBorderless;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			if (wd.fullscreen) {
				return wd.on_top;
			} else {
				return [wd.window_object level] == NSFloatingWindowLevel;
			}
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			return wd.layered_window;
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			return wd.no_focus;
		} break;
		default: {
		}
	}

	return false;
}

void DisplayServerOSX::window_request_attention(WindowID p_window) {
	// It's app global, ignore window id.
	[NSApp requestUserAttention:NSCriticalRequest];
}

void DisplayServerOSX::window_move_to_foreground(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	const WindowData &wd = windows[p_window];

	[[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
	if (wd.no_focus) {
		[wd.window_object orderFront:nil];
	} else {
		[wd.window_object makeKeyAndOrderFront:nil];
	}
}

bool DisplayServerOSX::window_can_draw(WindowID p_window) const {
	return window_get_mode(p_window) != WINDOW_MODE_MINIMIZED;
}

bool DisplayServerOSX::can_any_window_draw() const {
	_THREAD_SAFE_METHOD_

	for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
		if (window_get_mode(E->key()) != WINDOW_MODE_MINIMIZED) {
			return true;
		}
	}
	return false;
}

void DisplayServerOSX::window_set_ime_active(const bool p_active, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_active = p_active;

	if (!p_active) {
		[wd.window_view cancelComposition];
	}
}

void DisplayServerOSX::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_position = p_pos;
}

bool DisplayServerOSX::get_swap_cancel_ok() {
	return false;
}

void DisplayServerOSX::cursor_set_shape(CursorShape p_shape) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (cursor_shape == p_shape) {
		return;
	}

	if (mouse_mode != MOUSE_MODE_VISIBLE && mouse_mode != MOUSE_MODE_CONFINED) {
		cursor_shape = p_shape;
		return;
	}

	if (cursors[p_shape] != nullptr) {
		[cursors[p_shape] set];
	} else {
		switch (p_shape) {
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
				[_cursorFromSelector(@selector(_windowResizeNorthSouthCursor), @selector(resizeUpDownCursor)) set];
				break;
			case CURSOR_HSIZE:
				[_cursorFromSelector(@selector(_windowResizeEastWestCursor), @selector(resizeLeftRightCursor)) set];
				break;
			case CURSOR_BDIAGSIZE:
				[_cursorFromSelector(@selector(_windowResizeNorthEastSouthWestCursor)) set];
				break;
			case CURSOR_FDIAGSIZE:
				[_cursorFromSelector(@selector(_windowResizeNorthWestSouthEastCursor)) set];
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
				[_cursorFromSelector(@selector(_helpCursor)) set];
				break;
			default: {
			}
		}
	}

	cursor_shape = p_shape;
}

DisplayServerOSX::CursorShape DisplayServerOSX::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerOSX::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	_THREAD_SAFE_METHOD_

	if (p_cursor.is_valid()) {
		Map<CursorShape, Vector<Variant>>::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}
			cursors_cache.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_image();
		}

		if (!image.is_valid() && atlas_texture.is_valid()) {
			texture = atlas_texture->get_atlas();

			atlas_rect.size.width = texture->get_width();
			atlas_rect.size.height = texture->get_height();
			atlas_rect.position.x = atlas_texture->get_region().position.x;
			atlas_rect.position.y = atlas_texture->get_region().position.y;

			texture_size.width = atlas_texture->get_region().size.x;
			texture_size.height = atlas_texture->get_region().size.y;
		} else if (image.is_valid()) {
			texture_size.width = texture->get_width();
			texture_size.height = texture->get_height();
		}

		ERR_FAIL_COND(!texture.is_valid());
		ERR_FAIL_COND(p_hotspot.x < 0 || p_hotspot.y < 0);
		ERR_FAIL_COND(texture_size.width > 256 || texture_size.height > 256);
		ERR_FAIL_COND(p_hotspot.x > texture_size.width || p_hotspot.y > texture_size.height);

		image = texture->get_image();

		ERR_FAIL_COND(!image.is_valid());

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

		ERR_FAIL_COND(imgrep == nil);
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

		[cursors[p_shape] release];
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

		[imgrep release];
		[nsimage release];
	} else {
		// Reset to default system cursor
		if (cursors[p_shape] != nullptr) {
			[cursors[p_shape] release];
			cursors[p_shape] = nullptr;
		}

		CursorShape c = cursor_shape;
		cursor_shape = CURSOR_MAX;
		cursor_set_shape(c);

		cursors_cache.erase(p_shape);
	}
}

struct LayoutInfo {
	String name;
	String code;
};

static Vector<LayoutInfo> kbd_layouts;
static int current_layout = 0;
static bool keyboard_layout_dirty = true;
static void keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info) {
	kbd_layouts.clear();
	current_layout = 0;
	keyboard_layout_dirty = true;
}

void _update_keyboard_layouts() {
	@autoreleasepool {
		TISInputSourceRef cur_source = TISCopyCurrentKeyboardInputSource();
		NSString *cur_name = (NSString *)TISGetInputSourceProperty(cur_source, kTISPropertyLocalizedName);
		CFRelease(cur_source);

		// Enum IME layouts
		NSDictionary *filter_ime = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardInputMode };
		NSArray *list_ime = (NSArray *)TISCreateInputSourceList((CFDictionaryRef)filter_ime, false);
		for (NSUInteger i = 0; i < [list_ime count]; i++) {
			LayoutInfo ly;
			NSString *name = (NSString *)TISGetInputSourceProperty((TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyLocalizedName);
			ly.name.parse_utf8([name UTF8String]);

			NSArray *langs = (NSArray *)TISGetInputSourceProperty((TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyInputSourceLanguages);
			ly.code.parse_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
			kbd_layouts.push_back(ly);

			if ([name isEqualToString:cur_name]) {
				current_layout = kbd_layouts.size() - 1;
			}
		}
		[list_ime release];

		// Enum plain keyboard layouts
		NSDictionary *filter_kbd = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardLayout };
		NSArray *list_kbd = (NSArray *)TISCreateInputSourceList((CFDictionaryRef)filter_kbd, false);
		for (NSUInteger i = 0; i < [list_kbd count]; i++) {
			LayoutInfo ly;
			NSString *name = (NSString *)TISGetInputSourceProperty((TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyLocalizedName);
			ly.name.parse_utf8([name UTF8String]);

			NSArray *langs = (NSArray *)TISGetInputSourceProperty((TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyInputSourceLanguages);
			ly.code.parse_utf8([(NSString *)[langs objectAtIndex:0] UTF8String]);
			kbd_layouts.push_back(ly);

			if ([name isEqualToString:cur_name]) {
				current_layout = kbd_layouts.size() - 1;
			}
		}
		[list_kbd release];
	}

	keyboard_layout_dirty = false;
}

int DisplayServerOSX::keyboard_get_layout_count() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}
	return kbd_layouts.size();
}

void DisplayServerOSX::keyboard_set_current_layout(int p_index) {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX(p_index, kbd_layouts.size());

	NSString *cur_name = [NSString stringWithUTF8String:kbd_layouts[p_index].name.utf8().get_data()];

	NSDictionary *filter_kbd = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardLayout };
	NSArray *list_kbd = (NSArray *)TISCreateInputSourceList((CFDictionaryRef)filter_kbd, false);
	for (NSUInteger i = 0; i < [list_kbd count]; i++) {
		NSString *name = (NSString *)TISGetInputSourceProperty((TISInputSourceRef)[list_kbd objectAtIndex:i], kTISPropertyLocalizedName);
		if ([name isEqualToString:cur_name]) {
			TISSelectInputSource((TISInputSourceRef)[list_kbd objectAtIndex:i]);
			break;
		}
	}
	[list_kbd release];

	NSDictionary *filter_ime = @{ (NSString *)kTISPropertyInputSourceType : (NSString *)kTISTypeKeyboardInputMode };
	NSArray *list_ime = (NSArray *)TISCreateInputSourceList((CFDictionaryRef)filter_ime, false);
	for (NSUInteger i = 0; i < [list_ime count]; i++) {
		NSString *name = (NSString *)TISGetInputSourceProperty((TISInputSourceRef)[list_ime objectAtIndex:i], kTISPropertyLocalizedName);
		if ([name isEqualToString:cur_name]) {
			TISSelectInputSource((TISInputSourceRef)[list_ime objectAtIndex:i]);
			break;
		}
	}
	[list_ime release];
}

int DisplayServerOSX::keyboard_get_current_layout() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	return current_layout;
}

String DisplayServerOSX::keyboard_get_layout_language(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].code;
}

String DisplayServerOSX::keyboard_get_layout_name(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].name;
}

Key DisplayServerOSX::keyboard_get_keycode_from_physical(Key p_keycode) const {
	if (p_keycode == Key::PAUSE) {
		return p_keycode;
	}

	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = p_keycode & KeyModifierMask::CODE_MASK;
	unsigned int osx_keycode = unmapKey((Key)keycode_no_mod);
	return (Key)(remapKey(osx_keycode, 0) | modifiers);
}

void DisplayServerOSX::_push_input(const Ref<InputEvent> &p_event) {
	Ref<InputEvent> ev = p_event;
	Input::get_singleton()->parse_input_event(ev);
}

void DisplayServerOSX::_release_pressed_events() {
	_THREAD_SAFE_METHOD_
	if (Input::get_singleton()) {
		Input::get_singleton()->release_pressed_events();
	}
}

NSMenu *DisplayServerOSX::_get_dock_menu() const {
	return dock_menu;
}

void DisplayServerOSX::_menu_callback(id p_sender) {
	if (![p_sender representedObject]) {
		return;
	}

	GlobalMenuItem *value = [p_sender representedObject];

	if (value) {
		if (value->checkable) {
			if ([p_sender state] == NSControlStateValueOff) {
				[p_sender setState:NSControlStateValueOn];
			} else {
				[p_sender setState:NSControlStateValueOff];
			}
		}

		if (value->callback != Callable()) {
			Variant tag = value->meta;
			Variant *tagp = &tag;
			Variant ret;
			Callable::CallError ce;
			value->callback.call((const Variant **)&tagp, 1, ret, ce);
		}
	}
}

void DisplayServerOSX::_send_event(NSEvent *p_event) {
	// special case handling of command-period, which is traditionally a special
	// shortcut in macOS and doesn't arrive at our regular keyDown handler.
	if ([p_event type] == NSEventTypeKeyDown) {
		if (([p_event modifierFlags] & NSEventModifierFlagCommand) && [p_event keyCode] == 0x2f) {
			Ref<InputEventKey> k;
			k.instantiate();

			_get_key_modifier_state([p_event modifierFlags], k);
			k->set_window_id(DisplayServerOSX::INVALID_WINDOW_ID);
			k->set_pressed(true);
			k->set_keycode(Key::PERIOD);
			k->set_physical_keycode(Key::PERIOD);
			k->set_echo([p_event isARepeat]);

			Input::get_singleton()->parse_input_event(k);
		}
	}
}

void DisplayServerOSX::_process_key_events() {
	Ref<InputEventKey> k;
	for (int i = 0; i < key_event_pos; i++) {
		const KeyEvent &ke = key_event_buffer[i];
		if (ke.raw) {
			// Non IME input - no composite characters, pass events as is
			k.instantiate();

			k->set_window_id(ke.window_id);
			_get_key_modifier_state(ke.osx_state, k);
			k->set_pressed(ke.pressed);
			k->set_echo(ke.echo);
			k->set_keycode(ke.keycode);
			k->set_physical_keycode((Key)ke.physical_keycode);
			k->set_unicode(ke.unicode);

			_push_input(k);
		} else {
			// IME input
			if ((i == 0 && ke.keycode == Key::NONE) || (i > 0 && key_event_buffer[i - 1].keycode == Key::NONE)) {
				k.instantiate();

				k->set_window_id(ke.window_id);
				_get_key_modifier_state(ke.osx_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_keycode(Key::NONE);
				k->set_physical_keycode(Key::NONE);
				k->set_unicode(ke.unicode);

				_push_input(k);
			}
			if (ke.keycode != Key::NONE) {
				k.instantiate();

				k->set_window_id(ke.window_id);
				_get_key_modifier_state(ke.osx_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_keycode(ke.keycode);
				k->set_physical_keycode((Key)ke.physical_keycode);

				if (i + 1 < key_event_pos && key_event_buffer[i + 1].keycode == Key::NONE) {
					k->set_unicode(key_event_buffer[i + 1].unicode);
				}

				_push_input(k);
			}
		}
	}

	key_event_pos = 0;
}

void DisplayServerOSX::process_events() {
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

	if (!drop_events) {
		_process_key_events();
		Input::get_singleton()->flush_buffered_events();
	}

	for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
		WindowData &wd = E->get();
		if (wd.mpath.size() > 0) {
			const Vector2 mpos = _get_mouse_pos(wd, [wd.window_object mouseLocationOutsideOfEventStream]);
			if (Geometry2D::is_point_in_polygon(mpos, wd.mpath)) {
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

	[autoreleasePool drain];
	autoreleasePool = [[NSAutoreleasePool alloc] init];
}

void DisplayServerOSX::force_process_and_drop_events() {
	_THREAD_SAFE_METHOD_

	drop_events = true;
	process_events();
	drop_events = false;
}

void DisplayServerOSX::set_native_icon(const String &p_filename) {
	_THREAD_SAFE_METHOD_

	FileAccess *f = FileAccess::open(p_filename, FileAccess::READ);
	ERR_FAIL_COND(!f);

	Vector<uint8_t> data;
	uint64_t len = f->get_length();
	data.resize(len);
	f->get_buffer((uint8_t *)&data.write[0], len);
	memdelete(f);

	NSData *icon_data = [[[NSData alloc] initWithBytes:&data.write[0] length:len] autorelease];
	ERR_FAIL_COND_MSG(!icon_data, "Error reading icon data.");

	NSImage *icon = [[[NSImage alloc] initWithData:icon_data] autorelease];
	ERR_FAIL_COND_MSG(!icon, "Error loading icon.");

	[NSApp setApplicationIconImage:icon];
}

void DisplayServerOSX::set_icon(const Ref<Image> &p_icon) {
	_THREAD_SAFE_METHOD_

	Ref<Image> img = p_icon;
	img = img->duplicate();
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
	ERR_FAIL_COND(imgrep == nil);
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
	ERR_FAIL_COND(nsimg == nil);

	[nsimg addRepresentation:imgrep];
	[NSApp setApplicationIconImage:nsimg];

	[imgrep release];
	[nsimg release];
}

void DisplayServerOSX::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->swap_buffers();
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
	}
#endif
}

DisplayServer::VSyncMode DisplayServerOSX::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		return (gl_manager->is_using_vsync() ? DisplayServer::VSyncMode::VSYNC_ENABLED : DisplayServer::VSyncMode::VSYNC_DISABLED);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		return context_vulkan->get_vsync_mode(p_window);
	}
#endif
	return DisplayServer::VSYNC_ENABLED;
}

Vector<String> DisplayServerOSX::get_rendering_drivers_func() {
	Vector<String> drivers;

#if defined(VULKAN_ENABLED)
	drivers.push_back("vulkan");
#endif
#if defined(GLES3_ENABLED)
	drivers.push_back("opengl3");
#endif

	return drivers;
}

void DisplayServerOSX::gl_window_make_current(DisplayServer::WindowID p_window_id) {
#if defined(GLES3_ENABLED)
	gl_manager->window_make_current(p_window_id);
#endif
}

Point2i DisplayServerOSX::ime_get_selection() const {
	return im_selection;
}

String DisplayServerOSX::ime_get_text() const {
	return im_text;
}

DisplayServer::WindowID DisplayServerOSX::get_window_at_screen_position(const Point2i &p_position) const {
	Point2i position = p_position;
	position.y *= -1;
	position += _get_screens_origin();
	position /= screen_get_max_scale();

	NSInteger wnum = [NSWindow windowNumberAtPoint:NSMakePoint(position.x, position.y) belowWindowWithWindowNumber:0 /*topmost*/];
	for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
		if ([E->get().window_object windowNumber] == wnum) {
			return E->key();
		}
	}
	return INVALID_WINDOW_ID;
}

void DisplayServerOSX::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].instance_id = p_instance;
}

ObjectID DisplayServerOSX::window_get_attached_instance_id(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), ObjectID());
	return windows[p_window].instance_id;
}

DisplayServer *DisplayServerOSX::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerOSX(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported Vulkan or OpenGL versions.", "Unable to initialize Video driver");
	}
	return ds;
}

DisplayServerOSX::WindowID DisplayServerOSX::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, const Rect2i &p_rect) {
	WindowID id;
	const float scale = screen_get_max_scale();
	{
		WindowData wd;

		wd.window_delegate = [[GodotWindowDelegate alloc] init];
		ERR_FAIL_COND_V_MSG(wd.window_delegate == nil, INVALID_WINDOW_ID, "Can't create a window delegate");
		[wd.window_delegate setWindowID:window_id_counter];

		Point2i position = p_rect.position;
		// OS X native y-coordinate relative to _get_screens_origin() is negative,
		// Godot passes a positive value
		position.y *= -1;
		position += _get_screens_origin();

		// initWithContentRect uses bottom-left corner of the window’s frame as origin.
		wd.window_object = [[GodotWindow alloc]
				initWithContentRect:NSMakeRect(position.x / scale, (position.y - p_rect.size.height) / scale, p_rect.size.width / scale, p_rect.size.height / scale)
						  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
							backing:NSBackingStoreBuffered
							  defer:NO];
		ERR_FAIL_COND_V_MSG(wd.window_object == nil, INVALID_WINDOW_ID, "Can't create a window");

		wd.window_view = [[GodotContentView alloc] init];
		ERR_FAIL_COND_V_MSG(wd.window_view == nil, INVALID_WINDOW_ID, "Can't create a window view");
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

		CALayer *layer = [wd.window_view layer];
		if (layer) {
			layer.contentsScale = scale;
		}

#if defined(VULKAN_ENABLED)
		if (context_vulkan) {
			Error err = context_vulkan->window_create(window_id_counter, p_vsync_mode, wd.window_view, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create a Vulkan context");
		}
#endif
#if defined(GLES3_ENABLED)
		if (gl_manager) {
			Error err = gl_manager->window_create(window_id_counter, wd.window_view, p_rect.size.width, p_rect.size.height);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Can't create an OpenGL context");
		}
#endif
		id = window_id_counter++;
		windows[id] = wd;
	}

	WindowData &wd = windows[id];
	window_set_mode(p_mode, id);

	const NSRect contentRect = [wd.window_view frame];
	wd.size.width = contentRect.size.width * scale;
	wd.size.height = contentRect.size.height * scale;

	CALayer *layer = [wd.window_view layer];
	if (layer) {
		layer.contentsScale = scale;
	}

#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_resize(id, wd.size.width, wd.size.height);
	}
#endif
#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(id, wd.size.width, wd.size.height);
	}
#endif

	return id;
}

void DisplayServerOSX::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerOSX *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerOSX::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	_THREAD_SAFE_METHOD_
	if (!in_dispatch_input_event) {
		in_dispatch_input_event = true;

		Variant ev = p_event;
		Variant *evp = &ev;
		Variant ret;
		Callable::CallError ce;

		Ref<InputEventFromWindow> event_from_window = p_event;
		if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
			//send to a window
			if (windows.has(event_from_window->get_window_id())) {
				Callable callable = windows[event_from_window->get_window_id()].input_event_callback;
				if (callable.is_null()) {
					return;
				}
				callable.call((const Variant **)&evp, 1, ret, ce);
			}
		} else {
			//send to all windows
			for (Map<WindowID, WindowData>::Element *E = windows.front(); E; E = E->next()) {
				Callable callable = E->get().input_event_callback;
				if (callable.is_null()) {
					continue;
				}
				callable.call((const Variant **)&evp, 1, ret, ce);
			}
		}

		in_dispatch_input_event = false;
	}
}

void DisplayServerOSX::release_rendering_thread() {
}

void DisplayServerOSX::make_rendering_thread() {
}

void DisplayServerOSX::swap_buffers() {
#if defined(GLES3_ENABLED)
	gl_manager->swap_buffers();
#endif
}

void DisplayServerOSX::console_set_visible(bool p_enabled) {
	//TODO - open terminal and redirect
}

bool DisplayServerOSX::is_console_visible() const {
	return isatty(STDIN_FILENO);
}

DisplayServerOSX::DisplayServerOSX(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
	drop_events = false;

	memset(cursors, 0, sizeof(cursors));
	cursor_shape = CURSOR_ARROW;

	key_event_pos = 0;
	mouse_mode = MOUSE_MODE_VISIBLE;

	autoreleasePool = [[NSAutoreleasePool alloc] init];

	eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
	ERR_FAIL_COND(!eventSource);

	CGEventSourceSetLocalEventsSuppressionInterval(eventSource, 0.0);

	keyboard_layout_dirty = true;
	displays_arrangement_dirty = true;
	displays_scale_dirty = true;

	// Register to be notified on keyboard layout changes
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
			nullptr, keyboard_layout_changed,
			kTISNotifySelectedKeyboardInputSourceChanged, nullptr,
			CFNotificationSuspensionBehaviorDeliverImmediately);

	// Register to be notified on displays arrangement changes
	CGDisplayRegisterReconfigurationCallback(displays_arrangement_changed, nullptr);

	NSMenuItem *menu_item;
	NSString *title;

	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname == nil) {
		nsappname = [[NSProcessInfo processInfo] processName];
	}

	// Setup Dock menu
	dock_menu = [[NSMenu alloc] initWithTitle:@"_dock"];

	// Setup Apple menu
	apple_menu = [[[NSMenu alloc] initWithTitle:@""] autorelease];
	title = [NSString stringWithFormat:NSLocalizedString(@"About %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(showAbout:) keyEquivalent:@""];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	NSMenu *services = [[NSMenu alloc] initWithTitle:@""];
	menu_item = [apple_menu addItemWithTitle:NSLocalizedString(@"Services", nil) action:nil keyEquivalent:@""];
	[apple_menu setSubmenu:services forItem:menu_item];
	[NSApp setServicesMenu:services];
	[services release];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	title = [NSString stringWithFormat:NSLocalizedString(@"Hide %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(hide:) keyEquivalent:@"h"];

	menu_item = [apple_menu addItemWithTitle:NSLocalizedString(@"Hide Others", nil) action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
	[menu_item setKeyEquivalentModifierMask:(NSEventModifierFlagOption | NSEventModifierFlagCommand)];

	[apple_menu addItemWithTitle:NSLocalizedString(@"Show all", nil) action:@selector(unhideAllApplications:) keyEquivalent:@""];

	[apple_menu addItem:[NSMenuItem separatorItem]];

	title = [NSString stringWithFormat:NSLocalizedString(@"Quit %@", nil), nsappname];
	[apple_menu addItemWithTitle:title action:@selector(terminate:) keyEquivalent:@"q"];

	// Add items to the menu bar
	NSMenu *main_menu = [NSApp mainMenu];
	menu_item = [main_menu addItemWithTitle:@"" action:nil keyEquivalent:@""];
	[main_menu setSubmenu:apple_menu forItem:menu_item];

	//!!!!!!!!!!!!!!!!!!!!!!!!!!
	//TODO - do Vulkan and OpenGL support checks, driver selection and fallback
	rendering_driver = p_rendering_driver;

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		GLManager_OSX::ContextType opengl_api_type = GLManager_OSX::GLES_3_0_COMPATIBLE;
		gl_manager = memnew(GLManager_OSX(opengl_api_type));
		if (gl_manager->initialize() != OK) {
			memdelete(gl_manager);
			gl_manager = nullptr;
			r_error = ERR_UNAVAILABLE;
			ERR_FAIL_MSG("Could not initialize OpenGL");
			return;
		}
	}
#endif
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextOSX);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_CANT_CREATE;
			ERR_FAIL_MSG("Could not initialize Vulkan");
		}
	}
#endif

	Point2i window_position(
			screen_get_position(0).x + (screen_get_size(0).width - p_resolution.width) / 2,
			screen_get_position(0).y + (screen_get_size(0).height - p_resolution.height) / 2);
	WindowID main_window = _create_window(p_mode, p_vsync_mode, Rect2i(window_position, p_resolution));
	ERR_FAIL_COND(main_window == INVALID_WINDOW_ID);
	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, main_window);
		}
	}
	show_window(MAIN_WINDOW_ID);

#if defined(GLES3_ENABLED)
	if (rendering_driver == "opengl3") {
		RasterizerGLES3::make_current();
	}
#endif
#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif
}

DisplayServerOSX::~DisplayServerOSX() {
	if (dock_menu) {
		[dock_menu release];
	}

	for (Map<String, NSMenu *>::Element *E = submenu.front(); E; E = E->next()) {
		[E->get() release];
	}

	//destroy all windows
	for (Map<WindowID, WindowData>::Element *E = windows.front(); E;) {
		Map<WindowID, WindowData>::Element *F = E;
		E = E->next();
		[F->get().window_object setContentView:nil];
		[F->get().window_object close];
	}

	//destroy drivers
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		memdelete(gl_manager);
		gl_manager = nullptr;
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
	CGDisplayRemoveReconfigurationCallback(displays_arrangement_changed, nullptr);

	cursors_cache.clear();
}

void DisplayServerOSX::register_osx_driver() {
	register_create_function("osx", create_func, get_rendering_drivers_func);
}
