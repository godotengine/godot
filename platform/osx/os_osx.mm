/**************************************************************************/
/*  os_osx.mm                                                             */
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

#include "os_osx.h"

#include "core/math/geometry.h"
#include "core/os/keyboard.h"
#include "core/print_string.h"
#include "core/version_generated.gen.h"
#include "dir_access_osx.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles3/rasterizer_gles3.h"
#include "main/main.h"
#include "servers/visual/visual_server_raster.h"

#ifdef SDL_ENABLED
#include "drivers/sdl/joypad_sdl.h"
#endif

#include <mach-o/dyld.h>
#include <os/log.h>
#include <sys/sysctl.h>

#include <Carbon/Carbon.h>
#import <Cocoa/Cocoa.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/hid/IOHIDKeys.h>
#include <IOKit/hid/IOHIDLib.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <libproc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tts_osx.h"

#if MAC_OS_X_VERSION_MAX_ALLOWED < 101200
#define NSEventMaskAny NSAnyEventMask
#define NSEventTypeKeyDown NSKeyDown
#define NSEventTypeKeyUp NSKeyUp
#define NSEventModifierFlagShift NSShiftKeyMask
#define NSEventModifierFlagCommand NSCommandKeyMask
#define NSEventModifierFlagControl NSControlKeyMask
#define NSEventModifierFlagOption NSAlternateKeyMask
#define NSWindowStyleMaskTitled NSTitledWindowMask
#define NSWindowStyleMaskResizable NSResizableWindowMask
#define NSWindowStyleMaskMiniaturizable NSMiniaturizableWindowMask
#define NSWindowStyleMaskClosable NSClosableWindowMask
#define NSWindowStyleMaskBorderless NSBorderlessWindowMask
#endif

#ifndef NSAppKitVersionNumber10_12
#define NSAppKitVersionNumber10_12 1504
#endif
#ifndef NSAppKitVersionNumber10_14
#define NSAppKitVersionNumber10_14 1671
#endif

static void get_key_modifier_state(unsigned int p_osx_state, Ref<InputEventWithModifiers> state) {
	state->set_shift((p_osx_state & NSEventModifierFlagShift));
	state->set_control((p_osx_state & NSEventModifierFlagControl));
	state->set_alt((p_osx_state & NSEventModifierFlagOption));
	state->set_metakey((p_osx_state & NSEventModifierFlagCommand));
}

static void push_to_key_event_buffer(const OS_OSX::KeyEvent &p_event) {
	Vector<OS_OSX::KeyEvent> &buffer = OS_OSX::singleton->key_event_buffer;
	if (OS_OSX::singleton->key_event_pos >= buffer.size()) {
		buffer.resize(1 + OS_OSX::singleton->key_event_pos);
	}
	buffer.write[OS_OSX::singleton->key_event_pos++] = p_event;
}

static int mouse_x = 0;
static int mouse_y = 0;
static int button_mask = 0;
static bool mouse_down_control = false;
static bool ignore_momentum_scroll = false;

static Vector2 get_mouse_pos(NSPoint locationInWindow) {
	const NSRect contentRect = [OS_OSX::singleton->window_view frame];
	const NSPoint p = locationInWindow;
	const float s = OS_OSX::singleton->get_screen_max_scale();
	mouse_x = p.x * s;
	mouse_y = (contentRect.size.height - p.y) * s;
	return Vector2(mouse_x, mouse_y);
}

static NSCursor *cursorFromSelector(SEL selector, SEL fallback = nil) {
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

@interface GodotApplication : NSApplication
@end

@implementation GodotApplication

- (void)sendEvent:(NSEvent *)event {
	// special case handling of command-period, which is traditionally a special
	// shortcut in macOS and doesn't arrive at our regular keyDown handler.
	if ([event type] == NSEventTypeKeyDown) {
		if ((([event modifierFlags] & NSEventModifierFlagDeviceIndependentFlagsMask) == NSEventModifierFlagCommand) && [event keyCode] == 0x2f) {
			Ref<InputEventKey> k;
			k.instance();

			get_key_modifier_state([event modifierFlags], k);
			k->set_pressed(true);
			k->set_scancode(KEY_PERIOD);
			k->set_physical_scancode(KEY_PERIOD);
			k->set_echo([event isARepeat]);

			OS_OSX::singleton->push_input(k);
		}
	}

	// From http://cocoadev.com/index.pl?GameKeyboardHandlingAlmost
	// This works around an AppKit bug, where key up events while holding
	// down the command key don't get sent to the key window.
	if ([event type] == NSEventTypeKeyUp && ([event modifierFlags] & NSEventModifierFlagCommand))
		[[self keyWindow] sendEvent:event];
	else
		[super sendEvent:event];
}

@end

@interface GodotApplicationDelegate : NSObject
- (void)forceUnbundledWindowActivationHackStep1;
- (void)forceUnbundledWindowActivationHackStep2;
- (void)forceUnbundledWindowActivationHackStep3;
@end

@implementation GodotApplicationDelegate

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication *)app {
	return YES;
}

- (void)forceUnbundledWindowActivationHackStep1 {
	// Step 1: Switch focus to macOS SystemUIServer process.
	// Required to perform step 2, TransformProcessType will fail if app is already the in focus.
	for (NSRunningApplication *app in [NSRunningApplication runningApplicationsWithBundleIdentifier:@"com.apple.systemuiserver"]) {
		[app activateWithOptions:NSApplicationActivateIgnoringOtherApps];
		break;
	}
	[self performSelector:@selector(forceUnbundledWindowActivationHackStep2) withObject:nil afterDelay:0.02];
}

- (void)forceUnbundledWindowActivationHackStep2 {
	// Step 2: Register app as foreground process.
	ProcessSerialNumber psn = { 0, kCurrentProcess };
	(void)TransformProcessType(&psn, kProcessTransformToForegroundApplication);

	[self performSelector:@selector(forceUnbundledWindowActivationHackStep3) withObject:nil afterDelay:0.02];
}

- (void)forceUnbundledWindowActivationHackStep3 {
	// Step 3: Switch focus back to app window.
	[[NSRunningApplication currentApplication] activateWithOptions:NSApplicationActivateIgnoringOtherApps];
}

- (void)applicationDidFinishLaunching:(NSNotification *)notice {
	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	NSString *nsbundleid_env = [NSString stringWithUTF8String:getenv("__CFBundleIdentifier")];
	NSString *nsbundleid = [[NSBundle mainBundle] bundleIdentifier];
	if (nsappname == nil || isatty(STDOUT_FILENO) || isatty(STDIN_FILENO) || isatty(STDERR_FILENO) || ![nsbundleid isEqualToString:nsbundleid_env]) {
		// If the executable is started from terminal or is not bundled, macOS WindowServer won't register and activate app window correctly (menu and title bar are grayed out and input ignored).
		[self performSelector:@selector(forceUnbundledWindowActivationHackStep1) withObject:nil afterDelay:0.02];
	}
}

- (void)globalMenuCallback:(id)sender {
	if (![sender representedObject])
		return;

	OS_OSX::GlobalMenuItem *item = (OS_OSX::GlobalMenuItem *)[[sender representedObject] pointerValue];

	if (!item)
		return;

	OS_OSX::singleton->main_loop->global_menu_action(item->signal, item->meta);
}

- (NSMenu *)applicationDockMenu:(NSApplication *)sender {
	NSMenu *menu = [[[NSMenu alloc] initWithTitle:@""] autorelease];

	Vector<OS_OSX::GlobalMenuItem> &E = OS_OSX::singleton->global_menus["_dock"];
	for (int i = 0; i < E.size(); i++) {
		if (E[i].label == String()) {
			[menu addItem:[NSMenuItem separatorItem]];
		} else {
			NSMenuItem *menu_item = [menu addItemWithTitle:[NSString stringWithUTF8String:E[i].label.utf8().get_data()] action:@selector(globalMenuCallback:) keyEquivalent:@""];
			[menu_item setRepresentedObject:[NSValue valueWithPointer:&(E[i])]];
		}
	}

	return menu;
}

- (BOOL)application:(NSApplication *)sender openFile:(NSString *)filename {
	// Note: may be called called before main loop init!
	char *utfs = strdup([filename UTF8String]);
	OS_OSX::singleton->open_with_filename.parse_utf8(utfs);
	free(utfs);

#ifdef TOOLS_ENABLED
	// Open new instance
	if (OS_OSX::singleton->get_main_loop()) {
		List<String> args;
		args.push_back(OS_OSX::singleton->open_with_filename);
		String exec = OS::get_singleton()->get_executable_path();

		OS::ProcessID pid = 0;
		OS::get_singleton()->execute(exec, args, false, &pid);
	}
#endif
	return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);

	return NSTerminateCancel;
}

- (void)applicationDidHide:(NSNotification *)notification {
	/*
	_Godotwindow* window;
	for (window = _Godot.windowListHead;  window;  window = window->next)
		_GodotInputWindowVisibility(window, GL_FALSE);
*/
}

- (void)applicationDidUnhide:(NSNotification *)notification {
	/*
	_Godotwindow* window;

	for (window = _Godot.windowListHead;  window;  window = window->next) {
		if ([window_object isVisible])
			_GodotInputWindowVisibility(window, GL_TRUE);
	}
*/
}

- (void)applicationDidChangeScreenParameters:(NSNotification *)notification {
	//_GodotInputMonitorChange();
}

- (void)showAbout:(id)sender {
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_ABOUT);
}

@end

@interface GodotWindowDelegate : NSObject {
	//_Godotwindow* window;
}

@end

@implementation GodotWindowDelegate

- (BOOL)windowShouldClose:(id)sender {
	//_GodotInputWindowCloseRequest(window);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);

	return NO;
}

- (void)windowDidEnterFullScreen:(NSNotification *)notification {
	OS_OSX::singleton->zoomed = true;

	[OS_OSX::singleton->window_object setContentMinSize:NSMakeSize(0, 0)];
	[OS_OSX::singleton->window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
	// Force window resize event.
	[self windowDidResize:notification];
}

- (void)windowDidExitFullScreen:(NSNotification *)notification {
	OS_OSX::singleton->zoomed = false;

	if (OS_OSX::singleton->min_size != Size2()) {
		Size2 size = OS_OSX::singleton->min_size / OS_OSX::singleton->get_screen_max_scale();
		[OS_OSX::singleton->window_object setContentMinSize:NSMakeSize(size.x, size.y)];
	}
	if (OS_OSX::singleton->max_size != Size2()) {
		Size2 size = OS_OSX::singleton->max_size / OS_OSX::singleton->get_screen_max_scale();
		[OS_OSX::singleton->window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
	}

	if (!OS_OSX::singleton->resizable)
		[OS_OSX::singleton->window_object setStyleMask:[OS_OSX::singleton->window_object styleMask] & ~NSWindowStyleMaskResizable];

	if (OS_OSX::singleton->on_top)
		[OS_OSX::singleton->window_object setLevel:NSFloatingWindowLevel];

	[NSApp setPresentationOptions:NSApplicationPresentationDefault];

	// Force window resize event.
	[self windowDidResize:notification];
}

- (void)windowDidChangeBackingProperties:(NSNotification *)notification {
	if (!OS_OSX::singleton)
		return;

	NSWindow *window = (NSWindow *)[notification object];
	CGFloat newBackingScaleFactor = [window backingScaleFactor];
	CGFloat oldBackingScaleFactor = [[[notification userInfo] objectForKey:@"NSBackingPropertyOldScaleFactorKey"] doubleValue];
	if (OS_OSX::singleton->is_hidpi_allowed()) {
		[OS_OSX::singleton->window_view setWantsBestResolutionOpenGLSurface:YES];
	} else {
		[OS_OSX::singleton->window_view setWantsBestResolutionOpenGLSurface:NO];
	}

	if (newBackingScaleFactor != oldBackingScaleFactor) {
		//Set new display scale and window size
		float newDisplayScale = OS_OSX::singleton->get_screen_max_scale();

		const NSRect contentRect = [OS_OSX::singleton->window_view frame];
		const NSRect fbRect = contentRect;

		OS_OSX::singleton->window_size.width = fbRect.size.width * newDisplayScale;
		OS_OSX::singleton->window_size.height = fbRect.size.height * newDisplayScale;

		if (OS_OSX::singleton->context) {
			GLint dim[2];
			dim[0] = OS_OSX::singleton->window_size.width;
			dim[1] = OS_OSX::singleton->window_size.height;
			CGLSetParameter((CGLContextObj)[OS_OSX::singleton->context CGLContextObj], kCGLCPSurfaceBackingSize, &dim[0]);
			CGLEnable((CGLContextObj)[OS_OSX::singleton->context CGLContextObj], kCGLCESurfaceBackingSize);
		}

		//Update context
		if (OS_OSX::singleton->main_loop) {
			//Force window resize event
			[self windowDidResize:notification];
		}
	}
}

- (void)windowWillStartLiveResize:(NSNotification *)notification {
	OS_OSX::singleton->is_resizing = true;
}

- (void)windowDidEndLiveResize:(NSNotification *)notification {
	OS_OSX::singleton->is_resizing = false;
}

- (void)windowDidResize:(NSNotification *)notification {
	[OS_OSX::singleton->context update];

	const NSRect contentRect = [OS_OSX::singleton->window_view frame];
	const NSRect fbRect = contentRect;

	float displayScale = OS_OSX::singleton->get_screen_max_scale();
	OS_OSX::singleton->window_size.width = fbRect.size.width * displayScale;
	OS_OSX::singleton->window_size.height = fbRect.size.height * displayScale;

	if (OS_OSX::singleton->context) {
		GLint dim[2];
		dim[0] = OS_OSX::singleton->window_size.width;
		dim[1] = OS_OSX::singleton->window_size.height;
		CGLSetParameter((CGLContextObj)[OS_OSX::singleton->context CGLContextObj], kCGLCPSurfaceBackingSize, &dim[0]);
		CGLEnable((CGLContextObj)[OS_OSX::singleton->context CGLContextObj], kCGLCESurfaceBackingSize);
	}

	/*
	_GodotInputFramebufferSize(window, fbRect.size.width, fbRect.size.height);
	_GodotInputWindowSize(window, contentRect.size.width, contentRect.size.height);
	_GodotInputWindowDamage(window);

	if (window->cursorMode == Godot_CURSOR_DISABLED)
		centerCursor(window);
*/
}

- (void)windowDidMove:(NSNotification *)notification {
	if (OS_OSX::singleton->get_main_loop()) {
		OS_OSX::singleton->input->release_pressed_events();
	}

	/*
	[window->nsgl.context update];

	int x, y;
	_GodotPlatformGetWindowPos(window, &x, &y);
	_GodotInputWindowPos(window, x, y);

	if (window->cursorMode == Godot_CURSOR_DISABLED)
		centerCursor(window);
*/
}

- (void)windowDidBecomeKey:(NSNotification *)notification {
	if (OS_OSX::singleton->get_main_loop()) {
		_ALLOW_DISCARD_ get_mouse_pos([OS_OSX::singleton->window_object mouseLocationOutsideOfEventStream]);
		OS_OSX::singleton->input->set_mouse_position(Point2(mouse_x, mouse_y));

		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	}

	OS_OSX::singleton->window_focused = true;
}

- (void)windowDidResignKey:(NSNotification *)notification {
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);

	OS_OSX::singleton->window_focused = false;
}

- (void)windowDidMiniaturize:(NSNotification *)notification {
	OS_OSX::singleton->wm_minimized(true);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);

	OS_OSX::singleton->window_focused = false;
};

- (void)windowDidDeminiaturize:(NSNotification *)notification {
	OS_OSX::singleton->wm_minimized(false);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);

	OS_OSX::singleton->window_focused = true;
};

- (void)windowDidChangeOcclusionState:(NSNotification *)notification {
	OS_OSX::singleton->is_visible = ([OS_OSX::singleton->window_object occlusionState] & NSWindowOcclusionStateVisible) && [OS_OSX::singleton->window_object isVisible];
}

@end

@interface GodotContentView : NSOpenGLView <NSTextInputClient> {
	NSTrackingArea *trackingArea;
	NSMutableAttributedString *markedText;
	bool imeInputEventInProgress;
}
- (void)cancelComposition;
- (BOOL)wantsUpdateLayer;
- (void)updateLayer;
@end

@implementation GodotContentView

- (void)drawRect:(NSRect)dirtyRect {
	if (OS_OSX::singleton->get_main_loop() && (OS_OSX::singleton->get_render_thread_mode() != OS::RENDER_SEPARATE_THREAD) && OS_OSX::singleton->is_resizing) {
		Main::force_redraw();
		if (!Main::is_iterating()) { // Avoid cyclic loop.
			Main::iteration();
		}
	}
}

- (void)setFrameSize:(NSSize)newSize {
	[super setFrameSize:newSize];
	[self setNeedsDisplay:YES]; // Force "drawRect" call.
}

+ (void)initialize {
	if (self == [GodotContentView class]) {
		// nothing left to do here at the moment..
	}
}

- (BOOL)wantsUpdateLayer {
	return YES;
}

- (void)updateLayer {
	[OS_OSX::singleton->context update];
}

- (id)init {
	self = [super init];
	trackingArea = nil;
	imeInputEventInProgress = false;
	[self updateTrackingAreas];
	[self registerForDraggedTypes:[NSArray arrayWithObject:NSPasteboardTypeFileURL]];
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
	if (OS_OSX::singleton->im_active) {
		imeInputEventInProgress = true;
		OS_OSX::singleton->im_text.parse_utf8([[markedText mutableString] UTF8String]);
		OS_OSX::singleton->im_selection = Point2(selectedRange.location, selectedRange.length);

		if (OS_OSX::singleton->get_main_loop())
			OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
	}
}

- (void)doCommandBySelector:(SEL)aSelector {
	if ([self respondsToSelector:aSelector])
		[self performSelector:aSelector];
}

- (void)unmarkText {
	imeInputEventInProgress = false;
	[[markedText mutableString] setString:@""];
	if (OS_OSX::singleton->im_active) {
		OS_OSX::singleton->im_text = String();
		OS_OSX::singleton->im_selection = Point2();

		if (OS_OSX::singleton->get_main_loop())
			OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_OS_IME_UPDATE);
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
	const NSRect contentRect = [OS_OSX::singleton->window_view frame];
	float displayScale = OS_OSX::singleton->get_screen_max_scale();
	NSRect pointInWindowRect = NSMakeRect(OS_OSX::singleton->im_position.x / displayScale, contentRect.size.height - (OS_OSX::singleton->im_position.y / displayScale) - 1, 0, 0);
	NSPoint pointOnScreen = [[OS_OSX::singleton->window_view window] convertRectToScreen:pointInWindowRect].origin;

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

	NSUInteger i, length = [characters length];

	NSCharacterSet *ctrlChars = [NSCharacterSet controlCharacterSet];
	NSCharacterSet *wsnlChars = [NSCharacterSet whitespaceAndNewlineCharacterSet];
	if ([characters rangeOfCharacterFromSet:ctrlChars].length && [characters rangeOfCharacterFromSet:wsnlChars].length == 0) {
		NSTextInputContext *currentInputContext = [NSTextInputContext currentInputContext];
		[currentInputContext discardMarkedText];
		[self cancelComposition];
		return;
	}

	for (i = 0; i < length; i++) {
		const unichar codepoint = [characters characterAtIndex:i];
		if ((codepoint & 0xFF00) == 0xF700)
			continue;

		OS_OSX::KeyEvent ke;

		ke.osx_state = [event modifierFlags];
		ke.pressed = true;
		ke.echo = false;
		ke.raw = false; // IME input event
		ke.scancode = 0;
		ke.physical_scancode = 0;
		ke.unicode = codepoint;

		push_to_key_event_buffer(ke);
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
	Vector<String> files;
	NSPasteboard *pboard = [sender draggingPasteboard];

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

	if (files.size()) {
		OS_OSX::singleton->main_loop->drop_files(files, 0);
		OS_OSX::singleton->move_window_to_foreground();
	}

	return NO;
}

- (BOOL)isOpaque {
	return YES;
}

- (BOOL)canBecomeKeyView {
	return YES;
}

- (BOOL)acceptsFirstResponder {
	return YES;
}

- (void)cursorUpdate:(NSEvent *)event {
	OS::CursorShape p_shape = OS_OSX::singleton->cursor_shape;
	OS_OSX::singleton->cursor_shape = OS::CURSOR_MAX;
	OS_OSX::singleton->set_cursor_shape(p_shape);
}

static void _mouseDownEvent(NSEvent *event, int index, int mask, bool pressed) {
	if (pressed) {
		button_mask |= mask;
	} else {
		button_mask &= ~mask;
	}

	Ref<InputEventMouseButton> mb;
	mb.instance();
	const Vector2 pos = get_mouse_pos([event locationInWindow]);
	get_key_modifier_state([event modifierFlags], mb);
	mb->set_button_index(index);
	mb->set_pressed(pressed);
	mb->set_position(pos);
	mb->set_global_position(pos);
	mb->set_button_mask(button_mask);
	if (index == BUTTON_LEFT && pressed) {
		mb->set_doubleclick([event clickCount] == 2);
	}
	OS_OSX::singleton->push_input(mb);
}

- (void)mouseDown:(NSEvent *)event {
	if (([event modifierFlags] & NSEventModifierFlagControl)) {
		mouse_down_control = true;
		_mouseDownEvent(event, BUTTON_RIGHT, BUTTON_MASK_RIGHT, true);
	} else {
		mouse_down_control = false;
		_mouseDownEvent(event, BUTTON_LEFT, BUTTON_MASK_LEFT, true);
	}
}

- (void)mouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)mouseUp:(NSEvent *)event {
	if (mouse_down_control) {
		_mouseDownEvent(event, BUTTON_RIGHT, BUTTON_MASK_RIGHT, false);
	} else {
		_mouseDownEvent(event, BUTTON_LEFT, BUTTON_MASK_LEFT, false);
	}
}

- (void)mouseMoved:(NSEvent *)event {
	NSPoint delta = NSMakePoint([event deltaX], [event deltaY]);
	NSPoint mpos = [event locationInWindow];

	if (OS_OSX::singleton->ignore_warp) {
		// Discard late events, before warp
		if (([event timestamp]) < OS_OSX::singleton->last_warp) {
			return;
		}
		OS_OSX::singleton->ignore_warp = false;
		return;
	}

	if (OS_OSX::singleton->mouse_mode == OS::MOUSE_MODE_CONFINED || OS_OSX::singleton->mouse_mode == OS::MOUSE_MODE_CONFINED_HIDDEN) {
		// Discard late events
		if (([event timestamp]) < OS_OSX::singleton->last_warp) {
			return;
		}

		// Warp affects next event delta, subtract previous warp deltas
		List<OS_OSX::WarpEvent>::Element *F = OS_OSX::singleton->warp_events.front();
		while (F) {
			if (F->get().timestamp < [event timestamp]) {
				List<OS_OSX::WarpEvent>::Element *E = F;
				delta.x -= E->get().delta.x;
				delta.y -= E->get().delta.y;
				F = F->next();
				OS_OSX::singleton->warp_events.erase(E);
			} else {
				F = F->next();
			}
		}

		// Confine mouse position to the window, and update delta
		NSRect frame = [OS_OSX::singleton->window_object frame];
		NSPoint conf_pos = mpos;
		conf_pos.x = CLAMP(conf_pos.x + delta.x, 0.f, frame.size.width);
		conf_pos.y = CLAMP(conf_pos.y - delta.y, 0.f, frame.size.height);
		delta.x = conf_pos.x - mpos.x;
		delta.y = mpos.y - conf_pos.y;
		mpos = conf_pos;

		// Move mouse cursor
		NSRect pointInWindowRect = NSMakeRect(conf_pos.x, conf_pos.y, 0, 0);
		conf_pos = [[OS_OSX::singleton->window_view window] convertRectToScreen:pointInWindowRect].origin;
		conf_pos.y = CGDisplayBounds(CGMainDisplayID()).size.height - conf_pos.y;
		CGWarpMouseCursorPosition(conf_pos);

		// Save warp data
		OS_OSX::singleton->last_warp = [[NSProcessInfo processInfo] systemUptime];
		OS_OSX::WarpEvent ev;
		ev.timestamp = OS_OSX::singleton->last_warp;
		ev.delta = delta;
		OS_OSX::singleton->warp_events.push_back(ev);
	}

	Ref<InputEventMouseMotion> mm;
	mm.instance();

	mm->set_button_mask(button_mask);
	const Vector2 pos = get_mouse_pos(mpos);
	mm->set_position(pos);
	mm->set_pressure([event pressure]);
	NSEventSubtype subtype = [event subtype];
	if (subtype == NSEventSubtypeTabletPoint) {
		const NSPoint p = [event tilt];
		mm->set_tilt(Vector2(p.x, -p.y));
		mm->set_pen_inverted(OS_OSX::singleton->last_pen_inverted);
	} else if (subtype == NSEventSubtypeTabletProximity) {
		// Check if using the eraser end of pen only on proximity event.
		OS_OSX::singleton->last_pen_inverted = [event pointingDeviceType] == NSPointingDeviceTypeEraser;
		mm->set_pen_inverted(OS_OSX::singleton->last_pen_inverted);
	}
	mm->set_global_position(pos);
	mm->set_speed(OS_OSX::singleton->input->get_last_mouse_speed());
	const Vector2 relativeMotion = Vector2(delta.x, delta.y) * OS_OSX::singleton->get_screen_max_scale();
	mm->set_relative(relativeMotion);
	get_key_modifier_state([event modifierFlags], mm);

	OS_OSX::singleton->push_input(mm);
}

- (void)rightMouseDown:(NSEvent *)event {
	_mouseDownEvent(event, BUTTON_RIGHT, BUTTON_MASK_RIGHT, true);
}

- (void)rightMouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)rightMouseUp:(NSEvent *)event {
	_mouseDownEvent(event, BUTTON_RIGHT, BUTTON_MASK_RIGHT, false);
}

- (void)otherMouseDown:(NSEvent *)event {
	if ((int)[event buttonNumber] == 2) {
		_mouseDownEvent(event, BUTTON_MIDDLE, BUTTON_MASK_MIDDLE, true);

	} else if ((int)[event buttonNumber] == 3) {
		_mouseDownEvent(event, BUTTON_XBUTTON1, BUTTON_MASK_XBUTTON1, true);

	} else if ((int)[event buttonNumber] == 4) {
		_mouseDownEvent(event, BUTTON_XBUTTON2, BUTTON_MASK_XBUTTON2, true);

	} else {
		return;
	}
}

- (void)otherMouseDragged:(NSEvent *)event {
	[self mouseMoved:event];
}

- (void)otherMouseUp:(NSEvent *)event {
	if ((int)[event buttonNumber] == 2) {
		_mouseDownEvent(event, BUTTON_MIDDLE, BUTTON_MASK_MIDDLE, false);

	} else if ((int)[event buttonNumber] == 3) {
		_mouseDownEvent(event, BUTTON_XBUTTON1, BUTTON_MASK_XBUTTON1, false);

	} else if ((int)[event buttonNumber] == 4) {
		_mouseDownEvent(event, BUTTON_XBUTTON2, BUTTON_MASK_XBUTTON2, false);

	} else {
		return;
	}
}

- (void)mouseExited:(NSEvent *)event {
	if (!OS_OSX::singleton)
		return;

	if (OS_OSX::singleton->main_loop && OS_OSX::singleton->mouse_mode != OS::MOUSE_MODE_CAPTURED)
		OS_OSX::singleton->main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_EXIT);
}

- (void)mouseEntered:(NSEvent *)event {
	if (!OS_OSX::singleton)
		return;
	if (OS_OSX::singleton->main_loop && OS_OSX::singleton->mouse_mode != OS::MOUSE_MODE_CAPTURED)
		OS_OSX::singleton->main_loop->notification(MainLoop::NOTIFICATION_WM_MOUSE_ENTER);

	OS::CursorShape p_shape = OS_OSX::singleton->cursor_shape;
	OS_OSX::singleton->cursor_shape = OS::CURSOR_MAX;
	OS_OSX::singleton->set_cursor_shape(p_shape);
}

- (void)magnifyWithEvent:(NSEvent *)event {
	Ref<InputEventMagnifyGesture> ev;
	ev.instance();
	get_key_modifier_state([event modifierFlags], ev);
	ev->set_position(get_mouse_pos([event locationInWindow]));
	ev->set_factor([event magnification] + 1.0);
	OS_OSX::singleton->push_input(ev);
}

- (void)viewDidChangeBackingProperties {
	// nothing left to do here
}

- (void)updateTrackingAreas {
	if (trackingArea != nil) {
		[self removeTrackingArea:trackingArea];
		[trackingArea release];
	}

	NSTrackingAreaOptions options =
			NSTrackingMouseEnteredAndExited |
			NSTrackingActiveInKeyWindow |
			NSTrackingCursorUpdate |
			NSTrackingInVisibleRect;

	trackingArea = [[NSTrackingArea alloc]
			initWithRect:[self bounds]
				 options:options
				   owner:self
				userInfo:nil];

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
		if (key == table[i])
			return true;
	}
	return false;
}

// Keyboard symbol translation table
static const unsigned int _osx_to_godot_table[128] = {
	/* 00 */ KEY_A,
	/* 01 */ KEY_S,
	/* 02 */ KEY_D,
	/* 03 */ KEY_F,
	/* 04 */ KEY_H,
	/* 05 */ KEY_G,
	/* 06 */ KEY_Z,
	/* 07 */ KEY_X,
	/* 08 */ KEY_C,
	/* 09 */ KEY_V,
	/* 0a */ KEY_SECTION, /* ISO Section */
	/* 0b */ KEY_B,
	/* 0c */ KEY_Q,
	/* 0d */ KEY_W,
	/* 0e */ KEY_E,
	/* 0f */ KEY_R,
	/* 10 */ KEY_Y,
	/* 11 */ KEY_T,
	/* 12 */ KEY_1,
	/* 13 */ KEY_2,
	/* 14 */ KEY_3,
	/* 15 */ KEY_4,
	/* 16 */ KEY_6,
	/* 17 */ KEY_5,
	/* 18 */ KEY_EQUAL,
	/* 19 */ KEY_9,
	/* 1a */ KEY_7,
	/* 1b */ KEY_MINUS,
	/* 1c */ KEY_8,
	/* 1d */ KEY_0,
	/* 1e */ KEY_BRACERIGHT,
	/* 1f */ KEY_O,
	/* 20 */ KEY_U,
	/* 21 */ KEY_BRACELEFT,
	/* 22 */ KEY_I,
	/* 23 */ KEY_P,
	/* 24 */ KEY_ENTER,
	/* 25 */ KEY_L,
	/* 26 */ KEY_J,
	/* 27 */ KEY_APOSTROPHE,
	/* 28 */ KEY_K,
	/* 29 */ KEY_SEMICOLON,
	/* 2a */ KEY_BACKSLASH,
	/* 2b */ KEY_COMMA,
	/* 2c */ KEY_SLASH,
	/* 2d */ KEY_N,
	/* 2e */ KEY_M,
	/* 2f */ KEY_PERIOD,
	/* 30 */ KEY_TAB,
	/* 31 */ KEY_SPACE,
	/* 32 */ KEY_QUOTELEFT,
	/* 33 */ KEY_BACKSPACE,
	/* 34 */ KEY_UNKNOWN,
	/* 35 */ KEY_ESCAPE,
	/* 36 */ KEY_META,
	/* 37 */ KEY_META,
	/* 38 */ KEY_SHIFT,
	/* 39 */ KEY_CAPSLOCK,
	/* 3a */ KEY_ALT,
	/* 3b */ KEY_CONTROL,
	/* 3c */ KEY_SHIFT,
	/* 3d */ KEY_ALT,
	/* 3e */ KEY_CONTROL,
	/* 3f */ KEY_UNKNOWN, /* Function */
	/* 40 */ KEY_UNKNOWN, /* F17 */
	/* 41 */ KEY_KP_PERIOD,
	/* 42 */ KEY_UNKNOWN,
	/* 43 */ KEY_KP_MULTIPLY,
	/* 44 */ KEY_UNKNOWN,
	/* 45 */ KEY_KP_ADD,
	/* 46 */ KEY_UNKNOWN,
	/* 47 */ KEY_NUMLOCK, /* Really KeypadClear... */
	/* 48 */ KEY_VOLUMEUP, /* VolumeUp */
	/* 49 */ KEY_VOLUMEDOWN, /* VolumeDown */
	/* 4a */ KEY_VOLUMEMUTE, /* Mute */
	/* 4b */ KEY_KP_DIVIDE,
	/* 4c */ KEY_KP_ENTER,
	/* 4d */ KEY_UNKNOWN,
	/* 4e */ KEY_KP_SUBTRACT,
	/* 4f */ KEY_UNKNOWN, /* F18 */
	/* 50 */ KEY_UNKNOWN, /* F19 */
	/* 51 */ KEY_EQUAL, /* KeypadEqual */
	/* 52 */ KEY_KP_0,
	/* 53 */ KEY_KP_1,
	/* 54 */ KEY_KP_2,
	/* 55 */ KEY_KP_3,
	/* 56 */ KEY_KP_4,
	/* 57 */ KEY_KP_5,
	/* 58 */ KEY_KP_6,
	/* 59 */ KEY_KP_7,
	/* 5a */ KEY_UNKNOWN, /* F20 */
	/* 5b */ KEY_KP_8,
	/* 5c */ KEY_KP_9,
	/* 5d */ KEY_YEN, /* JIS Yen */
	/* 5e */ KEY_UNDERSCORE, /* JIS Underscore */
	/* 5f */ KEY_COMMA, /* JIS KeypadComma */
	/* 60 */ KEY_F5,
	/* 61 */ KEY_F6,
	/* 62 */ KEY_F7,
	/* 63 */ KEY_F3,
	/* 64 */ KEY_F8,
	/* 65 */ KEY_F9,
	/* 66 */ KEY_UNKNOWN, /* JIS Eisu */
	/* 67 */ KEY_F11,
	/* 68 */ KEY_UNKNOWN, /* JIS Kana */
	/* 69 */ KEY_F13,
	/* 6a */ KEY_F16,
	/* 6b */ KEY_F14,
	/* 6c */ KEY_UNKNOWN,
	/* 6d */ KEY_F10,
	/* 6e */ KEY_MENU,
	/* 6f */ KEY_F12,
	/* 70 */ KEY_UNKNOWN,
	/* 71 */ KEY_F15,
	/* 72 */ KEY_INSERT, /* Really Help... */
	/* 73 */ KEY_HOME,
	/* 74 */ KEY_PAGEUP,
	/* 75 */ KEY_DELETE,
	/* 76 */ KEY_F4,
	/* 77 */ KEY_END,
	/* 78 */ KEY_F2,
	/* 79 */ KEY_PAGEDOWN,
	/* 7a */ KEY_F1,
	/* 7b */ KEY_LEFT,
	/* 7c */ KEY_RIGHT,
	/* 7d */ KEY_DOWN,
	/* 7e */ KEY_UP,
	/* 7f */ KEY_UNKNOWN,
};

// Translates a OS X keycode to a Godot keycode
static int translateKey(unsigned int key) {
	if (key >= 128)
		return KEY_UNKNOWN;

	return _osx_to_godot_table[key];
}

// Translates a Godot keycode back to a OSX keycode
static unsigned int unmapKey(unsigned int key) {
	for (int i = 0; i <= 126; i++) {
		if (_osx_to_godot_table[i] == key) {
			return i;
		}
	}
	return 127;
}

struct _KeyCodeMap {
	UniChar kchar;
	int kcode;
};

static const _KeyCodeMap _keycodes[55] = {
	{ '`', KEY_QUOTELEFT },
	{ '~', KEY_ASCIITILDE },
	{ '0', KEY_0 },
	{ '1', KEY_1 },
	{ '2', KEY_2 },
	{ '3', KEY_3 },
	{ '4', KEY_4 },
	{ '5', KEY_5 },
	{ '6', KEY_6 },
	{ '7', KEY_7 },
	{ '8', KEY_8 },
	{ '9', KEY_9 },
	{ '-', KEY_MINUS },
	{ '_', KEY_UNDERSCORE },
	{ '=', KEY_EQUAL },
	{ '+', KEY_PLUS },
	{ 'q', KEY_Q },
	{ 'w', KEY_W },
	{ 'e', KEY_E },
	{ 'r', KEY_R },
	{ 't', KEY_T },
	{ 'y', KEY_Y },
	{ 'u', KEY_U },
	{ 'i', KEY_I },
	{ 'o', KEY_O },
	{ 'p', KEY_P },
	{ '[', KEY_BRACELEFT },
	{ ']', KEY_BRACERIGHT },
	{ '{', KEY_BRACELEFT },
	{ '}', KEY_BRACERIGHT },
	{ 'a', KEY_A },
	{ 's', KEY_S },
	{ 'd', KEY_D },
	{ 'f', KEY_F },
	{ 'g', KEY_G },
	{ 'h', KEY_H },
	{ 'j', KEY_J },
	{ 'k', KEY_K },
	{ 'l', KEY_L },
	{ ';', KEY_SEMICOLON },
	{ ':', KEY_COLON },
	{ '\'', KEY_APOSTROPHE },
	{ '\"', KEY_QUOTEDBL },
	{ '\\', KEY_BACKSLASH },
	{ '#', KEY_NUMBERSIGN },
	{ 'z', KEY_Z },
	{ 'x', KEY_X },
	{ 'c', KEY_C },
	{ 'v', KEY_V },
	{ 'b', KEY_B },
	{ 'n', KEY_N },
	{ 'm', KEY_M },
	{ ',', KEY_COMMA },
	{ '.', KEY_PERIOD },
	{ '/', KEY_SLASH }
};

static int remapKey(unsigned int key, unsigned int state) {
	if (isNumpadKey(key))
		return translateKey(key);

	TISInputSourceRef currentKeyboard = TISCopyCurrentKeyboardInputSource();
	if (!currentKeyboard)
		return translateKey(key);

	CFDataRef layoutData = (CFDataRef)TISGetInputSourceProperty(currentKeyboard, kTISPropertyUnicodeKeyLayoutData);
	if (!layoutData)
		return translateKey(key);

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
	ignore_momentum_scroll = true;

	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		NSString *characters = [event characters];
		NSUInteger length = [characters length];

		if (!OS_OSX::singleton->im_active && length > 0 && keycode_has_unicode(remapKey([event keyCode], [event modifierFlags]))) {
			// Fallback unicode character handler used if IME is not active
			for (NSUInteger i = 0; i < length; i++) {
				OS_OSX::KeyEvent ke;

				ke.osx_state = [event modifierFlags];
				ke.pressed = true;
				ke.echo = [event isARepeat];
				ke.scancode = remapKey([event keyCode], [event modifierFlags]);
				ke.physical_scancode = translateKey([event keyCode]);
				ke.raw = true;
				ke.unicode = [characters characterAtIndex:i];

				push_to_key_event_buffer(ke);
			}
		} else {
			OS_OSX::KeyEvent ke;

			ke.osx_state = [event modifierFlags];
			ke.pressed = true;
			ke.echo = [event isARepeat];
			ke.scancode = remapKey([event keyCode], [event modifierFlags]);
			ke.physical_scancode = translateKey([event keyCode]);
			ke.raw = false;
			ke.unicode = 0;

			push_to_key_event_buffer(ke);
		}
	}

	// Pass events to IME handler
	if (OS_OSX::singleton->im_active)
		[self interpretKeyEvents:[NSArray arrayWithObject:event]];
}

- (void)flagsChanged:(NSEvent *)event {
	ignore_momentum_scroll = true;

	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		OS_OSX::KeyEvent ke;

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
		ke.scancode = remapKey(key, mod);
		ke.physical_scancode = translateKey(key);
		ke.unicode = 0;

		push_to_key_event_buffer(ke);
	}
}

- (void)keyUp:(NSEvent *)event {
	// Ignore all input if IME input is in progress
	if (!imeInputEventInProgress) {
		NSString *characters = [event characters];
		NSUInteger length = [characters length];

		// Fallback unicode character handler used if IME is not active
		if (!OS_OSX::singleton->im_active && length > 0 && keycode_has_unicode(remapKey([event keyCode], [event modifierFlags]))) {
			for (NSUInteger i = 0; i < length; i++) {
				OS_OSX::KeyEvent ke;

				ke.osx_state = [event modifierFlags];
				ke.pressed = false;
				ke.echo = [event isARepeat];
				ke.scancode = remapKey([event keyCode], [event modifierFlags]);
				ke.physical_scancode = translateKey([event keyCode]);
				ke.raw = true;
				ke.unicode = [characters characterAtIndex:i];

				push_to_key_event_buffer(ke);
			}
		} else {
			OS_OSX::KeyEvent ke;

			ke.osx_state = [event modifierFlags];
			ke.pressed = false;
			ke.echo = [event isARepeat];
			ke.scancode = remapKey([event keyCode], [event modifierFlags]);
			ke.physical_scancode = translateKey([event keyCode]);
			ke.raw = true;
			ke.unicode = 0;

			push_to_key_event_buffer(ke);
		}
	}
}

inline void sendScrollEvent(int button, double factor, int modifierFlags) {
	unsigned int mask = 1 << (button - 1);
	Vector2 mouse_pos = Vector2(mouse_x, mouse_y);

	Ref<InputEventMouseButton> sc;
	sc.instance();

	get_key_modifier_state(modifierFlags, sc);
	sc->set_button_index(button);
	sc->set_factor(factor);
	sc->set_pressed(true);
	sc->set_position(mouse_pos);
	sc->set_global_position(mouse_pos);
	button_mask |= mask;
	sc->set_button_mask(button_mask);
	OS_OSX::singleton->push_input(sc);

	sc.instance();
	sc->set_button_index(button);
	sc->set_factor(factor);
	sc->set_pressed(false);
	sc->set_position(mouse_pos);
	sc->set_global_position(mouse_pos);
	button_mask &= ~mask;
	sc->set_button_mask(button_mask);
	OS_OSX::singleton->push_input(sc);
}

inline void sendPanEvent(double dx, double dy, int modifierFlags) {
	Ref<InputEventPanGesture> pg;
	pg.instance();

	get_key_modifier_state(modifierFlags, pg);
	Vector2 mouse_pos = Vector2(mouse_x, mouse_y);
	pg->set_position(mouse_pos);
	pg->set_delta(Vector2(-dx, -dy));
	OS_OSX::singleton->push_input(pg);
}

- (void)scrollWheel:(NSEvent *)event {
	double deltaX, deltaY;

	_ALLOW_DISCARD_ get_mouse_pos([event locationInWindow]);

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
		sendPanEvent(deltaX, deltaY, [event modifierFlags]);
	} else {
		if (fabs(deltaX)) {
			sendScrollEvent(0 > deltaX ? BUTTON_WHEEL_RIGHT : BUTTON_WHEEL_LEFT, fabs(deltaX * 0.3), [event modifierFlags]);
		}
		if (fabs(deltaY)) {
			sendScrollEvent(0 < deltaY ? BUTTON_WHEEL_UP : BUTTON_WHEEL_DOWN, fabs(deltaY * 0.3), [event modifierFlags]);
		}
	}
}

@end

@interface GodotWindow : NSWindow {
}
@end

@implementation GodotWindow

- (BOOL)canBecomeKeyWindow {
	// Required for NSBorderlessWindowMask windows
	return YES;
}

@end

void OS_OSX::_update_global_menu() {
	NSMenu *main_menu = [NSApp mainMenu];

	for (int i = [main_menu numberOfItems] - 1; i > 0; i--) {
		[main_menu removeItemAtIndex:i];
	}
	for (List<String>::Element *E = global_menus_order.front(); E; E = E->next()) {
		Vector<GlobalMenuItem> &items = global_menus[E->get()];
		NSMenu *menu = [[[NSMenu alloc] initWithTitle:[NSString stringWithUTF8String:E->get().utf8().get_data()]] autorelease];
		for (int i = 0; i < items.size(); i++) {
			if (items[i].label == String()) {
				[menu addItem:[NSMenuItem separatorItem]];
			} else {
				NSMenuItem *menu_item = [menu addItemWithTitle:[NSString stringWithUTF8String:items[i].label.utf8().get_data()] action:@selector(globalMenuCallback:) keyEquivalent:@""];
				[menu_item setRepresentedObject:[NSValue valueWithPointer:&(items[i])]];
			}
		}
		NSMenuItem *menu_item = [main_menu addItemWithTitle:[NSString stringWithUTF8String:E->get().utf8().get_data()] action:nil keyEquivalent:@""];
		[main_menu setSubmenu:menu forItem:menu_item];
	}
}

void OS_OSX::global_menu_add_item(const String &p_menu, const String &p_label, const Variant &p_signal, const Variant &p_meta) {
	if (!global_menus.has(p_menu) && (p_menu != "_dock")) {
		global_menus_order.push_back(p_menu);
	}
	global_menus[p_menu].push_back(GlobalMenuItem(p_label, p_signal, p_meta));
	_update_global_menu();
}

void OS_OSX::global_menu_add_separator(const String &p_menu) {
	if (!global_menus.has(p_menu) && (p_menu != "_dock")) {
		global_menus_order.push_back(p_menu);
	}
	global_menus[p_menu].push_back(GlobalMenuItem());
	_update_global_menu();
}

void OS_OSX::global_menu_remove_item(const String &p_menu, int p_idx) {
	ERR_FAIL_INDEX(p_idx, global_menus[p_menu].size());

	global_menus[p_menu].remove(p_idx);
	_update_global_menu();
}

void OS_OSX::global_menu_clear(const String &p_menu) {
	if (global_menus.has(p_menu)) {
		global_menus[p_menu].clear();
		if (p_menu != "_dock") {
			global_menus.erase(p_menu);
			global_menus_order.erase(p_menu);
		}
	}
	_update_global_menu();
}

Point2 OS_OSX::get_ime_selection() const {
	return im_selection;
}

String OS_OSX::get_ime_text() const {
	return im_text;
}

String OS_OSX::get_unique_id() const {
	static String serial_number;

	if (serial_number.empty()) {
		io_service_t platformExpert = IOServiceGetMatchingService(kIOMasterPortDefault, IOServiceMatching("IOPlatformExpertDevice"));
		CFStringRef serialNumberAsCFString = NULL;
		if (platformExpert) {
			serialNumberAsCFString = (CFStringRef)IORegistryEntryCreateCFProperty(platformExpert, CFSTR(kIOPlatformSerialNumberKey), kCFAllocatorDefault, 0);
			IOObjectRelease(platformExpert);
		}

		NSString *serialNumberAsNSString = nil;
		if (serialNumberAsCFString) {
			serialNumberAsNSString = [NSString stringWithString:(NSString *)serialNumberAsCFString];
			CFRelease(serialNumberAsCFString);
		}

		serial_number = [serialNumberAsNSString UTF8String];
	}

	return serial_number;
}

void OS_OSX::set_ime_active(const bool p_active) {
	im_active = p_active;
	if (!im_active)
		[window_view cancelComposition];
}

void OS_OSX::set_ime_position(const Point2 &p_pos) {
	im_position = p_pos;
}

String OS_OSX::get_processor_name() const {
	char buffer[256];
	size_t buffer_len = 256;
	if (sysctlbyname("machdep.cpu.brand_string", &buffer, &buffer_len, NULL, 0) == 0) {
		return String::utf8(buffer, buffer_len);
	}
	ERR_FAIL_V_MSG("", String("Couldn't get the CPU model name. Returning an empty string."));
}

void OS_OSX::initialize_core() {
	crash_handler.initialize();

	OS_Unix::initialize_core();

	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessOSX>(DirAccess::ACCESS_FILESYSTEM);
}

struct LayoutInfo {
	String name;
	String code;
};

static Vector<LayoutInfo> kbd_layouts;
static int current_layout = 0;
static bool keyboard_layout_dirty = true;
static OS::LatinKeyboardVariant latin_variant = OS::LATIN_KEYBOARD_QWERTY;
static void keyboard_layout_changed(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef user_info) {
	kbd_layouts.clear();
	current_layout = 0;
	keyboard_layout_dirty = true;
}

static bool displays_arrangement_dirty = true;
static bool displays_scale_dirty = true;
static void displays_arrangement_changed(CGDirectDisplayID display_id, CGDisplayChangeSummaryFlags flags, void *user_info) {
	displays_arrangement_dirty = true;
	displays_scale_dirty = true;
}

int OS_OSX::get_current_video_driver() const {
	return video_driver_index;
}

bool OS_OSX::tts_is_speaking() const {
	ERR_FAIL_COND_V_MSG(!tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND_V(!tts, false);
	return [tts isSpeaking];
}

bool OS_OSX::tts_is_paused() const {
	ERR_FAIL_COND_V_MSG(!tts, false, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND_V(!tts, false);
	return [tts isPaused];
}

Array OS_OSX::tts_get_voices() const {
	ERR_FAIL_COND_V_MSG(!tts, Array(), "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND_V(!tts, Array());
	return [tts getVoices];
}

void OS_OSX::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND(!tts);
	[tts speak:p_text voice:p_voice volume:p_volume pitch:p_pitch rate:p_rate utterance_id:p_utterance_id interrupt:p_interrupt];
}

void OS_OSX::tts_pause() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND(!tts);
	[tts pauseSpeaking];
}

void OS_OSX::tts_resume() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND(!tts);
	[tts resumeSpeaking];
}

void OS_OSX::tts_stop() {
	ERR_FAIL_COND_MSG(!tts, "Enable the \"audio/general/text_to_speech\" project setting to use text-to-speech.");
	ERR_FAIL_COND(!tts);
	[tts stopSpeaking];
}

Error OS_OSX::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	/*** OSX INITIALIZATION ***/
	/*** OSX INITIALIZATION ***/
	/*** OSX INITIALIZATION ***/

	keyboard_layout_dirty = true;
	displays_arrangement_dirty = true;
	displays_scale_dirty = true;

	// Register to be notified on keyboard layout changes
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
			NULL, keyboard_layout_changed,
			kTISNotifySelectedKeyboardInputSourceChanged, NULL,
			CFNotificationSuspensionBehaviorDeliverImmediately);

	// Register to be notified on displays arrangement changes
	CGDisplayRegisterReconfigurationCallback(displays_arrangement_changed, NULL);

	// Init TTS
	bool tts_enabled = GLOBAL_GET("audio/general/text_to_speech");
	if (tts_enabled) {
		tts = [[TTS_OSX alloc] init];
	}

	window_delegate = [[GodotWindowDelegate alloc] init];

	// Don't use accumulation buffer support; it's not accelerated
	// Aux buffers probably aren't accelerated either

	unsigned int styleMask;

	if (p_desired.borderless_window) {
		styleMask = NSWindowStyleMaskBorderless;
	} else {
		resizable = p_desired.resizable;
		styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (p_desired.resizable ? NSWindowStyleMaskResizable : 0);
	}

	float displayScale = get_screen_max_scale();

	window_object = [[GodotWindow alloc]
			initWithContentRect:NSMakeRect(0, 0, p_desired.width / displayScale, p_desired.height / displayScale)
					  styleMask:styleMask
						backing:NSBackingStoreBuffered
						  defer:NO];

	ERR_FAIL_COND_V(window_object == nil, ERR_UNAVAILABLE);

	window_view = [[GodotContentView alloc] init];
	if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_14) {
		[window_view setWantsLayer:TRUE];
	}

	window_size.width = p_desired.width;
	window_size.height = p_desired.height;

	if (displayScale > 1.0) {
		[window_view setWantsBestResolutionOpenGLSurface:YES];
		//if (current_videomode.resizable)
		[window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
	} else {
		[window_view setWantsBestResolutionOpenGLSurface:NO];
	}

	[window_object setColorSpace:[NSColorSpace sRGBColorSpace]];

	//[window_object setTitle:[NSString stringWithUTF8String:"GodotEnginies"]];
	[window_object setContentView:window_view];
	[window_object setDelegate:window_delegate];
	if (!is_no_window_mode_enabled()) {
		[window_object setAcceptsMouseMovedEvents:YES];
		[((NSWindow *)window_object) center];
	}

	[window_object setRestorable:NO];

	unsigned int attributeCount = 0;

	// OS X needs non-zero color size, so set reasonable values
	int colorBits = 32;

	// Fail if a robustness strategy was requested

#define ADD_ATTR(x) \
	{ attributes[attributeCount++] = x; }
#define ADD_ATTR2(x, y) \
	{                   \
		ADD_ATTR(x);    \
		ADD_ATTR(y);    \
	}

	// Arbitrary array size here
	NSOpenGLPixelFormatAttribute attributes[40];

	ADD_ATTR(NSOpenGLPFADoubleBuffer);
	ADD_ATTR(NSOpenGLPFAClosestPolicy);

	if (p_video_driver == VIDEO_DRIVER_GLES2) {
		ADD_ATTR2(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersionLegacy);
	} else {
		//we now need OpenGL 3 or better, maybe even change this to 3_3Core ?
		ADD_ATTR2(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core);
	}

	ADD_ATTR2(NSOpenGLPFAColorSize, colorBits);

	/*
	if (fbconfig->alphaBits > 0)
		ADD_ATTR2(NSOpenGLPFAAlphaSize, fbconfig->alphaBits);
*/

	ADD_ATTR2(NSOpenGLPFADepthSize, 24);

	ADD_ATTR2(NSOpenGLPFAStencilSize, 8);

	/*
	if (fbconfig->stereo)
		ADD_ATTR(NSOpenGLPFAStereo);
*/

	/*
	if (fbconfig->samples > 0) {
		ADD_ATTR2(NSOpenGLPFASampleBuffers, 1);
		ADD_ATTR2(NSOpenGLPFASamples, fbconfig->samples);
	}
*/

	// NOTE: All NSOpenGLPixelFormats on the relevant cards support sRGB
	//       framebuffer, so there's no need (and no way) to request it

	ADD_ATTR(0);

#undef ADD_ATTR
#undef ADD_ATTR2

	pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
	ERR_FAIL_COND_V(pixelFormat == nil, ERR_UNAVAILABLE);

	context = [[NSOpenGLContext alloc] initWithFormat:pixelFormat shareContext:nil];

	ERR_FAIL_COND_V(context == nil, ERR_UNAVAILABLE);

	[window_view setOpenGLContext:context];

	context_offscreen = [[NSOpenGLContext alloc] initWithFormat:pixelFormat shareContext:nil];

	[context makeCurrentContext];

	GLint dim[2];
	dim[0] = window_size.width;
	dim[1] = window_size.height;
	CGLSetParameter((CGLContextObj)[context CGLContextObj], kCGLCPSurfaceBackingSize, &dim[0]);
	CGLEnable((CGLContextObj)[context CGLContextObj], kCGLCESurfaceBackingSize);

	set_use_vsync(p_desired.use_vsync);

	if (!is_no_window_mode_enabled()) {
		[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
		[NSApp activateIgnoringOtherApps:YES];
	}

	_update_window();

	if (!is_no_window_mode_enabled()) {
		[window_object makeKeyAndOrderFront:nil];
	}

	on_top = p_desired.always_on_top;
	if (p_desired.always_on_top && !p_desired.fullscreen) {
		[window_object setLevel:NSFloatingWindowLevel];
	}

	if (p_desired.fullscreen)
		zoomed = true;

	/*** END OSX INITIALIZATION ***/

	bool gles3 = true;
	if (p_video_driver == VIDEO_DRIVER_GLES2) {
		gles3 = false;
	}

	bool editor = Engine::get_singleton()->is_editor_hint();
	bool gl_initialization_error = false;

	while (true) {
		if (gles3) {
			if (RasterizerGLES3::is_viable() == OK) {
				RasterizerGLES3::register_config();
				RasterizerGLES3::make_current();
				break;
			} else {
				if (GLOBAL_GET("rendering/quality/driver/fallback_to_gles2") || editor) {
					p_video_driver = VIDEO_DRIVER_GLES2;
					gles3 = false;
					continue;
				} else {
					gl_initialization_error = true;
					break;
				}
			}
		} else {
			if (RasterizerGLES2::is_viable() == OK) {
				RasterizerGLES2::register_config();
				RasterizerGLES2::make_current();
				break;
			} else {
				gl_initialization_error = true;
				break;
			}
		}
	}

	if (gl_initialization_error) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported OpenGL versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU, upgrade it.",
				"Unable to initialize Video driver");
		return ERR_UNAVAILABLE;
	}

	video_driver_index = p_video_driver;

	visual_server = memnew(VisualServerRaster);
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	visual_server->init();
	AudioDriverManager::initialize(p_audio_driver);

	input = memnew(InputDefault);
#ifdef SDL_ENABLED
	joypad_sdl = memnew(JoypadSDL(input));
	if (joypad_sdl->initialize() != OK) {
		ERR_PRINT("Couldn't initialize SDL joypad input driver.");
		memdelete(joypad_sdl);
		joypad_sdl = nullptr;
	}
#endif

	power_manager = memnew(PowerOSX);

	restore_rect = Rect2(get_window_position(), get_window_size());

	if (p_desired.layered) {
		set_window_per_pixel_transparency_enabled(true);
	}

	update_real_mouse_position();

	if (is_no_window_mode_enabled()) {
		[NSApp hide:nil];
	}

	return OK;
}

void OS_OSX::finalize() {
#ifdef COREMIDI_ENABLED
	midi_driver.close();
#endif

	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), NULL, kTISNotifySelectedKeyboardInputSourceChanged, NULL);
	CGDisplayRemoveReconfigurationCallback(displays_arrangement_changed, NULL);

	delete_main_loop();

#ifdef SDL_ENABLED
	if (joypad_sdl) {
		memdelete(joypad_sdl);
	}
#endif
	memdelete(input);

	cursors_cache.clear();
	visual_server->finish();
	memdelete(visual_server);
	//memdelete(rasterizer);
}

void OS_OSX::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
}

void OS_OSX::delete_main_loop() {
	if (!main_loop)
		return;
	memdelete(main_loop);
	main_loop = NULL;
}

String OS_OSX::get_name() const {
	return "OSX";
}

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101200
class OSXTerminalLogger : public StdLogger {
public:
	virtual void log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type = ERR_ERROR) {
		if (!should_log(true)) {
			return;
		}

		const char *err_details;
		if (p_rationale && p_rationale[0])
			err_details = p_rationale;
		else
			err_details = p_code;

		switch (p_type) {
			case ERR_WARNING:
				if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_12) {
					os_log_info(OS_LOG_DEFAULT,
							"WARNING: %{public}s\nat: %{public}s (%{public}s:%i)",
							err_details, p_function, p_file, p_line);
				}
				logf_error("\E[1;33mWARNING:\E[0;93m %s\n", err_details);
				logf_error("\E[0;90m     at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_SCRIPT:
				if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_12) {
					os_log_error(OS_LOG_DEFAULT,
							"SCRIPT ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
							err_details, p_function, p_file, p_line);
				}
				logf_error("\E[1;35mSCRIPT ERROR:\E[0;95m %s\n", err_details);
				logf_error("\E[0;90m          at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_SHADER:
				if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_12) {
					os_log_error(OS_LOG_DEFAULT,
							"SHADER ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
							err_details, p_function, p_file, p_line);
				}
				logf_error("\E[1;36mSHADER ERROR:\E[0;96m %s\n", err_details);
				logf_error("\E[0;90m          at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
			case ERR_ERROR:
			default:
				if (NSAppKitVersionNumber >= NSAppKitVersionNumber10_12) {
					os_log_error(OS_LOG_DEFAULT,
							"ERROR: %{public}s\nat: %{public}s (%{public}s:%i)",
							err_details, p_function, p_file, p_line);
				}
				logf_error("\E[1;31mERROR:\E[0;91m %s\n", err_details);
				logf_error("\E[0;90m   at: %s (%s:%i)\E[0m\n", p_function, p_file, p_line);
				break;
		}
	}
};

#else

typedef UnixTerminalLogger OSXTerminalLogger;
#endif

void OS_OSX::alert(const String &p_alert, const String &p_title) {
	if (is_no_window_mode_enabled()) {
		print_line("ALERT: " + p_title + ": " + p_alert);
		return;
	}

	// Set OS X-compliant variables
	NSAlert *window = [[NSAlert alloc] init];
	NSString *ns_title = [NSString stringWithUTF8String:p_title.utf8().get_data()];
	NSString *ns_alert = [NSString stringWithUTF8String:p_alert.utf8().get_data()];

	[window addButtonWithTitle:@"OK"];
	[window setMessageText:ns_title];
	[window setInformativeText:ns_alert];
	[window setAlertStyle:NSAlertStyleWarning];

	// Display it, then release
	id key_window = [[NSApplication sharedApplication] keyWindow];
	[window runModal];
	[window release];
	if (key_window) {
		[key_window makeKeyAndOrderFront:nil];
	}
}

Error OS_OSX::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = p_path;

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .dylib files from within the executable path
		path = get_executable_path().get_base_dir().plus_file(p_path.get_file());
	}

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .dylib files from a standard macOS location
		path = get_executable_path().get_base_dir().plus_file("../Frameworks").plus_file(p_path.get_file());
	}

	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ", error: " + dlerror() + ".");
	return OK;
}

void OS_OSX::set_cursor_shape(CursorShape p_shape) {
	if (cursor_shape == p_shape)
		return;

	if (mouse_mode != MOUSE_MODE_VISIBLE && mouse_mode != MOUSE_MODE_CONFINED) {
		cursor_shape = p_shape;
		return;
	}

	if (cursors[p_shape] != NULL) {
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
				[cursorFromSelector(@selector(_windowResizeNorthSouthCursor), @selector(resizeUpDownCursor)) set];
				break;
			case CURSOR_HSIZE:
				[cursorFromSelector(@selector(_windowResizeEastWestCursor), @selector(resizeLeftRightCursor)) set];
				break;
			case CURSOR_BDIAGSIZE:
				[cursorFromSelector(@selector(_windowResizeNorthEastSouthWestCursor)) set];
				break;
			case CURSOR_FDIAGSIZE:
				[cursorFromSelector(@selector(_windowResizeNorthWestSouthEastCursor)) set];
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
				[cursorFromSelector(@selector(_helpCursor)) set];
				break;
			default: {
			};
		}
	}

	cursor_shape = p_shape;
}

OS::CursorShape OS_OSX::get_cursor_shape() const {
	return cursor_shape;
}

void OS_OSX::set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	if (p_cursor.is_valid()) {
		Map<CursorShape, Vector<Variant>>::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				set_cursor_shape(p_shape);
				return;
			}

			cursors_cache.erase(p_shape);
		}

		Ref<Texture> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_data();
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

		image = texture->get_data();

		ERR_FAIL_COND(!image.is_valid());

		NSBitmapImageRep *imgrep = [[NSBitmapImageRep alloc]
				initWithBitmapDataPlanes:NULL
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
		PoolVector<uint8_t> data = image->get_data();
		PoolVector<uint8_t>::Read r = data.read();

		image->lock();

		/* Premultiply the alpha channel */
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

		image->unlock();

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
		if (cursors[p_shape] != NULL) {
			[cursors[p_shape] release];
			cursors[p_shape] = NULL;
		}

		CursorShape c = cursor_shape;
		cursor_shape = CURSOR_MAX;
		set_cursor_shape(c);

		cursors_cache.erase(p_shape);
	}
}

void OS_OSX::set_mouse_show(bool p_show) {
}

void OS_OSX::set_mouse_grab(bool p_grab) {
}

bool OS_OSX::is_mouse_grab_enabled() const {
	return mouse_grab;
}

void OS_OSX::warp_mouse_position(const Point2 &p_to) {
	//copied from windows impl with osx native calls
	if (mouse_mode == MOUSE_MODE_CAPTURED) {
		mouse_x = p_to.x;
		mouse_y = p_to.y;
	} else { //set OS position

		//local point in window coords
		const NSRect contentRect = [window_view frame];
		float displayScale = get_screen_max_scale();
		NSRect pointInWindowRect = NSMakeRect(p_to.x / displayScale, contentRect.size.height - (p_to.y / displayScale) - 1, 0, 0);
		NSPoint pointOnScreen = [[window_view window] convertRectToScreen:pointInWindowRect].origin;

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

void OS_OSX::update_real_mouse_position() {
	_ALLOW_DISCARD_ get_mouse_pos([window_object mouseLocationOutsideOfEventStream]);
	input->set_mouse_position(Point2(mouse_x, mouse_y));
}

Point2 OS_OSX::get_mouse_position() const {
	return Vector2(mouse_x, mouse_y);
}

int OS_OSX::get_mouse_button_state() const {
	return button_mask;
}

void OS_OSX::set_window_title(const String &p_title) {
	title = p_title;

	[window_object setTitle:[NSString stringWithUTF8String:p_title.utf8().get_data()]];
}

void OS_OSX::set_window_mouse_passthrough(const PoolVector2Array &p_region) {
	mpath.clear();
	for (int i = 0; i < p_region.size(); i++) {
		mpath.push_back(p_region[i]);
	}
}

void OS_OSX::set_native_icon(const String &p_filename) {
	FileAccess *f = FileAccess::open(p_filename, FileAccess::READ);
	ERR_FAIL_COND(!f);

	Vector<uint8_t> data;
	uint64_t len = f->get_len();
	data.resize(len);
	f->get_buffer((uint8_t *)&data.write[0], len);
	memdelete(f);

	NSData *icon_data = [[[NSData alloc] initWithBytes:&data.write[0] length:len] autorelease];
	ERR_FAIL_COND_MSG(!icon_data, "Error reading icon data.");

	NSImage *icon = [[[NSImage alloc] initWithData:icon_data] autorelease];
	ERR_FAIL_COND_MSG(!icon, "Error loading icon.");

	[NSApp setApplicationIconImage:icon];
}

void OS_OSX::set_icon(const Ref<Image> &p_icon) {
	Ref<Image> img = p_icon;
	img = img->duplicate();
	img->convert(Image::FORMAT_RGBA8);
	NSBitmapImageRep *imgrep = [[[NSBitmapImageRep alloc]
			initWithBitmapDataPlanes:NULL
						  pixelsWide:img->get_width()
						  pixelsHigh:img->get_height()
					   bitsPerSample:8
					 samplesPerPixel:4
							hasAlpha:YES
							isPlanar:NO
					  colorSpaceName:NSDeviceRGBColorSpace
						 bytesPerRow:img->get_width() * 4
						bitsPerPixel:32] autorelease];
	ERR_FAIL_COND(imgrep == nil);
	uint8_t *pixels = [imgrep bitmapData];

	int len = img->get_width() * img->get_height();
	PoolVector<uint8_t> data = img->get_data();
	PoolVector<uint8_t>::Read r = data.read();

	/* Premultiply the alpha channel */
	for (int i = 0; i < len; i++) {
		uint8_t alpha = r[i * 4 + 3];
		pixels[i * 4 + 0] = (uint8_t)(((uint16_t)r[i * 4 + 0] * alpha) / 255);
		pixels[i * 4 + 1] = (uint8_t)(((uint16_t)r[i * 4 + 1] * alpha) / 255);
		pixels[i * 4 + 2] = (uint8_t)(((uint16_t)r[i * 4 + 2] * alpha) / 255);
		pixels[i * 4 + 3] = alpha;
	}

	NSImage *nsimg = [[[NSImage alloc] initWithSize:NSMakeSize(img->get_width(), img->get_height())] autorelease];
	ERR_FAIL_COND(nsimg == nil);
	[nsimg addRepresentation:imgrep];

	[NSApp setApplicationIconImage:nsimg];
}

MainLoop *OS_OSX::get_main_loop() const {
	return main_loop;
}

String OS_OSX::get_config_path() const {
	if (has_environment("HOME")) {
		return get_environment("HOME").plus_file("Library/Application Support");
	}
	return ".";
}

String OS_OSX::get_data_path() const {
	return get_config_path();
}

String OS_OSX::get_cache_path() const {
	if (has_environment("HOME")) {
		return get_environment("HOME").plus_file("Library/Caches");
	}
	return get_config_path();
}

String OS_OSX::get_bundle_resource_dir() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *resourcePath = [main resourcePath];
		ret.parse_utf8([resourcePath UTF8String]);
	}
	return ret;
}

String OS_OSX::get_bundle_icon_path() const {
	String ret;

	NSBundle *main = [NSBundle mainBundle];
	if (main) {
		NSString *iconPath = [[main infoDictionary] objectForKey:@"CFBundleIconFile"];
		if (iconPath) {
			ret.parse_utf8([iconPath UTF8String]);
		}
	}
	return ret;
}

// Get properly capitalized engine name for system paths
String OS_OSX::get_godot_dir_name() const {
	return String(VERSION_SHORT_NAME).capitalize();
}

String OS_OSX::get_system_dir(SystemDir p_dir, bool p_shared_storage) const {
	NSSearchPathDirectory id;
	bool found = true;

	switch (p_dir) {
		case SYSTEM_DIR_DESKTOP: {
			id = NSDesktopDirectory;
		} break;
		case SYSTEM_DIR_DOCUMENTS: {
			id = NSDocumentDirectory;
		} break;
		case SYSTEM_DIR_DOWNLOADS: {
			id = NSDownloadsDirectory;
		} break;
		case SYSTEM_DIR_MOVIES: {
			id = NSMoviesDirectory;
		} break;
		case SYSTEM_DIR_MUSIC: {
			id = NSMusicDirectory;
		} break;
		case SYSTEM_DIR_PICTURES: {
			id = NSPicturesDirectory;
		} break;
		default: {
			found = false;
		}
	}

	String ret;
	if (found) {
		NSArray *paths = NSSearchPathForDirectoriesInDomains(id, NSUserDomainMask, YES);
		if (paths && [paths count] >= 1) {
			char *utfs = strdup([[paths firstObject] UTF8String]);
			ret.parse_utf8(utfs);
			free(utfs);
		}
	}

	return ret;
}

bool OS_OSX::can_draw() const {
	return is_visible;
}

void OS_OSX::set_clipboard(const String &p_text) {
	NSString *copiedString = [NSString stringWithUTF8String:p_text.utf8().get_data()];
	NSArray *copiedStringArray = [NSArray arrayWithObject:copiedString];

	NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
	[pasteboard clearContents];
	[pasteboard writeObjects:copiedStringArray];
}

String OS_OSX::get_clipboard() const {
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

void OS_OSX::release_rendering_thread() {
	[NSOpenGLContext clearCurrentContext];
}

void OS_OSX::make_rendering_thread() {
	[context makeCurrentContext];
}

Error OS_OSX::shell_open(String p_uri) {
	NSString *string = [NSString stringWithUTF8String:p_uri.utf8().get_data()];
	NSURL *uri = [[NSURL alloc] initWithString:string];
	// Escape special characters in filenames
	if (!uri || !uri.scheme || [uri.scheme isEqual:@"file"]) {
		uri = [[NSURL alloc] initWithString:[string stringByAddingPercentEncodingWithAllowedCharacters:[NSCharacterSet URLFragmentAllowedCharacterSet]]];
	}
	[[NSWorkspace sharedWorkspace] openURL:uri];
	return OK;
}

String OS_OSX::get_locale() const {
	NSString *locale_code = [[NSLocale preferredLanguages] objectAtIndex:0];
	return String([locale_code UTF8String]).replace("-", "_");
}

void OS_OSX::swap_buffers() {
	[context flushBuffer];
}

void OS_OSX::wm_minimized(bool p_minimized) {
	minimized = p_minimized;
};

void OS_OSX::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
}

OS::VideoMode OS_OSX::get_video_mode(int p_screen) const {
	VideoMode vm;
	vm.width = window_size.width;
	vm.height = window_size.height;

	return vm;
}

void OS_OSX::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
}

bool OS_OSX::is_offscreen_gl_available() const {
	return context_offscreen != nil;
}

void OS_OSX::set_offscreen_gl_current(bool p_current) {
	if (p_current) {
		[context_offscreen makeCurrentContext];
	} else {
		[NSOpenGLContext clearCurrentContext];
	}
}

int OS_OSX::get_screen_count() const {
	NSArray *screenArray = [NSScreen screens];
	return [screenArray count];
};

// Returns the native top-left screen coordinate of the smallest rectangle
// that encompasses all screens. Needed in get_screen_position(),
// get_window_position, and set_window_position()
// to convert between OS X native screen coordinates and the ones expected by Godot
Point2 OS_OSX::get_screens_origin() const {
	static Point2 origin;

	if (displays_arrangement_dirty) {
		origin = Point2();

		for (int i = 0; i < get_screen_count(); i++) {
			Point2 position = get_native_screen_position(i);
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

static int get_screen_index(NSScreen *screen) {
	const NSUInteger index = [[NSScreen screens] indexOfObject:screen];
	return index == NSNotFound ? 0 : index;
}

int OS_OSX::get_current_screen() const {
	if (window_object) {
		return get_screen_index([window_object screen]);
	} else {
		return get_screen_index([NSScreen mainScreen]);
	}
};

void OS_OSX::set_current_screen(int p_screen) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if (get_current_screen() == p_screen) {
		return;
	}

	bool was_fullscreen = false;
	if (zoomed) {
		// Temporary exit fullscreen mode to move window.
		[window_object toggleFullScreen:nil];
		was_fullscreen = true;
	}

	Vector2 wpos = get_window_position() - get_screen_position(get_current_screen());
	set_window_position(wpos + get_screen_position(p_screen));

	if (was_fullscreen) {
		// Re-enter fullscreen mode.
		[window_object toggleFullScreen:nil];
	}
};

Point2 OS_OSX::get_native_screen_position(int p_screen) const {
	if (p_screen < 0) {
		p_screen = get_current_screen();
	}

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		float display_scale = get_screen_max_scale();
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		// Return the top-left corner of the screen, for OS X the y starts at the bottom
		return Point2(nsrect.origin.x, nsrect.origin.y + nsrect.size.height) * display_scale;
	}

	return Point2();
}

Point2 OS_OSX::get_screen_position(int p_screen) const {
	Point2 position = get_native_screen_position(p_screen) - get_screens_origin();
	// OS X native y-coordinate relative to get_screens_origin() is negative,
	// Godot expects a positive value
	position.y *= -1;
	return position;
}

int OS_OSX::get_screen_dpi(int p_screen) const {
	if (p_screen < 0) {
		p_screen = get_current_screen();
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

float OS_OSX::get_screen_refresh_rate(int p_screen) const {
	if (p_screen < 0) {
		p_screen = get_current_screen();
	}

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		NSDictionary *description = [[screenArray objectAtIndex:p_screen] deviceDescription];
		const CGDisplayModeRef displayMode = CGDisplayCopyDisplayMode([[description objectForKey:@"NSScreenNumber"] unsignedIntValue]);
		const double displayRefreshRate = CGDisplayModeGetRefreshRate(displayMode);
		return (float)displayRefreshRate;
	}
	ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
	return OS::get_singleton()->SCREEN_REFRESH_RATE_FALLBACK;
}

Size2 OS_OSX::get_screen_size(int p_screen) const {
	if (p_screen < 0) {
		p_screen = get_current_screen();
	}

	NSArray *screenArray = [NSScreen screens];
	if ((NSUInteger)p_screen < [screenArray count]) {
		float displayScale = get_screen_max_scale();
		// Note: Use frame to get the whole screen size
		NSRect nsrect = [[screenArray objectAtIndex:p_screen] frame];
		return Size2(nsrect.size.width, nsrect.size.height) * displayScale;
	}

	return Size2();
}

void OS_OSX::_update_window() {
	bool borderless_full = false;

	if (get_borderless_window()) {
		NSRect frameRect = [window_object frame];
		NSRect screenRect = [[window_object screen] frame];

		// Check if our window covers up the screen
		if (frameRect.origin.x <= screenRect.origin.x && frameRect.origin.y <= frameRect.origin.y &&
				frameRect.size.width >= screenRect.size.width && frameRect.size.height >= screenRect.size.height) {
			borderless_full = true;
		}
	}

	if (borderless_full) {
		// If the window covers up the screen set the level to above the main menu and hide on deactivate
		[window_object setLevel:NSMainMenuWindowLevel + 1];
		[window_object setHidesOnDeactivate:YES];
	} else {
		// Reset these when our window is not a borderless window that covers up the screen
		if (on_top & !zoomed) {
			[window_object setLevel:NSFloatingWindowLevel];
		} else {
			[window_object setLevel:NSNormalWindowLevel];
		}
		[window_object setHidesOnDeactivate:NO];
	}
}

float OS_OSX::get_screen_scale(int p_screen) const {
	if (p_screen < 0) {
		p_screen = get_current_screen();
	}

	if (is_hidpi_allowed()) {
		NSArray *screenArray = [NSScreen screens];
		if ((NSUInteger)p_screen < [screenArray count]) {
			if ([[screenArray objectAtIndex:p_screen] respondsToSelector:@selector(backingScaleFactor)]) {
				return fmax(1.0, [[screenArray objectAtIndex:p_screen] backingScaleFactor]);
			}
		}
	}

	return 1.f;
}

float OS_OSX::get_screen_max_scale() const {
	static float scale = 1.f;
	if (displays_scale_dirty) {
		int screen_count = get_screen_count();
		for (int i = 0; i < screen_count; i++) {
			scale = fmax(scale, get_screen_scale(i));
		}
		displays_scale_dirty = false;
	}
	return scale;
}

Point2 OS_OSX::get_native_window_position() const {
	NSRect nsrect = [window_object frame];
	Point2 pos;
	float display_scale = get_screen_max_scale();

	// Return the position of the top-left corner, for OS X the y starts at the bottom
	pos.x = nsrect.origin.x * display_scale;
	pos.y = (nsrect.origin.y + nsrect.size.height) * display_scale;

	return pos;
};

Point2 OS_OSX::get_window_position() const {
	Point2 position = get_native_window_position() - get_screens_origin();
	// OS X native y-coordinate relative to get_screens_origin() is negative,
	// Godot expects a positive value
	position.y *= -1;
	return position;
}

void OS_OSX::set_native_window_position(const Point2 &p_position) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	NSPoint pos;
	float displayScale = get_screen_max_scale();

	pos.x = p_position.x / displayScale;
	pos.y = p_position.y / displayScale;

	[window_object setFrameTopLeftPoint:pos];

	_update_window();
};

void OS_OSX::set_window_position(const Point2 &p_position) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	Point2 position = p_position;
	// OS X native y-coordinate relative to get_screens_origin() is negative,
	// Godot passes a positive value
	position.y *= -1;
	set_native_window_position(get_screens_origin() + position);

	update_real_mouse_position();
};

Size2 OS_OSX::get_window_size() const {
	return window_size;
};

Size2 OS_OSX::get_real_window_size() const {
	NSRect frame = [window_object frame];
	return Size2(frame.size.width, frame.size.height) * get_screen_max_scale();
}

Size2 OS_OSX::get_max_window_size() const {
	return max_size;
}

Size2 OS_OSX::get_min_window_size() const {
	return min_size;
}

void OS_OSX::set_min_window_size(const Size2 p_size) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if ((p_size != Size2()) && (max_size != Size2()) && ((p_size.x > max_size.x) || (p_size.y > max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	min_size = p_size;

	if ((min_size != Size2()) && !zoomed) {
		Size2 size = min_size / get_screen_max_scale();
		[window_object setContentMinSize:NSMakeSize(size.x, size.y)];
	} else {
		[window_object setContentMinSize:NSMakeSize(0, 0)];
	}
}

void OS_OSX::set_max_window_size(const Size2 p_size) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if ((p_size != Size2()) && ((p_size.x < min_size.x) || (p_size.y < min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	max_size = p_size;

	if ((max_size != Size2()) && !zoomed) {
		Size2 size = max_size / get_screen_max_scale();
		[window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
	} else {
		[window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
	}
}

void OS_OSX::set_window_size(const Size2 p_size) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	Size2 size = p_size / get_screen_max_scale();

	NSPoint top_left;
	NSRect old_frame = [window_object frame];
	top_left.x = old_frame.origin.x;
	top_left.y = NSMaxY(old_frame);

	NSRect new_frame = NSMakeRect(0, 0, size.x, size.y);
	new_frame = [window_object frameRectForContentRect:new_frame];

	new_frame.origin.x = top_left.x;
	new_frame.origin.y = top_left.y - new_frame.size.height;

	[window_object setFrame:new_frame display:YES];

	_update_window();
};

void OS_OSX::set_window_fullscreen(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if (zoomed != p_enabled) {
		[window_object setLevel:NSNormalWindowLevel];
		if (layered_window)
			set_window_per_pixel_transparency_enabled(false);
		if (!resizable)
			[window_object setStyleMask:[window_object styleMask] | NSWindowStyleMaskResizable];
		if (p_enabled) {
			[window_object setContentMinSize:NSMakeSize(0, 0)];
			[window_object setContentMaxSize:NSMakeSize(FLT_MAX, FLT_MAX)];
		} else {
			if (min_size != Size2()) {
				Size2 size = min_size / get_screen_max_scale();
				[window_object setContentMinSize:NSMakeSize(size.x, size.y)];
			}
			if (max_size != Size2()) {
				Size2 size = max_size / get_screen_max_scale();
				[window_object setContentMaxSize:NSMakeSize(size.x, size.y)];
			}
		}
		if (p_enabled) {
			const NSUInteger presentationOptions = NSApplicationPresentationHideDock | NSApplicationPresentationHideMenuBar;
			[NSApp setPresentationOptions:presentationOptions];
		} else {
			[NSApp setPresentationOptions:NSApplicationPresentationDefault];
		}
		[window_object toggleFullScreen:nil];
	}
	zoomed = p_enabled;
};

bool OS_OSX::is_window_fullscreen() const {
	return zoomed;
};

void OS_OSX::set_window_resizable(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if (p_enabled)
		[window_object setStyleMask:[window_object styleMask] | NSWindowStyleMaskResizable];
	else if (!zoomed)
		[window_object setStyleMask:[window_object styleMask] & ~NSWindowStyleMaskResizable];

	resizable = p_enabled;
};

bool OS_OSX::is_window_resizable() const {
	return [window_object styleMask] & NSWindowStyleMaskResizable;
};

void OS_OSX::set_window_minimized(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if (p_enabled)
		[window_object performMiniaturize:nil];
	else
		[window_object deminiaturize:nil];
};

bool OS_OSX::is_window_minimized() const {
	if ([window_object respondsToSelector:@selector(isMiniaturized)])
		return [window_object isMiniaturized];

	return minimized;
};

void OS_OSX::set_window_maximized(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	if (p_enabled) {
		restore_rect = Rect2(get_window_position(), get_window_size());
		[window_object setFrame:[[[NSScreen screens] objectAtIndex:get_current_screen()] visibleFrame] display:YES];
	} else {
		set_window_size(restore_rect.size);
		set_window_position(restore_rect.position);
	};
	maximized = p_enabled;
};

bool OS_OSX::is_window_maximized() const {
	// don't know
	return maximized;
};

void OS_OSX::move_window_to_foreground() {
	if (is_no_window_mode_enabled()) {
		return;
	}

	[[NSApplication sharedApplication] activateIgnoringOtherApps:YES];
	[window_object makeKeyAndOrderFront:nil];
}

void OS_OSX::set_window_always_on_top(bool p_enabled) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	on_top = p_enabled;

	if (zoomed)
		return;

	if (is_window_always_on_top() == p_enabled)
		return;

	if (p_enabled)
		[window_object setLevel:NSFloatingWindowLevel];
	else
		[window_object setLevel:NSNormalWindowLevel];
}

bool OS_OSX::is_window_always_on_top() const {
	if (zoomed) {
		return on_top;
	} else {
		return [window_object level] == NSFloatingWindowLevel;
	}
}

bool OS_OSX::is_window_focused() const {
	return window_focused;
}

void OS_OSX::request_attention() {
	if (is_no_window_mode_enabled()) {
		return;
	}

	[NSApp requestUserAttention:NSCriticalRequest];
}

bool OS_OSX::get_window_per_pixel_transparency_enabled() const {
	if (!is_layered_allowed())
		return false;
	return layered_window;
}

void OS_OSX::set_window_per_pixel_transparency_enabled(bool p_enabled) {
	if (!is_layered_allowed())
		return;
	if (layered_window != p_enabled) {
		if (p_enabled) {
			GLint opacity = 0;
			[window_object setBackgroundColor:[NSColor clearColor]];
			[window_object setOpaque:NO];
			[window_object setHasShadow:NO];
			[context setValues:&opacity forParameter:NSOpenGLContextParameterSurfaceOpacity];
			layered_window = true;
		} else {
			GLint opacity = 1;
			[window_object setBackgroundColor:[NSColor colorWithCalibratedWhite:1 alpha:1]];
			[window_object setOpaque:YES];
			[window_object setHasShadow:YES];
			[context setValues:&opacity forParameter:NSOpenGLContextParameterSurfaceOpacity];
			layered_window = false;
		}
		[context update];

		if (!is_no_window_mode_enabled()) {
			// Force resize window frame to update OpenGL context.
			NSRect frameRect = [window_object frame];
			[window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
			[window_object setFrame:frameRect display:YES];
		}
	}
}

void OS_OSX::set_borderless_window(bool p_borderless) {
	if (is_no_window_mode_enabled()) {
		return;
	}

	// OrderOut prevents a lose focus bug with the window
	[window_object orderOut:nil];

	if (p_borderless) {
		[window_object setStyleMask:NSWindowStyleMaskBorderless];
	} else {
		[window_object setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | (resizable ? NSWindowStyleMaskResizable : 0)];

		// Force update of the window styles
		NSRect frameRect = [window_object frame];
		[window_object setFrame:NSMakeRect(frameRect.origin.x, frameRect.origin.y, frameRect.size.width + 1, frameRect.size.height) display:NO];
		[window_object setFrame:frameRect display:NO];

		// Restore the window title
		[window_object setTitle:[NSString stringWithUTF8String:title.utf8().get_data()]];
	}

	_update_window();

	[window_object makeKeyAndOrderFront:nil];
}

bool OS_OSX::get_borderless_window() {
	return [window_object styleMask] == NSWindowStyleMaskBorderless;
}

String OS_OSX::get_executable_path() const {
	int ret;
	pid_t pid;
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];

	pid = getpid();
	ret = proc_pidpath(pid, pathbuf, sizeof(pathbuf));
	if (ret <= 0) {
		return OS::get_executable_path();
	} else {
		String path;
		path.parse_utf8(pathbuf);

		return path;
	}
}

Error OS_OSX::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
	if (@available(macOS 10.15, *)) {
		NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
		// If executable is bundled, always execute editor instances using NSWorkspace to ensure app window is registered and activated correctly.
		if (p_path == get_executable_path() && nsappname != nil) {
			if (p_blocking && r_pipe) {
				ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Cannot pipe stream from process running as app bundle.");
			}
			NSMutableArray *arguments = [[NSMutableArray alloc] init];
			for (const List<String>::Element *E = p_arguments.front(); E; E = E->next()) {
				[arguments addObject:[NSString stringWithUTF8String:E->get().utf8().get_data()]];
			}
			NSWorkspaceOpenConfiguration *configuration = [[NSWorkspaceOpenConfiguration alloc] init];
			[configuration setArguments:arguments];
			[configuration setCreatesNewApplicationInstance:YES];
			__block dispatch_semaphore_t lock = dispatch_semaphore_create(0);
			__block Error err = ERR_TIMEOUT;
			__block pid_t pid = 0;
			[[NSWorkspace sharedWorkspace] openApplicationAtURL:[[NSBundle mainBundle] bundleURL]
												  configuration:configuration
											  completionHandler:^(NSRunningApplication *app, NSError *error) {
												  if (error) {
													  err = ERR_CANT_FORK;
													  NSLog(@"Failed to execute: %@", error.localizedDescription);
												  } else {
													  pid = [app processIdentifier];
													  err = OK;
												  }
												  dispatch_semaphore_signal(lock);
											  }];
			dispatch_semaphore_wait(lock, dispatch_time(DISPATCH_TIME_NOW, 20000000000)); // 20 sec timeout, wait for app to launch.
			dispatch_release(lock);

			if (err == OK) {
				if (p_blocking) {
					int status;
					waitpid(pid, &status, 0);
					if (r_exitcode) {
						*r_exitcode = WIFEXITED(status) ? WEXITSTATUS(status) : status;
					}
				} else if (r_child_id) {
					*r_child_id = (ProcessID)pid;
				}
			}

			return err;
		} else {
			return OS_Unix::execute(p_path, p_arguments, p_blocking, r_child_id, r_pipe, r_exitcode, read_stderr, p_pipe_mutex, p_open_console);
		}
	} else {
		return OS_Unix::execute(p_path, p_arguments, p_blocking, r_child_id, r_pipe, r_exitcode, read_stderr, p_pipe_mutex, p_open_console);
	}
}

// Returns string representation of keys, if they are printable.
//
static NSString *createStringForKeys(const CGKeyCode *keyCode, int length) {
	TISInputSourceRef currentKeyboard = TISCopyCurrentKeyboardInputSource();
	if (!currentKeyboard)
		return nil;

	CFDataRef layoutData = (CFDataRef)TISGetInputSourceProperty(currentKeyboard, kTISPropertyUnicodeKeyLayoutData);
	if (!layoutData)
		return nil;

	const UCKeyboardLayout *keyboardLayout = (const UCKeyboardLayout *)CFDataGetBytePtr(layoutData);

	OSStatus err;
	CFMutableStringRef output = CFStringCreateMutable(NULL, 0);

	for (int i = 0; i < length; ++i) {
		UInt32 keysDown = 0;
		UniChar chars[4];
		UniCharCount realLength;

		err = UCKeyTranslate(keyboardLayout,
				keyCode[i],
				kUCKeyActionDisplay,
				0,
				LMGetKbdType(),
				kUCKeyTranslateNoDeadKeysBit,
				&keysDown,
				sizeof(chars) / sizeof(chars[0]),
				&realLength,
				chars);

		if (err != noErr) {
			CFRelease(output);
			return nil;
		}

		CFStringAppendCharacters(output, chars, 1);
	}

	//CFStringUppercase(output, NULL);

	return (NSString *)output;
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

	// Update latin variant
	latin_variant = OS::LATIN_KEYBOARD_QWERTY;

	CGKeyCode keys[] = { kVK_ANSI_Q, kVK_ANSI_W, kVK_ANSI_E, kVK_ANSI_R, kVK_ANSI_T, kVK_ANSI_Y };
	NSString *test = createStringForKeys(keys, 6);

	if ([test isEqualToString:@"qwertz"]) {
		latin_variant = OS::LATIN_KEYBOARD_QWERTZ;
	} else if ([test isEqualToString:@"azerty"]) {
		latin_variant = OS::LATIN_KEYBOARD_AZERTY;
	} else if ([test isEqualToString:@"qzerty"]) {
		latin_variant = OS::LATIN_KEYBOARD_QZERTY;
	} else if ([test isEqualToString:@"',.pyf"]) {
		latin_variant = OS::LATIN_KEYBOARD_DVORAK;
	} else if ([test isEqualToString:@"xvlcwk"]) {
		latin_variant = OS::LATIN_KEYBOARD_NEO;
	} else if ([test isEqualToString:@"qwfpgj"]) {
		latin_variant = OS::LATIN_KEYBOARD_COLEMAK;
	}

	[test release];

	keyboard_layout_dirty = false;
}

OS::LatinKeyboardVariant OS_OSX::get_latin_keyboard_variant() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	return latin_variant;
}

int OS_OSX::keyboard_get_layout_count() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}
	return kbd_layouts.size();
}

void OS_OSX::keyboard_set_current_layout(int p_index) {
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

int OS_OSX::keyboard_get_current_layout() const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	return current_layout;
}

String OS_OSX::keyboard_get_layout_language(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].code;
}

String OS_OSX::keyboard_get_layout_name(int p_index) const {
	if (keyboard_layout_dirty) {
		_update_keyboard_layouts();
	}

	ERR_FAIL_INDEX_V(p_index, kbd_layouts.size(), "");
	return kbd_layouts[p_index].name;
}

uint32_t OS_OSX::keyboard_get_scancode_from_physical(uint32_t p_scancode) const {
	if (p_scancode == KEY_PAUSE) {
		return p_scancode;
	}

	unsigned int modifiers = p_scancode & KEY_MODIFIER_MASK;
	unsigned int scancode_no_mod = p_scancode & KEY_CODE_MASK;
	unsigned int osx_scancode = unmapKey((uint32_t)scancode_no_mod);
	return (uint32_t)(remapKey(osx_scancode, 0) | modifiers);
}

void OS_OSX::process_events() {
	while (true) {
		NSEvent *event = [NSApp
				nextEventMatchingMask:NSEventMaskAny
							untilDate:[NSDate distantPast]
							   inMode:NSDefaultRunLoopMode
							  dequeue:YES];

		if (event == nil)
			break;

		[NSApp sendEvent:event];
	}
	process_key_events();

	if (mpath.size() > 0) {
		const Vector2 mpos = get_mouse_pos([window_object mouseLocationOutsideOfEventStream]);
		if (Geometry::is_point_in_polygon(mpos, mpath)) {
			if ([window_object ignoresMouseEvents]) {
				[window_object setIgnoresMouseEvents:NO];
			}
		} else {
			if (![window_object ignoresMouseEvents]) {
				[window_object setIgnoresMouseEvents:YES];
			}
		}
	} else {
		if ([window_object ignoresMouseEvents]) {
			[window_object setIgnoresMouseEvents:NO];
		}
	}

	[autoreleasePool drain];
	autoreleasePool = [[NSAutoreleasePool alloc] init];

	input->flush_buffered_events();
}

void OS_OSX::process_key_events() {
	Ref<InputEventKey> k;
	for (int i = 0; i < key_event_pos; i++) {
		const KeyEvent &ke = key_event_buffer[i];

		if (ke.raw) {
			// Non IME input - no composite characters, pass events as is
			k.instance();

			get_key_modifier_state(ke.osx_state, k);
			k->set_pressed(ke.pressed);
			k->set_echo(ke.echo);
			k->set_scancode(ke.scancode);
			k->set_physical_scancode(ke.physical_scancode);
			k->set_unicode(ke.unicode);

			push_input(k);
		} else {
			// IME input
			if ((i == 0 && ke.scancode == 0) || (i > 0 && key_event_buffer[i - 1].scancode == 0)) {
				k.instance();

				get_key_modifier_state(ke.osx_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_scancode(0);
				k->set_physical_scancode(0);
				k->set_unicode(ke.unicode);

				push_input(k);
			}
			if (ke.scancode != 0) {
				k.instance();

				get_key_modifier_state(ke.osx_state, k);
				k->set_pressed(ke.pressed);
				k->set_echo(ke.echo);
				k->set_scancode(ke.scancode);
				k->set_physical_scancode(ke.physical_scancode);

				if (i + 1 < key_event_pos && key_event_buffer[i + 1].scancode == 0) {
					k->set_unicode(key_event_buffer[i + 1].unicode);
				}

				push_input(k);
			}
		}
	}

	key_event_pos = 0;
}

void OS_OSX::push_input(const Ref<InputEvent> &p_event) {
	Ref<InputEvent> ev = p_event;
	input->parse_input_event(ev);
}

void OS_OSX::force_process_input() {
	process_events(); // get rid of pending events
#ifdef SDL_ENABLED
	if (joypad_sdl) {
		joypad_sdl->process_events();
	}
#endif
}

void OS_OSX::pre_wait_observer_cb(CFRunLoopObserverRef p_observer, CFRunLoopActivity p_activiy, void *p_context) {
	// Prevent main loop from sleeping and redraw window during modal popup display.
	// Do not redraw when rendering is done from the separate thread, it will conflict with the OpenGL context updates.

	if (get_singleton()->get_main_loop() && (get_singleton()->get_render_thread_mode() != RENDER_SEPARATE_THREAD) && !OS_OSX::singleton->is_resizing) {
		Main::force_redraw();
		if (!Main::is_iterating()) { // Avoid cyclic loop.
			Main::iteration();
		}
	}

	CFRunLoopWakeUp(CFRunLoopGetCurrent()); // Prevent main loop from sleeping.
}

void OS_OSX::run() {
	force_quit = false;

	if (!main_loop) {
		return;
	}

	main_loop->init();

	if (zoomed) {
		zoomed = false;
		set_window_fullscreen(true);
	}

	CFRunLoopObserverRef pre_wait_observer = CFRunLoopObserverCreate(kCFAllocatorDefault, kCFRunLoopBeforeWaiting, true, 0, &pre_wait_observer_cb, nullptr);
	CFRunLoopAddObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);

	bool quit = false;

	while (!force_quit && !quit) {
		@try {
			process_events(); // get rid of pending events
#ifdef SDL_ENABLED
			if (joypad_sdl) {
				joypad_sdl->process_events();
			}
#endif

			if (Main::iteration()) {
				quit = true;
			}
		} @catch (NSException *exception) {
			ERR_PRINT("NSException: " + String::utf8([exception reason].UTF8String));
		}
	};

	CFRunLoopRemoveObserver(CFRunLoopGetCurrent(), pre_wait_observer, kCFRunLoopCommonModes);
	CFRelease(pre_wait_observer);

	main_loop->finish();
}

void OS_OSX::set_mouse_mode(MouseMode p_mode) {
	if (p_mode == mouse_mode)
		return;

	if (p_mode == MOUSE_MODE_CAPTURED) {
		// Apple Docs state that the display parameter is not used.
		// "This parameter is not used. By default, you may pass kCGDirectMainDisplay."
		// https://developer.apple.com/library/mac/documentation/graphicsimaging/reference/Quartz_Services_Ref/Reference/reference.html
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		CGAssociateMouseAndMouseCursorPosition(false);

		const NSRect contentRect = [window_view frame];
		NSRect pointInWindowRect = NSMakeRect(contentRect.size.width / 2, contentRect.size.height / 2, 0, 0);
		NSPoint pointOnScreen = [[window_view window] convertRectToScreen:pointInWindowRect].origin;
		CGPoint lMouseWarpPos = { pointOnScreen.x, CGDisplayBounds(CGMainDisplayID()).size.height - pointOnScreen.y };
		CGWarpMouseCursorPosition(lMouseWarpPos);
	} else if (p_mode == MOUSE_MODE_HIDDEN) {
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		CGAssociateMouseAndMouseCursorPosition(true);
	} else if (p_mode == MOUSE_MODE_CONFINED) {
		CGDisplayShowCursor(kCGDirectMainDisplay);
		CGAssociateMouseAndMouseCursorPosition(false);
	} else if (p_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
			CGDisplayHideCursor(kCGDirectMainDisplay);
		}
		CGAssociateMouseAndMouseCursorPosition(false);
	} else { // MOUSE_MODE_VISIBLE
		CGDisplayShowCursor(kCGDirectMainDisplay);
		CGAssociateMouseAndMouseCursorPosition(true);
	}

	last_warp = [[NSProcessInfo processInfo] systemUptime];
	ignore_warp = true;
	warp_events.clear();
	mouse_mode = p_mode;

	if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
		CursorShape p_shape = cursor_shape;
		cursor_shape = OS::CURSOR_MAX;
		set_cursor_shape(p_shape);
	}
}

OS::MouseMode OS_OSX::get_mouse_mode() const {
	return mouse_mode;
}

String OS_OSX::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

OS::PowerState OS_OSX::get_power_state() {
	return power_manager->get_power_state();
}

int OS_OSX::get_power_seconds_left() {
	return power_manager->get_power_seconds_left();
}

int OS_OSX::get_power_percent_left() {
	return power_manager->get_power_percent_left();
}

Error OS_OSX::move_to_trash(const String &p_path) {
	NSFileManager *fm = [NSFileManager defaultManager];
	NSURL *url = [NSURL fileURLWithPath:@(p_path.utf8().get_data())];
	NSError *err;

	if (![fm trashItemAtURL:url resultingItemURL:nil error:&err]) {
		ERR_PRINT("trashItemAtURL error: " + String::utf8(err.localizedDescription.UTF8String));
		return FAILED;
	}

	return OK;
}

void OS_OSX::_set_use_vsync(bool p_enable) {
	CGLContextObj ctx = CGLGetCurrentContext();
	if (ctx) {
		GLint swapInterval = p_enable ? 1 : 0;
		CGLSetParameter(ctx, kCGLCPSwapInterval, &swapInterval);
	}
}

OS_OSX *OS_OSX::singleton = NULL;

OS_OSX::OS_OSX() {
	context = nullptr;
	context_offscreen = nullptr;

	memset(cursors, 0, sizeof(cursors));
	key_event_pos = 0;
	mouse_mode = OS::MOUSE_MODE_VISIBLE;
	main_loop = NULL;
	singleton = this;
	im_active = false;
	im_position = Point2();
	layered_window = false;
	autoreleasePool = [[NSAutoreleasePool alloc] init];
	is_visible = true;

	eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
	ERR_FAIL_COND(!eventSource);

	CGEventSourceSetLocalEventsSuppressionInterval(eventSource, 0.0);

	/*
	if (pthread_key_create(&_Godot.nsgl.current, NULL) != 0) {
		_GodotInputError(Godot_PLATFORM_ERROR, "NSGL: Failed to create context TLS");
		return GL_FALSE;
	}
*/

	framework = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
	ERR_FAIL_COND(!framework);

	// Implicitly create shared NSApplication instance
	[GodotApplication sharedApplication];

	// Menu bar setup must go between sharedApplication above and
	// finishLaunching below, in order to properly emulate the behavior
	// of NSApplicationMain
	NSMenuItem *menu_item;
	NSString *title;

	NSString *nsappname = [[[NSBundle mainBundle] infoDictionary] objectForKey:@"CFBundleName"];
	if (nsappname == nil)
		nsappname = [[NSProcessInfo processInfo] processName];

	// Setup Apple menu
	NSMenu *apple_menu = [[NSMenu alloc] initWithTitle:@""];
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

	// Setup menu bar
	NSMenu *main_menu = [[NSMenu alloc] initWithTitle:@""];
	menu_item = [main_menu addItemWithTitle:@"" action:nil keyEquivalent:@""];
	[main_menu setSubmenu:apple_menu forItem:menu_item];
	[NSApp setMainMenu:main_menu];

	[main_menu release];
	[apple_menu release];

	[NSApp finishLaunching];

	delegate = [[GodotApplicationDelegate alloc] init];
	ERR_FAIL_COND(!delegate);
	[NSApp setDelegate:delegate];

	cursor_shape = CURSOR_ARROW;

	maximized = false;
	minimized = false;
	window_size = Vector2(1024, 600);
	zoomed = false;
	resizable = false;
	window_focused = true;
	on_top = false;

	Vector<Logger *> loggers;
	loggers.push_back(memnew(OSXTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));

	//process application:openFile: event
	while (true) {
		NSEvent *event = [NSApp
				nextEventMatchingMask:NSEventMaskAny
							untilDate:[NSDate distantPast]
							   inMode:NSDefaultRunLoopMode
							  dequeue:YES];

		if (event == nil)
			break;

		[NSApp sendEvent:event];
	}

#ifdef COREAUDIO_ENABLED
	AudioDriverManager::add_driver(&audio_driver);
#endif
}

bool OS_OSX::_check_internal_feature_support(const String &p_feature) {
	return p_feature == "pc";
}

void OS_OSX::disable_crash_handler() {
	crash_handler.disable();
}

bool OS_OSX::is_disable_crash_handler() const {
	return crash_handler.is_disabled();
}
