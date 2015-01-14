/*************************************************************************/
/*  os_osx.mm                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#import <Cocoa/Cocoa.h>

#include <Carbon/Carbon.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/hid/IOHIDLib.h>
#include <IOKit/hid/IOHIDKeys.h>

#include "sem_osx.h"
#include "servers/visual/visual_server_raster.h"
//#include "drivers/opengl/rasterizer_gl.h"
//#include "drivers/gles2/rasterizer_gles2.h"
#include "os_osx.h"
#include <stdio.h>
#include <stdlib.h>
#include "print_string.h"
#include "servers/physics/physics_server_sw.h"
#include "drivers/gles2/rasterizer_instance_gles2.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include "main/main.h"
#include "os/keyboard.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <libproc.h>
//uses portions of glfw

//========================================================================
// GLFW 3.0 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2010 Camilla Berglund <elmindreda@elmindreda.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================

static NSRect convertRectToBacking(NSRect contentRect) {

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
    if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_6)
	return [OS_OSX::singleton->window_view convertRectToBacking:contentRect];
    else
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/
	return contentRect;

}

static InputModifierState translateFlags(NSUInteger flags)
{
    InputModifierState mod;


    mod.shift = (flags & NSShiftKeyMask);
    mod.control = (flags & NSControlKeyMask);
    mod.alt = (flags & NSAlternateKeyMask);
    mod.meta = (flags & NSCommandKeyMask);

    return mod;
}

static int mouse_x=0;
static int mouse_y=0;
static int prev_mouse_x=0;
static int prev_mouse_y=0;
static int button_mask=0;


@interface GodotApplication : NSApplication
@end

@implementation GodotApplication

// From http://cocoadev.com/index.pl?GameKeyboardHandlingAlmost
// This works around an AppKit bug, where key up events while holding
// down the command key don't get sent to the key window.
- (void)sendEvent:(NSEvent *)event
{
    if ([event type] == NSKeyUp && ([event modifierFlags] & NSCommandKeyMask))
	[[self keyWindow] sendEvent:event];
    else
	[super sendEvent:event];
}

@end

@interface GodotApplicationDelegate : NSObject
@end

@implementation GodotApplicationDelegate

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender
{
/*    _Godotwindow* window;

    for (window = _Godot.windowListHead;  window;  window = window->next)
	_GodotInputWindowCloseRequest(window);
*/
    return NSTerminateCancel;
}

- (void)applicationDidHide:(NSNotification *)notification
{
  /*  _Godotwindow* window;

    for (window = _Godot.windowListHead;  window;  window = window->next)
	_GodotInputWindowVisibility(window, GL_FALSE);
	*/
}

- (void)applicationDidUnhide:(NSNotification *)notification
{
	/*
    _Godotwindow* window;

    for (window = _Godot.windowListHead;  window;  window = window->next)
    {
	if ([window_object isVisible])
	    _GodotInputWindowVisibility(window, GL_TRUE);
    }
    */
}

- (void)applicationDidChangeScreenParameters:(NSNotification *) notification
{
    //_GodotInputMonitorChange();
}

@end

@interface GodotWindowDelegate : NSObject
{
 //   _Godotwindow* window;
}

@end

@implementation GodotWindowDelegate


- (BOOL)windowShouldClose:(id)sender
{
    //_GodotInputWindowCloseRequest(window);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
    return NO;
}




- (void)windowDidResize:(NSNotification *)notification
{
    [OS_OSX::singleton->context update];

    const NSRect contentRect = [OS_OSX::singleton->window_view frame];
    const NSRect fbRect = convertRectToBacking(contentRect);

    OS_OSX::singleton->current_videomode.width=fbRect.size.width;
    OS_OSX::singleton->current_videomode.height=fbRect.size.height;


   // _GodotInputFramebufferSize(window, fbRect.size.width, fbRect.size.height);
   // _GodotInputWindowSize(window, contentRect.size.width, contentRect.size.height);
    //_GodotInputWindowDamage(window);

    //if (window->cursorMode == Godot_CURSOR_DISABLED)
     //   centerCursor(window);
}

- (void)windowDidMove:(NSNotification *)notification
{
   // [window->nsgl.context update];

   // int x, y;
  //  _GodotPlatformGetWindowPos(window, &x, &y);
   // _GodotInputWindowPos(window, x, y);

    //if (window->cursorMode == Godot_CURSOR_DISABLED)
      //  centerCursor(window);
}

- (void)windowDidMiniaturize:(NSNotification *)notification
{
   // _GodotInputWindowIconify(window, GL_TRUE);
}

- (void)windowDidDeminiaturize:(NSNotification *)notification
{
    //if (window->monitor)
//        enterFullscreenMode(window);

  //  _GodotInputWindowIconify(window, GL_FALSE);
}

- (void)windowDidBecomeKey:(NSNotification *)notification
{
   // _GodotInputWindowFocus(window, GL_TRUE);
   // _GodotPlatformSetCursorMode(window, window->cursorMode);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
}

- (void)windowDidResignKey:(NSNotification *)notification
{
   // _GodotInputWindowFocus(window, GL_FALSE);
   // _GodotPlatformSetCursorMode(window, Godot_CURSOR_NORMAL);
	if (OS_OSX::singleton->get_main_loop())
		OS_OSX::singleton->get_main_loop()->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
}

@end

@interface GodotContentView : NSView
{
      NSTrackingArea* trackingArea;
}



@end

@implementation GodotContentView

+ (void)initialize
{
    if (self == [GodotContentView class])
    {
       /* if (_glfw.ns.cursor == nil)
	{
	    NSImage* data = [[NSImage alloc] initWithSize:NSMakeSize(1, 1)];
	    _glfw.ns.cursor = [[NSCursor alloc] initWithImage:data
						      hotSpot:NSZeroPoint];
	    [data release];
	}*/
    }
}

- (id)init
{
    self = [super init];
    trackingArea = nil;
    [self updateTrackingAreas];

    return self;
}

-(void)dealloc
{
    [trackingArea release];
    [super dealloc];
}

- (BOOL)isOpaque
{
    return YES;
}

- (BOOL)canBecomeKeyView
{
    return YES;
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (void)cursorUpdate:(NSEvent *)event
{
 //   setModeCursor(window, window->cursorMode);
}

- (void)mouseDown:(NSEvent *)event
{

	print_line("mouse down:");
	button_mask|=BUTTON_MASK_LEFT;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_LEFT;
	ev.mouse_button.pressed=true;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.doubleclick = [event clickCount]==2;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);


  /*  _glfwInputMouseClick(window,
			 GLFW_MOUSE_BUTTON_LEFT,
			 GLFW_PRESS,
			 translateFlags([event modifierFlags]));*/
}

- (void)mouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)mouseUp:(NSEvent *)event
{

	button_mask&=~BUTTON_MASK_LEFT;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_LEFT;
	ev.mouse_button.pressed=false;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);

   /* _glfwInputMouseClick(window,
			 GLFW_MOUSE_BUTTON_LEFT,
			 GLFW_RELEASE,
			 translateFlags([event modifierFlags]));*/
}

- (void)mouseMoved:(NSEvent *)event
{

	InputEvent ev;
	ev.type=InputEvent::MOUSE_MOTION;
	ev.mouse_motion.button_mask=button_mask;
	prev_mouse_x=mouse_x;
	prev_mouse_y=mouse_y;
	const NSRect contentRect = [OS_OSX::singleton->window_view frame];
	const NSPoint p = [event locationInWindow];
	mouse_x = p.x * [[event window] backingScaleFactor];
	mouse_y = (contentRect.size.height - p.y) * [[event window] backingScaleFactor];
	ev.mouse_motion.x=mouse_x;
	ev.mouse_motion.y=mouse_y;
	ev.mouse_motion.global_x=mouse_x;
	ev.mouse_motion.global_y=mouse_y;
	ev.mouse_motion.relative_x=[event deltaX] * [[event window] backingScaleFactor];
    ev.mouse_motion.relative_y=[event deltaY] * [[event window] backingScaleFactor];
	ev.mouse_motion.mod = translateFlags([event modifierFlags]);

	OS_OSX::singleton->input->set_mouse_pos(Point2(mouse_x,mouse_y));
	OS_OSX::singleton->push_input(ev);


  /*  if (window->cursorMode == GLFW_CURSOR_DISABLED)
	_glfwInputCursorMotion(window, [event deltaX], [event deltaY]);
    else
    {
	const NSRect contentRect = [window->ns.view frame];
	const NSPoint p = [event locationInWindow];

	_glfwInputCursorMotion(window, p.x, contentRect.size.height - p.y);
    }*/
}

- (void)rightMouseDown:(NSEvent *)event
{

	button_mask|=BUTTON_MASK_RIGHT;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_RIGHT;
	ev.mouse_button.pressed=true;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);

  /*  _glfwInputMouseClick(window,
			 GLFW_MOUSE_BUTTON_RIGHT,
			 GLFW_PRESS,
			 translateFlags([event modifierFlags]));*/
}

- (void)rightMouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)rightMouseUp:(NSEvent *)event
{

	button_mask&=~BUTTON_MASK_RIGHT;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_RIGHT;
	ev.mouse_button.pressed=false;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);

    /*_glfwInputMouseClick(window,
			 GLFW_MOUSE_BUTTON_RIGHT,
			 GLFW_RELEASE,
			 translateFlags([event modifierFlags]));*/
}

- (void)otherMouseDown:(NSEvent *)event
{

	if ((int) [event buttonNumber]!=2)
		return;

	button_mask|=BUTTON_MASK_MIDDLE;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_MIDDLE;
	ev.mouse_button.pressed=true;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);

    /*_glfwInputMouseClick(window,
			 (int) [event buttonNumber],
			 GLFW_PRESS,
			 translateFlags([event modifierFlags]));*/
}

- (void)otherMouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)otherMouseUp:(NSEvent *)event
{

	if ((int) [event buttonNumber]!=2)
		return;

	button_mask&=~BUTTON_MASK_MIDDLE;
	InputEvent ev;
	ev.type=InputEvent::MOUSE_BUTTON;
	ev.mouse_button.button_index=BUTTON_MIDDLE;
	ev.mouse_button.pressed=false;
	ev.mouse_button.x=mouse_x;
	ev.mouse_button.y=mouse_y;
	ev.mouse_button.global_x=mouse_x;
	ev.mouse_button.global_y=mouse_y;
	ev.mouse_button.button_mask=button_mask;
	ev.mouse_button.mod = translateFlags([event modifierFlags]);
	OS_OSX::singleton->push_input(ev);
   /* _glfwInputMouseClick(window,
			 (int) [event buttonNumber],
			 GLFW_RELEASE,
			 translateFlags([event modifierFlags]));*/
}

- (void)mouseExited:(NSEvent *)event
{
   // _glfwInputCursorEnter(window, GL_FALSE);
}

- (void)mouseEntered:(NSEvent *)event
{
  //  _glfwInputCursorEnter(window, GL_TRUE);
}

- (void)viewDidChangeBackingProperties
{
  /*  const NSRect contentRect = [window->ns.view frame];
    const NSRect fbRect = convertRectToBacking(window, contentRect);

    _glfwInputFramebufferSize(window, fbRect.size.width, fbRect.size.height);*/
}

- (void)updateTrackingAreas
{
    if (trackingArea != nil)
    {
	[self removeTrackingArea:trackingArea];
	[trackingArea release];
    }

    NSTrackingAreaOptions options = NSTrackingMouseEnteredAndExited |
				    NSTrackingActiveInKeyWindow |
				    NSTrackingCursorUpdate |
				    NSTrackingInVisibleRect;

    trackingArea = [[NSTrackingArea alloc] initWithRect:[self bounds]
						options:options
						  owner:self
					       userInfo:nil];

    [self addTrackingArea:trackingArea];
    [super updateTrackingAreas];
}

// Translates a OS X keycode to a Godot keycode
//
static int translateKey(unsigned int key)
{
    // Keyboard symbol translation table
    static const unsigned int table[128] =
    {
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
	/* 0a */ KEY_UNKNOWN,
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
	/* 24 */ KEY_RETURN,
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
	/* 40 */ KEY_UNKNOWN,
	/* 41 */ KEY_KP_PERIOD,
	/* 42 */ KEY_UNKNOWN,
	/* 43 */ KEY_KP_MULTIPLY,
	/* 44 */ KEY_UNKNOWN,
	/* 45 */ KEY_KP_ADD,
	/* 46 */ KEY_UNKNOWN,
	/* 47 */ KEY_NUMLOCK, /* Really KeypadClear... */
	/* 48 */ KEY_UNKNOWN, /* VolumeUp */
	/* 49 */ KEY_UNKNOWN, /* VolumeDown */
	/* 4a */ KEY_UNKNOWN, /* Mute */
	/* 4b */ KEY_KP_DIVIDE,
	/* 4c */ KEY_KP_ENTER,
	/* 4d */ KEY_UNKNOWN,
	/* 4e */ KEY_KP_SUBSTRACT,
	/* 4f */ KEY_UNKNOWN,
	/* 50 */ KEY_UNKNOWN,
	/* 51 */ KEY_EQUAL, //wtf equal?
	/* 52 */ KEY_KP_0,
	/* 53 */ KEY_KP_1,
	/* 54 */ KEY_KP_2,
	/* 55 */ KEY_KP_3,
	/* 56 */ KEY_KP_4,
	/* 57 */ KEY_KP_5,
	/* 58 */ KEY_KP_6,
	/* 59 */ KEY_KP_7,
	/* 5a */ KEY_UNKNOWN,
	/* 5b */ KEY_KP_8,
	/* 5c */ KEY_KP_9,
	/* 5d */ KEY_UNKNOWN,
	/* 5e */ KEY_UNKNOWN,
	/* 5f */ KEY_UNKNOWN,
	/* 60 */ KEY_F5,
	/* 61 */ KEY_F6,
	/* 62 */ KEY_F7,
	/* 63 */ KEY_F3,
	/* 64 */ KEY_F8,
	/* 65 */ KEY_F9,
	/* 66 */ KEY_UNKNOWN,
	/* 67 */ KEY_F11,
	/* 68 */ KEY_UNKNOWN,
	/* 69 */ KEY_F13,
	/* 6a */ KEY_F16,
	/* 6b */ KEY_F14,
	/* 6c */ KEY_UNKNOWN,
	/* 6d */ KEY_F10,
	/* 6e */ KEY_UNKNOWN,
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

    if (key >= 128)
	return KEY_UNKNOWN;

    return table[key];
}
- (void)keyDown:(NSEvent *)event
{
    InputEvent ev;
    ev.type=InputEvent::KEY;
    ev.key.pressed=true;
    ev.key.mod=translateFlags([event modifierFlags]);
    ev.key.scancode = latin_keyboard_keycode_convert(translateKey([event keyCode]));
    ev.key.echo = [event isARepeat];

    NSString* characters = [event characters];
    NSUInteger i, length = [characters length];


    if (length>0 && keycode_has_unicode(ev.key.scancode)) {


	for (i = 0;  i < length;  i++) {
		ev.key.unicode=[characters characterAtIndex:i];
		OS_OSX::singleton->push_input(ev);
		ev.key.scancode=0;
	}

    } else {
	    OS_OSX::singleton->push_input(ev);
    }
}

- (void)flagsChanged:(NSEvent *)event
{
   /* int action;
    unsigned int newModifierFlags =
	[event modifierFlags] & NSDeviceIndependentModifierFlagsMask;

    if (newModifierFlags > window->ns.modifierFlags)
	action = GLFW_PRESS;
    else
	action = GLFW_RELEASE;

    window->ns.modifierFlags = newModifierFlags;

    const int key = translateKey([event keyCode]);
    const int mods = translateFlags([event modifierFlags]);
    _glfwInputKey(window, key, [event keyCode], action, mods);*/
}

- (void)keyUp:(NSEvent *)event
{

	InputEvent ev;
	ev.type=InputEvent::KEY;
	ev.key.pressed=false;
	ev.key.mod=translateFlags([event modifierFlags]);
	ev.key.scancode = latin_keyboard_keycode_convert(translateKey([event keyCode]));
	OS_OSX::singleton->push_input(ev);


 /*   const int key = translateKey([event keyCode]);
    const int mods = translateFlags([event modifierFlags]);
    _glfwInputKey(window, key, [event keyCode], GLFW_RELEASE, mods);*/
}

- (void)scrollWheel:(NSEvent *)event
{

	 double deltaX, deltaY;

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
    if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_6)
    {
	deltaX = [event scrollingDeltaX];
	deltaY = [event scrollingDeltaY];

	if ([event hasPreciseScrollingDeltas])
	{
	    deltaX *= 0.1;
	    deltaY *= 0.1;
	}
    }
    else
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/
    {
	deltaX = [event deltaX];
	deltaY = [event deltaY];
    }


	if (fabs(deltaY)) {

		InputEvent ev;
		ev.type=InputEvent::MOUSE_BUTTON;
		ev.mouse_button.button_index=deltaY >0 ? BUTTON_WHEEL_UP : BUTTON_WHEEL_DOWN;
		ev.mouse_button.pressed=true;
		ev.mouse_button.x=mouse_x;
		ev.mouse_button.y=mouse_y;
		ev.mouse_button.global_x=mouse_x;
		ev.mouse_button.global_y=mouse_y;
		ev.mouse_button.button_mask=button_mask;
		OS_OSX::singleton->push_input(ev);
		ev.mouse_button.pressed=false;
		OS_OSX::singleton->push_input(ev);
	}

}

@end

@interface GodotWindow : NSWindow {}
@end

@implementation GodotWindow

- (BOOL)canBecomeKeyWindow
{
    // Required for NSBorderlessWindowMask windows
    return YES;
}

@end


int OS_OSX::get_video_driver_count() const {

	return 1;
}
const char * OS_OSX::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OS_OSX::get_default_video_mode() const {

	VideoMode vm;
	vm.width=800;
	vm.height=600;
	vm.fullscreen=false;
	vm.resizable=true;
	return vm;
}


void OS_OSX::initialize_core() {

	OS_Unix::initialize_core();
	SemaphoreOSX::make_default();

}

static bool keyboard_layout_dirty = true;
static void keyboardLayoutChanged(CFNotificationCenterRef center, void *observer, CFStringRef name, const void *object, CFDictionaryRef userInfo) {
	keyboard_layout_dirty = true;
}

void OS_OSX::initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

	/*** OSX INITIALIZATION ***/
	/*** OSX INITIALIZATION ***/
	/*** OSX INITIALIZATION ***/

	keyboard_layout_dirty = true;

	// Register to be notified on keyboard layout changes
	CFNotificationCenterAddObserver(CFNotificationCenterGetDistributedCenter(),
									NULL, keyboardLayoutChanged,
									kTISNotifySelectedKeyboardInputSourceChanged, NULL,
									CFNotificationSuspensionBehaviorDeliverImmediately);
    
	window_delegate = [[GodotWindowDelegate alloc] init];

       // Don't use accumulation buffer support; it's not accelerated
       // Aux buffers probably aren't accelerated either

	unsigned int styleMask = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | (p_desired.resizable?NSResizableWindowMask:0);


	window_object = [[GodotWindow alloc]
	    initWithContentRect:NSMakeRect(0, 0, p_desired.width, p_desired.height)
		      styleMask:styleMask
			backing:NSBackingStoreBuffered
			  defer:NO];

	ERR_FAIL_COND( window_object==nil );

	window_view = [[GodotContentView alloc] init];

	current_videomode = p_desired;

	// Adjust for display density
	const NSRect fbRect = convertRectToBacking(NSMakeRect(0, 0, p_desired.width, p_desired.height));
	current_videomode.width = fbRect.size.width;
	current_videomode.height = fbRect.size.height;

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
	if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_6) {
	    [window_view setWantsBestResolutionOpenGLSurface:YES];
	    if (current_videomode.resizable)
		[window_object setCollectionBehavior:NSWindowCollectionBehaviorFullScreenPrimary];
	}
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/

//	[window_object setTitle:[NSString stringWithUTF8String:"GodotEnginies"]];
	[window_object setContentView:window_view];
	[window_object setDelegate:window_delegate];
	[window_object setAcceptsMouseMovedEvents:YES];
	[window_object center];

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
	if (floor(NSAppKitVersionNumber) > NSAppKitVersionNumber10_6)
		[window_object setRestorable:NO];
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/

	unsigned int attributeCount = 0;

	// OS X needs non-zero color size, so set resonable values
	int colorBits = 24;

	// Fail if a robustness strategy was requested


#define ADD_ATTR(x) { attributes[attributeCount++] = x; }
#define ADD_ATTR2(x, y) { ADD_ATTR(x); ADD_ATTR(y); }

	// Arbitrary array size here
	NSOpenGLPixelFormatAttribute attributes[40];

	ADD_ATTR(NSOpenGLPFADoubleBuffer);
	ADD_ATTR(NSOpenGLPFAClosestPolicy);

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
	if (false/* use gl3*/)
		ADD_ATTR2(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core);
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/

	ADD_ATTR2(NSOpenGLPFAColorSize, colorBits);

	/* if (fbconfig->alphaBits > 0)
	     ADD_ATTR2(NSOpenGLPFAAlphaSize, fbconfig->alphaBits);*/

	ADD_ATTR2(NSOpenGLPFADepthSize, 24);

	ADD_ATTR2(NSOpenGLPFAStencilSize, 8);

	/*if (fbconfig->stereo)
	     ADD_ATTR(NSOpenGLPFAStereo);*/

	/* if (fbconfig->samples > 0)
	 {
	     ADD_ATTR2(NSOpenGLPFASampleBuffers, 1);
	     ADD_ATTR2(NSOpenGLPFASamples, fbconfig->samples);
	 }*/

	// NOTE: All NSOpenGLPixelFormats on the relevant cards support sRGB
	//       frambuffer, so there's no need (and no way) to request it

	ADD_ATTR(0);

#undef ADD_ATTR
#undef ADD_ATTR2

	pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attributes];
	ERR_FAIL_COND( pixelFormat == nil);


	context = [[NSOpenGLContext alloc] initWithFormat:pixelFormat
							  shareContext:nil];

	ERR_FAIL_COND(context==nil);


	[context setView:window_view];

	[context makeCurrentContext];

	[NSApp activateIgnoringOtherApps:YES];

	 [window_object makeKeyAndOrderFront:nil];


	/*** END OSX INITIALIZATION ***/
	/*** END OSX INITIALIZATION ***/
	/*** END OSX INITIALIZATION ***/

	bool use_gl2=p_video_driver!=1;



	AudioDriverManagerSW::add_driver(&audio_driver_osx);


	rasterizer = instance_RasterizerGLES2();

	visual_server = memnew( VisualServerRaster(rasterizer) );

	if (get_render_thread_mode()!=RENDER_THREAD_UNSAFE) {

		visual_server =memnew(VisualServerWrapMT(visual_server,get_render_thread_mode()==RENDER_SEPARATE_THREAD));
	}
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	AudioDriverManagerSW::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManagerSW::get_driver(p_audio_driver)->init()!=OK) {

		ERR_PRINT("Initializing audio failed.");
	}

	sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew( AudioServerSW(sample_manager) );

	audio_server->set_mixer_params(AudioMixerSW::INTERPOLATION_LINEAR,false);
	audio_server->init();

	spatial_sound_server = memnew( SpatialSoundServerSW );
	spatial_sound_server->init();

	spatial_sound_2d_server = memnew( SpatialSound2DServerSW );
	spatial_sound_2d_server->init();

	//
	physics_server = memnew( PhysicsServerSW );
	physics_server->init();
	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();

	input = memnew( InputDefault );

	_ensure_data_dir();


}
void OS_OSX::finalize() {

	CFNotificationCenterRemoveObserver(CFNotificationCenterGetDistributedCenter(), NULL, kTISNotifySelectedKeyboardInputSourceChanged, NULL);

}

void OS_OSX::set_main_loop( MainLoop * p_main_loop ) {

	main_loop=p_main_loop;
	input->set_main_loop(p_main_loop);

}

void OS_OSX::delete_main_loop() {

	memdelete(main_loop);
	main_loop=NULL;
}


String OS_OSX::get_name() {

	return "OSX";
}

void OS_OSX::set_cursor_shape(CursorShape p_shape) {

	if (cursor_shape==p_shape)
		return;

	switch(p_shape) {
		case CURSOR_ARROW: [[NSCursor arrowCursor] set]; break;
		case CURSOR_IBEAM: [[NSCursor IBeamCursor] set]; break;
		case CURSOR_POINTING_HAND: [[NSCursor pointingHandCursor] set]; break;
		case CURSOR_CROSS: [[NSCursor crosshairCursor] set]; break;
		case CURSOR_WAIT: [[NSCursor arrowCursor] set]; break;
		case CURSOR_BUSY: [[NSCursor arrowCursor] set]; break;
		case CURSOR_DRAG: [[NSCursor closedHandCursor] set]; break;
		case CURSOR_CAN_DROP: [[NSCursor openHandCursor] set]; break;
		case CURSOR_FORBIDDEN: [[NSCursor arrowCursor] set]; break;
		case CURSOR_VSIZE: [[NSCursor resizeUpDownCursor] set]; break;
		case CURSOR_HSIZE: [[NSCursor resizeLeftRightCursor] set]; break;
		case CURSOR_BDIAGSIZE: [[NSCursor arrowCursor] set]; break;
		case CURSOR_FDIAGSIZE: [[NSCursor arrowCursor] set]; break;
		case CURSOR_MOVE: [[NSCursor arrowCursor] set]; break;
		case CURSOR_VSPLIT: [[NSCursor resizeUpDownCursor] set]; break;
		case CURSOR_HSPLIT: [[NSCursor resizeLeftRightCursor] set]; break;
		case CURSOR_HELP: [[NSCursor arrowCursor] set]; break;
		default: {};
	}

	cursor_shape=p_shape;
}

void OS_OSX::set_mouse_show(bool p_show) {

}
void OS_OSX::set_mouse_grab(bool p_grab) {

}
bool OS_OSX::is_mouse_grab_enabled() const {

	return mouse_grab;
}

void OS_OSX::warp_mouse_pos(const Point2& p_to) {

    //copied from windows impl with osx native calls
    if (mouse_mode == MOUSE_MODE_CAPTURED){
        mouse_x = p_to.x;
        mouse_y = p_to.y;
    }
    else{ //set OS position
        CGPoint lMouseWarpPos = {p_to.x, p_to.y};
        
        CGEventSourceRef lEventRef = CGEventSourceCreate(kCGEventSourceStateCombinedSessionState);
        CGEventSourceSetLocalEventsSuppressionInterval(lEventRef, 0.0);
        CGAssociateMouseAndMouseCursorPosition(false);
        CGWarpMouseCursorPosition(lMouseWarpPos);
        CGAssociateMouseAndMouseCursorPosition(true);
    }
}

Point2 OS_OSX::get_mouse_pos() const {

	return Vector2(mouse_x,mouse_y);
}
int OS_OSX::get_mouse_button_state() const {
	return button_mask;
}
void OS_OSX::set_window_title(const String& p_title) {

	[window_object setTitle:[NSString stringWithUTF8String:p_title.utf8().get_data()]];

}

void OS_OSX::set_icon(const Image& p_icon) {

	Image img=p_icon;
	img.convert(Image::FORMAT_RGBA);
	NSBitmapImageRep *imgrep= [[[NSBitmapImageRep alloc] initWithBitmapDataPlanes: NULL
			  pixelsWide: p_icon.get_width()
			  pixelsHigh: p_icon.get_height()
			  bitsPerSample: 8
			  samplesPerPixel: 4
			  hasAlpha: YES
			  isPlanar: NO
			  colorSpaceName: NSDeviceRGBColorSpace
			  bytesPerRow: p_icon.get_width()*4
			  bitsPerPixel: 32] autorelease];
	ERR_FAIL_COND(imgrep==nil);
	uint8_t *pixels = [imgrep bitmapData];

	int len = img.get_width()*img.get_height();
	DVector<uint8_t> data = img.get_data();
	DVector<uint8_t>::Read r = data.read();

	/* Premultiply the alpha channel */
	for (int i = 0; i<len ; i++) {
		uint8_t alpha = r[i*4+3];
		pixels[i*4+0] = (uint8_t)(((uint16_t)r[i*4+0] * alpha) / 255);
		pixels[i*4+1] = (uint8_t)(((uint16_t)r[i*4+1] * alpha) / 255);
		pixels[i*4+2] = (uint8_t)(((uint16_t)r[i*4+2] * alpha) / 255);
		pixels[i*4+3] = alpha;

	}

	NSImage *nsimg = [[[NSImage alloc] initWithSize: NSMakeSize(img.get_width(),img.get_height())] autorelease];
	ERR_FAIL_COND(nsimg == nil);
	[nsimg addRepresentation: imgrep];

	[NSApp setApplicationIconImage:nsimg];

}

MainLoop *OS_OSX::get_main_loop() const {

	return main_loop;
}

bool OS_OSX::can_draw() const {

	return true;
}

void OS_OSX::set_clipboard(const String& p_text) {

	NSArray* types = [NSArray arrayWithObjects:NSStringPboardType, nil];

	NSPasteboard* pasteboard = [NSPasteboard generalPasteboard];
	[pasteboard declareTypes:types owner:nil];
	[pasteboard setString:[NSString stringWithUTF8String:p_text.utf8().get_data()]
			forType:NSStringPboardType];
}
String OS_OSX::get_clipboard() const {

	NSPasteboard* pasteboard = [NSPasteboard generalPasteboard];

	if (![[pasteboard types] containsObject:NSStringPboardType])
	{
		return "";
	}

	NSString* object = [pasteboard stringForType:NSStringPboardType];
	if (!object)
	{
		return "";
	}

	char *utfs = strdup([object UTF8String]);
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

	[[NSWorkspace sharedWorkspace] openURL:[[NSURL alloc] initWithString:[NSString stringWithUTF8String:p_uri.utf8().get_data()]]];
	return OK;
}

void OS_OSX::swap_buffers() {

	[context flushBuffer];

}



void OS_OSX::set_video_mode(const VideoMode& p_video_mode,int p_screen) {

}

OS::VideoMode OS_OSX::get_video_mode(int p_screen) const {

	return current_videomode;
}
void OS_OSX::get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const {

}

void OS_OSX::move_window_to_foreground() {

	[window_object orderFrontRegardless];
}

String OS_OSX::get_executable_path() const {

	int ret;
	pid_t pid;
	char pathbuf[PROC_PIDPATHINFO_MAXSIZE];

	pid = getpid();
	ret = proc_pidpath (pid, pathbuf, sizeof(pathbuf));
	if ( ret <= 0 ) {
		return OS::get_executable_path();
	} else {
		String path;
		path.parse_utf8(pathbuf);

		return path;
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

	for (int i=0; i<length; ++i) {

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
OS::LatinKeyboardVariant OS_OSX::get_latin_keyboard_variant() const {

	static LatinKeyboardVariant layout = LATIN_KEYBOARD_QWERTY;

	if (keyboard_layout_dirty) {

		layout = LATIN_KEYBOARD_QWERTY;

		CGKeyCode keys[] = {kVK_ANSI_Q, kVK_ANSI_W, kVK_ANSI_E, kVK_ANSI_R, kVK_ANSI_T, kVK_ANSI_Y};
		NSString *test = createStringForKeys(keys, 6);

		if ([test isEqualToString:@"qwertz"]) {
			layout = LATIN_KEYBOARD_QWERTZ;
		} else if ([test isEqualToString:@"azerty"]) {
			layout = LATIN_KEYBOARD_AZERTY;
		} else if ([test isEqualToString:@"qzerty"]) {
			layout = LATIN_KEYBOARD_QZERTY;
		} else if ([test isEqualToString:@"',.pyf"]) {
			layout = LATIN_KEYBOARD_DVORAK;
		} else if ([test isEqualToString:@"xvlcwk"]) {
			layout = LATIN_KEYBOARD_NEO;
		}

		[test release];

		keyboard_layout_dirty = false;
		return layout;
	}

	return layout;
}

void OS_OSX::process_events() {

	while (true) {
		NSEvent* event = [NSApp nextEventMatchingMask:NSAnyEventMask
				untilDate:[NSDate distantPast]
				inMode:NSDefaultRunLoopMode
				dequeue:YES];
		if (event == nil)
			break;

		[NSApp sendEvent:event];
	}

	[autoreleasePool drain];
	autoreleasePool = [[NSAutoreleasePool alloc] init];
}



void OS_OSX::push_input(const InputEvent& p_event) {

	InputEvent ev=p_event;
	ev.ID=last_id++;
	//print_line("EV: "+String(ev));
	input->parse_input_event(ev);
}

void OS_OSX::run() {

	force_quit = false;

	if (!main_loop)
		return;

	main_loop->init();

//	uint64_t last_ticks=get_ticks_usec();

//	int frames=0;
//	uint64_t frame=0;

	while (!force_quit) {

		process_events(); // get rid of pending events
//		process_joysticks();
		if (Main::iteration()==true)
			break;
	};

	main_loop->finish();
}

void OS_OSX::set_mouse_mode(MouseMode p_mode) {

    if (p_mode==mouse_mode)
        return;

    if (p_mode==MOUSE_MODE_CAPTURED) {
        // Apple Docs state that the display parameter is not used.
        // "This parameter is not used. By default, you may pass kCGDirectMainDisplay."
        // https://developer.apple.com/library/mac/documentation/graphicsimaging/reference/Quartz_Services_Ref/Reference/reference.html
        CGDisplayHideCursor(kCGDirectMainDisplay);
        CGAssociateMouseAndMouseCursorPosition(false);
    } else if (p_mode==MOUSE_MODE_HIDDEN) {
        CGDisplayHideCursor(kCGDirectMainDisplay);
        CGAssociateMouseAndMouseCursorPosition(true);
    } else {
        CGDisplayShowCursor(kCGDirectMainDisplay);
        CGAssociateMouseAndMouseCursorPosition(true);
    }

    mouse_mode=p_mode;
}

OS::MouseMode OS_OSX::get_mouse_mode() const {

    return mouse_mode;
}

OS_OSX* OS_OSX::singleton=NULL;

OS_OSX::OS_OSX() {

	main_loop=NULL;
	singleton=this;
	autoreleasePool = [[NSAutoreleasePool alloc] init];

	eventSource = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
	ERR_FAIL_COND(!eventSource);

	CGEventSourceSetLocalEventsSuppressionInterval(eventSource, 0.0);


	/*if (pthread_key_create(&_Godot.nsgl.current, NULL) != 0)
	  {
	      _GodotInputError(Godot_PLATFORM_ERROR,
			      "NSGL: Failed to create context TLS");
	      return GL_FALSE;
	  }*/

	framework = CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
	ERR_FAIL_COND(!framework);

	// Implicitly create shared NSApplication instance
	[GodotApplication sharedApplication];

	// In case we are unbundled, make us a proper UI application
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    #if 0
	// Menu bar setup must go between sharedApplication above and
	// finishLaunching below, in order to properly emulate the behavior
	// of NSApplicationMain
	createMenuBar();
    #endif

	[NSApp finishLaunching];

	delegate = [[GodotApplicationDelegate alloc] init];
	ERR_FAIL_COND(!delegate);
	[NSApp setDelegate:delegate];


	last_id=1;
	cursor_shape=CURSOR_ARROW;


}
