//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// OSXWindow.mm: Implementation of OSWindow for OSX

#include "util/osx/OSXWindow.h"

#include <set>
// Include Carbon to use the keycode names in Carbon's Event.h
#include <Carbon/Carbon.h>

#include "anglebase/no_destructor.h"
#include "common/debug.h"

// On OSX 10.12 a number of AppKit interfaces have been renamed for consistency, and the previous
// symbols tagged as deprecated. However we can't simply use the new symbols as it would break
// compilation on our automated testing that doesn't use OSX 10.12 yet. So we just ignore the
// warnings.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Some events such as "ShouldTerminate" are sent to the whole application so we keep a list of
// all the windows in order to forward the event to each of them. However this and calling pushEvent
// in ApplicationDelegate is inherently unsafe in a multithreaded environment.
static std::set<OSXWindow *> &AllWindows()
{
    static angle::base::NoDestructor<std::set<OSXWindow *>> allWindows;
    return *allWindows;
}

@interface Application : NSApplication
@end

@implementation Application
- (void)sendEvent:(NSEvent *)nsEvent
{
    if ([nsEvent type] == NSApplicationDefined)
    {
        for (auto window : AllWindows())
        {
            if ([window->getNSWindow() windowNumber] == [nsEvent windowNumber])
            {
                Event event;
                event.Type = Event::EVENT_TEST;
                window->pushEvent(event);
            }
        }
    }
    [super sendEvent:nsEvent];
}
@end

// The Delegate receiving application-wide events.
@interface ApplicationDelegate : NSObject
@end

@implementation ApplicationDelegate
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender
{
    Event event;
    event.Type = Event::EVENT_CLOSED;
    for (auto window : AllWindows())
    {
        window->pushEvent(event);
    }
    return NSTerminateCancel;
}
@end
static ApplicationDelegate *gApplicationDelegate = nil;

static bool InitializeAppKit()
{
    if (NSApp != nil)
    {
        return true;
    }

    // Initialize the global variable "NSApp"
    [Application sharedApplication];

    // Make us appear in the dock
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Register our global event handler
    gApplicationDelegate = [[ApplicationDelegate alloc] init];
    if (gApplicationDelegate == nil)
    {
        return false;
    }
    [NSApp setDelegate:static_cast<id>(gApplicationDelegate)];

    // Set our status to "started" so we are not bouncing in the doc and can activate
    [NSApp finishLaunching];
    return true;
}

// NS's and CG's coordinate systems start at the bottom left, while OSWindow's coordinate
// system starts at the top left. This function converts the Y coordinate accordingly.
static float YCoordToFromCG(float y)
{
    float screenHeight = CGDisplayBounds(CGMainDisplayID()).size.height;
    return screenHeight - y;
}

// Delegate for window-wide events, note that the protocol doesn't contain anything input related.
@implementation WindowDelegate
- (id)initWithWindow:(OSXWindow *)window
{
    self = [super init];
    if (self != nil)
    {
        mWindow = window;
    }
    return self;
}

- (void)onOSXWindowDeleted
{
    mWindow = nil;
}

- (BOOL)windowShouldClose:(id)sender
{
    Event event;
    event.Type = Event::EVENT_CLOSED;
    mWindow->pushEvent(event);
    return NO;
}

- (void)windowDidResize:(NSNotification *)notification
{
    NSSize windowSize = [[mWindow->getNSWindow() contentView] frame].size;
    Event event;
    event.Type        = Event::EVENT_RESIZED;
    event.Size.Width  = (int)windowSize.width;
    event.Size.Height = (int)windowSize.height;
    mWindow->pushEvent(event);
}

- (void)windowDidMove:(NSNotification *)notification
{
    NSRect screenspace = [mWindow->getNSWindow() frame];
    Event event;
    event.Type   = Event::EVENT_MOVED;
    event.Move.X = (int)screenspace.origin.x;
    event.Move.Y = (int)YCoordToFromCG(screenspace.origin.y + screenspace.size.height);
    mWindow->pushEvent(event);
}

- (void)windowDidBecomeKey:(NSNotification *)notification
{
    Event event;
    event.Type = Event::EVENT_GAINED_FOCUS;
    mWindow->pushEvent(event);
    [self retain];
}

- (void)windowDidResignKey:(NSNotification *)notification
{
    if (mWindow != nil)
    {
        Event event;
        event.Type = Event::EVENT_LOST_FOCUS;
        mWindow->pushEvent(event);
    }
    [self release];
}
@end

static Key NSCodeToKey(int keyCode)
{
    // Missing KEY_PAUSE
    switch (keyCode)
    {
        case kVK_Shift:
            return KEY_LSHIFT;
        case kVK_RightShift:
            return KEY_RSHIFT;
        case kVK_Option:
            return KEY_LALT;
        case kVK_RightOption:
            return KEY_RALT;
        case kVK_Control:
            return KEY_LCONTROL;
        case kVK_RightControl:
            return KEY_RCONTROL;
        case kVK_Command:
            return KEY_LSYSTEM;
        // Right System doesn't have a name, but shows up as 0x36.
        case 0x36:
            return KEY_RSYSTEM;
        case kVK_Function:
            return KEY_MENU;

        case kVK_ANSI_Semicolon:
            return KEY_SEMICOLON;
        case kVK_ANSI_Slash:
            return KEY_SLASH;
        case kVK_ANSI_Equal:
            return KEY_EQUAL;
        case kVK_ANSI_Minus:
            return KEY_DASH;
        case kVK_ANSI_LeftBracket:
            return KEY_LBRACKET;
        case kVK_ANSI_RightBracket:
            return KEY_RBRACKET;
        case kVK_ANSI_Comma:
            return KEY_COMMA;
        case kVK_ANSI_Period:
            return KEY_PERIOD;
        case kVK_ANSI_Backslash:
            return KEY_BACKSLASH;
        case kVK_ANSI_Grave:
            return KEY_TILDE;
        case kVK_Escape:
            return KEY_ESCAPE;
        case kVK_Space:
            return KEY_SPACE;
        case kVK_Return:
            return KEY_RETURN;
        case kVK_Delete:
            return KEY_BACK;
        case kVK_Tab:
            return KEY_TAB;
        case kVK_PageUp:
            return KEY_PAGEUP;
        case kVK_PageDown:
            return KEY_PAGEDOWN;
        case kVK_End:
            return KEY_END;
        case kVK_Home:
            return KEY_HOME;
        case kVK_Help:
            return KEY_INSERT;
        case kVK_ForwardDelete:
            return KEY_DELETE;
        case kVK_ANSI_KeypadPlus:
            return KEY_ADD;
        case kVK_ANSI_KeypadMinus:
            return KEY_SUBTRACT;
        case kVK_ANSI_KeypadMultiply:
            return KEY_MULTIPLY;
        case kVK_ANSI_KeypadDivide:
            return KEY_DIVIDE;

        case kVK_F1:
            return KEY_F1;
        case kVK_F2:
            return KEY_F2;
        case kVK_F3:
            return KEY_F3;
        case kVK_F4:
            return KEY_F4;
        case kVK_F5:
            return KEY_F5;
        case kVK_F6:
            return KEY_F6;
        case kVK_F7:
            return KEY_F7;
        case kVK_F8:
            return KEY_F8;
        case kVK_F9:
            return KEY_F9;
        case kVK_F10:
            return KEY_F10;
        case kVK_F11:
            return KEY_F11;
        case kVK_F12:
            return KEY_F12;
        case kVK_F13:
            return KEY_F13;
        case kVK_F14:
            return KEY_F14;
        case kVK_F15:
            return KEY_F15;

        case kVK_LeftArrow:
            return KEY_LEFT;
        case kVK_RightArrow:
            return KEY_RIGHT;
        case kVK_DownArrow:
            return KEY_DOWN;
        case kVK_UpArrow:
            return KEY_UP;

        case kVK_ANSI_Keypad0:
            return KEY_NUMPAD0;
        case kVK_ANSI_Keypad1:
            return KEY_NUMPAD1;
        case kVK_ANSI_Keypad2:
            return KEY_NUMPAD2;
        case kVK_ANSI_Keypad3:
            return KEY_NUMPAD3;
        case kVK_ANSI_Keypad4:
            return KEY_NUMPAD4;
        case kVK_ANSI_Keypad5:
            return KEY_NUMPAD5;
        case kVK_ANSI_Keypad6:
            return KEY_NUMPAD6;
        case kVK_ANSI_Keypad7:
            return KEY_NUMPAD7;
        case kVK_ANSI_Keypad8:
            return KEY_NUMPAD8;
        case kVK_ANSI_Keypad9:
            return KEY_NUMPAD9;

        case kVK_ANSI_A:
            return KEY_A;
        case kVK_ANSI_B:
            return KEY_B;
        case kVK_ANSI_C:
            return KEY_C;
        case kVK_ANSI_D:
            return KEY_D;
        case kVK_ANSI_E:
            return KEY_E;
        case kVK_ANSI_F:
            return KEY_F;
        case kVK_ANSI_G:
            return KEY_G;
        case kVK_ANSI_H:
            return KEY_H;
        case kVK_ANSI_I:
            return KEY_I;
        case kVK_ANSI_J:
            return KEY_J;
        case kVK_ANSI_K:
            return KEY_K;
        case kVK_ANSI_L:
            return KEY_L;
        case kVK_ANSI_M:
            return KEY_M;
        case kVK_ANSI_N:
            return KEY_N;
        case kVK_ANSI_O:
            return KEY_O;
        case kVK_ANSI_P:
            return KEY_P;
        case kVK_ANSI_Q:
            return KEY_Q;
        case kVK_ANSI_R:
            return KEY_R;
        case kVK_ANSI_S:
            return KEY_S;
        case kVK_ANSI_T:
            return KEY_T;
        case kVK_ANSI_U:
            return KEY_U;
        case kVK_ANSI_V:
            return KEY_V;
        case kVK_ANSI_W:
            return KEY_W;
        case kVK_ANSI_X:
            return KEY_X;
        case kVK_ANSI_Y:
            return KEY_Y;
        case kVK_ANSI_Z:
            return KEY_Z;

        case kVK_ANSI_1:
            return KEY_NUM1;
        case kVK_ANSI_2:
            return KEY_NUM2;
        case kVK_ANSI_3:
            return KEY_NUM3;
        case kVK_ANSI_4:
            return KEY_NUM4;
        case kVK_ANSI_5:
            return KEY_NUM5;
        case kVK_ANSI_6:
            return KEY_NUM6;
        case kVK_ANSI_7:
            return KEY_NUM7;
        case kVK_ANSI_8:
            return KEY_NUM8;
        case kVK_ANSI_9:
            return KEY_NUM9;
        case kVK_ANSI_0:
            return KEY_NUM0;
    }

    return Key(0);
}

static void AddNSKeyStateToEvent(Event *event, NSEventModifierFlags state)
{
    event->Key.Shift   = state & NSShiftKeyMask;
    event->Key.Control = state & NSControlKeyMask;
    event->Key.Alt     = state & NSAlternateKeyMask;
    event->Key.System  = state & NSCommandKeyMask;
}

static MouseButton TranslateMouseButton(NSInteger button)
{
    switch (button)
    {
        case 2:
            return MOUSEBUTTON_MIDDLE;
        case 3:
            return MOUSEBUTTON_BUTTON4;
        case 4:
            return MOUSEBUTTON_BUTTON5;
        default:
            return MOUSEBUTTON_UNKNOWN;
    }
}

// Delegate for NSView events, mostly the input events
@implementation ContentView
- (id)initWithWindow:(OSXWindow *)window
{
    self = [super init];
    if (self != nil)
    {
        mWindow          = window;
        mTrackingArea    = nil;
        mCurrentModifier = 0;
        [self updateTrackingAreas];
    }
    return self;
}

- (void)dealloc
{
    [mTrackingArea release];
    [super dealloc];
}

- (void)updateTrackingAreas
{
    if (mTrackingArea != nil)
    {
        [self removeTrackingArea:mTrackingArea];
        [mTrackingArea release];
        mTrackingArea = nil;
    }

    NSRect bounds               = [self bounds];
    NSTrackingAreaOptions flags = NSTrackingMouseEnteredAndExited | NSTrackingActiveInKeyWindow |
                                  NSTrackingCursorUpdate | NSTrackingInVisibleRect |
                                  NSTrackingAssumeInside;
    mTrackingArea = [[NSTrackingArea alloc] initWithRect:bounds
                                                 options:flags
                                                   owner:self
                                                userInfo:nil];

    [self addTrackingArea:mTrackingArea];
    [super updateTrackingAreas];
}

// Helps with performance
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

// Handle mouse events from the NSResponder protocol
- (float)translateMouseY:(float)y
{
    return [self frame].size.height - y;
}

- (void)addButtonEvent:(NSEvent *)nsEvent
                  type:(Event::EventType)eventType
                button:(MouseButton)button
{
    Event event;
    event.Type               = eventType;
    event.MouseButton.Button = button;
    event.MouseButton.X      = (int)[nsEvent locationInWindow].x;
    event.MouseButton.Y      = (int)[self translateMouseY:[nsEvent locationInWindow].y];
    mWindow->pushEvent(event);
}

- (void)mouseDown:(NSEvent *)event
{
    [self addButtonEvent:event type:Event::EVENT_MOUSE_BUTTON_PRESSED button:MOUSEBUTTON_LEFT];
}

- (void)mouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)mouseUp:(NSEvent *)event
{
    [self addButtonEvent:event type:Event::EVENT_MOUSE_BUTTON_RELEASED button:MOUSEBUTTON_LEFT];
}

- (void)mouseMoved:(NSEvent *)nsEvent
{
    Event event;
    event.Type        = Event::EVENT_MOUSE_MOVED;
    event.MouseMove.X = (int)[nsEvent locationInWindow].x;
    event.MouseMove.Y = (int)[self translateMouseY:[nsEvent locationInWindow].y];
    mWindow->pushEvent(event);
}

- (void)mouseEntered:(NSEvent *)nsEvent
{
    Event event;
    event.Type = Event::EVENT_MOUSE_ENTERED;
    mWindow->pushEvent(event);
}

- (void)mouseExited:(NSEvent *)nsEvent
{
    Event event;
    event.Type = Event::EVENT_MOUSE_LEFT;
    mWindow->pushEvent(event);
}

- (void)rightMouseDown:(NSEvent *)event
{
    [self addButtonEvent:event type:Event::EVENT_MOUSE_BUTTON_PRESSED button:MOUSEBUTTON_RIGHT];
}

- (void)rightMouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)rightMouseUp:(NSEvent *)event
{
    [self addButtonEvent:event type:Event::EVENT_MOUSE_BUTTON_RELEASED button:MOUSEBUTTON_RIGHT];
}

- (void)otherMouseDown:(NSEvent *)event
{
    [self addButtonEvent:event
                    type:Event::EVENT_MOUSE_BUTTON_PRESSED
                  button:TranslateMouseButton([event buttonNumber])];
}

- (void)otherMouseDragged:(NSEvent *)event
{
    [self mouseMoved:event];
}

- (void)otherMouseUp:(NSEvent *)event
{
    [self addButtonEvent:event
                    type:Event::EVENT_MOUSE_BUTTON_RELEASED
                  button:TranslateMouseButton([event buttonNumber])];
}

- (void)scrollWheel:(NSEvent *)nsEvent
{
    if (static_cast<int>([nsEvent deltaY]) == 0)
    {
        return;
    }

    Event event;
    event.Type             = Event::EVENT_MOUSE_WHEEL_MOVED;
    event.MouseWheel.Delta = (int)[nsEvent deltaY];
    mWindow->pushEvent(event);
}

// Handle key events from the NSResponder protocol
- (void)keyDown:(NSEvent *)nsEvent
{
    // TODO(cwallez) also send text events
    Event event;
    event.Type     = Event::EVENT_KEY_PRESSED;
    event.Key.Code = NSCodeToKey([nsEvent keyCode]);
    AddNSKeyStateToEvent(&event, [nsEvent modifierFlags]);
    mWindow->pushEvent(event);
}

- (void)keyUp:(NSEvent *)nsEvent
{
    Event event;
    event.Type     = Event::EVENT_KEY_RELEASED;
    event.Key.Code = NSCodeToKey([nsEvent keyCode]);
    AddNSKeyStateToEvent(&event, [nsEvent modifierFlags]);
    mWindow->pushEvent(event);
}

// Modifier keys do not trigger keyUp/Down events but only flagsChanged events.
- (void)flagsChanged:(NSEvent *)nsEvent
{
    Event event;

    // Guess if the key has been pressed or released with the change of modifiers
    // It currently doesn't work when modifiers are unchanged, such as when pressing
    // both shift keys. GLFW has a solution for this but it requires tracking the
    // state of the keys. Implementing this is still TODO(cwallez)
    int modifier = [nsEvent modifierFlags] & NSDeviceIndependentModifierFlagsMask;
    if (modifier < mCurrentModifier)
    {
        event.Type = Event::EVENT_KEY_RELEASED;
    }
    else
    {
        event.Type = Event::EVENT_KEY_PRESSED;
    }
    mCurrentModifier = modifier;

    event.Key.Code = NSCodeToKey([nsEvent keyCode]);
    AddNSKeyStateToEvent(&event, [nsEvent modifierFlags]);
    mWindow->pushEvent(event);
}
@end

OSXWindow::OSXWindow() : mWindow(nil), mDelegate(nil), mView(nil) {}

OSXWindow::~OSXWindow()
{
    destroy();
}

bool OSXWindow::initialize(const std::string &name, int width, int height)
{
    if (!InitializeAppKit())
    {
        return false;
    }

    unsigned int styleMask = NSTitledWindowMask | NSClosableWindowMask | NSResizableWindowMask |
                             NSMiniaturizableWindowMask;
    mWindow = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, height)
                                          styleMask:styleMask
                                            backing:NSBackingStoreBuffered
                                              defer:NO];

    if (mWindow == nil)
    {
        return false;
    }

    mDelegate = [[WindowDelegate alloc] initWithWindow:this];
    if (mDelegate == nil)
    {
        return false;
    }
    [mWindow setDelegate:static_cast<id>(mDelegate)];

    mView = [[ContentView alloc] initWithWindow:this];
    if (mView == nil)
    {
        return false;
    }
    [mView setWantsLayer:YES];

    // Disable scaling for this view.
    mView.layer.contentsScale = 1;

    [mWindow setContentView:mView];
    [mWindow setTitle:[NSString stringWithUTF8String:name.c_str()]];
    [mWindow setAcceptsMouseMovedEvents:YES];
    [mWindow center];

    [NSApp activateIgnoringOtherApps:YES];

    mX      = 0;
    mY      = 0;
    mWidth  = width;
    mHeight = height;

    AllWindows().insert(this);
    return true;
}

void OSXWindow::destroy()
{
    AllWindows().erase(this);

    [mView release];
    mView = nil;
    [mDelegate onOSXWindowDeleted];
    [mDelegate release];
    mDelegate = nil;
    [mWindow setContentView:nil];
    [mWindow release];
    mWindow = nil;
}

void OSXWindow::resetNativeWindow() {}

EGLNativeWindowType OSXWindow::getNativeWindow() const
{
    return [mView layer];
}

EGLNativeDisplayType OSXWindow::getNativeDisplay() const
{
    // TODO(cwallez): implement it once we have defined what EGLNativeDisplayType is
    return static_cast<EGLNativeDisplayType>(0);
}

void OSXWindow::messageLoop()
{
    @autoreleasepool
    {
        while (true)
        {
            NSEvent *event = [NSApp nextEventMatchingMask:NSAnyEventMask
                                                untilDate:[NSDate distantPast]
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES];
            if (event == nil)
            {
                break;
            }

            if ([event type] == NSAppKitDefined)
            {
                continue;
            }
            [NSApp sendEvent:event];
        }
    }
}

void OSXWindow::setMousePosition(int x, int y)
{
    y = (int)([mWindow frame].size.height) - y - 1;
    NSPoint screenspace;

#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_7
    screenspace = [mWindow convertBaseToScreen:NSMakePoint(x, y)];
#else
    screenspace = [mWindow convertRectToScreen:NSMakeRect(x, y, 0, 0)].origin;
#endif
    CGWarpMouseCursorPosition(CGPointMake(screenspace.x, YCoordToFromCG(screenspace.y)));
}

bool OSXWindow::setPosition(int x, int y)
{
    // Given CG and NS's coordinate system, the "Y" position of a window is the Y coordinate
    // of the bottom of the window.
    int newBottom    = (int)([mWindow frame].size.height) + y;
    NSRect emptyRect = NSMakeRect(x, YCoordToFromCG(newBottom), 0, 0);
    [mWindow setFrameOrigin:[mWindow frameRectForContentRect:emptyRect].origin];
    return true;
}

bool OSXWindow::resize(int width, int height)
{
    [mWindow setContentSize:NSMakeSize(width, height)];
    return true;
}

void OSXWindow::setVisible(bool isVisible)
{
    if (isVisible)
    {
        [mWindow makeKeyAndOrderFront:nil];
    }
    else
    {
        [mWindow orderOut:nil];
    }
}

void OSXWindow::signalTestEvent()
{
    @autoreleasepool
    {
        NSEvent *event = [NSEvent otherEventWithType:NSApplicationDefined
                                            location:NSMakePoint(0, 0)
                                       modifierFlags:0
                                           timestamp:0.0
                                        windowNumber:[mWindow windowNumber]
                                             context:nil
                                             subtype:0
                                               data1:0
                                               data2:0];
        [NSApp postEvent:event atStart:YES];
    }
}

NSWindow *OSXWindow::getNSWindow() const
{
    return mWindow;
}

// static
OSWindow *OSWindow::New()
{
    return new OSXWindow;
}
